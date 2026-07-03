// src/device/cuda/cuda_backend_qpfstats.cu
//
// CudaBackend's qpfstats smoothing-solve subsystem: the shared-factor batched
// least-squares fit that turns genome-wide f-statistics into smoothed,
// bias-corrected estimates. A private CUDA TU split out of cuda_backend.cu,
// with the qpfstats solve family's bodies moved verbatim.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_qpfstats.cu.md
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/internal/nvtx.hpp"
#include "core/internal/qpfstats_jackknife.hpp"
#include "core/internal/small_linalg.hpp"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/device_decode_result_impl.cuh"
#include "device/cuda/dstat_kernel.cuh"
#include "device/cuda/f2_block_kernel.cuh"
#include "device/cuda/qpadm_fit_kernels.cuh"
#include "device/cuda/qpfstats_jackknife_kernel.cuh"
#include "device/cuda/qpfstats_kernel.cuh"
#include "steppe/config.hpp"

namespace steppe::device {

// qpfstats_smooth: the shared-factor batched smoothing solve — reference §3
QpfstatsSmooth CudaBackend::qpfstats_smooth(std::span<const double> x,
                                             std::span<const double> ymat,
                                             std::span<const double> y,
                                             int npopcomb, int npairs,
                                             int n_block, double ridge,
                                             const Precision& precision) {
    guard_device();
    QpfstatsSmooth out;
    out.npairs = npairs;
    out.n_block = n_block;
    if (npopcomb <= 0 || npairs <= 0 || n_block <= 0) { out.status = Status::Ok; return out; }

    const std::size_t nc = static_cast<std::size_t>(npopcomb);
    const std::size_t np = static_cast<std::size_t>(npairs);
    const std::size_t nb = static_cast<std::size_t>(n_block);
    const int ncols = n_block + 1;
    const std::size_t ncols_sz = static_cast<std::size_t>(ncols);

    DeviceBuffer<double> dX(nc * np);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dX.data(), x.data(), nc * np * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    DeviceBuffer<double> dRhsSrc(nc * ncols_sz);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRhsSrc.data(), ymat.data(), nc * nb * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRhsSrc.data() + nc * nb, y.data(),
                                      nc * sizeof(double), cudaMemcpyHostToDevice,
                                      stream_.get()));
    DeviceBuffer<int> dNanPerCol(ncols_sz);
    launch_qpfstats_zero_nan_ymat(dRhsSrc.data(), npopcomb, ncols, dNanPerCol.data(),
                                  stream_.get());
    std::vector<int> nan_per_col(ncols_sz, 0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(nan_per_col.data(), dNanPerCol.data(),
                                      ncols_sz * sizeof(int), cudaMemcpyDeviceToHost,
                                      stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    DeviceBuffer<double> dA(np * np);
    {
        const double alpha = 1.0, beta = 0.0;
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                 npairs, npopcomb, &alpha, dX.data(), npopcomb,
                                 &beta, dA.data(), npairs));
    }
    launch_symmetrize_lower_to_full(dA.data(), npairs, stream_.get());
    launch_qpfstats_add_ridge_diag(dA.data(), npairs, ridge, stream_.get());

    DeviceBuffer<double> dRhs(np * ncols_sz);
    {
        const double alpha = 1.0, beta = 0.0;
        const MathModeScope gemm_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                                 npairs, ncols, npopcomb, &alpha,
                                 dX.data(), npopcomb, dRhsSrc.data(), npopcomb,
                                 &beta, dRhs.data(), npairs));
    }

    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    DeviceBuffer<int> dInfo(1);
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                               npairs, dA.data(), npairs, &lwork));
    const std::size_t lwork_need = static_cast<std::size_t>(lwork > 0 ? lwork : 1);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, npairs,
                                    dA.data(), npairs, solver_work_.data(), lwork, dInfo.data()));
    int info = 0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) { out.status = Status::NonSpdCovariance; return out; }

    {
        const double one = 1.0;
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
    }

    std::vector<double> sol(np * ncols_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(sol.data(), dRhs.data(), np * ncols_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.b.assign(np * nb, 0.0);
    std::copy(sol.begin(), sol.begin() + static_cast<std::ptrdiff_t>(np * nb), out.b.begin());
    out.bglob.assign(np, 0.0);
    std::copy(sol.begin() + static_cast<std::ptrdiff_t>(np * nb), sol.end(),
              out.bglob.begin());

    bool any_partial = false;
    for (int col = 0; col < ncols; ++col)
        if (nan_per_col[static_cast<std::size_t>(col)] > 0 &&
            nan_per_col[static_cast<std::size_t>(col)] < npopcomb) { any_partial = true; break; }
    if (any_partial) {
        const auto xat = [&](int c, int p) -> double {
            return x[static_cast<std::size_t>(c) + nc * static_cast<std::size_t>(p)];
        };
        std::vector<double> Abase(np * np, 0.0);
        for (int i = 0; i < npairs; ++i)
            for (int j = 0; j < npairs; ++j) {
                long double s = 0.0L;
                for (int c = 0; c < npopcomb; ++c)
                    s += static_cast<long double>(xat(c, i)) * static_cast<long double>(xat(c, j));
                double v = static_cast<double>(s);
                if (i == j) v += ridge;
                Abase[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] = v;
            }
        std::vector<double> rhs(np), bcol, Adown;
        for (int col = 0; col < ncols; ++col) {
            const int kc = nan_per_col[static_cast<std::size_t>(col)];
            if (kc <= 0 || kc >= npopcomb) continue;
            std::fill(rhs.begin(), rhs.end(), 0.0);
            Adown = Abase;
            for (int c = 0; c < npopcomb; ++c) {
                const double sv = (col < n_block)
                    ? ymat[nc * static_cast<std::size_t>(col) + static_cast<std::size_t>(c)]
                    : y[static_cast<std::size_t>(c)];
                if (!std::isfinite(sv)) {
                    for (int i = 0; i < npairs; ++i)
                        for (int j = 0; j < npairs; ++j)
                            Adown[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] -=
                                xat(c, i) * xat(c, j);
                    continue;
                }
                for (int p = 0; p < npairs; ++p) rhs[static_cast<std::size_t>(p)] += xat(c, p) * sv;
            }
            const core::LinAlgStatus st = core::solve(Adown, npairs, rhs, bcol);
            if (!st.ok) { out.status = Status::NonSpdCovariance; return out; }
            if (col < n_block)
                std::copy(bcol.begin(), bcol.end(),
                          out.b.begin() + static_cast<std::ptrdiff_t>(np * static_cast<std::size_t>(col)));
            else
                out.bglob = bcol;
        }
    }

    out.status = Status::Ok;
    return out;
}

// qpfstats_blocks_smooth (host-ptr upload door) — reference §6
QpfstatsSmooth CudaBackend::qpfstats_blocks_smooth(
    const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
    std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
    std::span<const int> block_sizes, double ridge, const Precision& precision) {
    guard_device();
    STEPPE_NVTX_RANGE("qpfstats");
    if (npopcomb <= 0 || npairs <= 0 || n_block <= 0 || P <= 0 || M <= 0 ||
        quadruples.size() < 4) {
        QpfstatsSmooth out;
        out.npairs = npairs;
        out.n_block = n_block;
        out.status = Status::Ok;
        return out;
    }
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    DeviceBuffer<double> dQ(pm), dV(pm);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ.data(), Q, pm * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV.data(), V, pm * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    return qpfstats_blocks_smooth_device(dQ.data(), dV.data(), P, M, block_id, n_block,
                                         quadruples, x, npopcomb, npairs, block_sizes,
                                         ridge, precision);
}

// qpfstats_blocks_smooth (resident-borrow door) — reference §6
QpfstatsSmooth CudaBackend::qpfstats_blocks_smooth(
    const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
    std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
    std::span<const int> block_sizes, double ridge, const Precision& precision) {
    guard_device();
    if (dec.empty() || dec.q_device() == nullptr) {
        QpfstatsSmooth out;
        out.npairs = npairs;
        out.n_block = n_block;
        out.status = Status::Ok;
        return out;
    }
    return qpfstats_blocks_smooth_device(dec.q_device(), dec.v_device(), dec.P,
                                         dec.M_kept, block_id, n_block, quadruples, x,
                                         npopcomb, npairs, block_sizes, ridge, precision);
}

// qpfstats_blocks_smooth_device: the fused device pipeline — reference §7
QpfstatsSmooth CudaBackend::qpfstats_blocks_smooth_device(
    const double* dQ, const double* dV, int P, long M, const int* block_id, int n_block,
    std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
    std::span<const int> block_sizes, double ridge, const Precision& precision) {
    QpfstatsSmooth out;
    out.npairs = npairs;
    out.n_block = n_block;
    const int N = static_cast<int>(quadruples.size() / 4);
    if (npopcomb <= 0 || npairs <= 0 || n_block <= 0 || N <= 0 || P <= 0 || M <= 0) {
        out.status = Status::Ok;
        return out;
    }

    const std::size_t nc = static_cast<std::size_t>(npopcomb);
    const std::size_t np = static_cast<std::size_t>(npairs);
    const std::size_t nbb = static_cast<std::size_t>(n_block);
    const std::size_t nb_out = nc * nbb;

    const std::vector<core::BlockRange> ranges =
        core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                           M, n_block);
    std::vector<int> begin(nbb), size(nbb);
    for (int b = 0; b < n_block; ++b) {
        begin[static_cast<std::size_t>(b)] = static_cast<int>(ranges[static_cast<std::size_t>(b)].begin);
        size[static_cast<std::size_t>(b)]  = static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }

    const std::size_t nq = static_cast<std::size_t>(N) * 4u;

    DeviceBuffer<int> dQuad(nq), dBegin(nbb), dSize(nbb);
    DeviceBuffer<double> dX(nc * np);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuad.data(), quadruples.data(), nq * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBegin.data(), begin.data(), nbb * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSize.data(), size.data(), nbb * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dX.data(), x.data(), nc * np * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));

    DeviceBuffer<double> dNum(nb_out), dDen(nb_out), dCnt(nb_out);
    launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                              dBegin.data(), dSize.data(), n_block,
                              dNum.data(), dDen.data(), dCnt.data(), stream_.get());

    const int ncols = n_block + 1;
    const std::size_t ncols_sz = static_cast<std::size_t>(ncols);
    DeviceBuffer<double> dNumer(nb_out);
    DeviceBuffer<double> dRhsSrc(nc * ncols_sz);
    launch_qpfstats_numer_jackknife(dNum.data(), dCnt.data(), npopcomb, n_block,
                                    dNumer.data(), dRhsSrc.data(),
                                    dRhsSrc.data() + nc * nbb, stream_.get());

    DeviceBuffer<int> dNanPerCol(ncols_sz);
    launch_qpfstats_zero_nan_ymat(dRhsSrc.data(), npopcomb, ncols, dNanPerCol.data(),
                                  stream_.get());
    std::vector<int> nan_per_col(ncols_sz, 0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(nan_per_col.data(), dNanPerCol.data(),
                                      ncols_sz * sizeof(int), cudaMemcpyDeviceToHost,
                                      stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    DeviceBuffer<double> dA(np * np);
    {
        const double alpha = 1.0, beta = 0.0;
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                 npairs, npopcomb, &alpha, dX.data(), npopcomb,
                                 &beta, dA.data(), npairs));
    }
    launch_symmetrize_lower_to_full(dA.data(), npairs, stream_.get());
    launch_qpfstats_add_ridge_diag(dA.data(), npairs, ridge, stream_.get());

    DeviceBuffer<double> dRhs(np * ncols_sz);
    {
        const double alpha = 1.0, beta = 0.0;
        const MathModeScope gemm_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                                 npairs, ncols, npopcomb, &alpha,
                                 dX.data(), npopcomb, dRhsSrc.data(), npopcomb,
                                 &beta, dRhs.data(), npairs));
    }

    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    DeviceBuffer<int> dInfo(1);
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                               npairs, dA.data(), npairs, &lwork));
    const std::size_t lwork_need = static_cast<std::size_t>(lwork > 0 ? lwork : 1);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, npairs,
                                    dA.data(), npairs, solver_work_.data(), lwork, dInfo.data()));
    int info = 0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) { out.status = Status::NonSpdCovariance; return out; }

    {
        const double one = 1.0;
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
    }

    DeviceBuffer<int> dBlockSizes(nbb);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                      nbb * sizeof(int), cudaMemcpyHostToDevice,
                                      stream_.get()));
    DeviceBuffer<double> dShift(np);
    launch_qpfstats_recenter_shift(dRhs.data(), dRhs.data() + np * nbb, dBlockSizes.data(),
                                   npairs, n_block, dShift.data(), stream_.get());

    std::vector<double> sol(np * ncols_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(sol.data(), dRhs.data(), np * ncols_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    out.recenter_shift.assign(np, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.recenter_shift.data(), dShift.data(),
                                      np * sizeof(double), cudaMemcpyDeviceToHost,
                                      stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.b.assign(np * nbb, 0.0);
    std::copy(sol.begin(), sol.begin() + static_cast<std::ptrdiff_t>(np * nbb), out.b.begin());
    out.bglob.assign(np, 0.0);
    std::copy(sol.begin() + static_cast<std::ptrdiff_t>(np * nbb), sol.end(),
              out.bglob.begin());

    bool any_partial = false;
    for (int col = 0; col < ncols; ++col)
        if (nan_per_col[static_cast<std::size_t>(col)] > 0 &&
            nan_per_col[static_cast<std::size_t>(col)] < npopcomb) { any_partial = true; break; }
    if (any_partial) {
        std::vector<double> ymat(nb_out, 0.0), yv(nc, 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(ymat.data(), dRhsSrc.data(),
                                          nb_out * sizeof(double), cudaMemcpyDeviceToHost,
                                          stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(yv.data(), dRhsSrc.data() + nc * nbb,
                                          nc * sizeof(double), cudaMemcpyDeviceToHost,
                                          stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        const auto xat = [&](int c, int p) -> double {
            return x[static_cast<std::size_t>(c) + nc * static_cast<std::size_t>(p)];
        };
        std::vector<double> Abase(np * np, 0.0);
        for (int i = 0; i < npairs; ++i)
            for (int j = 0; j < npairs; ++j) {
                long double s = 0.0L;
                for (int c = 0; c < npopcomb; ++c)
                    s += static_cast<long double>(xat(c, i)) * static_cast<long double>(xat(c, j));
                double v = static_cast<double>(s);
                if (i == j) v += ridge;
                Abase[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] = v;
            }
        std::vector<double> rhs(np), bcol, Adown;
        for (int col = 0; col < ncols; ++col) {
            const int kc = nan_per_col[static_cast<std::size_t>(col)];
            if (kc <= 0 || kc >= npopcomb) continue;
            std::fill(rhs.begin(), rhs.end(), 0.0);
            Adown = Abase;
            for (int c = 0; c < npopcomb; ++c) {
                const double sv = (col < n_block)
                    ? ymat[nc * static_cast<std::size_t>(col) + static_cast<std::size_t>(c)]
                    : yv[static_cast<std::size_t>(c)];
                if (!std::isfinite(sv)) {
                    for (int i = 0; i < npairs; ++i)
                        for (int j = 0; j < npairs; ++j)
                            Adown[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] -=
                                xat(c, i) * xat(c, j);
                    continue;
                }
                for (int p = 0; p < npairs; ++p) rhs[static_cast<std::size_t>(p)] += xat(c, p) * sv;
            }
            const core::LinAlgStatus st = core::solve(Adown, npairs, rhs, bcol);
            if (!st.ok) { out.status = Status::NonSpdCovariance; return out; }
            if (col < n_block)
                std::copy(bcol.begin(), bcol.end(),
                          out.b.begin() + static_cast<std::ptrdiff_t>(np * static_cast<std::size_t>(col)));
            else
                out.bglob = bcol;
        }
        std::vector<int> bl(block_sizes.begin(), block_sizes.end());
        std::vector<double> series(nbb, 0.0);
        for (int p = 0; p < npairs; ++p) {
            for (int b = 0; b < n_block; ++b)
                series[static_cast<std::size_t>(b)] =
                    out.b[static_cast<std::size_t>(p) + np * static_cast<std::size_t>(b)];
            out.recenter_shift[static_cast<std::size_t>(p)] =
                out.bglob[static_cast<std::size_t>(p)] - core::f2blocks_pair_est(series, bl);
        }
    }

    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
