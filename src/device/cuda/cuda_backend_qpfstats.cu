// src/device/cuda/cuda_backend_qpfstats.cu
//
// CudaBackend — qpfstats smoothing-solve subsystem TU
// (cuda_backend.cu split T5; docs/kimiactions/05-cuda-backend-split.md §2.3 TU-F).
// Out-of-line homes of the qpfstats shared-factor batched smoothing-solve family:
// `CudaBackend::qpfstats_smooth` (the x'x+ridge SYRK gram → the multi-column Cholesky
// solve over ALL n_block+1 RHS columns in two Dtrsm; EmulatedFp64 matmul + native-FP64
// carve-out potrf, plus the cold partial-NaN host downdate fallback),
// `qpfstats_blocks_smooth(host-ptr)` + `qpfstats_blocks_smooth(DeviceDecodeResult)` (the
// Q/V H2D-upload vs resident-borrow doors), and the fused device core
// `qpfstats_blocks_smooth_device` (genotype-f4 numerator reduce → on-device per-comb
// block jackknife → the SAME smoothing solve straight from VRAM → on-device per-pair
// recenter). Bodies MOVED VERBATIM from cuda_backend.cu; nothing about codegen / math /
// precision / file-order changed by the split.
//
// Shared mutable state stays as class members (cuda_backend.cuh): `solve_precision_`
// (READER here — the native-FP64 potrf carve-out; WRITER set_solve_precision in the
// lifecycle TU), `blas_`/`solver_` (the SYRK/GEMM/Dtrsm + potrf) and the pooled potrf
// workspace `solver_work_`.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). Joins the SAME
// steppe_device target, so it inherits identical codegen/macros/RDC.
#include <algorithm>  // std::copy / std::fill — the solution-slice copy + the partial-NaN rhs/Adown reset (NOT in §2.3 TU-F list; carried so the move compiles)
#include <cmath>      // std::isfinite — the partial-NaN comb-row test in the downdate fallback
#include <stdexcept>  // (per the §2.3 TU-F include set; the qpfstats bodies themselves use no throw)
#include <vector>     // std::vector — host nan_per_col / sol / b / bglob / Abase / series staging buffers

#include "core/domain/block_partition_rule.hpp"        // core::block_ranges / core::BlockRange (per-block SNP layout; the assign_blocks inverse)
#include "core/internal/nvtx.hpp"                       // STEPPE_NVTX_RANGE (coarse "qpfstats" phase-boundary marker)
#include "core/internal/qpfstats_jackknife.hpp"         // core::f2blocks_pair_est (the cold partial-NaN recenter recompute)
#include "core/internal/small_linalg.hpp"               // core::solve / core::LinAlgStatus (the host downdated-A partial-NaN solve)
#include "device/cuda/cuda_backend.cuh"                 // the CudaBackend class declaration (split T0)
#include "device/cuda/check.cuh"                        // STEPPE_CUDA_CHECK / CUBLAS_CHECK / CUSOLVER_CHECK
#include "device/cuda/device_decode_result_impl.cuh"   // DeviceDecodeResult::Impl (resident Q/V owners behind dec.q_device/v_device)
#include "device/cuda/dstat_kernel.cuh"                 // launch_dstat_block_reduce (the genotype-f4 numerator per-block reduce)
#include "device/cuda/f2_block_kernel.cuh"              // MathModeScope/engage_f2_precision + CusolverMathModeScope/engage_solver_precision/emulation_honorable (the §12 scoped precision)
#include "device/cuda/qpadm_fit_kernels.cuh"            // launch_symmetrize_lower_to_full (the lower→full gram symmetrize; declared HERE, NOT in §2.3 TU-F — carried so the move compiles)
#include "device/cuda/qpfstats_jackknife_kernel.cuh"   // launch_qpfstats_numer_jackknife / _recenter_shift (the fused on-device PERF path)
#include "device/cuda/qpfstats_kernel.cuh"              // launch_qpfstats_zero_nan_ymat / _add_ridge_diag (the smoother solve prep)
#include "steppe/config.hpp"                            // Precision (the typed precision config in every compute sig)

namespace steppe::device {

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
    const int ncols = n_block + 1;  // the n_block ymat columns + the bglob (y) column
    const std::size_t ncols_sz = static_cast<std::size_t>(ncols);

    // ---- Upload x [npopcomb × npairs] + the RHS source [npopcomb × ncols] -----------
    // RHS source columns 0..n_block-1 = ymat; column n_block = y. NaN entries are zeroed
    // on device (launch_qpfstats_zero_nan_ymat); the per-column NaN count rides back so
    // the host can detect any PARTIAL-NaN column for the downdate fallback.
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

    // ---- A_shared = x'x + ridge·I  (cublasDsyrk, EmulatedFp64 via the f2 policy) -----
    // x is [npopcomb × npairs] COLUMN-MAJOR ⇒ x'x is the OP_T·OP_N gram = syrk(OP_T,
    // n=npairs, k=npopcomb, A=dX lda=npopcomb). The well-conditioned gram ENGAGES
    // `precision` exactly like the f2/jackknife SYRK (engage_f2_precision +
    // MathModeScope; the §12 scoped emulated-FP64 pattern). Then symmetrize + add ridge.
    DeviceBuffer<double> dA(np * np);
    {
        const double alpha = 1.0, beta = 0.0;
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                 npairs, npopcomb, &alpha, dX.data(), npopcomb,
                                 &beta, dA.data(), npairs));
    }  // restores the math mode before the native Cholesky/Dtrsm
    launch_symmetrize_lower_to_full(dA.data(), npairs, stream_.get());
    launch_qpfstats_add_ridge_diag(dA.data(), npairs, ridge, stream_.get());

    // ---- RHS = x' · RhsSrc  (cublasDgemm, [npairs × ncols], EmulatedFp64) -----------
    // x' [npairs × npopcomb] · RhsSrc [npopcomb × ncols] = RHS [npairs × ncols].
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

    // ---- L = chol(A_shared)  (cusolverDnDpotrf LOWER; native FP64 carve-out) ---------
    // A_shared = L·Lᵀ. The ill-conditioned solve DEFAULTS native (solve_precision_=Fp64);
    // promotable via set_solve_precision under per-stage S8 validation (the f2-cache fit
    // policy). The scope is local so it never leaks the mode to the next op.
    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    DeviceBuffer<int> dInfo(1);
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                               npairs, dA.data(), npairs, &lwork));
    // POOLED potrf workspace (B5): grow the persistent solver_work_ to lwork
    // (monotonic, never-shrink) instead of a per-call cudaMalloc/cudaFree. Same
    // bytes fed to potrf ⇒ bit-identical (§12). lwork is queried AFTER the
    // math-mode scope above so it reflects the engaged mode (CUDA 13.x cuSOLVER).
    const std::size_t lwork_need = static_cast<std::size_t>(lwork > 0 ? lwork : 1);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, npairs,
                                    dA.data(), npairs, solver_work_.data(), lwork, dInfo.data()));
    int info = 0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) { out.status = Status::NonSpdCovariance; return out; }

    // ---- The multi-column solve A·B = RHS via a cublasDtrsm PAIR (NOT potrsBatched,
    // which is nrhs=1-only): A = L·Lᵀ ⇒ forward L·Z = RHS (lower, no-trans) then back
    // Lᵀ·B = Z (lower, trans). ALL ncols columns in ONE Dtrsm each — the whole n_block
    // axis in two calls, NO host per-block loop. dRhs is overwritten with the solution B.
    {
        const double one = 1.0;
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
    }

    // ---- D2H the solution: b = columns 0..n_block-1, bglob = column n_block ----------
    std::vector<double> sol(np * ncols_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(sol.data(), dRhs.data(), np * ncols_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.b.assign(np * nb, 0.0);
    std::copy(sol.begin(), sol.begin() + static_cast<std::ptrdiff_t>(np * nb), out.b.begin());
    out.bglob.assign(np, 0.0);
    std::copy(sol.begin() + static_cast<std::ptrdiff_t>(np * nb), sol.end(),
              out.bglob.begin());

    // ---- PARTIAL-NaN downdate fallback (0 < k_col < npopcomb): re-solve ONLY those
    // columns with A_b = A_shared - x[nan]'x[nan]. ABSENT on the real golden (block 536
    // is all-NaN, handled by the zero-RHS shared solve above) ⇒ this loop runs ZERO
    // iterations in production. When present it is a host small_linalg solve over the few
    // partial-NaN columns (NOT all blocks), the AT2 generic-solve branch — bit-faithful.
    bool any_partial = false;
    for (int col = 0; col < ncols; ++col)
        if (nan_per_col[static_cast<std::size_t>(col)] > 0 &&
            nan_per_col[static_cast<std::size_t>(col)] < npopcomb) { any_partial = true; break; }
    if (any_partial) {
        // A_shared (host): x'x + ridge·I (recompute on host; tiny npairs×npairs).
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
            if (kc <= 0 || kc >= npopcomb) continue;  // skip no-NaN + all-NaN (already correct)
            // rhs = x' · (this column's source with NaN→0). The source is dRhsSrc col
            // (already zeroed on device) — recompute from the host inputs (col<n_block:
            // ymat; col==n_block: y), zeroing NaN, to match exactly.
            std::fill(rhs.begin(), rhs.end(), 0.0);
            Adown = Abase;
            for (int c = 0; c < npopcomb; ++c) {
                const double sv = (col < n_block)
                    ? ymat[nc * static_cast<std::size_t>(col) + static_cast<std::size_t>(c)]
                    : y[static_cast<std::size_t>(c)];
                if (!std::isfinite(sv)) {
                    // downdate A by this NaN row's outer product.
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

QpfstatsSmooth CudaBackend::qpfstats_blocks_smooth(
    const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
    std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
    std::span<const int> block_sizes, double ridge, const Precision& precision) {
    guard_device();
    STEPPE_NVTX_RANGE("qpfstats");  // coarse phase boundary: fused reduce->jackknife->smooth->recenter
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

    // ---- Per-block contiguous SNP layout (the SINGLE-SOURCE inverse of assign_blocks; the
    // SAME primitive dstat_block_reduce uses). begin/size as int (M fits int by B22). ----
    const std::vector<core::BlockRange> ranges =
        core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                           M, n_block);
    std::vector<int> begin(nbb), size(nbb);
    for (int b = 0; b < n_block; ++b) {
        begin[static_cast<std::size_t>(b)] = static_cast<int>(ranges[static_cast<std::size_t>(b)].begin);
        size[static_cast<std::size_t>(b)]  = static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }

    const std::size_t nq = static_cast<std::size_t>(N) * 4u;

    // ---- Upload the quad table + block layout + the design x (col-major). Q/V are
    // BORROWED resident pointers (the host overload uploaded them; the device
    // overload passes the resident DeviceDecodeResult Q/V — NO Q/V H2D here). ----
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

    // ---- 1. The genotype-f4 numerator reduce → dNum/dCnt RESIDENT (densum ignored). ----
    DeviceBuffer<double> dNum(nb_out), dDen(nb_out), dCnt(nb_out);
    launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                              dBegin.data(), dSize.data(), n_block,
                              dNum.data(), dDen.data(), dCnt.data(), stream_.get());

    // ---- 2. numer + ymat (col-major) + the GLOBAL per-comb jackknife y, ON-DEVICE. -----
    // The RHS source for the solve is [npopcomb × (n_block+1)]: columns 0..n_block-1 = ymat,
    // column n_block = y. We write ymat into the first nc*nb of dRhsSrc and y into the last.
    const int ncols = n_block + 1;
    const std::size_t ncols_sz = static_cast<std::size_t>(ncols);
    DeviceBuffer<double> dNumer(nb_out);
    DeviceBuffer<double> dRhsSrc(nc * ncols_sz);
    launch_qpfstats_numer_jackknife(dNum.data(), dCnt.data(), npopcomb, n_block,
                                    dNumer.data(), dRhsSrc.data(),
                                    dRhsSrc.data() + nc * nbb, stream_.get());

    // ---- The shared-factor batched smoothing solve, reading dRhsSrc straight from VRAM
    // (IDENTICAL math to qpfstats_smooth; NO ymat/y H2D re-upload). Zero the NaN entries +
    // count NaN comb-rows per column (the AT2 ymat_chunk[nan]=0).
    DeviceBuffer<int> dNanPerCol(ncols_sz);
    launch_qpfstats_zero_nan_ymat(dRhsSrc.data(), npopcomb, ncols, dNanPerCol.data(),
                                  stream_.get());
    std::vector<int> nan_per_col(ncols_sz, 0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(nan_per_col.data(), dNanPerCol.data(),
                                      ncols_sz * sizeof(int), cudaMemcpyDeviceToHost,
                                      stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // A_shared = x'x + ridge·I  (cublasDsyrk, EmulatedFp64 via the f2 policy).
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

    // RHS = x' · RhsSrc  (cublasDgemm, [npairs × ncols], EmulatedFp64).
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

    // L = chol(A_shared) (cusolverDnDpotrf LOWER; native FP64 carve-out).
    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    DeviceBuffer<int> dInfo(1);
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                               npairs, dA.data(), npairs, &lwork));
    // POOLED potrf workspace (B5): grow the persistent solver_work_ to lwork
    // (monotonic, never-shrink); same bytes fed to potrf ⇒ bit-identical (§12).
    const std::size_t lwork_need = static_cast<std::size_t>(lwork > 0 ? lwork : 1);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, npairs,
                                    dA.data(), npairs, solver_work_.data(), lwork, dInfo.data()));
    int info = 0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) { out.status = Status::NonSpdCovariance; return out; }

    // The multi-column solve A·B = RHS via a cublasDtrsm PAIR over ALL ncols at once.
    {
        const double one = 1.0;
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
        CUBLAS_CHECK(cublasDtrsm(blas_.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, npairs, ncols, &one,
                                 dA.data(), npairs, dRhs.data(), npairs));
    }
    // dRhs now holds the solution B: columns 0..n_block-1 = b, column n_block = bglob.
    // It STAYS RESIDENT so the recenter kernel reads b/bglob straight from VRAM.

    // ---- The per-pair RECENTER jackknife ON-DEVICE → dShift (reads the resident b/bglob).
    DeviceBuffer<int> dBlockSizes(nbb);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                      nbb * sizeof(int), cudaMemcpyHostToDevice,
                                      stream_.get()));
    DeviceBuffer<double> dShift(np);
    launch_qpfstats_recenter_shift(dRhs.data(), dRhs.data() + np * nbb, dBlockSizes.data(),
                                   npairs, n_block, dShift.data(), stream_.get());

    // ---- D2H ONLY the small outputs: b, bglob, recenter_shift. --------------------------
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

    // ---- PARTIAL-NaN downdate fallback (0 < k_col < npopcomb): the AT2 generic-solve
    // branch. ABSENT on the real golden (block 536 is all-NaN ⇒ the zero-RHS shared solve
    // is exact) ⇒ ZERO iterations in production. When present, re-solve ONLY those columns
    // host-side with A_b = A_shared - x[nan]'x[nan]; the recenter is then recomputed on the
    // corrected b. Bit-faithful to qpfstats_smooth's fallback. We D2H ymat/y ONLY here
    // (the cold path) to feed the host recompute — the hot path never pays this.
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
        // Recompute the recenter on the corrected b (host; the same f2blocks_pair_est ref).
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
