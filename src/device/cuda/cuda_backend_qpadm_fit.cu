// src/device/cuda/cuda_backend_qpadm_fit.cu
//
// Out-of-line CUDA definitions for the CudaBackend qpAdm fit engine: the block
// jackknife, the small/large fit paths, the rank sweep, GLS weights with
// leave-one-out standard errors, and the batched multi-model fit. The class is
// declared in cuda_backend.cuh; this is a private steppe_device translation unit.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_qpadm_fit.cu.md
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "core/internal/pchisq.hpp"
#include "core/qpadm/qpadm_bounds.hpp"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/device_f2_blocks_impl.cuh"
#include "device/cuda/f2_block_kernel.cuh"
#include "device/cuda/qpadm_fit_kernels.cuh"
#include "steppe/config.hpp"

namespace steppe::device {

namespace {

// shared rank-drop table fill — reference §14
void fill_rankdrop(int rmax,
                   const std::vector<int>& dof, const std::vector<double>& chisq,
                   const std::vector<double>& p,
                   std::vector<int>& rd_f4rank, std::vector<int>& rd_dof,
                   std::vector<int>& rd_dofdiff, std::vector<double>& rd_chisq,
                   std::vector<double>& rd_p, std::vector<double>& rd_chisqdiff,
                   std::vector<double>& rd_p_nested) {
    const std::size_t n = static_cast<std::size_t>(rmax) + 1;
    rd_f4rank.resize(n); rd_dof.resize(n); rd_chisq.resize(n);
    rd_p.resize(n); rd_dofdiff.resize(n);
    rd_chisqdiff.resize(n); rd_p_nested.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        const int r = rmax - static_cast<int>(k);
        rd_f4rank[k] = r;
        rd_dof[k] = dof[static_cast<std::size_t>(r)];
        rd_chisq[k] = chisq[static_cast<std::size_t>(r)];
        rd_p[k] = p[static_cast<std::size_t>(r)];
        if (r - 1 >= 0) {
            const int dd = dof[static_cast<std::size_t>(r - 1)] - dof[static_cast<std::size_t>(r)];
            const double cd = chisq[static_cast<std::size_t>(r - 1)] -
                              chisq[static_cast<std::size_t>(r)];
            rd_dofdiff[k] = dd;
            rd_chisqdiff[k] = cd;
            rd_p_nested[k] = core::internal::pchisq_upper(cd, dd);
        } else {
            rd_dofdiff[k] = INT_MIN;
            rd_chisqdiff[k] = std::numeric_limits<double>::quiet_NaN();
            rd_p_nested[k] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

}  // namespace


// block jackknife covariance — reference §3
JackknifeCov CudaBackend::jackknife_cov(const F4Blocks& x,
                                         std::span<const int> block_sizes,
                                         double fudge,
                                         const Precision& precision) {
    guard_device();

    const int m = x.nl * x.nr;
    const int nb = x.n_block;
    JackknifeCov out;
    out.m = m;
    if (m <= 0 || nb <= 0) { out.status = Status::Ok; return out; }
    const std::size_t m_sz = static_cast<std::size_t>(m);

    long long n_ll = 0;
    for (int b = 0; b < nb; ++b) n_ll += block_sizes[static_cast<std::size_t>(b)];
    const double n = static_cast<double>(n_ll);

    DeviceBuffer<double> dLoo(m_sz * static_cast<std::size_t>(nb));
    DeviceBuffer<double> dEst(m_sz);
    DeviceBuffer<double> dTotLine(m_sz);
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb));
    DeviceBuffer<double> dXtau(m_sz * static_cast<std::size_t>(nb));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLoo.data(), x.x_loo.data(),
                                      m_sz * static_cast<std::size_t>(nb) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dEst.data(), x.x_total.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotLine.data(), tot_line_.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                      static_cast<std::size_t>(nb) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_f4_xtau(dLoo.data(), dEst.data(), dTotLine.data(), dBlockSizes.data(),
                   m, nb, n, dXtau.data(), stream_.get());

    DeviceBuffer<double> dQ(m_sz * m_sz);
    const double alpha = 1.0 / static_cast<double>(nb);
    const double beta = 0.0;
    {
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                 m, nb, &alpha, dXtau.data(), m, &beta, dQ.data(), m));
    }
    launch_symmetrize_lower_to_full(dQ.data(), m, stream_.get());
    out.Q.assign(m_sz * m_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.Q.data(), dQ.data(), m_sz * m_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    double tr = 0.0;
    for (int k = 0; k < m; ++k) tr += out.Q[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(k)];
    DeviceBuffer<double> dQf(m_sz * m_sz);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQf.data(), dQ.data(), m_sz * m_sz * sizeof(double),
                                      cudaMemcpyDeviceToDevice, stream_.get()));
    launch_add_fudge_diag(dQf.data(), m, fudge, tr, stream_.get());

    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    DeviceBuffer<int> dInfo(1);
    int lwork_f = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                               m, dQf.data(), m, &lwork_f));
    int lwork_i = 0;
    CUSOLVER_CHECK(cusolverDnDpotri_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                               m, dQf.data(), m, &lwork_i));
    const int lwork_max = std::max(lwork_f, lwork_i);
    const std::size_t lwork_need = static_cast<std::size_t>(lwork_max > 0 ? lwork_max : 1);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                    dQf.data(), m, solver_work_.data(), lwork_f, dInfo.data()));
    int info = 0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) {
        out.status = Status::NonSpdCovariance;
        return out;
    }
    CUSOLVER_CHECK(cusolverDnDpotri(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                    dQf.data(), m, solver_work_.data(), lwork_i, dInfo.data()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) {
        out.status = Status::NonSpdCovariance;
        return out;
    }
    launch_symmetrize_lower_to_full(dQf.data(), m, stream_.get());
    out.Qinv.assign(m_sz * m_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.Qinv.data(), dQf.data(), m_sz * m_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.status = Status::Ok;
    return out;
}

// block jackknife diagonal — reference §4
JackknifeDiag CudaBackend::jackknife_diag(const F4Blocks& x,
                                           std::span<const int> block_sizes,
                                           const Precision& precision) {
    (void)precision;
    guard_device();

    const int m = x.nl * x.nr;
    const int nb = x.n_block;
    JackknifeDiag out;
    out.m = m;
    if (m <= 0 || nb <= 0) { out.status = Status::Ok; return out; }
    const std::size_t m_sz = static_cast<std::size_t>(m);

    long long n_ll = 0;
    for (int b = 0; b < nb; ++b) n_ll += block_sizes[static_cast<std::size_t>(b)];
    const double n = static_cast<double>(n_ll);

    DeviceBuffer<double> dLoo(m_sz * static_cast<std::size_t>(nb));
    DeviceBuffer<double> dEst(m_sz);
    DeviceBuffer<double> dTotLine(m_sz);
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb));
    DeviceBuffer<double> dXtau(m_sz * static_cast<std::size_t>(nb));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLoo.data(), x.x_loo.data(),
                                      m_sz * static_cast<std::size_t>(nb) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dEst.data(), x.x_total.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotLine.data(), tot_line_.data(),
                                      m_sz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                      static_cast<std::size_t>(nb) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_f4_xtau(dLoo.data(), dEst.data(), dTotLine.data(), dBlockSizes.data(),
                   m, nb, n, dXtau.data(), stream_.get());

    DeviceBuffer<double> dVar(m_sz);
    launch_f4_diag_var(dXtau.data(), m, nb, dVar.data(), stream_.get());

    out.var.assign(m_sz, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.var.data(), dVar.data(), m_sz * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.status = Status::Ok;
    return out;
}

// small-vs-large fit-path predicates — reference §5
bool CudaBackend::model_fits_small_path(int nl, int nr, int r) {
    return core::qpadm::model_fits_small_path(nl, nr, r);
}

bool CudaBackend::gesvdj_applicable(int nl, int nr) {
    return nl <= kGesvdjMaxDim && nr <= kGesvdjMaxDim;
}

// large-path SVD (compute V) — reference §6
CudaBackend::SvdScratchSizes CudaBackend::large_svd_scratch_sizes(int nl, int nr) {
    SvdScratchSizes sz;
    const int rows = (nr >= nl) ? nr : nl;
    const int cols = (nr >= nl) ? nl : nr;
    sz.s  = static_cast<std::size_t>(cols);
    sz.u  = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    sz.vt = static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols);
    sz.a2 = (nr >= nl) ? 0
                       : static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    solver_.set_stream(stream_.get());
    if (gesvdj_applicable(nl, nr)) {
        GesvdjInfo params;
        CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
            solver_.get(), CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, rows, cols,
            nullptr, rows, nullptr, nullptr, rows, nullptr, cols, &sz.lwork,
            params.get()));
    } else {
        CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(solver_.get(), rows, cols, &sz.lwork));
    }
    if (sz.lwork <= 0) sz.lwork = 1;
    return sz;
}

void CudaBackend::large_svd_V(const double* dXmat, int nl, int nr, int r,
                 double* dVout, double* dXt,
                 double* sS, double* sU, double* sVt, double* sA2,
                 int* sInfo, double* sWork, int lwork, cudaStream_t stream) {
    if (r <= 0) return;
    solver_.set_stream(stream);
    if (nr >= nl) {
        launch_transpose_small(dXmat, nl, nr, dXt, stream);
        const int rows = nr, cols = nl;
        if (gesvdj_applicable(nl, nr)) {
            GesvdjInfo params;
            CUSOLVER_CHECK(cusolverDnDgesvdj(
                solver_.get(), CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, rows, cols,
                dXt, rows, sS, sU, rows, sVt, cols,
                sWork, lwork, sInfo, params.get()));
        } else {
            CUSOLVER_CHECK(cusolverDnDgesvd(
                solver_.get(), 'S', 'N', rows, cols, dXt, rows, sS,
                sU, rows, /*VT*/nullptr, cols, sWork, lwork,
                /*rwork*/nullptr, sInfo));
        }
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(
            dVout, sU,
            static_cast<std::size_t>(nr) * static_cast<std::size_t>(r) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
    } else {
        const int rows = nl, cols = nr;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(
            sA2, dXmat,
            static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
        if (gesvdj_applicable(nl, nr)) {
            GesvdjInfo params;
            CUSOLVER_CHECK(cusolverDnDgesvdj(
                solver_.get(), CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, rows, cols,
                sA2, rows, sS, sU, rows, sVt, cols,
                sWork, lwork, sInfo, params.get()));
        } else {
            CUSOLVER_CHECK(cusolverDnDgesvd(
                solver_.get(), 'N', 'S', rows, cols, sA2, rows, sS,
                /*U*/nullptr, rows, sVt, cols, sWork, lwork,
                /*rwork*/nullptr, sInfo));
        }
        launch_transpose_small(sVt, cols, cols, dXt, stream);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(
            dVout, dXt,
            static_cast<std::size_t>(nr) * static_cast<std::size_t>(r) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
    }
}

void CudaBackend::large_svd_V(const double* dXmat, int nl, int nr, int r,
                 double* dVout, double* dXt, cudaStream_t stream) {
    if (r <= 0) return;
    const SvdScratchSizes sz = large_svd_scratch_sizes(nl, nr);
    const std::size_t lwork_need = static_cast<std::size_t>(sz.lwork);
    if (svd_s_.size()       < sz.s)       svd_s_       = DeviceBuffer<double>(sz.s);
    if (svd_u_.size()       < sz.u)       svd_u_       = DeviceBuffer<double>(sz.u);
    if (svd_vt_.size()      < sz.vt)      svd_vt_      = DeviceBuffer<double>(sz.vt);
    if (svd_a2_.size()      < sz.a2)      svd_a2_      = DeviceBuffer<double>(sz.a2);
    if (svd_info_.size()    < sz.info)    svd_info_    = DeviceBuffer<int>(sz.info);
    if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
    large_svd_V(dXmat, nl, nr, r, dVout, dXt,
                svd_s_.data(), svd_u_.data(), svd_vt_.data(), svd_a2_.data(),
                svd_info_.data(), solver_work_.data(), sz.lwork, stream);
}

// large-path scratch-size helpers — reference §7
std::size_t CudaBackend::large_dbl_scratch(int nl, int nr, int r) {
    const std::size_t m = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    const std::size_t t = static_cast<std::size_t>(nl > nr ? nl : nr) *
                          static_cast<std::size_t>(r > 0 ? r : 1);
    const std::size_t als = m + m * t + t * t + t + t + t * t + t;
    const std::size_t weight_chisq =
                          static_cast<std::size_t>(nl) * static_cast<std::size_t>(nl)
                          + static_cast<std::size_t>(nl)
                          + static_cast<std::size_t>(nl) * static_cast<std::size_t>(nl)
                          + static_cast<std::size_t>(nl) + m;
    return als > weight_chisq ? als : weight_chisq;
}

std::size_t CudaBackend::large_int_scratch(int nl, int nr, int r) {
    const std::size_t t = static_cast<std::size_t>(nl > nr ? nl : nr) *
                          static_cast<std::size_t>(r > 0 ? r : 1);
    return (t > static_cast<std::size_t>(nl) ? t : static_cast<std::size_t>(nl));
}

std::size_t CudaBackend::large_loo_dbl_refit(int nl, int nr, int r) {
    const std::size_t m  = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    const std::size_t rr = static_cast<std::size_t>(r > 0 ? r : 1);
    const std::size_t a  = static_cast<std::size_t>(nl) * rr;
    const std::size_t bb = rr * static_cast<std::size_t>(nr);
    return m + a + bb + large_dbl_scratch(nl, nr, r);
}

// single-model large-path fit — reference §8
void CudaBackend::large_fit_one(const double* dXmat, const double* dQinv, int nl, int nr, int r,
                   double fudge, int als_iters, double* dA, double* dB, double* dW,
                   double* dchisq, int* dStatus, double* dVout, double* dXt,
                   double* dScratch, int* dIntScratch, cudaStream_t stream) {
    if (r > 0) {
        large_svd_V(dXmat, nl, nr, r, dVout, dXt, stream);
        launch_qpadm_seed_from_V(dXmat, dVout, nl, nr, r, dA, dB, stream);
        launch_qpadm_als_large(dXmat, dQinv, nl, nr, r, fudge, als_iters,
                               dA, dB, dScratch, dIntScratch, stream);
    }
    launch_qpadm_weights_chisq_large(dXmat, dQinv, dA, dB, nl, nr, r,
                                     dW, dchisq, dStatus, dScratch, dIntScratch, stream);
}

// rank sweep — reference §9
bool CudaBackend::provides_rank_sweep() const { return true; }

RankSweep CudaBackend::rank_sweep(const F4Blocks& x,
                                   const JackknifeCov& cov,
                                   double alpha,
                                   const QpAdmOptions& opts,
                                   const Precision& precision) {
    (void)precision;
    guard_device();
    RankSweep rs;
    const int nl = x.nl, nr = x.nr, m = nl * nr;
    const int rmax = (nl < nr ? nl : nr) - 1;
    rs.svd_path = gesvdj_applicable(nl, nr) ? 1 : 2;
    if (rmax < 0) {
        rs.status = Status::RankDeficient;
        return rs;
    }
    const bool small = model_fits_small_path(nl, nr, rmax);
    if (m <= 0) { rs.status = Status::Ok; return rs; }

    DeviceBuffer<double> dTotal(static_cast<std::size_t>(m));
    DeviceBuffer<double> dXmat(static_cast<std::size_t>(m));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    const std::size_t cap = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    DeviceBuffer<double> dA(cap > 0 ? cap : 1);
    DeviceBuffer<double> dB(cap > 0 ? cap : 1);
    DeviceBuffer<double> dW(static_cast<std::size_t>(nl > 0 ? nl : 1));
    DeviceBuffer<double> dchisq(static_cast<std::size_t>(rmax) + 1);
    DeviceBuffer<int> dStatus(static_cast<std::size_t>(rmax) + 1);
    DeviceBuffer<double> dVout(small ? 1 : static_cast<std::size_t>(nr) * static_cast<std::size_t>(rmax > 0 ? rmax : 1));
    DeviceBuffer<double> dXt(small ? 1 : static_cast<std::size_t>(nr) * static_cast<std::size_t>(nl > 0 ? nl : 1));
    DeviceBuffer<double> dScratch(small ? 1 : large_dbl_scratch(nl, nr, rmax));
    DeviceBuffer<int>    dIntScratch(small ? 1 : large_int_scratch(nl, nr, rmax));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotal.data(), x.x_total.data(),
                                      static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), cov.Qinv.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_qpadm_xmat_from_rowmajor(dTotal.data(), nl, nr, dXmat.data(), stream_.get());

    const std::size_t n = static_cast<std::size_t>(rmax) + 1;
    rs.chisq.assign(n, 0.0);
    rs.dof.assign(n, 0);
    rs.p.assign(n, 0.0);
    bool degenerate = false;
    for (int r = 0; r <= rmax; ++r) {
        if (small) {
            if (r > 0) {
                launch_qpadm_seed_ab(dXmat.data(), nl, nr, r, dA.data(), dB.data(), stream_.get());
                launch_qpadm_als(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                                 opts.als_iterations, dA.data(), dB.data(), stream_.get());
            }
            launch_qpadm_weights_chisq(dXmat.data(), dQinv.data(), dA.data(), dB.data(),
                                       nl, nr, r, dW.data(),
                                       dchisq.data() + r, dStatus.data() + r,
                                       stream_.get());
        } else {
            large_fit_one(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                          opts.als_iterations, dA.data(), dB.data(), dW.data(),
                          dchisq.data() + r, dStatus.data() + r, dVout.data(), dXt.data(),
                          dScratch.data(), dIntScratch.data(), stream_.get());
        }
    }
    std::vector<double> chisq_h(n);
    std::vector<int> status_h(n);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(status_h.data(), dStatus.data(),
                                      n * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(chisq_h.data(), dchisq.data(),
                                      n * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    for (int r = 0; r <= rmax; ++r) {
        const std::size_t rr = static_cast<std::size_t>(r);
        if (status_h[rr] != 0) degenerate = true;
        rs.chisq[rr] = chisq_h[rr];
        rs.dof[rr] = core::qpadm::qpadm_dof(nl, nr, r);
        rs.p[rr] = core::internal::pchisq_upper(chisq_h[rr], rs.dof[rr]);
    }

    fill_rankdrop(rmax, rs.dof, rs.chisq, rs.p,
                  rs.rd_f4rank, rs.rd_dof, rs.rd_dofdiff, rs.rd_chisq,
                  rs.rd_p, rs.rd_chisqdiff, rs.rd_p_nested);

    rs.f4rank = rmax;
    for (int r = 0; r <= rmax; ++r) {
        if (rs.p[static_cast<std::size_t>(r)] > alpha) { rs.f4rank = r; break; }
    }

    if (!cov.Q.empty() && cov.m == m && m > 0) {
        DeviceBuffer<double> dQfull(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
        DeviceBuffer<double> dRankScratch(static_cast<std::size_t>(m) * static_cast<std::size_t>(m)
                                          + static_cast<std::size_t>(m));
        DeviceBuffer<int> dRankOrder(static_cast<std::size_t>(m));
        DeviceBuffer<int> dRank(1);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQfull.data(), cov.Q.data(),
                                          static_cast<std::size_t>(m) * static_cast<std::size_t>(m)
                                              * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
        launch_qpadm_rank_via_jacobi(dQfull.data(), m,
                                     std::numeric_limits<double>::epsilon(),
                                     dRankScratch.data(), dRankOrder.data(),
                                     dRank.data(), stream_.get());
        int rk_h = m;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&rk_h, dRank.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        rs.rank_Q = rk_h;
    } else {
        rs.rank_Q = m;
    }

    rs.status = degenerate ? Status::RankDeficient
                           : (cov.status != Status::Ok ? cov.status : Status::Ok);
    return rs;
}

// GLS weights for one model — reference §10
GlsWeights CudaBackend::gls_weights(const F4Blocks& x,
                                     const JackknifeCov& cov,
                                     int r,
                                     const QpAdmOptions& opts,
                                     const Precision& precision) {
    (void)precision;
    guard_device();
    GlsWeights gw;
    gw.r = r;
    const int nl = x.nl, nr = x.nr, m = nl * nr;
    const bool small = model_fits_small_path(nl, nr, r);
    gw.A.assign(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r), 0.0);
    gw.B.assign(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
    gw.w.assign(static_cast<std::size_t>(nl), 0.0);
    if (m <= 0 || nl <= 0) { gw.status = Status::Ok; return gw; }

    DeviceBuffer<double> dTotal(static_cast<std::size_t>(m));
    DeviceBuffer<double> dXmat(static_cast<std::size_t>(m));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    DeviceBuffer<double> dA(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r > 0 ? r : 1));
    DeviceBuffer<double> dB(static_cast<std::size_t>(r > 0 ? r : 1) * static_cast<std::size_t>(nr));
    DeviceBuffer<double> dW(static_cast<std::size_t>(nl));
    DeviceBuffer<double> dchisq(1);
    DeviceBuffer<int> dStatus(1);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTotal.data(), x.x_total.data(),
                                      static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), cov.Qinv.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_qpadm_xmat_from_rowmajor(dTotal.data(), nl, nr, dXmat.data(), stream_.get());
    if (small) {
        if (r > 0) {
            launch_qpadm_seed_ab(dXmat.data(), nl, nr, r, dA.data(), dB.data(), stream_.get());
            launch_qpadm_als(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                             opts.als_iterations, dA.data(), dB.data(), stream_.get());
        }
        launch_qpadm_weights_chisq(dXmat.data(), dQinv.data(), dA.data(), dB.data(),
                                   nl, nr, r, dW.data(), dchisq.data(), dStatus.data(),
                                   stream_.get());
    } else {
        DeviceBuffer<double> dVout(static_cast<std::size_t>(nr) * static_cast<std::size_t>(r > 0 ? r : 1));
        DeviceBuffer<double> dXt(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nl > 0 ? nl : 1));
        DeviceBuffer<double> dScratch(large_dbl_scratch(nl, nr, r));
        DeviceBuffer<int>    dIntScratch(large_int_scratch(nl, nr, r));
        large_fit_one(dXmat.data(), dQinv.data(), nl, nr, r, opts.fudge,
                      opts.als_iterations, dA.data(), dB.data(), dW.data(),
                      dchisq.data(), dStatus.data(), dVout.data(), dXt.data(),
                      dScratch.data(), dIntScratch.data(), stream_.get());
    }
    int status_i = 0;
    double chisq = 0.0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&status_i, dStatus.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&chisq, dchisq.data(), sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(gw.w.data(), dW.data(),
                                      static_cast<std::size_t>(nl) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    if (r > 0) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(gw.A.data(), dA.data(),
                                          gw.A.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(gw.B.data(), dB.data(),
                                          gw.B.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
    }
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    gw.chisq = chisq;
    if (status_i == core::qpadm::kQpStatusRankDeficient) { gw.status = Status::RankDeficient; return gw; }
    gw.status = Status::Ok;
    return gw;
}

// leave-one-out standard errors — reference §11
std::vector<double> CudaBackend::gls_weights_loo_batched(
    const F4Blocks& x, const JackknifeCov& cov, int r,
    const QpAdmOptions& opts, const Precision& precision) {
    guard_device();
    const int nl = x.nl, nb = x.n_block;
    const int m = nl * x.nr;
    std::vector<double> wmat(static_cast<std::size_t>(nb < 0 ? 0 : nb) *
                             static_cast<std::size_t>(nl), 0.0);
    if (m <= 0 || nb <= 0 || nl <= 0) return wmat;
    DeviceBuffer<double> dWmat(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl));
    populate_loo_wmat_resident(x, cov, r, opts, precision, dWmat);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(wmat.data(), dWmat.data(),
                                      static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return wmat;
}

std::vector<double> CudaBackend::se_from_wmat(
    const F4Blocks& x, const JackknifeCov& cov, int r,
    const QpAdmOptions& opts, const Precision& precision) {
    guard_device();
    const int nl = x.nl, nb = x.n_block;
    const int m = nl * x.nr;
    std::vector<double> se(static_cast<std::size_t>(nl < 0 ? 0 : nl), 0.0);
    constexpr int kMinJackknifeBlocks = 2;
    if (m <= 0 || nb < kMinJackknifeBlocks || nl <= 0) return se;

    DeviceBuffer<double> dWmat(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl));
    populate_loo_wmat_resident(x, cov, r, opts, precision, dWmat);
    DeviceBuffer<double> dSe(static_cast<std::size_t>(nl));
    launch_qpadm_se_from_wmat_batched(dWmat.data(), nl, nb, /*n_models=*/1,
                                      dSe.data(), stream_.get());
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(se.data(), dSe.data(),
                                      static_cast<std::size_t>(nl) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    const double scale = static_cast<double>(nb - 1) / std::sqrt(static_cast<double>(nb));
    for (double& v : se) v *= scale;
    return se;
}

void CudaBackend::populate_loo_wmat_resident(const F4Blocks& x, const JackknifeCov& cov, int r,
                                const QpAdmOptions& opts, const Precision& precision,
                                DeviceBuffer<double>& dWmat) {
    (void)precision;
    const int nl = x.nl, nr = x.nr, m = nl * nr, nb = x.n_block;
    const bool small = model_fits_small_path(nl, nr, r);
    if (m <= 0 || nb <= 0 || nl <= 0) return;

    DeviceBuffer<double> dLoo(static_cast<std::size_t>(m) * static_cast<std::size_t>(nb));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLoo.data(), x.x_loo.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(nb) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), cov.Qinv.data(),
                                      static_cast<std::size_t>(m) * static_cast<std::size_t>(m) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (small) {
        launch_qpadm_loo_batched(dLoo.data(), dQinv.data(), nl, nr, r, opts.fudge,
                                 opts.als_iterations, nb, dWmat.data(), stream_.get());
    } else {
        const int n_models = 1;
        DeviceBuffer<double> dXmatB(static_cast<std::size_t>(m) * static_cast<std::size_t>(nb));
        DeviceBuffer<double> dVout(static_cast<std::size_t>(nr) * static_cast<std::size_t>(r > 0 ? r : 1) *
                                   static_cast<std::size_t>(nb));
        DeviceBuffer<double> dXt(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nl > 0 ? nl : 1) *
                                 static_cast<std::size_t>(nb));
        DeviceBuffer<double> dAseed(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r > 0 ? r : 1) *
                                    static_cast<std::size_t>(nb));
        DeviceBuffer<double> dBseed(static_cast<std::size_t>(r > 0 ? r : 1) * static_cast<std::size_t>(nr) *
                                    static_cast<std::size_t>(nb));
        const std::size_t dbl_refit = large_loo_dbl_refit(nl, nr, r);
        const std::size_t int_refit = large_int_scratch(nl, nr, r);
        const std::size_t n_refit   = static_cast<std::size_t>(nb) * static_cast<std::size_t>(n_models);
        DeviceBuffer<double> dScratch(dbl_refit * n_refit);
        DeviceBuffer<int>    dIntScratch(int_refit * n_refit);
        if (r > 0) {
            const SvdScratchSizes sz = large_svd_scratch_sizes(nl, nr);
            DeviceBuffer<double> dSvdS(sz.s);
            DeviceBuffer<double> dSvdU(sz.u);
            DeviceBuffer<double> dSvdVt(sz.vt);
            DeviceBuffer<double> dSvdA2(sz.a2);
            DeviceBuffer<int>    dSvdInfo(sz.info);
            DeviceBuffer<double> dSvdWork(static_cast<std::size_t>(sz.lwork));
            const int svd_lwork = sz.lwork;
            for (int b = 0; b < nb; ++b) {
                const std::size_t ob = static_cast<std::size_t>(b);
                double* xmat_b = dXmatB.data() + static_cast<std::size_t>(m) * ob;
                launch_qpadm_xmat_from_rowmajor(
                    dLoo.data() + static_cast<std::size_t>(m) * ob, nl, nr, xmat_b,
                    stream_.get());
                large_svd_V(xmat_b, nl, nr, r,
                            dVout.data() + static_cast<std::size_t>(nr) * r * ob,
                            dXt.data() + static_cast<std::size_t>(nr) * nl * ob,
                            dSvdS.data(), dSvdU.data(), dSvdVt.data(), dSvdA2.data(),
                            dSvdInfo.data(), dSvdWork.data(), svd_lwork,
                            stream_.get());
                launch_qpadm_seed_from_V(
                    xmat_b, dVout.data() + static_cast<std::size_t>(nr) * r * ob,
                    nl, nr, r,
                    dAseed.data() + static_cast<std::size_t>(nl) * r * ob,
                    dBseed.data() + static_cast<std::size_t>(r) * nr * ob,
                    stream_.get());
            }
        }
        launch_qpadm_loo_large_batched(
            dLoo.data(), dQinv.data(), dAseed.data(), dBseed.data(),
            nl, nr, r, opts.fudge, opts.als_iterations, nb, n_models,
            static_cast<long>(dbl_refit), static_cast<long>(int_refit),
            dScratch.data(), dIntScratch.data(), dWmat.data(), stream_.get());
    }
}

// batched multi-model fit — reference §12
bool CudaBackend::provides_batched_fit() const { return true; }

std::vector<QpAdmResult> CudaBackend::fit_models_batched(
    const steppe::device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models,
    const QpAdmOptions& opts,
    const Precision& precision) {
    guard_device();
    STEPPE_NVTX_RANGE("qpadm_fit");
    std::vector<QpAdmResult> results(models.size());
    if (models.empty()) return results;

    const Precision::Kind tag =
        (precision.kind == Precision::Kind::EmulatedFp64 &&
         capabilities().emulated_fp64_honorable)
            ? Precision::Kind::EmulatedFp64
            : Precision::Kind::Fp64;

    struct Key { int nl, nr, r; };
    std::vector<std::vector<std::size_t>> bucket_members;
    std::vector<Key> bucket_keys;
    for (std::size_t mi = 0; mi < models.size(); ++mi) {
        const QpAdmModel& mdl = models[mi];
        const int nl = static_cast<int>(mdl.left.size());
        const int nr = static_cast<int>(mdl.right.size()) - 1;
        const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
        std::size_t bk = bucket_keys.size();
        for (std::size_t k = 0; k < bucket_keys.size(); ++k)
            if (bucket_keys[k].nl == nl && bucket_keys[k].nr == nr &&
                bucket_keys[k].r == r) { bk = k; break; }
        if (bk == bucket_keys.size()) {
            bucket_keys.push_back(Key{nl, nr, r});
            bucket_members.emplace_back();
        }
        bucket_members[bk].push_back(mi);
    }

    const std::vector<int> surv = device_survivor_blocks(f2, f2.n_block, f2.P);
    const int nb_s = static_cast<int>(surv.size());
    const bool dropped = (nb_s != f2.n_block);
    std::vector<int> surv_block_sizes(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        surv_block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    if (dropped)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
    const int* d_surv = dropped ? dSurv.data() : nullptr;

    for (std::size_t bk = 0; bk < bucket_keys.size(); ++bk) {
        fit_one_bucket(f2, models, bucket_members[bk], bucket_keys[bk].nl,
                       bucket_keys[bk].nr, bucket_keys[bk].r, nb_s, surv_block_sizes,
                       d_surv, opts, precision, tag, results);
    }
    return results;
}

void CudaBackend::fit_one_bucket(const steppe::device::DeviceF2Blocks& f2,
                    std::span<const QpAdmModel> models,
                    const std::vector<std::size_t>& mem, int nl, int nr, int r,
                    int nb, const std::vector<int>& survivor_block_sizes,
                    const int* d_surv, const QpAdmOptions& opts,
                    const Precision& precision,
                    Precision::Kind tag, std::vector<QpAdmResult>& results) {
    if (mem.empty()) return;
    const int m = nl * nr;
    const int P = f2.P;
    const int rmax = (nl < nr ? nl : nr) - 1;
    const int r_fit = r;
    const std::size_t m_sz = static_cast<std::size_t>(m);
    const std::size_t Mm = m_sz * m_sz;

    long long n_ll = 0;
    for (int v : survivor_block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb > 0 ? nb : 1));
    if (nb > 0)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), survivor_block_sizes.data(),
                                          static_cast<std::size_t>(nb) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

    const bool se_pass = (opts.jackknife != JackknifePolicy::None);
    const std::size_t se_per_model_dbl =
        se_pass
            ? (m_sz * static_cast<std::size_t>(nb)
               + Mm
               + static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl)
               + static_cast<std::size_t>(nl))
            : 0;
    const std::size_t per_model_dbl =
        3 * m_sz * static_cast<std::size_t>(nb) + 4 * Mm +
        static_cast<std::size_t>(2 * nl + 1 + (rmax + 1) + (nl + 1) + nl) +
        se_per_model_dbl;
    const std::size_t per_model_bytes = per_model_dbl * sizeof(double) +
        static_cast<std::size_t>((nl + 1) + (nr + 1)) * sizeof(int) +
        sizeof(int) /*status*/ + 3 * sizeof(double*) /*ptr arrays*/ +
        (se_pass ? sizeof(int) : 0) /*dSurv (JK-gated)*/;
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kFitBudgetFreeVramFallbackBytes;
    const std::size_t headroom = kFitBudgetHeadroomBytes;
    std::size_t budget = (free_b > headroom) ? (free_b - headroom) : free_b / 2;
    std::size_t B_max = (per_model_bytes > 0) ? (budget / per_model_bytes) : mem.size();
    if (B_max < 1) B_max = 1;
    if (B_max > mem.size()) B_max = mem.size();

    for (std::size_t off = 0; off < mem.size(); off += B_max) {
        const std::size_t B = std::min(B_max, mem.size() - off);
        fit_chunk(f2, models, mem, off, B, nl, nr, r_fit, rmax, m, nb, P, n,
                  dBlockSizes.data(), d_surv, opts, precision, tag, results);
    }
}

void CudaBackend::fit_chunk(const steppe::device::DeviceF2Blocks& f2,
               std::span<const QpAdmModel> models,
               const std::vector<std::size_t>& mem, std::size_t off, std::size_t B,
               int nl, int nr, int r_fit, int rmax, int m, int nb, int P, double n,
               const int* dBlockSizes, const int* d_surv, const QpAdmOptions& opts,
               const Precision& precision, Precision::Kind tag,
               std::vector<QpAdmResult>& results) {
    ++batched_dispatch_count_;
    const std::size_t m_sz = static_cast<std::size_t>(m);
    const std::size_t Mm = m_sz * m_sz;
    const std::size_t Bnb = B * m_sz * static_cast<std::size_t>(nb);

    std::vector<int> h_left(B * static_cast<std::size_t>(nl + 1));
    std::vector<int> h_right(B * static_cast<std::size_t>(nr + 1));
    for (std::size_t j = 0; j < B; ++j) {
        const QpAdmModel& mdl = models[mem[off + j]];
        int* lp = h_left.data() + j * (nl + 1);
        lp[0] = mdl.target;
        for (int i = 0; i < nl; ++i) lp[i + 1] = mdl.left[static_cast<std::size_t>(i)];
        int* rp = h_right.data() + j * (nr + 1);
        for (int i = 0; i <= nr; ++i) rp[i] = mdl.right[static_cast<std::size_t>(i)];
    }
    DeviceBuffer<int> dLeft(h_left.size());
    DeviceBuffer<int> dRight(h_right.size());
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLeft.data(), h_left.data(),
                                      h_left.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRight.data(), h_right.data(),
                                      h_right.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));

    DeviceBuffer<double> dX(Bnb);
    DeviceBuffer<double> dLoo(Bnb);
    DeviceBuffer<double> dXtau(Bnb);
    DeviceBuffer<double> dTotal(B * m_sz);
    DeviceBuffer<double> dTotLine(B * m_sz);
    DeviceBuffer<double> dQ(B * Mm);
    DeviceBuffer<double> dQf(B * Mm);

    launch_assemble_f4_gather_models_batched(f2.f2_device(), P, dLeft.data(),
                                             dRight.data(), nl, nr, nb,
                                             static_cast<int>(B), d_surv, dX.data(),
                                             stream_.get());
    launch_f4_loo_total_models_batched(dX.data(), dBlockSizes, m, nb, n,
                                       static_cast<int>(B), dLoo.data(),
                                       dTotal.data(), dTotLine.data(), stream_.get());
    launch_f4_xtau_models_batched(dLoo.data(), dTotal.data(), dTotLine.data(),
                                  dBlockSizes, m, nb, n, static_cast<int>(B),
                                  dXtau.data(), stream_.get());

    {
        const double alpha = 1.0 / static_cast<double>(nb);
        const double beta = 0.0;
        const MathModeScope gemm_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, m, m, nb, &alpha,
            dXtau.data(), m, static_cast<long long>(m_sz) * nb,
            dXtau.data(), m, static_cast<long long>(m_sz) * nb, &beta,
            dQ.data(), m, static_cast<long long>(Mm),
            static_cast<int>(B)));
    }

    launch_add_fudge_diag_models_batched(dQ.data(), dQf.data(), m, opts.fudge,
                                         static_cast<int>(B), stream_.get());

    DeviceBuffer<double> dQinv(B * Mm);
    launch_fill_identity_batched(dQinv.data(), m, static_cast<int>(B), stream_.get());
    std::vector<double*> h_Aptr(B);
    for (std::size_t j = 0; j < B; ++j) h_Aptr[j] = dQf.data() + j * Mm;
    DeviceBuffer<double*> dAptr(B);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dAptr.data(), h_Aptr.data(),
                                      B * sizeof(double*), cudaMemcpyHostToDevice,
                                      stream_.get()));
    DeviceBuffer<int> dInfo(B);
    const CusolverMathModeScope solve_scope =
        engage_solver_precision(solver_.get(), solve_precision_, &emulation_honorable);
    CUSOLVER_CHECK(cusolverDnDpotrfBatched(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                           dAptr.data(), m, dInfo.data(),
                                           static_cast<int>(B)));
    std::vector<int> h_info(B, 0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_info.data(), dInfo.data(), B * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    DeviceBuffer<double*> dBptrAll(static_cast<std::size_t>(m) * B);
    std::vector<double*> h_BptrAll(static_cast<std::size_t>(m) * B);
    for (int c = 0; c < m; ++c) {
        for (std::size_t j = 0; j < B; ++j)
            h_BptrAll[static_cast<std::size_t>(c) * B + j] =
                dQinv.data() + j * Mm + static_cast<std::size_t>(c) * m_sz;
    }
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(
        dBptrAll.data(), h_BptrAll.data(),
        static_cast<std::size_t>(m) * B * sizeof(double*), cudaMemcpyHostToDevice,
        stream_.get()));
    DeviceBuffer<int> dSolveInfo(1);
    for (int c = 0; c < m; ++c) {
        CUSOLVER_CHECK(cusolverDnDpotrsBatched(
            solver_.get(), CUBLAS_FILL_MODE_LOWER, m, 1 /*nrhs*/, dAptr.data(), m,
            dBptrAll.data() + static_cast<std::size_t>(c) * B, m, dSolveInfo.data(),
            static_cast<int>(B)));
    }
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    DeviceBuffer<double> dWeight(B * static_cast<std::size_t>(nl));
    DeviceBuffer<double> dSe(B * static_cast<std::size_t>(nl));
    DeviceBuffer<double> dChisq(B);
    DeviceBuffer<int> dStatus(B);
    DeviceBuffer<double> dRankChisq(B * static_cast<std::size_t>(rmax + 1));
    DeviceBuffer<double> dPopChisq(B * static_cast<std::size_t>(nl + 1));
    DeviceBuffer<double> dPopWfull(B * static_cast<std::size_t>(nl));
    launch_qpadm_fit_models_batched(
        dTotal.data(), dQinv.data(), dLoo.data(), dBlockSizes, nl, nr, r_fit, rmax,
        opts.fudge, opts.als_iterations, nb, static_cast<int>(B),
        dWeight.data(), dSe.data(), dChisq.data(), dStatus.data(),
        dRankChisq.data(), dPopChisq.data(), dPopWfull.data(), stream_.get());


    std::vector<double> h_weight(B * nl), h_se(B * nl, 0.0), h_chisq(B),
        h_rankchisq(B * (rmax + 1)), h_popchisq(B * (nl + 1)), h_popwfull(B * nl);
    std::vector<int> h_status(B);
    auto d2h = [&](double* dst, const DeviceBuffer<double>& src, std::size_t cnt) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), cnt * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
    };
    d2h(h_weight.data(), dWeight, B * nl);
    d2h(h_chisq.data(), dChisq, B);
    d2h(h_rankchisq.data(), dRankChisq, B * (rmax + 1));
    d2h(h_popchisq.data(), dPopChisq, B * (nl + 1));
    d2h(h_popwfull.data(), dPopWfull, B * nl);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_status.data(), dStatus.data(), B * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    const int dof_full = core::qpadm::qpadm_dof(nl, nr, r_fit);
    auto feasible_from_wfull = [nl](const double* w) {
        bool any = false;
        for (int i = 0; i < nl; ++i) {
            const double v = w[i];
            if (std::isnan(v)) continue;
            any = true;
            if (v < 0.0 || v > 1.0) return false;
        }
        return any;
    };
    std::vector<char> se_computed(B, 0);
    std::vector<std::size_t> surv;
    surv.reserve(B);
    std::size_t n_eligible = 0;
    for (std::size_t j = 0; j < B; ++j) {
        const bool ok_fit = (h_info[j] == 0) && (h_status[j] == 0);
        if (!ok_fit) continue;
        ++n_eligible;
        bool survivor = false;
        switch (opts.jackknife) {
            case JackknifePolicy::None:
                survivor = false;
                break;
            case JackknifePolicy::FeasibleOnly: {
                const bool feas = feasible_from_wfull(h_popwfull.data() + j * nl);
                const double p = core::internal::pchisq_upper(h_chisq[j], dof_full);
                survivor = feas && (!opts.se_require_p || p >= opts.p_se_threshold);
                break;
            }
            case JackknifePolicy::All:
            default:
                survivor = true;
                break;
        }
        if (survivor) { surv.push_back(j); se_computed[j] = 1; }
    }

    if (nb >= 2 && !surv.empty()) {
        const double jackknife_scale =
            static_cast<double>(nb - 1) / std::sqrt(static_cast<double>(nb));
        const bool all_survive = (surv.size() == n_eligible) &&
                                 (n_eligible == B);
        if (all_survive) {
            DeviceBuffer<double> dWmat(B * static_cast<std::size_t>(nb) *
                                       static_cast<std::size_t>(nl));
            launch_qpadm_loo_models_batched(dLoo.data(), dQinv.data(), nl, nr, r_fit,
                                            opts.fudge, opts.als_iterations, nb,
                                            static_cast<int>(B), jackknife_scale,
                                            dWmat.data(),
                                            stream_.get());
            launch_qpadm_se_from_wmat_batched(dWmat.data(), nl, nb,
                                              static_cast<int>(B), dSe.data(),
                                              stream_.get());
            d2h(h_se.data(), dSe, B * nl);
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        } else {
            const std::size_t Sn = surv.size();
            std::vector<int> h_surv(Sn);
            for (std::size_t k = 0; k < Sn; ++k)
                h_surv[k] = static_cast<int>(surv[k]);
            DeviceBuffer<int> dSurv(Sn);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), h_surv.data(),
                                              Sn * sizeof(int), cudaMemcpyHostToDevice,
                                              stream_.get()));
            DeviceBuffer<double> dLooS(Sn * m_sz * static_cast<std::size_t>(nb));
            DeviceBuffer<double> dQinvS(Sn * Mm);
            launch_qpadm_gather_loo_qinv(dLoo.data(), dQinv.data(), dSurv.data(),
                                         m, nb, static_cast<int>(Sn), dLooS.data(),
                                         dQinvS.data(), stream_.get());
            DeviceBuffer<double> dWmatS(Sn * static_cast<std::size_t>(nb) *
                                        static_cast<std::size_t>(nl));
            DeviceBuffer<double> dSeS(Sn * static_cast<std::size_t>(nl));
            launch_qpadm_loo_models_batched(dLooS.data(), dQinvS.data(), nl, nr, r_fit,
                                            opts.fudge, opts.als_iterations, nb,
                                            static_cast<int>(Sn), jackknife_scale,
                                            dWmatS.data(),
                                            stream_.get());
            launch_qpadm_se_from_wmat_batched(dWmatS.data(), nl, nb,
                                              static_cast<int>(Sn), dSeS.data(),
                                              stream_.get());
            std::vector<double> h_seS(Sn * nl);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_seS.data(), dSeS.data(),
                                              Sn * nl * sizeof(double),
                                              cudaMemcpyDeviceToHost, stream_.get()));
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            for (std::size_t k = 0; k < Sn; ++k) {
                const std::size_t j = surv[k];
                for (int i = 0; i < nl; ++i)
                    h_se[j * nl + i] = h_seS[k * nl + i];
            }
        }
    }

    for (std::size_t j = 0; j < B; ++j) {
        const std::size_t pos = mem[off + j];
        const QpAdmModel& mdl = models[pos];
        assemble_result(mdl, nl, nr, r_fit, rmax, tag,
                        AssembleFlags{ .nonspd = h_info[j] != 0,
                                       .se_computed = se_computed[j] != 0 },
                        h_status[j],
                        h_weight.data() + j * nl, h_se.data() + j * nl, h_chisq[j],
                        h_rankchisq.data() + j * (rmax + 1),
                        h_popchisq.data() + j * (nl + 1),
                        h_popwfull.data() + j * nl,
                        opts.rank_alpha,
                        results[pos]);
    }
}

// assemble a result record — reference §13
void CudaBackend::assemble_result(const QpAdmModel& mdl, int nl, int nr, int r_fit, int rmax,
                     Precision::Kind tag, AssembleFlags flags, int fit_status,
                     const double* weight, const double* se, double chisq,
                     const double* rank_chisq, const double* pop_chisq,
                     const double* pop_wfull, double rank_alpha, QpAdmResult& res) {
    res.model_index = mdl.model_index;
    res.precision_tag = tag;
    res.est_rank = r_fit;
    res.dof = core::qpadm::qpadm_dof(nl, nr, r_fit);
    if (flags.nonspd) { res.status = Status::NonSpdCovariance; return; }
    if (fit_status != 0) { res.status = Status::RankDeficient; return; }

    res.weight.assign(weight, weight + nl);
    if (flags.se_computed) {
        res.se.assign(se, se + nl);
        res.z.assign(static_cast<std::size_t>(nl), 0.0);
        for (int i = 0; i < nl; ++i)
            res.z[static_cast<std::size_t>(i)] =
                (se[i] > 0.0) ? weight[i] / se[i] : 0.0;
    }
    res.chisq = chisq;
    res.p = core::internal::pchisq_upper(chisq, res.dof);
    res.rank_p.assign(static_cast<std::size_t>(r_fit) + 1, 0.0);
    if (r_fit >= 0 && static_cast<std::size_t>(r_fit) < res.rank_p.size())
        res.rank_p[static_cast<std::size_t>(r_fit)] = res.p;

    const std::size_t nrk = static_cast<std::size_t>(rmax) + 1;
    res.rank_chisq.assign(nrk, 0.0);
    res.rank_dof.assign(nrk, 0);
    std::vector<double> rankp(nrk, 0.0);
    for (int rr = 0; rr <= rmax; ++rr) {
        res.rank_chisq[static_cast<std::size_t>(rr)] = rank_chisq[rr];
        res.rank_dof[static_cast<std::size_t>(rr)] = core::qpadm::qpadm_dof(nl, nr, rr);
        rankp[static_cast<std::size_t>(rr)] =
            core::internal::pchisq_upper(rank_chisq[rr],
                                         res.rank_dof[static_cast<std::size_t>(rr)]);
    }
    res.f4rank = rmax;
    for (int rr = 0; rr <= rmax; ++rr)
        if (rankp[static_cast<std::size_t>(rr)] > rank_alpha) { res.f4rank = rr; break; }

    fill_rankdrop(rmax, res.rank_dof, res.rank_chisq, rankp,
                  res.rankdrop_f4rank, res.rankdrop_dof, res.rankdrop_dofdiff,
                  res.rankdrop_chisq, res.rankdrop_p, res.rankdrop_chisqdiff,
                  res.rankdrop_p_nested);

    auto push_pop = [&](const std::string& pat, int wt, int nl_red, double cq,
                        const double* w_for_feas) {
        const int rr = nl_red - 1;
        const int dof = core::qpadm::qpadm_dof(nl_red, nr, rr);
        res.popdrop_pat.push_back(pat);
        res.popdrop_wt.push_back(wt);
        res.popdrop_dof.push_back(dof);
        res.popdrop_f4rank.push_back(rr);
        res.popdrop_chisq.push_back(cq);
        res.popdrop_p.push_back(core::internal::pchisq_upper(cq, dof));
        bool any = false, feas = true;
        if (w_for_feas) {
            for (int i = 0; i < nl; ++i) {
                const double w = w_for_feas[i];
                if (std::isnan(w)) continue;
                any = true;
                if (w < 0.0 || w > 1.0) { feas = false; break; }
            }
        }
        res.popdrop_feasible.push_back((any && feas) ? char{1} : char{0});
    };
    if (nl >= 1) {
        std::string pat_full(static_cast<std::size_t>(nl), '0');
        push_pop(pat_full, 0, nl, pop_chisq[0], pop_wfull);
        if (nl >= 2) {
            for (int drop = nl - 1; drop >= 0; --drop) {
                std::string pat(static_cast<std::size_t>(nl), '0');
                pat[static_cast<std::size_t>(drop)] = '1';
                const int row = 1 + (nl - 1 - drop);
                push_pop(pat, 1, nl - 1, pop_chisq[row], nullptr);
            }
        }
    }
    res.status = (res.dof <= 0) ? Status::ChisqUndefined : Status::Ok;
}

}  // namespace steppe::device
