// src/device/cuda/pca_truncated_eig.cu
//
// Randomized subspace-iteration top-K eigensolver on a device-resident, column-major N x N
// SPD covariance (the `steppe pca` GRM). See pca_truncated_eig.cuh for the why (it replaces
// the full-spectrum cusolverDnDsyevd whose ~8.5 GB workspace OOMs the 23k AADR cohort).
//
// Pipeline (Halko 2011, Alg. 4.4 + 5.3), all device-resident:
//   Omega  (N x L) deterministic Gaussian sketch, L = min(K + oversample, N)
//   Q      = orth(C * Omega)                         (Dgemm emulated + Householder QR native)
//   repeat subspace_iters:  Q = orth(C * Q)          (sharpens the near-degenerate tail)
//   B      = Q^T (C Q)      (L x L)                   (C*Q emulated; Q^T(CQ) NATIVE — carve-out)
//   (Wb, Yb) = eigh(B)      (L x L, ascending)        (Dsyevd native — tiny, ~1 MB workspace)
//   V      = Q * Yb         (N x L lifted Ritz vecs)  (Dgemm emulated)
// The L ascending Ritz values Wb approximate C's top-L eigenvalues; the top K (well-separated)
// are the PCs. Extra workspace is O(N*L) (~190 MB at N=23k, L=256), not O(N^2).
#include "device/cuda/pca_truncated_eig.cuh"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "device/cuda/check.cuh"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/f2_block_kernel.cuh"       // engage_f2_precision
#include "device/cuda/handles.hpp"               // MathModeScope, CusolverMathModeScope
#include "device/cuda/pca_standardize_kernel.cuh"  // launch_pca_fill_omega

namespace steppe::device {

namespace {

// Fixed seed for the Gaussian sketch — reproducible across runs (golden-parity friendly).
constexpr unsigned long long kPcaOmegaSeed = 0x9E3779B97F4A7C15ULL;

// C_out (N x L) = dC (N x N) * Q_in (N x L), emulated-FP64 (the matmul-heavy sketch path).
void sketch_CQ(cublasHandle_t blas, const Precision& precision, const double* dC,
               const double* Q_in, double* C_out, int N, int L) {
    const MathModeScope mode(blas, CUBLAS_PEDANTIC_MATH);
    engage_f2_precision(blas, precision);
    const double one = 1.0;
    const double zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, L, N, &one, dC, N, Q_in, N,
                             &zero, C_out, N));
}

// Thin Householder QR that ORTHONORMALIZES A (N x L, column-major, ld N) in place -> Q.
// Native-FP64 (the eigen/factorization carve-out). Returns false on a nonzero cuSOLVER info.
bool orthonormalize(cusolverDnHandle_t solver, double* A, int N, int L,
                    DeviceBuffer<double>& dTau, DeviceBuffer<int>& dInfo, cudaStream_t stream) {
    const CusolverMathModeScope qr_scope(solver, /*honorable=*/false);
    int lw_geqrf = 0;
    int lw_orgqr = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver, N, L, A, N, &lw_geqrf));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(solver, N, L, L, A, N, dTau.data(), &lw_orgqr));
    const int lwork = std::max(std::max(lw_geqrf, lw_orgqr), 1);
    DeviceBuffer<double> work(static_cast<std::size_t>(lwork));
    CUSOLVER_CHECK(cusolverDnDgeqrf(solver, N, L, A, N, dTau.data(), work.data(), lwork,
                                    dInfo.data()));
    CUSOLVER_CHECK(cusolverDnDorgqr(solver, N, L, L, A, N, dTau.data(), work.data(), lwork,
                                    dInfo.data()));
    int info = 0;
    d2h_async(&info, dInfo, 1, stream);
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
    return info == 0;
}

}  // namespace

PcaTruncatedEig pca_truncated_topk(cublasHandle_t blas, cusolverDnHandle_t solver,
                                   cudaStream_t stream, const double* dC, int N, int K,
                                   int oversample, int subspace_iters,
                                   const Precision& precision, double* d_evecL,
                                   double* d_evalL, int* out_L) {
    PcaTruncatedEig out;
    if (N <= 0 || K <= 0 || K > N) {
        out.status = Status::InvalidConfig;
        return out;
    }
    if (oversample < 0) oversample = 0;
    if (subspace_iters < 0) subspace_iters = 0;
    const long Lw = std::min<long>(static_cast<long>(K) + oversample, N);
    const int L = static_cast<int>(Lw);
    out.L = L;
    if (out_L) *out_L = L;

    const std::size_t NL = static_cast<std::size_t>(N) * static_cast<std::size_t>(L);
    DeviceBuffer<double> dOmega(NL);
    DeviceBuffer<double> dQ(NL);
    DeviceBuffer<double> dScratch(NL);
    DeviceBuffer<double> dB(static_cast<std::size_t>(L) * static_cast<std::size_t>(L));
    DeviceBuffer<double> dTau(static_cast<std::size_t>(L));
    DeviceBuffer<int> dInfo(1);

    // 1) Deterministic Gaussian sketch Omega (N x L).
    launch_pca_fill_omega(dOmega.data(), NL, kPcaOmegaSeed, stream);

    // 2) Y = C * Omega  ->  3) Q = orth(Y).
    sketch_CQ(blas, precision, dC, dOmega.data(), dQ.data(), N, L);
    if (!orthonormalize(solver, dQ.data(), N, L, dTau, dInfo, stream)) {
        out.status = Status::NonSpdCovariance;
        return out;
    }

    // 4) Subspace iteration: Q = orth(C * Q), re-orthonormalizing each pass (stable form —
    //    never forms C*C*Omega raw, which loses the trailing PCs to rounding).
    for (int it = 0; it < subspace_iters; ++it) {
        sketch_CQ(blas, precision, dC, dQ.data(), dScratch.data(), N, L);
        if (!orthonormalize(solver, dScratch.data(), N, L, dTau, dInfo, stream)) {
            out.status = Status::NonSpdCovariance;
            return out;
        }
        std::swap(dQ, dScratch);  // dQ now holds the freshly orthonormalized basis
    }

    // 5) Rayleigh-Ritz projection. dScratch = C*Q emulated; B = Q^T*(C*Q) NATIVE (the small
    //    O(N*L^2) inner-product reduction whose eigenvalues ARE the Ritz values — carve-out).
    sketch_CQ(blas, precision, dC, dQ.data(), dScratch.data(), N, L);
    {
        const MathModeScope mode(blas, CUBLAS_PEDANTIC_MATH);  // native FP64 (no emulation)
        const double one = 1.0;
        const double zero = 0.0;
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, N, &one, dQ.data(), N,
                                 dScratch.data(), N, &zero, dB.data(), L));
    }

    // 6) Small-B eigen (native, carve-out on L x L, NOT the N x N GRM). B_evecs overwrite dB,
    //    ascending eigenvalues into d_evalL. Dsyevd reads only the LOWER triangle, so any
    //    emulated-rounding asymmetry in B's upper triangle is harmless (no symmetrize needed).
    {
        const CusolverMathModeScope eig_scope(solver, /*honorable=*/false);
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR,
                                                   CUBLAS_FILL_MODE_LOWER, L, dB.data(), L,
                                                   d_evalL, &lwork));
        DeviceBuffer<double> work(static_cast<std::size_t>(std::max(lwork, 1)));
        CUSOLVER_CHECK(cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR,
                                        CUBLAS_FILL_MODE_LOWER, L, dB.data(), L, d_evalL,
                                        work.data(), lwork, dInfo.data()));
        int info = 0;
        d2h_async(&info, dInfo, 1, stream);
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (info != 0) {
            out.status = Status::NonSpdCovariance;
            return out;
        }
    }

    // 7) Lift the Ritz vectors: V (N x L) = Q (N x L) * B_evecs (L x L). Emulated (matmul).
    {
        const MathModeScope mode(blas, CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas, precision);
        const double one = 1.0;
        const double zero = 0.0;
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, L, L, &one, dQ.data(), N,
                                 dB.data(), L, &zero, d_evecL, N));
    }

    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
