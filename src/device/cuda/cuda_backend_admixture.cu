// src/device/cuda/cuda_backend_admixture.cu
//
// CudaBackend override for the ADMIXTURE Q/F ML fit (`steppe admixture`). It uploads the
// packed individual-major genotype tile, decodes the per-individual dosage matrix G [N x M]
// + validity mask V device-resident, seeds Q/F deterministically (the shared admix_init),
// and runs the frappe/ADMIXTURE block-EM SNP-tiled and device-resident:
//
//   F-update pass (given Q):  A = Q F^T (GEMM)  ->  R2=g/A, R1=(2-g)/(1-A) (native FP64)
//                             ->  S2 = Q^T R2, S1 = Q^T R1 (GEMMs)  ->  multiplicative F-update
//   Q-update pass (given F):  A = Q F^T  ->  R2/R1  ->  T2 = R2 F, T1 = R1 (1-F) (GEMMs)
//                             ->  multiplicative Q-update + row-simplex renormalize
//   loglik pass:              A = Q F^T  ->  native-FP64 binomial log-lik reduction
//
// The GEMMs run emulated-FP64 (matmul-heavy default, engage_f2_precision + PEDANTIC math);
// the responsibility/log-lik elementwise runs native FP64 (the near-0/1 cancellation
// carve-out). Never materializes the full N x M reconstruction — A/R1/R2 are one SNP tile at
// a time (the pca_covariance_eig SNP-tiling idiom). When fixed_F is supplied the F-update is
// skipped (supervised/projection: solve only Q). Multi-seed restarts keep the best log-lik.
// A CUDA TU private to steppe_device, mirroring cuda_backend_pcangsd.cu.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <cublas_v2.h>

#include "core/internal/admix_init.hpp"
#include "device/cuda/admixture_kernels.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/f2_block_kernel.cuh"  // engage_f2_precision

namespace steppe::device {

namespace {
constexpr double kAdmixEps = 1e-5;
constexpr std::size_t kAdmixTileDivisor = 6;  // bound A/R1/R2 tiles to ~free/6
constexpr std::size_t kAdmixFreeFallback = std::size_t{1} << 30;
}  // namespace

AdmixtureFit CudaBackend::admixture_fit(const DecodeTileView& tile, int K, const double* fixed_F,
                                        long fixed_F_M, unsigned long long seed, int seeds,
                                        int max_iter, double tol, int init_mode,
                                        const Precision& precision) {
    guard_device();
    (void)init_mode;  // v1: random init only (svd warm-start is Phase 2)
    AdmixtureFit out;
    out.precision_tag = (precision.kind == Precision::Kind::EmulatedFp64 &&
                         capabilities().emulated_fp64_honorable)
                            ? Precision::Kind::EmulatedFp64
                            : Precision::Kind::Fp64;

    const long N = static_cast<long>(tile.n_individuals);
    const long M = static_cast<long>(tile.n_snp);
    const bool supervised = (fixed_F != nullptr);
    if (N <= 0 || M <= 0 || K <= 0 || K > N) {
        out.status = Status::InvalidConfig;
        return out;
    }
    if (supervised && fixed_F_M != M) {
        out.status = Status::InvalidConfig;
        return out;
    }
    out.N = static_cast<int>(N);
    out.M = static_cast<int>(M);
    out.K = K;
    const int nseed = supervised ? 1 : std::max(1, seeds);
    if (max_iter < 1) max_iter = 1;

    // --- upload packed tile + decode G/V device-resident -------------------------------
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
    if (packed_bytes > 0) h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());

    const std::size_t NM = static_cast<std::size_t>(N) * static_cast<std::size_t>(M);
    DeviceBuffer<double> dG(NM), dV(NM);
    launch_admix_decode(dPacked.data(), tile.bytes_per_record, N, M, dG.data(), dV.data(),
                        stream_.get());

    // Per-SNP mean allele-2 frequency (for the random F init).
    DeviceBuffer<double> dPhat(static_cast<std::size_t>(M));
    launch_admix_snp_mean(dG.data(), dV.data(), N, M, dPhat.data(), stream_.get());
    std::vector<double> phat(static_cast<std::size_t>(M), 0.5);
    d2h_async(phat.data(), dPhat, static_cast<std::size_t>(M), stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // --- SNP tile size bounding the A/R1/R2 operands to ~free/kAdmixTileDivisor ---------
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kAdmixFreeFallback;
    std::size_t budget = free_b / kAdmixTileDivisor;
    const std::size_t col_bytes = 3u * static_cast<std::size_t>(N) * sizeof(double);  // A+R2+R1
    if (budget < col_bytes) budget = col_bytes;
    long tileM = static_cast<long>(budget / col_bytes);
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;

    // --- resident EM buffers ------------------------------------------------------------
    const std::size_t NK = static_cast<std::size_t>(N) * static_cast<std::size_t>(K);
    const std::size_t MK = static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t KM = static_cast<std::size_t>(K) * static_cast<std::size_t>(M);
    DeviceBuffer<double> dQ(NK), dF(MK), dFc(MK);
    DeviceBuffer<double> dA(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
    DeviceBuffer<double> dR2(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
    DeviceBuffer<double> dR1(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
    DeviceBuffer<double> dS2(KM), dS1(KM);
    DeviceBuffer<double> dT2(NK), dT1(NK);
    DeviceBuffer<double> dLL(1);

    // Fixed F (supervised/projection): host row-major [M x K] -> col-major dF, once.
    std::vector<double> fixedF_cm;
    if (supervised) {
        fixedF_cm.assign(MK, 0.0);
        for (long s = 0; s < M; ++s)
            for (int k = 0; k < K; ++k) {
                double f = fixed_F[static_cast<std::size_t>(s) * static_cast<std::size_t>(K) +
                                   static_cast<std::size_t>(k)];
                if (f < kAdmixEps) f = kAdmixEps;
                if (f > 1.0 - kAdmixEps) f = 1.0 - kAdmixEps;
                fixedF_cm[static_cast<std::size_t>(s) + static_cast<std::size_t>(M) *
                                                            static_cast<std::size_t>(k)] = f;
            }
    }

    std::vector<double> hQ(NK, 0.0), hF(MK, 0.0);  // host col-major init scratch
    std::vector<double> bestQ, bestF;
    double best_ll = -std::numeric_limits<double>::infinity();
    out.seed_loglik.assign(static_cast<std::size_t>(nseed), 0.0);
    out.seed_iters.assign(static_cast<std::size_t>(nseed), 0);
    out.seed_converged.assign(static_cast<std::size_t>(nseed), 0);

    const MathModeScope gemm_mode(blas_.get(), CUBLAS_PEDANTIC_MATH);
    const double one = 1.0, zero = 0.0;
    const int Ni = static_cast<int>(N);

    for (int si = 0; si < nseed; ++si) {
        // ---- init Q (and F unless fixed) ----
        const std::uint64_t sd = seed + static_cast<std::uint64_t>(si) * 0x9E3779B1ull;
        core::admix_init_q(sd, N, K, hQ.data());
        h2d_async(dQ, hQ.data(), NK, stream_.get());
        if (supervised) {
            h2d_async(dF, fixedF_cm.data(), MK, stream_.get());
        } else {
            core::admix_init_f(sd, M, K, phat.data(), kAdmixEps, hF.data());
            h2d_async(dF, hF.data(), MK, stream_.get());
        }
        launch_admix_complement(dF.data(), dFc.data(), M, K, stream_.get());

        double L_prev = -std::numeric_limits<double>::infinity();
        int iters = 0;
        bool converged = false;
        for (int it = 0; it < max_iter; ++it) {
            engage_f2_precision(blas_.get(), precision);

            // ---- F-update pass (skip when F fixed) ----
            if (!supervised) {
                for (long s0 = 0; s0 < M; s0 += tileM) {
                    const long t = std::min<long>(tileM, M - s0);
                    const int tt = static_cast<int>(t);
                    // A = Q F_tile^T  (N x t)
                    CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, &one,
                                             dQ.data(), Ni, dF.data() + s0, static_cast<int>(M),
                                             &zero, dA.data(), Ni));
                    launch_admix_responsibility(dG.data() + static_cast<std::size_t>(N) * s0,
                                                dV.data() + static_cast<std::size_t>(N) * s0,
                                                dA.data(), N, t, kAdmixEps, dR2.data(), dR1.data(),
                                                stream_.get());
                    // S2_col = Q^T R2  (K x t) into dS2 at column s0 (ld K)
                    CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, K, tt, Ni, &one,
                                             dQ.data(), Ni, dR2.data(), Ni, &zero,
                                             dS2.data() + static_cast<std::size_t>(K) * s0, K));
                    CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, K, tt, Ni, &one,
                                             dQ.data(), Ni, dR1.data(), Ni, &zero,
                                             dS1.data() + static_cast<std::size_t>(K) * s0, K));
                }
                launch_admix_update_f(dF.data(), dS2.data(), dS1.data(), M, K, kAdmixEps,
                                      stream_.get());
                launch_admix_complement(dF.data(), dFc.data(), M, K, stream_.get());
            }

            // ---- Q-update pass ----
            bool first = true;
            for (long s0 = 0; s0 < M; s0 += tileM) {
                const long t = std::min<long>(tileM, M - s0);
                const int tt = static_cast<int>(t);
                CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, &one,
                                         dQ.data(), Ni, dF.data() + s0, static_cast<int>(M), &zero,
                                         dA.data(), Ni));
                launch_admix_responsibility(dG.data() + static_cast<std::size_t>(N) * s0,
                                            dV.data() + static_cast<std::size_t>(N) * s0, dA.data(),
                                            N, t, kAdmixEps, dR2.data(), dR1.data(), stream_.get());
                const double beta = first ? 0.0 : 1.0;
                // T2 += R2 F_tile   (N x K)
                CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, Ni, K, tt, &one,
                                         dR2.data(), Ni, dF.data() + s0, static_cast<int>(M), &beta,
                                         dT2.data(), Ni));
                // T1 += R1 (1-F_tile)
                CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, Ni, K, tt, &one,
                                         dR1.data(), Ni, dFc.data() + s0, static_cast<int>(M), &beta,
                                         dT1.data(), Ni));
                first = false;
            }
            launch_admix_update_q(dQ.data(), dT2.data(), dT1.data(), N, K, stream_.get());

            // ---- log-likelihood pass (native FP64) ----
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dLL.data(), 0, sizeof(double), stream_.get()));
            for (long s0 = 0; s0 < M; s0 += tileM) {
                const long t = std::min<long>(tileM, M - s0);
                const int tt = static_cast<int>(t);
                CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, &one,
                                         dQ.data(), Ni, dF.data() + s0, static_cast<int>(M), &zero,
                                         dA.data(), Ni));
                launch_admix_loglik(dG.data() + static_cast<std::size_t>(N) * s0,
                                    dV.data() + static_cast<std::size_t>(N) * s0, dA.data(), N, t,
                                    kAdmixEps, dLL.data(), stream_.get());
            }
            double L = 0.0;
            d2h_async(&L, dLL, 1, stream_.get());
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            iters = it + 1;
            if (std::isfinite(L) && std::isfinite(L_prev) &&
                std::fabs(L - L_prev) < tol * std::max(1.0, std::fabs(L))) {
                converged = true;
                L_prev = L;
                break;
            }
            L_prev = L;
        }

        out.seed_loglik[static_cast<std::size_t>(si)] = L_prev;
        out.seed_iters[static_cast<std::size_t>(si)] = iters;
        out.seed_converged[static_cast<std::size_t>(si)] = converged ? 1 : 0;

        if (si == 0 || L_prev > best_ll) {
            best_ll = L_prev;
            out.best_seed = si;
            out.iters_run = iters;
            out.converged = converged;
            bestQ.assign(NK, 0.0);
            bestF.assign(MK, 0.0);
            d2h_async(bestQ.data(), dQ, NK, stream_.get());
            d2h_async(bestF.data(), dF, MK, stream_.get());
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        }
    }

    // --- col-major Q/F -> row-major result ----------------------------------------------
    out.best_loglik = best_ll;
    out.Q.assign(NK, 0.0);
    out.F.assign(MK, 0.0);
    for (long i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k)
            out.Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(K) +
                  static_cast<std::size_t>(k)] =
                bestQ[static_cast<std::size_t>(i) +
                      static_cast<std::size_t>(N) * static_cast<std::size_t>(k)];
    for (long s = 0; s < M; ++s)
        for (int k = 0; k < K; ++k)
            out.F[static_cast<std::size_t>(s) * static_cast<std::size_t>(K) +
                  static_cast<std::size_t>(k)] =
                bestF[static_cast<std::size_t>(s) +
                      static_cast<std::size_t>(M) * static_cast<std::size_t>(k)];

    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
