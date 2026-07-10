// src/device/cuda/cuda_backend_admixture.cu
//
// CudaBackend override for the ADMIXTURE Q/F ML fit (`steppe admixture`). It uploads the
// packed individual-major genotype tile and keeps ONLY the 2-bit packed genotypes resident
// (Tier-1): the per-individual dosage G [N x M] + validity mask V are NOT materialized
// (that was a 32 GB / 16-B-per-genotype wall at ~8.5k samples). Instead each consumer decodes
// the [N x tileM] slice it needs per SNP-tile straight from dPacked (launch_admix_decode_tile),
// exactly as PCA re-standardizes from d_packed per SNP-block. It seeds Q/F deterministically
// (the shared admix_init) and runs the frappe/ADMIXTURE block-EM SNP-tiled and device-resident:
//
//   F-update pass (given Q):  A = Q F^T (GEMM)  ->  R2=g/A, R1=(2-g)/(1-A) (native FP64)
//                             ->  S2 = Q^T R2, S1 = Q^T R1 (GEMMs)  ->  multiplicative F-update
//   Q-update pass (given F):  A = Q F^T  ->  R2/R1  ->  T2 = R2 F, T1 = R1 (1-F) (GEMMs)
//                             ->  multiplicative Q-update + row-simplex renormalize
//   loglik pass:              A = Q F^T  ->  native-FP64 binomial log-lik reduction
//
// That one full F-then-Q iteration is the base map M. Two ACCELERATION layers sit over it,
// selected by `accel_mode`:
//   - accel_mode==1 (em):      plain fixed-point iteration of M, one seed at a time (the
//                              original shipped loop, kept bit-identical below).
//   - accel_mode==0 (squarem): the SqS3 quasi-Newton control layer wrapping the SAME map M
//                              (~8x fewer OUTER steps), with all restarts run CONCURRENTLY as
//                              one strided-batched-GEMM stack (robustness ~free). The base map
//                              math is untouched — the accelerator is a swappable layer, never
//                              baked into M. See admixture_kernels.{cu,cuh} for the batched
//                              primitives + the SQUAREM norms/combine/project kernels.
//
// The GEMMs run emulated-FP64 (matmul-heavy default, engage_f2_precision + PEDANTIC math);
// the responsibility/log-lik elementwise + the SQUAREM norms/extrapolation run native FP64.
// Never materializes the full N x M reconstruction — A/R1/R2 are one SNP tile at a time.
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
constexpr std::size_t kAdmixTileDivisor = 6;  // bound A/R1/R2 tiles to ~free/6 (per seed / S)
constexpr std::size_t kAdmixFreeFallback = std::size_t{1} << 30;
// SQUAREM backtracking: the accept slack (MUST be > 0 so emulated-GEMM / boundary-clamp noise
// in M cannot make the terminal test fail forever) and the terminal cap + near-alpha=-1 band
// that force an UNCONDITIONAL plain-EM step (M(theta2)), guaranteeing termination + monotone.
constexpr double kSqDelta = 1e-9;
constexpr double kSqVvTiny = 1e-30;
constexpr double kSqTermBand = 1e-4;  // |alpha+1| < band -> snap to the plain-EM floor
constexpr int kSqMaxBacktrack = 30;

// cudaMalloc reserves device memory in fixed-size pages, so a request of B bytes reduces
// cudaMemGetInfo's reported free by round_up(B, page). On the CUDA 13 / Blackwell (sm_120)
// target this page is 2 MiB (empirically verified on the gate box: an 836,203,888 B request
// dropped free by 836,763,648 B = 399 x 2 MiB). Used ONLY to reproduce the pre-fix free-VRAM
// reading for tileM parity (see reproduced_free_b) — it is not a correctness-critical value.
constexpr std::size_t kCudaMallocPageBytes = std::size_t{2} << 20;  // 2 MiB
[[nodiscard]] inline std::size_t admix_reserved_bytes(std::size_t logical) noexcept {
    return ((logical + kCudaMallocPageBytes - 1) / kCudaMallocPageBytes) * kCudaMallocPageBytes;
}
}  // namespace

AdmixtureFit CudaBackend::admixture_fit(const DecodeTileView& tile, int K, const double* fixed_F,
                                        long fixed_F_M, unsigned long long seed, int seeds,
                                        int max_iter, double tol, int init_mode, int accel_mode,
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

    // --- upload packed tile; keep ONLY the 2-bit packed genotypes resident (Tier-1) ---------
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
    if (packed_bytes > 0) h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());

    const std::size_t NM = static_cast<std::size_t>(N) * static_cast<std::size_t>(M);

    // Per-SNP mean allele-2 frequency (for the random F init), computed BY SNP-TILE from
    // dPacked so no full N x M buffer is held. phat[s] is a per-SNP reduction over all N and is
    // independent of the tile boundary, so this stays byte-identical to the old full decode.
    DeviceBuffer<double> dPhat(static_cast<std::size_t>(M));
    {
        // Modest, parity-irrelevant width (phat is per-SNP): bound the transient G/V decode
        // scratch to ~256 MiB. Freed at block exit BEFORE the tileM budget query below, so the
        // free-VRAM reading there differs from the pre-fix reading by exactly dG+dV (which the
        // pre-fix path held live at that point) — see reproduced_free_b.
        constexpr std::size_t kPhatScratchCap = std::size_t{256} << 20;
        long tileP = static_cast<long>(kPhatScratchCap /
                                       (2u * static_cast<std::size_t>(N) * sizeof(double)));
        if (tileP < 1) tileP = 1;
        if (tileP > M) tileP = M;
        DeviceBuffer<double> dGp(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileP));
        DeviceBuffer<double> dVp(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileP));
        for (long s0 = 0; s0 < M; s0 += tileP) {
            const long t = std::min<long>(tileP, M - s0);
            launch_admix_decode_tile(dPacked.data(), tile.bytes_per_record, N, s0, t, dGp.data(),
                                     dVp.data(), stream_.get());
            launch_admix_snp_mean(dGp.data(), dVp.data(), N, t, dPhat.data() + s0, stream_.get());
        }
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));  // finish before dGp/dVp free
    }
    std::vector<double> phat(static_cast<std::size_t>(M), 0.5);
    d2h_async(phat.data(), dPhat, static_cast<std::size_t>(M), stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    const std::size_t NK = static_cast<std::size_t>(N) * static_cast<std::size_t>(K);
    const std::size_t MK = static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t KM = static_cast<std::size_t>(K) * static_cast<std::size_t>(M);
    const int Ni = static_cast<int>(N);

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

    std::vector<double> bestQ, bestF;
    double best_ll = -std::numeric_limits<double>::infinity();
    out.seed_loglik.assign(static_cast<std::size_t>(nseed), 0.0);
    out.seed_iters.assign(static_cast<std::size_t>(nseed), 0);
    out.seed_converged.assign(static_cast<std::size_t>(nseed), 0);

    // Reproduce the pre-fix free-VRAM reading so the SNP-tile width tileM — hence the cross-tile
    // T2/T1 Q-update FP summation order — stays byte-identical to the pre-fix binary. The pre-fix
    // path allocated dG+dV (two full N x M FP64 buffers) BEFORE the budget query; this fix does
    // not, so their footprint no longer shrinks the measured free VRAM. Whenever the pre-fix path
    // would itself have fit (2*reserved(N*M*8) <= free) we subtract that footprint to reproduce
    // its reading exactly (a valid byte-exact baseline exists there); once it would have OOM'd —
    // the wall this fix removes — we keep the true, larger free VRAM (no baseline to match, and
    // it is what lets the run proceed with sensible tiles). dPacked/dPhat are resident in BOTH
    // paths at the query point, so they cancel and the reproduction is exact on the gate box.
    const std::size_t admix_phantom_free = 2u * admix_reserved_bytes(NM * sizeof(double));
    auto reproduced_free_b = [&]() -> std::size_t {
        std::size_t free_b = capabilities().free_vram_bytes;
        if (free_b == 0) free_b = kAdmixFreeFallback;
        if (admix_phantom_free < free_b) free_b -= admix_phantom_free;
        return free_b;
    };

    // ==========================================================================================
    // accel_mode == 1 : PLAIN EM (bit-identical to the shipped loop; base map untouched).
    // ==========================================================================================
    if (accel_mode == 1) {
        std::size_t free_b = reproduced_free_b();
        std::size_t budget = free_b / kAdmixTileDivisor;
        // col_bytes counts ONLY dA/dR2/dR1 (3 cols) — NOT the added dGt/dVt decode scratch — so
        // tileM is byte-identical to the pre-fix value (parity); the 2 extra tile columns are
        // absorbed by the divisor headroom (and the freed dG+dV dwarfs them).
        const std::size_t col_bytes = 3u * static_cast<std::size_t>(N) * sizeof(double);
        if (budget < col_bytes) budget = col_bytes;
        long tileM = static_cast<long>(budget / col_bytes);
        if (tileM < 1) tileM = 1;
        if (tileM > M) tileM = M;

        DeviceBuffer<double> dQ(NK), dF(MK), dFc(MK);
        DeviceBuffer<double> dA(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dR2(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dR1(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
        // Per-SNP-tile decode scratch (dosage + validity), decoded fresh from dPacked each pass.
        DeviceBuffer<double> dGt(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dVt(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dS2(KM), dS1(KM);
        DeviceBuffer<double> dT2(NK), dT1(NK);
        DeviceBuffer<double> dLL(1);

        std::vector<double> hQ(NK, 0.0), hF(MK, 0.0);
        const MathModeScope gemm_mode(blas_.get(), CUBLAS_PEDANTIC_MATH);
        const double one = 1.0, zero = 0.0;

        for (int si = 0; si < nseed; ++si) {
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
                if (!supervised) {
                    for (long s0 = 0; s0 < M; s0 += tileM) {
                        const long t = std::min<long>(tileM, M - s0);
                        const int tt = static_cast<int>(t);
                        CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K,
                                                 &one, dQ.data(), Ni, dF.data() + s0,
                                                 static_cast<int>(M), &zero, dA.data(), Ni));
                        launch_admix_decode_tile(dPacked.data(), tile.bytes_per_record, N, s0, t,
                                                 dGt.data(), dVt.data(), stream_.get());
                        launch_admix_responsibility(dGt.data(), dVt.data(), dA.data(), N, t,
                                                    kAdmixEps, dR2.data(), dR1.data(),
                                                    stream_.get());
                        CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, K, tt, Ni,
                                                 &one, dQ.data(), Ni, dR2.data(), Ni, &zero,
                                                 dS2.data() + static_cast<std::size_t>(K) * s0, K));
                        CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, K, tt, Ni,
                                                 &one, dQ.data(), Ni, dR1.data(), Ni, &zero,
                                                 dS1.data() + static_cast<std::size_t>(K) * s0, K));
                    }
                    launch_admix_update_f(dF.data(), dS2.data(), dS1.data(), M, K, kAdmixEps,
                                          stream_.get());
                    launch_admix_complement(dF.data(), dFc.data(), M, K, stream_.get());
                }

                bool first = true;
                for (long s0 = 0; s0 < M; s0 += tileM) {
                    const long t = std::min<long>(tileM, M - s0);
                    const int tt = static_cast<int>(t);
                    CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, &one,
                                             dQ.data(), Ni, dF.data() + s0, static_cast<int>(M),
                                             &zero, dA.data(), Ni));
                    launch_admix_decode_tile(dPacked.data(), tile.bytes_per_record, N, s0, t,
                                             dGt.data(), dVt.data(), stream_.get());
                    launch_admix_responsibility(dGt.data(), dVt.data(), dA.data(), N, t, kAdmixEps,
                                                dR2.data(), dR1.data(), stream_.get());
                    const double beta = first ? 0.0 : 1.0;
                    CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, Ni, K, tt, &one,
                                             dR2.data(), Ni, dF.data() + s0, static_cast<int>(M),
                                             &beta, dT2.data(), Ni));
                    CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, Ni, K, tt, &one,
                                             dR1.data(), Ni, dFc.data() + s0, static_cast<int>(M),
                                             &beta, dT1.data(), Ni));
                    first = false;
                }
                launch_admix_update_q(dQ.data(), dT2.data(), dT1.data(), N, K, stream_.get());

                STEPPE_CUDA_CHECK(cudaMemsetAsync(dLL.data(), 0, sizeof(double), stream_.get()));
                for (long s0 = 0; s0 < M; s0 += tileM) {
                    const long t = std::min<long>(tileM, M - s0);
                    const int tt = static_cast<int>(t);
                    CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, &one,
                                             dQ.data(), Ni, dF.data() + s0, static_cast<int>(M),
                                             &zero, dA.data(), Ni));
                    launch_admix_decode_tile(dPacked.data(), tile.bytes_per_record, N, s0, t,
                                             dGt.data(), dVt.data(), stream_.get());
                    launch_admix_loglik(dGt.data(), dVt.data(), dA.data(), N, t, kAdmixEps,
                                        dLL.data(), stream_.get());
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
                out.base_map_evals = iters;  // one base map per plain-EM iteration
                out.converged = converged;
                bestQ.assign(NK, 0.0);
                bestF.assign(MK, 0.0);
                d2h_async(bestQ.data(), dQ, NK, stream_.get());
                d2h_async(bestF.data(), dF, MK, stream_.get());
                STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            }
        }
    } else {
        // ======================================================================================
        // accel_mode == 0 : SQUAREM (SqS3) over the SAME base map, all restarts BATCHED (S=nseed).
        // ======================================================================================
        const int S = nseed;
        const std::size_t Sz = static_cast<std::size_t>(S);

        // MUST_FIX 1: the S x SNP-tile scratch (dA/dR2/dR1) is the dominant buffer, so fold S
        // into the tile budget divisor: budget = free / (divisor * S). Keeps the S-batched tile
        // region within ~free/divisor (was silently violated by S x with the per-seed sizing).
        std::size_t free_b = reproduced_free_b();
        std::size_t budget = free_b / (kAdmixTileDivisor * Sz);
        // col_bytes counts ONLY the S-batched dA/dR2/dR1 (3 cols) — NOT the added seed-shared
        // dGt/dVt decode scratch (2 non-batched cols, tiny vs S*3) — so tileM is byte-identical
        // to the pre-fix value (the parity-load-bearing invariant).
        const std::size_t col_bytes = 3u * static_cast<std::size_t>(N) * sizeof(double);
        if (budget < col_bytes) budget = col_bytes;
        long tileM = static_cast<long>(budget / col_bytes);
        if (tileM < 1) tileM = 1;
        if (tileM > M) tileM = M;
        const long strideTile = static_cast<long>(N) * tileM;

        // Resident batched stack. F snapshots exist only unsupervised (theta = vecQ only when F
        // fixed -> F excluded from theta/r/v/combine/projection).
        const std::size_t fcopy = supervised ? 1u : Sz * MK;
        DeviceBuffer<double> dQ(Sz * NK), dF(Sz * MK), dFc(Sz * MK);
        DeviceBuffer<double> dQ0(Sz * NK), dQ1(Sz * NK), dQ2(Sz * NK), dQp(Sz * NK);
        DeviceBuffer<double> dF0(fcopy), dF1(fcopy), dF2(fcopy), dFp(fcopy), dFcp(fcopy);
        DeviceBuffer<double> dA(Sz * static_cast<std::size_t>(strideTile));
        DeviceBuffer<double> dR2(Sz * static_cast<std::size_t>(strideTile));
        DeviceBuffer<double> dR1(Sz * static_cast<std::size_t>(strideTile));
        // Per-SNP-tile decode scratch — SEED-INDEPENDENT (G/V are shared across restarts), so a
        // single [N x tileM] copy (not S-batched), decoded fresh from dPacked in each sweep.
        DeviceBuffer<double> dGt(static_cast<std::size_t>(strideTile));
        DeviceBuffer<double> dVt(static_cast<std::size_t>(strideTile));
        DeviceBuffer<double> dS2(Sz * KM), dS1(Sz * KM);
        DeviceBuffer<double> dT2(Sz * NK), dT1(Sz * NK);
        DeviceBuffer<double> dLL(Sz), dLL2(Sz), drr(Sz), dvv(Sz), dAlpha(Sz), dActive(Sz);

        const MathModeScope gemm_mode(blas_.get(), CUBLAS_PEDANTIC_MATH);
        const double one = 1.0;
        cudaStream_t stream = stream_.get();
        cublasHandle_t blas = blas_.get();

        auto gemm_sb = [&](cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                           const double* A, int lda, long sA, const double* B, int ldb, long sB,
                           double beta, double* C, int ldc, long sC) {
            CUBLAS_CHECK(cublasDgemmStridedBatched(
                blas, ta, tb, m, n, k, &one, A, lda, static_cast<long long>(sA), B, ldb,
                static_cast<long long>(sB), &beta, C, ldc, static_cast<long long>(sC), S));
        };
        auto d2d = [&](const double* src, double* dst, std::size_t n) {
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(double),
                                              cudaMemcpyDeviceToDevice, stream));
        };
        auto d2d_seed = [&](const double* src, double* dst, int s, std::size_t per) {
            const std::size_t off = static_cast<std::size_t>(s) * per;
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst + off, src + off, per * sizeof(double),
                                              cudaMemcpyDeviceToDevice, stream));
        };

        // One base-map iteration M in place on (Q,F,Fc). `active` (nullptr=all) freezes seeds.
        auto em_map = [&](double* Q, double* F, double* Fc, const double* active) {
            engage_f2_precision(blas, precision);
            if (!supervised) {
                for (long s0 = 0; s0 < M; s0 += tileM) {
                    const long t = std::min<long>(tileM, M - s0);
                    const int tt = static_cast<int>(t);
                    gemm_sb(CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, Q, Ni, static_cast<long>(NK),
                            F + s0, static_cast<int>(M), static_cast<long>(MK), 0.0, dA.data(), Ni,
                            strideTile);
                    launch_admix_decode_tile(dPacked.data(), tile.bytes_per_record, N, s0, t,
                                             dGt.data(), dVt.data(), stream);
                    launch_admix_responsibility_b(dGt.data(), dVt.data(), dA.data(), N, t, kAdmixEps,
                                                  dR2.data(), dR1.data(), S, strideTile, stream);
                    gemm_sb(CUBLAS_OP_T, CUBLAS_OP_N, K, tt, Ni, Q, Ni, static_cast<long>(NK),
                            dR2.data(), Ni, strideTile, 0.0,
                            dS2.data() + static_cast<std::size_t>(K) * s0, K, static_cast<long>(KM));
                    gemm_sb(CUBLAS_OP_T, CUBLAS_OP_N, K, tt, Ni, Q, Ni, static_cast<long>(NK),
                            dR1.data(), Ni, strideTile, 0.0,
                            dS1.data() + static_cast<std::size_t>(K) * s0, K, static_cast<long>(KM));
                }
                launch_admix_update_f_b(F, dS2.data(), dS1.data(), M, K, kAdmixEps, S,
                                        static_cast<long>(MK), static_cast<long>(KM), active,
                                        stream);
                launch_admix_complement_b(F, Fc, static_cast<long>(MK), S, static_cast<long>(MK),
                                          active, stream);
            }
            bool first = true;
            for (long s0 = 0; s0 < M; s0 += tileM) {
                const long t = std::min<long>(tileM, M - s0);
                const int tt = static_cast<int>(t);
                gemm_sb(CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, Q, Ni, static_cast<long>(NK), F + s0,
                        static_cast<int>(M), static_cast<long>(MK), 0.0, dA.data(), Ni, strideTile);
                launch_admix_decode_tile(dPacked.data(), tile.bytes_per_record, N, s0, t,
                                         dGt.data(), dVt.data(), stream);
                launch_admix_responsibility_b(dGt.data(), dVt.data(), dA.data(), N, t, kAdmixEps,
                                              dR2.data(), dR1.data(), S, strideTile, stream);
                const double beta = first ? 0.0 : 1.0;
                gemm_sb(CUBLAS_OP_N, CUBLAS_OP_N, Ni, K, tt, dR2.data(), Ni, strideTile, F + s0,
                        static_cast<int>(M), static_cast<long>(MK), beta, dT2.data(), Ni,
                        static_cast<long>(NK));
                gemm_sb(CUBLAS_OP_N, CUBLAS_OP_N, Ni, K, tt, dR1.data(), Ni, strideTile, Fc + s0,
                        static_cast<int>(M), static_cast<long>(MK), beta, dT1.data(), Ni,
                        static_cast<long>(NK));
                first = false;
            }
            launch_admix_update_q_b(Q, dT2.data(), dT1.data(), N, K, S, static_cast<long>(NK),
                                    active, stream);
        };

        // Per-seed loglik(Q,F) -> dll[S] (native FP64).
        auto loglik = [&](const double* Q, const double* F, double* dll) {
            engage_f2_precision(blas, precision);
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dll, 0, Sz * sizeof(double), stream));
            for (long s0 = 0; s0 < M; s0 += tileM) {
                const long t = std::min<long>(tileM, M - s0);
                const int tt = static_cast<int>(t);
                gemm_sb(CUBLAS_OP_N, CUBLAS_OP_T, Ni, tt, K, Q, Ni, static_cast<long>(NK), F + s0,
                        static_cast<int>(M), static_cast<long>(MK), 0.0, dA.data(), Ni, strideTile);
                launch_admix_decode_tile(dPacked.data(), tile.bytes_per_record, N, s0, t, dGt.data(),
                                         dVt.data(), stream);
                launch_admix_loglik_b(dGt.data(), dVt.data(), dA.data(), N, t, kAdmixEps, dll, S,
                                      strideTile, stream);
            }
        };

        // --- seed the batched stack ---------------------------------------------------------
        std::vector<double> hQall(Sz * NK, 0.0);
        std::vector<double> hFall(supervised ? std::size_t{0} : Sz * MK, 0.0);
        for (int si = 0; si < S; ++si) {
            const std::uint64_t sd = seed + static_cast<std::uint64_t>(si) * 0x9E3779B1ull;
            core::admix_init_q(sd, N, K, hQall.data() + static_cast<std::size_t>(si) * NK);
            if (!supervised)
                core::admix_init_f(sd, M, K, phat.data(), kAdmixEps,
                                   hFall.data() + static_cast<std::size_t>(si) * MK);
        }
        h2d_async(dQ, hQall.data(), Sz * NK, stream);
        if (supervised) {
            h2d_async(dF, fixedF_cm.data(), MK, stream);  // S==1 when supervised
        } else {
            h2d_async(dF, hFall.data(), Sz * MK, stream);
        }
        launch_admix_complement_b(dF.data(), dFc.data(), static_cast<long>(MK), S,
                                  static_cast<long>(MK), nullptr, stream);

        // --- per-seed host state ------------------------------------------------------------
        std::vector<double> L_prev(Sz, -std::numeric_limits<double>::infinity());
        std::vector<double> L2(Sz, 0.0), Lpp(Sz, 0.0), L_new(Sz, 0.0);
        std::vector<double> rr(Sz, 0.0), vv(Sz, 0.0), alpha(Sz, -1.0);
        std::vector<double> hActive(Sz, 1.0);
        std::vector<char> converged(Sz, 0), settled(Sz, 0), forceterm(Sz, 0);
        std::vector<int> outer_iters(Sz, 0), base_maps(Sz, 0);

        for (int outer = 0; outer < max_iter; ++outer) {
            bool all_conv = true;
            for (int s = 0; s < S; ++s)
                if (!converged[static_cast<std::size_t>(s)]) all_conv = false;
            if (all_conv) break;

            for (int s = 0; s < S; ++s)
                hActive[static_cast<std::size_t>(s)] =
                    converged[static_cast<std::size_t>(s)] ? 0.0 : 1.0;
            h2d_async(dActive, hActive.data(), Sz, stream);

            // theta0 = current; theta1 = M(theta0); theta2 = M(theta1).
            d2d(dQ.data(), dQ0.data(), Sz * NK);
            if (!supervised) d2d(dF.data(), dF0.data(), Sz * MK);
            em_map(dQ.data(), dF.data(), dFc.data(), dActive.data());
            d2d(dQ.data(), dQ1.data(), Sz * NK);
            if (!supervised) d2d(dF.data(), dF1.data(), Sz * MK);
            em_map(dQ.data(), dF.data(), dFc.data(), dActive.data());
            d2d(dQ.data(), dQ2.data(), Sz * NK);
            if (!supervised) d2d(dF.data(), dF2.data(), Sz * MK);
            for (int s = 0; s < S; ++s)
                if (!converged[static_cast<std::size_t>(s)]) base_maps[static_cast<std::size_t>(s)] += 2;

            // L2 = loglik(theta2); dQ/dF now hold theta2 (the plain-2-EM baseline / accumulator).
            loglik(dQ.data(), dF.data(), dLL2.data());
            d2h_async(L2.data(), dLL2, Sz, stream);

            // Fused SQUAREM norms rr = ||r||^2, vv = ||v||^2 over theta0/1/2 (Q part + F part).
            STEPPE_CUDA_CHECK(cudaMemsetAsync(drr.data(), 0, Sz * sizeof(double), stream));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dvv.data(), 0, Sz * sizeof(double), stream));
            launch_admix_squarem_norms(dQ0.data(), dQ1.data(), dQ2.data(), static_cast<long>(NK),
                                       static_cast<long>(NK), drr.data(), dvv.data(), S, stream);
            if (!supervised)
                launch_admix_squarem_norms(dF0.data(), dF1.data(), dF2.data(),
                                           static_cast<long>(MK), static_cast<long>(MK), drr.data(),
                                           dvv.data(), S, stream);
            d2h_async(rr.data(), drr, Sz, stream);
            d2h_async(vv.data(), dvv, Sz, stream);
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));

            // Steplength (SqS3): alpha = -sqrt(rr/vv), clamped to the EM floor alpha <= -1;
            // degenerate vv ~ 0 -> alpha = -1 (a plain EM step). MUST_FIX 2: delta strictly > 0.
            std::vector<double> delta(Sz, 0.0);
            for (int s = 0; s < S; ++s) {
                const std::size_t z = static_cast<std::size_t>(s);
                settled[z] = converged[z];
                forceterm[z] = 0;
                L_new[z] = L2[z];  // default accept = theta2 (dQ/dF already hold it)
                delta[z] = kSqDelta * std::max(1.0, std::fabs(L2[z]));
                if (converged[z]) continue;
                double a;
                if (!(std::isfinite(rr[z]) && std::isfinite(vv[z])) || vv[z] <= kSqVvTiny) {
                    a = -1.0;
                } else {
                    a = -std::sqrt(rr[z] / vv[z]);
                    if (a > -1.0) a = -1.0;
                }
                alpha[z] = a;
            }
            h2d_async(dAlpha, alpha.data(), Sz, stream);

            // --- per-seed backtracking to the plain-EM floor (monotone accept) --------------
            for (int bt = 0; bt <= kSqMaxBacktrack; ++bt) {
                bool any_unsettled = false;
                for (int s = 0; s < S; ++s) {
                    const std::size_t z = static_cast<std::size_t>(s);
                    hActive[z] = settled[z] ? 0.0 : 1.0;  // frozen once accepted/converged
                    if (!settled[z]) { any_unsettled = true; base_maps[z] += 1; }
                }
                if (!any_unsettled) break;
                h2d_async(dActive, hActive.data(), Sz, stream);

                // theta' = c0 theta0 + c1 theta1 + c2 theta2 (avoids materializing r,v).
                launch_admix_squarem_combine(dQ0.data(), dQ1.data(), dQ2.data(), dQp.data(),
                                             static_cast<long>(NK), static_cast<long>(NK),
                                             dAlpha.data(), dActive.data(), S, stream);
                launch_admix_project_q(dQp.data(), N, K, S, static_cast<long>(NK), dActive.data(),
                                       stream);
                double* Fcand = dF.data();
                double* Fccand = dFc.data();
                if (!supervised) {
                    launch_admix_squarem_combine(dF0.data(), dF1.data(), dF2.data(), dFp.data(),
                                                 static_cast<long>(MK), static_cast<long>(MK),
                                                 dAlpha.data(), dActive.data(), S, stream);
                    launch_admix_project_f(dFp.data(), dFcp.data(), static_cast<long>(MK), S,
                                           static_cast<long>(MK), kAdmixEps, dActive.data(), stream);
                    Fcand = dFp.data();
                    Fccand = dFcp.data();
                }
                // theta'' = M(theta') : stabilizing base map (restores exact feasibility).
                em_map(dQp.data(), Fcand, Fccand, dActive.data());
                loglik(dQp.data(), Fcand, dLL.data());
                d2h_async(Lpp.data(), dLL, Sz, stream);
                STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));

                bool changed_alpha = false;
                for (int s = 0; s < S; ++s) {
                    const std::size_t z = static_cast<std::size_t>(s);
                    if (settled[z]) continue;
                    const bool ok = std::isfinite(Lpp[z]) && (Lpp[z] >= L2[z] - delta[z]);
                    if (forceterm[z] || ok) {
                        // Accept theta'' as the seed's next iterate (commit into dQ/dF).
                        d2d_seed(dQp.data(), dQ.data(), s, NK);
                        if (!supervised) d2d_seed(dFp.data(), dF.data(), s, MK);
                        L_new[z] = Lpp[z];
                        settled[z] = 1;
                    } else {
                        // Reject: halve toward the EM floor alpha -> -1. MUST_FIX 3: on the cap
                        // OR the near-(-1) band, snap to alpha=-1 EXACTLY and force an
                        // unconditional plain-EM step next round (theta'' = M(theta2) = theta3).
                        double a = (alpha[z] - 1.0) * 0.5;
                        if (bt + 1 >= kSqMaxBacktrack || std::fabs(a + 1.0) < kSqTermBand) {
                            a = -1.0;
                            forceterm[z] = 1;
                        }
                        alpha[z] = a;
                        changed_alpha = true;
                    }
                }
                if (changed_alpha) h2d_async(dAlpha, alpha.data(), Sz, stream);
            }

            // Commit done in dQ/dF (accepted seeds = theta''; any unsettled kept theta2).
            launch_admix_complement_b(dF.data(), dFc.data(), static_cast<long>(MK), S,
                                      static_cast<long>(MK), nullptr, stream);
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));

            // Per-seed convergence on the accepted post-stabilization loglik.
            for (int s = 0; s < S; ++s) {
                const std::size_t z = static_cast<std::size_t>(s);
                if (converged[z]) continue;
                outer_iters[z] += 1;
                if (std::isfinite(L_new[z]) && std::isfinite(L_prev[z]) &&
                    std::fabs(L_new[z] - L_prev[z]) < tol * std::max(1.0, std::fabs(L_new[z])))
                    converged[z] = 1;
                L_prev[z] = L_new[z];
            }
        }

        // --- select the best restart + read back its Q/F ------------------------------------
        int best = 0;
        for (int s = 0; s < S; ++s) {
            const std::size_t z = static_cast<std::size_t>(s);
            out.seed_loglik[z] = L_prev[z];
            out.seed_iters[z] = outer_iters[z];
            out.seed_converged[z] = converged[z] ? 1 : 0;
            if (s == 0 || L_prev[z] > L_prev[static_cast<std::size_t>(best)]) best = s;
        }
        best_ll = L_prev[static_cast<std::size_t>(best)];
        out.best_seed = best;
        out.iters_run = outer_iters[static_cast<std::size_t>(best)];
        out.base_map_evals = base_maps[static_cast<std::size_t>(best)];
        out.converged = converged[static_cast<std::size_t>(best)] != 0;
        bestQ.assign(NK, 0.0);
        bestF.assign(MK, 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(bestQ.data(),
                                          dQ.data() + static_cast<std::size_t>(best) * NK,
                                          NK * sizeof(double), cudaMemcpyDeviceToHost, stream));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(bestF.data(),
                                          dF.data() + static_cast<std::size_t>(best) * MK,
                                          MK * sizeof(double), cudaMemcpyDeviceToHost, stream));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // --- col-major Q/F -> row-major result --------------------------------------------------
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
