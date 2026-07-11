// src/device/cuda/cuda_backend_pca.cu
//
// CudaBackend override for standalone genotype PCA (`steppe pca`). It uploads the packed
// genotype tile device-resident, SNP-tiles the Patterson-standardize kernel + cuBLAS SYRK
// to accumulate the sample x sample covariance C = Z*Z^T entirely on the GPU (emulated-FP64
// default, the matmul-heavy path — copied from the fit engine's SYRK block), symmetrizes it,
// eigendecomposes it with cuSOLVER Dsyevd (native FP64 — the cancellation/eigen carve-out),
// projects the top-K eigenvectors to sample coordinates, and D2H's ONLY the small N*K coords
// + the eigen spectrum + the used-SNP counter. The covariance/eigen compute is all on the
// GPU (no host genotype loop). A CUDA TU private to steppe_device, mirroring
// cuda_backend_fst.cu / cuda_backend_sfs.cu.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <span>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "steppe/pca.hpp"  // PcaProjectMode

#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/f2_block_kernel.cuh"      // engage_f2_precision, emulation_honorable
#include "device/cuda/pca_standardize_kernel.cuh"
#include "device/cuda/pca_truncated_eig.cuh"     // pca_truncated_topk (randomized top-K)
#include "device/cuda/qpadm_fit_kernels.cuh"    // launch_symmetrize_lower_to_full

namespace steppe::device {

namespace {
// Fraction of free VRAM the standardized Z operand tile may occupy (leaves room for the
// packed tile, the N x N covariance, and the eigensolver workspace).
constexpr std::size_t kPcaZBudgetDivisor = 4;
constexpr std::size_t kPcaFreeVramFallbackBytes = std::size_t{1} << 30;  // 1 GiB

// Matrix-free path: the Z-tile budget is a SMALLER fraction of the free VRAM that remains AFTER
// the resident packed tile, because the emulated-FP64 (fixed-point Ozaki) GEMM allocates an
// internal working set that scales with the Z_t operand — and the matrix-free sweep issues TWO
// such GEMMs per tile against Z_t. Taking free/8 (measured post-packed) keeps dZ + that working
// set + the O(N*L) sketches well inside VRAM at biobank N; a bigger tile buys nothing (the extra
// SNP-tiles are cheap, memory-bound re-standardizes) and risks OOM in the emulated GEMM scratch.
constexpr std::size_t kPcaRandZBudgetDivisor = 8;

// Randomized truncated-eigensolver tuning. Oversample p and subspace-iteration count q
// govern top-K convergence: the error on the K-th Ritz pair ~ (λ_{L+1}/λ_K)^(2q+1). The AADR
// covariance spectrum decays SLOWLY (dozens–hundreds of significant PCs, no clean gap at
// index K), so the near-degenerate leading block needs a large subspace L=K+p and several
// iterations to converge the top-K to the exact-solver reference (the rotation-invariant Gram
// gate). This is affordable: the sketch cost is ~5*N^2*L flops — for N=23k, L=256 that is
// ~180x FEWER flops than the old (4/3)N^3 Dsyevd, and the O(N*L) sketch/basis buffers total
// ~190 MB (4x N*L*8B at N=23k, L=256), vs the removed ~8.5 GB full-Dsyevd workspace.
// Chosen against the real-AADR scikit-allel Gram gate (fit9, 430 samples, K=10): p=246/q=6
// drives the top-10 subspace to rel≈2.5e-6 vs the exact solver (5e-3 gate), with wide margin
// for the slower-decaying worldwide 23k cohort. Cost is a rounding error on the total wall —
// the eigensolve is ~15x fewer flops than the removed (4/3)N^3 Dsyevd and ~1e-6 of the SYRK
// covariance formation — so this buys robustness essentially for free.
constexpr int kPcaOversampleDefault = 246;   // L = K + p = 256 for the K=10 default
constexpr int kPcaSubspaceItersDefault = 6;

// Optional env overrides (reproducibility preserved — the sketch seed is fixed regardless).
int pca_oversample() {
    if (const char* e = std::getenv("STEPPE_PCA_OVERSAMPLE")) {
        const int v = std::atoi(e);
        if (v >= 0) return v;
    }
    return kPcaOversampleDefault;
}
int pca_subspace_iters() {
    if (const char* e = std::getenv("STEPPE_PCA_SUBSPACE_ITERS")) {
        const int v = std::atoi(e);
        if (v >= 0) return v;
    }
    return kPcaSubspaceItersDefault;
}

// AUTO solver resolution (solver_mode 2 -> exact/randomized). Use the matrix-free randomized
// path once the sample count is clearly large OR the would-be dense N x N Gram (8*N^2 bytes)
// would claim a big slice of free VRAM (the quadratic wall the exact path hits ~23k on 32 GB).
// The sample floor sits ABOVE the committed 430-sample golden and the 2-8k gate subset so
// `auto` stays on the byte-identical exact path there; the 22-27k present-day cohort trips it.
constexpr long kPcaAutoRandomizedSampleFloor = 8000;
constexpr std::size_t kPcaAutoDenseVramDivisor = 4;  // dense Gram > free/4 -> randomized

bool pca_use_randomized(int solver_mode, long N, std::size_t free_vram_bytes) {
    if (solver_mode == 1) return true;   // explicit --pca-solver randomized
    if (solver_mode == 0) return false;  // explicit --pca-solver exact
    std::size_t free_b = free_vram_bytes;
    if (free_b == 0) free_b = kPcaFreeVramFallbackBytes;
    const std::size_t dense_gram_bytes =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(N) * sizeof(double);
    return (N > kPcaAutoRandomizedSampleFloor) ||
           (dense_gram_bytes > free_b / kPcaAutoDenseVramDivisor);
}
}  // namespace

PcaEig CudaBackend::pca_covariance_eig(const DecodeTileView& tile, int k, int solver_mode,
                                       const Precision& precision) {
    guard_device();
    PcaEig out;
    out.precision_tag =
        (precision.kind == Precision::Kind::EmulatedFp64 &&
         capabilities().emulated_fp64_honorable)
            ? Precision::Kind::EmulatedFp64
            : Precision::Kind::Fp64;

    const long N = static_cast<long>(tile.n_individuals);
    const long M = static_cast<long>(tile.n_snp);
    if (N <= 0 || M <= 0 || k <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }
    const int Nn = static_cast<int>(N);
    const int K = static_cast<int>(std::min<long>(k, N));

    const bool use_randomized =
        pca_use_randomized(solver_mode, N, capabilities().free_vram_bytes);

    // Randomized subspace-iteration tuning + the L=K+p subspace, shared by both paths (the
    // exact path runs the SAME truncated solver on the resident dense Gram; the randomized
    // path runs it matrix-free). The lifted Ritz block dEvecL (N x L, ascending) and the L
    // Ritz values dEvalL, plus the trace + used-SNP counters, are produced by whichever branch
    // runs and consumed by the shared projection/D2H tail below.
    solver_.set_stream(stream_.get());
    const int oversample = pca_oversample();
    const int subspace_iters = pca_subspace_iters();
    const int L = static_cast<int>(std::min<long>(static_cast<long>(K) + oversample, N));
    DeviceBuffer<double> dEvecL(static_cast<std::size_t>(N) * static_cast<std::size_t>(L));
    DeviceBuffer<double> dEvalL(static_cast<std::size_t>(L));
    DeviceBuffer<double> dTrace(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dTrace.data(), 0, sizeof(double), stream_.get()));
    DeviceBuffer<unsigned long long> dUsed(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dUsed.data(), 0, sizeof(unsigned long long), stream_.get()));
    int L_used = 0;

    if (!use_randomized) {
        // ---- EXACT path (byte-unchanged): form the dense N x N Gram then solve on it ----
        // Sample x sample covariance accumulator (column-major N x N), zeroed for beta=0/1 SYRK.
        const std::size_t NN = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
        DeviceBuffer<double> dC(NN);
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dC.data(), 0, NN * sizeof(double), stream_.get()));

        // Choose a SNP tile that bounds the Z operand footprint to ~free/kPcaZBudgetDivisor,
        // keeping the standardize+SYRK memory O(N*tileM + N^2) (the extract_f2 tiling idiom).
        std::size_t free_b = capabilities().free_vram_bytes;
        if (free_b == 0) free_b = kPcaFreeVramFallbackBytes;
        std::size_t budget = free_b / kPcaZBudgetDivisor;
        const std::size_t z_row_bytes = static_cast<std::size_t>(N) * sizeof(double);
        if (budget < z_row_bytes) budget = z_row_bytes;
        long tileM = static_cast<long>(budget / z_row_bytes);
        if (tileM < 1) tileM = 1;
        if (tileM > M) tileM = M;

        {
            // The packed genotype tile, the standardize center/scale scratch, and the Z operand
            // tile ALL live ONLY inside this block so RAII frees them (the ~7 GB packed tile +
            // the ~free/kPcaZBudgetDivisor Z tile, together ~11 GB at 23k) BEFORE the
            // eigensolver allocates its workspace — the marginal alloc that OOMs the full
            // cohort. Nothing after the block references them (symmetrize + the truncated solve
            // read only dC).
            const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
            DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
            if (packed_bytes > 0) {
                h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());
            }

            DeviceBuffer<double> dCenter(static_cast<std::size_t>(tileM));
            DeviceBuffer<double> dInv(static_cast<std::size_t>(tileM));
            DeviceBuffer<double> dZ(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));

            // Covariance SYRK: emulated-FP64 default (matmul-heavy), the fit-engine block.
            const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            for (long s_lo = 0; s_lo < M; s_lo += tileM) {
                const long tm = std::min<long>(tileM, M - s_lo);
                launch_pca_snp_scale(dPacked.data(), tile.bytes_per_record,
                                     tile.n_individuals, s_lo, tm, dCenter.data(),
                                     dInv.data(), dUsed.data(), stream_.get());
                launch_pca_standardize(dPacked.data(), tile.bytes_per_record,
                                       tile.n_individuals, s_lo, tm, dCenter.data(),
                                       dInv.data(), dZ.data(), stream_.get());
                const double alpha = 1.0;
                const double beta = (s_lo == 0) ? 0.0 : 1.0;
                CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                         Nn, static_cast<int>(tm), &alpha, dZ.data(), Nn,
                                         &beta, dC.data(), Nn));
            }
        }
        launch_symmetrize_lower_to_full(dC.data(), Nn, stream_.get());

        // Truncated top-K eigensolve on the resident dense Gram (the pre-factoring reference
        // path, byte-identical). dC is READ-ONLY, so trace(dC) below stays valid.
        const PcaTruncatedEig te = pca_truncated_topk(
            blas_.get(), solver_.get(), stream_.get(), dC.data(), Nn, K, oversample,
            subspace_iters, precision, dEvecL.data(), dEvalL.data(), &L_used);
        if (te.status != Status::Ok) {
            out.status = te.status;
            return out;
        }

        // var_explained denominator = trace(dC) = Σ_all λ (native-FP64 diagonal reduction).
        launch_pca_trace(dC.data(), Nn, dTrace.data(), stream_.get());
    } else {
        // ---- MATRIX-FREE randomized path (biobank-scale): NEVER form the N x N Gram ----
        // Top-K eigenvectors of C = Z Zᵀ via a randomized range finder whose covariance action
        // C·Q is a streamed two-GEMM sweep CQ = Σ_tiles Z_t (Z_tᵀ Q) over SNP tiles — the exact
        // same operator the dense path applies, but no O(N^2) object is ever allocated (peak
        // memory is O(N*(bytes_per_record + L)), linear in N). Trace(C) = ||Z||_F^2 is
        // accumulated in the precompute sweep, replacing trace(dC).

        // The packed genotype tile + the Z/T scratch stay resident ACROSS every sweep (the
        // matvec re-standardizes them q+2 times) — so, unlike the exact path, they are NOT
        // freed before the solve; there is no N x N object competing for the VRAM. Allocate the
        // packed tile FIRST so the Z-tile budget below is taken from the free VRAM that REMAINS
        // after it (mirrors the exact path measuring free after dC), not the full pool.
        const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
        DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
        if (packed_bytes > 0) h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());

        // Cache the full-range Patterson center[M]/inv_scale[M] once (no re-fold per sweep).
        DeviceBuffer<double> dCenterAll(static_cast<std::size_t>(M));
        DeviceBuffer<double> dInvAll(static_cast<std::size_t>(M));

        // SNP tile bounding BOTH the Z operand (N x tileM) and the T = Zᵀ Q scratch (tileM x L):
        // each SNP column costs (N + L) doubles across the two. Sized to a conservative fraction
        // of the free VRAM that remains after the packed tile, leaving headroom for the emulated
        // GEMM's fixed-point working set (see kPcaRandZBudgetDivisor).
        std::size_t free_b = capabilities().free_vram_bytes;
        if (free_b == 0) free_b = kPcaFreeVramFallbackBytes;
        std::size_t budget = free_b / kPcaRandZBudgetDivisor;
        const std::size_t col_bytes =
            (static_cast<std::size_t>(N) + static_cast<std::size_t>(L)) * sizeof(double);
        if (budget < col_bytes) budget = col_bytes;
        long tileM = static_cast<long>(budget / col_bytes);
        if (tileM < 1) tileM = 1;
        if (tileM > M) tileM = M;

        DeviceBuffer<double> dZ(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dT(static_cast<std::size_t>(tileM) * static_cast<std::size_t>(L));

        // Precompute sweep: cache center/inv, count used SNPs, accumulate trace = ||Z||_F^2.
        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            launch_pca_snp_scale(dPacked.data(), tile.bytes_per_record, tile.n_individuals, s_lo,
                                 tm, dCenterAll.data() + s_lo, dInvAll.data() + s_lo, dUsed.data(),
                                 stream_.get());
            launch_pca_standardize(dPacked.data(), tile.bytes_per_record, tile.n_individuals, s_lo,
                                   tm, dCenterAll.data() + s_lo, dInvAll.data() + s_lo, dZ.data(),
                                   stream_.get());
            launch_pca_accumulate_sumsq(dZ.data(), static_cast<long>(N) * tm, dTrace.data(),
                                        stream_.get());
        }

        // Streamed covariance matvec CQ (N x ncols) = Σ_tiles Z_t (Z_tᵀ Q). Per tile: (a)
        // re-standardize Z_t from the resident packed tile (cached center/inv, no re-fold); (b)
        // T = Z_tᵀ Q (tm x ncols); (c) CQ += Z_t T (N x ncols). Both GEMMs emulated-FP64 (match
        // the dense sketch + SYRK). Buffers captured by value (pointers/handles/POD).
        cublasHandle_t blas = blas_.get();
        cudaStream_t stream = stream_.get();
        const std::uint8_t* packed_ptr = dPacked.data();
        const std::size_t bpr = tile.bytes_per_record;
        const std::size_t Nrec = tile.n_individuals;
        double* dZp = dZ.data();
        double* dTp = dT.data();
        const double* dCenterP = dCenterAll.data();
        const double* dInvP = dInvAll.data();
        const long tileM_c = tileM;
        const long M_c = M;
        const PcaCovMatvec matvec = [=](const double* Q_in, double* CQ_out, int ncols) {
            const MathModeScope mode(blas, CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas, precision);
            const double one = 1.0;
            for (long s_lo = 0; s_lo < M_c; s_lo += tileM_c) {
                const long tm = std::min<long>(tileM_c, M_c - s_lo);
                launch_pca_standardize(packed_ptr, bpr, Nrec, s_lo, tm, dCenterP + s_lo,
                                       dInvP + s_lo, dZp, stream);
                // T (tm x ncols) = Z_tᵀ Q : (OP_T, OP_N) m=tm, n=ncols, k=N, beta=0.
                const double zero = 0.0;
                CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(tm),
                                         ncols, Nn, &one, dZp, Nn, Q_in, Nn, &zero, dTp,
                                         static_cast<int>(tm)));
                // CQ (N x ncols) += Z_t T : (OP_N, OP_N) m=N, n=ncols, k=tm, beta = first?0:1.
                const double beta = (s_lo == 0) ? 0.0 : 1.0;
                CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, Nn, ncols,
                                         static_cast<int>(tm), &one, dZp, Nn, dTp,
                                         static_cast<int>(tm), &beta, CQ_out, Nn));
            }
        };

        const PcaTruncatedEig te = pca_truncated_topk_op(
            blas, solver_.get(), stream, matvec, Nn, K, oversample, subspace_iters, precision,
            dEvecL.data(), dEvalL.data(), &L_used);
        if (te.status != Status::Ok) {
            out.status = te.status;
            return out;
        }
        // dTrace already holds ||Z||_F^2 from the precompute sweep.
    }

    // ---- Shared tail: project top-K coords, D2H the small results, assemble the spectrum ----
    // The eigenvector block is N x L, ascending Ritz value (largest at the tail => PC k reads
    // column L-1-k). coord = evec*sqrt(eval).
    DeviceBuffer<double> dCoords(static_cast<std::size_t>(N) * static_cast<std::size_t>(K));
    launch_pca_coords(dEvecL.data(), dEvalL.data(), Nn, L, K, dCoords.data(), stream_.get());

    out.coords.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(K), 0.0);
    std::vector<double> eval_L(static_cast<std::size_t>(L), 0.0);
    double trace_h = 0.0;
    unsigned long long used_h = 0;
    d2h_async(out.coords.data(), dCoords,
              static_cast<std::size_t>(N) * static_cast<std::size_t>(K), stream_.get());
    d2h_async(eval_L.data(), dEvalL, static_cast<std::size_t>(L), stream_.get());
    d2h_async(&trace_h, dTrace, 1, stream_.get());
    d2h_async(&used_h, dUsed, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    const long double total = static_cast<long double>(trace_h);
    out.eigenvalues.assign(static_cast<std::size_t>(K), 0.0);
    out.var_explained.assign(static_cast<std::size_t>(K), 0.0);
    for (int kk = 0; kk < K; ++kk) {
        // Ritz values ascending -> descending PCs read from the tail.
        const double lambda = eval_L[static_cast<std::size_t>(L - 1 - kk)];
        out.eigenvalues[static_cast<std::size_t>(kk)] = lambda;
        out.var_explained[static_cast<std::size_t>(kk)] =
            (total != 0.0L) ? static_cast<double>(static_cast<long double>(lambda) / total) : 0.0;
    }

    out.N = Nn;
    out.K = K;
    out.n_snp_used = static_cast<long>(used_h);
    out.n_snp_monomorphic = M - out.n_snp_used;
    out.status = Status::Ok;
    return out;
}

PcaEig CudaBackend::pca_covariance_eig_dosage(const DosageTileView& tile, int k,
                                              const Precision& precision) {
    guard_device();
    PcaEig out;
    out.precision_tag =
        (precision.kind == Precision::Kind::EmulatedFp64 &&
         capabilities().emulated_fp64_honorable)
            ? Precision::Kind::EmulatedFp64
            : Precision::Kind::Fp64;

    const long N = static_cast<long>(tile.n_individuals);
    const long M = static_cast<long>(tile.n_snp);
    if (N <= 0 || M <= 0 || k <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }
    const int Nn = static_cast<int>(N);
    const int K = static_cast<int>(std::min<long>(k, N));

    // v1: the EXACT dense-Gram path only (region/AADR scale). The randomized matrix-free path
    // is a documented follow-on; it reuses this SAME dosage standardize with a streamed matvec.
    solver_.set_stream(stream_.get());
    const int oversample = pca_oversample();
    const int subspace_iters = pca_subspace_iters();
    const int L = static_cast<int>(std::min<long>(static_cast<long>(K) + oversample, N));
    DeviceBuffer<double> dEvecL(static_cast<std::size_t>(N) * static_cast<std::size_t>(L));
    DeviceBuffer<double> dEvalL(static_cast<std::size_t>(L));
    DeviceBuffer<double> dTrace(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dTrace.data(), 0, sizeof(double), stream_.get()));
    DeviceBuffer<unsigned long long> dUsed(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dUsed.data(), 0, sizeof(unsigned long long), stream_.get()));
    int L_used = 0;

    const std::size_t NN = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    DeviceBuffer<double> dC(NN);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dC.data(), 0, NN * sizeof(double), stream_.get()));

    // SNP tile bounding the Z operand footprint to ~free/kPcaZBudgetDivisor (mirrors the exact
    // integer path). The resident FP32 dosage tile is N*M*4 bytes; the Z tile stays FP64.
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kPcaFreeVramFallbackBytes;
    std::size_t budget = free_b / kPcaZBudgetDivisor;
    const std::size_t z_row_bytes = static_cast<std::size_t>(N) * sizeof(double);
    if (budget < z_row_bytes) budget = z_row_bytes;
    long tileM = static_cast<long>(budget / z_row_bytes);
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;

    {
        // The resident dosage tile + the standardize scratch + the Z operand live ONLY inside
        // this block so RAII frees them before the eigensolver allocates its workspace.
        const std::size_t dosage_elems = static_cast<std::size_t>(N) * static_cast<std::size_t>(M);
        DeviceBuffer<float> dDosage(dosage_elems == 0 ? 1u : dosage_elems);
        if (dosage_elems > 0) {
            h2d_async(dDosage, tile.dosage, dosage_elems, stream_.get());
        }

        DeviceBuffer<double> dCenter(static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dInv(static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dZ(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));

        // Covariance SYRK: emulated-FP64 default (matmul-heavy), the fit-engine block.
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            launch_pca_dosage_snp_scale(dDosage.data(), tile.n_individuals, tile.n_snp, s_lo, tm,
                                        dCenter.data(), dInv.data(), dUsed.data(), stream_.get());
            launch_pca_dosage_standardize(dDosage.data(), tile.n_individuals, tile.n_snp, s_lo, tm,
                                          dCenter.data(), dInv.data(), dZ.data(), stream_.get());
            const double alpha = 1.0;
            const double beta = (s_lo == 0) ? 0.0 : 1.0;
            CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, Nn,
                                     static_cast<int>(tm), &alpha, dZ.data(), Nn, &beta,
                                     dC.data(), Nn));
        }
    }
    launch_symmetrize_lower_to_full(dC.data(), Nn, stream_.get());

    // Truncated top-K eigensolve on the resident dense Gram (shared with the integer path).
    const PcaTruncatedEig te = pca_truncated_topk(
        blas_.get(), solver_.get(), stream_.get(), dC.data(), Nn, K, oversample, subspace_iters,
        precision, dEvecL.data(), dEvalL.data(), &L_used);
    if (te.status != Status::Ok) {
        out.status = te.status;
        return out;
    }
    launch_pca_trace(dC.data(), Nn, dTrace.data(), stream_.get());

    // ---- Shared tail: project top-K coords, D2H the small results, assemble the spectrum ----
    DeviceBuffer<double> dCoords(static_cast<std::size_t>(N) * static_cast<std::size_t>(K));
    launch_pca_coords(dEvecL.data(), dEvalL.data(), Nn, L, K, dCoords.data(), stream_.get());

    out.coords.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(K), 0.0);
    std::vector<double> eval_L(static_cast<std::size_t>(L), 0.0);
    double trace_h = 0.0;
    unsigned long long used_h = 0;
    d2h_async(out.coords.data(), dCoords,
              static_cast<std::size_t>(N) * static_cast<std::size_t>(K), stream_.get());
    d2h_async(eval_L.data(), dEvalL, static_cast<std::size_t>(L), stream_.get());
    d2h_async(&trace_h, dTrace, 1, stream_.get());
    d2h_async(&used_h, dUsed, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    const long double total = static_cast<long double>(trace_h);
    out.eigenvalues.assign(static_cast<std::size_t>(K), 0.0);
    out.var_explained.assign(static_cast<std::size_t>(K), 0.0);
    for (int kk = 0; kk < K; ++kk) {
        const double lambda = eval_L[static_cast<std::size_t>(L - 1 - kk)];
        out.eigenvalues[static_cast<std::size_t>(kk)] = lambda;
        out.var_explained[static_cast<std::size_t>(kk)] =
            (total != 0.0L) ? static_cast<double>(static_cast<long double>(lambda) / total) : 0.0;
    }

    out.N = Nn;
    out.K = K;
    out.n_snp_used = static_cast<long>(used_h);
    out.n_snp_monomorphic = M - out.n_snp_used;
    out.status = Status::Ok;
    return out;
}

namespace {
// Native-FP64 Cholesky solve of the SPD K x K system A x = b (A column-major, K tiny — the
// default K=10 per-target normal matrix). Returns false if A is not positive definite (a
// rank-deficient / singular target -> the caller falls back to the diagonal ratio estimate).
bool pca_chol_solve(const double* A, int K, const double* b, double* x) {
    std::vector<double> L(A, A + static_cast<std::size_t>(K) * static_cast<std::size_t>(K));
    for (int j = 0; j < K; ++j) {
        double d = L[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * K];
        for (int p = 0; p < j; ++p) {
            const double ljp = L[static_cast<std::size_t>(j) + static_cast<std::size_t>(p) * K];
            d -= ljp * ljp;
        }
        if (!(d > 0.0)) return false;
        const double ljj = std::sqrt(d);
        L[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * K] = ljj;
        for (int i = j + 1; i < K; ++i) {
            double s = L[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * K];
            for (int p = 0; p < j; ++p)
                s -= L[static_cast<std::size_t>(i) + static_cast<std::size_t>(p) * K] *
                     L[static_cast<std::size_t>(j) + static_cast<std::size_t>(p) * K];
            L[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * K] = s / ljj;
        }
    }
    std::vector<double> y(static_cast<std::size_t>(K), 0.0);
    for (int i = 0; i < K; ++i) {
        double s = b[i];
        for (int p = 0; p < i; ++p)
            s -= L[static_cast<std::size_t>(i) + static_cast<std::size_t>(p) * K] * y[static_cast<std::size_t>(p)];
        y[static_cast<std::size_t>(i)] = s / L[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * K];
    }
    for (int i = K - 1; i >= 0; --i) {
        double s = y[static_cast<std::size_t>(i)];
        for (int p = i + 1; p < K; ++p)
            s -= L[static_cast<std::size_t>(p) + static_cast<std::size_t>(i) * K] * x[p];
        x[i] = s / L[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * K];
    }
    return true;
}
}  // namespace

PcaProject CudaBackend::pca_project_lsq(const DecodeTileView& tile, int k,
                                        std::span<const int> ref_rows,
                                        std::span<const int> tgt_rows, int project_mode,
                                        const Precision& precision) {
    guard_device();
    PcaProject out;
    out.precision_tag =
        (precision.kind == Precision::Kind::EmulatedFp64 &&
         capabilities().emulated_fp64_honorable)
            ? Precision::Kind::EmulatedFp64
            : Precision::Kind::Fp64;

    const long M = static_cast<long>(tile.n_snp);
    const long N_ref = static_cast<long>(ref_rows.size());
    const long N_tgt = static_cast<long>(tgt_rows.size());
    if (M <= 0 || N_ref <= 0 || N_tgt <= 0 || k <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }
    const int Nr = static_cast<int>(N_ref);
    const int Nt = static_cast<int>(N_tgt);
    const int K = static_cast<int>(std::min<long>(k, N_ref));

    // Row-index lists (indices into the tile individual axis).
    DeviceBuffer<int> dRef(static_cast<std::size_t>(N_ref));
    DeviceBuffer<int> dTgt(static_cast<std::size_t>(N_tgt));
    h2d_async(dRef, ref_rows.data(), static_cast<std::size_t>(N_ref), stream_.get());
    h2d_async(dTgt, tgt_rows.data(), static_cast<std::size_t>(N_tgt), stream_.get());

    // Reference sample x sample covariance accumulator (column-major N_ref x N_ref).
    const std::size_t NN = static_cast<std::size_t>(N_ref) * static_cast<std::size_t>(N_ref);
    DeviceBuffer<double> dC(NN);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dC.data(), 0, NN * sizeof(double), stream_.get()));
    DeviceBuffer<unsigned long long> dUsed(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dUsed.data(), 0, sizeof(unsigned long long), stream_.get()));

    // Cache the full-range Patterson center[M] / inv_scale[M] (folded over ref rows only) so
    // pass 2 re-standardizes with NO re-fold (~19 MB at 1.2M SNPs).
    DeviceBuffer<double> dCenterAll(static_cast<std::size_t>(M));
    DeviceBuffer<double> dInvAll(static_cast<std::size_t>(M));

    // SNP tile bounding the Z_ref operand to ~free/kPcaZBudgetDivisor (over the ref rows).
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kPcaFreeVramFallbackBytes;
    std::size_t budget = free_b / kPcaZBudgetDivisor;
    const std::size_t z_row_bytes = static_cast<std::size_t>(N_ref) * sizeof(double);
    if (budget < z_row_bytes) budget = z_row_bytes;
    long tileM = static_cast<long>(budget / z_row_bytes);
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;

    // The packed genotype tile stays resident across BOTH passes (pass 2 re-standardizes the
    // same rows for the loadings — the design's "reuse the resident packed tile, no re-read").
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
    if (packed_bytes > 0) h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());

    // ---- Pass 1: ref-only fold + SYRK covariance ----
    {
        DeviceBuffer<double> dZ(static_cast<std::size_t>(N_ref) * static_cast<std::size_t>(tileM));
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            launch_pca_snp_scale_gather(dPacked.data(), tile.bytes_per_record, dRef.data(), N_ref,
                                        s_lo, tm, dCenterAll.data() + s_lo, dInvAll.data() + s_lo,
                                        dUsed.data(), stream_.get());
            launch_pca_standardize_gather(dPacked.data(), tile.bytes_per_record, dRef.data(), N_ref,
                                          s_lo, tm, dCenterAll.data() + s_lo, dInvAll.data() + s_lo,
                                          dZ.data(), stream_.get());
            const double alpha = 1.0;
            const double beta = (s_lo == 0) ? 0.0 : 1.0;
            CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, Nr,
                                     static_cast<int>(tm), &alpha, dZ.data(), Nr, &beta, dC.data(),
                                     Nr));
        }
    }
    launch_symmetrize_lower_to_full(dC.data(), Nr, stream_.get());

    // var_explained denominator = trace(dC) (native-FP64 diagonal reduction).
    DeviceBuffer<double> dTrace(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dTrace.data(), 0, sizeof(double), stream_.get()));
    launch_pca_trace(dC.data(), Nr, dTrace.data(), stream_.get());

    // Truncated top-K eigensolve of the reference covariance.
    solver_.set_stream(stream_.get());
    const int oversample = pca_oversample();
    const int subspace_iters = pca_subspace_iters();
    const int L = static_cast<int>(std::min<long>(static_cast<long>(K) + oversample, N_ref));
    DeviceBuffer<double> dEvecL(static_cast<std::size_t>(N_ref) * static_cast<std::size_t>(L));
    DeviceBuffer<double> dEvalL(static_cast<std::size_t>(L));
    int L_used = 0;
    const PcaTruncatedEig te = pca_truncated_topk(
        blas_.get(), solver_.get(), stream_.get(), dC.data(), Nr, K, oversample, subspace_iters,
        precision, dEvecL.data(), dEvalL.data(), &L_used);
    if (te.status != Status::Ok) {
        out.status = te.status;
        return out;
    }

    // Reference coords U*S, plus the loadings basis U (N_ref x K) and inv_S (1/S_k).
    DeviceBuffer<double> dCoordsRef(static_cast<std::size_t>(N_ref) * static_cast<std::size_t>(K));
    launch_pca_coords(dEvecL.data(), dEvalL.data(), Nr, te.L, K, dCoordsRef.data(), stream_.get());
    DeviceBuffer<double> dU(static_cast<std::size_t>(N_ref) * static_cast<std::size_t>(K));
    DeviceBuffer<double> dInvS(static_cast<std::size_t>(K));
    launch_pca_pack_basis(dEvecL.data(), dEvalL.data(), Nr, te.L, K, dU.data(), dInvS.data(),
                          stream_.get());

    // ---- Pass 2: loadings W = Z_ref^T U / S, then b = W_O^T z_O and A = W_O^T W_O ----
    DeviceBuffer<double> dB(static_cast<std::size_t>(K) * static_cast<std::size_t>(N_tgt));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dB.data(), 0,
                                      static_cast<std::size_t>(K) * static_cast<std::size_t>(N_tgt) *
                                          sizeof(double),
                                      stream_.get()));
    DeviceBuffer<double> dA(static_cast<std::size_t>(N_tgt) * static_cast<std::size_t>(K) *
                            static_cast<std::size_t>(K));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dA.data(), 0,
                                      static_cast<std::size_t>(N_tgt) * static_cast<std::size_t>(K) *
                                          static_cast<std::size_t>(K) * sizeof(double),
                                      stream_.get()));
    DeviceBuffer<unsigned long long> dMobs(static_cast<std::size_t>(N_tgt));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dMobs.data(), 0,
                                      static_cast<std::size_t>(N_tgt) * sizeof(unsigned long long),
                                      stream_.get()));
    {
        DeviceBuffer<double> dZref(static_cast<std::size_t>(N_ref) *
                                   static_cast<std::size_t>(tileM));
        DeviceBuffer<double> dW(static_cast<std::size_t>(tileM) * static_cast<std::size_t>(K));
        DeviceBuffer<double> dZtgt(static_cast<std::size_t>(N_tgt) *
                                   static_cast<std::size_t>(tileM));
        DeviceBuffer<std::uint8_t> dMask(static_cast<std::size_t>(N_tgt) *
                                         static_cast<std::size_t>(tileM));
        const MathModeScope gemm_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            // Re-standardize Z_ref (cached center/inv, no re-fold).
            launch_pca_standardize_gather(dPacked.data(), tile.bytes_per_record, dRef.data(), N_ref,
                                          s_lo, tm, dCenterAll.data() + s_lo, dInvAll.data() + s_lo,
                                          dZref.data(), stream_.get());
            // W_tile (tm x K) = Z_ref^T U.
            const double one = 1.0, zero = 0.0;
            CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(tm), K,
                                     Nr, &one, dZref.data(), Nr, dU.data(), Nr, &zero, dW.data(),
                                     static_cast<int>(tm)));
            launch_pca_scale_loadings(dW.data(), tm, K, dInvS.data(), stream_.get());
            // Target standardize + observed mask.
            launch_pca_standardize_target(dPacked.data(), tile.bytes_per_record, dTgt.data(), N_tgt,
                                          s_lo, tm, dCenterAll.data() + s_lo, dInvAll.data() + s_lo,
                                          dZtgt.data(), dMask.data(), stream_.get());
            // b (K x N_tgt) += W^T Z_tgt^T.
            CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_T, K, Nt,
                                     static_cast<int>(tm), &one, dW.data(), static_cast<int>(tm),
                                     dZtgt.data(), Nt, &one, dB.data(), K));
            // A += masked W W^T, and m_obs += Σ mask (native FP64 assembly).
            launch_pca_accumulate_A(dW.data(), dMask.data(), tm, Nt, K, dA.data(), stream_.get());
            launch_pca_accumulate_mobs(dMask.data(), tm, Nt, dMobs.data(), stream_.get());
        }
    }

    // D2H the small results: ref coords, spectrum, trace, used cnt, per-target A/b/m_obs.
    out.coords_ref.assign(static_cast<std::size_t>(N_ref) * static_cast<std::size_t>(K), 0.0);
    std::vector<double> eval_L(static_cast<std::size_t>(te.L), 0.0);
    std::vector<double> A_h(static_cast<std::size_t>(N_tgt) * static_cast<std::size_t>(K) *
                                static_cast<std::size_t>(K),
                            0.0);
    std::vector<double> b_h(static_cast<std::size_t>(K) * static_cast<std::size_t>(N_tgt), 0.0);
    std::vector<unsigned long long> mobs_h(static_cast<std::size_t>(N_tgt), 0);
    double trace_h = 0.0;
    unsigned long long used_h = 0;
    d2h_async(out.coords_ref.data(), dCoordsRef,
              static_cast<std::size_t>(N_ref) * static_cast<std::size_t>(K), stream_.get());
    d2h_async(eval_L.data(), dEvalL, static_cast<std::size_t>(te.L), stream_.get());
    d2h_async(A_h.data(), dA, A_h.size(), stream_.get());
    d2h_async(b_h.data(), dB, b_h.size(), stream_.get());
    d2h_async(mobs_h.data(), dMobs, mobs_h.size(), stream_.get());
    d2h_async(&trace_h, dTrace, 1, stream_.get());
    d2h_async(&used_h, dUsed, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // Reference spectrum.
    const long double total = static_cast<long double>(trace_h);
    out.eigenvalues.assign(static_cast<std::size_t>(K), 0.0);
    out.var_explained.assign(static_cast<std::size_t>(K), 0.0);
    for (int kk = 0; kk < K; ++kk) {
        const double lambda = eval_L[static_cast<std::size_t>(te.L - 1 - kk)];
        out.eigenvalues[static_cast<std::size_t>(kk)] = lambda;
        out.var_explained[static_cast<std::size_t>(kk)] =
            (total != 0.0L) ? static_cast<double>(static_cast<long double>(lambda) / total) : 0.0;
    }

    // Per-target K x K solve (native FP64): full lsq, or the diagonal ratio (scaled mode /
    // rank-deficient / no-data fallback). M_used = the reference-polymorphic site count.
    const long M_used = static_cast<long>(used_h);
    out.coords_tgt.assign(static_cast<std::size_t>(N_tgt) * static_cast<std::size_t>(K), 0.0);
    out.m_obs.assign(static_cast<std::size_t>(N_tgt), 0);
    out.status_per_target.assign(static_cast<std::size_t>(N_tgt), 0);
    for (int j = 0; j < Nt; ++j) {
        const long m_obs = static_cast<long>(mobs_h[static_cast<std::size_t>(j)]);
        out.m_obs[static_cast<std::size_t>(j)] = m_obs;
        const double* bj = b_h.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(K);
        double* xj =
            out.coords_tgt.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(K);
        if (m_obs <= 0) {
            out.status_per_target[static_cast<std::size_t>(j)] = 2;  // no usable data -> zeros
            continue;
        }
        bool solved = false;
        if (project_mode == static_cast<int>(PcaProjectMode::Lsq) && m_obs >= K) {
            const double* Aj =
                A_h.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(K) *
                                 static_cast<std::size_t>(K);
            solved = pca_chol_solve(Aj, K, bj, xj);
        }
        if (solved) {
            out.status_per_target[static_cast<std::size_t>(j)] = 0;  // full lsq
        } else {
            // Diagonal ratio a = (M_used/m_obs)*b (== the W_O^T W_O ≈ (m_obs/M)*I limit).
            const double ratio =
                m_obs > 0 ? static_cast<double>(M_used) / static_cast<double>(m_obs) : 0.0;
            for (int kk = 0; kk < K; ++kk) xj[kk] = ratio * bj[kk];
            out.status_per_target[static_cast<std::size_t>(j)] = 1;  // ratio path
        }
    }

    out.N_ref = Nr;
    out.N_tgt = Nt;
    out.K = K;
    out.n_snp_used = M_used;
    out.n_snp_monomorphic = M - M_used;
    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
