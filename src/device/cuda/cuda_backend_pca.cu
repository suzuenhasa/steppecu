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
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

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
}  // namespace

PcaEig CudaBackend::pca_covariance_eig(const DecodeTileView& tile, int k,
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

    // Sample x sample covariance accumulator (column-major N x N), zeroed for beta=0/1 SYRK.
    const std::size_t NN = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    DeviceBuffer<double> dC(NN);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dC.data(), 0, NN * sizeof(double), stream_.get()));

    DeviceBuffer<unsigned long long> dUsed(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dUsed.data(), 0, sizeof(unsigned long long), stream_.get()));

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
        // tile ALL live ONLY inside this block so RAII frees them (the ~7 GB packed tile + the
        // ~free/kPcaZBudgetDivisor Z tile, together ~11 GB at 23k) BEFORE the eigensolver
        // allocates its workspace — the marginal alloc that OOMs the full cohort. Nothing after
        // the block references them (symmetrize + the truncated solve read only dC).
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

    // Truncated top-K eigensolve (randomized subspace iteration): only the K PCs are needed,
    // so we skip the full N-spectrum Dsyevd (~2N^2*8B ≈ 8.5 GB workspace at 23k — the OOM
    // wall) and Rayleigh-Ritz project onto an L = K+p subspace (default L=256; sketch/basis
    // buffers ~190 MB at 23k, the tiny L x L Dsyevd ~1 MB). The matmul-heavy sketches run
    // emulated-FP64; the QR + the tiny L x L B-eigen run
    // native-FP64 (the eigen carve-out, now on B not the 23k GRM). dC is READ-ONLY here so
    // trace(dC) below stays valid.
    solver_.set_stream(stream_.get());
    const int oversample = pca_oversample();
    const int subspace_iters = pca_subspace_iters();
    const int L = static_cast<int>(std::min<long>(static_cast<long>(K) + oversample, N));
    DeviceBuffer<double> dEvecL(static_cast<std::size_t>(N) * static_cast<std::size_t>(L));
    DeviceBuffer<double> dEvalL(static_cast<std::size_t>(L));
    int L_used = 0;
    const PcaTruncatedEig te = pca_truncated_topk(
        blas_.get(), solver_.get(), stream_.get(), dC.data(), Nn, K, oversample,
        subspace_iters, precision, dEvecL.data(), dEvalL.data(), &L_used);
    if (te.status != Status::Ok) {
        out.status = te.status;
        return out;
    }

    // var_explained denominator = trace(dC) = Σ_all λ (native-FP64 diagonal reduction) — the
    // truncated solve no longer returns the full N-vector we used to sum, but the trace is
    // exactly Σ of ALL eigenvalues, so the ratio is unchanged in value.
    DeviceBuffer<double> dTrace(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dTrace.data(), 0, sizeof(double), stream_.get()));
    launch_pca_trace(dC.data(), Nn, dTrace.data(), stream_.get());

    // Project the top-K eigenvectors to sample PC coordinates (coord = evec*sqrt(eval)); the
    // eigenvector block is N x L (ncol = te.L), ascending Ritz value (largest at the tail).
    DeviceBuffer<double> dCoords(static_cast<std::size_t>(N) * static_cast<std::size_t>(K));
    launch_pca_coords(dEvecL.data(), dEvalL.data(), Nn, te.L, K, dCoords.data(), stream_.get());

    // D2H only the small results: coords (N*K), the L Ritz values, the trace scalar, used cnt.
    out.coords.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(K), 0.0);
    std::vector<double> eval_L(static_cast<std::size_t>(te.L), 0.0);
    double trace_h = 0.0;
    unsigned long long used_h = 0;
    d2h_async(out.coords.data(), dCoords,
              static_cast<std::size_t>(N) * static_cast<std::size_t>(K), stream_.get());
    d2h_async(eval_L.data(), dEvalL, static_cast<std::size_t>(te.L), stream_.get());
    d2h_async(&trace_h, dTrace, 1, stream_.get());
    d2h_async(&used_h, dUsed, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    const long double total = static_cast<long double>(trace_h);
    out.eigenvalues.assign(static_cast<std::size_t>(K), 0.0);
    out.var_explained.assign(static_cast<std::size_t>(K), 0.0);
    for (int kk = 0; kk < K; ++kk) {
        // Ritz values ascending -> descending PCs read from the tail.
        const double lambda = eval_L[static_cast<std::size_t>(te.L - 1 - kk)];
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

}  // namespace steppe::device
