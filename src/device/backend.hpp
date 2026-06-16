// src/device/backend.hpp
//
// ComputeBackend — the dependency-injection seam between `core` orchestration
// and the device layer (architecture.md §4, §8; ROADMAP §2, M0).
//
// THIS HEADER IS CUDA-FREE BY CONTRACT. It is the only door `core` uses to reach
// the GPU: `core` is pure host C++20 and never includes a CUDA header or calls
// cuBLAS/cuSOLVER directly (architecture.md §2, §4). CUDA is PRIVATE to
// steppe_device, so this interface compiles into `core` and the CLI without
// dragging in the device toolkit. Two implementations satisfy it — `CudaBackend`
// (the 3-GEMM reformulation on the GPU) and `CpuBackend` (the scalar reference
// oracle) — and the compute layer is written once against the interface, never
// branching on GPU-vs-CPU (architecture.md §8). The CPU backend is both the DRY
// reference and the correctness anchor the GPU is continuously diffed against
// (architecture.md §13; ROADMAP §5).
//
// M0 scope: the minimal-but-real method is `compute_f2` — compute the f2 matrix
// from the Q/V/N contract at a given Precision, returning f2 [P × P] and the
// pairwise-valid-SNP count Vpair [P × P]. Vpair is RETAINED, not discarded: it
// is the weighted-block-jackknife weight at S4 (architecture.md §5 S2 caveat
// (a)). Later milestones add decode / gemm / jackknife / svd methods here.
#ifndef STEPPE_DEVICE_BACKEND_HPP
#define STEPPE_DEVICE_BACKEND_HPP

#include <vector>

#include "steppe/config.hpp"        // steppe::Precision
#include "core/internal/views.hpp"  // steppe::core::MatView (Q/V/N contract)

namespace steppe {

/// Output of an f2 computation: the symmetric f2 matrix and the pairwise-valid
/// SNP counts, both column-major [P × P] (element (i,j) at `i + P·j`). Plain
/// host vectors so the type crosses the CUDA-free seam (architecture.md §4);
/// the CUDA backend copies its device results into these before returning.
struct F2Result {
    /// f2 matrix, column-major [P × P]. Bias-corrected, AT2-unbiased estimator
    /// f2(i,j) = mean over jointly-valid SNPs of (p_i − p_j)² − hc_i − hc_j.
    std::vector<double> f2;

    /// Pairwise-valid SNP count, column-major [P × P]: Vpair(i,j) = number of
    /// SNPs valid in BOTH i and j. RETAINED as the S4 jackknife weight
    /// (architecture.md §5 S2 caveat (a)); the per-pair divide and S4 weighting
    /// must compose to AT2's f2_blocks definition, not double-normalize.
    std::vector<double> vpair;

    /// Number of populations P (the leading dimension of both matrices).
    int P = 0;
};

/// Abstract compute backend. One interface, two implementations (CUDA, CPU
/// reference). All device operations route through here; `core` never issues a
/// GEMM/SVD/Cholesky itself (architecture.md §2, §8). Move-only ownership of
/// concrete backends is by `std::unique_ptr<ComputeBackend>` in `Resources`
/// (architecture.md §9).
class ComputeBackend {
public:
    ComputeBackend() = default;
    ComputeBackend(const ComputeBackend&) = delete;
    ComputeBackend& operator=(const ComputeBackend&) = delete;
    ComputeBackend(ComputeBackend&&) = delete;
    ComputeBackend& operator=(ComputeBackend&&) = delete;
    virtual ~ComputeBackend() = default;

    /// Compute the bias-corrected f2 matrix and pairwise-valid counts from the
    /// Q/V/N contract (column-major [P × M] views, views.hpp) at the requested
    /// precision.
    ///
    /// @param Q  reference-allele frequencies in [0,1], zero-filled where
    ///           invalid (the zero is what makes the masked GEMM correct).
    /// @param V  validity mask (1.0 valid / 0.0 missing).
    /// @param N  non-missing haploid count (2 × diploids, or 1 × pseudo-haploids
    ///           for ancient DNA). Enters only the het correction.
    /// @param precision  governs the matmul-heavy f2 GEMMs ONLY (default
    ///           EmulatedFp64{40} ⇒ ≈ native, MEASURED 7–17× faster on real
    ///           AADR; ROADMAP §0). The small numerator/divide stays native
    ///           FP64 regardless (architecture.md §12). The CPU backend computes
    ///           the native-FP64 reference and ignores the matmul mode.
    ///
    /// Preconditions: Q, V, N share the same P and M and refer to the same SNP
    /// block. Returns f2 [P × P] + Vpair [P × P] (column-major). Both backends
    /// must agree at the f2/Vpair seam within the tight tolerance tier
    /// (architecture.md §12, §13) — the formula is shared via
    /// core/internal/f2_estimator.hpp so they cannot diverge.
    [[nodiscard]] virtual F2Result compute_f2(const core::MatView& Q,
                                              const core::MatView& V,
                                              const core::MatView& N,
                                              const Precision& precision) = 0;
};

}  // namespace steppe

#endif  // STEPPE_DEVICE_BACKEND_HPP
