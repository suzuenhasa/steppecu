// src/core/internal/views.hpp
//
// The Q/V/N contract as non-owning, column-major [P × M] views — the seam
// between the data front-end and the f2 kernel (ROADMAP §2; architecture.md §5).
//
// This is host-pure, CUDA-free DRY plumbing: a single tiny `MatView` struct that
// describes how the f2 inputs Q, V, N are laid out, so the CPU reference and the
// GPU feeder agree on indexing exactly. The full `span_view.hpp` (architecture.md
// §8) generalizes this over cuda::std::span/mdspan for the device layer; this is
// the minimal host-side anchor the M0 contract needs. No ownership, no CUDA.
//
// ---------------------------------------------------------------------------
// THE Q/V/N CONTRACT (stable regardless of decode method, ploidy, or precision)
// ---------------------------------------------------------------------------
// Three column-major [P × M] arrays (leading dim `P`; element (pop i, snp s) at
// flat index `i + P·s`), per SNP-block. P = number of populations, M = number
// of SNPs in the block. Matches the SHARED BINARY FORMAT written by
// build_tgeno_matrix.py (Q.f64 / V.f64 / N.f64) and consumed by the spike's
// --load path, so the contract is byte-compatible with the validated inputs.
//
//   Q  -- frequency of the FIXED reference allele in [0,1]. Zero-filled where
//         invalid; the zero is what makes the masked GEMM correct (Q²=0 and the
//         cross term vanish at invalid entries). (.snp col 5 is the ref allele;
//         value 0/1/2 = copies of the reference allele.)
//   V  -- validity mask: 1.0 if the population has a non-missing genotype at that
//         SNP, else 0.0.
//   N  -- NON-MISSING HAPLOID COUNT: 2 × non-missing diploids, OR 1 × non-missing
//         pseudo-haploids (the ancient-DNA case — MUST be honored; it changes how
//         N is computed, not this contract). It is alleles, not individuals — the
//         AT2 bias-correction convention. N enters only the het correction
//         hc = q(1-q)/max(N-1, 1) (see f2_estimator.hpp).
//
// Invariant the producer guarantees and consumers may assume: V != 0  ⟺  N > 0.
//
// Block membership comes from the shared `block_partition_rule` (cM-based,
// core/domain/block_partition_rule.hpp), NOT from these views.
#ifndef STEPPE_CORE_INTERNAL_VIEWS_HPP
#define STEPPE_CORE_INTERNAL_VIEWS_HPP

namespace steppe::core {

/// Non-owning, column-major [P × M] matrix view over double data.
///
/// Layout (cuBLAS column-major, leading dimension = `P`): the element for
/// (population `i`, SNP `s`) lives at `data[i + P·s]`. Used for Q, V, and N in
/// the Q/V/N contract above; the same struct describes any column-major
/// [rows × cols] double matrix where the view does not own its storage.
///
/// `M` is `long` so a [P × M] view over a large SNP block does not overflow a
/// 32-bit count; the flat index promotes to `long` before multiplying by `P`.
struct MatView {
    /// Non-owning pointer to column-major storage (leading dim = `P`). The view
    /// never frees this; lifetime is the caller's responsibility.
    const double* data = nullptr;

    /// Number of rows = number of populations P (also the leading dimension).
    int P = 0;

    /// Number of columns = number of SNPs M in this block.
    long M = 0;

    /// Element (population `i`, SNP `s`), i.e. `data[i + P·s]`. Column-major
    /// with leading dimension `P` — the one indexing rule shared by the CPU
    /// reference and the GPU feeder. No bounds check (hot path); callers respect
    /// 0 ≤ i < P and 0 ≤ s < M.
    [[nodiscard]] double element(int i, long s) const noexcept {
        // `P * s` already promotes to `long` (`s` is `long`), so only `i` needs the
        // explicit widening cast; the redundant `static_cast<long>(P)` is dropped
        // (DRY; NAMING-STYLE-STANDARD §2.5; findings group-7 7.3). Value unchanged.
        return data[static_cast<long>(i) + P * s];
    }
};

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_VIEWS_HPP
