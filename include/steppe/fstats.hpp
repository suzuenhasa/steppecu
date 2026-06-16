// include/steppe/fstats.hpp
//
// PUBLIC f-statistics API surface (architecture.md §4 "include/steppe/fstats.hpp
// (planned, M4) public F2BlockTensor handle"; ROADMAP §3 M4). This is the
// installed, CUDA-FREE header that carries the M4 deliverable: the per-block f2
// tensor `f2_blocks [P × P × n_block]` + the RETAINED per-block pairwise-valid
// count `Vpair` (the S4 weighted-block-jackknife weight — architecture.md §5 S2
// caveat (a)) + the block metadata.
//
// WHY THIS LIVES IN include/ (not src/core): it is the cacheable, ADMIXTOOLS-
// compatible interchange artifact (architecture.md §0, §1) — the seam between the
// GPU precompute engine (S0–S2) and the small-LA fit engine (S3–S8). It is plain
// host-accessible FP64 storage so it crosses the CUDA-free seam (architecture.md
// §4) and is consumable by `core`, the CLI, and the Python bindings without
// dragging in the device toolkit. No CUDA header here.
//
// STORAGE IS FP64 IN EVERY PRECISION MODE (architecture.md §11.2): the `Precision`
// knob is an OPERATION mode for the matmul-heavy GEMMs, never a storage type — the
// resident tensor is always FP64 (`P² · n_block · 8` bytes, the §11.2 budget term).
#ifndef STEPPE_FSTATS_HPP
#define STEPPE_FSTATS_HPP

#include <cstddef>
#include <vector>

namespace steppe {

/// The M4 deliverable: the per-block f2 tensor + the retained per-block Vpair +
/// block metadata, as plain host FP64 storage (architecture.md §0, §5 S2, §11.2).
///
/// LAYOUT. Both `f2` and `vpair` are a stack of `n_block` column-major [P × P]
/// matrices: the entry for (population i, population j, block b) lives at flat
/// index `i + P·j + P·P·b` (block-major outer, column-major within a block). This
/// is the [P × P × n_block] resident tensor of architecture.md §11.1/§11.2 — the
/// `n_block` axis is the jackknife batch axis the fit engine (S3–S8) contracts
/// over. Each [P × P] slab is SYMMETRIC. The DIAGONAL carries the full (i,i)
/// computation (= -2·mean within-pop het correction, generally nonzero), NOT a
/// forced 0 — the SAME diagonal convention as the single-block F2Result (pinned
/// in src/device/backend.hpp; cleanup X-2/B4), so the GPU grouped path, the CPU
/// per-block oracle, and the single-block == M0 F2Result all agree on the
/// diagonal. Downstream f3/f4 read off-diagonal f2 only, so the diagonal is never
/// consumed but is kept consistent across paths.
struct F2BlockTensor {
    /// Per-block bias-corrected f2: `f2[i + P·j + P·P·b]` is the AT2-unbiased f2
    /// for pair (i, j) over the SNPs of block b that are valid in BOTH i and j
    /// (the pairwise-complete path). FP64 in every precision mode. Length
    /// `P · P · n_block`.
    std::vector<double> f2;

    /// Per-block pairwise-valid SNP count: `vpair[i + P·j + P·P·b]` is the number
    /// of SNPs in block b valid in both i and j. RETAINED — it is the S4
    /// weighted-block-jackknife weight (architecture.md §5 S2 caveat (a)); the
    /// per-block divide here and the S4 weighting must COMPOSE to AT2's f2_blocks
    /// definition, not double-normalize. Integer-valued (carried as double).
    /// Length `P · P · n_block`.
    std::vector<double> vpair;

    /// Per-block SNP count: `block_sizes[b]` = number of SNPs assigned to block b
    /// (from the shared block_partition_rule). Σ over blocks == the total SNP
    /// count. Length `n_block`.
    std::vector<int> block_sizes;

    /// Number of populations P (the leading dimension of each [P × P] slab).
    int P = 0;

    /// Number of jackknife blocks (the outer/batch axis length).
    int n_block = 0;

    /// Flat element count `P · P · n_block` of `f2` / `vpair` (convenience).
    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
               static_cast<std::size_t>(n_block);
    }
};

}  // namespace steppe

#endif  // STEPPE_FSTATS_HPP
