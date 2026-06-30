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
// resident tensors are always FP64. The M4 device path keeps BOTH `f2` AND the
// retained `vpair` co-resident, each `P² · n_block · 8` bytes, so the resident pair
// is `2 · P² · n_block · 8` bytes — both §11.2 budget terms (cleanup X-13/B26;
// counting only `f2` under-reserves the resident set by 2× and OOMs mid-stream).
// The device budget helper reserves for both (src/device/vram_budget.hpp).
#ifndef STEPPE_FSTATS_HPP
#define STEPPE_FSTATS_HPP

#include <cstddef>
#include <span>
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

    /// Unchecked f2 accessor for the LAYOUT-block convention `i + P·j + P·P·b`
    /// (see the struct-level LAYOUT note). `noexcept`, no bounds check — hot path.
    /// The index math runs in `std::size_t` (NOT `int`): at production scale
    /// (P≈768, n_block≈800) the flat index exceeds INT_MAX, so a plain-`int`
    /// computation would overflow (the design-for-scale rule; same cast pattern as
    /// `size()`). Const read overload.
    [[nodiscard]] double f2_at(int i, int j, int b) const noexcept {
        return f2[flat_index(i, j, b)];
    }
    /// Mutable overload of `f2_at` (lvalue into `f2`); identical `i + P·j + P·P·b`
    /// index.
    double& f2_at(int i, int j, int b) noexcept { return f2[flat_index(i, j, b)]; }

    /// Vpair counterpart of `f2_at` — identical `i + P·j + P·P·b` index into
    /// `vpair` (see the LAYOUT note). Const read overload.
    [[nodiscard]] double vpair_at(int i, int j, int b) const noexcept {
        return vpair[flat_index(i, j, b)];
    }
    /// Mutable overload of `vpair_at` (lvalue into `vpair`).
    double& vpair_at(int i, int j, int b) noexcept {
        return vpair[flat_index(i, j, b)];
    }

    /// The contiguous `[P × P]` f2 slab for block `b` as a span (column-major
    /// `i + P·j` within the slab; see the LAYOUT note). Length `P · P`; the byte
    /// offset is computed in `std::size_t` for the same scale reason as `f2_at`.
    [[nodiscard]] std::span<const double> block(int b) const noexcept {
        const std::size_t slab =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
        return std::span<const double>{
            f2.data() + slab * static_cast<std::size_t>(b), slab};
    }

  private:
    /// The SINGLE canonical home for the `i + P·j + P·P·b` flat-index convention
    /// (the struct LAYOUT note): every accessor routes through here so the formula
    /// cannot drift from the writer (src/app/f2_dir_writer.cpp) or the kernel
    /// layout. `std::size_t` arithmetic, same cast pattern as `size()`.
    [[nodiscard]] std::size_t flat_index(int i, int j, int b) const noexcept {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(P) * static_cast<std::size_t>(j) +
               static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                   static_cast<std::size_t>(b);
    }
};

}  // namespace steppe

#endif  // STEPPE_FSTATS_HPP
