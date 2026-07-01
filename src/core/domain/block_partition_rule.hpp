// src/core/domain/block_partition_rule.hpp
//
// THE SNP→block assignment rule — the single source of truth for which jackknife
// block a SNP belongs to (architecture.md §2, §5, §8, §12; ROADMAP §3 M3, §4).
//
// This is the one shared DOMAIN rule that both the `io` front-end and the device
// kernels consume bit-for-bit (architecture.md §4): it is host-pure and
// CUDA-free, so it does not break the layering. It lives in `core`, NOT in `io`
// — re-deriving the block rule in `io` is an explicitly-named smell
// (architecture.md §2). The block id must be identical across the single-dataset
// and merged-dataset paths and across the CPU/GPU backends, which is exactly why
// there is one function here and no second copy anywhere.
//
// Units (verified against ADMIXTOOLS 2 semantics, architecture.md §9): AT2's
// `blgsize` default is 0.05 Morgans = 5 cM. The block math is done in MORGANS to
// match upstream; the cM-facing config accessor (RunConfig::block_size_cm,
// default kDefaultBlockSizeCm = 5.0) converts at exactly one place. Do not
// conflate cM and Morgans.
//
// M3 status: the floor-of-position-over-block-size core (`block_of`) is the
// stable __host__-shareable per-SNP primitive from M0; it survives as a valid
// primitive but `assign_blocks` no longer calls it (the AT2 walk does not bin to
// a fixed grid). M3 adds the whole-ordering assignment (`assign_blocks`) — the
// AT2 SNP-anchored cumulative walk (`setblocks()`; see assign_blocks below) that
// turns the per-SNP (chrom, genpos) into the dense, per-chromosome-reset,
// contiguous global block ids the jackknife consumes — plus the ONE cM→Morgan
// conversion site (`block_size_cm_to_morgans`). All host-pure, CUDA-free; the
// assignment loop lives in block_partition_rule.cpp.
#ifndef STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
#define STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "steppe/config.hpp"  // kCentimorgansPerMorgan (the conversion constant)

namespace steppe::core {

/// Index-cast helper: the per-SNP `long` column index → the `std::size_t` a
/// std::vector / std::span subscript wants ([7.3] dedup). The
/// `block_id[static_cast<std::size_t>(...)]` cast boilerplate was hand-repeated in
/// the per-SNP loops of BOTH assign_blocks and block_ranges; this writes the widening
/// ONCE. Callers guarantee a non-negative index (file-order SNP columns in [0, M)),
/// so the widening is value-preserving (parity-NEUTRAL).
[[nodiscard]] constexpr std::size_t idx(long i) noexcept {
    return static_cast<std::size_t>(i);
}

/// Block index of a SNP from its genetic position, both in MORGANS.
///
/// The deterministic pure function of genetic position that defines jackknife
/// block membership (architecture.md §12 "Seed control"): a SNP at genetic
/// position `genpos_morgans` falls in block `floor(genpos_morgans /
/// block_size_morgans)`. Same arithmetic on host and device; no SNP-block
/// assignment is ever re-derived elsewhere.
///
/// PRECONDITION: `block_size_morgans > 0` (it is the divisor). This per-SNP
/// primitive does NOT validate it — the guard lives in `assign_blocks` (which
/// rejects 0 / negative / NaN before ever calling this; cleanup X-3/B13), and
/// will also live in `ConfigBuilder::build()` if/when it lands. Calling this
/// directly with a non-positive width is UB at the `static_cast<int>` of the
/// resulting ±Inf / NaN ([conv.fpint]).
///
/// @param genpos_morgans     genetic position in Morgans (>= 0 within a
///                           chromosome; callers supply per-chromosome positions
///                           and handle chromosome boundaries — M3).
/// @param block_size_morgans block width in Morgans (AT2 `blgsize`, default 0.05
///                           = 5 cM via kDefaultBlockSizeCm). Must be > 0 (caller
///                           guaranteed; see precondition above).
/// @return                   zero-based block index (>= 0 for non-negative
///                           position).
///
/// Note on units: callers convert the cM-facing config to Morgans
/// (kCentimorgansPerMorgan) before calling — this function speaks Morgans only,
/// matching the upstream block math.
[[nodiscard]] inline int block_of(double genpos_morgans, double block_size_morgans) noexcept {
    return static_cast<int>(std::floor(genpos_morgans / block_size_morgans));
}

/// THE single cM↔Morgan conversion site (architecture.md §9 "Block-size unit";
/// ROADMAP §4). The config surface speaks centimorgans (RunConfig::block_size_cm,
/// default kDefaultBlockSizeCm = 5.0); the block math is done in Morgans to match
/// ADMIXTOOLS 2 (`blgsize` default 0.05 Morgans = 5 cM). This is the ONLY place
/// the factor is applied — no bare `* 0.01` / `/ 100` anywhere else in the tree
/// (the conversion constant kCentimorgansPerMorgan is its single home).
///
/// `block_size_cm_to_morgans(kDefaultBlockSizeCm) == 0.05` exactly (5.0 / 100.0).
///
/// @param cm  block width in centimorgans (must be > 0 for a meaningful rule).
/// @return    the same width in Morgans, ready for `block_of` / `assign_blocks`.
[[nodiscard]] constexpr double block_size_cm_to_morgans(double cm) noexcept {
    return cm / kCentimorgansPerMorgan;
}

/// Result of the whole-ordering block assignment: a dense per-SNP block id and
/// the resulting block count (architecture.md §5, §12; ROADMAP §3 M3).
///
/// `block_id` is parallel to the input SNPs (in file order); `block_id[s]` is the
/// global jackknife block of SNP `s`. The ids are DENSE: every value in
/// `0 .. n_block-1` is used at least once, and the sequence is monotonically
/// non-decreasing in file order. `n_block` is the number of distinct blocks.
struct BlockPartition {
    /// Per-SNP global block id, parallel to the input arrays (file order). Stays
    /// `int`: block counts are O(1e3) even on a whole genome (architecture.md §5).
    std::vector<int> block_id;

    /// Number of distinct blocks == max(block_id) + 1 (0 when there are no SNPs).
    int n_block = 0;
};

/// Assign every SNP to a dense, contiguous jackknife block in a single
/// deterministic pass over the SNPs IN FILE ORDER (architecture.md §5, §8, §12;
/// ROADMAP §3 M3). This is the host-pure whole-ordering rule that BOTH the `io`
/// front-end and the device kernels consume bit-for-bit — re-deriving it anywhere
/// else is the named smell (architecture.md §2, §8). It lives in `core`, never in
/// `io`.
///
/// The rule (the AT2 `setblocks()` convention — ADMIXTOOLS DReichLab
/// `qpsubs.c:1698-1759`, admixtools R 2.0.10 `get_block_lengths`; the bit-tight
/// parity TARGET, docs/research/block-partition-at2.md): a SNP-ANCHORED
/// cumulative walk. Carry the genetic position of the current block's FIRST SNP
/// (the anchor). A NEW global block opens when EITHER the chromosome changes OR
/// the cumulative distance from the anchor reaches `block_size_morgans` (`>=`,
/// inclusive — porting C `dis >= blocklen`); on a cut the anchor RE-SETS to the
/// SNP that opens the block, so the sub-width remainder rolls FORWARD. Blocks are
/// anchored at actual SNP positions, NEVER at fixed grid multiples
/// `k*block_size_morgans` — that re-anchoring is the difference from a floor-bin
/// grid. Consequences, each a property the M3 tests pin:
///   * per-chromosome reset — a chromosome boundary ALWAYS forces a fresh block,
///     so blocks never straddle chromosomes (the cut test includes `c != prev`);
///   * a block spans `>= block_size_morgans` (it can be WIDER, since the SNP that
///     trips the threshold may overshoot it) EXCEPT the trailing/short-chrom
///     remnant, which is kept as-is — matching AT2's tiny chr-end blocks;
///   * the count is the genome's SNP-anchored walk count, NOT the occupied-grid
///     count: a wide SNP-sparse stretch is ONE block, not one block per empty
///     grid cell.
///
/// Inputs are parallel arrays of the SAME length M (one entry per SNP, file
/// order). M can be ~584k (whole-genome AADR), so the loop index is `long` to
/// match views.hpp `MatView::M`; `block_id` stays `std::vector<int>` (the counts
/// are small). No allocation beyond the result vector.
///
/// THE bp BLOCK-FALLBACK (AT2 parity, the all-zero-genetic-map case). A dataset
/// with NO genetic linkage map ships an ALL-ZERO genetic-position column (common
/// in VCF/PLINK-derived modern data: .snp col3 / .bim col3 = 0 everywhere). Then
/// `(genpos - anchor)` is always 0, the only cuts are chromosome boundaries, and
/// the walk collapses to ONE block per chromosome — the block-jackknife SE breaks
/// (a single-chrom subset → 1 block → NA SE / non-SPD covariance). ADMIXTOOLS 2
/// handles this in `get_block_lengths`: it prints "No genetic linkage map found!
/// Defining blocks by base pair distance of 2e+06" and partitions by a 2 Mb
/// PHYSICAL-position window (admixtools 2.0.10 R/resampling.R; the 2e6 is
/// HARDCODED, independent of `blgsize`). `assign_blocks` reproduces this EXACTLY:
/// when `genpos_morgans` is all zero AND a NON-DEGENERATE physical axis `physpos`
/// (length >= M, not all zero) is supplied, it runs the IDENTICAL SNP-anchored
/// walk over `physpos` with window `bp_window` (default kBpFallbackWindow = 2e6),
/// and warns on stderr like AT2. The walk over raw bp with a 2e6 window gives the
/// same cuts as feeding bp·1e-8 pseudo-Morgans with a 0.02-Morgans window (1
/// cM/Mb), but stays in exact integer-valued bp (bp < 2^53), so the partition is
/// robust and matches AT2's ~357-block HGDP result. The fallback fires ONLY on an
/// all-zero map: a dataset WITH a real map (e.g. the AADR) takes the genetic-map
/// walk unchanged — bit-identical to the pre-fallback behavior (the pass gate).
///
/// @param chrom           per-SNP chromosome code (any integer scheme; only
///                        equality between adjacent SNPs matters). Length M.
/// @param genpos_morgans  per-SNP genetic position in Morgans. Length M (== chrom
///                        length; mismatched lengths are a programming error).
/// @param block_size_morgans  block width in Morgans (use
///                        `block_size_cm_to_morgans(RunConfig::block_size_cm)`).
///                        Must be > 0; a non-positive or NaN width is rejected
///                        fail-fast (see the empty-partition note below).
/// @param physpos         per-SNP physical position in base pairs (from the .snp
///                        col4 / .bim col4 reader), parallel to `chrom`. Used ONLY
///                        for the bp fallback when `genpos_morgans` is all zero;
///                        ignored entirely on a real genetic map. Empty (the
///                        default) disables the fallback (the walk stays on
///                        `genpos_morgans`, i.e. 1 block/chrom on an all-zero map).
/// @param bp_window       the bp fallback window (default kBpFallbackWindow = 2e6,
///                        the AT2 hardcoded 2 Mb). Only consulted in the fallback.
/// @return  a BlockPartition with `block_id` of length M (dense 0..n_block-1,
///          non-decreasing) and `n_block` set. Empty input → empty `block_id`,
///          `n_block == 0`. An ILLEGAL `block_size_morgans` (0, negative, or NaN)
///          likewise yields the empty partition (`n_block == 0`) rather than the
///          float→int UB / silently-inverted bins it would otherwise produce
///          (architecture.md §2 fail-fast; cleanup X-3/B13) — this is the only
///          enforceable site today, as `ConfigBuilder::build()` does not exist.
[[nodiscard]] BlockPartition assign_blocks(std::span<const int> chrom,
                                           std::span<const double> genpos_morgans,
                                           double block_size_morgans,
                                           std::span<const double> physpos = {},
                                           double bp_window = kBpFallbackWindow);

/// The half-open SNP column range `[begin, end)` of one jackknife block in the
/// per-SNP arrays (file order). `begin` is the block's first column (the CUDA
/// path's `block_offsets[b]`); `size() == end - begin` is the block's SNP count
/// (the S4 jackknife weighting denominator base, also `F2BlockTensor::block_sizes`).
/// Widths are `long` to match `MatView::M` / the per-SNP column index.
struct BlockRange {
    long begin = 0;  ///< first SNP column of the block (inclusive).
    long end = 0;    ///< one past the block's last SNP column (exclusive).

    /// SNP count of the block (`end - begin`).
    [[nodiscard]] long size() const noexcept { return end - begin; }
};

/// THE single-source inverse of `assign_blocks`: turn the dense, non-decreasing
/// per-SNP `block_id[]` into the per-block half-open column ranges `[begin, end)`,
/// validating the partition contract ONCE (architecture.md §2 fail-fast, §5, §8;
/// ROADMAP M4; cleanup X-3/B3).
///
/// Both backends (`device/cuda/cuda_backend.cu`, `device/cpu/cpu_backend.cpp`)
/// derive their per-block layout from `block_id` — the CUDA path as
/// `block_offsets`/`block_sizes` it copies to the device and the kernel
/// dereferences (`f2_batched_kernel.cu`), the CPU oracle as `begin`/`end`. That
/// scan was hand-duplicated in both backends (and a third near-copy in the M4
/// equivalence test) and NONE validated the partition, so a malformed `block_id`
/// (an id `< 0` or `>= n_block`, or `block_id.size() < M`) was a silent
/// out-of-bounds write on the host range vectors and an out-of-bounds device read
/// of `block_offsets[id]`. This is the one home for that inverse; both backends
/// call it and the validation lives here, exactly once.
///
/// CONTRACT (the `assign_blocks` postcondition every consumer relies on, made
/// fail-fast here): `block_id` must have at least `M` entries, every
/// `block_id[s]` (for `s ∈ [0, M)`) must satisfy `0 <= id < n_block`, and the
/// sequence must be non-decreasing (so each block's SNPs form one contiguous
/// half-open range). A violation is a programming/data error — it throws
/// `std::runtime_error` rather than corrupting memory (the OOB B3 closes). The
/// result is dense in `[0, n_block)`: a block with no SNPs gets an empty
/// `[begin, end)` (`begin == end`); `assign_blocks` never produces one, but the
/// validation does not forbid it (it only forbids the unsafe ids above).
///
/// @param block_id  per-SNP global block ids in file order (length `>= M`); the
///                  `BlockPartition::block_id` from `assign_blocks`.
/// @param M         number of SNP columns to scan (the `MatView::M` the backend
///                  trusts — the count travels separately from `block_id`'s
///                  length, so this also pins `block_id.size() >= M`).
/// @param n_block   number of distinct blocks (`BlockPartition::n_block`); the
///                  length of the returned vector.
/// @return  `n_block` ranges, indexed by block id; `out[b]` is block `b`'s
///          half-open `[begin, end)`. Empty input (`M <= 0` or `n_block <= 0`) →
///          empty vector.
/// @throws std::runtime_error if the partition violates the contract above.
///
/// HEADER-INLINE (unlike the out-of-line `assign_blocks`): both backends — which
/// compile into `steppe_device`, NOT `steppe_core` — call this, and `device`
/// cannot link `steppe_core` (that would cycle: `steppe_core` links
/// `steppe::device` PRIVATE). Like `block_of`, this is host-pure CUDA-free code
/// reachable from BOTH layers only through the header-only `steppe::core_internal`
/// INTERFACE target, so its body lives here. It is one O(M) scan, called once per
/// `compute_f2_blocks` (the inlining cost is negligible vs the f2 GEMMs).
[[nodiscard]] inline std::vector<BlockRange> block_ranges(std::span<const int> block_id,
                                                          long M, int n_block) {
    // Empty input → empty layout (the backends early-out on this too). Guard both
    // axes: M is the SNP count the backend trusts (MatView::M), n_block the
    // distinct-block count; either non-positive means there is nothing to range.
    if (M <= 0 || n_block <= 0) {
        return {};
    }

    // The three fail-fast paths below share the "core::block_ranges: " prefix; this
    // local throw lambda writes the prefix ONCE ([7.4] dedup). The distinct per-check
    // message text is still supplied at each call site (the checks remain genuinely
    // different); only the common prefix is single-homed.
    const auto fail = [](const std::string& msg) {
        throw std::runtime_error("core::block_ranges: " + msg);
    };

    // FAIL-FAST contract guard (architecture.md §2): the scan below indexes
    // out[block_id[s]] for s in [0, M), so a short block_id or an id outside
    // [0, n_block) would be an out-of-bounds read/write. assign_blocks guarantees
    // these, but a hand-built or recomputed partition (the M4 test, a future
    // merged-dataset path) might not — surface it here, ONCE, with context,
    // rather than silently corrupting memory in the backend (cleanup X-3/B3).
    // This is the single home: both backends and the M4 test call this; the OOB
    // is closed in every build config, not just debug.
    if (block_id.size() < static_cast<std::size_t>(M)) {
        fail("block_id has " + std::to_string(block_id.size()) +
             " entries but M = " + std::to_string(M) +
             " columns are required (partition shorter than the SNP count)");
    }

    std::vector<BlockRange> ranges(static_cast<std::size_t>(n_block));

    // Single forward scan over the contiguous runs. block_id is non-decreasing
    // (assign_blocks), so each block id's SNPs are exactly one half-open run
    // [s, e); we validate both the range bound and monotonicity as we go.
    long s = 0;
    int prev_b = -1;  // last block id seen; ids must be non-decreasing.
    while (s < M) {
        const int b = block_id[idx(s)];
        if (b < 0 || b >= n_block) {
            fail("block_id[" + std::to_string(s) + "] = " +
                 std::to_string(b) + " is out of range [0, " + std::to_string(n_block) + ")");
        }
        if (b < prev_b) {
            fail("block_id is not non-decreasing at column " +
                 std::to_string(s) + " (" + std::to_string(b) + " < " +
                 std::to_string(prev_b) + "); the partition is not contiguous");
        }

        long e = s;
        while (e < M && block_id[idx(e)] == b) ++e;
        ranges[static_cast<std::size_t>(b)] = BlockRange{s, e};
        prev_b = b;
        s = e;
    }

    return ranges;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
