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
// stable __host__-shareable per-SNP primitive from M0. M3 adds the whole-ordering
// assignment (`assign_blocks`) that turns the per-SNP local bins into the dense,
// per-chromosome-reset, contiguous global block ids the jackknife consumes, plus
// the ONE cM→Morgan conversion site (`block_size_cm_to_morgans`). All host-pure,
// CUDA-free; the assignment loop lives in block_partition_rule.cpp.
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
/// The rule: track the previous SNP's chromosome and its local floor-bin
/// (`block_of`). A NEW global block opens when EITHER the chromosome changes OR
/// the local bin changes from the previous SNP; otherwise the current global
/// block is reused. Consequences, each a property the M3 tests pin:
///   * per-chromosome reset — a chromosome boundary ALWAYS forces a fresh block,
///     so blocks never straddle chromosomes (the local bin is per-chromosome);
///   * interior empty bins are ABSORBED — a gap in the local bins (e.g. bins
///     0,1,3) yields dense global ids (0,1,2), not a reserved slot for the empty
///     bin (this is what makes the count the genome's *occupied*-block count);
///   * negative positions (`std::floor` → -1) get their own block and never alias
///     bin 0 — they form a distinct local bin like any other.
///
/// Inputs are parallel arrays of the SAME length M (one entry per SNP, file
/// order). M can be ~584k (whole-genome AADR), so the loop index is `long` to
/// match views.hpp `MatView::M`; `block_id` stays `std::vector<int>` (the counts
/// are small). No allocation beyond the result vector.
///
/// @param chrom           per-SNP chromosome code (any integer scheme; only
///                        equality between adjacent SNPs matters). Length M.
/// @param genpos_morgans  per-SNP genetic position in Morgans. Length M (== chrom
///                        length; mismatched lengths are a programming error).
/// @param block_size_morgans  block width in Morgans (use
///                        `block_size_cm_to_morgans(RunConfig::block_size_cm)`).
///                        Must be > 0; a non-positive or NaN width is rejected
///                        fail-fast (see the empty-partition note below).
/// @return  a BlockPartition with `block_id` of length M (dense 0..n_block-1,
///          non-decreasing) and `n_block` set. Empty input → empty `block_id`,
///          `n_block == 0`. An ILLEGAL `block_size_morgans` (0, negative, or NaN)
///          likewise yields the empty partition (`n_block == 0`) rather than the
///          float→int UB / silently-inverted bins it would otherwise produce
///          (architecture.md §2 fail-fast; cleanup X-3/B13) — this is the only
///          enforceable site today, as `ConfigBuilder::build()` does not exist.
[[nodiscard]] BlockPartition assign_blocks(std::span<const int> chrom,
                                           std::span<const double> genpos_morgans,
                                           double block_size_morgans);

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
/// dereferences (`f2_blocks_kernel.cu`), the CPU oracle as `begin`/`end`. That
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

    // FAIL-FAST contract guard (architecture.md §2): the scan below indexes
    // out[block_id[s]] for s in [0, M), so a short block_id or an id outside
    // [0, n_block) would be an out-of-bounds read/write. assign_blocks guarantees
    // these, but a hand-built or recomputed partition (the M4 test, a future
    // merged-dataset path) might not — surface it here, ONCE, with context,
    // rather than silently corrupting memory in the backend (cleanup X-3/B3).
    // This is the single home: both backends and the M4 test call this; the OOB
    // is closed in every build config, not just debug.
    if (block_id.size() < static_cast<std::size_t>(M)) {
        throw std::runtime_error(
            "core::block_ranges: block_id has " + std::to_string(block_id.size()) +
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
        const int b = block_id[static_cast<std::size_t>(s)];
        if (b < 0 || b >= n_block) {
            throw std::runtime_error(
                "core::block_ranges: block_id[" + std::to_string(s) + "] = " +
                std::to_string(b) + " is out of range [0, " + std::to_string(n_block) + ")");
        }
        if (b < prev_b) {
            throw std::runtime_error(
                "core::block_ranges: block_id is not non-decreasing at column " +
                std::to_string(s) + " (" + std::to_string(b) + " < " +
                std::to_string(prev_b) + "); the partition is not contiguous");
        }

        long e = s;
        while (e < M && block_id[static_cast<std::size_t>(e)] == b) ++e;
        ranges[static_cast<std::size_t>(b)] = BlockRange{s, e};
        prev_b = b;
        s = e;
    }

    return ranges;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
