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
#include <span>
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
/// @param genpos_morgans     genetic position in Morgans (>= 0 within a
///                           chromosome; callers supply per-chromosome positions
///                           and handle chromosome boundaries — M3).
/// @param block_size_morgans block width in Morgans (AT2 `blgsize`, default 0.05
///                           = 5 cM via kDefaultBlockSizeCm). Must be > 0.
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
///                        Must be > 0.
/// @return  a BlockPartition with `block_id` of length M (dense 0..n_block-1,
///          non-decreasing) and `n_block` set. Empty input → empty `block_id`,
///          `n_block == 0`.
[[nodiscard]] BlockPartition assign_blocks(std::span<const int> chrom,
                                           std::span<const double> genpos_morgans,
                                           double block_size_morgans);

}  // namespace steppe::core

#endif  // STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
