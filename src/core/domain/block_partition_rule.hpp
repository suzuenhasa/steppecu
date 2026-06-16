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
// M0 status: this is the stub that pins the SIGNATURE and the rule shape. The
// full rule (per-chromosome reset, gap handling, contiguous block renumbering)
// arrives at M3; the floor-of-position-over-block-size core is stable and lives
// here so downstream code can compile and test against it now.
#ifndef STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
#define STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP

#include <cmath>

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

}  // namespace steppe::core

#endif  // STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
