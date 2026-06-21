// src/core/fstats/f2_combine.hpp
//
// The HOST-STAGED fixed-order f2 combine — the PORTABLE PARITY BASELINE of the
// single-node multi-GPU (SPMG) precompute (architecture.md §11.4 "the host-staged
// fixed-order combine remains the portable parity baseline", §12 parity; design
// §3). It PLACES the G per-device COMPACT partials into one full-shape
// [P × P × n_block] F2BlockTensor in the FIXED g=0..G-1 device order
// (DeviceConfig::devices order, pinned in Resources). Placing in a
// configuration-INDEPENDENT fixed order is exactly what makes the result
// BIT-IDENTICAL across G and to the single-GPU reference (architecture.md §12 —
// NEVER an NCCL AllReduce, whose order varies with G).
//
// CUDA-FREE, host-pure, in `steppe::core`: it names only the public CUDA-free
// `F2BlockTensor` and the CUDA-free `DeviceShard` plan, so it is unit-testable
// host-only and compiles into steppe_core without the device toolkit (design §3,
// §8). It is the ONLY combine path on the budget box (no peer access); the
// device-resident cudaMemcpyPeer combine (device/p2p_combine.hpp, a separate unit)
// is the opt-in fast-path and is BIT-IDENTICAL to this one by construction (both
// place the same fixed-order shards onto a zero-initialized full tensor; architecture
// .md §11.4, §12).
#ifndef STEPPE_CORE_FSTATS_F2_COMBINE_HPP
#define STEPPE_CORE_FSTATS_F2_COMBINE_HPP

#include <span>

#include "steppe/fstats.hpp"      // steppe::F2BlockTensor (public, CUDA-free)
#include "device/shard_plan.hpp"  // steppe::device::DeviceShard (CUDA-free plan)

namespace steppe::core {

/// Combine the G per-device COMPACT partials into one full-shape F2BlockTensor by
/// PLACING each device g's partial at its block offset (`shards[g].b0`), visiting
/// the devices in the FIXED g=0..G-1 order (architecture.md §11.4, §12; design §3).
///
/// With block-aligned sharding the per-device block ranges are DISJOINT, so this is
/// a PLACEMENT — each global slab is written by exactly one device, exactly once.
/// Each device's owned slabs are a contiguous run, so the placement is one
/// `std::copy_n` of `f2`, one of `vpair`, and one of `block_sizes` per device
/// (memcpy-grade). Visiting the devices in g=0..G-1 order is literally the §12 "sum
/// the per-device partials host-side in fixed device order"; the device-resident
/// P2P combine performs the SAME fixed-order placement on-device (onto a
/// cudaMemset(0) accumulator), so the two combine tiers are bit-identical to each
/// other (design §4).
///
/// The placement is EXACT and bit-identical to the single-GPU reference: that
/// reference computes each slab DIRECTLY (it never adds the slab onto a zero), so a
/// faithful copy reproduces its exact bits — INCLUDING a −0.0 element. (A `+= onto
/// +0.0` would flip −0.0 to +0.0 — IEEE-754: `(+0.0)+(−0.0) == +0.0` under
/// round-to-nearest, a different bit pattern — so `x + 0.0 == x` holds for finite
/// x ≠ ±0.0 but NOT for x = −0.0; the `std::copy_n` placement avoids that flip
/// unconditionally. cleanup B7 / f2_combine N2.) Non-owned slabs remain the +0.0
/// init, but the disjoint tiling proves none exist on the real path.
///
/// f2, vpair, AND block_sizes are all placed: block_sizes is copied from each
/// partial at offset b0 (the backend already computed each block's SNP count from
/// its local ranges — == the global block's count, design §2 — so no host recompute).
///
/// PRECONDITIONS (fail-fast, architecture.md §2): `partials.size() == shards.size()`
/// (== G); every NON-EMPTY partial shares `P`; the union of the shard block ranges
/// tiles `[0, n_block_full)` contiguously; each partial g spans exactly its shard's
/// blocks (`partials[g].n_block == shards[g].b1 - shards[g].b0`). A violation throws.
///
/// @param partials       G compact F2BlockTensors in g=0..G-1 order; `partials[g]`
///                       is device g's `[P × P × (shards[g].b1 - shards[g].b0)]`
///                       result (host storage, from each device's compute_f2_blocks).
///                       An EMPTY shard's partial has n_block == 0 (skipped).
/// @param shards         the block-aligned plan (plan_block_shards); `shards[g].b0`
///                       is partial g's placement offset.
/// @param P              population count (the leading dim of every slab). MUST be
///                       the P of every non-empty partial.
/// @param n_block_full   total block count of the combined tensor.
/// @return  the full `[P × P × n_block_full]` F2BlockTensor, BIT-IDENTICAL to the
///          single-GPU compute_f2_blocks over the full inputs (design §0).
/// @throws std::runtime_error on a precondition violation (size mismatch, P
///         disagreement, or a partial whose n_block != its shard's block span).
[[nodiscard]] F2BlockTensor combine_f2_partials_host(
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_COMBINE_HPP
