// src/core/fstats/f2_combine.cpp
//
// combine_f2_partials_host — the HOST-STAGED fixed-order combine (the PORTABLE
// PARITY BASELINE; architecture.md §11.4, §12; design §3). Host-pure, CUDA-FREE,
// in steppe::core. It allocates the full-shape zero tensor and sums each device's
// compact partial into it at the device's block offset, in the FIXED g=0..G-1
// order — the configuration-independent order that makes the result bit-identical
// across G and to the single-GPU reference (NEVER an NCCL AllReduce, §12).
#include "core/fstats/f2_combine.hpp"

#include <cstddef>
#include <span>

#include "steppe/fstats.hpp"                     // steppe::F2BlockTensor
#include "device/shard_plan.hpp"                 // steppe::device::DeviceShard
#include "core/fstats/f2_partials_validate.hpp"  // shared validate_f2_partials (cleanup B5)

namespace steppe::core {

F2BlockTensor combine_f2_partials_host(
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full) {
    // Fail-fast precondition guard — the ONE validator SHARED with the device P2P
    // combine (cleanup B5; architecture.md §2, §8): the two tiers MUST reject
    // identically or their parity-neutrality (§11.4, §12) breaks. It runs ONCE up
    // front, O(G), off the bandwidth-critical path. validate_f2_partials throws on a
    // size/P/span/storage/tiling violation; after it, P >= 0 and n_block_full >= 0
    // are guaranteed.
    validate_f2_partials("steppe::core::combine_f2_partials_host",
                         partials, shards, P, n_block_full);

    // ---- Allocate the full-shape tensor, ZERO-initialized --------------------
    // f2 + vpair are [P × P × n_block_full] FP64 (the §11.2 resident pair, both
    // budget terms); block_sizes is the per-block SNP count. The 0.0 init is the
    // value of every NON-OWNED slab: summing exact zeros for those is exact
    // (x + 0.0 == x), so the combined tensor equals the single-GPU run slab-for-slab
    // (design §0). block_sizes init 0 likewise; each owned block's count is placed.
    F2BlockTensor out;
    out.P = P;
    out.n_block = n_block_full;
    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    // validate_f2_partials already rejected a negative n_block_full, so the cast is
    // safe with no clamp (cleanup B5 / C2 — the old `< 0 ? 0 :` ternary was dead).
    const std::size_t total = slab * static_cast<std::size_t>(n_block_full);
    out.f2.assign(total, 0.0);
    out.vpair.assign(total, 0.0);
    out.block_sizes.assign(static_cast<std::size_t>(n_block_full), 0);

    // ---- FIXED-ORDER SUM g = 0 .. G-1 (THE parity law) -----------------------
    // For each device in fixed order, place/sum its compact partial at its block
    // offset. The loop order is g=0..G-1 verbatim (architecture.md §12): even
    // though the disjoint shards make this a placement (each global slab written
    // once, onto a 0.0), the explicit += in fixed order keeps the algorithm
    // literally the §12 fixed-order combine and bit-matches the on-device P2P sum
    // (design §3/§4). Reference precision: f2/vpair are FP64 and the add is onto an
    // exact zero, so the placement is EXACT regardless of precision policy.
    for (std::size_t g = 0; g < partials.size(); ++g) {
        const F2BlockTensor& part = partials[g];
        const int b0 = shards[g].b0;
        for (int lb = 0; lb < part.n_block; ++lb) {
            const int b = b0 + lb;  // global block id of this device's local block lb
            const std::size_t out_base =
                slab * static_cast<std::size_t>(b);
            const std::size_t in_base =
                slab * static_cast<std::size_t>(lb);
            for (std::size_t e = 0; e < slab; ++e) {
                out.f2[out_base + e]    += part.f2[in_base + e];
                out.vpair[out_base + e] += part.vpair[in_base + e];
            }
            // block_sizes: the backend computed this block's SNP count from its
            // local ranges (== the global block's count, design §2); place it
            // (single-homed, no host recompute). Disjoint shards ⇒ a plain set.
            out.block_sizes[static_cast<std::size_t>(b)] =
                part.block_sizes[static_cast<std::size_t>(lb)];
        }
    }

    return out;
}

}  // namespace steppe::core
