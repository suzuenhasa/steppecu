// src/core/fstats/f2_combine.cpp
//
// combine_f2_partials_host — the HOST-STAGED fixed-order combine (the PORTABLE
// PARITY BASELINE; architecture.md §11.4, §12; design §3). Host-pure, CUDA-FREE,
// in steppe::core. It allocates the full-shape zero tensor and PLACES each device's
// compact partial into it at the device's block offset, visiting devices in the
// FIXED g=0..G-1 order — the configuration-independent order that makes the result
// bit-identical across G and to the single-GPU reference (NEVER an NCCL AllReduce,
// §12). Because the block-aligned shards are DISJOINT (each global block is owned by
// exactly one device) the per-device contribution is a PLACEMENT, not an
// accumulation: each owned slab is written exactly once, so the placement is a
// contiguous std::copy_n (memcpy-grade) rather than a scalar read-add-store loop
// (cleanup B7 / f2_combine N2+P2).
#include "core/fstats/f2_combine.hpp"

#include <algorithm>  // std::copy_n (the contiguous per-device placement, B7)
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
    // budget terms); block_sizes is the per-block SNP count. The 0.0 (= +0.0) init
    // is the value of every slab; the disjoint shards then PLACE every owned slab
    // verbatim and leave the rest at +0.0. With block-aligned sharding the shards
    // tile [0, n_block_full) exactly (plan_block_shards; validate_f2_partials
    // re-checks `covered == n_block_full`), so NO slab is left unowned on the real
    // path — the +0.0 init survives only for a degenerate genuinely-unowned tail
    // that the disjoint contract proves cannot exist. Verbatim placement (not a
    // sum onto +0.0) is what makes the result bit-identical to the single-GPU run
    // slab-for-slab (design §0): the single-GPU reference computes each slab
    // DIRECTLY, never adding it onto a zero, so a faithful copy reproduces its exact
    // bits — INCLUDING a −0.0 element, which a `+= onto +0.0` would silently flip to
    // +0.0 (IEEE-754: (+0.0)+(−0.0)==+0.0 under round-to-nearest, a DIFFERENT bit
    // pattern — the old "x + 0.0 == x for all finite x" comment was wrong on x=−0.0;
    // cleanup B7 / f2_combine N2). block_sizes init 0 likewise; each owned block's
    // count is placed.
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

    // ---- FIXED-ORDER PLACEMENT g = 0 .. G-1 (THE parity law) -----------------
    // For each device in the FIXED g=0..G-1 order (architecture.md §12), place its
    // compact partial at its block offset shards[g].b0. The loop order over devices
    // IS the §12 fixed-order combine: it bit-matches the on-device P2P sibling,
    // which performs the same fixed-order placement-into onto a cudaMemset(0)
    // accumulator (design §3/§4).
    //
    // The block-aligned shards are DISJOINT (validate_f2_partials enforced the
    // tiling: each global slab is owned by exactly one device), so a device's
    // contribution is a PLACEMENT — its owned region is written exactly once, with
    // no other device touching it and no read-modify-write against the +0.0 init.
    // For device g's compact partial, local block lb maps to global block b = b0+lb,
    // so out_base = slab·(b0+lb) and in_base = slab·lb: as lb runs 0..n_block-1 the
    // SOURCE spans [0, slab·n_block) contiguously and the DESTINATION spans
    // [slab·b0, slab·(b0+n_block)) contiguously. The whole per-device placement of
    // f2 (and of vpair) is therefore ONE contiguous std::copy_n of slab·n_block
    // doubles, and block_sizes is one std::copy_n of n_block ints at offset b0.
    //
    // std::copy_n of distinct, non-overlapping ranges copies element-by-element in
    // increasing index order and reproduces the source bits EXACTLY — it lowers to
    // memcpy/memmove for trivially-copyable element types (C++23 [alg.copy]; the
    // libstdc++ __builtin_memmove fast path). It is therefore (a) memcpy-grade (a
    // single streaming store, no load-of-zero + add) and (b) STRICTLY more faithful
    // to the single-GPU reference than the prior scalar `+=` onto +0.0: a copy
    // reproduces a −0.0 partial element byte-for-byte, where the `+=` flipped it to
    // +0.0 (cleanup B7 / f2_combine N2). FP64 storage is mandated in every mode
    // (fstats.hpp); precision policy never reaches the combine, so the placement is
    // exact regardless of it.
    for (std::size_t g = 0; g < partials.size(); ++g) {
        const F2BlockTensor& part = partials[g];
        if (part.n_block <= 0) continue;  // empty shard owns nothing (b0 == b1)
        const std::size_t b0 = static_cast<std::size_t>(shards[g].b0);
        const std::size_t part_elems =
            slab * static_cast<std::size_t>(part.n_block);  // f2/vpair run length
        // f2 + vpair: one contiguous copy of the device's owned slabs into place.
        std::copy_n(part.f2.data(),    part_elems, out.f2.data()    + slab * b0);
        std::copy_n(part.vpair.data(), part_elems, out.vpair.data() + slab * b0);
        // block_sizes: the backend computed each block's SNP count from its local
        // ranges (== the global block's count, design §2); copy them into place
        // (single-homed, no host recompute). Disjoint shards ⇒ a plain placement.
        std::copy_n(part.block_sizes.data(),
                    static_cast<std::size_t>(part.n_block),
                    out.block_sizes.data() + b0);
    }

    return out;
}

}  // namespace steppe::core
