// src/core/fstats/f2_partials_validate.hpp
//
// validate_f2_partials — the ONE fail-fast precondition guard SHARED by BOTH f2
// combine tiers (cleanup B5 / X7; architecture.md §2 fail-fast, §8 single-source).
//
// The host-staged combine (core/fstats/f2_combine.cpp) and the device-resident
// cudaMemcpyPeer combine (device/cuda/p2p_combine.cu) MUST reject malformed inputs
// IDENTICALLY: the two tiers are parity-NEUTRAL siblings (architecture.md §11.4,
// §12) and the §12 bit-identity story rests on them agreeing — a drift in which
// inputs each tier accepts would let one tier combine bytes the other refuses. The
// guard used to be DUPLICATED byte-for-byte between the two TUs (kept "in lock-step"
// only by a comment), which is exactly the §8 single-source violation this header
// closes: one home, both tiers call it, so they cannot drift.
//
// CUDA-FREE BY CONTRACT (like device/shard_plan.hpp / device/backend.hpp): it names
// only the public CUDA-free `steppe::F2BlockTensor` and the CUDA-free
// `steppe::device::DeviceShard` plan + std types, so it compiles into BOTH
// steppe_core (the host-staged tier, host-pure) AND a CUDA TU (the P2P tier) without
// dragging <cuda_runtime.h> into core. It is header-only INLINE: O(G) and off the
// bandwidth-critical combine path (architecture.md §11.4 — the combine is
// "essentially free"), so there is no out-of-line TU to link and no link-layer
// coupling between the tiers — each #includes the same CUDA-free header.
//
// The P2P tier additionally checks `device_ids.size() == G` (it threads a third
// parallel span the host tier has no notion of); that one extra check stays at the
// P2P call site (it names a transport detail core does not see), and this shared
// guard validates everything the two tiers have in common.
#ifndef STEPPE_CORE_FSTATS_F2_PARTIALS_VALIDATE_HPP
#define STEPPE_CORE_FSTATS_F2_PARTIALS_VALIDATE_HPP

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include "steppe/fstats.hpp"          // steppe::F2BlockTensor (public, CUDA-free)
#include "device/shard_plan.hpp"      // steppe::device::DeviceShard (CUDA-free plan)
#include "device/device_partial.hpp"  // steppe::device::DevicePartial (CUDA-free opaque resident handle)

namespace steppe::core {

/// Fail-fast precondition guard for BOTH f2 combine tiers (architecture.md §2): the
/// G partials must align with the G shards, agree on P, carry storage matching their
/// declared extent, and tile `[0, n_block_full)` exactly. Validated ONCE up front —
/// before any placement / device allocation / DMA — so a malformed combine is
/// attributed to its own inputs with context (the offending `g`, its `n_block`, and
/// the shard span) rather than silently producing a wrong tensor or reading past a
/// short partial. Cheap O(G), off the bandwidth-critical path (architecture.md §11.4).
///
/// The checks (the SAME contract both tiers depend on):
///   1. `partials.size() == shards.size()` (== G).
///   2. `P >= 0` and `n_block_full >= 0`.
///   3. each partial g spans exactly its shard's block range
///      (`partials[g].n_block == shards[g].b1 - shards[g].b0`).
///   4. every NON-EMPTY partial agrees on P (`partials[g].P == P`).
///   5. every NON-EMPTY partial's storage matches its declared extent
///      (`f2.size() == vpair.size() == P*P*n_block` and
///      `block_sizes.size() == n_block`) — the short-partial OOB guard (cleanup
///      B5 / C1): both combines index `f2[in_base + e]` up to `P*P*n_block - 1`
///      (host) and drive the `cudaMemcpyPeer` byte count off `P*P*n_block` (P2P),
///      so a partial with the right scalar `n_block`/`P` but an under-sized vector
///      would read out of bounds. This closes that gap for both tiers at once.
///   6. the shards together tile `[0, n_block_full)` contiguously (`Σ span == n_block_full`).
///
/// @param who           the calling combine's qualified name, prefixed onto every
///                      error message so a throw still names its own tier
///                      (e.g. "steppe::core::combine_f2_partials_host").
/// @param partials      G compact F2BlockTensors in g=0..G-1 order.
/// @param shards        the block-aligned plan (plan_block_shards); `shards[g]` is
///                      partial g's owned block range.
/// @param P             the combined population count (leading dim of every slab).
/// @param n_block_full  total block count of the combined tensor.
/// @throws std::runtime_error on any precondition violation, with context.
inline void validate_f2_partials(
    const char* who,
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full) {
    const std::string prefix = std::string(who) + ": ";

    if (partials.size() != shards.size()) {
        throw std::runtime_error(
            prefix + "partials count (" + std::to_string(partials.size()) +
            ") != shards count (" + std::to_string(shards.size()) + ")");
    }
    if (P < 0 || n_block_full < 0) {
        throw std::runtime_error(prefix + "negative P or n_block_full");
    }

    // P*P*n_block element count of a non-empty partial's f2/vpair (computed once
    // per g in size_t to match F2BlockTensor::size()'s widening; check 5).
    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

    // Each non-empty partial must agree on P, carry storage matching its extent, and
    // span exactly its shard's block range; together the shards must tile
    // [0, n_block_full) contiguously.
    long covered = 0;  // running count of blocks the shards account for
    for (std::size_t g = 0; g < partials.size(); ++g) {
        const F2BlockTensor& part = partials[g];
        const steppe::device::DeviceShard& sh = shards[g];
        const int span_blocks = sh.b1 - sh.b0;  // shard's block count (>= 0)

        if (part.n_block != span_blocks) {
            throw std::runtime_error(
                prefix + "partial[" + std::to_string(g) + "].n_block (" +
                std::to_string(part.n_block) + ") != shard block span [" +
                std::to_string(sh.b0) + ", " + std::to_string(sh.b1) + ") = " +
                std::to_string(span_blocks));
        }
        // P is the leading dim of every slab; a non-empty partial must match the
        // combined P (an empty partial carries no slab and is exempt).
        if (part.n_block > 0 && part.P != P) {
            throw std::runtime_error(
                prefix + "partial[" + std::to_string(g) + "].P (" +
                std::to_string(part.P) + ") != combined P (" +
                std::to_string(P) + ")");
        }
        // Storage must match the declared extent (the short-partial OOB guard,
        // cleanup B5 / C1): both combines index P*P*n_block doubles and n_block ints.
        if (part.n_block > 0) {
            const std::size_t want_slabs =
                slab * static_cast<std::size_t>(part.n_block);
            const std::size_t want_counts =
                static_cast<std::size_t>(part.n_block);
            if (part.f2.size() != want_slabs ||
                part.vpair.size() != want_slabs ||
                part.block_sizes.size() != want_counts) {
                throw std::runtime_error(
                    prefix + "partial[" + std::to_string(g) +
                    "] storage size mismatch: expected f2/vpair=" +
                    std::to_string(want_slabs) + " (P*P*n_block), block_sizes=" +
                    std::to_string(want_counts) + " (n_block); got f2=" +
                    std::to_string(part.f2.size()) + ", vpair=" +
                    std::to_string(part.vpair.size()) + ", block_sizes=" +
                    std::to_string(part.block_sizes.size()) +
                    " (a short partial would read past its storage)");
            }
        }
        covered += span_blocks;
    }
    if (covered != static_cast<long>(n_block_full)) {
        throw std::runtime_error(
            prefix + "shards cover " + std::to_string(covered) +
            " blocks but n_block_full = " + std::to_string(n_block_full) +
            " (the shards must tile [0, n_block_full))");
    }
}

/// The DEVICE-RESIDENT sibling of validate_f2_partials (cleanup B5 / Option A): the
/// SAME fail-fast contract, validated over `DevicePartial` handles instead of host
/// `F2BlockTensor` partials, so the device-resident combine
/// (combine_f2_partials_resident) rejects malformed inputs IDENTICALLY to the
/// host-staged tier — the two tiers stay parity-NEUTRAL siblings (architecture.md
/// §11.4, §12). A separate overload (NOT an edit to the F2BlockTensor validator)
/// keeps the host tier's reject behavior frozen and risk-free, while CUDA-free
/// `DevicePartial` keeps this header CUDA-free (it compiles into a CUDA TU only).
///
/// The checks mirror validate_f2_partials, per g:
///   1. `partials.size() == shards.size()` (== G).
///   2. `P >= 0` and `n_block_full >= 0`.
///   3. each handle g spans exactly its shard's block range
///      (`partials[g].n_block_local == shards[g].b1 - shards[g].b0`).
///   4. each handle's placement offset matches its shard (`partials[g].b0 == shards[g].b0`).
///   5. every NON-EMPTY handle agrees on P (`partials[g].P == P`).
///   6. every NON-EMPTY handle's host block_sizes is sized n_block_local (the count axis;
///      the resident f2/vpair extent is P*P*n_block_local by construction in
///      run_f2_blocks_resident — not re-checkable from this CUDA-free header).
///   7. the shards together tile `[0, n_block_full)` contiguously (`Σ span == n_block_full`).
///
/// @param who           the calling combine's qualified name, prefixed onto every error.
/// @param partials      G resident DevicePartial handles in g=0..G-1 order.
/// @param shards        the block-aligned plan (plan_block_shards); the authoritative
///                      tiling the handles are cross-checked against.
/// @param P             the combined population count (leading dim of every slab).
/// @param n_block_full  total block count of the combined tensor.
/// @throws std::runtime_error on any precondition violation, with context.
inline void validate_resident_partials(
    const char* who,
    std::span<const steppe::device::DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full) {
    const std::string prefix = std::string(who) + ": ";

    if (partials.size() != shards.size()) {
        throw std::runtime_error(
            prefix + "partials count (" + std::to_string(partials.size()) +
            ") != shards count (" + std::to_string(shards.size()) + ")");
    }
    if (P < 0 || n_block_full < 0) {
        throw std::runtime_error(prefix + "negative P or n_block_full");
    }

    long covered = 0;  // running count of blocks the shards account for
    for (std::size_t g = 0; g < partials.size(); ++g) {
        const steppe::device::DevicePartial& part = partials[g];
        const steppe::device::DeviceShard& sh = shards[g];
        const int span_blocks = sh.b1 - sh.b0;  // shard's block count (>= 0)

        if (part.n_block_local != span_blocks) {
            throw std::runtime_error(
                prefix + "partial[" + std::to_string(g) + "].n_block_local (" +
                std::to_string(part.n_block_local) + ") != shard block span [" +
                std::to_string(sh.b0) + ", " + std::to_string(sh.b1) + ") = " +
                std::to_string(span_blocks));
        }
        if (part.b0 != sh.b0) {
            throw std::runtime_error(
                prefix + "partial[" + std::to_string(g) + "].b0 (" +
                std::to_string(part.b0) + ") != shard.b0 (" +
                std::to_string(sh.b0) + ")");
        }
        if (part.n_block_local > 0 && part.P != P) {
            throw std::runtime_error(
                prefix + "partial[" + std::to_string(g) + "].P (" +
                std::to_string(part.P) + ") != combined P (" +
                std::to_string(P) + ")");
        }
        if (part.n_block_local > 0 &&
            part.block_sizes.size() != static_cast<std::size_t>(part.n_block_local)) {
            throw std::runtime_error(
                prefix + "partial[" + std::to_string(g) +
                "] block_sizes size mismatch: expected " +
                std::to_string(part.n_block_local) + " (n_block_local), got " +
                std::to_string(part.block_sizes.size()));
        }
        covered += span_blocks;
    }
    if (covered != static_cast<long>(n_block_full)) {
        throw std::runtime_error(
            prefix + "shards cover " + std::to_string(covered) +
            " blocks but n_block_full = " + std::to_string(n_block_full) +
            " (the shards must tile [0, n_block_full))");
    }
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_PARTIALS_VALIDATE_HPP
