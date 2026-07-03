// src/core/fstats/f2_partials_validate.hpp
//
// Fail-fast precondition guard shared by both f2 combine tiers, so they reject
// malformed partials identically. Header-only and CUDA-free so it compiles into
// both the host core and a CUDA TU.
//
// Reference: docs/reference/src_core_fstats_f2_partials_validate.hpp.md
#ifndef STEPPE_CORE_FSTATS_F2_PARTIALS_VALIDATE_HPP
#define STEPPE_CORE_FSTATS_F2_PARTIALS_VALIDATE_HPP

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include "steppe/fstats.hpp"
#include "device/shard_plan.hpp"
#include "device/device_partial.hpp"

namespace steppe::core {

namespace detail {

// validate_partials_scaffold, the shared internal helper — reference §7
template <typename CheckG>
inline void validate_partials_scaffold(
    const char* who,
    std::size_t partials_count,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full,
    CheckG&& check_g) {
    const std::string prefix = std::string(who) + ": ";

    if (partials_count != shards.size()) {
        throw std::runtime_error(
            prefix + "partials count (" + std::to_string(partials_count) +
            ") != shards count (" + std::to_string(shards.size()) + ")");
    }
    if (P < 0 || n_block_full < 0) {
        throw std::runtime_error(prefix + "negative P or n_block_full");
    }

    long covered = 0;
    for (std::size_t g = 0; g < partials_count; ++g) {
        const steppe::device::DeviceShard& sh = shards[g];
        const int span_blocks = sh.b1 - sh.b0;
        check_g(prefix, g, sh, span_blocks);
        covered += span_blocks;
    }
    if (covered != static_cast<long>(n_block_full)) {
        throw std::runtime_error(
            prefix + "shards cover " + std::to_string(covered) +
            " blocks but n_block_full = " + std::to_string(n_block_full) +
            " (the shards must tile [0, n_block_full))");
    }
}

}  // namespace detail

// validate_f2_partials, the host-staged tier — reference §5
inline void validate_f2_partials(
    const char* who,
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full) {
    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

    detail::validate_partials_scaffold(
        who, partials.size(), shards, P, n_block_full,
        [&](const std::string& prefix, std::size_t g,
            const steppe::device::DeviceShard& sh, int span_blocks) {
            const F2BlockTensor& part = partials[g];

            if (part.n_block != span_blocks) {
                throw std::runtime_error(
                    prefix + "partial[" + std::to_string(g) + "].n_block (" +
                    std::to_string(part.n_block) + ") != shard block span [" +
                    std::to_string(sh.b0) + ", " + std::to_string(sh.b1) + ") = " +
                    std::to_string(span_blocks));
            }
            if (part.n_block > 0 && part.P != P) {
                throw std::runtime_error(
                    prefix + "partial[" + std::to_string(g) + "].P (" +
                    std::to_string(part.P) + ") != combined P (" +
                    std::to_string(P) + ")");
            }
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
        });
}

// validate_resident_partials, the device-resident tier — reference §6
inline void validate_resident_partials(
    const char* who,
    std::span<const steppe::device::DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full) {
    detail::validate_partials_scaffold(
        who, partials.size(), shards, P, n_block_full,
        [&](const std::string& prefix, std::size_t g,
            const steppe::device::DeviceShard& sh, int span_blocks) {
            const steppe::device::DevicePartial& part = partials[g];

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
        });
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_PARTIALS_VALIDATE_HPP
