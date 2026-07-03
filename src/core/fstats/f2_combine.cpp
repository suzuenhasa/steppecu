// src/core/fstats/f2_combine.cpp
//
// combine_f2_partials_host assembles the per-device f2 block partials into the
// full-shape output tensor. It visits devices in fixed order and copies each
// device's compact shard into place at its block offset; the shards are disjoint,
// so every slab is written exactly once and the result is bit-identical across
// GPU counts and to the single-GPU reference.
#include "core/fstats/f2_combine.hpp"

#include <algorithm>
#include <cstddef>
#include <span>

#include "steppe/fstats.hpp"
#include "device/shard_plan.hpp"
#include "core/fstats/f2_partials_validate.hpp"

namespace steppe::core {

F2BlockTensor combine_f2_partials_host(
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full) {
    validate_f2_partials("steppe::core::combine_f2_partials_host",
                         partials, shards, P, n_block_full);

    F2BlockTensor out;
    out.P = P;
    out.n_block = n_block_full;
    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total = slab * static_cast<std::size_t>(n_block_full);
    out.f2.assign(total, 0.0);
    out.vpair.assign(total, 0.0);
    out.block_sizes.assign(static_cast<std::size_t>(n_block_full), 0);

    for (std::size_t g = 0; g < partials.size(); ++g) {
        const F2BlockTensor& part = partials[g];
        if (part.n_block <= 0) continue;
        const std::size_t b0 = static_cast<std::size_t>(shards[g].b0);
        const std::size_t part_elems =
            slab * static_cast<std::size_t>(part.n_block);
        const std::size_t out_base = slab * b0;
        auto place = [&](const double* src, double* dst) {
            std::copy_n(src, part_elems, dst + out_base);
        };
        place(part.f2.data(),    out.f2.data());
        place(part.vpair.data(), out.vpair.data());
        std::copy_n(part.block_sizes.data(),
                    static_cast<std::size_t>(part.n_block),
                    out.block_sizes.data() + b0);
    }

    return out;
}

}  // namespace steppe::core
