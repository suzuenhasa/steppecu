// src/core/fstats/f2_combine.hpp
//
// The host-staged f2 combine: stitches the G per-GPU compact partials into one
// full-shape [P × P × n_block] result by placing each at its block offset in a
// fixed device order. CUDA-free and host-only, so it compiles into steppe_core
// and unit-tests without a GPU.
//
// Reference: docs/reference/src_core_fstats_f2_combine.hpp.md
#ifndef STEPPE_CORE_FSTATS_F2_COMBINE_HPP
#define STEPPE_CORE_FSTATS_F2_COMBINE_HPP

#include <span>

#include "steppe/fstats.hpp"
#include "device/shard_plan.hpp"

namespace steppe::core {

// combine_f2_partials_host — reference §4
[[nodiscard]] F2BlockTensor combine_f2_partials_host(
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_COMBINE_HPP
