// src/device/p2p_combine.hpp
//
// The opt-in device-resident P2P f2 combine: GPU 0 (the combine root) pulls each
// peer's resident partial straight GPU-to-GPU into its disjoint place in the full
// result. CUDA-free header — it names only CUDA-free types; GPU calls live in the .cpp.
//
// Reference: docs/reference/src_device_p2p_combine.hpp.md
#ifndef STEPPE_DEVICE_P2P_COMBINE_HPP
#define STEPPE_DEVICE_P2P_COMBINE_HPP

#include <span>

#include "steppe/fstats.hpp"
#include "device/shard_plan.hpp"
#include "device/device_partial.hpp"
#include "device/device_f2_blocks.hpp"

namespace steppe::device {

// Combine resident partials → host F2BlockTensor (one final D2H) — reference §2
[[nodiscard]] F2BlockTensor combine_f2_partials_resident(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id);

// Combine resident partials → root-resident DeviceF2Blocks (no D2H) — reference §2
[[nodiscard]] DeviceF2Blocks combine_f2_partials_resident_device(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_P2P_COMBINE_HPP
