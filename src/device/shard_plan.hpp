// src/device/shard_plan.hpp
//
// Host-pure, CUDA-free planning step for the single-node multi-GPU precompute:
// it splits the jackknife blocks across the G devices in contiguous whole-block
// ranges balanced by SNP count, so each block is computed entirely on one device
// — the invariant the multi-GPU bit-parity guarantee rests on.
//
// Reference: docs/reference/src_device_shard_plan.hpp.md
#ifndef STEPPE_DEVICE_SHARD_PLAN_HPP
#define STEPPE_DEVICE_SHARD_PLAN_HPP

#include <cstddef>
#include <span>
#include <vector>

#include "core/domain/block_partition_rule.hpp"

namespace steppe::device {

// DeviceShard: per-device block range + covering SNP-column range — reference §3
struct DeviceShard {
    int  b0 = 0;
    int  b1 = 0;
    long s0 = 0;
    long s1 = 0;

    [[nodiscard]] bool empty() const noexcept { return b0 >= b1; }
};

// plan_block_shards: SNP-count-balanced contiguous block partition — reference §§4-5
[[nodiscard]] std::vector<DeviceShard> plan_block_shards(
    std::span<const steppe::core::BlockRange> ranges,
    std::size_t G);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_SHARD_PLAN_HPP
