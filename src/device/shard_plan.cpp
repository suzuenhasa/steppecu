// src/device/shard_plan.cpp
//
// plan_block_shards: the block-aligned, SNP-count-balanced shard planner.
// Host-pure and CUDA-free, so it compiles into steppe_device as a plain .cpp
// and is equally exercisable host-only by the parity test.
#include "device/shard_plan.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  
#include "core/internal/launch_config.hpp"        

namespace steppe::device {

std::vector<DeviceShard> plan_block_shards(
    std::span<const steppe::core::BlockRange> ranges,
    std::size_t G) {
    if (G == 0) {
        throw std::runtime_error(
            "steppe::device::plan_block_shards: G == 0 — the SPMG shard plan "
            "requires at least one device (architecture.md §9)");
    }

    const std::size_t n_block = ranges.size();
    std::vector<DeviceShard> plan(G);  

    if (n_block == 0) {
        return plan;
    }

    long total_snps = 0;
    for (std::size_t b = 0; b < n_block; ++b) {
        total_snps += ranges[b].size();
    }
    const long G_signed = static_cast<long>(G);
    const long target_per_device =
        core::cdiv(total_snps, G_signed);  

    std::size_t g = 0;          
    std::size_t b0 = 0;         
    long device_snps = 0;       

    const auto make_shard = [&](std::size_t lo, std::size_t hi) {
        return DeviceShard{
            static_cast<int>(lo),
            static_cast<int>(hi),
            ranges[lo].begin,
            ranges[hi - 1].end};
    };

    for (std::size_t b = 0; b < n_block; ++b) {
        device_snps += ranges[b].size();

        const bool more_devices_left = (g + 1 < G);
        const bool reached_target = (device_snps >= target_per_device);
        if (more_devices_left && reached_target && (b + 1 < n_block)) {
            const std::size_t b1 = b + 1;
            plan[g] = make_shard(b0, b1);
            ++g;
            b0 = b1;
            device_snps = 0;
        }
    }

    plan[g] = make_shard(b0, n_block);

    return plan;
}

}  // namespace steppe::device
