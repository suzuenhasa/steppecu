// src/core/fstats/f2_blocks_multigpu_core.hpp
//
// The host-pure, CUDA-free core of the multi-GPU f2-block precompute: the
// block-aligned shard plan plus the concurrent per-device fan-out. The final
// P2P combine needs a CUDA device symbol, so it stays with the public entry
// point (compute_f2_blocks_multigpu), not here.
//
// Reference: docs/reference/src_core_fstats_f2_blocks_multigpu_core.hpp.md
#ifndef STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_CORE_HPP
#define STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_CORE_HPP

#include <cstddef>
#include <span>
#include <vector>

#include "core/internal/views.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"
#include "device/resources.hpp"
#include "device/shard_plan.hpp"
#include "device/device_partial.hpp"

namespace steppe::core {

// Block-aligned multi-GPU shard plan — reference §3
[[nodiscard]] std::vector<steppe::device::DeviceShard> plan_multigpu_shards(
    const BlockPartition& partition, long M, int n_block, std::size_t G);

// Concurrent fan-out, host-staged partials — reference §4, §5
[[nodiscard]] std::vector<F2BlockTensor> compute_multigpu_partials(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision);

// Concurrent fan-out, device-resident partials — reference §4, §6
[[nodiscard]] std::vector<steppe::device::DevicePartial> compute_multigpu_partials_resident(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision);

// Concurrent fan-out, direct into one shared buffer — reference §4, §7
void compute_multigpu_partials_into(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    double* dst_f2, double* dst_vpair, int* block_sizes_dst,
    const Precision& precision);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_CORE_HPP
