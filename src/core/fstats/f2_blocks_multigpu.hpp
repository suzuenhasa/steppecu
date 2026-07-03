// src/core/fstats/f2_blocks_multigpu.hpp
//
// Entry point for computing the per-block f2 tensor across all G GPUs of a
// machine. Host-pure and CUDA-free: whole blocks are sharded across devices and
// the partials combined in fixed device order, so the result is bit-identical to
// the single-GPU path and identical for any G.
//
// Reference: docs/reference/src_core_fstats_f2_blocks_multigpu.hpp.md
#ifndef STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_HPP
#define STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_HPP

#include "core/internal/views.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"
#include "device/resources.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/f2_blocks_out.hpp"

namespace steppe::core {

// Adaptive tiered precompute — reference §6
[[nodiscard]] steppe::device::F2BlocksOut compute_f2_blocks_multigpu_tiered(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision);

// Device-resident multi-GPU precompute — reference §5
[[nodiscard]] steppe::device::DeviceF2Blocks compute_f2_blocks_multigpu_device(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision);

// Combined host-tensor precompute — reference §4
[[nodiscard]] F2BlockTensor compute_f2_blocks_multigpu(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_HPP
