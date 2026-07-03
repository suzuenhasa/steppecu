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
#include "device/stream_f2_blocks.hpp"

namespace steppe::core {

// Adaptive tiered precompute — reference §6. When redecode != nullptr the streamed
// tiers re-decode each chunk from the packed genotypes instead of a dense host
// Q/V/N (which the caller passes as null-data MatViews carrying only P and M_kept);
// this is the extract-f2 host-RAM-wall path. Redecode never resolves to Resident.
[[nodiscard]] steppe::device::F2BlocksOut compute_f2_blocks_multigpu_tiered(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision,
    const steppe::device::RedecodeSource* redecode = nullptr);

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
