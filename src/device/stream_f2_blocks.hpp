// src/device/stream_f2_blocks.hpp
//
// CUDA-FREE declaration of the M5 STREAMED f2_blocks request — the seam the CUDA-free
// orchestrator (steppe::core) uses to drive the HostRam / Disk tiers without naming a
// CUDA type. The actual block-stream loop (stream_f2_blocks.cu) REUSES
// run_f2_blocks_resident's prologue + per-block gather/GEMM/assemble VERBATIM and spills
// each block's [P²] slab through a triple-buffered sink (block_sink.cuh) — the per-block
// bits are BIT-IDENTICAL to the device-resident path (§12); the ONLY difference is the
// result is spilled block-by-block instead of left whole.
//
// The orchestrator does NOT construct a CUDA BlockSink (it is CUDA-free). It calls the
// backend virtual compute_f2_blocks_streamed with a CUDA-FREE StreamTarget (the tier +
// the CUDA-free destinations: an F2BlockTensor* for HostRam, a path + DiskF2Blocks* for
// Disk); the CUDA backend builds the matching HostRamSink / DiskSink internally. Resident
// NEVER reaches here (the orchestrator calls compute_f2_blocks_device directly).
#ifndef STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP
#define STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP

#include <string>

#include "device/tier_select.hpp"      // OutputTier
#include "device/f2_blocks_out.hpp"    // DiskF2Blocks (CUDA-free descriptor)
#include "steppe/fstats.hpp"           // F2BlockTensor (HostRam destination)

namespace steppe::device {

/// CUDA-FREE request descriptor for compute_f2_blocks_streamed (§5 seam). EXACTLY the
/// destination the resolved tier needs:
///   * tier == HostRam : `host_dst` non-null; the backend's HostRamSink streams into it.
///   * tier == Disk    : `disk_path` is the cache file; the backend's DiskSink streams to
///                       it and reopens it read-only into `disk_dst`.
/// tier == Resident NEVER reaches this seam (the orchestrator calls
/// compute_f2_blocks_device directly). All members are CUDA-free, so the orchestrator
/// constructs this without a CUDA header; the CUDA backend builds the concrete sink.
struct StreamTarget {
    OutputTier tier = OutputTier::HostRam;
    F2BlockTensor* host_dst = nullptr;   ///< HostRam destination (owned by F2BlocksOut.host).
    std::string disk_path;               ///< Disk cache file path.
    DiskF2Blocks* disk_dst = nullptr;    ///< Disk descriptor to populate (F2BlocksOut.disk).
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP
