// src/device/stream_f2_blocks.hpp
//
// The CUDA-free StreamTarget descriptor: the seam the CUDA-free orchestrator uses to
// tell the CUDA backend where a streamed f2_blocks result should be spilled (host RAM
// or disk). It names the destination without naming any CUDA sink type.
//
// Reference: docs/reference/src_device_stream_f2_blocks.hpp.md
#ifndef STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP
#define STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP

#include <string>

#include "device/tier_select.hpp"
#include "device/f2_blocks_out.hpp"
#include "steppe/fstats.hpp"

namespace steppe::device {

// StreamTarget request descriptor — reference §4
struct StreamTarget {
    OutputTier tier = OutputTier::HostRam;
    F2BlockTensor* host_dst = nullptr;
    std::string disk_path;
    DiskF2Blocks* disk_dst = nullptr;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP
