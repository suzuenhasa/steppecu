// src/device/device_f2_blocks.hpp — owning handle for the phase-1 f2/paired-variance
// result [P × P × n_block], left resident in GPU memory for the fit phase to read.
// CUDA-free at this seam: the DeviceBuffer<double> owners live in a private Impl (.cu).
//
// Reference: docs/reference/src_device_device_f2_blocks.hpp.md
#ifndef STEPPE_DEVICE_DEVICE_F2_BLOCKS_HPP
#define STEPPE_DEVICE_DEVICE_F2_BLOCKS_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "steppe/fstats.hpp"

namespace steppe::device {

// DeviceF2Blocks handle — reference §2
class DeviceF2Blocks {
public:
    DeviceF2Blocks();
    ~DeviceF2Blocks();
    DeviceF2Blocks(DeviceF2Blocks&&) noexcept;
    DeviceF2Blocks& operator=(DeviceF2Blocks&&) noexcept;
    DeviceF2Blocks(const DeviceF2Blocks&) = delete;
    DeviceF2Blocks& operator=(const DeviceF2Blocks&) = delete;

    // Shape & block metadata — reference §3
    int P = 0;
    int n_block = 0;
    int device_id = -1;

    std::vector<int> block_sizes;

    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
               static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
    }
    [[nodiscard]] bool empty() const noexcept { return n_block <= 0 || P <= 0; }

    // Resident device pointers & layout — reference §4
    [[nodiscard]] const double* f2_device() const noexcept;
    [[nodiscard]] const double* vpair_device() const noexcept;

    // Host materialization (to_host) — reference §5
    [[nodiscard]] F2BlockTensor to_host() const;

    struct Impl;
    std::unique_ptr<Impl> impl;

};

// Host→device upload — reference §6
[[nodiscard]] DeviceF2Blocks upload_f2_blocks_to_device(const F2BlockTensor& host, int device_id);

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_DEVICE_F2_BLOCKS_HPP
