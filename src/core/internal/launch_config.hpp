// src/core/internal/launch_config.hpp
//
// The single home for GPU launch-grid math: ceiling division, the hardware
// grid-dimension limits, and the two extent helpers that turn a problem size
// into a block count. Pure host/device constexpr integer arithmetic with no
// CUDA, so both the host-pure core and the device kernels can include it.
//
// Reference: docs/reference/src_core_internal_launch_config.hpp.md
#ifndef STEPPE_CORE_INTERNAL_LAUNCH_CONFIG_HPP
#define STEPPE_CORE_INTERNAL_LAUNCH_CONFIG_HPP

#include "core/internal/host_device.hpp"
#include "steppe/config.hpp"

namespace steppe::core {

// Hardware grid-dimension limits — reference §2
inline constexpr unsigned kMaxGridX = 2147483647u;
inline constexpr unsigned kMaxGridY = 65535u;
inline constexpr unsigned kMaxGridZ = kMaxGridY;

// Ceiling division — reference §3
[[nodiscard]] STEPPE_HD constexpr int cdiv(int n, int block) noexcept {
    return (n + block - 1) / block;
}

[[nodiscard]] STEPPE_HD constexpr long cdiv(long n, long block) noexcept {
    return (n + block - 1) / block;
}

// Grid extent for square f2 launches — reference §4
[[nodiscard]] STEPPE_HD constexpr int grid_for(int n, int block = kCdivBlock,
                                               [[maybe_unused]] unsigned max_grid = kMaxGridY) noexcept {
    const int extent = cdiv(n, block);
    STEPPE_ASSERT(extent >= 0 && static_cast<unsigned>(extent) <= max_grid,
                  "grid extent exceeds the hardware grid-dimension limit "
                  "(architecture.md §7; cleanup X-7/B6) — re-orient the large axis "
                  "onto gridDim.x or tile it");
    return extent;
}

// Batch (z-axis) extent — reference §5
[[nodiscard]] inline unsigned grid_z_extent(int n_in_group) noexcept {
    const unsigned z = static_cast<unsigned>(n_in_group);
    STEPPE_ASSERT(n_in_group >= 1 && z <= kMaxGridZ,
                  "M4 batch count (gridDim.z = n_in_group) must be in [1, kMaxGridZ] "
                  "(architecture.md §7; cleanup X-7/B6) — the backend tiles the batch "
                  "so this holds");
    return z;
}

// Decode-kernel block geometry — reference §6
inline constexpr int kDecodeBlockX = 32;
inline constexpr int kDecodeBlockY = 8;

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_LAUNCH_CONFIG_HPP
