// src/device/f2_disk_format.hpp
//
// On-disk f2_blocks cache format: a CUDA-free binary, little-endian, block-major
// layout whose bytes match the in-memory f2 tensor, so a whole-file load is one bulk
// copy and any single block is directly seekable.
//
// Reference: docs/reference/src_device_f2_disk_format.hpp.md
#ifndef STEPPE_DEVICE_F2_DISK_FORMAT_HPP
#define STEPPE_DEVICE_F2_DISK_FORMAT_HPP

#include <cstddef>
#include <cstdint>

namespace steppe::device {

// Named constants (magic, version, dtype) — reference §3
inline constexpr char     kF2DiskMagic[8]   = {'S','T','P','F','2','B','K','1'};
inline constexpr std::uint32_t kF2DiskVersion   = 1u;
inline constexpr std::uint32_t kF2DiskDtypeFp64 = 1u;

// Fixed 64-byte file header — reference §4
struct F2DiskHeader {
    char          magic[8];
    std::uint32_t version;
    std::uint32_t dtype;
    std::int32_t  P;
    std::int32_t  n_block;
    std::uint64_t f2_offset;
    std::uint64_t vpair_offset;
    std::uint64_t block_sizes_offset;
    std::uint8_t  reserved[16];
};
inline constexpr std::size_t kF2DiskHeaderSize = sizeof(F2DiskHeader);
static_assert(kF2DiskHeaderSize == 64, "F2DiskHeader must be exactly 64 bytes");

// Block offset arithmetic — reference §5
namespace detail {
[[nodiscard]] inline std::uint64_t slab_offset(std::uint64_t base, const F2DiskHeader& h,
                                               int b) noexcept {
    return base + static_cast<std::uint64_t>(h.P) * static_cast<std::uint64_t>(h.P) *
                      static_cast<std::uint64_t>(b) * sizeof(double);
}
}  // namespace detail

[[nodiscard]] inline std::uint64_t f2_block_offset(const F2DiskHeader& h, int b) noexcept {
    return detail::slab_offset(h.f2_offset, h, b);
}
[[nodiscard]] inline std::uint64_t vpair_block_offset(const F2DiskHeader& h, int b) noexcept {
    return detail::slab_offset(h.vpair_offset, h, b);
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_F2_DISK_FORMAT_HPP
