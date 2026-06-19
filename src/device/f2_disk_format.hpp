// src/device/f2_disk_format.hpp
//
// M5 TIER 2 on-disk f2_blocks cache file format (CUDA-FREE). Binary, little-endian,
// block-major: a fixed 64-byte header, then the f2 region (all blocks' [P²] slabs
// contiguous, block-major), then the vpair region (same), then the block_sizes
// trailer. The within-block layout is column-major i+P·j and the block axis is outer,
// so it is BYTE-IDENTICAL to F2BlockTensor's i + P·j + P·P·b (fstats.hpp:38-46) and to
// the resident tensor — a whole-file read is a single memcpy-equivalent into an
// F2BlockTensor, and the fit preads block b's f2 slab at a computed offset without
// scanning. AT2-compat is a GOAL: the [P²]-slab-per-block block-major layout matches
// AT2's on-disk f2_blocks ordering; the steppe header is a fixed 64-byte prefix the M7
// reader strips. Storage is FP64 in EVERY precision mode (fstats.hpp:18) -> dtype fixed.
#ifndef STEPPE_DEVICE_F2_DISK_FORMAT_HPP
#define STEPPE_DEVICE_F2_DISK_FORMAT_HPP

#include <cstddef>
#include <cstdint>

namespace steppe::device {

inline constexpr char     kF2DiskMagic[8]   = {'S','T','P','F','2','B','K','1'};  // "STPF2BK1"
inline constexpr std::uint32_t kF2DiskDtypeFp64 = 1u;  // FP64 little-endian (storage is FP64 in every precision mode, fstats.hpp:18)

/// 64-byte fixed header at file offset 0 (little-endian, packed). The remainder:
///   header[64] | f2[P²·n_block doubles] | vpair[P²·n_block doubles] | block_sizes[n_block int32]
struct F2DiskHeader {                       // sizeof == 64 (padded)
    char          magic[8];                 // kF2DiskMagic
    std::uint32_t version;                  // 1
    std::uint32_t dtype;                    // kF2DiskDtypeFp64
    std::int32_t  P;
    std::int32_t  n_block;
    std::uint64_t f2_offset;                // == 64
    std::uint64_t vpair_offset;             // == 64 + P²·n_block·8
    std::uint64_t block_sizes_offset;       // == vpair_offset + P²·n_block·8
    std::uint8_t  reserved[16];             // zero
};
static_assert(sizeof(F2DiskHeader) == 64, "F2DiskHeader must be exactly 64 bytes");

/// Byte offset of block b's [P²] f2 slab (column-major i+P·j within the slab).
[[nodiscard]] inline std::uint64_t f2_block_offset(const F2DiskHeader& h, int b) noexcept {
    return h.f2_offset + static_cast<std::uint64_t>(h.P) * static_cast<std::uint64_t>(h.P) *
                             static_cast<std::uint64_t>(b) * sizeof(double);
}
[[nodiscard]] inline std::uint64_t vpair_block_offset(const F2DiskHeader& h, int b) noexcept {
    return h.vpair_offset + static_cast<std::uint64_t>(h.P) * static_cast<std::uint64_t>(h.P) *
                                static_cast<std::uint64_t>(b) * sizeof(double);
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_F2_DISK_FORMAT_HPP
