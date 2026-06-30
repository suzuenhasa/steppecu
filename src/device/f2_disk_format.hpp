// src/device/f2_disk_format.hpp
//
// M5 TIER 2 on-disk f2_blocks cache file format (CUDA-FREE). Binary, little-endian,
// block-major: a fixed 64-byte header, then the f2 region (all blocks' [P²] slabs
// contiguous, block-major), then the vpair region (same), then the block_sizes
// trailer. The within-block layout is column-major i+P·j and the block axis is outer,
// so it is BYTE-IDENTICAL to F2BlockTensor's i + P·j + P·P·b (include/steppe/fstats.hpp:37-46) and to
// the resident tensor — a whole-file read is a single memcpy-equivalent into an
// F2BlockTensor, and the fit preads block b's f2 slab at a computed offset without
// scanning. AT2-compat is a GOAL: the [P²]-slab-per-block block-major layout matches
// AT2's on-disk f2_blocks ordering; the steppe header is a fixed 64-byte prefix the M7
// reader strips. Storage is FP64 in EVERY precision mode (include/steppe/fstats.hpp:17) -> dtype fixed.
#ifndef STEPPE_DEVICE_F2_DISK_FORMAT_HPP
#define STEPPE_DEVICE_F2_DISK_FORMAT_HPP

#include <cstddef>
#include <cstdint>

namespace steppe::device {

inline constexpr char     kF2DiskMagic[8]   = {'S','T','P','F','2','B','K','1'};  // "STPF2BK1"
inline constexpr std::uint32_t kF2DiskVersion   = 1u;  // f2.bin BINARY format version (the writer stamps this into F2DiskHeader.version; read_f2_dir gates on it). DISTINCT from the meta.json SIDECAR schema version (kF2MetaSchemaVersion, src/app/f2_dir_writer.hpp) — do NOT conflate: this versions the numeric payload bytes, that versions the JSON provenance shape. Single home so the writer/reader cannot drift
inline constexpr std::uint32_t kF2DiskDtypeFp64 = 1u;  // FP64 little-endian (storage is FP64 in every precision mode, include/steppe/fstats.hpp:17)

/// 64-byte fixed header at file offset 0 (little-endian, packed). The remainder:
///   header[64] | f2[P²·n_block doubles] | vpair[P²·n_block doubles] | block_sizes[n_block int32]
struct F2DiskHeader {                       // sizeof == 64 (padded)
    char          magic[8];                 // kF2DiskMagic
    std::uint32_t version;                  // kF2DiskVersion
    std::uint32_t dtype;                    // kF2DiskDtypeFp64
    std::int32_t  P;
    std::int32_t  n_block;
    std::uint64_t f2_offset;                // == 64
    std::uint64_t vpair_offset;             // == 64 + P²·n_block·8
    std::uint64_t block_sizes_offset;       // == vpair_offset + P²·n_block·8
    std::uint8_t  reserved[16];             // zero
};
/// The fixed on-disk header size in bytes (file offset 0; the f2 region begins here).
/// Single home for the "64" that the prose (above), the layout, and the M7 reader's
/// strip all reference, so they cannot desync. Value frozen by the on-disk format.
inline constexpr std::size_t kF2DiskHeaderSize = sizeof(F2DiskHeader);
static_assert(kF2DiskHeaderSize == 64, "F2DiskHeader must be exactly 64 bytes");

namespace detail {
/// Byte offset of block b's [P²] FP64 slab within a region that starts at `base`:
/// `base + P²·b·sizeof(double)`. The `P·P·b` widening is done in `std::uint64_t`
/// (cast each factor BEFORE multiplying) so the stride cannot wrap at P²·b scale.
/// SINGLE home for the f2/vpair slab arithmetic — both region accessors below differ
/// ONLY by their `base` offset, so the identical stride lives here once.
[[nodiscard]] inline std::uint64_t slab_offset(std::uint64_t base, const F2DiskHeader& h,
                                               int b) noexcept {
    return base + static_cast<std::uint64_t>(h.P) * static_cast<std::uint64_t>(h.P) *
                      static_cast<std::uint64_t>(b) * sizeof(double);
}
}  // namespace detail

/// Byte offset of block b's [P²] f2 slab (column-major i+P·j within the slab).
[[nodiscard]] inline std::uint64_t f2_block_offset(const F2DiskHeader& h, int b) noexcept {
    return detail::slab_offset(h.f2_offset, h, b);
}
[[nodiscard]] inline std::uint64_t vpair_block_offset(const F2DiskHeader& h, int b) noexcept {
    return detail::slab_offset(h.vpair_offset, h, b);
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_F2_DISK_FORMAT_HPP
