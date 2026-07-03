// src/device/f2_blocks_out.hpp
//
// F2BlocksOut — the unified f2-precompute result, held in exactly one of three
// storage tiers (GPU-resident / host RAM / on-disk) and read back through
// tier-agnostic accessors that yield a bit-identical host tensor. CUDA-free
// header: the accessor bodies live in cuda/f2_blocks_out.cu.
//
// Reference: docs/reference/src_device_f2_blocks_out.hpp.md
#ifndef STEPPE_DEVICE_F2_BLOCKS_OUT_HPP
#define STEPPE_DEVICE_F2_BLOCKS_OUT_HPP

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "steppe/fstats.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/tier_select.hpp"

namespace steppe::device {

// Per-block slab element count (P²) — reference §3
[[nodiscard]] inline std::size_t slab_elems(int P) noexcept {
    return static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
}

// Disk read-handle deleter — reference §4
struct FileCloser {
    void operator()(std::FILE* f) const noexcept;
};

// On-disk cache descriptor (TIER 2) — reference §5
struct DiskF2Blocks {
    DiskF2Blocks() = default;
    ~DiskF2Blocks() = default;
    DiskF2Blocks(DiskF2Blocks&&) noexcept = default;
    DiskF2Blocks& operator=(DiskF2Blocks&&) noexcept = default;
    DiskF2Blocks(const DiskF2Blocks&) = delete;
    DiskF2Blocks& operator=(const DiskF2Blocks&) = delete;

    std::string path;
    int P = 0;
    int n_block = 0;
    std::vector<int> block_sizes;
    std::unique_ptr<std::FILE, FileCloser> read_handle;
};

// The unified precompute result + tier-agnostic read-back accessors — reference §6
class F2BlocksOut {
public:
    F2BlocksOut() = default;
    ~F2BlocksOut() = default;
    F2BlocksOut(F2BlocksOut&&) noexcept = default;
    F2BlocksOut& operator=(F2BlocksOut&&) noexcept = default;
    F2BlocksOut(const F2BlocksOut&) = delete;
    F2BlocksOut& operator=(const F2BlocksOut&) = delete;

    OutputTier tier = OutputTier::Resident;
    int P = 0;
    int n_block = 0;
    std::vector<int> block_sizes;

    DeviceF2Blocks resident;
    F2BlockTensor  host;
    DiskF2Blocks   disk;


    [[nodiscard]] F2BlockTensor to_host() const;

    void read_block_to_host(int b, double* f2_slab_out, double* vpair_slab_out) const;

    [[nodiscard]] std::size_t size() const noexcept {
        return slab_elems(P) * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
    }
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_F2_BLOCKS_OUT_HPP
