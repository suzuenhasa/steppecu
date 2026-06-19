// src/device/device_f2_blocks.hpp — CUDA-FREE opaque handle to the FULL f2/Vpair
// result [P × P × n_block] left RESIDENT in VRAM (M4.5 device-resident output). The
// PRIMARY product of the precompute: the result STAYS on the GPU; the host
// F2BlockTensor is an opt-in to_host() materialization (the ONLY D2H site). Mirrors
// device_partial.hpp's CUDA-free-decl pattern: names NO CUDA type; the
// DeviceBuffer<double> owners live in the Impl in cuda/device_f2_blocks.cu.
#ifndef STEPPE_DEVICE_DEVICE_F2_BLOCKS_HPP
#define STEPPE_DEVICE_DEVICE_F2_BLOCKS_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "steppe/fstats.hpp"  // steppe::F2BlockTensor (the opt-in host materialization type)

namespace steppe::device {

/// Move-only, OPAQUE owner of the FULL [P × P × n_block] f2/Vpair result left
/// RESIDENT in VRAM on `device_id` — the PRIMARY output of compute_f2_blocks /
/// compute_f2_blocks_multigpu (the M4.5 device-resident-output cure). CUDA-FREE at
/// this seam: the DeviceBuffer<double> f2/vpair owners live in `Impl`
/// (cuda/device_f2_blocks_impl.cuh). The shape fields are plain host scalars so the
/// CUDA-free orchestrator (steppe::core) and the CUDA-free public API can hold and
/// forward the handle without seeing <cuda_runtime.h>.
///
/// LIFETIME: the DeviceBuffer<double> pair was cudaMalloc'd on `device_id`; this
/// handle MOVES out of the producer and frees in its destructor (cudaFree is
/// pointer-device-aware). It is the precompute->fit handoff: the fit reads
/// f2()/vpair() in VRAM; nothing copies to host unless to_host() is called.
class DeviceF2Blocks {
public:
    DeviceF2Blocks();                                    // empty / moved-from / degenerate
    ~DeviceF2Blocks();                                   // frees the resident DeviceBuffer pair
    DeviceF2Blocks(DeviceF2Blocks&&) noexcept;           // move-only
    DeviceF2Blocks& operator=(DeviceF2Blocks&&) noexcept;
    DeviceF2Blocks(const DeviceF2Blocks&) = delete;
    DeviceF2Blocks& operator=(const DeviceF2Blocks&) = delete;

    // ---- Shape (plain host scalars; CUDA-free) ----
    int P = 0;        ///< population count (leading dim of every slab).
    int n_block = 0;  ///< total blocks of the FULL tensor (NOT a shard count).
    int device_id = -1;  ///< the CUDA ordinal the f2/vpair buffers are resident on.

    // ---- Per-block SNP counts (host int; the S4 jackknife metadata, placed host-side).
    std::vector<int> block_sizes;  ///< length n_block (0 on degenerate).

    /// Flat element count P*P*n_block of the resident f2/vpair (convenience).
    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
               static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
    }
    /// True for a degenerate/empty result (no resident buffers).
    [[nodiscard]] bool empty() const noexcept { return n_block <= 0 || P <= 0; }

    /// Borrowed device pointers to the resident f2 / vpair (column-major
    /// [P × P × n_block], i + P·j + P·P·b). null iff empty(). The fit engine reads
    /// these in VRAM. Defined in cuda/device_f2_blocks.cu (it dereferences Impl).
    [[nodiscard]] const double* f2_device() const noexcept;
    [[nodiscard]] const double* vpair_device() const noexcept;

    /// THE ONLY D2H + host alloc in the whole pipeline (opt-in materialization).
    /// Allocates a host F2BlockTensor and copies the resident f2/vpair down with one
    /// D2H each, PINNING the host destinations for the D2H window
    /// (RegisteredHostRegion, graceful pageable degrade) EXACTLY like
    /// p2p_combine.cu:185-186 — parity-neutral (pinned vs pageable moves the same
    /// bytes). Re-selects device_id for the copy and restores the caller's device
    /// (RAII guard), mirroring p2p_combine.cu:79-85. Used ONLY by: the parity test,
    /// the future M7 disk cache, and an explicit host/CLI caller. BIT-IDENTICAL to
    /// the result the old host-returning path produced (same doubles, same layout;
    /// §12).
    [[nodiscard]] F2BlockTensor to_host() const;

    // ---- Opaque CUDA payload (the DeviceBuffer<double> f2/vpair owners) ----
    struct Impl;                  // defined in cuda/device_f2_blocks_impl.cuh
    std::unique_ptr<Impl> impl;   // null iff empty()
};

/// H2D inverse of DeviceF2Blocks::to_host (M4.5 no-peer assembly transport): allocate
/// the resident DeviceBuffer<double> f2/vpair pair on `device_id`, cudaMemcpy the host
/// F2BlockTensor's f2/vpair up, and return the handle. Used ONLY by the no-peer G>=2
/// orchestrator arm to re-upload the host-assembled tensor so the PRIMARY return is
/// still a DeviceF2Blocks (the host bounce is the cross-card assembly transport, NOT a
/// forced output copy; documented limitation, architecture.md §11.4 no-peer tier).
/// CUDA-FREE decl; defined in cuda/device_f2_blocks.cu. Bit-faithful (raw byte copy).
[[nodiscard]] DeviceF2Blocks upload_f2_blocks_to_device(const F2BlockTensor& host, int device_id);

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_DEVICE_F2_BLOCKS_HPP
