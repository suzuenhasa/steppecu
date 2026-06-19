// src/device/device_partial.hpp  — CUDA-FREE opaque handle to a per-device
// resident f2/Vpair partial (M4.5 device-resident combine). Mirrors the
// p2p_combine.hpp CUDA-free-decl pattern: this header names NO CUDA type; the
// DeviceBuffer<double> owners live in the Impl defined in cuda/device_partial.cu.
#ifndef STEPPE_DEVICE_DEVICE_PARTIAL_HPP
#define STEPPE_DEVICE_DEVICE_PARTIAL_HPP

#include <cstddef>
#include <memory>
#include <vector>

namespace steppe::device {

/// Move-only, OPAQUE owner of ONE device's f2/Vpair partial left RESIDENT on the
/// device that computed it (NO D2H, NO free) — the M4.5 device-resident combine
/// input. CUDA-FREE at this seam: the DeviceBuffer<double> f2/vpair owners + the
/// resident device pointers live in `Impl`, defined only in cuda/device_partial.cu
/// (the same CUDA-free-decl / CUDA-def split combine_f2_partials_p2p uses). The
/// shape fields are plain host scalars so the CUDA-free orchestrator
/// (compute_f2_blocks_multigpu, steppe::core) can hold and forward the handle
/// without seeing <cuda_runtime.h>.
///
/// LIFETIME (the jthread-survival + free-after-combine contract): the
/// underlying DeviceBuffer<double> pair was cudaMalloc'd on device `device_id` by
/// worker g; this handle MOVES out of the worker into the returned vector (it must
/// survive the jthread join) and frees in its destructor AFTER the combine has
/// consumed it (cudaFree is pointer-device-aware — it frees on the buffer's owning
/// device from any current-device context).
class DevicePartial {
public:
    DevicePartial();                                   // empty / moved-from / empty-shard handle
    ~DevicePartial();                                  // frees the resident DeviceBuffer pair
    DevicePartial(DevicePartial&&) noexcept;           // move-only
    DevicePartial& operator=(DevicePartial&&) noexcept;
    DevicePartial(const DevicePartial&) = delete;
    DevicePartial& operator=(const DevicePartial&) = delete;

    // ---- Shape (plain host scalars; CUDA-free) ----
    int P = 0;             ///< population count (leading dim of every slab).
    int n_block_local = 0; ///< blocks this device owns (== shard b1-b0); 0 == empty shard.
    int b0 = 0;            ///< this partial's global block placement offset (== shard.b0).
    int device_id = -1;    ///< the physical CUDA ordinal the buffers are resident on.

    // ---- Per-block SNP counts (host int, placed host-side; mirrors the existing
    // F2BlockTensor.block_sizes the host combine copies). Sized n_block_local. ----
    std::vector<int> block_sizes;

    /// True for an empty shard (n_block_local == 0): no resident buffers, the
    /// combine places only block_sizes (a no-op loop) for it.
    [[nodiscard]] bool empty() const noexcept { return n_block_local <= 0; }

    // ---- Opaque CUDA payload (the DeviceBuffer<double> f2/vpair owners) ----
    struct Impl;                                       // defined in cuda/device_partial_impl.cuh
    std::unique_ptr<Impl> impl;                        // null iff empty()
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_DEVICE_PARTIAL_HPP
