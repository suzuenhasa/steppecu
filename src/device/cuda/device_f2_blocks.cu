// src/device/cuda/device_f2_blocks.cu — the CUDA side of DeviceF2Blocks. Out-of-line
// special members (so unique_ptr<Impl> has a complete Impl at instantiation), the
// device-pointer accessors, to_host() — the ONLY D2H + host alloc in the
// device-resident pipeline — and upload_f2_blocks_to_device() (the H2D inverse for the
// no-peer assembly transport). PRIVATE to steppe_device (a CUDA TU, architecture.md §4).
#include "device/cuda/device_f2_blocks_impl.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<double>
#include "device/cuda/pinned_buffer.cuh"    // RegisteredHostRegion (pin the D2H; graceful degrade)
#include "steppe/fstats.hpp"                // F2BlockTensor

namespace steppe::device {

DeviceF2Blocks::DeviceF2Blocks() = default;
DeviceF2Blocks::~DeviceF2Blocks() = default;
DeviceF2Blocks::DeviceF2Blocks(DeviceF2Blocks&&) noexcept = default;
DeviceF2Blocks& DeviceF2Blocks::operator=(DeviceF2Blocks&&) noexcept = default;

const double* DeviceF2Blocks::f2_device() const noexcept {
    return impl ? impl->f2.data() : nullptr;
}
const double* DeviceF2Blocks::vpair_device() const noexcept {
    return impl ? impl->vpair.data() : nullptr;
}

// THE ONLY D2H + host alloc in the device-resident pipeline (opt-in). Bit-identical
// to the result the old host-returning compute_f2_blocks produced (same doubles,
// same [P×P×n_block] layout; §12). PINS the host destinations for the D2H window
// (RegisteredHostRegion, graceful pageable degrade), EXACTLY like p2p_combine.cu
// :185-186 — parity-neutral (pinned vs pageable moves the same bytes). Sets device_id
// current for the copy and restores the caller's device (RAII guard), mirroring
// p2p_combine.cu:79-85, because cudaMemcpy targets the buffers' device.
F2BlockTensor DeviceF2Blocks::to_host() const {
    F2BlockTensor out;
    out.P = P;
    out.n_block = (n_block < 0 ? 0 : n_block);
    out.block_sizes = block_sizes;
    const std::size_t total = size();
    out.f2.resize(total);
    out.vpair.resize(total);
    if (total == 0 || !impl) return out;

    int prev = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&prev));
    struct G { int d; ~G() { (void)cudaSetDevice(d); } } restore{prev};
    STEPPE_CUDA_CHECK(cudaSetDevice(device_id));

    const std::size_t bytes = total * sizeof(double);
    RegisteredHostRegion pin_f2(out.f2.data(), bytes);
    RegisteredHostRegion pin_vp(out.vpair.data(), bytes);
    STEPPE_CUDA_CHECK(cudaMemcpy(out.f2.data(), impl->f2.data(), bytes,
                                 cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(out.vpair.data(), impl->vpair.data(), bytes,
                                 cudaMemcpyDeviceToHost));
    return out;
}

// H2D inverse of to_host (M4.5 no-peer assembly transport): allocate the resident
// f2/vpair pair on device_id and cudaMemcpy the host tensor up. Raw byte copy =>
// bit-faithful (preserves −0.0). PINS the host SOURCES for the H2D window
// (RegisteredHostRegion, graceful pageable degrade), mirroring to_host's D2H dest
// pinning at :55-56 — parity-neutral (pinned vs pageable moves the same bytes).
// Re-selects device_id for the alloc+copy (DeviceBuffer cudaMalloc allocates on the
// current device) and restores the caller's device.
DeviceF2Blocks upload_f2_blocks_to_device(const F2BlockTensor& host, int device_id) {
    DeviceF2Blocks out;
    out.P = host.P;
    out.n_block = (host.n_block < 0 ? 0 : host.n_block);
    out.device_id = device_id;
    out.block_sizes = host.block_sizes;
    const std::size_t total = out.size();
    if (total == 0) return out;

    int prev = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&prev));
    struct G { int d; ~G() { (void)cudaSetDevice(d); } } restore{prev};
    STEPPE_CUDA_CHECK(cudaSetDevice(device_id));

    out.impl = std::make_unique<DeviceF2Blocks::Impl>();
    out.impl->f2 = DeviceBuffer<double>(total);
    out.impl->vpair = DeviceBuffer<double>(total);
    const std::size_t bytes = total * sizeof(double);
    // PIN the H2D source pages for the copy window (RegisteredHostRegion, graceful
    // pageable degrade — never throws, pinned_buffer.cuh:159-176), mirroring the D2H
    // dest pinning in to_host at :55-56 and resolving the documented direction
    // asymmetry. cudaHostRegister takes void* and page-locks in place WITHOUT writing
    // the bytes, so a const H2D source registers soundly via the ctor's const_cast
    // (pinned_buffer.cuh:159-163); pinned vs pageable moves the identical bytes, so
    // this is PARITY-NEUTRAL (§12). The registrations RAII-unregister at scope exit.
    RegisteredHostRegion pin_f2(host.f2.data(), bytes);
    RegisteredHostRegion pin_vp(host.vpair.data(), bytes);
    STEPPE_CUDA_CHECK(cudaMemcpy(out.impl->f2.data(), host.f2.data(), bytes,
                                 cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(out.impl->vpair.data(), host.vpair.data(), bytes,
                                 cudaMemcpyHostToDevice));
    return out;
}

}  // namespace steppe::device
