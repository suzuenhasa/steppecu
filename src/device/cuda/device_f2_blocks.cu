// src/device/cuda/device_f2_blocks.cu — the CUDA side of DeviceF2Blocks: out-of-line
// special members, the device-pointer accessors, to_host() (the only D2H + host alloc
// in the device-resident pipeline), and upload_f2_blocks_to_device() (its H2D inverse).
// PRIVATE to steppe_device (a CUDA TU, architecture.md §4).
#include "device/cuda/device_f2_blocks_impl.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

#include "device/cuda/check.cuh"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/device_guard.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "steppe/fstats.hpp"

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
    DeviceGuard restore{prev};
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
    DeviceGuard restore{prev};
    STEPPE_CUDA_CHECK(cudaSetDevice(device_id));

    out.impl = std::make_unique<DeviceF2Blocks::Impl>();
    out.impl->f2 = DeviceBuffer<double>(total);
    out.impl->vpair = DeviceBuffer<double>(total);
    const std::size_t bytes = total * sizeof(double);
    RegisteredHostRegion pin_f2(host.f2.data(), bytes);
    RegisteredHostRegion pin_vp(host.vpair.data(), bytes);
    STEPPE_CUDA_CHECK(cudaMemcpy(out.impl->f2.data(), host.f2.data(), bytes,
                                 cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(out.impl->vpair.data(), host.vpair.data(), bytes,
                                 cudaMemcpyHostToDevice));
    return out;
}

}  // namespace steppe::device
