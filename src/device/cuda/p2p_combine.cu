// src/device/cuda/p2p_combine.cu
//
// The opt-in device-resident P2P f2 combine: it assembles the per-device resident
// f2/vpair partials into one full tensor on a root GPU over the peer-to-peer link,
// with no host bounce. CUDA-private to steppe_device — it includes a CUDA-free
// declaration header so the CUDA-free core entry point can reach it.
//
// Reference: docs/reference/src_device_cuda_p2p_combine.cu.md
#include "device/p2p_combine.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <vector>

#include "device/cuda/check.cuh"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/device_guard.cuh"
#include "device/cuda/device_partial_impl.cuh"
#include "device/cuda/device_f2_blocks_impl.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "device/cuda/stream.hpp"
#include "steppe/fstats.hpp"
#include "device/shard_plan.hpp"
#include "core/fstats/f2_partials_validate.hpp"

namespace steppe::device {

namespace {

// shared fixed-order placement loop + peer-access handling — reference §3, §4
void place_partials_into(double* dst_f2_base, double* dst_vpair_base,
                         std::span<DevicePartial> partials, std::size_t slab,
                         int root_device_id, cudaStream_t root_stream,
                         std::vector<int>& out_block_sizes) {
    for (std::size_t g = 0; g < partials.size(); ++g) {
        DevicePartial& part = partials[g];

        for (int lb = 0; lb < part.n_block_local; ++lb) {
            out_block_sizes[static_cast<std::size_t>(part.b0 + lb)] =
                part.block_sizes[static_cast<std::size_t>(lb)];
        }
        if (part.empty()) continue;

        const std::size_t part_elems = slab * static_cast<std::size_t>(part.n_block_local);
        const std::size_t part_bytes = part_elems * sizeof(double);
        const std::size_t dst_off = slab * static_cast<std::size_t>(part.b0);

        double* dst_f2 = dst_f2_base + dst_off;
        double* dst_vpair = dst_vpair_base + dst_off;
        const double* src_f2 = part.impl->f2.data();
        const double* src_vpair = part.impl->vpair.data();

        if (part.device_id == root_device_id) {
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst_f2, src_f2, part_bytes,
                                              cudaMemcpyDeviceToDevice, root_stream));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst_vpair, src_vpair, part_bytes,
                                              cudaMemcpyDeviceToDevice, root_stream));
        } else {
            (void)STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(part.device_id, 0));
            (void)cudaGetLastError();
            STEPPE_CUDA_CHECK(cudaMemcpyPeerAsync(dst_f2, root_device_id,
                                                  src_f2, part.device_id, part_bytes, root_stream));
            STEPPE_CUDA_CHECK(cudaMemcpyPeerAsync(dst_vpair, root_device_id,
                                                  src_vpair, part.device_id, part_bytes, root_stream));
        }
    }
}

}  // namespace

// host-returning combine entry (final D2H + pinned copy) — reference §2, §5, §6, §7
F2BlockTensor combine_f2_partials_resident(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id) {
    steppe::core::validate_resident_partials(
        "steppe::device::combine_f2_partials_resident", partials, shards, P, n_block_full);

    int prev_device = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&prev_device));
    DeviceGuard restore{prev_device};
    STEPPE_CUDA_CHECK(cudaSetDevice(root_device_id));

    Stream root_stream_owner;
    const cudaStream_t root_stream = root_stream_owner.get();

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total = slab * static_cast<std::size_t>(n_block_full);

    DeviceBuffer<double> result_f2(total);
    DeviceBuffer<double> result_vpair(total);

    F2BlockTensor out;
    out.P = P;
    out.n_block = n_block_full;
    out.f2.resize(total);
    out.vpair.resize(total);
    out.block_sizes.assign(static_cast<std::size_t>(n_block_full), 0);

    place_partials_into(result_f2.data(), result_vpair.data(), partials, slab,
                        root_device_id, root_stream, out.block_sizes);

    STEPPE_CUDA_CHECK(cudaStreamSynchronize(root_stream));
    if (total > 0) {
        const std::size_t bytes = total * sizeof(double);
        RegisteredHostRegion pin_f2(out.f2.data(), bytes);
        RegisteredHostRegion pin_vpair(out.vpair.data(), bytes);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.f2.data(), result_f2.data(),
                                          bytes, cudaMemcpyDeviceToHost, root_stream));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.vpair.data(), result_vpair.data(),
                                          bytes, cudaMemcpyDeviceToHost, root_stream));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(root_stream));
    }
    return out;
}

// device-resident combine entry (no final D2H) — reference §2, §5, §6
DeviceF2Blocks combine_f2_partials_resident_device(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id) {
    steppe::core::validate_resident_partials(
        "steppe::device::combine_f2_partials_resident_device", partials, shards, P, n_block_full);

    int prev_device = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&prev_device));
    DeviceGuard restore{prev_device};
    STEPPE_CUDA_CHECK(cudaSetDevice(root_device_id));

    Stream root_stream_owner;
    const cudaStream_t root_stream = root_stream_owner.get();

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total = slab * static_cast<std::size_t>(n_block_full);

    DeviceF2Blocks out;
    out.P = P;
    out.n_block = n_block_full;
    out.device_id = root_device_id;
    out.block_sizes.assign(static_cast<std::size_t>(n_block_full), 0);
    out.impl = std::make_unique<DeviceF2Blocks::Impl>();
    out.impl->f2 = DeviceBuffer<double>(total);
    out.impl->vpair = DeviceBuffer<double>(total);
    double* result_f2 = out.impl->f2.data();
    double* result_vpair = out.impl->vpair.data();

    place_partials_into(result_f2, result_vpair, partials, slab,
                        root_device_id, root_stream, out.block_sizes);

    STEPPE_CUDA_CHECK(cudaStreamSynchronize(root_stream));
    return out;
}

}  // namespace steppe::device
