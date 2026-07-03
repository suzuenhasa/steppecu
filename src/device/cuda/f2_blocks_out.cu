// src/device/cuda/f2_blocks_out.cu — tier-agnostic read-back of F2BlocksOut (Resident /
// HostRam / Disk) plus the DiskF2Blocks file-handle cleanup. A CUDA TU because only the
// Resident tier's device-to-host copy needs CUDA; HostRam/Disk are plain host/file I/O.
// Reference: docs/reference/src_device_cuda_f2_blocks_out.cu.md
#include "device/f2_blocks_out.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "core/internal/host_device.hpp"
#include "core/internal/log.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/device_f2_blocks_impl.cuh"
#include "device/cuda/device_guard.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "device/f2_disk_format.hpp"

namespace steppe::device {

// FileCloser — the one close site for the disk read handle — reference §6
void FileCloser::operator()(std::FILE* f) const noexcept {
    if (!f) return;
    [[maybe_unused]] const int status = std::fclose(f);
    if (status != 0)
        STEPPE_LOG_WARN("std::fclose (F2BlocksOut(Disk) read handle teardown) failed");
}

namespace {
// pread_all — read a whole slab or region, or fail loudly — reference §5
void pread_all(std::FILE* f, void* buf, std::size_t bytes, std::uint64_t offset,
               const char* region) {
    static_assert(sizeof(long) >= 8, "F2BlocksOut(Disk) needs 64-bit file offsets");
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0)
        throw std::runtime_error(std::string("F2BlocksOut(Disk): fseek(") + region + ") failed");
    const std::size_t got = std::fread(buf, 1, bytes, f);
    if (got != bytes)
        throw std::runtime_error(std::string("F2BlocksOut(Disk): short read(") + region + ")");
}

// disk_header — rebuild the offsets from the shape, not the file — reference §5
F2DiskHeader disk_header(const DiskF2Blocks& d) {
    F2DiskHeader h{};
    h.P = d.P;
    h.n_block = d.n_block;
    h.f2_offset = sizeof(F2DiskHeader);
    const std::uint64_t region =
        static_cast<std::uint64_t>(slab_elems(d.P)) *
        static_cast<std::uint64_t>(d.n_block < 0 ? 0 : d.n_block) * sizeof(double);
    h.vpair_offset = h.f2_offset + region;
    h.block_sizes_offset = h.vpair_offset + region;
    return h;
}
}  // namespace

// read_block_to_host — the fit's one-block tile reader — reference §3
void F2BlocksOut::read_block_to_host(int b, double* f2_slab_out, double* vpair_slab_out) const {
    STEPPE_ASSERT(b >= 0 && b < n_block, "read_block_to_host: block index out of range");
    const std::size_t slab = slab_elems(P);
    const std::size_t bytes = slab * sizeof(double);
    if (slab == 0) return;

    switch (tier) {
        case OutputTier::Resident: {
            const double* f2_dev = resident.f2_device();
            const double* vpair_dev = resident.vpair_device();
            if (!f2_dev || !vpair_dev) return;
            int prev = 0;
            STEPPE_CUDA_CHECK(cudaGetDevice(&prev));
            DeviceGuard restore{prev};
            STEPPE_CUDA_CHECK(cudaSetDevice(resident.device_id));
            const std::size_t off = slab * static_cast<std::size_t>(b);
            RegisteredHostRegion pin_f2(f2_slab_out, bytes);
            RegisteredHostRegion pin_vp(vpair_slab_out, bytes);
            STEPPE_CUDA_CHECK(cudaMemcpy(f2_slab_out, f2_dev + off, bytes,
                                         cudaMemcpyDeviceToHost));
            STEPPE_CUDA_CHECK(cudaMemcpy(vpair_slab_out, vpair_dev + off, bytes,
                                         cudaMemcpyDeviceToHost));
            break;
        }
        case OutputTier::HostRam: {
            const std::size_t off = slab * static_cast<std::size_t>(b);
            std::memcpy(f2_slab_out, host.f2.data() + off, bytes);
            std::memcpy(vpair_slab_out, host.vpair.data() + off, bytes);
            break;
        }
        case OutputTier::Disk: {
            if (!disk.read_handle)
                throw std::runtime_error("F2BlocksOut::read_block_to_host: Disk tier has no "
                                         "open read handle");
            const F2DiskHeader h = disk_header(disk);
            pread_all(disk.read_handle.get(), f2_slab_out, bytes, f2_block_offset(h, b), "f2");
            pread_all(disk.read_handle.get(), vpair_slab_out, bytes, vpair_block_offset(h, b), "vpair");
            break;
        }
    }
}

// to_host — materialize the whole result at once — reference §4
F2BlockTensor F2BlocksOut::to_host() const {
    switch (tier) {
        case OutputTier::Resident:
            return resident.to_host();

        case OutputTier::HostRam: {
            F2BlockTensor out;
            out.P = P;
            out.n_block = (n_block < 0 ? 0 : n_block);
            out.block_sizes = block_sizes;
            out.f2 = host.f2;
            out.vpair = host.vpair;
            return out;
        }

        case OutputTier::Disk: {
            F2BlockTensor out;
            out.P = P;
            out.n_block = (n_block < 0 ? 0 : n_block);
            out.block_sizes = block_sizes;
            const std::size_t total =
                slab_elems(P) * static_cast<std::size_t>(out.n_block);
            out.f2.assign(total, 0.0);
            out.vpair.assign(total, 0.0);
            if (total == 0) return out;
            if (!disk.read_handle)
                throw std::runtime_error("F2BlocksOut::to_host: Disk tier has no open read handle");
            const F2DiskHeader h = disk_header(disk);
            pread_all(disk.read_handle.get(), out.f2.data(), total * sizeof(double),
                      h.f2_offset, "f2-region");
            pread_all(disk.read_handle.get(), out.vpair.data(), total * sizeof(double),
                      h.vpair_offset, "vpair-region");
            return out;
        }
    }
    return F2BlockTensor{};
}

}  // namespace steppe::device
