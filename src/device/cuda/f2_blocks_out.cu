// src/device/cuda/f2_blocks_out.cu — the tier-agnostic read-back of F2BlocksOut, plus
// the DiskF2Blocks special members (it owns a FILE* read handle). PRIVATE to
// steppe_device (a CUDA TU: the Resident arm needs a D2H; HostRam/Disk are plain
// host/file I/O). PARITY (architecture.md §12): to_host() / read_block_to_host() are
// BIT-IDENTICAL across all tiers — the D2H is a raw byte copy (the Resident arm pins
// its caller destinations via RegisteredHostRegion, a parity-neutral §11.4/§12
// data-movement lever, then byte-copies), the host copy is memcpy, the disk read is
// raw bytes; no recompute, no reorder, no precision change.
#include "device/f2_blocks_out.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK, STEPPE_CUDA_WARN (device-restore guard)
#include "device/cuda/device_f2_blocks_impl.cuh"  // DeviceF2Blocks::Impl (f2/vpair device pointers; pulls device_buffer.cuh)
#include "device/cuda/pinned_buffer.cuh"    // RegisteredHostRegion (pin the D2H; graceful degrade)
#include "device/f2_disk_format.hpp"        // F2DiskHeader, offsets

namespace steppe::device {

// ---- DiskF2Blocks special members (owns the read FILE*) -------------------
DiskF2Blocks::DiskF2Blocks() = default;
DiskF2Blocks::~DiskF2Blocks() {
    if (read_handle) { std::fclose(read_handle); read_handle = nullptr; }
}
DiskF2Blocks::DiskF2Blocks(DiskF2Blocks&& o) noexcept
    : path(std::move(o.path)), P(o.P), n_block(o.n_block),
      block_sizes(std::move(o.block_sizes)),
      read_handle(std::exchange(o.read_handle, nullptr)) {}
DiskF2Blocks& DiskF2Blocks::operator=(DiskF2Blocks&& o) noexcept {
    if (this != &o) {
        if (read_handle) std::fclose(read_handle);
        path = std::move(o.path);
        P = o.P;
        n_block = o.n_block;
        block_sizes = std::move(o.block_sizes);
        read_handle = std::exchange(o.read_handle, nullptr);
    }
    return *this;
}

namespace {
// pread the whole slab at offset, looping over partial reads. Throws on short read.
void pread_all(std::FILE* f, void* buf, std::size_t bytes, std::uint64_t offset,
               const char* what) {
    // 64-bit file offsets: `long` is 64-bit on the LP64 Linux/CUDA target, so
    // std::fseek covers the multi-GB f2/vpair regions. A static_assert pins the
    // assumption (a 32-bit long would silently wrap a large offset).
    static_assert(sizeof(long) >= 8, "F2BlocksOut(Disk) needs 64-bit file offsets");
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0)
        throw std::runtime_error(std::string("F2BlocksOut(Disk): fseek(") + what + ") failed");
    const std::size_t got = std::fread(buf, 1, bytes, f);
    if (got != bytes)
        throw std::runtime_error(std::string("F2BlocksOut(Disk): short read(") + what + ")");
}

// Reconstruct the disk header from the descriptor's shape (the on-disk layout is fully
// determined by P/n_block; we do not re-parse the file's 64-byte header for offsets).
F2DiskHeader disk_header(const DiskF2Blocks& d) {
    F2DiskHeader h{};
    h.P = d.P;
    h.n_block = d.n_block;
    h.f2_offset = sizeof(F2DiskHeader);
    const std::uint64_t region =
        static_cast<std::uint64_t>(d.P) * static_cast<std::uint64_t>(d.P) *
        static_cast<std::uint64_t>(d.n_block < 0 ? 0 : d.n_block) * sizeof(double);
    h.vpair_offset = h.f2_offset + region;
    h.block_sizes_offset = h.vpair_offset + region;
    return h;
}
}  // namespace

// ---- F2BlocksOut::read_block_to_host — the FIT's tile reader (tier-agnostic) ----
void F2BlocksOut::read_block_to_host(int b, double* f2_slab_out, double* vpair_slab_out) const {
    const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t bytes = slab * sizeof(double);
    if (slab == 0) return;

    switch (tier) {
        case OutputTier::Resident: {
            const double* f2_dev = resident.f2_device();
            const double* vp_dev = resident.vpair_device();
            if (!f2_dev || !vp_dev) return;
            int prev = 0;
            STEPPE_CUDA_CHECK(cudaGetDevice(&prev));
            // Dtor-must-not-throw, so the restore can't go through the throwing
            // STEPPE_CUDA_CHECK; route it through the non-throwing STEPPE_CUDA_WARN
            // so a failed restore logs one diagnostic line (debug) and yields its
            // status instead of vanishing — cudaSetDevice can surface a prior async
            // launch error (CUDA-13 Device-Management). The [[nodiscard]] return is
            // (void)-discarded for the -Werror build; happy path is byte-identical.
            struct G { int d; ~G() { (void)STEPPE_CUDA_WARN(cudaSetDevice(d)); } } restore{prev};
            STEPPE_CUDA_CHECK(cudaSetDevice(resident.device_id));
            const std::size_t off = slab * static_cast<std::size_t>(b);
            // PIN the caller's pageable D2H destinations for the copy window
            // (RegisteredHostRegion, graceful pageable degrade — never throws,
            // pinned_buffer.cuh:159-176), EXACTLY like to_host's D2H at
            // device_f2_blocks.cu:55-56. cudaHostRegister page-locks the range in
            // place (CUDA 13.x Runtime API: "Page-locks the memory range ... and
            // maps it for the device(s) ... to automatically accelerate calls to
            // functions such as cudaMemcpy()"); it changes only the page state, not
            // the bytes, so this is PARITY-NEUTRAL (§12 — a data-movement lever
            // only). The registrations stay alive through both synchronous memcpys
            // and RAII-unregister at scope exit.
            RegisteredHostRegion pin_f2(f2_slab_out, bytes);
            RegisteredHostRegion pin_vp(vpair_slab_out, bytes);
            STEPPE_CUDA_CHECK(cudaMemcpy(f2_slab_out, f2_dev + off, bytes,
                                         cudaMemcpyDeviceToHost));
            STEPPE_CUDA_CHECK(cudaMemcpy(vpair_slab_out, vp_dev + off, bytes,
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
            pread_all(disk.read_handle, f2_slab_out, bytes, f2_block_offset(h, b), "f2");
            pread_all(disk.read_handle, vpair_slab_out, bytes, vpair_block_offset(h, b), "vpair");
            break;
        }
    }
}

// ---- F2BlocksOut::to_host — materialize the WHOLE result (bit-identical across tiers) ----
F2BlockTensor F2BlocksOut::to_host() const {
    switch (tier) {
        case OutputTier::Resident:
            // The EXISTING ONLY D2H of the device-resident handle (a tautology vs the
            // single-GPU reference — same buffers, same layout; §12).
            return resident.to_host();

        case OutputTier::HostRam: {
            // A copy of the streamed host tensor (already block-major i + P·j + P·P·b).
            F2BlockTensor out;
            out.P = P;
            out.n_block = (n_block < 0 ? 0 : n_block);
            out.block_sizes = block_sizes;
            out.f2 = host.f2;
            out.vpair = host.vpair;
            return out;
        }

        case OutputTier::Disk: {
            // Read every block back via pread into one contiguous F2BlockTensor. The
            // on-disk f2/vpair regions are ALREADY block-major i + P·j + P·P·b, so each
            // region is a single contiguous read into the matching vector.
            F2BlockTensor out;
            out.P = P;
            out.n_block = (n_block < 0 ? 0 : n_block);
            out.block_sizes = block_sizes;
            const std::size_t total =
                static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                static_cast<std::size_t>(out.n_block);
            out.f2.assign(total, 0.0);
            out.vpair.assign(total, 0.0);
            if (total == 0) return out;
            if (!disk.read_handle)
                throw std::runtime_error("F2BlocksOut::to_host: Disk tier has no open read handle");
            const F2DiskHeader h = disk_header(disk);
            pread_all(disk.read_handle, out.f2.data(), total * sizeof(double),
                      h.f2_offset, "f2-region");
            pread_all(disk.read_handle, out.vpair.data(), total * sizeof(double),
                      h.vpair_offset, "vpair-region");
            return out;
        }
    }
    return F2BlockTensor{};  // unreachable (all enumerators handled)
}

}  // namespace steppe::device
