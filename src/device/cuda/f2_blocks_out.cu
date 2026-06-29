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

#include "core/internal/host_device.hpp"   // STEPPE_ASSERT (debug-only block-index bounds check)
#include "core/internal/log.hpp"           // STEPPE_LOG_WARN (the one teardown-warning sink)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK (fault check on the D2H copies)
#include "device/cuda/device_f2_blocks_impl.cuh"  // DeviceF2Blocks::Impl (f2/vpair device pointers; pulls device_buffer.cuh)
#include "device/cuda/device_guard.cuh"     // DeviceGuard (shared scoped device-restore RAII, §3.3)
#include "device/cuda/pinned_buffer.cuh"    // RegisteredHostRegion (pin the D2H; graceful degrade)
#include "device/f2_disk_format.hpp"        // F2DiskHeader, offsets

namespace steppe::device {

// ---- FileCloser — the ONE close site for the disk read handle -------------
// The descriptor's std::unique_ptr<std::FILE, FileCloser> now supplies move-only +
// null-on-move + the freeing dtor for DiskF2Blocks (all special members =default in the
// header; ~18 lines of hand-rolled rule-of-five deleted, group-16 16.5). Folding both
// former close sites (the old dtor + the move-assign close-old) into this one deleter
// also lets the previously-discarded std::fclose status emit ONE teardown warn here
// (16.1). NDEBUG: STEPPE_LOG_WARN compiles to (void)0 AND does not evaluate its args, so
// the std::fclose call MUST stay outside the macro (its close is the load-bearing side
// effect); the `status` local is consumed only by the (release-stripped) warn, so it is
// marked [[maybe_unused]] to stay -Werror=unused-variable clean under NDEBUG.
void FileCloser::operator()(std::FILE* f) const noexcept {
    if (!f) return;  // defensive (unique_ptr only invokes the deleter on a non-null get())
    [[maybe_unused]] const int status = std::fclose(f);  // 0 on success, EOF on failure
    if (status != 0)
        STEPPE_LOG_WARN("std::fclose (F2BlocksOut(Disk) read handle teardown) failed");
}

namespace {
// pread the whole slab at offset, looping over partial reads. Throws on short read.
void pread_all(std::FILE* f, void* buf, std::size_t bytes, std::uint64_t offset,
               const char* region) {
    // 64-bit file offsets: `long` is 64-bit on the LP64 Linux/CUDA target, so
    // std::fseek covers the multi-GB f2/vpair regions. A static_assert pins the
    // assumption (a 32-bit long would silently wrap a large offset).
    static_assert(sizeof(long) >= 8, "F2BlocksOut(Disk) needs 64-bit file offsets");
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0)
        throw std::runtime_error(std::string("F2BlocksOut(Disk): fseek(") + region + ") failed");
    const std::size_t got = std::fread(buf, 1, bytes, f);
    if (got != bytes)
        throw std::runtime_error(std::string("F2BlocksOut(Disk): short read(") + region + ")");
}

// Reconstruct the disk header from the descriptor's shape (the on-disk layout is fully
// determined by P/n_block; we do not re-parse the file's 64-byte header for offsets).
F2DiskHeader disk_header(const DiskF2Blocks& d) {
    F2DiskHeader h{};
    h.P = d.P;
    h.n_block = d.n_block;
    h.f2_offset = sizeof(F2DiskHeader);
    // P² (single-homed slab shape, 5.3) × n_block × 8; slab_elems widens P before the
    // multiply and returns std::size_t (64-bit on the LP64 target), so the whole
    // product is 64-bit — no int overflow before widening (the region reaches multi-GB).
    const std::uint64_t region =
        static_cast<std::uint64_t>(slab_elems(d.P)) *
        static_cast<std::uint64_t>(d.n_block < 0 ? 0 : d.n_block) * sizeof(double);
    h.vpair_offset = h.f2_offset + region;
    h.block_sizes_offset = h.vpair_offset + region;
    return h;
}
}  // namespace

// ---- F2BlocksOut::read_block_to_host — the FIT's tile reader (tier-agnostic) ----
void F2BlocksOut::read_block_to_host(int b, double* f2_slab_out, double* vpair_slab_out) const {
    // Debug-only bounds contract: every tier computes off = slab·(size_t)b, so a negative
    // b casts to a huge size_t (OOB D2H / host read / pread) and b >= n_block walks past
    // the last slab. Compiles out under NDEBUG (b unevaluated) — the trust-based contract
    // and the gate build are byte-identical.
    STEPPE_ASSERT(b >= 0 && b < n_block, "read_block_to_host: block index out of range");
    const std::size_t slab = slab_elems(P);  // P² per-block slab (single-homed shape, 5.3)
    const std::size_t bytes = slab * sizeof(double);
    if (slab == 0) return;

    switch (tier) {
        case OutputTier::Resident: {
            const double* f2_dev = resident.f2_device();
            const double* vpair_dev = resident.vpair_device();
            if (!f2_dev || !vpair_dev) return;
            int prev = 0;
            STEPPE_CUDA_CHECK(cudaGetDevice(&prev));
            // Restore the caller's device on scope exit via the shared DeviceGuard
            // (device_guard.cuh; standard §3.3 single device-restore helper). The dtor
            // must not throw, so its restore routes through the non-throwing
            // STEPPE_CUDA_WARN — a failed restore logs one diagnostic line (debug) and
            // yields its status instead of vanishing; cudaSetDevice can surface a prior
            // async launch error (CUDA 13.x Device Management). Happy path byte-identical.
            DeviceGuard restore{prev};
            STEPPE_CUDA_CHECK(cudaSetDevice(resident.device_id));
            const std::size_t off = slab * static_cast<std::size_t>(b);
            // PIN the caller's pageable D2H destinations for the copy window
            // (RegisteredHostRegion, graceful pageable degrade — never throws,
            // pinned_buffer.cuh:159-176), EXACTLY like DeviceF2Blocks::to_host's
            // D2H (device_f2_blocks.cu). cudaHostRegister page-locks the range in
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
                slab_elems(P) * static_cast<std::size_t>(out.n_block);  // P²·n_block (5.3)
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
    return F2BlockTensor{};  // unreachable (all enumerators handled)
}

}  // namespace steppe::device
