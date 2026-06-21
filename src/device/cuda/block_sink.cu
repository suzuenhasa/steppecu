// src/device/cuda/block_sink.cu
//
// The M5 block-spill SINK impls: HostRamSink (TIER 1) and DiskSink (TIER 2). Both use a
// SMALL PERSISTENT pinned staging RING (kStreamStagingSlots slots, pinned ONCE in begin()
// via PinnedBuffer / cudaHostAlloc, reused every spill_block — the 94c6d8e per-call
// cudaHostRegister serialization bug is designed out) + a BACKGROUND WRITER THREAD.
// PRIVATE to steppe_device (a CUDA TU, architecture.md §4).
//
// THE TRIPLE-BUFFER OVERLAP (§5): spill_block(b) ACQUIRES a free pinned slot (blocking
// only under backpressure when all slots are still in the writer's hands), issues the
// async D2H of block b into it on `stream`, records the slot's completion event, and
// ENQUEUES the slot to the writer thread. The writer thread waits the event (so the slab
// is fully drained — the frozen per-slot happens-before) then does the host-copy /
// pwrite, and returns the slot to the free pool. So the slow host-copy / pwrite runs
// CONCURRENTLY with the GPU's compute + D2H of the NEXT chunk — the spill OVERLAPS
// compute (at large P the GEMM dominates, so the spill hides). PARITY-NEUTRAL: the D2H is
// a raw byte copy, the host copy is memcpy, the disk write is raw bytes — no recompute,
// no reorder, no precision change (architecture.md §12). The per-block slabs land at
// disjoint offsets (block-major), so the writer's out-of-order drain is race-free.
#include "device/cuda/block_sink.cuh"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "device/cuda/check.cuh"        // STEPPE_CUDA_CHECK, CudaError
#include "device/f2_blocks_out.hpp"     // DiskF2Blocks (the finalized Disk descriptor)
#include "device/f2_disk_format.hpp"    // F2DiskHeader, kF2DiskMagic, kF2DiskVersion, kF2DiskDtypeFp64, offsets
#include "core/internal/log.hpp"        // STEPPE_LOG_WARN (teardown warnings)

namespace steppe::device {

/// POSIX file mode for the Disk-tier on-disk f2_blocks cache: rw-r--r-- (owner
/// read/write, group/other read). Named so the magic permission bits are not a bare
/// octal literal at the open() call (group-5 5.1).
namespace { constexpr int kCacheFileMode = 0644; }

// ===========================================================================
// HostRamSink (TIER 1)
// ===========================================================================

void HostRamSink::begin(int P, int n_block, const std::vector<int>& block_sizes) {
    P_ = P;
    n_block_ = n_block;
    slab_ = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    host_.P = P;
    host_.n_block = (n_block < 0 ? 0 : n_block);
    host_.block_sizes = block_sizes;
    const std::size_t total = slab_ * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
    host_.f2.assign(total, 0.0);
    host_.vpair.assign(total, 0.0);
    if (slab_ == 0 || n_block <= 0) return;
    // Start the shared pinned ring + writer ([7.1]). The drain callback is the ONLY
    // per-tier difference: std::memcpy the (drained) slot into host_.f2/vpair at the
    // block's base. PARITY-NEUTRAL: a raw byte copy, same bytes (§12). slab_ is captured
    // by value (the slab never changes after begin); host_ by reference (it is the dst).
    const std::size_t slab = slab_;
    F2BlockTensor& host = host_;
    ring_.begin(
        slab_,
        [slab, &host](SinkSlot& s) {
            const std::size_t dst = slab * static_cast<std::size_t>(s.block);
            std::memcpy(host.f2.data() + dst, s.f2.data(), slab * sizeof(double));
            std::memcpy(host.vpair.data() + dst, s.vpair.data(), slab * sizeof(double));
        },
        "HostRamSink writer");
}

void HostRamSink::spill_block(int b, const double* f2_dev, const double* vpair_dev,
                              std::size_t slab_elems, cudaStream_t stream) {
    (void)slab_elems;  // == slab_ by construction (P²); the caller passes P*P.
    ring_.spill_block(b, f2_dev, vpair_dev, stream);  // shared ([7.1])
}

void HostRamSink::finish() {
    ring_.stop_and_join();  // shared ([7.1]); idempotent / no-op for an empty ring.
    if (ring_.writer_failed())
        throw std::runtime_error("HostRamSink: background writer failed: " +
                                 ring_.writer_error());
}

// ===========================================================================
// DiskSink (TIER 2)
// ===========================================================================

namespace {
// pwrite the whole buffer at offset, looping over partial writes. Throws on error.
void pwrite_all(int fd, const void* buf, std::size_t bytes, std::uint64_t offset,
                const char* what) {
    const char* p = static_cast<const char*>(buf);
    std::size_t left = bytes;
    std::uint64_t off = offset;
    while (left > 0) {
        const ssize_t w = ::pwrite(fd, p, left, static_cast<off_t>(off));
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("DiskSink: pwrite(") + what +
                                     ") failed: " + std::strerror(errno));
        }
        if (w == 0)
            throw std::runtime_error(std::string("DiskSink: pwrite(") + what +
                                     ") wrote 0 bytes");
        p += w;
        left -= static_cast<std::size_t>(w);
        off += static_cast<std::uint64_t>(w);
    }
}
}  // namespace

void DiskSink::begin(int P, int n_block, const std::vector<int>& block_sizes) {
    P_ = P;
    n_block_ = n_block;
    block_sizes_ = block_sizes;
    slab_ = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    slab_bytes_ = slab_ * sizeof(double);
    const std::uint64_t region_bytes =
        slab_bytes_ * static_cast<std::uint64_t>(n_block < 0 ? 0 : n_block);
    f2_region_ = sizeof(F2DiskHeader);
    vpair_region_ = f2_region_ + region_bytes;

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, kCacheFileMode);
    if (fd_ < 0)
        throw std::runtime_error("DiskSink: cannot open '" + path_ + "' for write: " +
                                 std::strerror(errno));

    F2DiskHeader h{};
    std::memcpy(h.magic, kF2DiskMagic, sizeof(h.magic));
    h.version = kF2DiskVersion;
    h.dtype = kF2DiskDtypeFp64;
    h.P = P;
    h.n_block = (n_block < 0 ? 0 : n_block);
    h.f2_offset = f2_region_;
    h.vpair_offset = vpair_region_;
    h.block_sizes_offset = vpair_region_ + region_bytes;
    pwrite_all(fd_, &h, sizeof(h), 0, "header");

    if (slab_ == 0 || n_block <= 0) return;
    // Start the shared pinned ring + writer ([7.1]). The drain callback is the ONLY
    // per-tier difference: pwrite the (drained) slot to its block-major f2/vpair file
    // region offset. PARITY-NEUTRAL: raw bytes (§12). The offsets/fd never change after
    // begin so they are captured by value (fd_ is a plain int handle).
    const int fd = fd_;
    const std::size_t slab_bytes = slab_bytes_;
    const std::uint64_t f2_region = f2_region_;
    const std::uint64_t vpair_region = vpair_region_;
    ring_.begin(
        slab_,
        [fd, slab_bytes, f2_region, vpair_region](SinkSlot& s) {
            pwrite_all(fd, s.f2.data(), slab_bytes,
                       f2_region + slab_bytes * static_cast<std::uint64_t>(s.block), "f2");
            pwrite_all(fd, s.vpair.data(), slab_bytes,
                       vpair_region + slab_bytes * static_cast<std::uint64_t>(s.block), "vpair");
        },
        "DiskSink writer");
}

void DiskSink::spill_block(int b, const double* f2_dev, const double* vpair_dev,
                           std::size_t slab_elems, cudaStream_t stream) {
    (void)slab_elems;
    ring_.spill_block(b, f2_dev, vpair_dev, stream);  // shared ([7.1])
}

void DiskSink::finish() {
    if (finalized_) return;
    ring_.stop_and_join();  // shared ([7.1]); idempotent / no-op for an empty ring.
    if (ring_.writer_failed())
        throw std::runtime_error("DiskSink: background writer failed: " +
                                 ring_.writer_error());

    // block_sizes trailer (int32 each), at the header's block_sizes_offset.
    const std::uint64_t bs_off =
        vpair_region_ + slab_bytes_ * static_cast<std::uint64_t>(n_block_ < 0 ? 0 : n_block_);
    if (!block_sizes_.empty()) {
        static_assert(sizeof(int) == 4, "block_sizes trailer is int32");
        pwrite_all(fd_, block_sizes_.data(), block_sizes_.size() * sizeof(int),
                   bs_off, "block_sizes");
    }

    if (::fsync(fd_) != 0)
        throw std::runtime_error("DiskSink: fsync('" + path_ + "') failed: " +
                                 std::strerror(errno));
    ::close(fd_);
    fd_ = -1;
    read_handle_ = std::fopen(path_.c_str(), "rb");
    if (!read_handle_)
        throw std::runtime_error("DiskSink: cannot reopen '" + path_ + "' read-only: " +
                                 std::strerror(errno));
    finalized_ = true;
}

void DiskSink::take_descriptor(DiskF2Blocks& out) {
    out.path = path_;
    out.P = P_;
    out.n_block = (n_block_ < 0 ? 0 : n_block_);
    out.block_sizes = block_sizes_;
    // Hand the reopened read handle to the descriptor's owning unique_ptr; reset()
    // closes any handle it already held (via FileCloser, the single warn-on-fail close
    // site) before taking ownership — folds the old explicit close-old (group-16 16.5).
    out.read_handle.reset(read_handle_);
    read_handle_ = nullptr;
}

DiskSink::~DiskSink() {
    // STOP THE WRITER FIRST, before this dtor body closes fd_ (the writer's drain callback
    // pwrites to fd_; its lifetime MUST outlive any in-flight write). ring_'s OWN dtor also
    // stops/joins idempotently AND runs the cudaDeviceSynchronize teardown barrier (14.4)
    // before its pinned ring frees — but ring_ is the last-declared member, so it destructs
    // AFTER this body runs; we therefore stop_and_join HERE so no writer pwrite outlives
    // the fd_ close below. (The barrier itself stays in ring_'s dtor, which fires next.)
    ring_.stop_and_join();  // shared ([7.1]); no-op for an empty ring.
    if (read_handle_) { std::fclose(read_handle_); read_handle_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }  // only if finish() didn't run (error path)
}

}  // namespace steppe::device
