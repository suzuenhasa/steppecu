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
#include "device/f2_disk_format.hpp"    // F2DiskHeader, kF2DiskMagic, kF2DiskDtypeFp64, offsets
#include "core/internal/log.hpp"        // STEPPE_LOG_WARN (teardown warnings)

namespace steppe::device {

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
    // PIN ONCE: kStreamStagingSlots × P² pinned doubles each (f2 + vpair).
    slots_.resize(kStreamStagingSlots);
    free_.assign(kStreamStagingSlots, true);
    for (SinkSlot& s : slots_) {
        s.f2 = PinnedBuffer<double>(slab_);
        s.vpair = PinnedBuffer<double>(slab_);
        STEPPE_CUDA_CHECK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
        s.block = -1;
    }
    stop_ = false;
    writer_ = std::thread(&HostRamSink::writer_loop, this);
}

int HostRamSink::acquire_slot() {
    std::unique_lock<std::mutex> lk(mtx_);
    int idx = -1;
    cv_free_.wait(lk, [&] {
        for (int i = 0; i < static_cast<int>(free_.size()); ++i)
            if (free_[static_cast<std::size_t>(i)]) { idx = i; return true; }
        return false;
    });
    free_[static_cast<std::size_t>(idx)] = false;
    return idx;
}

void HostRamSink::writer_loop() {
    for (;;) {
        int idx = -1;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [&] { return stop_ || !ready_.empty(); });
            if (ready_.empty() && stop_) return;
            idx = ready_.front();
            ready_.pop();
        }
        SinkSlot& s = slots_[static_cast<std::size_t>(idx)];
        // Per-slot happens-before: the D2H into this slot must be fully drained before the
        // host reads it. cudaEventSynchronize blocks ONLY on this slot's D2H.
        const cudaError_t e = cudaEventSynchronize(s.done);
        if (e != cudaSuccess) {
            STEPPE_LOG_WARN("HostRamSink writer cudaEventSynchronize: %s", cudaGetErrorString(e));
        }
        std::memcpy(host_.f2.data() + slab_ * static_cast<std::size_t>(s.block),
                    s.f2.data(), slab_ * sizeof(double));
        std::memcpy(host_.vpair.data() + slab_ * static_cast<std::size_t>(s.block),
                    s.vpair.data(), slab_ * sizeof(double));
        {
            std::lock_guard<std::mutex> lk(mtx_);
            free_[static_cast<std::size_t>(idx)] = true;
        }
        cv_free_.notify_one();
    }
}

void HostRamSink::spill_block(int b, const double* f2_dev, const double* vpair_dev,
                              std::size_t slab_elems, cudaStream_t stream) {
    (void)slab_elems;  // == slab_ by construction (P²); the caller passes P*P.
    if (slab_ == 0) return;
    const int idx = acquire_slot();
    SinkSlot& s = slots_[static_cast<std::size_t>(idx)];
    s.block = b;
    const std::size_t bytes = slab_ * sizeof(double);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.f2.data(), f2_dev, bytes,
                                      cudaMemcpyDeviceToHost, stream));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.vpair.data(), vpair_dev, bytes,
                                      cudaMemcpyDeviceToHost, stream));
    STEPPE_CUDA_CHECK(cudaEventRecord(s.done, stream));
    {
        std::lock_guard<std::mutex> lk(mtx_);
        ready_.push(idx);
    }
    cv_work_.notify_one();
}

void HostRamSink::finish() {
    if (!writer_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    cv_work_.notify_all();
    writer_.join();
}

HostRamSink::~HostRamSink() {
    if (writer_.joinable()) {
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        writer_.join();
    }
    for (SinkSlot& s : slots_) {
        if (s.done) {
            const cudaError_t e = cudaEventDestroy(s.done);
            if (e != cudaSuccess)
                STEPPE_LOG_WARN("cudaEventDestroy (HostRamSink): %s", cudaGetErrorString(e));
            s.done = nullptr;
        }
    }
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
    f2_region_ = sizeof(F2DiskHeader);                  // == 64
    vpair_region_ = f2_region_ + region_bytes;

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0)
        throw std::runtime_error("DiskSink: cannot open '" + path_ + "' for write: " +
                                 std::strerror(errno));

    F2DiskHeader h{};
    std::memcpy(h.magic, kF2DiskMagic, sizeof(h.magic));
    h.version = 1u;
    h.dtype = kF2DiskDtypeFp64;
    h.P = P;
    h.n_block = (n_block < 0 ? 0 : n_block);
    h.f2_offset = f2_region_;
    h.vpair_offset = vpair_region_;
    h.block_sizes_offset = vpair_region_ + region_bytes;
    pwrite_all(fd_, &h, sizeof(h), 0, "header");

    if (slab_ == 0 || n_block <= 0) return;
    slots_.resize(kStreamStagingSlots);
    free_.assign(kStreamStagingSlots, true);
    for (SinkSlot& s : slots_) {
        s.f2 = PinnedBuffer<double>(slab_);
        s.vpair = PinnedBuffer<double>(slab_);
        STEPPE_CUDA_CHECK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
        s.block = -1;
    }
    stop_ = false;
    writer_failed_ = false;
    writer_ = std::thread(&DiskSink::writer_loop, this);
}

int DiskSink::acquire_slot() {
    std::unique_lock<std::mutex> lk(mtx_);
    int idx = -1;
    cv_free_.wait(lk, [&] {
        for (int i = 0; i < static_cast<int>(free_.size()); ++i)
            if (free_[static_cast<std::size_t>(i)]) { idx = i; return true; }
        return false;
    });
    free_[static_cast<std::size_t>(idx)] = false;
    return idx;
}

void DiskSink::writer_loop() {
    for (;;) {
        int idx = -1;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [&] { return stop_ || !ready_.empty(); });
            if (ready_.empty() && stop_) return;
            idx = ready_.front();
            ready_.pop();
        }
        SinkSlot& s = slots_[static_cast<std::size_t>(idx)];
        try {
            const cudaError_t e = cudaEventSynchronize(s.done);
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("DiskSink writer cudaEventSynchronize: ") +
                                         cudaGetErrorString(e));
            pwrite_all(fd_, s.f2.data(), slab_bytes_,
                       f2_region_ + slab_bytes_ * static_cast<std::uint64_t>(s.block), "f2");
            pwrite_all(fd_, s.vpair.data(), slab_bytes_,
                       vpair_region_ + slab_bytes_ * static_cast<std::uint64_t>(s.block), "vpair");
        } catch (const std::exception& ex) {
            // Record the first error; the compute thread re-throws it at finish(). Keep
            // draining the queue so the compute thread is not deadlocked on a full ring.
            std::lock_guard<std::mutex> lk(mtx_);
            if (!writer_failed_) { writer_failed_ = true; writer_error_ = ex.what(); }
        }
        {
            std::lock_guard<std::mutex> lk(mtx_);
            free_[static_cast<std::size_t>(idx)] = true;
        }
        cv_free_.notify_one();
    }
}

void DiskSink::spill_block(int b, const double* f2_dev, const double* vpair_dev,
                           std::size_t slab_elems, cudaStream_t stream) {
    (void)slab_elems;
    if (slab_ == 0) return;
    const int idx = acquire_slot();
    SinkSlot& s = slots_[static_cast<std::size_t>(idx)];
    s.block = b;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.f2.data(), f2_dev, slab_bytes_,
                                      cudaMemcpyDeviceToHost, stream));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.vpair.data(), vpair_dev, slab_bytes_,
                                      cudaMemcpyDeviceToHost, stream));
    STEPPE_CUDA_CHECK(cudaEventRecord(s.done, stream));
    {
        std::lock_guard<std::mutex> lk(mtx_);
        ready_.push(idx);
    }
    cv_work_.notify_one();
}

void DiskSink::finish() {
    if (finalized_) return;
    if (writer_.joinable()) {
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        writer_.join();
    }
    if (writer_failed_)
        throw std::runtime_error("DiskSink: background writer failed: " + writer_error_);

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
    if (out.read_handle) std::fclose(out.read_handle);
    out.read_handle = read_handle_;
    read_handle_ = nullptr;
}

DiskSink::~DiskSink() {
    if (writer_.joinable()) {
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        writer_.join();
    }
    for (SinkSlot& s : slots_) {
        if (s.done) {
            const cudaError_t e = cudaEventDestroy(s.done);
            if (e != cudaSuccess)
                STEPPE_LOG_WARN("cudaEventDestroy (DiskSink): %s", cudaGetErrorString(e));
            s.done = nullptr;
        }
    }
    if (read_handle_) { std::fclose(read_handle_); read_handle_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }  // only if finish() didn't run (error path)
}

}  // namespace steppe::device
