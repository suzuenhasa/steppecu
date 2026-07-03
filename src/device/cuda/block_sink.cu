// src/device/cuda/block_sink.cu
//
// The M5 block-spill sinks: HostRamSink (tier 1, host RAM) and DiskSink (tier 2,
// on-disk cache). Each stages a block's slab through a shared pinned ring plus a
// background writer thread so the storage write overlaps GPU compute, and the spill
// is parity-neutral (raw byte copies). Private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_block_sink.cu.md
#include "device/cuda/block_sink.cuh"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "device/cuda/check.cuh"
#include "device/f2_blocks_out.hpp"
#include "device/f2_disk_format.hpp"
#include "core/internal/log.hpp"

namespace steppe::device {

namespace { constexpr int kCacheFileMode = 0644; }

// HostRamSink (tier 1): drain each staged slot into host RAM — reference §3

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
    (void)slab_elems;
    ring_.spill_block(b, f2_dev, vpair_dev, stream);
}

void HostRamSink::finish() {
    ring_.stop_and_join();
    if (ring_.writer_failed())
        throw std::runtime_error("HostRamSink: background writer failed: " +
                                 ring_.writer_error());
}

// DiskSink (tier 2): write and finalize the on-disk cache file — reference §4

namespace {
// pwrite_all: complete a write despite short writes and interrupts — reference §5
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
    ring_.spill_block(b, f2_dev, vpair_dev, stream);
}

void DiskSink::finish() {
    if (finalized_) return;
    ring_.stop_and_join();
    if (ring_.writer_failed())
        throw std::runtime_error("DiskSink: background writer failed: " +
                                 ring_.writer_error());

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

// Shutdown ordering and handing off the read handle — reference §6
void DiskSink::take_descriptor(DiskF2Blocks& out) {
    out.path = path_;
    out.P = P_;
    out.n_block = (n_block_ < 0 ? 0 : n_block_);
    out.block_sizes = block_sizes_;
    out.read_handle.reset(read_handle_);
    read_handle_ = nullptr;
}

DiskSink::~DiskSink() {
    ring_.stop_and_join();
    if (read_handle_) { std::fclose(read_handle_); read_handle_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

}  // namespace steppe::device
