// src/device/cuda/block_sink.cuh
//
// The M5 block-spill SINK seam + its 2 streaming impls (CUDA-private, architecture.md §4
// — it takes device pointers and owns pinned staging, the same allowlisting domain as
// pinned_buffer.cuh). The block-stream loop computes block b's [P²] f2 + [P²] vpair slab
// RESIDENT on device, then hands the DEVICE pointers to the sink, which gets them to
// their tier. Each block is INDEPENDENT (architecture.md §12 block-axis exact); the sink
// only changes WHEN/WHERE a slab lands, never its bits.
//
// TIER 0 (Resident) BYPASSES this seam ENTIRELY — it never constructs a sink. Only
// HostRam + Disk use a sink, and BOTH use a SMALL PERSISTENT pinned staging RING
// (kStreamStagingSlots slots, pinned ONCE at begin(), reused every block — the 94c6d8e
// per-call-register bug is designed out) + a BACKGROUND WRITER THREAD so the host-copy /
// pwrite OVERLAPS the GPU compute of the next chunk (the triple-buffer the §5 perf claim
// rests on: compute b+1 / D2H b / write b-1). The compute thread only ENQUEUES a drained
// pinned slot to the writer; it never blocks on the slow disk write itself.
#ifndef STEPPE_DEVICE_CUDA_BLOCK_SINK_CUH
#define STEPPE_DEVICE_CUDA_BLOCK_SINK_CUH

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "device/cuda/pinned_buffer.cuh"  // PinnedBuffer<double> (persistent pinned staging)
#include "device/cuda/stream.hpp"         // steppe::device::Event (move-only RAII CUDA event)
#include "steppe/fstats.hpp"              // F2BlockTensor (HostRam destination)

namespace steppe::device {

/// Triple-buffer depth: enough pinned slots that the GPU can keep computing/D2H-ing while
/// the background writer drains earlier ones (compute b+1 / D2H b / write b-1). The DEVICE
/// chunk buffers (stream_f2_blocks_impl) are double-buffered separately (they only need to
/// survive their own D2H, not the slow write).
inline constexpr int kStreamStagingSlots = 3;

/// The block-spill SINK seam (M5). The stream loop computes block b's [P²] f2 + [P²]
/// vpair slab RESIDENT on device, then hands the DEVICE pointers to the sink. Each block
/// is INDEPENDENT (architecture.md §12); the sink only changes WHEN/WHERE a slab lands,
/// never its bits. Resident BYPASSES this seam entirely (never constructs a sink).
class BlockSink {
public:
    virtual ~BlockSink() = default;
    /// Allocate the destination + the persistent pinned staging ring + start the writer
    /// thread (pin ONCE here, reuse). Records P/n_block/block_sizes for the header.
    virtual void begin(int P, int n_block, const std::vector<int>& block_sizes) = 0;
    /// Spill block `b`'s slabs. `f2_dev`/`vpair_dev` are DEVICE pointers to its [P²]
    /// f2/vpair (valid on `stream` after the block's assemble). The sink claims the next
    /// pinned ring slot (blocking only if all slots are still in the writer's hands —
    /// the natural backpressure), issues the D2H into it on `stream` (async, genuine DMA),
    /// records a completion event, and ENQUEUES the slot to the background writer. It does
    /// NOT block on the disk write and does NOT free the device slab (the loop owns it).
    virtual void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                             std::size_t slab_elems, cudaStream_t stream) = 0;
    /// Drain the writer queue + finalize (Disk: write the header trailer + fsync + reopen
    /// for read; Host: join the writer). The populated F2BlocksOut is built by the
    /// orchestrator from the sink's tier outputs.
    virtual void finish() = 0;
};

// A pinned staging slot shared by both sinks: two P²-double pinned buffers (f2 + vpair),
// a CUDA event recorded after the slot's D2H, and the destination block id. `done` is a
// move-only RAII Event (16.1): it default-constructs a cudaEventDisableTiming event when the
// slots_ vector is resize()'d in begin() (throws-on-create, exactly as the old hand-rolled
// STEPPE_CUDA_CHECK(cudaEventCreateWithFlags) did) and self-frees in its warn-not-throw dtor.
// Wrapping it also makes SinkSlot's implicit member-wise move correct (16.2): Event's move
// std::exchanges the handle to null, so a moved-from slot no longer double-cudaEventDestroys.
struct SinkSlot {
    PinnedBuffer<double> f2;
    PinnedBuffer<double> vpair;
    Event done;
    int block = -1;
};

/// Shared fail-fast event-wait for BOTH sinks' writer threads. Blocks on this slot's D2H
/// completion event (the per-slot happens-before — the slab is fully drained before the
/// host reads it). A non-cudaSuccess return means the wait failed OR a prior async launch
/// on the stream (e.g. the D2H itself) failed, so the slot may hold stale/undrained bytes;
/// per architecture.md §12 (parity) the ONLY safe action is to fail fast — NEVER copy an
/// undrained slot (that would silently corrupt f2/vpair). `what` tags the throw site so the
/// two tiers attribute identically. (CUDA 13 cudaEventSynchronize returns cudaSuccess /
/// cudaErrorInvalidValue / cudaErrorInvalidResourceHandle / cudaErrorLaunchFailure, and may
/// also surface error codes from previous asynchronous launches.)
inline void sink_wait_slot_drained(cudaEvent_t done, const char* what) {
    const cudaError_t e = cudaEventSynchronize(done);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + " cudaEventSynchronize: " +
                                 cudaGetErrorString(e));
}

/// TIER 1: stream blocks into a host F2BlockTensor via the pinned ring + background
/// writer. The writer FAIL-FAST waits each slot's D2H event (a non-success sync means the
/// slot may be undrained → records writer_failed_ and SKIPS the copy, re-thrown at
/// finish() — identical to DiskSink, never a silently-corrupting WARN-and-continue per
/// §12), then std::memcpy's the slot into host.f2[P²·b]/host.vpair[P²·b] while the GPU
/// computes the next chunk. PARITY-NEUTRAL: the D2H is a raw byte copy, the host copy is
/// memcpy — same bytes (§12).
class HostRamSink final : public BlockSink {
public:
    explicit HostRamSink(F2BlockTensor& dst) noexcept : host_(dst) {}
    ~HostRamSink() override;
    // Owns a writer thread + pinned ring; non-copyable AND non-movable (16.3 — make the
    // ownership posture explicit, robust against a future member change silently re-enabling
    // a wrong-shaped special member).
    HostRamSink(const HostRamSink&) = delete;
    HostRamSink& operator=(const HostRamSink&) = delete;
    HostRamSink(HostRamSink&&) = delete;
    HostRamSink& operator=(HostRamSink&&) = delete;
    void begin(int P, int n_block, const std::vector<int>& block_sizes) override;
    void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                     std::size_t slab_elems, cudaStream_t stream) override;
    void finish() override;

private:
    void writer_loop();              ///< background: pop a slot, FAIL-FAST wait its event, memcpy.
    int acquire_slot();              ///< claim a free ring slot (blocks under backpressure).

    F2BlockTensor& host_;
    int P_ = 0;
    int n_block_ = 0;
    std::size_t slab_ = 0;           ///< P² (the per-block slab element count).
    std::vector<SinkSlot> slots_;    ///< kStreamStagingSlots pinned slots.
    // Writer-thread plumbing.
    std::thread writer_;
    std::mutex mtx_;
    std::condition_variable cv_free_;   ///< signaled when a slot returns to free.
    std::condition_variable cv_work_;   ///< signaled when a slot is queued / done.
    std::queue<int> ready_;             ///< slot indices queued FOR the writer.
    std::vector<bool> free_;            ///< slot free for the compute thread to claim.
    bool stop_ = false;
    // First error the writer thread hit (re-thrown on the compute thread at finish()) — a
    // failed event sync means the slot's D2H may be undrained, so we MUST NOT memcpy it.
    bool writer_failed_ = false;
    std::string writer_error_;
};

class DiskF2Blocks;  // fwd (CUDA-free type from f2_blocks_out.hpp; populated in finish())

/// TIER 2: stream blocks to a disk file via the SAME pinned ring + background writer
/// (device -> pinned staging -> disk pwrite ON THE WRITER THREAD), so the slow pwrite
/// OVERLAPS GPU compute and RAM stays tiny on a laptop. begin() opens the file + writes
/// the 64-byte header + starts the writer. spill_block() D2Hs into the next slot and
/// enqueues it; the writer pwrites it to the f2/vpair region offsets. finish() drains the
/// queue, writes the block_sizes trailer, fsyncs, closes, reopens read-only. PARITY-
/// NEUTRAL: raw bytes throughout (§12).
class DiskSink final : public BlockSink {
public:
    explicit DiskSink(std::string path) noexcept : path_(std::move(path)) {}
    ~DiskSink() override;
    // Owns an fd + writer thread + pinned ring; non-copyable AND non-movable (16.3 — make the
    // ownership posture explicit, robust against a future member change silently re-enabling
    // a wrong-shaped special member).
    DiskSink(const DiskSink&) = delete;
    DiskSink& operator=(const DiskSink&) = delete;
    DiskSink(DiskSink&&) = delete;
    DiskSink& operator=(DiskSink&&) = delete;
    void begin(int P, int n_block, const std::vector<int>& block_sizes) override;
    void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                     std::size_t slab_elems, cudaStream_t stream) override;
    void finish() override;

    /// Move the finalized on-disk descriptor (path + shape + reopened read handle) to the
    /// caller. Valid only AFTER finish(). The orchestrator stores it on F2BlocksOut.disk.
    void take_descriptor(DiskF2Blocks& out);

private:
    void writer_loop();
    int acquire_slot();

    std::string path_;
    int fd_ = -1;
    int P_ = 0;
    int n_block_ = 0;
    std::size_t slab_ = 0;
    std::size_t slab_bytes_ = 0;
    std::uint64_t f2_region_ = 0;
    std::uint64_t vpair_region_ = 0;
    std::vector<int> block_sizes_;
    std::vector<SinkSlot> slots_;
    std::thread writer_;
    std::mutex mtx_;
    std::condition_variable cv_free_;
    std::condition_variable cv_work_;
    std::queue<int> ready_;
    std::vector<bool> free_;
    bool stop_ = false;
    bool finalized_ = false;
    std::FILE* read_handle_ = nullptr;
    // First error the writer thread hit (re-thrown on the compute thread at finish()).
    bool writer_failed_ = false;
    std::string writer_error_;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_BLOCK_SINK_CUH
