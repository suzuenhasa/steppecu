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
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/internal/log.hpp"          // STEPPE_LOG_WARN (teardown-barrier warning)
#include "device/cuda/check.cuh"          // STEPPE_CUDA_CHECK (spill_block D2H/event-record)
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

/// The ONE pinned-staging-ring + background-writer body shared by BOTH sinks
/// ([7.1]). HostRamSink and DiskSink previously held BYTE-IDENTICAL ring plumbing
/// (slots_/free_/ready_/mtx_/cv_*/stop_/writer_ + writer_failed_/writer_error_) and
/// BYTE-IDENTICAL acquire_slot / writer-loop queue-pop skeleton / spill_block (D2H +
/// event-record + enqueue) / stop-and-join (4 copies) / teardown barrier; the ONLY
/// per-tier difference is the WRITER DRAIN ACTION (HostRam memcpy into the destination
/// tensor vs Disk pwrite to the file region) and the throw tag. So the ring owns ALL
/// of that ONE TIME and the caller injects ONLY the drain via a callback — one body
/// replacing the two copies, BEHAVIOR-IDENTICAL (no math, no threading change).
///
/// The fail-fast policy (9dbc610 [13.3] HIGH) is PRESERVED for BOTH tiers: the writer
/// FAIL-FAST waits each slot's D2H event via sink_wait_slot_drained, and on a non-
/// success sync (the slot may be undrained) it records writer_failed_ and SKIPS the
/// drain — re-thrown by the owning sink at finish(); it NEVER drains an undrained slot
/// (a §12 parity violation). The drain callback itself may also throw (Disk pwrite
/// errno); that is caught the same way (first error recorded, queue kept draining so
/// the compute thread is not deadlocked on a full ring).
class StagingRing {
public:
    /// The per-slot drain action: copy/write the (already-drained) slot to its tier
    /// destination. Runs ON THE WRITER THREAD, NOT under mtx_ (the slot is privately
    /// owned by the writer between ready_.pop and the free-return). May throw (Disk
    /// pwrite); the thrown error is recorded as the writer failure.
    using DrainFn = std::function<void(SinkSlot& slot)>;

    StagingRing() = default;
    ~StagingRing() { stop_and_join(); teardown_barrier(); }
    // Owns a writer thread + pinned ring; non-copyable AND non-movable (the mtx_/cv_*
    // members already suppress these implicitly — make it explicit, 16.3).
    StagingRing(const StagingRing&) = delete;
    StagingRing& operator=(const StagingRing&) = delete;
    StagingRing(StagingRing&&) = delete;
    StagingRing& operator=(StagingRing&&) = delete;

    /// Pin the kStreamStagingSlots × slab_elems pinned slots ONCE and start the writer.
    /// `drain` is the tier-specific per-slot action; `what` tags the writer throw site.
    /// The caller does all NON-ring begin() work (allocate the dst tensor / open the
    /// file + write the header) BEFORE calling this. No-op for an empty ring
    /// (slab_elems == 0) — the writer is not started.
    void begin(std::size_t slab_elems, DrainFn drain, const char* what) {
        slab_ = slab_elems;
        slab_bytes_ = slab_ * sizeof(double);
        drain_ = std::move(drain);
        what_ = what;
        if (slab_ == 0) return;
        // PIN ONCE: kStreamStagingSlots × slab_ pinned doubles each (f2 + vpair). Each
        // slot's RAII Event (cudaEventDisableTiming, throws-on-create) is default-
        // constructed by the resize() below (16.1).
        slots_.resize(kStreamStagingSlots);
        free_.assign(kStreamStagingSlots, true);
        for (SinkSlot& s : slots_) {
            s.f2 = PinnedBuffer<double>(slab_);
            s.vpair = PinnedBuffer<double>(slab_);
            s.block = -1;
        }
        stop_ = false;
        writer_failed_ = false;
        writer_ = std::thread(&StagingRing::writer_loop, this);
    }

    /// Spill block `b`: claim a free slot (blocking under backpressure), issue the two
    /// async D2H of its f2/vpair slab into the slot on `stream`, record the slot's
    /// completion event, and ENQUEUE the slot to the writer. Does NOT block on the
    /// drain and does NOT free the device slab (the stream loop owns it). No-op for an
    /// empty ring. Byte-identical to the two former spill_block member bodies (the only
    /// former difference — HostRam's inline `slab_*sizeof(double)` vs Disk's slab_bytes_
    /// member — is now the single slab_bytes_ here).
    void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                     cudaStream_t stream) {
        if (slab_ == 0) return;
        const int idx = acquire_slot();
        SinkSlot& s = slots_[static_cast<std::size_t>(idx)];
        s.block = b;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.f2.data(), f2_dev, slab_bytes_,
                                          cudaMemcpyDeviceToHost, stream));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.vpair.data(), vpair_dev, slab_bytes_,
                                          cudaMemcpyDeviceToHost, stream));
        // Raw cudaEventRecord on the wrapped handle: spill_block receives a raw
        // cudaStream_t, but Event::record takes a Stream&, so record via .get() (16.1).
        STEPPE_CUDA_CHECK(cudaEventRecord(s.done.get(), stream));
        {
            std::lock_guard<std::mutex> lk(mtx_);
            ready_.push(idx);
        }
        cv_work_.notify_one();
    }

    /// Signal stop + wake + join the writer (idempotent — a no-op once joined). Byte-
    /// identical to the four former finish()/dtor stop-and-join copies.
    void stop_and_join() {
        if (!writer_.joinable()) return;
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        writer_.join();
    }

    /// True if the writer recorded a (drain or event-sync) failure. The owning sink
    /// re-throws it on the compute thread at finish() AFTER stop_and_join().
    [[nodiscard]] bool writer_failed() const noexcept { return writer_failed_; }
    [[nodiscard]] const std::string& writer_error() const noexcept { return writer_error_; }

private:
    /// Claim a free ring slot, blocking under backpressure until one returns to the
    /// free pool. Byte-identical to the two former acquire_slot() member bodies.
    [[nodiscard]] int acquire_slot() {
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

    /// Background writer: pop a queued slot, FAIL-FAST wait its D2H event (NEVER drain
    /// an undrained slot — §12), run the tier drain, return the slot to the free pool.
    /// Byte-identical queue-pop skeleton to the two former writer_loop bodies; only the
    /// drain call (the injected callback) replaces the two former inline drain bodies.
    void writer_loop() {
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
                // Per-slot happens-before: the D2H into this slot must be fully drained
                // before the host reads it (sink_wait_slot_drained blocks ONLY on this
                // slot's D2H and FAIL-FASTS on a non-success sync — draining an undrained
                // slot would silently corrupt f2/vpair, a §12 parity violation).
                sink_wait_slot_drained(s.done.get(), what_);
                drain_(s);
            } catch (const std::exception& ex) {
                // Record the first error; the owning sink re-throws it at finish(). Keep
                // draining the queue so the compute thread is not deadlocked on a full
                // ring. We deliberately do NOT drain this (possibly undrained) slot.
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

    /// DEFENSIVE TEARDOWN BARRIER (14.4): the writer's per-ENQUEUED-slot event sync is
    /// the sole D2H-completion guarantee — but if spill_block faulted mid-issue (the 2nd
    /// cudaMemcpyAsync or the cudaEventRecord threw via STEPPE_CUDA_CHECK) the FIRST D2H
    /// is left in flight on `stream` and its slot was NEVER pushed to ready_, so the
    /// writer never synchronized it. Freeing the pinned ring (slots_ destruct ->
    /// PinnedBuffer::reset -> cudaFreeHost) under that still-running DMA is a use-after-
    /// free of pinned host memory. cudaDeviceSynchronize "blocks until the device has
    /// completed all preceding requested tasks ... in all streams" (CUDA 13.x Runtime
    /// API), so it drains any such orphaned in-flight transfer before teardown. Warn-not-
    /// throw (a dtor never throws, architecture.md §7); off the hot path so the sync cost
    /// is irrelevant; on the happy path every D2H was already drained before join, so
    /// this is a cheap no-op. Called from the dtor AFTER stop_and_join and BEFORE slots_
    /// (and thus each slot's RAII Event + pinned buffers) destructs.
    void teardown_barrier() noexcept {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            STEPPE_LOG_WARN("cudaDeviceSynchronize (StagingRing teardown): %s",
                            cudaGetErrorString(e));
    }

    std::size_t slab_ = 0;            ///< per-block slab element count (P²).
    std::size_t slab_bytes_ = 0;      ///< slab_ * sizeof(double) (hoisted ONCE, [7.3]).
    DrainFn drain_;                   ///< tier-specific per-slot drain (memcpy / pwrite).
    const char* what_ = "";          ///< writer throw tag (per tier).
    std::vector<SinkSlot> slots_;     ///< kStreamStagingSlots pinned slots.
    std::thread writer_;
    std::mutex mtx_;
    std::condition_variable cv_free_;   ///< signaled when a slot returns to free.
    std::condition_variable cv_work_;   ///< signaled when a slot is queued / done.
    std::queue<int> ready_;             ///< slot indices queued FOR the writer.
    std::vector<bool> free_;            ///< slot free for the compute thread to claim.
    bool stop_ = false;
    // First error the writer thread hit (re-thrown by the owning sink at finish()) — a
    // failed event sync means the slot's D2H may be undrained, so we MUST NOT drain it.
    bool writer_failed_ = false;
    std::string writer_error_;
};

/// TIER 1: stream blocks into a host F2BlockTensor via the SHARED StagingRing (pinned
/// ring + background writer). The ring's writer FAIL-FAST waits each slot's D2H event (a
/// non-success sync means the slot may be undrained → records writer_failed_ and SKIPS
/// the copy, re-thrown at finish() — identical to DiskSink, never a silently-corrupting
/// WARN-and-continue per §12), then runs THIS sink's drain callback: std::memcpy the slot
/// into host.f2[P²·b]/host.vpair[P²·b] while the GPU computes the next chunk. PARITY-
/// NEUTRAL: the D2H is a raw byte copy, the host copy is memcpy — same bytes (§12).
class HostRamSink final : public BlockSink {
public:
    explicit HostRamSink(F2BlockTensor& dst) noexcept : host_(dst) {}
    ~HostRamSink() override = default;  // ring_ dtor stops/joins + barriers (StagingRing).
    // Owns a StagingRing (writer thread + pinned ring); non-copyable AND non-movable
    // (16.3 — make the ownership posture explicit, robust against a future member change
    // silently re-enabling a wrong-shaped special member; the ring is itself non-movable).
    HostRamSink(const HostRamSink&) = delete;
    HostRamSink& operator=(const HostRamSink&) = delete;
    HostRamSink(HostRamSink&&) = delete;
    HostRamSink& operator=(HostRamSink&&) = delete;
    void begin(int P, int n_block, const std::vector<int>& block_sizes) override;
    void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                     std::size_t slab_elems, cudaStream_t stream) override;
    void finish() override;

private:
    F2BlockTensor& host_;
    int P_ = 0;
    int n_block_ = 0;
    std::size_t slab_ = 0;           ///< P² (the per-block slab element count).
    StagingRing ring_;               ///< the shared pinned ring + background writer ([7.1]).
};

class DiskF2Blocks;  // fwd (CUDA-free type from f2_blocks_out.hpp; populated in finish())

/// TIER 2: stream blocks to a disk file via the SAME shared StagingRing (device ->
/// pinned staging -> disk pwrite ON THE WRITER THREAD), so the slow pwrite OVERLAPS GPU
/// compute and RAM stays tiny on a laptop. begin() opens the file + writes the 64-byte
/// header, then starts the ring whose drain callback pwrites each slot to the f2/vpair
/// region offsets. spill_block() D2Hs into the next slot and enqueues it. finish() drains
/// the queue, writes the block_sizes trailer, fsyncs, closes, reopens read-only. PARITY-
/// NEUTRAL: raw bytes throughout (§12).
class DiskSink final : public BlockSink {
public:
    explicit DiskSink(std::string path) noexcept : path_(std::move(path)) {}
    ~DiskSink() override;
    // Owns an fd + a StagingRing (writer thread + pinned ring); non-copyable AND non-
    // movable (16.3 — make the ownership posture explicit, robust against a future member
    // change silently re-enabling a wrong-shaped special member; the ring is non-movable).
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
    std::string path_;
    int fd_ = -1;
    int P_ = 0;
    int n_block_ = 0;
    std::size_t slab_ = 0;
    std::size_t slab_bytes_ = 0;
    std::uint64_t f2_region_ = 0;
    std::uint64_t vpair_region_ = 0;
    std::vector<int> block_sizes_;
    bool finalized_ = false;
    std::FILE* read_handle_ = nullptr;
    StagingRing ring_;               ///< the shared pinned ring + background writer ([7.1]).
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_BLOCK_SINK_CUH
