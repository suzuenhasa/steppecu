// src/device/cuda/stream.hpp
//
// RAII Stream / Event wrappers (architecture.md §2 RAII, §7, §8; ROADMAP §5).
//
// Replaces the spike's bare `cudaEventCreate`/`cudaEventDestroy` pairs (one per
// timed GEMM in f2_emu_spike.cu:267-323, f2_prec_acc.cu, f2_timing.cu — ROADMAP
// §1 "no RAII"), where every `exit()` error path leaked the events. Here lifetime
// is tied to scope and destroyed exactly once.
//
// Shape (architecture.md §7): both types are fully move-only — move-construct AND
// move-assign (the old draft deleting move-assign silently made `s = std::move(o)`
// ill-formed). Copy is deleted. Destructors NEVER throw; a nonzero destroy status
// at teardown routes to a debug-only warning (architecture.md §2, §7, §10).
//
// Usage (architecture.md §7, §11.1): one Stream per independent lane — the single
// statistic stream on the bit-stable path (architecture.md §12), separate copy /
// search lanes for throughput; express cross-stream dependencies with an Event,
// never a device-wide sync.
//
// Despite the `.hpp` extension (matching architecture.md §4's `stream.hpp` entry
// under src/device/cuda/), this includes `cuda_runtime.h` and so is a CUDA header:
// PRIVATE to steppe_device, never compiled into core / api / the CLI
// (architecture.md §4 layering rule).
#ifndef STEPPE_DEVICE_CUDA_STREAM_HPP
#define STEPPE_DEVICE_CUDA_STREAM_HPP

#include <cuda_runtime.h>

#include <utility>

#include "core/internal/log.hpp"  // STEPPE_LOG_WARN (the one teardown-warning sink)
#include "device/cuda/check.cuh"  // STEPPE_CUDA_CHECK

namespace steppe::device {

/// Owning, move-only CUDA stream. A default-constructed Stream creates a fresh
/// non-blocking-capable stream; a moved-from Stream owns nothing and is safe to
/// destroy. Hand `get()` to kernel launches / cuBLAS / memcpyAsync.
class Stream {
public:
    /// Create a new stream. Throws CudaError via STEPPE_CUDA_CHECK on failure.
    Stream() { STEPPE_CUDA_CHECK(cudaStreamCreate(&s_)); }

    Stream(Stream&& o) noexcept : s_(std::exchange(o.s_, nullptr)) {}

    Stream& operator=(Stream&& o) noexcept {
        if (this != &o) {
            destroy();
            s_ = std::exchange(o.s_, nullptr);
        }
        return *this;
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    ~Stream() { destroy(); }

    /// The underlying stream handle (non-owning view for launch arguments).
    [[nodiscard]] cudaStream_t get() const noexcept { return s_; }

    /// Block the calling host thread until all work on this stream completes.
    void synchronize() const { STEPPE_CUDA_CHECK(cudaStreamSynchronize(s_)); }

private:
    void destroy() noexcept {
        // Destructor never throws (architecture.md §7); a nonzero destroy status
        // is reported to the debug-only warning sink, never thrown.
        if (s_) {
            const cudaError_t e = cudaStreamDestroy(s_);
            if (e != cudaSuccess) {
                STEPPE_LOG_WARN("cudaStreamDestroy at teardown: %s",
                                cudaGetErrorString(e));
            }
        }
        s_ = nullptr;
    }

    cudaStream_t s_ = nullptr;
};

/// Owning, move-only CUDA event. Created with timing DISABLED by default
/// (`cudaEventDisableTiming`), which is the cheaper, lower-latency variant used
/// for cross-stream ordering (architecture.md §7 "express cross-stream deps with
/// Event"); pass `enable_timing = true` for the elapsed-time measurement path
/// (the spike's GEMM timing, f2_timing.cu).
class Event {
public:
    /// Create an event. `enable_timing` selects whether `elapsed_ms` is usable;
    /// disabled by default for the ordering use case. Throws on failure.
    explicit Event(bool enable_timing = false) {
        const unsigned int flags =
            enable_timing ? cudaEventDefault : cudaEventDisableTiming;
        STEPPE_CUDA_CHECK(cudaEventCreateWithFlags(&e_, flags));
    }

    Event(Event&& o) noexcept : e_(std::exchange(o.e_, nullptr)) {}

    Event& operator=(Event&& o) noexcept {
        if (this != &o) {
            destroy();
            e_ = std::exchange(o.e_, nullptr);
        }
        return *this;
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    ~Event() { destroy(); }

    /// The underlying event handle.
    [[nodiscard]] cudaEvent_t get() const noexcept { return e_; }

    /// Record this event on `stream` (capture its progress point).
    void record(const Stream& stream) {
        STEPPE_CUDA_CHECK(cudaEventRecord(e_, stream.get()));
    }

    /// Make `stream` wait until this event completes — the cross-stream
    /// dependency primitive (architecture.md §7), preferred over device-wide sync.
    void wait(const Stream& stream) const {
        STEPPE_CUDA_CHECK(cudaStreamWaitEvent(stream.get(), e_, 0));
    }

    /// Block the host until this event completes.
    void synchronize() const { STEPPE_CUDA_CHECK(cudaEventSynchronize(e_)); }

    /// Milliseconds elapsed from `start` to this event. Both events must have been
    /// created with timing enabled and recorded (architecture.md §11.3 profiling).
    [[nodiscard]] float elapsed_ms(const Event& start) const {
        float ms = 0.0f;
        STEPPE_CUDA_CHECK(cudaEventElapsedTime(&ms, start.e_, e_));
        return ms;
    }

private:
    void destroy() noexcept {
        // Destructor never throws (architecture.md §7); a nonzero destroy status
        // is reported to the debug-only warning sink, never thrown.
        if (e_) {
            const cudaError_t err = cudaEventDestroy(e_);
            if (err != cudaSuccess) {
                STEPPE_LOG_WARN("cudaEventDestroy at teardown: %s",
                                cudaGetErrorString(err));
            }
        }
        e_ = nullptr;
    }

    cudaEvent_t e_ = nullptr;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_STREAM_HPP
