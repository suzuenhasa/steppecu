// src/device/cuda/stream.hpp
//
// RAII wrappers that own a CUDA stream and a CUDA event and free each exactly
// once at end of scope. A CUDA header (includes cuda_runtime.h), so it is
// private to the GPU layer — never compiled into core, api, or the CLI.
//
// Reference: docs/reference/src_device_cuda_stream.hpp.md
#ifndef STEPPE_DEVICE_CUDA_STREAM_HPP
#define STEPPE_DEVICE_CUDA_STREAM_HPP

#include <cuda_runtime.h>

#include <utility>

#include "core/internal/log.hpp"
#include "device/cuda/check.cuh"

namespace steppe::device {

// Owning, move-only CUDA stream — reference §2
class Stream {
public:
    explicit Stream(unsigned int flags = cudaStreamNonBlocking) {
        STEPPE_CUDA_CHECK(cudaStreamCreateWithFlags(&s_, flags));
    }

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

    [[nodiscard]] cudaStream_t get() const noexcept { return s_; }

    void synchronize() const { STEPPE_CUDA_CHECK(cudaStreamSynchronize(s_)); }

private:
    void destroy() noexcept {
        if (s_) {
            const cudaError_t err = cudaStreamDestroy(s_);
            if (err != cudaSuccess) {
                STEPPE_LOG_WARN("cudaStreamDestroy at teardown: %s",
                                cudaGetErrorString(err));
            }
        }
        s_ = nullptr;
    }

    cudaStream_t s_ = nullptr;
};

// Owning, move-only CUDA event — reference §3
class Event {
public:
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

    [[nodiscard]] cudaEvent_t get() const noexcept { return e_; }

    void record(const Stream& stream) {
        STEPPE_CUDA_CHECK(cudaEventRecord(e_, stream.get()));
    }

    void wait(const Stream& stream) const {
        STEPPE_CUDA_CHECK(cudaStreamWaitEvent(stream.get(), e_, cudaEventWaitDefault));
    }

    void synchronize() const { STEPPE_CUDA_CHECK(cudaEventSynchronize(e_)); }

    [[nodiscard]] float elapsed_ms(const Event& start) const {
        float ms = 0.0f;
        STEPPE_CUDA_CHECK(cudaEventElapsedTime(&ms, start.e_, e_));
        return ms;
    }

private:
    void destroy() noexcept {
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
