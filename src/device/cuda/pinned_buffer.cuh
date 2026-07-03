// src/device/cuda/pinned_buffer.cuh
//
// Move-only RAII owners of PAGE-LOCKED (pinned) host memory: PinnedBuffer<T>,
// RegisteredHostRegion, and the small PinnedRegistryCache. One of the three
// allowlisted translation units permitted to call the host-side allocation
// family (cudaHostAlloc/cudaHostRegister) directly.
//
// Reference: docs/reference/src_device_cuda_pinned_buffer.cuh.md
#ifndef STEPPE_DEVICE_CUDA_PINNED_BUFFER_CUH
#define STEPPE_DEVICE_CUDA_PINNED_BUFFER_CUH

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <limits>
#include <source_location>
#include <type_traits>
#include <utility>

#include "core/internal/host_device.hpp"
#include "core/internal/log.hpp"
#include "device/cuda/check.cuh"

namespace steppe::device {

// PinnedBuffer<T> — owning pinned host allocation — reference §3
template <class T>
    requires std::is_trivially_copyable_v<T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t n) : size_(n) {
        if (n) {
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw CudaError(cudaErrorMemoryAllocation,
                                "PinnedBuffer: n * sizeof(T) overflows size_t",
                                std::source_location::current());
            }
            STEPPE_CUDA_CHECK(
                cudaHostAlloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T),
                              cudaHostAllocDefault));
        }
    }

    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), size_(std::exchange(o.size_, 0)) {}

    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = std::exchange(o.ptr_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() { reset(); }

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }

private:
    void reset() noexcept {
        if (ptr_) {
            const cudaError_t e = cudaFreeHost(ptr_);
            if (e != cudaSuccess) {
                STEPPE_LOG_WARN("cudaFreeHost at PinnedBuffer teardown: %s",
                                cudaGetErrorString(e));
            }
        }
        ptr_ = nullptr;
        size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// RegisteredHostRegion — pin an existing host buffer in place — reference §4
class RegisteredHostRegion {
public:
    RegisteredHostRegion() = default;

    RegisteredHostRegion(const void* ptr, std::size_t bytes) {
        if (ptr == nullptr || bytes == 0) return;
        void* const p = const_cast<void*>(ptr);
        const cudaError_t s =
            STEPPE_CUDA_WARN(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
        if (s == cudaSuccess) {
            ptr_ = p;
        } else {
            (void)cudaGetLastError();
        }
    }

    RegisteredHostRegion(RegisteredHostRegion&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)) {}

    RegisteredHostRegion& operator=(RegisteredHostRegion&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = std::exchange(o.ptr_, nullptr);
        }
        return *this;
    }

    RegisteredHostRegion(const RegisteredHostRegion&) = delete;
    RegisteredHostRegion& operator=(const RegisteredHostRegion&) = delete;

    ~RegisteredHostRegion() { reset(); }

    [[nodiscard]] bool registered() const noexcept { return ptr_ != nullptr; }

private:
    void reset() noexcept {
        if (ptr_) {
            const cudaError_t e = cudaHostUnregister(ptr_);
            if (e != cudaSuccess) {
                STEPPE_LOG_WARN("cudaHostUnregister at teardown: %s",
                                cudaGetErrorString(e));
            }
        }
        ptr_ = nullptr;
    }

    void* ptr_ = nullptr;
};

// PinnedRegistryCache — amortized cache of persistent registrations — reference §5
class PinnedRegistryCache {
public:
    PinnedRegistryCache() = default;

    PinnedRegistryCache(PinnedRegistryCache&&) noexcept = default;
    PinnedRegistryCache& operator=(PinnedRegistryCache&&) noexcept = default;
    PinnedRegistryCache(const PinnedRegistryCache&) = delete;
    PinnedRegistryCache& operator=(const PinnedRegistryCache&) = delete;
    ~PinnedRegistryCache() = default;

    void ensure(const void* ptr, std::size_t bytes) {
        if (ptr == nullptr || bytes == 0) return;
        for (const Slot& s : slots_) {
            if (s.ptr == ptr && s.bytes == bytes && s.reg.registered()) return;
        }
        Slot& dst = slots_[next_];
        STEPPE_ASSERT(!dst.reg.registered(),
                      "PinnedRegistryCache::ensure: evicting a still-registered slot "
                      "(a 4th distinct H2D range exceeds kSlots); the evicted range must "
                      "have no outstanding async copy (no hidden sync is performed here)");
        dst.reg = RegisteredHostRegion(ptr, bytes);
        dst.ptr = dst.reg.registered() ? ptr : nullptr;
        dst.bytes = dst.reg.registered() ? bytes : 0;
        next_ = (next_ + 1) % kSlots;
    }

private:
    static constexpr std::size_t kSlots = 3;
    struct Slot {
        const void* ptr = nullptr;
        std::size_t bytes = 0;
        RegisteredHostRegion reg{};
    };
    std::array<Slot, kSlots> slots_{};
    std::size_t next_ = 0;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_PINNED_BUFFER_CUH
