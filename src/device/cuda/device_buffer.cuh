// src/device/cuda/device_buffer.cuh
//
// DeviceBuffer<T> — THE move-only RAII owner of device memory (architecture.md
// §2 RAII, §7; ROADMAP §5). This is one of the THREE allowlisted translation
// units permitted to call `cudaMalloc`/`cudaFree` directly (architecture.md §2
// DRY grep gate: `device_buffer.cuh`, `allocator.cu`, `pinned_buffer.cuh`); all
// other code takes non-owning views and never touches the allocation family.
//
// Replaces the spike's raw `cudaMalloc`/`cudaFree` pairs scattered through
// f2_emu_spike.cu / f2_timing.cu (ROADMAP §1 "no RAII"). Fully move-only:
// move-construct AND move-assign, so `buf = std::move(other)` is well-formed
// (architecture.md §7 — the old draft's deleted move-assign was a bug). The
// destructor NEVER throws, but it is not silent: a nonzero `cudaFree` status at
// teardown is routed to a debug-only warning so "fail-fast" does not become
// "fail-silent at teardown" (architecture.md §2, §7, §10).
//
// The allocating ctor is also fail-fast on a SIZE OVERFLOW: `n * sizeof(T)` is a
// `std::size_t` product whose unsigned overflow is *defined* (modular), so an
// unchecked multiply would silently WRAP and under-allocate rather than trap.
// The ctor rejects any `n` for which the byte product would exceed `SIZE_MAX`
// with a typed `CudaError` (cleanup B23, architecture.md §2 fail-fast, §11.2),
// which makes `bytes()` exact for the §11.2 VRAM budget.
//
// This is a CUDA header: PRIVATE to steppe_device (architecture.md §4).
#ifndef STEPPE_DEVICE_CUDA_DEVICE_BUFFER_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_BUFFER_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>           // std::numeric_limits — the n*sizeof(T) overflow guard
#include <source_location>  // std::source_location — call site for the typed throw
#include <utility>

#include "core/internal/log.hpp"  // STEPPE_LOG_WARN (the one teardown-warning sink)
#include "device/cuda/check.cuh"  // STEPPE_CUDA_CHECK, CudaError

namespace steppe::device {

/// Owning, move-only device allocation of `n` elements of `T`. Hands out a raw
/// device pointer via `data()` for cuBLAS / kernel arguments; never copies.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    /// Allocate `n` elements on the current device. `n == 0` is a no-op (null
    /// pointer, zero size) — not an error.
    ///
    /// FAIL-FAST on a size overflow (architecture.md §2): the byte request
    /// `n * sizeof(T)` is a `std::size_t` product, and unsigned overflow is
    /// *defined* (modular) — so an unchecked multiply would NOT trap, it would
    /// SILENTLY WRAP to a small value, `cudaMalloc` would succeed with a buffer
    /// far smaller than `size_` advertises, and every downstream kernel/copy
    /// would over-run it (silent heap corruption — the exact opposite of §2's
    /// "fail-fast, not silent corruption"). `n` is NOT bounded by hardware at
    /// this layer (cuda_backend.cu forms buffer sizes as products of three host
    /// values BEFORE any §11.2 VRAM-budget check sees them), so the owner is the
    /// single-source place to make this safe once. We reject any `n` for which
    /// `n * sizeof(T)` would exceed `SIZE_MAX` with a typed `CudaError` carrying
    /// `cudaErrorMemoryAllocation` — which the public API maps to
    /// `STEPPE_ERR_DEVICE_OOM` (architecture.md §10): the request is, by
    /// definition, larger than any allocatable size.
    explicit DeviceBuffer(std::size_t n) : size_(n) {
        if (n) {
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                // Synthesize the typed CudaError directly (the throwing
                // STEPPE_CUDA_CHECK only fires on a CUDA runtime status); the
                // call site is captured for the message, exactly like the macro.
                throw CudaError(cudaErrorMemoryAllocation,
                                "DeviceBuffer: n * sizeof(T) overflows size_t",
                                std::source_location::current());
            }
            STEPPE_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));  // ALLOWLISTED TU
        }
    }

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), size_(std::exchange(o.size_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = std::exchange(o.ptr_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() { reset(); }

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// Exact logical byte footprint, `size_ * sizeof(T)`. This product can NEVER
    /// overflow `std::size_t`: the ctor rejects any `n` for which
    /// `n * sizeof(T)` would exceed `SIZE_MAX`, so a constructed buffer's `size_`
    /// always satisfies `size_ <= SIZE_MAX / sizeof(T)`. `bytes()` is therefore
    /// the *exact* logical size every consumer reasons about — in particular the
    /// §11.2 VRAM-budget check that sums the resident terms relies on it being
    /// exact (a wrapped `bytes()` would corrupt the budget total in the
    /// too-small, safe-looking direction). It is the logical request, NOT
    /// `cudaMalloc`'s internally rounded-up (≥256-byte) actual allocation.
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }

private:
    void reset() noexcept {
        if (ptr_) {
            // DEVICE-AGNOSTIC FREE — THE single-home deleter for the M4.5 multi-GPU
            // escape design (architecture.md §7, §11.4; cleanup 17.5). This owner
            // records ONLY ptr_/size_, NOT the device ordinal that was current at
            // cudaMalloc, and reset() issues a BARE cudaFree with NO record-and-
            // restore cudaSetDevice — INTENTIONALLY. The M4.5 design deliberately
            // lets a buffer ESCAPE its device-guarded producer (it moves into a
            // DeviceF2Blocks/DevicePartial and is freed LATER by the host-side
            // combine under a possibly-different — typically entry/device-0 —
            // ambient device). That cross-device free is SOUND because cudaFree is
            // POINTER-DEVICE-AWARE: it frees the allocation regardless of which
            // device is current. VERIFIED against the CUDA 13.x Runtime API
            // (CUDART_MEMORY): the cudaFree description is SILENT on the current
            // device — it neither requires nor forbids the alloc device be current —
            // so this is an unspecified-but-relied-on invariant, pinned here as the
            // single home (this is also the [17.5] note for DevicePartial /
            // DeviceF2Blocks, satisfied by this comment — no separate edit there).
            // Adding cudaSetDevice here is barred by design (no hidden global state
            // in the owner — that is the caller's / Resources' job), and would be
            // wrong for the escape path anyway.
            //
            // RE-TRIGGER WARNING: a future switch to cudaMallocAsync / a per-device
            // memory pool makes the free DEVICE-CURRENT-SENSITIVE — cudaFreeAsync is
            // stream/pool-ordered and tied to a device's memory pool (CUDA 13.x
            // CUDART_MEMORY) — so that change RE-IMPOSES a record-and-restore
            // requirement HERE (record the alloc device in the ctor, set+restore it
            // around the free), invalidating the device-agnostic premise above.
            //
            // Destructor never throws (architecture.md §7); a nonzero teardown
            // status is reported to the debug-only warning sink (the §7
            // teardown-warning behavior — fail-fast must not become fail-silent),
            // never thrown. cudaFree(nullptr) is a no-op, hence the guard.
            const cudaError_t e = cudaFree(ptr_);  // ALLOWLISTED TU
            // At static-destruction / atexit the CUDA primary context may already be
            // torn down, in which case cudaFree returns cudaErrorCudartUnloading (or
            // cudaErrorContextIsDestroyed) for memory the OS reclaims regardless —
            // a non-real "leak". Treat those two teardown-order codes as BENIGN
            // (skip the warn) so a clean process exit does not emit a spurious
            // teardown WARN (cleanup [17.1]). steppe's DeviceBuffers are normally
            // backend-owned (not file-scope statics), so this rarely fires.
            if (e != cudaSuccess && e != cudaErrorCudartUnloading &&
                e != cudaErrorContextIsDestroyed) {
                STEPPE_LOG_WARN("cudaFree at DeviceBuffer teardown: %s",
                                cudaGetErrorString(e));
            }
        }
        ptr_ = nullptr;
        size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DEVICE_BUFFER_CUH
