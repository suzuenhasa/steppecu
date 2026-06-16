// src/device/cuda/handles.hpp
//
// RAII cuBLAS handle wrapper (architecture.md §2 RAII, §7, §8, §12; ROADMAP §5).
//
// Replaces the spike's bare `cublasCreate`/`cublasDestroy` pairs (one per .cu in
// the experiments, ROADMAP §1). Created ONCE at startup and reused —
// `cublasDestroy` implicitly synchronizes, so a handle is never recreated
// per-iteration (architecture.md §7). Fully move-only (move-construct AND
// move-assign, architecture.md §7). The destructor never throws, but routes a
// nonzero destroy status to a debug-only warning (architecture.md §2, §7, §10).
//
// THE (stream, workspace) INVARIANT (architecture.md §12; cleanup X-1/B1). The
// emulated-FP64 path needs a non-default cuBLAS workspace pinned via
// `cublasSetWorkspace` for run-to-run reproducibility (the FIXED-slice Ozaki
// path; cuBLAS §2.1.4 Results Reproducibility). But `cublasSetStream`
// "unconditionally resets the cuBLAS library workspace back to the default
// workspace pool" (cuBLAS §2.4.7, CUDA 13.x), so ANY stream change AFTER the
// workspace is bound silently discards it — and the §12 determinism guarantee
// the goldens depend on (architecture.md §13) is defeated. The hazard cannot be
// fixed at the call sites (it is order-dependent and invisible there); it must
// be owned by the type. So `CublasHandle` holds the (non-owning) workspace span
// and `set_stream()` RE-APPLIES it after every `cublasSetStream`. Callers must
// route stream changes through `set_stream()` and never call raw
// `cublasSetStream(get(), …)`.
//
// Despite the `.hpp` extension (matching architecture.md §4's `handles.hpp`
// entry under src/device/cuda/), this includes `cublas_v2.h` and so is a CUDA
// header: PRIVATE to steppe_device, never compiled into core / api / the CLI
// (architecture.md §4 layering rule).
#ifndef STEPPE_DEVICE_CUDA_HANDLES_HPP
#define STEPPE_DEVICE_CUDA_HANDLES_HPP

#include <cublas_v2.h>

#include <cstddef>
#include <utility>

#include "core/internal/log.hpp"  // STEPPE_LOG_WARN (the one teardown-warning sink)
#include "device/cuda/check.cuh"  // CUBLAS_CHECK, CublasError

namespace steppe::device {

/// Owning, move-only cuBLAS handle that also owns the (stream, workspace)
/// invariant the §12 emulated-FP64 determinism contract depends on. Create once;
/// reuse for every GEMM. Bind the workspace once via `set_workspace()`, the
/// single statistic stream once via `set_stream()` (architecture.md §12), then
/// hand `get()` only to the `cublasGemm*` compute entry points — never to a raw
/// `cublasSetStream`, which would discard the workspace (cuBLAS §2.4.7).
class CublasHandle {
public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&h_)); }

    CublasHandle(CublasHandle&& o) noexcept
        : h_(std::exchange(o.h_, nullptr)),
          ws_(std::exchange(o.ws_, nullptr)),
          ws_bytes_(std::exchange(o.ws_bytes_, 0)) {}

    CublasHandle& operator=(CublasHandle&& o) noexcept {
        if (this != &o) {
            destroy();
            h_ = std::exchange(o.h_, nullptr);
            ws_ = std::exchange(o.ws_, nullptr);
            ws_bytes_ = std::exchange(o.ws_bytes_, 0);
        }
        return *this;
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    ~CublasHandle() { destroy(); }

    /// Pin the cuBLAS workspace for emulated-FP64 reproducibility (cuBLAS §2.1.4;
    /// architecture.md §12). The handle remembers the (ptr, bytes) so it can
    /// RE-APPLY it after any `set_stream()` (cuBLAS §2.4.7 resets it otherwise).
    /// The buffer is NON-owning: the caller (the backend's `DeviceBuffer`) must
    /// outlive the handle's use of it — destruction is reverse declaration order,
    /// so the backend declares the handle BEFORE the workspace buffer.
    void set_workspace(void* ptr, std::size_t bytes) {
        ws_ = ptr;
        ws_bytes_ = bytes;
        CUBLAS_CHECK(cublasSetWorkspace(h_, ws_, ws_bytes_));
    }

    /// Bind the cuBLAS stream AND re-apply the owned workspace (architecture.md
    /// §12; cleanup X-1/B1). `cublasSetStream` "unconditionally resets the cuBLAS
    /// library workspace back to the default workspace pool" (cuBLAS §2.4.7), so
    /// re-applying the pinned workspace here is what keeps the emulated-FP64
    /// determinism guarantee intact across stream changes. The single statistic
    /// stream is bound once at startup (architecture.md §12); this method exists
    /// so the invariant lives in ONE place rather than at every GEMM call site.
    void set_stream(cudaStream_t stream) {
        CUBLAS_CHECK(cublasSetStream(h_, stream));
        if (ws_) CUBLAS_CHECK(cublasSetWorkspace(h_, ws_, ws_bytes_));
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return h_; }

private:
    void destroy() noexcept {
        // Destructor never throws (architecture.md §7); a nonzero destroy status
        // is reported to the debug-only warning sink (the §7 teardown-warning
        // behavior — fail-fast must not become fail-silent), never thrown.
        if (h_) {
            const cublasStatus_t s = cublasDestroy(h_);
            if (s != CUBLAS_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cublasDestroy at teardown: %s",
                                CublasError::status_name(s));
            }
        }
        h_ = nullptr;
        // The workspace is NON-owning (cleared, not freed): the backend's
        // DeviceBuffer owns the VRAM and frees it in its own dtor.
        ws_ = nullptr;
        ws_bytes_ = 0;
    }

    cublasHandle_t h_ = nullptr;
    void* ws_ = nullptr;          // non-owning cuBLAS workspace (architecture.md §12)
    std::size_t ws_bytes_ = 0;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_HANDLES_HPP
