// src/device/cuda/handles.hpp
//
// RAII cuBLAS handle wrapper (architecture.md §2 RAII, §7, §8; ROADMAP §5).
//
// Replaces the spike's bare `cublasCreate`/`cublasDestroy` pairs (one per .cu in
// the experiments, ROADMAP §1). Created ONCE at startup and reused —
// `cublasDestroy` implicitly synchronizes, so a handle is never recreated
// per-iteration (architecture.md §7). Fully move-only (move-construct AND
// move-assign, architecture.md §7). The destructor never throws, but routes a
// nonzero destroy status to a debug-only warning (architecture.md §2, §7, §10).
//
// Despite the `.hpp` extension (matching architecture.md §4's `handles.hpp`
// entry under src/device/cuda/), this includes `cublas_v2.h` and so is a CUDA
// header: PRIVATE to steppe_device, never compiled into core / api / the CLI
// (architecture.md §4 layering rule).
#ifndef STEPPE_DEVICE_CUDA_HANDLES_HPP
#define STEPPE_DEVICE_CUDA_HANDLES_HPP

#include <cublas_v2.h>

#include <utility>

#include "device/cuda/check.cuh"  // CUBLAS_CHECK, CublasError

// Debug-only teardown warning sink, mirroring device_buffer.cuh. Destroy-time
// failures route through internal/log.hpp's STEPPE_LOG_WARN (architecture.md §7,
// §10) once that facade is wired into steppe_device; until then, warn to stderr in
// debug builds only and stay silent in release. The destructor must never throw.
#if defined(NDEBUG)
#  define STEPPE_HANDLES_WARN_ON_TEARDOWN(what, statusname) ((void)0)
#else
#  include <cstdio>
#  define STEPPE_HANDLES_WARN_ON_TEARDOWN(what, statusname) \
    std::fprintf(stderr, "[steppe][warn] %s at teardown: %s\n", (what), (statusname))
#endif

namespace steppe::device {

/// Owning, move-only cuBLAS handle. Optionally bound to a stream at construction
/// (single statistic stream on the bit-stable path, architecture.md §12). Create
/// once; reuse for every GEMM.
class CublasHandle {
public:
    explicit CublasHandle(cudaStream_t stream = nullptr) {
        CUBLAS_CHECK(cublasCreate(&h_));
        if (stream) CUBLAS_CHECK(cublasSetStream(h_, stream));
    }

    CublasHandle(CublasHandle&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}

    CublasHandle& operator=(CublasHandle&& o) noexcept {
        if (this != &o) {
            destroy();
            h_ = std::exchange(o.h_, nullptr);
        }
        return *this;
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    ~CublasHandle() { destroy(); }

    [[nodiscard]] cublasHandle_t get() const noexcept { return h_; }

private:
    void destroy() noexcept {
        // Destructor never throws (architecture.md §7); a nonzero destroy status
        // is reported to the debug-only warning sink (the §7 teardown-warning
        // behavior — fail-fast must not become fail-silent), never thrown.
        if (h_) {
            const cublasStatus_t s = cublasDestroy(h_);
            if (s != CUBLAS_STATUS_SUCCESS) {
                STEPPE_HANDLES_WARN_ON_TEARDOWN("cublasDestroy",
                                                CublasError::status_name(s));
            }
        }
        h_ = nullptr;
    }

    cublasHandle_t h_ = nullptr;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_HANDLES_HPP
