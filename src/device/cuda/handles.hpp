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
// THE DEVICE-ORDINAL INVARIANT (architecture.md §2.1.2, §9, §11.4; cleanup
// device-cuda-handles 2.3/11.x, overview §(2).1). "A cuBLAS library context is
// tightly coupled with the CUDA context that is current at the time of the
// `cublasCreate()` call" (cuBLAS §2.1.2, CUDA 13.x). The §9 `PerGpuResources`
// holds one `CublasHandle` PER device and §11.4 switches devices with
// `cudaSetDevice`; a handle later used while a DIFFERENT device is current runs
// its GEMMs on the wrong GPU or fails `CUBLAS_STATUS_ARCH_MISMATCH`. So the ctor
// RECORDS the device ordinal that was current at creation (`cudaGetDevice`), and
// every use that mutates the cuBLAS context debug-ASSERTs the current device
// still matches it. This is record-and-ASSERT, NEVER `cudaSetDevice`: the wrapper
// must not introduce hidden global mutable state (architecture.md §7) — selecting
// the device is the caller's / `Resources`' job. This is scaffolding for the
// M4.5 multi-GPU pass (no sharding here yet); on single-GPU the assert is a
// no-op-true and parity is unaffected (§12 — observability only).
//
// Despite the `.hpp` extension (matching architecture.md §4's `handles.hpp`
// entry under src/device/cuda/), this includes `cublas_v2.h` and so is a CUDA
// header: PRIVATE to steppe_device, never compiled into core / api / the CLI
// (architecture.md §4 layering rule).
#ifndef STEPPE_DEVICE_CUDA_HANDLES_HPP
#define STEPPE_DEVICE_CUDA_HANDLES_HPP

#include <cublas_v2.h>
#include <cuda_runtime.h>  // cudaGetDevice (debug device-ordinal record-and-assert)

#include <cstddef>
#include <utility>

#include "core/internal/host_device.hpp"  // STEPPE_DEBUG_ONLY, STEPPE_ASSERT (the one debug gate)
#include "core/internal/log.hpp"          // STEPPE_LOG_WARN (the one teardown-warning sink)
#include "device/cuda/check.cuh"          // STEPPE_CUDA_CHECK, CUBLAS_CHECK, CublasError

namespace steppe::device {

/// Owning, move-only cuBLAS handle that also owns the (stream, workspace)
/// invariant the §12 emulated-FP64 determinism contract depends on, and records
/// the device ordinal it was created on for the §11.4 multi-GPU invariant. Create
/// once; reuse for every GEMM. Bind the workspace once via `set_workspace()`, the
/// single statistic stream once via `set_stream()` (architecture.md §12), then
/// hand `get()` only to the `cublasGemm*` compute entry points — never to a raw
/// `cublasSetStream`, which would discard the workspace (cuBLAS §2.4.7). A
/// moved-from handle owns nothing (`h_ == nullptr`) and is safe to destroy
/// (mirroring `Stream`/`DeviceBuffer`).
class CublasHandle {
public:
    /// Create the handle ON THE CURRENTLY-CURRENT DEVICE and record that ordinal.
    /// The cuBLAS context binds to the current CUDA context here (cuBLAS §2.1.2),
    /// so the recorded ordinal is the device every subsequent GEMM must run on.
    /// The ordinal capture is a `cudaGetDevice` query (no device mutation); the
    /// caller / `PerGpuResources` is responsible for having selected the intended
    /// device BEFORE constructing the handle (architecture.md §9, §11.4).
    CublasHandle() {
        STEPPE_CUDA_CHECK(cudaGetDevice(&device_id_));
        CUBLAS_CHECK(cublasCreate(&h_));
    }

    CublasHandle(CublasHandle&& o) noexcept
        : h_(std::exchange(o.h_, nullptr)),
          ws_(std::exchange(o.ws_, nullptr)),
          ws_bytes_(std::exchange(o.ws_bytes_, 0)),
          device_id_(o.device_id_) {}

    CublasHandle& operator=(CublasHandle&& o) noexcept {
        if (this != &o) {
            destroy();
            h_ = std::exchange(o.h_, nullptr);
            ws_ = std::exchange(o.ws_, nullptr);
            ws_bytes_ = std::exchange(o.ws_bytes_, 0);
            device_id_ = o.device_id_;
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
        assert_on_creation_device();
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
        assert_on_creation_device();
        CUBLAS_CHECK(cublasSetStream(h_, stream));
        if (ws_) CUBLAS_CHECK(cublasSetWorkspace(h_, ws_, ws_bytes_));
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return h_; }

    /// The CUDA device ordinal this handle's cuBLAS context is bound to — the
    /// device current at construction (cuBLAS §2.1.2). The §11.4 multi-GPU
    /// orchestration uses it to log which tier each per-device handle is on and to
    /// assert the right device is current before use; on single-GPU it is always 0.
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

private:
    /// Debug-only record-and-ASSERT that the CURRENTLY-current CUDA device matches
    /// the one this handle was created on (architecture.md §7 record-and-assert,
    /// §11.4). cuBLAS couples the context to the device current at `cublasCreate`
    /// (cuBLAS §2.1.2), so configuring or issuing work on the handle while a
    /// different device is current is a multi-GPU bug. This NEVER calls
    /// `cudaSetDevice` (no hidden global mutable state in the wrapper — that is the
    /// caller's job); it only verifies the precondition. Compiled out under NDEBUG
    /// (STEPPE_DEBUG_ONLY): the query + assert add nothing to the release hot path.
    void assert_on_creation_device() const noexcept {
        STEPPE_DEBUG_ONLY(
            int current = -1;
            // Query only (no device mutation). A query failure here is itself a
            // debug-only assert: this path runs only in debug builds.
            const cudaError_t e = cudaGetDevice(&current);
            STEPPE_ASSERT(e == cudaSuccess,
                          "cudaGetDevice failed while checking CublasHandle device ordinal");
            STEPPE_ASSERT(current == device_id_,
                          "CublasHandle used on a CUDA device different from the one it was "
                          "created on (cuBLAS context is bound to the creation device, "
                          "cuBLAS §2.1.2; architecture.md §11.4)"));
    }

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
    // The CUDA device ordinal current at construction; the device the cuBLAS
    // context is bound to (cuBLAS §2.1.2). Carried through moves (the moved-into
    // handle owns the same context). Initialized in the ctor body via
    // cudaGetDevice (the -1 is a defensive sentinel never observed: the ctor
    // always sets it before any use, and the body throws if cudaGetDevice fails).
    int device_id_ = -1;
};

/// Scoped, RAII restore of a cuBLAS handle's math mode (architecture.md §12;
/// cleanup device-cuda-f2_block_kernel N-5, TODO M4.5 line 98; L10).
///
/// `cublasSetMathMode` is STICKY handle state: it stays set until changed again
/// (unlike the workspace, it is NOT reset by `cublasSetStream` — cuBLAS §2.4.7
/// resets only the workspace). The f2 path engages a math mode once per compute
/// call (`engage_f2_precision`: `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` for an
/// honorable `EmulatedFp64`, else `CUBLAS_PEDANTIC_MATH`) and never restores it —
/// benign while one precision owns the handle, but a determinism hazard the
/// moment the §12 mandatory gate recomputes a sample of jackknife blocks in
/// native `Fp64` (PEDANTIC) on the SAME shared handle that an `EmulatedFp64` run
/// is using: whichever ran last silently leaks its math mode into the next.
///
/// This guard makes a math-mode change OBSERVABLY SCOPED: the ctor captures the
/// handle's current mode (`cublasGetMathMode`) and applies a requested one; the
/// dtor restores the captured mode. So the Fp64 parity-recompute can engage
/// PEDANTIC for its scope and leave the handle exactly as it found it — the
/// EmulatedFp64 mode the surrounding run depends on is intact afterward. This is
/// the M4.5 scaffold (architecture.md §12 oracle pass "use
/// `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`"); it is parity-NEUTRAL
/// (it only restores state that was already being set imperatively, §12).
///
/// Move-only (so it can be returned/held); the dtor NEVER throws — a nonzero
/// restore status routes to the §7 teardown-warning sink (`STEPPE_LOG_WARN`),
/// consistent with the RAII wrappers above. A moved-from scope is inert (it
/// restores nothing). Takes the raw `cublasHandle_t` (not a `CublasHandle&`) so
/// it composes with `engage_f2_precision`, which already operates on the raw
/// handle.
class MathModeScope {
public:
    /// Capture the handle's current math mode and apply `requested`. Throws
    /// CublasError via CUBLAS_CHECK if the get/set fails (a constructor may throw;
    /// the dtor may not). The captured mode is restored at scope exit.
    MathModeScope(cublasHandle_t handle, cublasMath_t requested) : h_(handle) {
        CUBLAS_CHECK(cublasGetMathMode(h_, &prev_));
        CUBLAS_CHECK(cublasSetMathMode(h_, requested));
    }

    MathModeScope(MathModeScope&& o) noexcept
        : h_(std::exchange(o.h_, nullptr)), prev_(o.prev_) {}

    MathModeScope& operator=(MathModeScope&& o) noexcept {
        if (this != &o) {
            restore();
            h_ = std::exchange(o.h_, nullptr);
            prev_ = o.prev_;
        }
        return *this;
    }

    MathModeScope(const MathModeScope&) = delete;
    MathModeScope& operator=(const MathModeScope&) = delete;

    ~MathModeScope() { restore(); }

private:
    void restore() noexcept {
        // Destructor never throws (architecture.md §7); a nonzero restore status is
        // reported to the debug-only warning sink (the §7 teardown-warning
        // behavior — fail-fast must not become fail-silent), never thrown. A
        // moved-from scope (`h_ == nullptr`) restores nothing.
        if (h_) {
            const cublasStatus_t s = cublasSetMathMode(h_, prev_);
            if (s != CUBLAS_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cublasSetMathMode restore at scope exit: %s",
                                CublasError::status_name(s));
            }
        }
        h_ = nullptr;
    }

    cublasHandle_t h_ = nullptr;  // non-owning: the CublasHandle owns the context
    cublasMath_t prev_ = CUBLAS_DEFAULT_MATH;  // captured mode to restore (defensive default)
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_HANDLES_HPP
