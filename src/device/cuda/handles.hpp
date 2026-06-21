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
#include <cusolverDn.h>    // cusolverDnHandle_t (the qpAdm fit small-LA: potrf/getrf/gesvdj)
#include <cuda_runtime.h>  // cudaGetDevice (debug device-ordinal record-and-assert)

#include <atomic>
#include <cstddef>
#include <utility>

#include "core/internal/host_device.hpp"  // STEPPE_DEBUG_ONLY, STEPPE_ASSERT (the one debug gate)
#include "core/internal/log.hpp"          // STEPPE_LOG_WARN (the one teardown-warning sink)
#include "device/cuda/check.cuh"          // STEPPE_CUDA_CHECK, CUBLAS_CHECK, CublasError
#include "steppe/config.hpp"              // steppe::Precision (the cuSOLVER promotion seam input)

// ---------------------------------------------------------------------------
// cuSOLVER FP64-EMULATED math-mode capability probe (the promotion-seam guard).
//
// The cuBLAS f2 path engages `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` (cublas_api.h)
// — that enum is present in CUDA 13.x cuBLAS. cuSOLVER's `cusolverMathMode_t`
// (cusolver_common.h) is a SEPARATE, NARROWER enum: as of CUDA 13.0 / cuSOLVER
// 12.0 on the box it carries only `CUSOLVER_DEFAULT_MATH` and
// `CUSOLVER_FP32_EMULATED_BF16X9_MATH` — there is NO FP64-emulated cuSOLVER mode
// yet (verified against /usr/local/cuda/include/cusolver_common.h on box5090).
// So the seam cannot HARDCODE an FP64-emulated cuSOLVER enum that does not
// compile. This macro names the FP64-emulated cuSOLVER mode IF the toolkit ever
// exposes it (a forward-compatible single point of truth): when present, the
// scope genuinely promotes an honorable EmulatedFp64 solve request to it; when
// absent (today), the scope DEGRADES to native `CUSOLVER_DEFAULT_MATH` with a
// one-shot capability tag — exactly the f2 path's downgrade discipline. Either
// way the scope makes a REAL `cusolverDnSetMathMode` call and restores the prior
// mode (the RAII seam is live today; only the *target* mode is gated).
//
// Detection precedence: an explicit `CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH`
// enumerator (if a future header adds one), else leave it undefined. We do NOT
// invent a numeric value — feeding `cusolverDnSetMathMode` an out-of-range int
// would be undefined cuSOLVER behavior, the opposite of a behavior-preserving
// seam.
#if defined(CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)
#define STEPPE_HAVE_CUSOLVER_FP64_EMULATED 1
#else
#define STEPPE_HAVE_CUSOLVER_FP64_EMULATED 0
#endif

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

/// Owning, move-only dense-cuSOLVER handle — the qpAdm fit's small-LA primitive
/// home (Cholesky potrf/potri for the SPD Qinv, LU getrf/getrs + batched for the
/// ALS/weight/LOO solves, Jacobi gesvdj for the rank-test SVD seed; the FROZEN
/// CONTRACT §1a). Mirrors `CublasHandle` (RAII, move-only, device-ordinal
/// record-and-assert): the context binds to the CUDA context current at
/// `cusolverDnCreate` (cuSOLVER, like cuBLAS §2.1.2), so the caller /
/// `PerGpuResources` selects the device BEFORE constructing it. Unlike
/// `CublasHandle`, cuSOLVER has NO workspace-reset hazard — `cusolverDnSetStream`
/// does not touch any pinned workspace, so `set_stream` does not re-apply anything
/// (workspaces in the modern cuSOLVER dense API are passed per-call). Created once
/// in the backend ctor and reused; the dtor routes a nonzero
/// `cusolverDnDestroy` to the §7 teardown-warning sink (never throws).
class CusolverDnHandle {
public:
    /// Create the dense-cuSOLVER handle ON THE CURRENTLY-CURRENT DEVICE and record
    /// that ordinal (cuSOLVER binds to the current CUDA context at create, like
    /// cuBLAS §2.1.2). Caller/PerGpuResources selects the device BEFORE constructing.
    CusolverDnHandle() {
        STEPPE_CUDA_CHECK(cudaGetDevice(&device_id_));
        CUSOLVER_CHECK(cusolverDnCreate(&h_));
    }
    CusolverDnHandle(CusolverDnHandle&& o) noexcept
        : h_(std::exchange(o.h_, nullptr)), device_id_(o.device_id_) {}
    CusolverDnHandle& operator=(CusolverDnHandle&& o) noexcept {
        if (this != &o) {
            destroy();
            h_ = std::exchange(o.h_, nullptr);
            device_id_ = o.device_id_;
        }
        return *this;
    }
    CusolverDnHandle(const CusolverDnHandle&) = delete;
    CusolverDnHandle& operator=(const CusolverDnHandle&) = delete;
    ~CusolverDnHandle() { destroy(); }

    /// Bind the cuSOLVER stream (no workspace-reset hazard — unlike
    /// `cublasSetStream` it does not touch any pinned workspace, so no re-apply).
    void set_stream(cudaStream_t stream) {
        assert_on_creation_device();
        CUSOLVER_CHECK(cusolverDnSetStream(h_, stream));
    }
    [[nodiscard]] cusolverDnHandle_t get() const noexcept { return h_; }
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

private:
    /// Debug-only record-and-ASSERT that the CURRENTLY-current CUDA device matches
    /// the one this handle was created on (architecture.md §7, §11.4) — identical in
    /// spirit to CublasHandle::assert_on_creation_device. cuSOLVER couples its
    /// context to the device current at `cusolverDnCreate`, so configuring/issuing
    /// work while a different device is current is a multi-GPU bug. NEVER calls
    /// `cudaSetDevice` (no hidden global mutable state). Compiled out under NDEBUG.
    void assert_on_creation_device() const noexcept {
        STEPPE_DEBUG_ONLY(
            int current = -1;
            const cudaError_t e = cudaGetDevice(&current);
            STEPPE_ASSERT(e == cudaSuccess,
                          "cudaGetDevice failed while checking CusolverDnHandle device ordinal");
            STEPPE_ASSERT(current == device_id_,
                          "CusolverDnHandle used on a CUDA device different from the one it was "
                          "created on (cuSOLVER context binds to the creation device, like "
                          "cuBLAS §2.1.2; architecture.md §11.4)"));
    }
    void destroy() noexcept {
        if (h_) {
            const cusolverStatus_t s = cusolverDnDestroy(h_);
            if (s != CUSOLVER_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cusolverDnDestroy at teardown: %s",
                                CusolverError::status_name(s));
            }
        }
        h_ = nullptr;
    }

    cusolverDnHandle_t h_ = nullptr;
    int device_id_ = -1;
};

/// Owning, move-only RAII wrapper for a cuSOLVER `gesvdjInfo_t` — the gesvdj
/// (one-sided Jacobi SVD) parameter structure (architecture.md §2 RAII, §7, §8;
/// cleanup device-cuda-cuda_backend group-14 [14.5]). `cusolverDnCreateGesvdjInfo`
/// heap-allocates the structure (it can return CUSOLVER_STATUS_ALLOC_FAILED —
/// VERIFIED against the CUDA 13.x cuSOLVER docs) and `cusolverDnDestroyGesvdjInfo`
/// releases it; the two are the paired create/destroy, exactly like
/// `cusolverDnCreate`/`cusolverDnDestroy`. The bare-pointer idiom (create →
/// throwing `cusolverDnDgesvdj*`/`STEPPE_CUDA_CHECK` work → destroy) LEAKS the
/// handle if any throwing check between create and destroy unwinds past the
/// destroy line (CUSOLVER_CHECK / STEPPE_CUDA_CHECK throw — check.cuh) — the
/// `gesvdjInfo_t` was the only non-RAII resource in cuda_backend.cu, violating
/// that TU's "RAII for ALL ... handles" standard. This makes it RAII: an
/// exception anywhere in the gesvdj branch still frees it on unwind.
///
/// Created and destroyed per gesvdj SHAPE (cheap host-side config struct, NOT a
/// per-call device alloc), so there is no reuse-across-calls contract to honor
/// here — unlike `CusolverDnHandle`, each gesvdj branch makes a fresh one. Stays
/// stateless w.r.t. device ordinal: a `gesvdjInfo_t` is a plain configuration
/// structure, not bound to a CUDA context (cuSOLVER docs: `info` is host memory),
/// so it carries no device-ordinal record-and-assert. Move-only, mirroring the
/// other handle wrappers above; the dtor NEVER throws — a nonzero destroy status
/// routes to the §7 teardown-warning sink (`STEPPE_LOG_WARN`). A moved-from
/// wrapper owns nothing (`info_ == nullptr`) and is safe to destroy.
class GesvdjInfo {
public:
    /// Create the gesvdj parameter structure (default tolerance/sweeps; the
    /// callers leave them at default). Throws CusolverError via CUSOLVER_CHECK on
    /// CUSOLVER_STATUS_ALLOC_FAILED (a ctor may throw; the dtor may not).
    GesvdjInfo() { CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&info_)); }

    GesvdjInfo(GesvdjInfo&& o) noexcept : info_(std::exchange(o.info_, nullptr)) {}
    GesvdjInfo& operator=(GesvdjInfo&& o) noexcept {
        if (this != &o) {
            destroy();
            info_ = std::exchange(o.info_, nullptr);
        }
        return *this;
    }
    GesvdjInfo(const GesvdjInfo&) = delete;
    GesvdjInfo& operator=(const GesvdjInfo&) = delete;
    ~GesvdjInfo() { destroy(); }

    [[nodiscard]] gesvdjInfo_t get() const noexcept { return info_; }

private:
    void destroy() noexcept {
        // Destructor never throws (architecture.md §7); a nonzero destroy status is
        // reported to the §7 teardown-warning sink, never thrown. A moved-from
        // wrapper (`info_ == nullptr`) destroys nothing.
        if (info_) {
            const cusolverStatus_t s = cusolverDnDestroyGesvdjInfo(info_);
            if (s != CUSOLVER_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cusolverDnDestroyGesvdjInfo at scope exit: %s",
                                CusolverError::status_name(s));
            }
        }
        info_ = nullptr;
    }

    gesvdjInfo_t info_ = nullptr;
};

// ---------------------------------------------------------------------------
// One-shot capability tag for the cuSOLVER FP64-emulated DOWNGRADE — the
// cuSOLVER analogue of f2_block_kernel.cu's `warn_emulated_fp64_downgraded_once`
// (cleanup X-6/B2, T-CAP-1). Emitted AT MOST ONCE per process when an honorable
// EmulatedFp64 SOLVE request is asked for but the toolkit exposes no FP64-emulated
// cuSOLVER math mode (STEPPE_HAVE_CUSOLVER_FP64_EMULATED == 0, the box today), so
// the promotion silently-but-OBSERVABLY degrades to native rather than spamming the
// per-solve hot path. Compiled only on the no-mode lane (the only build where the
// downgrade can fire); a build whose toolkit gains the mode carries no unused
// helper (warnings-as-errors clean). Routes through the ONE warn sink
// (STEPPE_LOG_WARN); the std::atomic_flag makes the one-shot guard thread-safe
// (M4.5 multi-GPU may engage from more than one host thread).
#if !STEPPE_HAVE_CUSOLVER_FP64_EMULATED
namespace detail {
inline void warn_cusolver_emulated_fp64_unavailable_once() {
    static std::atomic_flag emitted = ATOMIC_FLAG_INIT;
    if (!emitted.test_and_set(std::memory_order_relaxed)) {
        STEPPE_LOG_WARN(
            "[capability] EmulatedFp64 SOLVE promotion requested but this CUDA "
            "toolkit's cusolverMathMode_t exposes no FP64-emulated mode "
            "(CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH absent; CUDA 13.0/cuSOLVER "
            "12.0) -> the cuSOLVER solve runs native FP64 [tag: "
            "cusolver_emu_fp64_unavailable]. The reported precision is native FP64, "
            "NOT emulated (architecture.md §9, §12; fit-engine.md §1.4). The "
            "promotion SEAM is live — only the target math mode is gated on toolkit "
            "support, so a newer cuSOLVER promotes with no code change.");
    }
}
}  // namespace detail
#endif  // !STEPPE_HAVE_CUSOLVER_FP64_EMULATED

/// Scoped, RAII restore of a dense-cuSOLVER handle's math mode — the cuSOLVER
/// analogue of `MathModeScope` (architecture.md §12; ROADMAP §6 the fit-solve
/// promotion seam). This is the PROMOTION SEAM the M(fit-4) GPU qpAdm fit
/// deliberately left for later: native FP64 is free for ONE model, but the S8
/// model rotation runs MILLIONS of small Cholesky/SVD/GLS solves where native
/// FP64 on Blackwell is a tensor-core-throughput wall. The seam lets a solve
/// stage be PROMOTED to the emulated-FP64 (Ozaki) tensor-core path WITHOUT
/// changing the default — the qpAdm virtuals still pass native, so the af6a8c2
/// golden parity is unchanged. The deliverable is the CAPABILITY, not a default
/// flip.
///
/// Like `cublasSetMathMode`, `cusolverDnSetMathMode` is STICKY handle state: it
/// persists until changed again. The shared per-device solver handle is reused
/// across stages and across models in the rotation, so an imperative promote
/// would silently leak the emulated mode into the next native (oracle/ill-
/// conditioned) solve. This guard makes the change OBSERVABLY SCOPED: the ctor
/// captures the handle's current mode (`cusolverDnGetMathMode`) and applies the
/// requested one; the dtor restores the captured mode. A native solve inside an
/// emulated rotation, or the §12 oracle recompute, can therefore engage native
/// for its scope and leave the handle exactly as it found it.
///
/// HONORABILITY (DRY with the f2 path). The ctor takes a pre-decided `honorable`
/// flag — the caller computes it via the ONE `emulation_honorable` predicate
/// (f2_block_kernel.cuh) so the cuSOLVER seam and the cuBLAS f2 path can never
/// disagree on whether an EmulatedFp64 request is honored. `honorable == true`
/// requests the FP64-emulated cuSOLVER mode; `false` (native Fp64, Tf32, or an
/// unhonorable EmulatedFp64) requests native. The FP64-emulated cuSOLVER mode is
/// itself further gated on toolkit support (STEPPE_HAVE_CUSOLVER_FP64_EMULATED):
/// absent today, so an honorable request DEGRADES to native with a one-shot tag —
/// the seam still makes a real `cusolverDnSetMathMode` call and restores, it just
/// targets native until the toolkit grows the mode. This is NOT a no-op stub: the
/// get/apply/restore round-trip exercises the real cuSOLVER API on every scope.
///
/// Move-only (so it can be returned/held); the dtor NEVER throws — a nonzero
/// restore status routes to the §7 teardown-warning sink (`STEPPE_LOG_WARN`),
/// mirroring `MathModeScope`. A moved-from scope is inert. Takes the raw
/// `cusolverDnHandle_t` (not a `CusolverDnHandle&`) so it composes at the solve
/// call sites that already hold `solver_.get()`.
class CusolverMathModeScope {
public:
    /// Capture the handle's current math mode and apply the requested one
    /// (FP64-emulated for an honorable EmulatedFp64 request where the toolkit
    /// supports it; native `CUSOLVER_DEFAULT_MATH` otherwise). Throws
    /// CusolverError via CUSOLVER_CHECK if the get/set fails (a ctor may throw;
    /// the dtor may not). The captured mode is restored at scope exit.
    CusolverMathModeScope(cusolverDnHandle_t handle, bool honorable) : h_(handle) {
        CUSOLVER_CHECK(cusolverDnGetMathMode(h_, &prev_));
        CUSOLVER_CHECK(cusolverDnSetMathMode(h_, requested_mode(honorable)));
    }

    CusolverMathModeScope(CusolverMathModeScope&& o) noexcept
        : h_(std::exchange(o.h_, nullptr)), prev_(o.prev_) {}

    CusolverMathModeScope& operator=(CusolverMathModeScope&& o) noexcept {
        if (this != &o) {
            restore();
            h_ = std::exchange(o.h_, nullptr);
            prev_ = o.prev_;
        }
        return *this;
    }

    CusolverMathModeScope(const CusolverMathModeScope&) = delete;
    CusolverMathModeScope& operator=(const CusolverMathModeScope&) = delete;

    ~CusolverMathModeScope() { restore(); }

    /// True iff this scope ACTUALLY engaged the FP64-emulated cuSOLVER mode (an
    /// honorable request AND a toolkit that exposes the mode). When false for an
    /// honorable request, the solve ran native (the one-shot tag was emitted) —
    /// the observable promotion state, for tests/logging.
    [[nodiscard]] bool promoted() const noexcept { return promoted_; }

private:
    /// Decide the target cuSOLVER math mode from the honorability flag, gating the
    /// FP64-emulated mode on toolkit support. Records `promoted_` and emits the
    /// one-shot downgrade tag when an honorable request cannot be honored by the
    /// toolkit. Never returns an out-of-range enum.
    cusolverMathMode_t requested_mode(bool honorable) noexcept {
        if (honorable) {
#if STEPPE_HAVE_CUSOLVER_FP64_EMULATED
            promoted_ = true;
            return CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH;
#else
            // The seam is live but the toolkit has no FP64-emulated cuSOLVER mode:
            // degrade to native, observably (one-shot tag), exactly like the f2
            // path's STEPPE_HAVE_EMU_TUNING-off downgrade.
            detail::warn_cusolver_emulated_fp64_unavailable_once();
            promoted_ = false;
            return CUSOLVER_DEFAULT_MATH;
#endif
        }
        promoted_ = false;
        return CUSOLVER_DEFAULT_MATH;  // native Fp64 oracle / fallback
    }

    void restore() noexcept {
        // Destructor never throws (architecture.md §7); a nonzero restore status is
        // reported to the debug-only warning sink, never thrown. A moved-from scope
        // (`h_ == nullptr`) restores nothing.
        if (h_) {
            const cusolverStatus_t s = cusolverDnSetMathMode(h_, prev_);
            if (s != CUSOLVER_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cusolverDnSetMathMode restore at scope exit: %s",
                                CusolverError::status_name(s));
            }
        }
        h_ = nullptr;
    }

    cusolverDnHandle_t h_ = nullptr;  // non-owning: CusolverDnHandle owns the context
    cusolverMathMode_t prev_ = CUSOLVER_DEFAULT_MATH;  // captured mode to restore
    bool promoted_ = false;            // did we actually engage the emulated mode?
};

/// Build a `CusolverMathModeScope` from a typed `Precision`, routing the
/// honorability decision through the SAME `emulation_honorable` predicate the f2
/// path uses (declared in f2_block_kernel.cuh; defined in f2_block_kernel.cu).
/// This is the one-line call the qpAdm solve sites use to engage the seam: the
/// DEFAULT qpAdm virtuals pass native `Fp64` ⇒ `emulation_honorable == false` ⇒
/// the scope targets native and the af6a8c2 golden parity is unchanged. A future
/// per-stage policy can pass `EmulatedFp64{40}` to PROMOTE that one solve. Defined
/// in handles.hpp inline; takes `emulation_honorable` as a function pointer so this
/// header need not include the device-private kernel header (the caller passes
/// `&emulation_honorable`).
[[nodiscard]] inline CusolverMathModeScope engage_solver_precision(
    cusolverDnHandle_t handle, const Precision& precision,
    bool (*honorable_predicate)(const Precision&) noexcept) {
    return CusolverMathModeScope(handle, honorable_predicate(precision));
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_HANDLES_HPP
