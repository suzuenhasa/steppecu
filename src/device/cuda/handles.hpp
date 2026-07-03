// src/device/cuda/handles.hpp
//
// Owning, move-only RAII wrappers for the raw GPU-library handles steppe uses —
// cuBLAS, cuSOLVER, and cuFFT — each freeing itself on every exit path (normal
// return and error unwind). Despite the .hpp extension this is a CUDA header
// (includes cublas_v2.h/cufft.h/cusolverDn.h): private to steppe_device, never
// compiled into core / api / the CLI.
//
// Reference: docs/reference/src_device_cuda_handles.hpp.md
#ifndef STEPPE_DEVICE_CUDA_HANDLES_HPP
#define STEPPE_DEVICE_CUDA_HANDLES_HPP

#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <utility>

#include "core/internal/host_device.hpp"
#include "core/internal/log.hpp"
#include "device/cuda/check.cuh"
#include "steppe/config.hpp"

// cuSOLVER FP64-emulated math-mode capability probe — reference §10
#if defined(CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)
#define STEPPE_HAVE_CUSOLVER_FP64_EMULATED 1
#else
#define STEPPE_HAVE_CUSOLVER_FP64_EMULATED 0
#endif

namespace steppe::device {

// CublasHandle: owning, move-only cuBLAS handle (stream/workspace + device-ordinal invariants) — reference §5
class CublasHandle {
public:
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

    void set_workspace(void* ptr, std::size_t bytes) {
        assert_on_creation_device();
        ws_ = ptr;
        ws_bytes_ = bytes;
        CUBLAS_CHECK(cublasSetWorkspace(h_, ws_, ws_bytes_));
    }

    void set_stream(cudaStream_t stream) {
        assert_on_creation_device();
        CUBLAS_CHECK(cublasSetStream(h_, stream));
        if (ws_) CUBLAS_CHECK(cublasSetWorkspace(h_, ws_, ws_bytes_));
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return h_; }

    [[nodiscard]] int device_id() const noexcept { return device_id_; }

private:
    void assert_on_creation_device() const noexcept {
        STEPPE_DEBUG_ONLY(
            int current = -1;
            const cudaError_t e = cudaGetDevice(&current);
            STEPPE_ASSERT(e == cudaSuccess,
                          "cudaGetDevice failed while checking CublasHandle device ordinal");
            STEPPE_ASSERT(current == device_id_,
                          "CublasHandle used on a CUDA device different from the one it was "
                          "created on (cuBLAS context is bound to the creation device, "
                          "cuBLAS §2.1.2; architecture.md §11.4)"));
    }

    void destroy() noexcept {
        assert_on_creation_device();
        if (h_) {
            const cublasStatus_t s = cublasDestroy(h_);
            if (s != CUBLAS_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cublasDestroy at teardown: %s",
                                CublasError::status_name(s));
            }
        }
        h_ = nullptr;
        ws_ = nullptr;
        ws_bytes_ = 0;
    }

    cublasHandle_t h_ = nullptr;
    void* ws_ = nullptr;
    std::size_t ws_bytes_ = 0;
    int device_id_ = -1;
};

// MathModeScope: scoped, RAII restore of a cuBLAS handle's math mode — reference §6
class MathModeScope {
public:
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
        if (h_) {
            const cublasStatus_t s = cublasSetMathMode(h_, prev_);
            if (s != CUBLAS_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cublasSetMathMode restore at scope exit: %s",
                                CublasError::status_name(s));
            }
        }
        h_ = nullptr;
    }

    cublasHandle_t h_ = nullptr;
    cublasMath_t prev_ = CUBLAS_DEFAULT_MATH;
};

// CusolverDnHandle: owning, move-only dense-cuSOLVER handle — reference §7
class CusolverDnHandle {
public:
    CusolverDnHandle() {
        STEPPE_CUDA_CHECK(cudaGetDevice(&device_id_));
        CUSOLVER_CHECK(cusolverDnCreate(&h_));
        CUSOLVER_CHECK(cusolverDnSetDeterministicMode(h_, CUSOLVER_DETERMINISTIC_RESULTS));
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

    void set_stream(cudaStream_t stream) {
        assert_on_creation_device();
        CUSOLVER_CHECK(cusolverDnSetStream(h_, stream));
    }
    [[nodiscard]] cusolverDnHandle_t get() const noexcept { return h_; }
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

    [[nodiscard]] cusolverDeterministicMode_t deterministic_mode() const {
        assert_on_creation_device();
        cusolverDeterministicMode_t mode = CUSOLVER_DETERMINISTIC_RESULTS;
        CUSOLVER_CHECK(cusolverDnGetDeterministicMode(h_, &mode));
        return mode;
    }

private:
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
        assert_on_creation_device();
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

// GesvdjInfo: owning, move-only cuSOLVER gesvdjInfo_t wrapper — reference §8
class GesvdjInfo {
public:
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

// CufftPlan: owning, move-only cuFFT plan wrapper — reference §9
class CufftPlan {
public:
    CufftPlan() = default;

    CufftPlan(CufftPlan&& o) noexcept : plan_(std::exchange(o.plan_, 0)) {}
    CufftPlan& operator=(CufftPlan&& o) noexcept {
        if (this != &o) {
            destroy();
            plan_ = std::exchange(o.plan_, 0);
        }
        return *this;
    }
    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;
    ~CufftPlan() { destroy(); }

    void make(int rank, int* n, int* inembed, int istride, int idist,
              int* onembed, int ostride, int odist, cufftType type, int batch) {
        destroy();
        cufftHandle p = 0;
        CUFFT_CHECK(cufftPlanMany(&p, rank, n, inembed, istride, idist,
                                  onembed, ostride, odist, type, batch));
        plan_ = p;
    }

    void set_stream(cudaStream_t stream) {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
    }

    [[nodiscard]] cufftHandle get() const noexcept { return plan_; }

private:
    void destroy() noexcept {
        if (plan_ != 0) {
            const cufftResult s = cufftDestroy(plan_);
            if (s != CUFFT_SUCCESS) {
                STEPPE_LOG_WARN("cufftDestroy at scope exit: %s",
                                CufftError::status_name(s));
            }
        }
        plan_ = 0;
    }

    cufftHandle plan_ = 0;
};

// One-shot capability tag for the cuSOLVER FP64-emulated downgrade — reference §10
#if !STEPPE_HAVE_CUSOLVER_FP64_EMULATED
namespace detail {
inline void warn_cusolver_emulated_fp64_unavailable_once() {
    static std::atomic_flag emitted;
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

// CusolverMathModeScope: scoped cuSOLVER math-mode restore (precision-promotion seam) — reference §11
class CusolverMathModeScope {
public:
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

    [[nodiscard]] bool promoted() const noexcept { return promoted_; }

private:
    cusolverMathMode_t requested_mode(bool honorable) noexcept {
        if (honorable) {
#if STEPPE_HAVE_CUSOLVER_FP64_EMULATED
            promoted_ = true;
            return CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH;
#else
            detail::warn_cusolver_emulated_fp64_unavailable_once();
            promoted_ = false;
            return CUSOLVER_DEFAULT_MATH;
#endif
        }
        promoted_ = false;
        return CUSOLVER_DEFAULT_MATH;
    }

    void restore() noexcept {
        if (h_) {
            const cusolverStatus_t s = cusolverDnSetMathMode(h_, prev_);
            if (s != CUSOLVER_STATUS_SUCCESS) {
                STEPPE_LOG_WARN("cusolverDnSetMathMode restore at scope exit: %s",
                                CusolverError::status_name(s));
            }
        }
        h_ = nullptr;
    }

    cusolverDnHandle_t h_ = nullptr;
    cusolverMathMode_t prev_ = CUSOLVER_DEFAULT_MATH;
    bool promoted_ = false;
};

// engage_solver_precision: build a CusolverMathModeScope from a typed Precision — reference §12
[[nodiscard]] inline CusolverMathModeScope engage_solver_precision(
    cusolverDnHandle_t handle, const Precision& precision,
    bool (*honorable_predicate)(const Precision&) noexcept) {
    return CusolverMathModeScope(handle, honorable_predicate(precision));
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_HANDLES_HPP
