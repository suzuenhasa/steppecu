// src/device/cuda/check.cuh
//
// The single home for CUDA / cuBLAS / cuSOLVER / cuFFT error checking plus
// post-launch kernel validation: a failed call throws a typed exception rather
// than exiting. A CUDA header (.cuh) private to the device layer — core, the
// public API, and the CLI never include it.
//
// Reference: docs/reference/src_device_cuda_check.cuh.md
#ifndef STEPPE_DEVICE_CUDA_CHECK_CUH
#define STEPPE_DEVICE_CUDA_CHECK_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>

#include <exception>
#include <source_location>
#include <string>

#include "core/internal/host_device.hpp"
#include "core/internal/log.hpp"

namespace steppe::device {

// Shared call-site formatter — reference §3
[[nodiscard]] inline std::string format_call_site(
    const std::source_location& loc, const char* expr) {
    return std::string(loc.file_name()) + ":" + std::to_string(loc.line()) +
           " (" + loc.function_name() + "): '" + expr + "' -> ";
}

// CudaError — reference §3
class CudaError : public std::exception {
public:
    CudaError(cudaError_t status, const char* expr,
              const std::source_location& loc)
        : status_(status) {
        msg_ = format_call_site(loc, expr) + cudaGetErrorName(status) + ": " +
               cudaGetErrorString(status);
    }
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] cudaError_t status() const noexcept { return status_; }

private:
    cudaError_t status_;
    std::string msg_;
};

// CublasError — reference §3
class CublasError : public std::exception {
public:
    CublasError(cublasStatus_t status, const char* expr,
                const std::source_location& loc)
        : status_(status) {
        msg_ = format_call_site(loc, expr) + status_name(status);
    }
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] cublasStatus_t status() const noexcept { return status_; }

    [[nodiscard]] static const char* status_name(cublasStatus_t s) noexcept {
        switch (s) {
            case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
            default:                             return "CUBLAS_STATUS_UNKNOWN";
        }
    }

private:
    cublasStatus_t status_;
    std::string msg_;
};

// CusolverError — reference §3
class CusolverError : public std::exception {
public:
    CusolverError(cusolverStatus_t status, const char* expr,
                  const std::source_location& loc)
        : status_(status) {
        msg_ = format_call_site(loc, expr) + status_name(status);
    }
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] cusolverStatus_t status() const noexcept { return status_; }

    [[nodiscard]] static const char* status_name(cusolverStatus_t s) noexcept {
        switch (s) {
            case CUSOLVER_STATUS_SUCCESS:           return "CUSOLVER_STATUS_SUCCESS";
            case CUSOLVER_STATUS_NOT_INITIALIZED:   return "CUSOLVER_STATUS_NOT_INITIALIZED";
            case CUSOLVER_STATUS_ALLOC_FAILED:      return "CUSOLVER_STATUS_ALLOC_FAILED";
            case CUSOLVER_STATUS_INVALID_VALUE:     return "CUSOLVER_STATUS_INVALID_VALUE";
            case CUSOLVER_STATUS_ARCH_MISMATCH:     return "CUSOLVER_STATUS_ARCH_MISMATCH";
            case CUSOLVER_STATUS_EXECUTION_FAILED:  return "CUSOLVER_STATUS_EXECUTION_FAILED";
            case CUSOLVER_STATUS_INTERNAL_ERROR:    return "CUSOLVER_STATUS_INTERNAL_ERROR";
            case CUSOLVER_STATUS_NOT_SUPPORTED:     return "CUSOLVER_STATUS_NOT_SUPPORTED";
            default:                                return "CUSOLVER_STATUS_UNKNOWN";
        }
    }

private:
    cusolverStatus_t status_;
    std::string msg_;
};

// CufftError — reference §3
class CufftError : public std::exception {
public:
    CufftError(cufftResult status, const char* expr,
               const std::source_location& loc)
        : status_(status) {
        msg_ = format_call_site(loc, expr) + status_name(status);
    }
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] cufftResult status() const noexcept { return status_; }

    [[nodiscard]] static const char* status_name(cufftResult s) noexcept {
        switch (s) {
            case CUFFT_SUCCESS:            return "CUFFT_SUCCESS";
            case CUFFT_INVALID_PLAN:       return "CUFFT_INVALID_PLAN";
            case CUFFT_ALLOC_FAILED:       return "CUFFT_ALLOC_FAILED";
            case CUFFT_INVALID_TYPE:       return "CUFFT_INVALID_TYPE";
            case CUFFT_INVALID_VALUE:      return "CUFFT_INVALID_VALUE";
            case CUFFT_INTERNAL_ERROR:     return "CUFFT_INTERNAL_ERROR";
            case CUFFT_EXEC_FAILED:        return "CUFFT_EXEC_FAILED";
            case CUFFT_SETUP_FAILED:       return "CUFFT_SETUP_FAILED";
            case CUFFT_INVALID_SIZE:       return "CUFFT_INVALID_SIZE";
            case CUFFT_UNALIGNED_DATA:     return "CUFFT_UNALIGNED_DATA";
            case CUFFT_INVALID_DEVICE:     return "CUFFT_INVALID_DEVICE";
            case CUFFT_NO_WORKSPACE:       return "CUFFT_NO_WORKSPACE";
            case CUFFT_NOT_IMPLEMENTED:    return "CUFFT_NOT_IMPLEMENTED";
            case CUFFT_NOT_SUPPORTED:      return "CUFFT_NOT_SUPPORTED";
            case CUFFT_MISSING_DEPENDENCY: return "CUFFT_MISSING_DEPENDENCY";
            case CUFFT_NVRTC_FAILURE:      return "CUFFT_NVRTC_FAILURE";
            case CUFFT_NVJITLINK_FAILURE:  return "CUFFT_NVJITLINK_FAILURE";
            case CUFFT_NVSHMEM_FAILURE:    return "CUFFT_NVSHMEM_FAILURE";
            default:                       return "CUFFT_STATUS_UNKNOWN";
        }
    }

private:
    cufftResult status_;
    std::string msg_;
};

namespace detail {

inline void cuda_check(cudaError_t status, const char* expr,
                       const std::source_location& loc =
                           std::source_location::current()) {
    if (status != cudaSuccess) throw CudaError(status, expr, loc);
}

[[nodiscard]] inline cudaError_t cuda_warn(
    cudaError_t status, [[maybe_unused]] const char* expr,
    [[maybe_unused]] const std::source_location& loc =
        std::source_location::current()) {
    if (status != cudaSuccess) {
        STEPPE_LOG_WARN("%s:%u (%s): '%s' -> %s: %s",
                        loc.file_name(), static_cast<unsigned>(loc.line()),
                        loc.function_name(), expr,
                        cudaGetErrorName(status), cudaGetErrorString(status));
    }
    return status;
}

inline void cublas_check(cublasStatus_t status, const char* expr,
                         const std::source_location& loc =
                             std::source_location::current()) {
    if (status != CUBLAS_STATUS_SUCCESS) throw CublasError(status, expr, loc);
}

inline void cusolver_check(cusolverStatus_t status, const char* expr,
                           const std::source_location& loc =
                               std::source_location::current()) {
    if (status != CUSOLVER_STATUS_SUCCESS) throw CusolverError(status, expr, loc);
}

inline void cufft_check(cufftResult status, const char* expr,
                        const std::source_location& loc =
                            std::source_location::current()) {
    if (status != CUFFT_SUCCESS) throw CufftError(status, expr, loc);
}

}  // namespace detail

}  // namespace steppe::device

// Fault-check macro (CUDA runtime) — reference §4
#define STEPPE_CUDA_CHECK(expr) \
    ::steppe::device::detail::cuda_check((expr), #expr)

// Warn macro — reference §5
#define STEPPE_CUDA_WARN(expr) \
    ::steppe::device::detail::cuda_warn((expr), #expr)

// Fault-check macros (cuBLAS / cuSOLVER / cuFFT) — reference §4
#define CUBLAS_CHECK(expr) \
    ::steppe::device::detail::cublas_check((expr), #expr)

#define CUSOLVER_CHECK(expr) \
    ::steppe::device::detail::cusolver_check((expr), #expr)

#define CUFFT_CHECK(expr) \
    ::steppe::device::detail::cufft_check((expr), #expr)

// Post-launch kernel check — reference §6
#define STEPPE_CUDA_CHECK_KERNEL()                                              \
    do {                                                                        \
        STEPPE_CUDA_CHECK(cudaGetLastError());      /* bad launch config */     \
        STEPPE_DEBUG_ONLY(                                                      \
            STEPPE_CUDA_CHECK(cudaDeviceSynchronize())); /* attr async fault */ \
    } while (0)

#endif  // STEPPE_DEVICE_CUDA_CHECK_CUH
