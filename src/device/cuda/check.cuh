// src/device/cuda/check.cuh
//
// THE single CUDA / cuBLAS error-checking home + post-launch validation
// (architecture.md §2 DRY, §7 idioms, §8 helpers; ROADMAP §5).
//
// Replaces the spike's THREE duplicated `CUDA_CHECK`/`CUBLAS_CHECK` macros
// (ROADMAP §1, §5). There is exactly ONE STEPPE_CUDA_CHECK and ONE CUBLAS_CHECK
// in the codebase and every CUDA / cuBLAS *fault* call routes through them.
//
// Unlike the spike macros — which `std::exit(EXIT_FAILURE)` on failure (fine for
// a throwaway harness, fatal for a library) — these THROW a typed exception
// carrying the call site via std::source_location, so tests can `catch (const
// CudaError&)` / `catch (const CublasError&)` and the public API can translate to
// `steppe_status_t` instead of aborting the process (architecture.md §7, §10).
// cuBLAS status enums do not share `cudaGetErrorString`, so they get a sibling
// macro and a separate translation (architecture.md §8 table).
//
// STEPPE_CUDA_WARN is the NON-throwing sibling for *recoverable* statuses
// (capability probes / pollers, architecture.md §11.4 capability tiers, §10): it
// logs one STEPPE_LOG_WARN line and YIELDS the status instead of throwing, so an
// EXPECTED capability-degrade (P2P canAccessPeer="no", or
// cudaErrorPeerAccessAlreadyEnabled) tags-and-degrades rather than faulting. The
// fault checks above are for unrecoverable calls; the statistic path uses only
// those, so §12 parity is identical on both capability tiers.
// TODO(M4.5): exercise this two-tier (CAP-1/CAP-2) §12 parity claim in CI — run
// the goldens on both a P2P-enabled and a P2P-disabled tier so the "identical on
// both" assertion is gated, not just asserted in prose (architecture.md §11.4).
//
// This is a CUDA header (`.cuh`): it includes <cuda_runtime.h>/<cublas_v2.h> and
// is therefore PRIVATE to steppe_device (architecture.md §4 layering, §8 link
// wiring) — `core`/`api`/the CLI never see it. Only the device layer includes it.
#ifndef STEPPE_DEVICE_CUDA_CHECK_CUH
#define STEPPE_DEVICE_CUDA_CHECK_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>

#include <exception>
#include <source_location>
#include <string>

#include "core/internal/host_device.hpp"  // STEPPE_DEBUG_ONLY (the one debug gate)
#include "core/internal/log.hpp"           // STEPPE_LOG_WARN (the one warn sink, B7/X-4)

namespace steppe::device {

/// Builds the shared "file:line (function): 'expr' -> " call-site prefix for the
/// typed CUDA / cuBLAS / cuSOLVER / cuFFT exceptions below. Each ctor appends its
/// own per-API status rendering (cudaGetErrorName/String, or a status_name()
/// switch) — only that trailing part differs, so the prefix lives once here to
/// remove the four-way copy and its drift risk. `inline` because this is a header
/// (ODR-safe). Pure host std::string formatting — no kernel / parity surface; the
/// emitted message is byte-identical to the previously-inlined per-class build.
[[nodiscard]] inline std::string format_call_site(
    const std::source_location& loc, const char* expr) {
    return std::string(loc.file_name()) + ":" + std::to_string(loc.line()) +
           " (" + loc.function_name() + "): '" + expr + "' -> ";
}

/// Typed exception thrown on a nonzero CUDA runtime status, carrying the call
/// site (file:line:function via std::source_location), the failing expression,
/// and the runtime's error name + string. Thrown — never `exit()` — so tests can
/// catch it and the public API can map it to STEPPE_ERR_CUDA_RUNTIME (and
/// distinguish cudaErrorMemoryAllocation → STEPPE_ERR_DEVICE_OOM via `status()`,
/// architecture.md §10).
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

/// Typed exception thrown on a nonzero cuBLAS status. cuBLAS has no
/// `cudaGetErrorString` equivalent, so the status enum is mapped to its symbolic
/// name here (architecture.md §8 table).
class CublasError : public std::exception {
public:
    CublasError(cublasStatus_t status, const char* expr,
                const std::source_location& loc)
        : status_(status) {
        msg_ = format_call_site(loc, expr) + status_name(status);
    }
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] cublasStatus_t status() const noexcept { return status_; }

    /// Symbolic name for a cuBLAS status (cuBLAS has no `cudaGetErrorString`).
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

/// Typed exception thrown on a nonzero cuSOLVER status. Like cuBLAS, cuSOLVER has
/// no `cudaGetErrorString` equivalent, so the status enum is mapped to its symbolic
/// name here. Thrown — never `exit()` — so the qpAdm fit can catch it and the API
/// can translate it (architecture.md §7, §10). NOTE: cuSOLVER's per-call
/// `int* devInfo` (factorization/solve outcome) is NOT routed through this — a
/// `*devInfo > 0` (singular/not-SPD) is a DOMAIN OUTCOME mapped to a Status value
/// (the FROZEN CONTRACT §1c); only the API status (bad handle / alloc / arch) throws.
class CusolverError : public std::exception {
public:
    CusolverError(cusolverStatus_t status, const char* expr,
                  const std::source_location& loc)
        : status_(status) {
        msg_ = format_call_site(loc, expr) + status_name(status);
    }
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] cusolverStatus_t status() const noexcept { return status_; }

    /// Symbolic name for a cuSOLVER status (cuSOLVER has no `cudaGetErrorString`).
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

/// Typed exception thrown on a nonzero cuFFT status. Like cuBLAS/cuSOLVER, cuFFT
/// has no `cudaGetErrorString` equivalent, so the `cufftResult` enum is mapped to
/// its symbolic name here (architecture.md §8 table). Thrown — never `exit()` — so
/// the DATES `dates_curve` cuFFT path can catch it and the API can translate it
/// (architecture.md §7, §10). The cuFFT plan create/exec/destroy calls all return
/// `cufftResult` (CUDA 13.x cuFFT: `CUFFT_SUCCESS = 0`); a nonzero status is an
/// unrecoverable API fault, routed here.
class CufftError : public std::exception {
public:
    CufftError(cufftResult status, const char* expr,
               const std::source_location& loc)
        : status_(status) {
        msg_ = format_call_site(loc, expr) + status_name(status);
    }
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] cufftResult status() const noexcept { return status_; }

    /// Symbolic name for a cuFFT status (cuFFT has no `cudaGetErrorString`). The
    /// enumerators mirror `cufftResult` in CUDA 13.x `cufft.h` exactly (verified
    /// against /usr/local/cuda/include/cufft.h on box5090): the legacy
    /// CUFFT_INCOMPLETE_PARAMETER_LIST / CUFFT_PARSE_ERROR / CUFFT_LICENSE_ERROR
    /// values were dropped, and CUFFT_MISSING_DEPENDENCY / CUFFT_NVRTC_FAILURE /
    /// CUFFT_NVJITLINK_FAILURE / CUFFT_NVSHMEM_FAILURE were added — so naming only
    /// the present enumerators keeps the switch warnings-as-errors clean.
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

// The source_location defaults to the call site (std::source_location::current()
// evaluated at the default-argument expansion point), so the check macros carry
// no __FILE__/__LINE__ plumbing — they pass only the stringized expression.

inline void cuda_check(cudaError_t status, const char* expr,
                       const std::source_location& loc =
                           std::source_location::current()) {
    if (status != cudaSuccess) throw CudaError(status, expr, loc);
}

// Non-throwing sibling of cuda_check for *recoverable* statuses (capability
// probes / pollers), NOT faults. On a non-cudaSuccess status it emits exactly
// ONE STEPPE_LOG_WARN line (same file:line:function + error name/string as
// CudaError, via the §10 warn sink) and RETURNS the status so the caller can
// branch on it — it does NOT throw. This is the device-cuda-check CAP-1/CAP-2
// home (architecture.md §11.4 capability tiers, §10 log taxonomy; TODO M4.5):
// `cudaDeviceCanAccessPeer` returning "cannot" and `cudaDeviceEnablePeerAccess`
// returning cudaErrorPeerAccessAlreadyEnabled are EXPECTED capability-degrade
// outcomes on the budget tier (GeForce P2P-disabled), NOT errors — routing them
// through the throwing cuda_check would turn a graceful, tagged degrade into a
// hard failure. The status return is build-mode-INDEPENDENT (only the WARN line
// is NDEBUG-gated, matching STEPPE_LOG_WARN's release-silent contract), so a
// release caller still observes the status it must branch on.
// `expr`/`loc` are consumed ONLY by the STEPPE_LOG_WARN line, which compiles to
// `((void)0)` (args not evaluated) under NDEBUG — so they are `[[maybe_unused]]`
// to stay clean under the project's -Wextra/-Werror (-Wunused-parameter) in a
// release build, where only `status` is referenced.
[[nodiscard]] inline cudaError_t cuda_warn(
    cudaError_t status, [[maybe_unused]] const char* expr,
    [[maybe_unused]] const std::source_location& loc =
        std::source_location::current()) {
    if (status != cudaSuccess) {
        // One line, same shape as CudaError::what(). Built only in debug (the
        // STEPPE_LOG_WARN arms its <cstdio> sink there); under NDEBUG the macro
        // is `((void)0)` and the arguments are not evaluated — `status` is still
        // returned below regardless, so the caller's branch is unaffected.
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

/// Check a CUDA runtime call; throw CudaError with file:line on failure
/// (architecture.md §7). The ONE CUDA error check in the codebase. Use for
/// *fault* calls only — calls whose failure is unrecoverable. Capability probes
/// and pollers (cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess /
/// cudaStreamQuery / cudaEventQuery) whose "failure" is an EXPECTED degrade or a
/// not-ready status must use STEPPE_CUDA_WARN instead (architecture.md §11.4).
/// Usage: `STEPPE_CUDA_CHECK(cudaMemcpyAsync(...));`
#define STEPPE_CUDA_CHECK(expr) \
    ::steppe::device::detail::cuda_check((expr), #expr)

/// Non-throwing CUDA runtime check for RECOVERABLE statuses (capability tiers,
/// architecture.md §11.4 / §10; TODO M4.5 CAP-1/CAP-2). On a non-cudaSuccess
/// status it logs ONE STEPPE_LOG_WARN line (file:line + error name/string) and
/// CONTINUES; it does NOT throw and YIELDS the cudaError_t so the caller can
/// branch and tag the degrade. Use for capability probes (e.g. P2P
/// canAccessPeer="no", cudaErrorPeerAccessAlreadyEnabled) and pollers — every
/// capability lever is parity-neutral (§12), so this NEVER appears on the
/// statistic path; STEPPE_CUDA_CHECK remains the only check there.
/// Usage: `if (::steppe::device::detail::cuda_warn(... ) != cudaSuccess) degrade();`
///   or:  `cudaError_t s = STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(peer, 0));`
#define STEPPE_CUDA_WARN(expr) \
    ::steppe::device::detail::cuda_warn((expr), #expr)

/// Check a cuBLAS call; throw CublasError with file:line on failure. The ONE
/// cuBLAS error check in the codebase (architecture.md §8 table).
/// Usage: `CUBLAS_CHECK(cublasGemmEx(...));`
#define CUBLAS_CHECK(expr) \
    ::steppe::device::detail::cublas_check((expr), #expr)

/// Check a cuSOLVER (dense) API call; throw CusolverError with file:line on a
/// nonzero API status. The ONE cuSOLVER error check in the codebase. This is for
/// the API status only — the per-call `int* devInfo` (singular/not-SPD outcome) is
/// a DOMAIN OUTCOME the qpAdm fit maps to a Status value, NOT thrown (the FROZEN
/// CONTRACT §1c). Usage: `CUSOLVER_CHECK(cusolverDnDpotrf(...));`
#define CUSOLVER_CHECK(expr) \
    ::steppe::device::detail::cusolver_check((expr), #expr)

/// Check a cuFFT call; throw CufftError with file:line on a nonzero `cufftResult`.
/// The ONE cuFFT error check in the codebase (architecture.md §8 table) — used by
/// the DATES `dates_curve` cuFFT autocorrelation engine for plan create / exec /
/// set-stream, and by the `CufftPlan` RAII owner's warn-not-throw teardown
/// (handles.hpp), so the previously bare `cufftDestroy` is no longer unchecked.
/// Usage: `CUFFT_CHECK(cufftPlanMany(...));`
#define CUFFT_CHECK(expr) \
    ::steppe::device::detail::cufft_check((expr), #expr)

/// Post-launch kernel check (architecture.md §7). `cudaGetLastError()` surfaces a
/// bad launch configuration synchronously; the debug-only `cudaDeviceSynchronize`
/// — gated through the ONE STEPPE_DEBUG_ONLY facility (core/internal/host_device.hpp),
/// not a per-site `#if defined(NDEBUG)` — forces async kernel faults to attribute
/// to THIS launch under compute-sanitizer (architecture.md §7, §13). Release relies
/// on the next runtime call surfacing the sticky error — no forced sync in the hot
/// path. Place immediately after every `kernel<<<...>>>(...)`.
#define STEPPE_CUDA_CHECK_KERNEL()                                              \
    do {                                                                        \
        STEPPE_CUDA_CHECK(cudaGetLastError());      /* bad launch config */     \
        STEPPE_DEBUG_ONLY(                                                      \
            STEPPE_CUDA_CHECK(cudaDeviceSynchronize())); /* attr async fault */ \
    } while (0)

#endif  // STEPPE_DEVICE_CUDA_CHECK_CUH
