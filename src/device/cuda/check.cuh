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
// those, so §12 parity is identical on both capability tiers (TODO M4.5).
//
// This is a CUDA header (`.cuh`): it includes <cuda_runtime.h>/<cublas_v2.h> and
// is therefore PRIVATE to steppe_device (architecture.md §4 layering, §8 link
// wiring) — `core`/`api`/the CLI never see it. Only the device layer includes it.
#ifndef STEPPE_DEVICE_CUDA_CHECK_CUH
#define STEPPE_DEVICE_CUDA_CHECK_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <exception>
#include <source_location>
#include <string>

#include "core/internal/host_device.hpp"  // STEPPE_DEBUG_ONLY (the one debug gate)
#include "core/internal/log.hpp"           // STEPPE_LOG_WARN (the one warn sink, B7/X-4)

namespace steppe::device {

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
        msg_ = std::string(loc.file_name()) + ":" +
               std::to_string(loc.line()) + " (" + loc.function_name() +
               "): '" + expr + "' -> " + cudaGetErrorName(status) + ": " +
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
        msg_ = std::string(loc.file_name()) + ":" +
               std::to_string(loc.line()) + " (" + loc.function_name() +
               "): '" + expr + "' -> " + status_name(status);
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
