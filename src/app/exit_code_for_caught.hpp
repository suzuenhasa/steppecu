// src/app/exit_code_for_caught.hpp
//
// exit_code_for_caught — the app's thin, CUDA-FREE bridge from a caught
// std::exception to the §4.4 process exit code (cli-bindings.md §4.4; architecture.md
// §10 fault taxonomy). It exists so each command's top-level
// `catch (const std::exception& e)` returns kExitDeviceOom (3) on a GENUINE device
// out-of-memory instead of the catch-all kExitRuntimeError (5) — an honest fault
// taxonomy a calling script can branch on — WITHOUT the CUDA-free app ever naming a
// CUDA type.
//
// THE LAYERING POINT (architecture.md §4): the typed device exceptions (CudaError /
// CublasError / CusolverError) and their cudaError_t / cublasStatus_t /
// cusolverStatus_t status codes are PRIVATE to steppe_device (cuda/check.cuh is a
// .cuh — the app never sees it; the arch-grep gate enforces it). So the OOM
// classification CANNOT live here: it is delegated to device::device_fault_status,
// declared CUDA-FREE in device/resources.hpp and defined in the steppe_device TU
// cuda/cuda_backend.cu. steppe_app already links steppe::device, so the symbol
// resolves while src/app stays CUDA-free.
//
// A host std::bad_alloc (host RAM, not device VRAM) — and every non-allocation
// CUDA/cuBLAS/cuSOLVER fault — yields std::nullopt from device_fault_status, so it
// keeps the catch-all kExitRuntimeError (5), which is correct. Only a real device
// allocation failure is reclassified 5 -> 3.
#ifndef STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP
#define STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP

#include <exception>
#include <optional>

#include "core/config/exit_code.hpp"  // steppe::config::exit_code_for / kExitRuntimeError
#include "device/resources.hpp"       // steppe::device::device_fault_status (CUDA-FREE seam)
#include "steppe/error.hpp"           // steppe::Status

namespace steppe::app {

/// Map a caught std::exception to the process exit code (cli-bindings.md §4.4):
/// kExitDeviceOom (3) when it is a genuine device allocation fault
/// (device::device_fault_status -> Status::DeviceOom), else the catch-all
/// kExitRuntimeError (5). Pure dispatch — `noexcept`, no allocation. Used by every
/// command's top-level `catch (const std::exception& e)` (and main.cpp's) in place of
/// a hard `return kExitRuntimeError;`.
[[nodiscard]] inline int exit_code_for_caught(const std::exception& e) noexcept {
    if (const std::optional<Status> status = device::device_fault_status(e)) {
        return config::exit_code_for(*status);
    }
    return config::kExitRuntimeError;
}

}  // namespace steppe::app

#endif  // STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP
