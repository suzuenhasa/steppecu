// src/app/exit_code_for_caught.hpp
//
// Thin, CUDA-free bridge from a caught std::exception to the process exit code.
// Lets each command's top-level `catch (const std::exception&)` return the device-OOM
// code on a genuine device out-of-memory, rather than the catch-all runtime-error
// code — an honest fault taxonomy a calling script can branch on.
//
// The OOM classification can't live here: the typed device exceptions and their CUDA
// status codes are private to the device layer, and src/app stays CUDA-free. So it is
// delegated to device::device_fault_status, declared CUDA-free in device/resources.hpp
// and defined in the device backend TU. A host std::bad_alloc, or any non-allocation
// CUDA/cuBLAS/cuSOLVER fault, yields nullopt and keeps the catch-all runtime-error
// code; only a real device allocation failure is reclassified to device-OOM.
#ifndef STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP
#define STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP

#include <exception>
#include <optional>

#include "core/config/exit_code.hpp"  // steppe::config::exit_code_for / kExitRuntimeError
#include "device/resources.hpp"       // steppe::device::device_fault_status (CUDA-FREE seam)
#include "steppe/error.hpp"           // steppe::Status

namespace steppe::app {

/// Map a caught std::exception to a process exit code: kExitDeviceOom on a genuine
/// device allocation fault (as classified by device::device_fault_status), else the
/// catch-all kExitRuntimeError. Pure dispatch — noexcept, no allocation. Used by every
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
