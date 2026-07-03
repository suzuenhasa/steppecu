// src/app/exit_code_for_caught.hpp
//
// Maps a caught std::exception to the process exit code, reclassifying a genuine
// device out-of-memory fault to kExitDeviceOom. Stays CUDA-free by delegating the
// fault classification to the device layer across a CUDA-free seam.
//
// Reference: docs/reference/src_app_exit_code_for_caught.hpp.md
#ifndef STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP
#define STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP

#include <exception>
#include <optional>

#include "core/config/exit_code.hpp"
#include "device/resources.hpp"
#include "steppe/error.hpp"

namespace steppe::app {

// Caught-exception → exit-code bridge — reference §3
[[nodiscard]] inline int exit_code_for_caught(const std::exception& e) noexcept {
    if (const std::optional<Status> status = device::device_fault_status(e)) {
        return config::exit_code_for(*status);
    }
    return config::kExitRuntimeError;
}

}  // namespace steppe::app

#endif  // STEPPE_APP_EXIT_CODE_FOR_CAUGHT_HPP
