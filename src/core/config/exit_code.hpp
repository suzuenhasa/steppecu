// src/core/config/exit_code.hpp
//
// The single source of the CLI's status -> process exit-code mapping: a named
// enum of exit codes plus the one function that maps a run's top-level status to
// a code. Host-only and CUDA-free; the library computes the code but never calls
// std::exit — the app owns process control.
//
// Reference: docs/reference/src_core_config_exit_code.hpp.md
#ifndef STEPPE_CORE_CONFIG_EXIT_CODE_HPP
#define STEPPE_CORE_CONFIG_EXIT_CODE_HPP

#include <cstdlib>

#include "steppe/error.hpp"

namespace steppe::config {

// The exit codes (CliExitCode) — reference §3
enum CliExitCode : int {
    kExitOk            = EXIT_SUCCESS,
    kExitInvalidConfig = 2,
    kExitDeviceOom     = 3,
    kExitIoError       = 4,
    kExitRuntimeError  = 5,
};

// Mapping a status to a code (exit_code_for) — reference §4
[[nodiscard]] constexpr int exit_code_for(Status status) noexcept {
    switch (status) {
        case Status::Ok:
        case Status::RankDeficient:
        case Status::NonSpdCovariance:
        case Status::ChisqUndefined:
            return kExitOk;
        case Status::InvalidConfig:
            return kExitInvalidConfig;
        case Status::DeviceOom:
            return kExitDeviceOom;
    }
    return kExitRuntimeError;
}

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_EXIT_CODE_HPP
