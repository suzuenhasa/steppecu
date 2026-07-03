// src/app/cmd_f3.hpp
//
// Declares the entry point for the `steppe f3` command — the standalone f3
// statistic (sibling of f4, not qpAdm). Plain C++20, app-layer, no CUDA header:
// it reaches the GPU only through CUDA-free seams.
//
// Reference: docs/reference/src_app_cmd_f3.hpp.md
#ifndef STEPPE_APP_CMD_F3_HPP
#define STEPPE_APP_CMD_F3_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_f3_command — reference §6
[[nodiscard]] int run_f3_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F3_HPP
