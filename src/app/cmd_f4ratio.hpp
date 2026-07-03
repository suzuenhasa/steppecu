// src/app/cmd_f4ratio.hpp
//
// Entry point for the `steppe f4-ratio` subcommand: declares one function,
// run_f4ratio_command, and nothing else. Plain host C++20 — no CUDA here; it
// reaches the GPU only through CUDA-free seams.
//
// Reference: docs/reference/src_app_cmd_f4ratio.hpp.md
#ifndef STEPPE_APP_CMD_F4RATIO_HPP
#define STEPPE_APP_CMD_F4RATIO_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_f4ratio_command — reference §6
[[nodiscard]] int run_f4ratio_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F4RATIO_HPP
