// src/app/cmd_f4.hpp
//
// The `steppe f4` command: the standalone f4 statistic (point estimate + block-
// jackknife standard error per quartet). Plain C++20, app-only — no CUDA header;
// it reaches the GPU through CUDA-free seams (build_resources / upload / run_f4).
//
// Reference: docs/reference/src_app_cmd_f4.hpp.md
#ifndef STEPPE_APP_CMD_F4_HPP
#define STEPPE_APP_CMD_F4_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_f4_command contract — reference §5
[[nodiscard]] int run_f4_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F4_HPP
