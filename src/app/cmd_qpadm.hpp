// src/app/cmd_qpadm.hpp
//
// The `steppe qpadm` command implementation: reads an f2-blocks directory, runs the
// qpAdm fit on the GPU, and emits a tidy CSV/JSON table. Plain C++20 with no CUDA
// header — it reaches the GPU only through CUDA-free seams.
//
// Reference: docs/reference/src_app_cmd_qpadm.hpp.md
#ifndef STEPPE_APP_CMD_QPADM_HPP
#define STEPPE_APP_CMD_QPADM_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// The single entry point: run_qpadm_command — reference §2
[[nodiscard]] int run_qpadm_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPADM_HPP
