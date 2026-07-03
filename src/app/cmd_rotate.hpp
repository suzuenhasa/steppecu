// src/app/cmd_rotate.hpp
//
// Implements the `steppe qpadm-rotate` command: enumerates every k-source subset
// of a candidate pool against one fixed target and right set, then fits the whole
// model list in a single batched, f2-resident engine call. App-layer and
// CUDA-free — it reaches the GPU only through the library's CUDA-free seams.
//
// Reference: docs/reference/src_app_cmd_rotate.hpp.md
#ifndef STEPPE_APP_CMD_ROTATE_HPP
#define STEPPE_APP_CMD_ROTATE_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// qpadm-rotate entry point — reference §1
[[nodiscard]] int run_qpadm_rotate_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_ROTATE_HPP
