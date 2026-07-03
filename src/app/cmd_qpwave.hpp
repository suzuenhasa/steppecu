// src/app/cmd_qpwave.hpp
//
// Declares the single entry point for the `steppe qpwave` command — the rank/cladality
// test underlying qpAdm. Application-layer and CUDA-free: it reaches the GPU only through
// the same CUDA-free seams the qpAdm command uses.
//
// Reference: docs/reference/src_app_cmd_qpwave.hpp.md
#ifndef STEPPE_APP_CMD_QPWAVE_HPP
#define STEPPE_APP_CMD_QPWAVE_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_qpwave_command — reference §2
[[nodiscard]] int run_qpwave_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPWAVE_HPP
