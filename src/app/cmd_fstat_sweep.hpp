// src/app/cmd_fstat_sweep.hpp — the `steppe f4-sweep` / `steppe f3-sweep` commands.
//
// The all-combinations f-stat sweep: enumerate every C(P,k) quartet/triple, filter, and emit only
// the survivors. Plain C++20 with no CUDA header — the GPU is reached through a CUDA-free seam.
//
// Reference: docs/reference/src_app_cmd_fstat_sweep.hpp.md
#ifndef STEPPE_APP_CMD_FSTAT_SWEEP_HPP
#define STEPPE_APP_CMD_FSTAT_SWEEP_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// Dedicated sweep commands — reference §4
[[nodiscard]] int run_f4_sweep_command(const steppe::config::RunConfig& config);

[[nodiscard]] int run_f3_sweep_command(const steppe::config::RunConfig& config);

// Shared sweep body — reference §5
[[nodiscard]] int run_fstat_sweep(const steppe::config::RunConfig& config, int k,
                                  const char* cmd);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_FSTAT_SWEEP_HPP
