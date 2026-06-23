// src/app/cmd_fstat_sweep.hpp — the `steppe f4-sweep` / `steppe f3-sweep` commands.
//
// The GPU-ONLY all-combinations f-stat sweep: enumerate EVERY C(P,k) quartet/triple ON THE
// DEVICE, compute + filter (|z| / top-K) + compact survivors ON THE DEVICE, emit ONLY survivors.
// Plain C++20, app-only, NO CUDA header (the §4 layering gate): the GPU is reached solely through
// the CUDA-free run_f4_sweep / run_f3_sweep seam.
#ifndef STEPPE_APP_CMD_FSTAT_SWEEP_HPP
#define STEPPE_APP_CMD_FSTAT_SWEEP_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

/// Run `steppe f4-sweep` (k=4) — returns a process exit code (core/config/exit_code.hpp).
[[nodiscard]] int run_f4_sweep_command(const steppe::config::RunConfig& config);

/// Run `steppe f3-sweep` (k=3).
[[nodiscard]] int run_f3_sweep_command(const steppe::config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_FSTAT_SWEEP_HPP
