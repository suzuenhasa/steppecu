// src/app/cmd_fstat_sweep.hpp — the `steppe f4-sweep` / `steppe f3-sweep` commands.
//
// GPU-only all-combinations f-stat sweep: enumerate every C(P,k) quartet/triple, then compute,
// filter (|z| / top-K), and compact survivors entirely on the device, emitting only survivors.
// This header stays CUDA-free — the GPU is reached solely through the run_f4_sweep / run_f3_sweep
// seam — so it can be included from plain C++ translation units.
#ifndef STEPPE_APP_CMD_FSTAT_SWEEP_HPP
#define STEPPE_APP_CMD_FSTAT_SWEEP_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

/// Run `steppe f4-sweep` (k=4) — returns a process exit code (core/config/exit_code.hpp).
[[nodiscard]] int run_f4_sweep_command(const steppe::config::RunConfig& config);

/// Run `steppe f3-sweep` (k=3).
[[nodiscard]] int run_f3_sweep_command(const steppe::config::RunConfig& config);

/// Shared sweep body (arity `k`: 4 ⇒ f4 sweep, 3 ⇒ f3 sweep). Exported so the standalone
/// `f4` / `f3` / `qpdstat` commands can route to the same GPU sweep when --all-quartets /
/// --all-triples is set. `cmd` is the program name used in stderr diagnostics so they read
/// with the invoked command rather than always naming the sweep.
[[nodiscard]] int run_fstat_sweep(const steppe::config::RunConfig& config, int k,
                                  const char* cmd);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_FSTAT_SWEEP_HPP
