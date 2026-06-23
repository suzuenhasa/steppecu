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

/// The shared sweep body (arity `k`: 4 ⇒ f4 sweep, 3 ⇒ f3 sweep). EXPORTED so the standalone
/// `f4` / `f3` / `qpdstat` commands can route to the SAME GPU sweep when --all-quartets /
/// --all-triples is set (mining the explicit `--pops` SUBSET + --min-z/--top-k/--sure/
/// --shard-dir off the frozen config). `prog` is the program-name string for stderr
/// ("f4" / "f3" / "qpdstat") so the diagnostics read with the invoked command, not the sweep.
[[nodiscard]] int run_fstat_sweep(const steppe::config::RunConfig& config, int k,
                                  const char* prog);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_FSTAT_SWEEP_HPP
