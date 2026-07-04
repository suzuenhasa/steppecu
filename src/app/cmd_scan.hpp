// src/app/cmd_scan.hpp
//
// Implements the `steppe scan` command: the proxy/model scanner. Phase 0 enumerates
// the candidate pool (like qpadm-rotate — no guided search yet), fits the whole model
// list in one batched, f2-resident engine call, then applies the scanner objective —
// a hard gate (status / feasibility / tail p >= alpha) followed by a parsimony /
// stability / robustness rank — and emits a best-first table with the search census
// (models tested / feasible) and a selected-not-confirmed marker. App-layer and
// CUDA-free — it reaches the GPU only through the library's CUDA-free seams.
//
// Design: docs/planning/proxy-scanner-scope.md (§1a objective, §4 Phase 0).
#ifndef STEPPE_APP_CMD_SCAN_HPP
#define STEPPE_APP_CMD_SCAN_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// scan entry point (Phase 0: ranked, gated emitter over the pool rotation)
[[nodiscard]] int run_scan_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_SCAN_HPP
