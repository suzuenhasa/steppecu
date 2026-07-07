// src/app/cmd_paint.hpp
//
// The `steppe paint` command — Li-Stephens haplotype-copying (ChromoPainter-style
// coancestry). App-layer, plain C++20 and CUDA-free.
//
// Phase 0 (this build): the command parses + resolves its flags and runs the
// host-pure up-front validator (phased/haploid input, a monotonic cM map, the
// self-copy / leave-one-out policy, and the O(N·K·M) cost guard), then reports the
// validated run plan. The GPU forward-backward core and the coancestry face land in
// Phase 1/2 — there is no kernel launch here.
//
// Reference: docs/reference/src_app_cmd_paint.hpp.md
// Scope background: docs/planning/li-stephens-engine-scope.md §1, §3
#ifndef STEPPE_APP_CMD_PAINT_HPP
#define STEPPE_APP_CMD_PAINT_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_paint_command — the `steppe paint` entry point. Returns a CliExitCode.
int run_paint_command(const steppe::config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_PAINT_HPP
