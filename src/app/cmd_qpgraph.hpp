// src/app/cmd_qpgraph.hpp
//
// The `steppe qpgraph` command — single-graph qpGraph fit. Reads f2_blocks + an
// admixture-graph edge list, fits on the GPU, and emits the fitted edges, admix weights,
// and score. App-only, no CUDA header: the GPU is reached only through the CUDA-free seams
// (resources.hpp / device_f2_blocks.hpp / qpgraph.hpp). main() owns stdout/stderr; a domain
// outcome (non-SPD, graph parse failure) exits nonzero, a clean fit exits 0.
#ifndef STEPPE_APP_CMD_QPGRAPH_HPP
#define STEPPE_APP_CMD_QPGRAPH_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

/// Run the single-graph qpGraph fit for the frozen config and return the process exit
/// code. Owns its stdout/stderr.
[[nodiscard]] int run_qpgraph_command(const config::RunConfig& config);

/// Run the exhaustive qpGraph topology search for the frozen config (--f2-dir, --pops as the
/// leaf set, --max-nadmix) and return the process exit code. Enumerates every rooted topology
/// on the pop set (nadmix in {0..max}) and fits them all in one batched launch, then emits the
/// deterministic global-best, the coverage count, the heuristic recovery, and the wall-clock.
/// Owns stdout/stderr.
[[nodiscard]] int run_qpgraph_search_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPGRAPH_HPP
