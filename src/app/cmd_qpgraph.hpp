// src/app/cmd_qpgraph.hpp
//
// The `steppe qpgraph` command — single-graph qpGraph fit. Reads the f2_blocks dir + an
// admixture-graph edge-list file -> build_resources -> upload f2 RESIDENT -> run_qpgraph
// (the productized IDEA-1 fleet) -> emit the fitted edges + admix weights + score.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering): the GPU is reached ONLY
// through the CUDA-FREE seams (resources.hpp / device_f2_blocks.hpp / qpgraph.hpp). main()
// owns stdout/stderr. A domain outcome (NonSpd / a graph parse failure) is a clear message
// + a nonzero exit; a clean fit exits 0.
#ifndef STEPPE_APP_CMD_QPGRAPH_HPP
#define STEPPE_APP_CMD_QPGRAPH_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

/// Run the single-graph qpGraph fit for the frozen config and return the process exit
/// code. Owns its stdout/stderr.
[[nodiscard]] int run_qpgraph_command(const config::RunConfig& config);

/// Run the qpGraph TOPOLOGY SEARCH v1 (oracle C: exhaustive bounded enumeration) for the
/// frozen config (--f2-dir + --pops the bounded leaf set + --max-nadmix) and return the
/// process exit code. Enumerates every rooted topology on the pop-set (nadmix in {0..max}),
/// fits ALL in one heterogeneous-fleet launch, emits the deterministic global-best + the
/// exhaustive-coverage count + the heuristic recovery + the wall-clock. Owns stdout/stderr.
[[nodiscard]] int run_qpgraph_search_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPGRAPH_HPP
