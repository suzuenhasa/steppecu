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

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPGRAPH_HPP
