// src/app/cmd_qpgraph.hpp
//
// Entry points for the `steppe qpgraph` command: the single-graph fit and the topology
// search. App-only and CUDA-free — the GPU is reached only through the CUDA-free seams,
// and main() owns stdout/stderr.
#ifndef STEPPE_APP_CMD_QPGRAPH_HPP
#define STEPPE_APP_CMD_QPGRAPH_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

[[nodiscard]] int run_qpgraph_command(const config::RunConfig& config);

[[nodiscard]] int run_qpgraph_search_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPGRAPH_HPP
