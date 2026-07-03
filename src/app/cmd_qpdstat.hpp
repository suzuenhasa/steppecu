// src/app/cmd_qpdstat.hpp
//
// The `steppe qpdstat` subcommand: a D-statistic / f4 over four populations.
// A thin wrapper that adds no compute of its own — the --f2-dir path reports f4
// via the existing f4 engine, the --prefix path runs the genotype-path
// normalized D. Plain C++20, no CUDA header.
//
// Reference: docs/reference/src_app_cmd_qpdstat.hpp.md
#ifndef STEPPE_APP_CMD_QPDSTAT_HPP
#define STEPPE_APP_CMD_QPDSTAT_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_qpdstat_command — reference §6
[[nodiscard]] int run_qpdstat_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPDSTAT_HPP
