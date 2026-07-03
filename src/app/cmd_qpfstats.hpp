// src/app/cmd_qpfstats.hpp
//
// The `steppe qpfstats` command: reads raw genotypes and writes a smoothed f2
// directory that later qpAdm / f4 / qpGraph runs read like an extract-f2 cache.
// Plain C++20, app-only — reaches the GPU only through CUDA-free seams.
//
// Reference: docs/reference/src_app_cmd_qpfstats.hpp.md
#ifndef STEPPE_APP_CMD_QPFSTATS_HPP
#define STEPPE_APP_CMD_QPFSTATS_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_qpfstats_command — reference §5
[[nodiscard]] int run_qpfstats_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPFSTATS_HPP
