// src/app/cmd_dates.hpp
//
// The `steppe dates` command — admixture dating by ancestry-covariance decay
// (ALDER/DATES). App-layer, plain C++20 and CUDA-free: the GPU is reached only
// through the run_dates seam.
//
// Reference: docs/reference/src_app_cmd_dates.hpp.md
#ifndef STEPPE_APP_CMD_DATES_HPP
#define STEPPE_APP_CMD_DATES_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_dates_command contract — reference §6
int run_dates_command(const steppe::config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_DATES_HPP
