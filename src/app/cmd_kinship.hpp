// src/app/cmd_kinship.hpp
//
// The `steppe kinship` command: KING-robust between-family kinship over a diploid genotype
// triple, computed on the GPU. App-layer C++ only; no CUDA header (the GPU is reached only
// through the run_kinship_* seam).
#ifndef STEPPE_APP_CMD_KINSHIP_HPP
#define STEPPE_APP_CMD_KINSHIP_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

int run_kinship_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_KINSHIP_HPP
