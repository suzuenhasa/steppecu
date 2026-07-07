// src/app/cmd_fst.hpp
//
// The `steppe fst` command: standalone per-SNP Weir & Cockerham 1984 FST over a
// population pair, computed on the GPU from a genotype triple. App-layer C++ only;
// no CUDA header (the GPU is reached only through the run_fst seam).
#ifndef STEPPE_APP_CMD_FST_HPP
#define STEPPE_APP_CMD_FST_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

int run_fst_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_FST_HPP
