// src/app/cmd_sfs.hpp
//
// The `steppe sfs` command: the 2D joint site-frequency spectrum over a population pair,
// accumulated on the GPU from a genotype triple. App-layer C++ only; no CUDA header (the
// GPU is reached only through the run_sfs seam).
#ifndef STEPPE_APP_CMD_SFS_HPP
#define STEPPE_APP_CMD_SFS_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

int run_sfs_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_SFS_HPP
