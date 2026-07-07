// src/app/cmd_readv2.hpp
//
// Reference: docs/reference/src_app_cmd_readv2.hpp.md
//
// The `steppe readv2` command entry point: pseudo-haploid windowed-mismatch kinship
// over a genotype triple (--prefix). Genotype-path command shaped like `steppe dates`;
// app-layer C++ only, no CUDA header.
#ifndef STEPPE_APP_CMD_READV2_HPP
#define STEPPE_APP_CMD_READV2_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

int run_readv2_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_READV2_HPP
