// src/app/cmd_extract_f2.hpp
//
// Declares the `steppe extract-f2` command: the genotype -> f2_blocks-dir
// precompute. This is steppe's sole io->compute wiring seam (the app layer is
// the only layer allowed to feed io into GPU compute), yet the header stays
// plain C++20 with no CUDA include.
//
// Reference: docs/reference/src_app_cmd_extract_f2.hpp.md
#ifndef STEPPE_APP_CMD_EXTRACT_F2_HPP
#define STEPPE_APP_CMD_EXTRACT_F2_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

// run_extract_f2_command — reference §4
[[nodiscard]] int run_extract_f2_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_EXTRACT_F2_HPP
