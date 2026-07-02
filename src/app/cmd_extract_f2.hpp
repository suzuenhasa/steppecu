// src/app/cmd_extract_f2.hpp
//
// The `steppe extract-f2` command: the genotype -> f2_blocks-dir precompute, and the one
// place in steppe that wires the io layer into compute. It does the up-front
// sizing/validation reads, delegates the decode -> filters -> assign_blocks -> compute_f2
// chain to steppe::run_extract_f2 in extract_f2_core.cpp, then writes the f2_blocks
// directory that the fit commands (steppe qpadm --f2-dir) reuse across many models.
//
// Plain C++20, app-only, NO CUDA header: the GPU is reached only through the CUDA-free
// seams (resources.hpp, the backend's decode_af, f2_blocks_multigpu.hpp). GPU-only — a
// box with no CUDA device surfaces a clear "no CUDA device" fault.
#ifndef STEPPE_APP_CMD_EXTRACT_F2_HPP
#define STEPPE_APP_CMD_EXTRACT_F2_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the extract-f2 precompute for the frozen config and return the process exit code.
/// Owns its stdout/stderr. Faults — bad pop names, missing files, device OOM,
/// file/format/CUDA-runtime errors — return a nonzero code; a clean extract returns 0.
/// --dry-run reports the resolved sizes / tier / precision and returns 0 without compute.
[[nodiscard]] int run_extract_f2_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_EXTRACT_F2_HPP
