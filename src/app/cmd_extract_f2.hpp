// src/app/cmd_extract_f2.hpp
//
// The `steppe extract-f2` command (M(cli-4); cli-bindings.md §4.1 row `extract-f2`).
// The genotype -> f2_blocks-dir precompute: the ONLY io->compute wiring in steppe (the
// §4 layering rule — app is the only layer that may feed the io leaf into compute). It
// does the up-front sizing/validation reads, delegates the genotype -> f2_blocks chain
// (decode -> filters -> assign_blocks -> compute_f2) to the library entry
// steppe::run_extract_f2 (extract_f2_core.cpp), then writes the result via the STPF2BK1
// dir WRITER, producing the <dir> the fit commands (steppe qpadm --f2-dir) consume
// (ADR-0005 precompute-once / fit-many).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is
// reached ONLY through the CUDA-FREE seams (resources.hpp build_resources, the
// backend's decode_af via Resources, f2_blocks_multigpu.hpp compute_*). GPU-only
// (cli-bindings.md §5.4 — no --device cpu); a no-GPU box surfaces a clear "no CUDA
// device" fault. main() owns stdout/stderr (architecture.md §10).
#ifndef STEPPE_APP_CMD_EXTRACT_F2_HPP
#define STEPPE_APP_CMD_EXTRACT_F2_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the extract-f2 precompute for the frozen config and return the process exit
/// code (steppe::config::CliExitCode). Owns its stdout/stderr. Faults
/// (InvalidConfig from bad pop names / missing files, DeviceOom, file/format/CUDA-
/// runtime errors) return a nonzero code; a clean extract returns 0. --dry-run
/// reports the resolved sizes / tier / precision and returns 0 with no compute.
[[nodiscard]] int run_extract_f2_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_EXTRACT_F2_HPP
