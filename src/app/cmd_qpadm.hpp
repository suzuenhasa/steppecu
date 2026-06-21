// src/app/cmd_qpadm.hpp
//
// The `steppe qpadm` command implementation (M(cli-1); cli-bindings.md §4.1 row
// `qpadm`). The GPU path is the deliverable (cli-bindings.md §5.4 GPU-only): read the
// f2_blocks dir -> resolve names->indices via pops.txt -> build_resources(DeviceConfig)
// -> upload_f2_blocks_to_device -> run_qpadm(DeviceF2Blocks, ...) -> emit tidy CSV/JSON.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches
// the GPU ONLY through the CUDA-FREE seams (resources.hpp build_resources;
// device_f2_blocks.hpp upload_f2_blocks_to_device; qpadm.hpp run_qpadm). A no-GPU box
// surfaces a clear "no CUDA device" fault (cli-bindings.md §5.4).
#ifndef STEPPE_APP_CMD_QPADM_HPP
#define STEPPE_APP_CMD_QPADM_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the qpAdm fit for the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its stdout/stderr (the library never prints,
/// architecture.md §10). A model-level DOMAIN outcome (RankDeficient/NonSpd/
/// ChisqUndefined) is emitted as a `status` column/field and EXITS 0
/// (record-and-continue); only FAULTS (InvalidConfig from bad names/dir, DeviceOom,
/// file/format/CUDA-runtime errors) return a nonzero code (cli-bindings.md §1.3, §4.4).
[[nodiscard]] int run_qpadm_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPADM_HPP
