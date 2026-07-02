// src/app/cmd_qpadm.hpp
//
// The `steppe qpadm` command: read the f2_blocks dir, resolve population names to
// indices, upload the blocks to the GPU, run the fit, emit tidy CSV/JSON. Plain C++20
// and app-only, with no CUDA header — it reaches the GPU only through the CUDA-free
// seams in resources.hpp, device_f2_blocks.hpp, and qpadm.hpp; a box with no GPU
// surfaces a clear "no CUDA device" fault.
#ifndef STEPPE_APP_CMD_QPADM_HPP
#define STEPPE_APP_CMD_QPADM_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the qpAdm fit for the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its own stdout/stderr; the library layer never
/// prints. A model-level domain outcome (RankDeficient/NonSpd/ChisqUndefined) is
/// reported in a `status` column and still exits 0 (record-and-continue); only faults
/// (bad names/dir, device OOM, file/format/CUDA-runtime errors) return a nonzero code.
[[nodiscard]] int run_qpadm_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPADM_HPP
