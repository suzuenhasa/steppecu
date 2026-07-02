// src/app/cmd_qpwave.hpp
//
// The `steppe qpwave` command implementation (M(cli-2); cli-bindings.md §4.1 row
// `qpwave`). qpWave is the rank/cladality test underlying qpAdm: given a left set + a
// right set (NO target; left[0] is the reference row) it sweeps the minimum f4 rank that
// relates them. The GPU path is the deliverable (cli-bindings.md §5.4 GPU-only): read the
// f2_blocks dir -> resolve names->indices via pops.txt -> build_resources(DeviceConfig)
// -> upload_f2_blocks_to_device -> run_qpwave(DeviceF2Blocks, ...) -> emit the rank-sweep
// table in tidy CSV/TSV/JSON.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches
// the GPU ONLY through the CUDA-FREE seams (resources.hpp build_resources;
// device_f2_blocks.hpp upload_f2_blocks_to_device; qpadm.hpp run_qpwave) — exactly as
// cmd_qpadm.cpp does. It REUSES the cmd_qpadm f2-dir loader + PopResolver + build/upload
// chain and the result_emit format primitives (through emit_qpwave_result); the ONLY new
// logic is the no-target resolve (left[0]=reference) and the rank-sweep emit.
#ifndef STEPPE_APP_CMD_QPWAVE_HPP
#define STEPPE_APP_CMD_QPWAVE_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the qpWave rank sweep for the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its stdout/stderr (the library never prints,
/// architecture.md §10). qpWave returns ONE result with a `status`; a DOMAIN outcome
/// (RankDeficient/NonSpd/ChisqUndefined) is emitted as a `status` column/field and EXITS 0
/// (record-and-continue); only FAULTS (InvalidConfig from bad names/dir, DeviceOom,
/// file/format/CUDA-runtime errors) return a nonzero code (cli-bindings.md §1.3, §4.4).
[[nodiscard]] int run_qpwave_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPWAVE_HPP
