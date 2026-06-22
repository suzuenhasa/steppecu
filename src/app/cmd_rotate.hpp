// src/app/cmd_rotate.hpp
//
// The `steppe qpadm-rotate` command implementation (M(cli-3); cli-bindings.md §4.1 row
// `qpadm-rotate`). The S8 ROTATION: ONE target + ONE fixed right set + a POOL of
// candidate left sources; ENUMERATE every k-source subset of the pool for k in
// [min_sources, max_sources], build a QpAdmModel per subset, and fit the WHOLE list in
// ONE batched run_qpadm_search call (the GPU-batched, f2-resident engine). Emits a
// per-model feasibility table (model_index, target, left, p, chisq, dof, f4rank,
// feasible, status, weights/se) in CSV/TSV/JSON.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches
// the GPU ONLY through the CUDA-FREE seams (resources.hpp build_resources;
// device_f2_blocks.hpp upload_f2_blocks_to_device; qpadm.hpp run_qpadm_search) — exactly
// as cmd_qpadm.cpp does. It REUSES the cmd_qpadm f2-dir loader + PopResolver +
// build/upload chain and the result_emit format primitives verbatim; the ONLY new logic
// is the pool-subset enumerator and the per-model table emit (no fit math is duplicated).
#ifndef STEPPE_APP_CMD_ROTATE_HPP
#define STEPPE_APP_CMD_ROTATE_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the qpAdm rotation for the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its stdout/stderr (the library never prints,
/// architecture.md §10). RECORD-AND-CONTINUE: a per-model DOMAIN outcome
/// (RankDeficient/NonSpd/ChisqUndefined) is a row + EXIT 0; only FAULTS (InvalidConfig
/// from bad names/dir/empty-enumeration, DeviceOom, file/format/CUDA-runtime errors)
/// return a nonzero code (cli-bindings.md §1.3, §4.4 — the rotation never routes through
/// a single result's exit_code_for; there is no single result).
[[nodiscard]] int run_qpadm_rotate_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_ROTATE_HPP
