// src/app/cmd_qpwave.hpp
//
// The `steppe qpwave` command. qpWave is the rank/cladality test underlying qpAdm: given a
// left set and a right set (no target; left[0] is the reference row) it sweeps the minimum
// f4 rank relating them, emitting the rank-sweep table as CSV/TSV/JSON.
//
// Plain C++20, app-only, with no CUDA header: it reaches the GPU only through the
// CUDA-free seams (build_resources, upload_f2_blocks_to_device, run_qpwave), exactly as
// cmd_qpadm.cpp does — and reuses that command's f2-dir loader, PopResolver, and emit
// primitives. The only qpWave-specific logic is the no-target resolve and the sweep emit.
#ifndef STEPPE_APP_CMD_QPWAVE_HPP
#define STEPPE_APP_CMD_QPWAVE_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the qpWave rank sweep for the frozen config and return the process exit code.
/// Owns its stdout/stderr; the library layer never prints. qpWave returns one result with
/// a `status`: a domain outcome (RankDeficient/NonSpd/ChisqUndefined) is recorded in the
/// `status` field and still exits 0 (record-and-continue). Only faults — bad names/dir,
/// device OOM, file/format/CUDA-runtime errors — return a nonzero code.
[[nodiscard]] int run_qpwave_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPWAVE_HPP
