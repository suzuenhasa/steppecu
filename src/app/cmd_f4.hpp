// src/app/cmd_f4.hpp
//
// The `steppe f4` command: the standalone f4 statistic. f4 is the sibling of qpwave, not a
// fork of qpAdm — no target, no ALS, no rank test. It computes the ADMIXTOOLS 2 weighted
// block-jackknife f4 point estimate per quartet plus the jackknife-diagonal SE, and emits a
// tidy table (pop1,pop2,pop3,pop4,est,se,z,p) as CSV/TSV/JSON. Quartets come from either the
// row-aligned --pop1/--pop2/--pop3/--pop4 columns (one quartet per row) or the single-quartet
// --pops A,B,C,D convenience.
//
// Plain C++20, app-only, no CUDA header: it reaches the GPU only through the CUDA-free seams
// (build_resources, upload_f2_blocks_to_device, run_f4). Reuses the cmd_qpadm f2-dir loader,
// PopResolver, build/upload chain, and result_emit primitives; the only new logic is the
// quartet resolve and the f4 emit.
#ifndef STEPPE_APP_CMD_F4_HPP
#define STEPPE_APP_CMD_F4_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the standalone f4 over the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its stdout/stderr; the library layer never prints.
/// A domain outcome (e.g. non-SPD covariance over the m-batch) rides on the result `status`
/// and exits 0 (record-and-continue); only faults — bad names/dir/mismatched quartet
/// columns, device OOM, file/format/CUDA-runtime errors — return a nonzero code.
[[nodiscard]] int run_f4_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F4_HPP
