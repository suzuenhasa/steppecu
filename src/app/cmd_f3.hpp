// src/app/cmd_f3.hpp
//
// The `steppe f3` command: the standalone f3 statistic. f3 is the sibling of f4, not a fork of
// qpAdm — no target, no ALS, no rank. It computes the AT2 weighted block-jackknife f3 point
// estimate per population triple plus the jackknife-diagonal SE, emitting a tidy CSV/TSV/JSON
// table (schema pop1,pop2,pop3,est,se,z,p). Triples come from the row-aligned
// --pop1/--pop2/--pop3 columns (as admixtools::f3 with comb=FALSE, triple k = (C=pop1[k],
// A=pop2[k], B=pop3[k])) or from the --pops C,A,B convenience (any multiple of 3 names).
//
// Plain C++20, app-only, no CUDA header: it reaches the GPU only through the CUDA-free seams
// (build_resources, upload_f2_blocks_to_device, run_f3), exactly as cmd_f4 does.
#ifndef STEPPE_APP_CMD_F3_HPP
#define STEPPE_APP_CMD_F3_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the standalone f3 over the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its own stdout/stderr; the library never prints. A
/// domain outcome (e.g. a non-SPD covariance over the m-batch) rides on the result `status`
/// and still exits 0 (record-and-continue); only faults (InvalidConfig from bad
/// names/dir/mismatched triple columns, DeviceOom, file/format/CUDA-runtime errors) return
/// a nonzero code.
[[nodiscard]] int run_f3_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F3_HPP
