// src/app/cmd_f4.hpp
//
// The `steppe f4` command implementation (standalone f4 statistic; fit-engine §6). f4 is
// the SIBLING of qpwave, NOT a fork of qpAdm: NO target / NO ALS / NO rank — it computes
// the AT2 weighted block-jackknife f4 POINT ESTIMATE per quartet + the jackknife-DIAGONAL
// SE. The GPU path is the deliverable (cli-bindings.md §5.4 GPU-only): read the f2_blocks
// dir -> resolve names->indices via pops.txt -> build_resources(DeviceConfig) ->
// upload_f2_blocks_to_device -> run_f4(DeviceF2Blocks, quartets) -> emit the table in tidy
// CSV/TSV/JSON (the golden_fit0_f4_readf2.csv schema: pop1,pop2,pop3,pop4,est,se,z,p).
//
// QUARTETS come from EITHER the row-aligned --pop1/--pop2/--pop3/--pop4 columns (admixtools
// ::f4 comb=FALSE) OR the single-quartet --pops A,B,C,D convenience (4 names = one quartet).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches the
// GPU ONLY through the CUDA-FREE seams (resources.hpp build_resources; device_f2_blocks.hpp
// upload_f2_blocks_to_device; f4.hpp run_f4) — exactly as cmd_qpadm.cpp / cmd_qpwave.cpp do.
// It REUSES the cmd_qpadm f2-dir loader + PopResolver + build/upload chain and the
// result_emit format primitives (through emit_f4_result); the ONLY new logic is the quartet
// resolve + the f4 emit.
#ifndef STEPPE_APP_CMD_F4_HPP
#define STEPPE_APP_CMD_F4_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the standalone f4 over the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its stdout/stderr (the library never prints,
/// architecture.md §10). A DOMAIN outcome (e.g. NonSpd covariance over the m-batch) rides
/// on the result `status` and EXITS 0 (record-and-continue); only FAULTS (InvalidConfig
/// from bad names/dir/mismatched quartet columns, DeviceOom, file/format/CUDA-runtime
/// errors) return a nonzero code (cli-bindings.md §1.3, §4.4).
[[nodiscard]] int run_f4_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F4_HPP
