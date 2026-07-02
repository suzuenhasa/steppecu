// src/app/cmd_f3.hpp
//
// The `steppe f3` command implementation (standalone f3 statistic; fit-engine §6). f3 is
// the SIBLING of f4, NOT a fork of qpAdm: NO target / NO ALS / NO rank — it computes the
// AT2 weighted block-jackknife f3 POINT ESTIMATE per triple + the jackknife-DIAGONAL SE.
// The GPU path is the deliverable (cli-bindings.md §5.4 GPU-only): read the f2_blocks dir ->
// resolve names->indices via pops.txt -> build_resources(DeviceConfig) ->
// upload_f2_blocks_to_device -> run_f3(DeviceF2Blocks, triples) -> emit the table in tidy
// CSV/TSV/JSON (the golden_fit0_f3_readf2.csv schema: pop1,pop2,pop3,est,se,z,p).
//
// TRIPLES come from EITHER the row-aligned --pop1/--pop2/--pop3 columns (admixtools::f3
// comb=FALSE; triple k = (C=pop1[k], A=pop2[k], B=pop3[k])) OR the single-triple --pops
// C,A,B convenience (3 names = one triple, any multiple of 3 = several).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches the
// GPU ONLY through the CUDA-FREE seams (resources.hpp build_resources; device_f2_blocks.hpp
// upload_f2_blocks_to_device; f3.hpp run_f3) — exactly as cmd_f4.cpp does. It REUSES the
// cmd_qpwave f2-dir loader + PopResolver + build/upload chain and the result_emit format
// primitives (through emit_f3_result); the ONLY new logic is the triple resolve + the f3 emit.
#ifndef STEPPE_APP_CMD_F3_HPP
#define STEPPE_APP_CMD_F3_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the standalone f3 over the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its stdout/stderr (the library never prints,
/// architecture.md §10). A DOMAIN outcome (e.g. NonSpd covariance over the m-batch) rides
/// on the result `status` and EXITS 0 (record-and-continue); only FAULTS (InvalidConfig
/// from bad names/dir/mismatched triple columns, DeviceOom, file/format/CUDA-runtime
/// errors) return a nonzero code (cli-bindings.md §1.3, §4.4).
[[nodiscard]] int run_f3_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F3_HPP
