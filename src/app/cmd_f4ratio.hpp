// src/app/cmd_f4ratio.hpp
//
// The `steppe f4-ratio` command implementation (standalone f4-ratio statistic; fit-engine
// §6). f4-ratio is the SIBLING of f4/f3, NOT a fork of qpAdm: NO target / NO ALS / NO rank —
// it computes the AT2 qpf4ratio admixture proportion alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4)
// per 5-tuple + the jackknife-of-the-RATIO SE. The GPU path is the deliverable (cli-bindings
// .md §5.4 GPU-only): read the f2_blocks dir -> resolve names->indices via pops.txt ->
// build_resources(DeviceConfig) -> upload_f2_blocks_to_device -> run_f4ratio(DeviceF2Blocks,
// tuples) -> emit the table in tidy CSV/TSV/JSON (the golden_fit0_f4ratio_readf2.csv schema:
// pop1,pop2,pop3,pop4,pop5,alpha,se,z — NO p column).
//
// TUPLES come from EITHER the row-aligned --pop1..--pop5 columns (admixtools::qpf4ratio;
// tuple k = (pop1[k]..pop5[k])) OR the single-tuple --pops p1,p2,p3,p4,p5 convenience (5
// names = one tuple, any multiple of 5 = several).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches the
// GPU ONLY through the CUDA-FREE seams (resources.hpp build_resources; device_f2_blocks.hpp
// upload_f2_blocks_to_device; f4ratio.hpp run_f4ratio) — exactly as cmd_f4.cpp/cmd_f3.cpp do.
// It REUSES the cmd_qpwave f2-dir loader + PopResolver + build/upload chain and the
// result_emit format primitives (through emit_f4ratio_result); the ONLY new logic is the
// 5-tuple resolve + the f4-ratio emit.
#ifndef STEPPE_APP_CMD_F4RATIO_HPP
#define STEPPE_APP_CMD_F4RATIO_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the standalone f4-ratio over the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its stdout/stderr (the library never prints,
/// architecture.md §10). A DOMAIN outcome (e.g. a degenerate block batch) rides on the
/// result `status` / the per-row NaN sentinel and EXITS 0 (record-and-continue); only FAULTS
/// (InvalidConfig from bad names/dir/mismatched tuple columns, DeviceOom, file/format/CUDA-
/// runtime errors) return a nonzero code (cli-bindings.md §1.3, §4.4).
[[nodiscard]] int run_f4ratio_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F4RATIO_HPP
