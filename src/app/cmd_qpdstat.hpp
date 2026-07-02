// src/app/cmd_qpdstat.hpp
//
// The `steppe qpdstat` command: D-statistic / f4 over a quartet (p1,p2,p3,p4).
// A thin wrapper with no new compute and no new emitter.
//
// --f2-dir reports f4, reusing run_f4 / assemble_f4_quartets and
// emit_f4_result. This is the AT2 convention: on f2 data qpdstat == f4, since
// f4mode is a no-op without per-SNP genotypes. --prefix instead reads the
// PREFIX.{geno,snp,ind} triple and runs the genotype-path normalized
// D = mean_snp(num)/mean_snp(den), block-jackknifed (run_dstat), emitting the
// same p1..p4,est,se,z,p table. In both modes z = est/se and
// p = 2*(1-Phi(|z|)) — the AT2 D-stat sign/Z/p convention.
//
// Quartets come from the row-aligned --pop1..--pop4 columns or the single
// --pops A,B,C,D convenience. Plain C++20, app-only, no CUDA header: it reaches
// the GPU only through the CUDA-free seams (resources.hpp, device_f2_blocks.hpp,
// f4.hpp), as cmd_f4.cpp does.
#ifndef STEPPE_APP_CMD_QPDSTAT_HPP
#define STEPPE_APP_CMD_QPDSTAT_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run `steppe qpdstat` over the frozen config and return the process exit code
/// (steppe::config::CliExitCode). Owns its own stdout/stderr; the library never
/// prints. A domain outcome (e.g. a degenerate quartet) rides on the result
/// status / per-row NaN and exits 0 (record-and-continue); only faults (bad
/// names or dir, missing genotype files, DeviceOom, file/format/CUDA-runtime
/// errors) return a nonzero code.
[[nodiscard]] int run_qpdstat_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPDSTAT_HPP
