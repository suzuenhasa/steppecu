// src/app/cmd_qpdstat.hpp
//
// The `steppe qpdstat` command (D-statistic / f4 over an f2_blocks dir; the qpDstat A+B
// plan, Part A). THIN wrapper over the EXISTING f4 path: the --f2-dir mode reports f4 — the
// AT2 f2-path convention (admixtools::qpdstat on the f2_data path == f4; f4mode is a no-op
// without per-SNP genotypes, so qpdstat(f2dir,f4mode=TRUE) is byte-identical to f4mode=FALSE
// and to f4). NO new compute and NO new emitter: it REUSES steppe::run_f4 /
// assemble_f4_quartets verbatim (the m-axis batched f4 POINT ESTIMATE + the jackknife-
// diagonal SE) and emit_f4_result (the exact p1,p2,p3,p4,est,se,z,p columns), where z = est/se
// and p = 2*(1-Phi(|z|)) ARE the AT2 D-stat sign/Z/p convention.
//
// The normalized-D MAGNITUDE (per-SNP genotypes) is Part B (--prefix): when --prefix is given
// THIS command reads the genotype triple PREFIX.{geno,snp,ind} through run_dstat (the
// genotype-path D = mean_snp(num)/mean_snp(den) block-jackknifed; include/steppe/dstat.hpp)
// and emits the SAME p1..p4,est,se,z,p table (REUSING emit_f4_result — the D convention).
//
// QUARTETS come from EITHER the row-aligned --pop1/--pop2/--pop3/--pop4 columns OR the single-
// quartet --pops A,B,C,D convenience (4 names = one quartet) — the QUADRUPLE input.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches the
// GPU ONLY through the CUDA-FREE seams (resources.hpp; device_f2_blocks.hpp; f4.hpp run_f4) —
// exactly as cmd_f4.cpp does.
#ifndef STEPPE_APP_CMD_QPDSTAT_HPP
#define STEPPE_APP_CMD_QPDSTAT_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run `steppe qpdstat` over the frozen config and return the process exit code
/// (steppe::config::CliExitCode). --f2-dir reports f4 (the AT2 f2-path convention); --prefix
/// runs the genotype-path NORMALIZED-D (Part B, run_dstat). Owns its stdout/stderr (the
/// library never prints, architecture.md §10). A DOMAIN outcome (e.g. a degenerate quadruple)
/// rides on the result `status` / per-row NaN and EXITS 0 (record-and-continue); only FAULTS
/// (bad names/dir, missing genotype files, DeviceOom, file/format/CUDA-runtime errors) return
/// a nonzero code (cli-bindings.md §1.3, §4.4).
[[nodiscard]] int run_qpdstat_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPDSTAT_HPP
