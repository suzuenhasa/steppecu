// src/app/cmd_qpfstats.hpp
//
// The `steppe qpfstats` command (the genotype-path JOINT f2 SMOOTHER; include/steppe/
// qpfstats.hpp). UNLIKE f4/f3/qpdstat(--f2-dir) (which READ an f2 cache and report a
// statistic), qpfstats READS GENOTYPES (--prefix the .geno/.snp/.ind triple) and WRITES a
// smoothed f2 DIR (--out-dir) that qpAdm / f4 / qpGraph then consume like any extract-f2
// cache. It DRIVES the qpDstat-B genotype-f4 numerator engine over the FULL f2∪f3∪f4
// popcomb set, runs the on-device shared-factor smoothing regression, and scatters the
// smoothed coefficients into a [P × P × n_block] F2BlockTensor (the AT2 qpfstats() output).
//
// INPUT: --prefix PATH (the genotype triple prefix) + --pops A,B,C,... (the pop set to
// smooth over; SORTED ASC internally = the AT2 dimnames order). --out-dir DIR (the smoothed
// f2 dir destination). --blgsize (Morgans; default 0.05). --precision (the matmul sub-step
// precision; default emu40 — the LANDED fit precision policy).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches the
// GPU ONLY through the CUDA-FREE seam (resources.hpp build_resources; qpfstats.hpp
// run_qpfstats) and writes the dir via the app's write_f2_dir — exactly as cmd_extract_f2.cpp
// composes its decode -> compute -> write_f2_dir chain.
#ifndef STEPPE_APP_CMD_QPFSTATS_HPP
#define STEPPE_APP_CMD_QPFSTATS_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run qpfstats over the frozen config and return the process exit code (the
/// steppe::config::kExit* codes from core/config/exit_code.hpp). Owns its stdout/stderr
/// (the library never prints, architecture.md §10). A FAULT (bad prefix/pops/out-dir, no
/// device, file/CUDA error) returns a nonzero code; a populated smoothed-f2 dir write exits 0.
[[nodiscard]] int run_qpfstats_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPFSTATS_HPP
