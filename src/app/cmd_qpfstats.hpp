// src/app/cmd_qpfstats.hpp
//
// The `steppe qpfstats` command. Unlike f4/f3/qpdstat, which read an f2 cache and report a
// statistic, qpfstats reads genotypes (--prefix) and writes a smoothed f2 dir (--out-dir) that
// qpAdm / f4 / qpGraph then consume like any extract-f2 cache. It runs the shared-factor
// smoothing regression over the full f2∪f3∪f4 popcomb set and scatters the coefficients into a
// [P × P × n_block] tensor matching AT2's qpfstats() output. --pops is sorted ascending
// internally to match AT2's dimname order.
//
// App-only, no CUDA header: reaches the GPU only through the CUDA-free seam (build_resources,
// run_qpfstats) and writes via write_f2_dir, mirroring cmd_extract_f2.cpp's compose chain.
#ifndef STEPPE_APP_CMD_QPFSTATS_HPP
#define STEPPE_APP_CMD_QPFSTATS_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run qpfstats over the frozen config and return the process exit code (the kExit* codes in
/// core/config/exit_code.hpp). Owns its stdout/stderr — the library layer never prints. A fault
/// (bad prefix/pops/out-dir, no device, file or CUDA error) returns nonzero; a successful
/// smoothed-f2 dir write exits 0.
[[nodiscard]] int run_qpfstats_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_QPFSTATS_HPP
