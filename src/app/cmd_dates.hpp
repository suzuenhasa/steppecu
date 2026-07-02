// src/app/cmd_dates.hpp
//
// The `steppe dates` command — admixture dating via the weighted ancestry-covariance decay
// (the DATES method; Loh 2013 ALDER / Chintalapati-Patterson-Moorjani 2022). Reports the date
// in generations plus a leave-one-chromosome block-jackknife SE.
//
// This is a plain C++20 app-layer header with no CUDA include: it reaches the GPU only through
// the CUDA-free run_dates seam, which drives the cuFFT autocorrelation LD engine (not the f2
// cache, not a host O(M^2) SNP-pair loop).
#ifndef STEPPE_APP_CMD_DATES_HPP
#define STEPPE_APP_CMD_DATES_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

/// Run the `steppe dates` command from the frozen RunConfig. Returns a process exit code
/// (0 on success; nonzero on a config/io/device fault). A degenerate run (no decay) is a
/// table-with-NaN + exit 0 (record-and-continue), like the other genotype-path commands.
int run_dates_command(const steppe::config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_DATES_HPP
