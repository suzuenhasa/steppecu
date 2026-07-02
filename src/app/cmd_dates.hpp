// src/app/cmd_dates.hpp
//
// The `steppe dates` command — admixture DATING via the weighted ancestry-covariance decay
// (the DATES tool; Loh 2013 ALDER / Chintalapati-Patterson-Moorjani 2022). Reads the genotype
// triple PREFIX.{geno,snp,ind} (--prefix), the admixed --target, and the two reference sources
// (--left, exactly two), and reports the date in generations + its leave-one-chromosome
// block-jackknife SE through run_dates (the cuFFT autocorrelation LD engine; NEVER the f2
// cache, NEVER a host O(M²) SNP-pair loop). PLAIN C++20, app-only, NO CUDA header (the §4
// layering): the GPU is reached ONLY through the CUDA-free run_dates seam.
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
