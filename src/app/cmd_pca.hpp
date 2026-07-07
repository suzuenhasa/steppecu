// src/app/cmd_pca.hpp
//
// The `steppe pca` command: standalone genotype PCA (Patterson-2006 standardization ->
// sample covariance -> eigendecomposition -> top-K principal components) computed on the
// GPU from a genotype triple. App-layer C++ only; no CUDA header (the GPU is reached only
// through the run_pca seam). Emits a per-sample PC-coordinate TSV/CSV/JSON (or the scree
// table with --eigenvalues) plus, with --emit-html, a self-contained interactive scatter.
#ifndef STEPPE_APP_CMD_PCA_HPP
#define STEPPE_APP_CMD_PCA_HPP

#include "core/config/run_config.hpp"

namespace steppe::app {

int run_pca_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_PCA_HPP
