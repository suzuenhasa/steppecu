// src/app/cmd_rotate.hpp
//
// The `steppe qpadm-rotate` command: one target, one fixed right set, and a pool of
// candidate left sources. Enumerate every k-source subset of the pool for k in
// [min_sources, max_sources] and fit the whole list in one batched, f2-resident GPU run,
// emitting a per-model feasibility table (target, left, p, chisq, dof, f4rank, feasible,
// status, weights/se) in CSV/TSV/JSON.
//
// App-only plain C++20 with no CUDA header: reaches the GPU only through the CUDA-free
// seams, as cmd_qpadm.cpp does. Reuses cmd_qpadm's f2-dir loader, PopResolver,
// build/upload chain, and emit primitives; the only new logic is the subset enumerator
// and the per-model table emit.
#ifndef STEPPE_APP_CMD_ROTATE_HPP
#define STEPPE_APP_CMD_ROTATE_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the qpAdm rotation for the frozen config and return the process exit code. Owns
/// its stdout/stderr; the library itself never prints. Record-and-continue: a per-model
/// domain outcome (rank-deficient, non-SPD, undefined chisq) is just a table row and
/// exits 0. Only faults (invalid config from bad names/dir/empty enumeration, device OOM,
/// file/format/CUDA-runtime errors) return a nonzero code, since a rotation has no single
/// result to derive one exit code from.
[[nodiscard]] int run_qpadm_rotate_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_ROTATE_HPP
