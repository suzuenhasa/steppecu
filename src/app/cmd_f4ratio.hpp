// src/app/cmd_f4ratio.hpp
//
// The `steppe f4-ratio` command: the standalone f4-ratio statistic. It is a sibling of f4/f3,
// not a variant of qpAdm — no target, no ALS, no rank. It computes the AT2 qpf4ratio admixture
// proportion alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) per 5-tuple, plus the jackknife-of-the-
// ratio SE, and emits pop1..pop5,alpha,se,z as CSV/TSV/JSON.
//
// Tuples come from either the row-aligned --pop1..--pop5 columns (tuple k = pop1[k]..pop5[k])
// or the --pops p1,p2,p3,p4,p5 convenience (any multiple of 5 names).
//
// Plain C++20, no CUDA header: this reaches the GPU only through the CUDA-free seams
// (build_resources, upload_f2_blocks_to_device, run_f4ratio), reusing the f2-dir loader,
// PopResolver, build/upload chain and result-emit primitives. The only new logic here is the
// 5-tuple resolve and the f4-ratio emit.
#ifndef STEPPE_APP_CMD_F4RATIO_HPP
#define STEPPE_APP_CMD_F4RATIO_HPP

#include "core/config/run_config.hpp"  // steppe::config::RunConfig

namespace steppe::app {

/// Run the standalone f4-ratio over the frozen config and return the process exit code. Owns
/// its stdout/stderr — the library never prints. A domain outcome (e.g. a degenerate block
/// batch) rides on the result `status` and the per-row NaN sentinel and exits 0 (record-and-
/// continue); only faults (InvalidConfig from bad names/dir/mismatched tuple columns, DeviceOom,
/// file/format/CUDA-runtime errors) return a nonzero code.
[[nodiscard]] int run_f4ratio_command(const config::RunConfig& config);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_F4RATIO_HPP
