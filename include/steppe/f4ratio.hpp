// include/steppe/f4ratio.hpp
//
// PUBLIC, CUDA-FREE standalone f4-RATIO entry point — the SIBLING of run_f4
// (include/steppe/f4.hpp) / run_f3 (include/steppe/f3.hpp). An f4-ratio is the AT2
// qpf4ratio admixture proportion
//   alpha = f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)
// i.e. a RATIO of two f4 statistics sharing pops p1,p2,p4 (only the 3rd slot swaps:
// p3 in the numerator, p5 in the denominator). It has NO ALS / NO rank test: the
// numerator and denominator are each a per-block f4 (REUSING assemble_f4_quartets, the
// SAME four-slab AT2 identity the fit/f4 paths use, with ZERO new assemble math), and
// the ONE new math seam is the per-block-RATIO weighted block-jackknife (the AT2
// jack_mat_stats `$est` of the ratio statistic + its xtau variance) — host-pure, in
// the .cpp (fit-engine §6 / the standalone-f4ratio design, mirroring f4/f3).
//
// AT2 qpf4ratio CONVENTION (verified against admixtools 4.3.3 R source): the input is a
// 5-column matrix c(p1,p2,p3,p4,p5). f4_num = f4(p1,p2;p3,p4), f4_den = f4(p1,p2;p5,p4).
// The per-block num uses the four-slab combine with (a=p1,b=p2,c=p3,d=p4), the den with
// (a=p1,b=p2,c=p5,d=p4). alpha REPORTED = jack_mat_stats$est (the weighted-block-jackknife
// $est of the per-block ratio R_b = f4_num_loo_b / f4_den_loo_b) — NOT tot=totnum/totden
// (tot is only the variance-centering term in xtau). z = alpha / se.
//
// ONE SHARED ASSEMBLE (essential): the num + den quartets are assembled in ONE
// assemble_f4_quartets call over the interleaved 2N-quartet flat array (length 8N), so the
// F1/OQ-12 survivor-block compaction (backend.hpp:113-126) is IDENTICAL for num and den —
// AT2 forces this (setmiss makes num/den NA together); one shared assemble guarantees one
// shared survivor-block set + one block_sizes. The returned F4Blocks has m=2N: num rows
// are k in [0,N), den rows in [N,2N) (contiguous halves). It carries x_blocks (per-block
// f4), x_loo (per-block LOO replicate, the AT2 est_to_loo) and block_sizes (survivor
// weights) — exactly the inputs the ratio jackknife needs.
//
// CUDA-FREE BY CONTRACT (architecture.md §4): standard-C++ only; populations are P-axis
// INDICES into the device-resident f2_blocks (name->index resolution is an app/binding
// concern). The DeviceF2Blocks overload is the GPU-first primary (zero D2H); the const
// F2BlockTensor& overload is the host-oracle/parity door. device::DeviceF2Blocks /
// device::Resources are forward-declared CUDA-free; the .cpp includes their real headers.
#ifndef STEPPE_F4RATIO_HPP
#define STEPPE_F4RATIO_HPP

#include <array>
#include <span>
#include <vector>

#include "steppe/config.hpp"  // steppe::Precision
#include "steppe/error.hpp"   // steppe::Status
#include "steppe/fstats.hpp"  // steppe::F2BlockTensor (host-oracle overload input)
#include "steppe/qpadm.hpp"   // steppe::QpAdmOptions (the shared per-call options) +
                              // the device::DeviceF2Blocks / device::Resources fwd-decls

namespace steppe {

/// One standalone-f4ratio result table, one parallel-array slot per input 5-tuple (in
/// INPUT order). pX[k] is the P-axis index of pop X of tuple k (echoed for the
/// emitter/binding to label the rows). NO `p` column (AT2 qpf4ratio emits only alpha/se/z).
/// A degenerate batch (empty tuples / all-missing blocks) is a per-call `status` VALUE,
/// never an exception (architecture.md §10; record-and-continue).
struct F4RatioResult {
    std::vector<int>    p1, p2, p3, p4, p5;  ///< the P-axis indices of each 5-tuple (len N).
    std::vector<double> alpha;  ///< f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) (the AT2 jack_mat_stats $est).
    std::vector<double> se;     ///< sqrt of the AT2 jackknife-of-the-ratio variance.
    std::vector<double> z;      ///< alpha / se.

    /// PER-CALL outcome (Ok for a populated result; the per-row NaN sentinel rides on a
    /// degenerate block batch). NEVER an exception for a domain outcome (architecture.md §10).
    Status status = Status::Ok;

    /// Which arithmetic produced this (the assemble honored tag; assemble_f4_quartets is
    /// the cancellation carve-out and stays native, so alpha is always native FP64).
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

/// Standalone f4-ratio over DEVICE-RESIDENT f2 (the GPU-first primary entry; zero D2H on
/// the CUDA path). `tuples` is a span of (p1,p2,p3,p4,p5) P-axis index 5-tuples; the result
/// carries one row per tuple in input order. Routes through resources.gpus[0].backend.
[[nodiscard]] F4RatioResult run_f4ratio(const device::DeviceF2Blocks& f2,
                                        std::span<const std::array<int, 5>> tuples,
                                        const QpAdmOptions& opts,
                                        device::Resources& resources);

/// HOST-ORACLE / parity overload: takes a host F2BlockTensor directly (the CpuBackend
/// reads host memory). The parity test stages the golden f2 as a host tensor and calls
/// THIS (or the device form on a real GPU).
[[nodiscard]] F4RatioResult run_f4ratio(const F2BlockTensor& f2_host,
                                        std::span<const std::array<int, 5>> tuples,
                                        const QpAdmOptions& opts,
                                        device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_F4RATIO_HPP
