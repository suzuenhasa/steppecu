// include/steppe/f3.hpp
//
// PUBLIC, CUDA-FREE standalone f3-statistic entry point — the SIBLING of run_f4
// (include/steppe/f4.hpp), swapping the four-slab QUARTET identity for the three-slab
// TRIPLE identity. A standalone f3 has NO ALS / NO rank test: it is just the AT2 weighted
// block-jackknife POINT ESTIMATE per triple + the jackknife-diagonal SE, computed by
// REUSING the SAME two seams the fit/f4 paths use — assemble_f3_triples (the three-slab AT2
// identity, backend.hpp) and jackknife_cov (the block jackknife covariance, backend.hpp) —
// with ONE new math seam (the 3-slab combine) and ZERO new infrastructure (fit-engine §6 /
// the standalone-f3 design, mirroring f4).
//
// KEY MAPPING (AT2 f3(C;A,B); proven against the fixture-matched regen golden): each triple
// (C=pop1, A=pop2, B=pop3) is one column of an f3 matrix with nl=1, nr=1, m=1, so per block
//   est = 0.5*( f2(C,A) + f2(C,B) - f2(A,B) ) = f3(C;A,B).
// Outgroup-f3 = f3(Outgroup;A,B) (shared drift); admixture-f3 = f3(Target;Src1,Src2)
// (negative ⇒ admixture). The SAME formula; only the apex/arg roles differ.
// BATCHED (the GPU-first production envelope, design-for-scale): ONE F4Blocks (reused as the
// generic per-block X carrier) whose m axis is the N triples, ONE jackknife_cov over the
// whole m-batch — est[k] = X.x_total[k], se[k] = sqrt(Q[k + m*k]) (the UNFUDGED diagonal).
// fudge = 0 for a bare f3 SE (qpAdm's 1e-4 ridge is a GLS-invert concern only; an f3 SE has
// no matrix inverse to regularize). z = est/se; p = 2*pnorm_upper(|z|) (the two-sided normal
// convention — REUSES f4_two_sided_p; AT2 ztop == erfc(|z|/sqrt2) == f4_two_sided_p).
//
// CUDA-FREE BY CONTRACT (architecture.md §4): standard-C++ only; populations are P-axis
// INDICES into the device-resident f2_blocks (name->index resolution is an app/binding
// concern). The DeviceF2Blocks overload is the GPU-first primary (zero D2H); the const
// F2BlockTensor& overload is the host-oracle/parity door. device::DeviceF2Blocks /
// device::Resources are forward-declared CUDA-free; the .cpp includes their real headers.
#ifndef STEPPE_F3_HPP
#define STEPPE_F3_HPP

#include <array>
#include <span>
#include <vector>

#include "steppe/config.hpp"  // steppe::Precision
#include "steppe/error.hpp"   // steppe::Status
#include "steppe/f4.hpp"      // steppe::f4_two_sided_p (REUSED — the SAME z->p convention)
#include "steppe/fstats.hpp"  // steppe::F2BlockTensor (host-oracle overload input)
#include "steppe/qpadm.hpp"   // steppe::QpAdmOptions (the shared per-call options; fudge) +
                              // the device::DeviceF2Blocks / device::Resources fwd-decls

namespace steppe {

/// One standalone-f3 result table, one parallel-array slot per input triple (in INPUT
/// order). pX[k] is the P-axis index of pop X of triple k (echoed for the emitter/binding
/// to label the rows); p1=C (apex/outgroup/target), p2=A, p3=B. A degenerate batch (empty
/// triples / non-SPD covariance) is a per-call `status` VALUE, never an exception
/// (architecture.md §10; record-and-continue).
struct F3Result {
    std::vector<int>    p1, p2, p3;  ///< the P-axis indices of each triple (len N); p1=C apex.
    std::vector<double> est;         ///< f3(p1;p2,p3) per triple (the AT2 jackknife $est).
    std::vector<double> se;          ///< sqrt of the UNFUDGED jackknife variance (the Q diagonal).
    std::vector<double> z;           ///< est / se.
    std::vector<double> p;           ///< 2*pnorm_upper(|z|) (two-sided normal; f4_two_sided_p).

    /// PER-CALL outcome (Ok / NonSpdCovariance for a degenerate batch). NEVER an
    /// exception for a domain outcome (architecture.md §10).
    Status status = Status::Ok;

    /// Which arithmetic produced this (the covariance SYRK honored tag; assemble_f3 is
    /// the cancellation carve-out and stays native, so the est is always native FP64).
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

/// Standalone f3 over DEVICE-RESIDENT f2 (the GPU-first primary entry; zero D2H on the
/// CUDA path). `triples` is a span of (p1,p2,p3) = (C,A,B) P-axis index 3-tuples; the
/// result carries one row per triple in input order. Routes through resources.gpus[0].
[[nodiscard]] F3Result run_f3(const device::DeviceF2Blocks& f2,
                              std::span<const std::array<int, 3>> triples,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

/// HOST-ORACLE / parity overload: takes a host F2BlockTensor directly (the CpuBackend
/// reads host memory). The parity test stages the golden f2 as a host tensor and calls
/// THIS (or the device form on a real GPU).
[[nodiscard]] F3Result run_f3(const F2BlockTensor& f2_host,
                              std::span<const std::array<int, 3>> triples,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_F3_HPP
