// include/steppe/f4.hpp
//
// PUBLIC, CUDA-FREE standalone f4-statistic entry point — the SIBLING of run_qpwave
// (include/steppe/qpadm.hpp), NOT a fork of qpAdm. A standalone f4 has NO ALS / NO rank
// test: it is just the AT2 weighted block-jackknife POINT ESTIMATE per quartet + the
// jackknife-diagonal SE, computed by REUSING the SAME two seams the fit engine uses —
// assemble_f4 (the four-slab AT2 identity, backend.hpp) and jackknife_cov (the block
// jackknife covariance, backend.hpp) — with ZERO new math (fit-engine §6 / the standalone-
// f4 design).
//
// KEY MAPPING (AT2 f4(p1,p2;p3,p4); proven against the regen golden, max rel delta
// 1.36e-12): each quartet (p1,p2;p3,p4) is one column of an f4 matrix with left =
// {p1 (target), p2 (source)}, right = {p3 (R0), p4 (R1)}, i.e. nl=1, nr=1, m=1, so the
// four-slab identity collapses to
//   est = 0.5*( f2(p2,p3) + f2(p1,p4) - f2(p1,p3) - f2(p2,p4) ) = f4(p1,p2;p3,p4).
// BATCHED (the GPU-first production envelope, design-for-scale): ONE F4Blocks whose m
// axis is the N quartets, ONE jackknife_cov over the whole m-batch — est[k] = X.x_total[k],
// se[k] = sqrt(Q[k + m*k]) (the UNFUDGED diagonal). fudge = 0 for a bare f4 SE (qpAdm's
// 1e-4 ridge is a GLS-invert concern only; an f4 SE has no matrix inverse to regularize).
// z = est/se; p = 2*pnorm_upper(|z|) (the two-sided normal AT2 f4 convention).
//
// CUDA-FREE BY CONTRACT (architecture.md §4): standard-C++ only; populations are P-axis
// INDICES into the device-resident f2_blocks (name->index resolution is an app/binding
// concern). The DeviceF2Blocks overload is the GPU-first primary (zero D2H); the const
// F2BlockTensor& overload is the host-oracle/parity door. device::DeviceF2Blocks /
// device::Resources are forward-declared CUDA-free; the .cpp includes their real headers.
#ifndef STEPPE_F4_HPP
#define STEPPE_F4_HPP

#include <array>
#include <span>
#include <vector>

#include "steppe/config.hpp"  // steppe::Precision
#include "steppe/error.hpp"   // steppe::Status
#include "steppe/fstats.hpp"  // steppe::F2BlockTensor (host-oracle overload input)
#include "steppe/qpadm.hpp"   // steppe::QpAdmOptions (the shared per-call options; fudge) +
                              // the device::DeviceF2Blocks / device::Resources fwd-decls

namespace steppe {

/// One standalone-f4 result table, one parallel-array slot per input quartet (in INPUT
/// order). pX[k] is the P-axis index of pop X of quartet k (echoed for the emitter/binding
/// to label the rows). A degenerate batch (empty quartets / non-SPD covariance) is a
/// per-call `status` VALUE, never an exception (architecture.md §10; record-and-continue).
struct F4Result {
    std::vector<int>    p1, p2, p3, p4;  ///< the P-axis indices of each quartet (len N).
    std::vector<double> est;             ///< f4(p1,p2;p3,p4) per quartet (the AT2 jackknife $est).
    std::vector<double> se;              ///< sqrt of the UNFUDGED jackknife variance (the Q diagonal).
    std::vector<double> z;               ///< est / se.
    std::vector<double> p;               ///< 2*pnorm_upper(|z|) (two-sided normal; AT2 f4 convention).

    /// PER-CALL outcome (Ok / NonSpdCovariance for a degenerate batch). NEVER an
    /// exception for a domain outcome (architecture.md §10).
    Status status = Status::Ok;

    /// Which arithmetic produced this (the covariance SYRK honored tag; assemble_f4 is
    /// the cancellation carve-out and stays native, so the est is always native FP64).
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

/// Two-sided normal tail probability p = 2*(1 - Phi(|z|)) (the AT2 f4 z->p convention).
/// Host-pure, native FP64; single-homed here (the ONE special function the f4 path adds
/// over the qpAdm pchisq tail). Declared so the impl + any test reference one source.
[[nodiscard]] double f4_two_sided_p(double z);

/// Standalone f4 over DEVICE-RESIDENT f2 (the GPU-first primary entry; zero D2H on the
/// CUDA path). `quartets` is a span of (p1,p2,p3,p4) P-axis index 4-tuples; the result
/// carries one row per quartet in input order. Routes through resources.gpus[0].backend.
[[nodiscard]] F4Result run_f4(const device::DeviceF2Blocks& f2,
                              std::span<const std::array<int, 4>> quartets,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

/// HOST-ORACLE / parity overload: takes a host F2BlockTensor directly (the CpuBackend
/// reads host memory). The parity test stages the golden f2 as a host tensor and calls
/// THIS (or the device form on a real GPU).
[[nodiscard]] F4Result run_f4(const F2BlockTensor& f2_host,
                              std::span<const std::array<int, 4>> quartets,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_F4_HPP
