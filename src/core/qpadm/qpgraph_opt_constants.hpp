// src/core/qpadm/qpgraph_opt_constants.hpp
//
// THE qpGraph projected-Newton optimizer constant set — the single source of truth
// for the deterministic-multistart splitmix RNG constants AND the projected-Newton
// hyperparameters shared, BIT-FOR-BIT, between the two backends:
//
//   * the CpuBackend oracle   src/device/cpu/cpu_backend.cpp (qpgraph_fit_fleet)
//   * the CUDA fleet kernels  src/device/cuda/qpgraph_fit_kernels.cu
//                             (d_init_theta / d_fit_one_restart)
//
// WHY this header exists. The CudaBackend qpGraph fleet is validated against the
// CpuBackend oracle under the §12/§13 parity diff (the real-AADR goldens: the
// single-fit score 80.0674 and the 5-pop search). The optimizer is a deterministic
// projected-Newton multistart, so the two backends must produce the SAME restart
// trajectories — which requires the splitmix multipliers, mantissa masks, and every
// step/curvature/backtracking/convergence constant to be IDENTICAL on both sides. The
// constants used to be hand-duplicated verbatim at the two sites; a one-sided edit
// silently broke parity with no compile error. Hoisting them here makes a drift a
// single-line change in ONE place (or a compile error), not a silent parity break.
//
// HOST-PURE, CUDA-FREE — exactly like core/domain/block_partition_rule.hpp (the one
// shared DOMAIN rule both `core` and the device kernels read, architecture.md §4).
// This header declares ONLY constexpr variables of built-in integral / floating-point
// type. Per the CUDA C++ Programming Guide §5.3 (C++ Language Support), a
// const-qualified variable initialized with a constant expression, of a built-in
// integral or floating-point type, "may be directly used in device code" with NO
// memory-space annotation — so these constants are usable verbatim from __device__
// code in nvcc and from host code in g++, with no CUDA types and no reliance on the
// experimental --expt-relaxed-constexpr function flag.
//   ref: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html
//
// NOTE: the RNG body and the projected-Newton loop themselves are deliberately NOT
// factored into a shared function here — a shared function would need a __host__
// __device__ annotation (a CUDA keyword, which would break the CUDA-free rule) or the
// experimental --expt-relaxed-constexpr flag. Instead, each backend keeps its own loop
// surface (host std::vector vs. device fixed-stack arrays) but references EVERY magic
// number from this header, so the only thing that must match — the constants — is
// single-sourced. This mirrors the block_partition_rule.hpp precedent (one constant
// set, two consumers).
#ifndef STEPPE_CORE_QPADM_QPGRAPH_OPT_CONSTANTS_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_OPT_CONSTANTS_HPP

namespace steppe::core::qpadm {

/// The deterministic-multistart splitmix64 RNG constants. d_init_theta / init_theta
/// seed z from (inst * kSplitmixInstMul) + (dim * kSplitmixDimMul) + kSplitmixSeedInc,
/// then run two splitmix64 mix rounds (kSplitmixMix1, kSplitmixMix2) with the canonical
/// 30/27/31 right-shift schedule, and normalize the low 52 mantissa bits
/// ((z & kMantissaMask) / kMantissaDiv, kMantissaDiv == 2^52) into a [0,1) double.
/// These are the standard
/// splitmix64 / Fibonacci-hashing magic constants; they are fixed seeds, not tunables —
/// changing them re-rolls every restart's initial theta on BOTH backends at once.
namespace qpgraph_opt {

// --- splitmix multistart seeding (the per-(inst,dim) deterministic theta init) ---
inline constexpr unsigned long long kSplitmixInstMul = 0x100000001B3ULL;       // FNV prime, inst weight
inline constexpr unsigned long long kSplitmixDimMul  = 0x9E3779B97F4A7C15ULL;   // golden-ratio (2^64/phi), dim weight
inline constexpr unsigned long long kSplitmixSeedInc = 0xD1B54A32D192ED03ULL;   // splitmix64 increment
inline constexpr unsigned long long kSplitmixMix1    = 0xBF58476D1CE4E5B9ULL;   // splitmix64 mix multiplier #1
inline constexpr unsigned long long kSplitmixMix2    = 0x94D049BB133111EBULL;   // splitmix64 mix multiplier #2
inline constexpr unsigned long long kMantissaMask    = 0xFFFFFFFFFFFFFULL;      // low 52 bits == 2^52 - 1
inline constexpr double             kMantissaDiv     = 4503599627370496.0;      // 0x10000000000000 == 2^52, normalizer into [0,1)

// --- the projected-Newton step (per-dim forward/central diff + diagonal curvature) ---
inline constexpr double kFdStep        = 1e-4;     // h: the finite-difference probe step
inline constexpr double kCurvGuard     = 1e-30;    // additive guard on the curvature denominator (no /0)
inline constexpr double kCurvHalf      = 0.5;      // the 0.5 in the central-second-difference denominator
inline constexpr double kCurvThresh    = 1e-8;     // curv > this ⇒ Newton step (g/curv); else gradient fallback
inline constexpr double kGradStepScale = 0.5;      // the gradient-fallback step factor (g * this)
inline constexpr double kTrustClamp    = 0.5;      // the projected trust-region step clamp (|step| ≤ this)

// --- the backtracking line search (halve toward the current point until non-increase) ---
inline constexpr int    kMaxBacktrack    = 8;      // backtracking-iteration cap (bt < this)
inline constexpr double kBacktrackHalf   = 0.5;    // wn ← this*(wn + w): halve toward w each backtrack

// --- the per-restart convergence test (both must hold to stop) ---
inline constexpr double kTolDxScale = 1e-2;        // max |Δθ| < tol * this
inline constexpr double kTolDsScale = 1e-3;        // max |Δscore| < tol * this

}  // namespace qpgraph_opt

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_OPT_CONSTANTS_HPP
