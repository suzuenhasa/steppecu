// src/core/qpadm/qpgraph_opt_constants.hpp
//
// The single source of truth for the qpGraph projected-Newton optimizer's
// constants — the splitmix multistart RNG seeds and the optimizer
// hyperparameters — shared bit-for-bit between the CPU oracle and the CUDA
// fleet kernels. Host-pure and CUDA-free (only built-in-typed constexpr), so
// both g++ and nvcc can include it.
//
// Reference: docs/reference/src_core_qpadm_qpgraph_opt_constants.hpp.md
#ifndef STEPPE_CORE_QPADM_QPGRAPH_OPT_CONSTANTS_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_OPT_CONSTANTS_HPP

namespace steppe::core::qpadm {

namespace qpgraph_opt {

// The multistart seeding constants — reference §4
inline constexpr unsigned long long kSplitmixInstMul = 0x100000001B3ULL;
inline constexpr unsigned long long kSplitmixDimMul  = 0x9E3779B97F4A7C15ULL;
inline constexpr unsigned long long kSplitmixSeedInc = 0xD1B54A32D192ED03ULL;
inline constexpr unsigned long long kSplitmixMix1    = 0xBF58476D1CE4E5B9ULL;
inline constexpr unsigned long long kSplitmixMix2    = 0x94D049BB133111EBULL;
inline constexpr unsigned long long kMantissaMask    = 0xFFFFFFFFFFFFFULL;
inline constexpr double             kMantissaDiv     = 4503599627370496.0;

// The projected-Newton step constants — reference §5
inline constexpr double kFdStep        = 1e-4;
inline constexpr double kCurvGuard     = 1e-30;
inline constexpr double kCurvHalf      = 0.5;
inline constexpr double kCurvThresh    = 1e-8;
inline constexpr double kGradStepScale = 0.5;
inline constexpr double kTrustClamp    = 0.5;

// The backtracking line-search constants — reference §6
inline constexpr int    kMaxBacktrack    = 8;
inline constexpr double kBacktrackHalf   = 0.5;

// The per-restart convergence test — reference §7
inline constexpr double kTolDxScale = 1e-2;
inline constexpr double kTolDsScale = 1e-3;

}  // namespace qpgraph_opt

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_OPT_CONSTANTS_HPP
