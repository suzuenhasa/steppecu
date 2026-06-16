// include/steppe/config.hpp
//
// Public, host-only configuration contract for steppe.
//
// This header is the SINGLE SOURCE OF TRUTH for the typed, immutable knobs that
// every other layer builds against:
//   * the precision policy (`Precision`)         -- architecture.md §9, §12; ROADMAP §0, §4
//   * device/resource selection (`DeviceConfig`) -- architecture.md §9, §11.4
//   * on-the-fly QC filters (`FilterConfig`)     -- architecture.md §1, §5 S0'; ROADMAP M2
//   * named numeric constants that REPLACE the spike's magic numbers (ROADMAP §4)
//
// It is deliberately CUDA-FREE and depends only on the C++ standard library, so
// it compiles into `core`, `api`, the CLI, and the bindings without dragging in
// the device layer (architecture.md §4 layering rule). No CUDA headers here.
//
// PRECISION POLICY IS THE LAW (MEASURED on real AADR v66, 2× RTX 5090, CUDA 13 --
// never on synthetic data; ROADMAP §0 cautionary tale). The matmul-heavy f2
// GEMMs default to FIXED-slice Ozaki emulation at mantissa_bits = 40 (≈ native
// FP64 accuracy, 7–13× faster). 32 bits is the faster, 8.6e-9-worst-case option.
// Native FP64 is the oracle / fallback. DYNAMIC mantissa control is the rejected
// parity trap (it overshoots to ~60 bits on real data and collapses to native
// speed for no win) and is INTENTIONALLY NOT OFFERED by this type.
#ifndef STEPPE_CONFIG_HPP
#define STEPPE_CONFIG_HPP

#include <cstddef>
#include <vector>

namespace steppe {

// ---------------------------------------------------------------------------
// Named constants — promoted from the spike's magic numbers (ROADMAP §4).
//
// No raw numeric literal from the spike may survive into production except true
// mathematical constants (e.g. the `2` in a²−2ab+b²). These are the named homes
// for the rest.
// ---------------------------------------------------------------------------

/// Default fixed-slice Ozaki mantissa-bit count for `Precision::EmulatedFp64`.
/// 40 ⇒ ≈ native FP64 (worst-case f2 error 2.2e-11); 32 ⇒ 8.6e-9 (faster);
/// 48 ⇒ exceeds native (1e-12). MEASURED on real AADR (ROADMAP §0). FIXED only —
/// dynamic mantissa control is the rejected trap and is not selectable here.
inline constexpr int kDefaultMantissaBits = 40;

/// Square thread-block edge for the elementwise f2 assemble/numerator kernels:
/// a `dim3(kCdivBlock, kCdivBlock)` 2-D block over the [P × P] output.
/// Replaces the spike's `dim3 block(16,16)` (f2_emu_spike.cu:311). The single
/// source of launch geometry lives in `core/internal/f2_estimator.hpp` /
/// `launch_config.hpp`; kernels never re-pick a block size.
inline constexpr int kCdivBlock = 16;

/// Relative-error floor: pairs whose reference |f2| is below this are scored by
/// absolute error so near-zero entries do not blow up the relative metric, and
/// it is the divide-by-zero guard in relative-error comparisons. Replaces the
/// spike's bare `1e-12` (f2_prec_acc.cu:87). Used by tests / accuracy gates.
inline constexpr double kRelFloor = 1e-12;

/// Absolute floor used to guard denominators in relative-error math, e.g.
/// `max(|ref|, kAbsFloor)`. Replaces the spike's bare `1e-300`
/// (f2_emu_spike.cu:474, f2_timing.cu:90,111). Smallest normalized-ish double
/// magnitude we treat as nonzero in those guards.
inline constexpr double kAbsFloor = 1e-300;

/// Het-bias-correction denominator floor: the `1` in `max(N-1, 1)` for the
/// per-SNP het correction `q(1-q)/max(N-1,1)`. A true mathematical floor that
/// avoids divide-by-zero when a population has a single non-missing haploid.
/// Replaces the spike's bare `1.0` in `std::max(n - 1.0, 1.0)`
/// (f2_emu_spike.cu:511, :709). Lives here AND is consumed by the shared
/// `f2_estimator` primitive so the CPU oracle and GPU feeder cannot diverge.
inline constexpr double kHetCorrDenomFloor = 1.0;

/// Default jackknife block size in centimorgans. ADMIXTOOLS 2's `blgsize`
/// default is 0.05 Morgans = 5 cM (architecture.md §9; ROADMAP §4). The accessor
/// surface speaks cM; the block math stores Morgans (1 cM = 0.01 Morgans). The
/// conversion lives in exactly one place, next to `block_partition_rule.hpp`.
inline constexpr double kDefaultBlockSizeCm = 5.0;

/// Centimorgans per Morgan — the single conversion constant between the
/// cM-facing config accessor and the Morgan-based block rule.
inline constexpr double kCentimorgansPerMorgan = 100.0;

// ---------------------------------------------------------------------------
// Precision — the typed precision knob (architecture.md §9, §12; ROADMAP §4).
//
// Three named modes assigned by the CONDITIONING of the operation, not its
// shape. The default (`EmulatedFp64`, 40-bit) governs the well-conditioned
// matmul-heavy stages including the f2 GEMMs; native `Fp64` is the oracle and
// the path for the cancellation-prone numerator/divide; `Tf32` is screening
// only. There is deliberately NO `Fp32` and NO dynamic-mantissa option.
// ---------------------------------------------------------------------------
struct Precision {
    /// Which arithmetic the matmul-heavy stages run in.
    enum class Kind {
        /// Native FP64. The validation ORACLE / gold reference and the FALLBACK.
        /// Used for the catastrophic-cancellation elementwise math (the small f2
        /// numerator/divide) and the ill-conditioned GLS/SVD, regardless of the
        /// selected matmul precision. Bit-reproducible single-stream; the mode
        /// every other mode is validated against (architecture.md §12).
        Fp64,

        /// Fixed-slice Ozaki FP64 emulation — the DEFAULT for all matmul-heavy
        /// stages including the f2 GEMMs (architecture.md §5 S2, §12; ROADMAP
        /// §0). MEASURED 7–17× over native FP64 on real AADR at native-grade
        /// accuracy. The slice count is FIXED via `mantissa_bits`: 40 ≈ native,
        /// 32 = faster/8.6e-9. Accuracy-approximate, NOT bit-identical to native
        /// and not IEEE-754 on specials — which is why `Fp64` stays the oracle.
        /// Dynamic mantissa control is NOT used: on real data's wide dynamic
        /// range it overshoots to ~60 bits and collapses to parity (the trap).
        EmulatedFp64,

        /// TF32 tensor-core. Opt-in, model-space SCREENING / ranking ONLY.
        /// Results carry a precision tag and land in the loose tolerance tier;
        /// NEVER bit-compared to ADMIXTOOLS 2 goldens and never emitted as a
        /// reported est/se/z/p without recomputation in EmulatedFp64/Fp64.
        Tf32
    };

    /// Selected arithmetic for the matmul-heavy stages. Default: emulated FP64.
    Kind kind = Kind::EmulatedFp64;

    /// FIXED-slice Ozaki mantissa-bit count — meaningful only when
    /// `kind == Kind::EmulatedFp64`. 40 ≈ native FP64 (default), 32 = faster
    /// (8.6e-9 worst-case), 48 = exceeds native. ALWAYS a fixed cap; the dynamic
    /// trap is not representable by this struct (ROADMAP §0, §4). Ignored for
    /// `Fp64` and `Tf32`.
    int mantissa_bits = kDefaultMantissaBits;
};

// ---------------------------------------------------------------------------
// DeviceConfig — resources, injected not globally discovered (architecture.md
// §9, §11.4). Promotes the spike's stray device-id / stream-count literals.
// ---------------------------------------------------------------------------
struct DeviceConfig {
    /// SINGLE source of truth for which / how many GPUs to use. Empty ⇒
    /// auto-enumerate every visible CUDA device in enumeration order; a
    /// non-empty list PINS both the set AND the ordering, which is the fixed,
    /// configuration-independent `f2_blocks` host-side combine order
    /// (architecture.md §11.4, §12). Size 1 ⇒ single-GPU. The CPU backend
    /// ignores it. There is no separate count field — count == `devices.size()`.
    std::vector<int> devices;

    /// Precision policy for the matmul-heavy stages (default: emulated FP64,
    /// 40-bit). FP64 is the oracle/fallback; TF32 is screening-only (§12).
    Precision precision;

    /// Statistic streams PER GPU. Must be 1 on the bit-stable statistic path:
    /// cuBLAS reproducibility does not hold across concurrent streams
    /// (architecture.md §12). Replaces the spike's implicit single-stream use.
    std::size_t stream_count = 1;

    /// Throughput-only lanes for the model-space search (S8), where results are
    /// recomputed in EmulatedFp64/Fp64 before any reported number (§12).
    std::size_t search_streams = 4;

    /// Use a pool-backed (cudaMallocAsync) allocator so per-iteration
    /// allocations recycle from cache rather than round-tripping the OS
    /// (architecture.md §7, §11.2).
    bool use_mem_pool = true;

    /// Enable peer access opportunistically (cudaDeviceEnablePeerAccess when
    /// canAccessPeer) for single-node multi-GPU (architecture.md §11.4).
    bool enable_peer_access = true;
};

// ---------------------------------------------------------------------------
// FilterConfig — on-the-fly QC thresholds (architecture.md §1, §5 S0'; ROADMAP
// M2). Promotes the spike's absent-but-implied filter knobs. Defaults are
// no-ops (keep everything) so the parity / pairwise-complete path is unaffected
// unless a filter is explicitly requested.
// ---------------------------------------------------------------------------
struct FilterConfig {
    /// Minimum minor-allele frequency to keep a SNP. 0.0 ⇒ no MAF filter.
    double maf_min = 0.0;

    /// Maximum per-SNP missing fraction (the `geno` filter): drop a SNP whose
    /// missing fraction exceeds this. 1.0 ⇒ keep every SNP.
    double geno_max_missing = 1.0;

    /// Maximum per-sample missing fraction (the `mind` filter): drop a sample
    /// whose missing fraction exceeds this. 1.0 ⇒ keep every sample.
    double mind_max_missing = 1.0;
};

}  // namespace steppe

#endif  // STEPPE_CONFIG_HPP
