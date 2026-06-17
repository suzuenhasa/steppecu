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
#include <string>
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
/// source of launch geometry (`cdiv`, `grid_for`, and the per-kernel block dims)
/// lives in `core/internal/launch_config.hpp`; kernels never re-pick a block size.
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

/// Base of the per-block size-bucketing used by the M4 batched f2 path: blocks
/// are grouped by `ceil_pow{kBlockGroupPadBase}(block_size)` and each group is
/// run as one cublasDgemmStridedBatched call padded to that group's bucket width.
/// Base 2 (power-of-two buckets) bounds the per-block padding waste to < this
/// factor WITHIN a group while keeping the number of strided-batched calls
/// O(log max_block_size) (MEASURED: real-AADR P=768 → 10 groups, 1.43× pad waste
/// vs the global-s_max design's 2.76×; the grouped design is the spike-chosen M4
/// design — fastest AND VRAM-viable, ROADMAP M4 spike). The pad columns carry
/// V=0 ⇒ they contribute nothing to the masked GEMMs (architecture.md §5 S2).
inline constexpr int kBlockGroupPadBase = 2;

/// cuBLAS workspace bytes for the f2 GEMMs (architecture.md §12 — an explicit
/// workspace is REQUIRED for run-to-run reproducibility of emulated FP64). Ample
/// for the reduce-to-[P×P] GEMMs; promoted out of the spike's bare 64 MiB literal
/// (f2_emu_spike.cu) into a named constant shared by the M0 and M4 device paths.
inline constexpr std::size_t kCublasWorkspaceBytes = 64u * 1024u * 1024u;

/// Target fraction of device VRAM the resident working set may occupy
/// (architecture.md §11.1/§11.2 — the `build()`-validated budget fraction; ROADMAP
/// §4 — promoted out of the bare `0.80` literal in cuda_backend.cu). It is the one
/// home for the `budget · free` fraction in §11.2's `total_vram ≤ budget · free`
/// check AND the in-stream chunk-sizing in the M4 backend, so the up-front reject
/// and the runtime chunk budget can never drift to two different fractions.
///
/// VALUE — reconciliation to architecture.md §11.1 ("a target fraction (say 60–70%)
/// of free VRAM"). The §11.1 60–70% figure is the budget against the WHOLE device's
/// free VRAM at chunk-size derivation time, where the headroom must also absorb the
/// resident `f2_blocks`/`Vpair` tensors, the cuSOLVER/cuBLAS workspaces, AND the
/// double-buffered pinned/device tile staging that §11.1 still has to allocate. The
/// M4 chunk budget this constant gates is applied to the free VRAM that REMAINS
/// AFTER the resident tensors + the cuBLAS workspace are already subtracted (the
/// budget helper in device/vram_budget.hpp subtracts `2·P²·n_block·8` for f2+Vpair
/// and `kCublasWorkspaceBytes` before applying this fraction), so the residual it
/// scales is the headroom for ONE transient strided-batched chunk — a narrower
/// quantity than §11.1's gross-free budget. 0.80 of that already-net residual keeps
/// a ≥20% margin for cuBLAS GEMM scratch beyond the bound workspace and for
/// allocator fragmentation, while staying within the §11.1 spirit (never commit all
/// free VRAM). Tunable policy number, NOT a mathematical constant.
inline constexpr double kMaxVramUtilizationFraction = 0.80;

/// Default jackknife block size in centimorgans. ADMIXTOOLS 2's `blgsize`
/// default is 0.05 Morgans = 5 cM (architecture.md §9; ROADMAP §4). The accessor
/// surface speaks cM; the block math stores Morgans (1 cM = 0.01 Morgans). The
/// conversion lives in exactly one place, next to `block_partition_rule.hpp`.
inline constexpr double kDefaultBlockSizeCm = 5.0;

/// Centimorgans per Morgan — the single conversion constant between the
/// cM-facing config accessor and the Morgan-based block rule.
inline constexpr double kCentimorgansPerMorgan = 100.0;

/// Inclusive autosome chromosome-code range for the `autosomes_only` filter
/// (M2). ADMIXTOOLS 2's `extract_f2` default is `auto_only = TRUE` ("keep only
/// SNPs on chromosomes 1 to 22"), so AT2 parity = chromosomes 1..22 — the sex
/// chromosomes (X, Y) and MT/other non-autosomal codes are dropped. The exact
/// EIGENSTRAT codes those non-autosomal labels map to (X→23, Y→24, MT→90) are
/// single-homed as `kChromCodeX`/`kChromCodeY`/`kChromCodeMt` in
/// src/io/eigenstrat_format.hpp — `read_snp` EMITS them and this 1..22 range is
/// the complement that DROPS them, so the filter's correctness depends on the two
/// agreeing on those codes (cleanup X-8/B16). Named here (not a bare 22) so the
/// single AT2 autosome definition lives in one place and the filter predicate
/// reads it.
/// (Verified against the AT2 extract_f2 reference: auto_only default TRUE = chr
/// 1-22. The M3 real-AADR finding — chr 1-24 → 757 blocks, chr 1-23 → 756 —
/// is consistent: dropping chr 23 AND 24 here is the AT2-parity autosome set.)
inline constexpr int kAutosomeChromMin = 1;
inline constexpr int kAutosomeChromMax = 22;

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
        ///
        /// HONORABILITY COUPLING (this header cannot enforce it — the macro is
        /// device-private — but it is the authoritative cross-reference): the
        /// FIXED-slice pin is engaged in the device layer ONLY when steppe_device
        /// is built with `STEPPE_HAVE_EMU_TUNING` (default ON; SteppeOptions.cmake).
        /// A build with it OFF cannot call the FIXED mantissa-control API, so
        /// selecting `EmulatedFp64` is NOT honorable there — the device path
        /// DOWNGRADES to native `Fp64` with a logged capability tag rather than
        /// silently running the rejected DYNAMIC mantissa (device
        /// `emulation_honorable` / `engage_f2_precision`; architecture.md §9
        /// build() "fall back to native Fp64 or error"; cleanup X-6/B2).
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

    // -----------------------------------------------------------------------
    // MULTI-GPU OVERRIDE-KNOB BANNER (architecture.md §11.4; cleanup overview
    // (2).3, finding include-config 11.1/9.1).
    //
    // There are TWO distinct knob types in the capability-tier design and they
    // live in TWO distinct places — do not conflate them:
    //
    //   * OVERRIDE INTENT  → here, in `DeviceConfig`. The user-facing levers
    //     that say what the run MAY or PREFERS to do. This set is APPEND-ONLY as
    //     capability levers land: `devices` (the fixed combine order, §11.4),
    //     `enable_peer_access` (M4), `prefer_p2p_combine` (M4.5), and a future
    //     `enable_gds_ingest` (M5/M7). Every lever is parity-NEUTRAL — it moves
    //     bytes or changes observability only, never a reported number (§12), so
    //     the host-staged fixed-order combine and the device-resident P2P
    //     combine are bit-identical on both capability tiers (§11.4).
    //
    //   * DISCOVERED CAPABILITY + the WHICH-PATH tag → NEVER here. The runtime
    //     probe result (canAccessPeer, free/total VRAM, emulated-FP64-honorable
    //     state) and the recorded "which path did this run actually take, and
    //     why did it degrade" tag are RUNTIME STATE, not intent: they live in
    //     `Resources` / the result metadata, never on `DeviceConfig` and never
    //     on the pure-numeric `F2BlockTensor` (cleanup overview (2).2). A
    //     non-throwing tagged-degrade path (e.g. canAccessPeer == "no") records
    //     the fallback there; it is NOT an error (§11.4).
    // -----------------------------------------------------------------------

    /// Enable peer access opportunistically (cudaDeviceEnablePeerAccess when
    /// canAccessPeer) for single-node multi-GPU (architecture.md §11.4). This is
    /// the MAY-WE knob: whether the backend is permitted to call
    /// cudaDeviceEnablePeerAccess at all. DISTINCT from `prefer_p2p_combine`
    /// (which path to take WHEN peer access is available) — see below.
    bool enable_peer_access = true;

    /// Prefer the device-resident P2P combine over the host-staged combine WHEN
    /// peer access is available (architecture.md §11.4; cleanup overview (2).3,
    /// finding include-config 11.1). This is the WHICH-PATH knob and is DISTINCT
    /// from `enable_peer_access`:
    ///   * `enable_peer_access` = MAY we call cudaDeviceEnablePeerAccess at all;
    ///   * `prefer_p2p_combine` = once peer access IS available, prefer the
    ///     device-resident `cudaMemcpyPeer` combine (GPU 0 pulls each peer's
    ///     partial via a byte-exact DMA and sums in the fixed `g = 0..G-1`
    ///     `devices` order on-device) over gathering the G partials to the host
    ///     and summing them there.
    /// BOTH combine paths sum in the SAME fixed device order and are therefore
    /// BIT-IDENTICAL to each other and to the single-GPU reference — the
    /// transport only moves bytes; software fixes the order; NEVER NCCL
    /// AllReduce (architecture.md §11.4, §12). So this knob is parity-NEUTRAL: it
    /// is a data-movement/cleanliness lever, never a reported number.
    /// Default `true` (capable-first): prefer the device-resident combine when
    /// `enable_peer_access` is set AND the runtime `cudaDeviceCanAccessPeer`
    /// probe succeeds; otherwise the backend DEGRADES — non-throwing and tagged
    /// — to the host-staged fixed-order combine, which is the portable parity
    /// BASELINE and the only path on a budget box with peer access disabled
    /// (e.g. stock-driver GeForce). The probe result and the which-path tag are
    /// DISCOVERED state recorded in `Resources`/result metadata, NOT here (see
    /// the override-knob banner above). The sharding + combine ALGORITHM that
    /// reads this knob lands in the M4.5 multi-GPU workflow; this is the
    /// capability-tier SCAFFOLD that makes the knob expressible.
    bool prefer_p2p_combine = true;

    /// Bit-stability INTENT for the statistic path (default ON). When true the
    /// run is held to the §12 reproducibility contract and `build()` enforces the
    /// constraints that make it hold (architecture.md §9 build()-validation list,
    /// §12):
    ///   * the statistic-bearing reductions run `run_to_run`-deterministic and
    ///     cuSOLVER runs in its scoped deterministic mode;
    ///   * `stream_count` is forced to 1 on the statistic path (cuBLAS
    ///     reproducibility does not hold across concurrent streams, §12) — set
    ///     >1 explicitly here while `deterministic` is the error `build()` raises;
    ///   * `precision == Precision::Kind::EmulatedFp64` requires the explicit
    ///     `cublasSetWorkspace` workspace (`kCublasWorkspaceBytes`), since
    ///     fixed-point emulation voids the run-to-run guarantee without an
    ///     adequate workspace (§12);
    ///   * the multi-GPU partials are combined in the fixed host-side device
    ///     order pinned by `devices` (§11.4) rather than a non-deterministic
    ///     AllReduce.
    /// This is OVERRIDE INTENT, not discovered state: it is the knob the §12
    /// stream_count/workspace/combine rules the M4.5 parity-recompute path relies
    /// on are phrased against, and they are inexpressible without it. Set false
    /// only for throughput-only lanes whose results are recomputed in
    /// EmulatedFp64/Fp64 before any reported number (§12).
    bool deterministic = true;
};

// ---------------------------------------------------------------------------
// FilterConfig — on-the-fly QC thresholds (architecture.md §1, §5 S0'; ROADMAP
// M2). Promotes the spike's absent-but-implied filter knobs. Defaults are
// no-ops (keep everything) so the parity / pairwise-complete path is unaffected
// unless a filter is explicitly requested.
// ---------------------------------------------------------------------------
struct FilterConfig {
    /// Minimum minor-allele frequency to keep a SNP. 0.0 ⇒ no MAF filter.
    /// MAF is the POOLED folded minor allele frequency: min(q, 1-q) of the
    /// pooled reference-allele frequency across all KEPT samples
    /// (Σ_pop ref_count / Σ_pop allele_count), matching build_tgeno_matrix.py's
    /// ssum/scnt pooling — NOT a per-population frequency. Pinned and documented
    /// in src/io/filter/filter_decision.hpp (the single source of the rule).
    double maf_min = 0.0;

    /// Maximum per-SNP missing fraction (the `geno` filter): drop a SNP whose
    /// missing fraction exceeds this. 1.0 ⇒ keep every SNP. The fraction is over
    /// the SAMPLE (individual) axis: 1 - (Σ non-missing individuals across kept
    /// pops) / (total kept individuals) — the PLINK `--geno` convention. NOTE:
    /// ADMIXTOOLS 2's `maxmiss` is a per-SNP fraction over POPULATIONS, a
    /// different denominator; see filter_decision.hpp for the pinned convention.
    double geno_max_missing = 1.0;

    /// Maximum per-sample missing fraction (the `mind` filter): drop a sample
    /// whose missing fraction exceeds this. 1.0 ⇒ keep every sample. Decidable
    /// only from a streaming pre-pass over ALL SNPs (the conditional S-1 pass),
    /// since per-sample missingness is not a single-tile quantity.
    double mind_max_missing = 1.0;

    // ----- NEW M2 flag-gated filters (each defaults to a NO-OP so the parity /
    // pairwise-complete path is untouched unless the flag is explicitly set).
    // These are the "additions beyond bare filter" (architecture.md §1): gated
    // behind explicit flags and tagged, never on by default. -----------------

    /// Keep only autosomal SNPs (chromosomes kAutosomeChromMin..kAutosomeChromMax
    /// == 1..22). false ⇒ keep every chromosome (no-op). ADMIXTOOLS 2's
    /// extract_f2 default is `auto_only = TRUE` (chr 1-22) — set this true for
    /// AT2 autosome parity; it is OFF here so the bare-filter default keeps the
    /// sex chromosomes (architecture.md §1: autosome-only is an explicit add-on).
    bool autosomes_only = false;

    /// Drop monomorphic SNPs (pooled MAF exactly 0 ⇒ no variation across kept
    /// samples). false ⇒ keep monomorphic SNPs (no-op). An explicit add-on
    /// beyond bare filter (architecture.md §1); equivalent to maf_min at the
    /// strict-positive boundary but kept as a separate, named flag.
    bool drop_monomorphic = false;

    /// Keep only transversion SNPs (ref/alt is a transversion: a purine↔pyrimidine
    /// change, NOT a transition A↔G or C↔T). false ⇒ keep transitions too
    /// (no-op). An explicit add-on beyond bare filter (architecture.md §1, the
    /// ts/tv option). NB: strand-AMBIGUOUS (A/T, C/G) and multiallelic SNPs are
    /// DROPPED regardless of this flag (drop-not-flip; architecture.md §1).
    bool transversions_only = false;

    // ----- Include / exclude SNP membership (resolved against the .snp ids by
    // include_exclude.{hpp,cpp}). Empty ⇒ no constraint (no-op). An external
    // prune.in (LD-pruned SNP id list) is READ here, NEVER computed — steppe
    // does not compute LD itself (architecture.md §1). -----------------------

    /// Explicit SNP-id keep set (the `keepsnps`/`--extract` analogue). Non-empty
    /// ⇒ keep ONLY these ids (intersected with whatever else passes). Empty ⇒
    /// no include constraint. These are SNP IDs (.snp column 1), not indices.
    std::vector<std::string> include_snp_ids;

    /// Explicit SNP-id drop set (the `--exclude` analogue). Any id here is
    /// dropped even if it would otherwise pass. Empty ⇒ no exclude constraint.
    std::vector<std::string> exclude_snp_ids;

    /// Path to an external prune.in file (one SNP id per line) of LD-pruned SNPs
    /// to KEEP. READ, never computed (architecture.md §1: we accept only an
    /// externally supplied prune.in and never compute LD). Empty ⇒ unused. When
    /// set it composes with `include_snp_ids` as an additional keep set.
    std::string prune_in_path;
};

}  // namespace steppe

#endif  // STEPPE_CONFIG_HPP
