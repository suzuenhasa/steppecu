// include/steppe/config.hpp
//
// Public, host-only configuration contract for steppe.
//
// This header is the SINGLE SOURCE OF TRUTH for the typed, immutable knobs that
// every other layer builds against:
//   * the precision policy (`Precision`)         -- architecture.md §9, §12
//   * device/resource selection (`DeviceConfig`) -- architecture.md §9, §11.4
//   * on-the-fly QC filters (`FilterConfig`)     -- architecture.md §1, §5 S0'
//   * named numeric constants that replace bare magic numbers
//
// It is deliberately CUDA-FREE and depends only on the C++ standard library, so
// it compiles into `core`, `api`, the CLI, and the bindings without dragging in
// the device layer (architecture.md §4 layering rule). No CUDA headers here.
//
// PRECISION POLICY IS THE LAW (MEASURED on real AADR v66, 2× RTX 5090, CUDA 13 --
// never on synthetic data). The matmul-heavy f2
// GEMMs default to FIXED-slice Ozaki emulation at mantissa_bits = 40 (≈ native
// FP64 accuracy, 7–17× faster). 32 bits is the faster, 8.6e-9-worst-case option.
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
// Named constants — promoted from bare magic numbers.
//
// No bare numeric literal may survive into production except true
// mathematical constants (e.g. the `2` in a²−2ab+b²). These are the named homes
// for the rest.
// ---------------------------------------------------------------------------

/// Default fixed-slice Ozaki mantissa-bit count for `Precision::EmulatedFp64`.
/// 40 ⇒ ≈ native FP64 (worst-case f2 error 2.2e-11); 32 ⇒ 8.6e-9 (faster);
/// 48 ⇒ exceeds native (1e-12). MEASURED on real AADR. FIXED only —
/// dynamic mantissa control is the rejected trap and is not selectable here.
inline constexpr int kDefaultMantissaBits = 40;

/// Maxcomb CAP on the f-stat SWEEP enumeration count: a sweep that would enumerate more
/// than this many combinations C(P,k) REFUSES up front (before any compute or allocation)
/// unless the caller passes --sure. The cap guards COMPUTE TIME, not just output memory:
/// every enumerated item is computed on the GPU to TEST the |z|/top-K filter, so a
/// billions-item sweep is hours of GPU work even when few survive — the filter bounds the
/// OUTPUT, never the WORK. 1e8 is the working ceiling (a few minutes of GPU sweep at the
/// measured rate); the user lifts it explicitly with --sure. A pure choose() compare, no
/// allocation. The AT2 qpAdm `sure` analogue for the all-combinations f4/f3 sweep.
inline constexpr unsigned long long kFstatMaxComb = 100000000ULL;  // 1e8

/// Default device-bounded top-K for a bare `--all-quartets`/`--all-triples` sweep (no explicit
/// --top-k, no --min-z override). The sweep computes EVERY C(P,k) item on the GPU but keeps only
/// the K most-significant (largest |z|) in a FIXED device reservoir (rising-tau top-K), so a
/// full C(500,4)=2.57B sweep cannot OOM host RAM: only K rows (~40 MB at K=1e6) ever cross the
/// CUDA-free seam, INDEPENDENT of how many billions are computed. --top-k overrides K; --min-z
/// raises the device tau floor (a pre-filter that coexists with the K cap).
inline constexpr std::size_t kFstatDefaultSweepTopK = 1000000;  // 1e6

/// Square thread-block edge for the elementwise f2 assemble/numerator kernels:
/// a `dim3(kCdivBlock, kCdivBlock)` 2-D block over the [P × P] output.
/// Replaces a bare `dim3 block(16,16)`. The single
/// source of launch geometry (`cdiv`, `grid_for`, and the per-kernel block dims)
/// lives in `core/internal/launch_config.hpp`; kernels never re-pick a block size.
inline constexpr int kCdivBlock = 16;

/// Relative-error floor: pairs whose reference |f2| is below this are scored by
/// absolute error so near-zero entries do not blow up the relative metric, and
/// it is the divide-by-zero guard in relative-error comparisons. Replaces a
/// bare `1e-12` literal. Used by tests / accuracy gates.
inline constexpr double kRelFloor = 1e-12;

/// Absolute floor used to guard denominators in relative-error math, e.g.
/// `max(|ref|, kAbsFloor)`. Replaces a bare `1e-300` literal.
/// Smallest normalized-ish double
/// magnitude we treat as nonzero in those guards.
inline constexpr double kAbsFloor = 1e-300;

/// Het-bias-correction denominator floor: the `1` in `max(N-1, 1)` for the
/// per-SNP het correction `q(1-q)/max(N-1,1)`. A true mathematical floor that
/// avoids divide-by-zero when a population has a single non-missing haploid.
/// Replaces a bare `1.0` in `std::max(n - 1.0, 1.0)`.
/// Lives here AND is consumed by the shared
/// `f2_estimator` primitive so the CPU oracle and GPU feeder cannot diverge.
inline constexpr double kHetCorrDenomFloor = 1.0;

/// Base of the per-block size-bucketing used by the batched f2 path: blocks
/// are grouped by `ceil_pow{kBlockGroupPadBase}(block_size)` and each group is
/// run as one cublasDgemmStridedBatched call padded to that group's bucket width.
/// Base 2 (power-of-two buckets) bounds the per-block padding waste to < this
/// factor WITHIN a group while keeping the number of strided-batched calls
/// O(log max_block_size) (MEASURED: real-AADR P=768 → 10 groups, 1.43× pad waste
/// vs the global-s_max design's 2.76×; the grouped design is the chosen
/// design — fastest AND VRAM-viable). The pad columns carry
/// V=0 ⇒ they contribute nothing to the masked GEMMs (architecture.md §5 S2).
inline constexpr int kBlockGroupPadBase = 2;

/// Per-dimension cap for routing a small dense SVD to cuSOLVER's one-sided Jacobi
/// (gesvdj) instead of the QR-based gesvd: gesvdj is selected for a matrix when BOTH
/// dims (nl, nr) are <= this cap, else gesvd. This is the cuSOLVER gesvdjBatched
/// per-dimension limit (m, n <= 32 for the batched Jacobi), reused here as the
/// single dense-Jacobi/gesvd crossover so the qpAdm LARGE-path SVD dispatch
/// (cuda_backend.cu large_svd_V) and the svd_path observability report cannot drift
/// to two different thresholds (architecture.md §8 single-source).
/// It is the cuSOLVER routine-selection threshold, NOT a warp size. FROZEN by §12
/// parity (the NRBIG golden asserts svd_path==2 ⇒ gesvd at nr=39 > 32; the 9-pop
/// asserts ==1 ⇒ gesvdj at nl=2,nr=5 <= 32): name only, do NOT change the value.
inline constexpr int kGesvdjMaxDim = 32;

/// cuBLAS workspace bytes for the f2 GEMMs (architecture.md §12 — an explicit
/// workspace is REQUIRED for run-to-run reproducibility of emulated FP64). Ample
/// for the reduce-to-[P×P] GEMMs; promoted out of a bare 64 MiB literal
/// into a named constant shared by the device paths.
inline constexpr std::size_t kCublasWorkspaceBytes = 64u * 1024u * 1024u;

/// Phase-2 FIT-path chunk-budget knobs (cuda_backend.cu fit_one_bucket): the
/// per-bucket batched fit sizes its model-chunk against free VRAM. When the
/// runtime free-VRAM probe returns 0 (unknown) the fit assumes this much free
/// VRAM as a conservative fallback, and it subtracts this fixed headroom before
/// dividing by the per-model footprint so the chunk never commits all of free
/// VRAM. Named here (not bare `4<<30` / `512<<20` literals mid-function) so the
/// fit path's VRAM levers live beside the f2 path's centralized VRAM policy
/// (kCublasWorkspaceBytes / the kMaxVramUtilizationFraction block) rather than
/// drifting separately. Tunable policy numbers, parity-neutral — they change the
/// model-chunk size, never a reported number (§12).
inline constexpr std::size_t kFitBudgetFreeVramFallbackBytes = static_cast<std::size_t>(4) << 30;
inline constexpr std::size_t kFitBudgetHeadroomBytes = static_cast<std::size_t>(512) << 20;

/// Per-population buffer counts of the single-GPU f2 FEEDER phase — the named home
/// for the feeder's VRAM footprint coefficients so the tier-select policy math
/// (tier_select.hpp resident_working_set_bytes / streamed_working_set_bytes) can
/// derive the footprint from a single source instead of re-spelling bare literals
/// that silently drift from the real feeder malloc in cuda_backend.cu.
///   kFeederRawBufsPerPop  = 3·P·M : the raw decoded inputs dQ_raw/dV_raw/dN_raw.
///   kFeederOutBufsPerPop  = 4·P·M : the persisted feeder outputs (dQt + dVt +
///                                   dSt, where dSt is 2·P·M = 1+1+2 = 4 per pop).
/// Their sum (7) is the Resident-tier feeder envelope coefficient (7·P·M doubles)
/// and the per-tile feeder coefficient (3+4) of the streamed path. Plain counts —
/// CUDA-free, parity-neutral (they change WHERE a slab lands, never its bits, §12).
inline constexpr unsigned kFeederRawBufsPerPop = 3u;
inline constexpr unsigned kFeederOutBufsPerPop = 4u;

/// Resident-tensor count: the f2 and vpair tensors held co-resident for the whole
/// bucket loop (each [P²·n_block] FP64, the 2× term). Single home so the resident
/// footprint coefficient in vram_budget.hpp (resident_tensor_bytes) is not a bare 2.
/// Structural count — CUDA-free, parity-neutral (changes WHERE bytes land, §12).
inline constexpr std::size_t kResidentTensorCount = 2;

/// Strided-batched chunk per-block stack counts (vram_budget.hpp
/// per_block_chunk_bytes / tier_select.hpp per_block_chunk_elems):
///   kChunkInputStacks  = 4·P·s_pad : gathered inputs Qg + Vg (P·s_pad each) + Sg
///                                    (2·P·s_pad) = 4 stacks.
///   kChunkOutputStacks = 4·P²     : GEMM outputs Gg + Vpairg (P² each) + Rg (2·P²)
///                                    = 4 stacks.
/// Single home so the per-block footprint coefficients are not maintained twice (the
/// bytes and the elems helpers). Structural counts — CUDA-free, parity-neutral (§12).
inline constexpr std::size_t kChunkInputStacks = 4;
inline constexpr std::size_t kChunkOutputStacks = 4;

/// No-device sentinel for a CUDA-ordinal field/member (an empty / moved-from /
/// not-yet-resolved device id). Shared home so the -1 "no device" convention is one
/// symbol rather than a bare literal at each device-id default. CUDA-free.
inline constexpr int kInvalidDeviceId = -1;

/// Streamed-path device ring depth: the number of per-chunk [P²·max_nb] f2/vpair
/// device buffers the streamed (HostRam/Disk) backend cycles through so a chunk's D2H
/// can drain while the next chunk computes (cuda_backend.cu stream_f2_blocks_impl).
/// Two (not three) keeps the device ring's VRAM small — the device buffer only needs
/// to survive its own D2H, the SINK's pinned ring absorbs the slow write. Single-homed
/// here so the real ring alloc (cuda_backend.cu) and the tier-select working-set budget
/// (tier_select.hpp streamed_working_set_bytes) cannot drift apart. Plain count
/// — CUDA-free, parity-neutral (it changes WHERE bytes land, §12).
inline constexpr int kStreamDeviceChunks = 2;

/// FORWARD-RESERVED (parked multi-GPU / S8 model-space search; NOT wired). Named
/// home for the forward-reserved `DeviceConfig::search_streams` default below — no
/// production consumer reads it, so it gates no compute and is parity-neutral by
/// construction: any S8 result is recomputed in EmulatedFp64/Fp64 before it is
/// reported (§12).
inline constexpr std::size_t kDefaultSearchStreams = 4;

/// Target fraction of device VRAM the resident working set may occupy
/// (architecture.md §11.1/§11.2 — the `build()`-validated budget fraction;
/// promoted out of the bare `0.80` literal in cuda_backend.cu). It is the one
/// home for the `budget · free` fraction in §11.2's `total_vram ≤ budget · free`
/// check AND the in-stream chunk-sizing in the batched backend, so the up-front reject
/// and the runtime chunk budget can never drift to two different fractions.
///
/// VALUE — reconciliation to architecture.md §11.1 ("a target fraction (say 60–70%)
/// of free VRAM"). The §11.1 60–70% figure is the budget against the WHOLE device's
/// free VRAM at chunk-size derivation time, where the headroom must also absorb the
/// resident `f2_blocks`/`Vpair` tensors, the cuSOLVER/cuBLAS workspaces, AND the
/// double-buffered pinned/device tile staging that §11.1 still has to allocate. The
/// The chunk budget this constant gates is applied to the free VRAM that REMAINS
/// AFTER the resident tensors + the cuBLAS workspace are already subtracted (the
/// budget helper in device/vram_budget.hpp subtracts `2·P²·n_block·8` for f2+Vpair
/// and `kCublasWorkspaceBytes` before applying this fraction), so the residual it
/// scales is the headroom for ONE transient strided-batched chunk — a narrower
/// quantity than §11.1's gross-free budget. 0.80 of that already-net residual keeps
/// a ≥20% margin for cuBLAS GEMM scratch beyond the bound workspace and for
/// allocator fragmentation, while staying within the §11.1 spirit (never commit all
/// free VRAM). Tunable policy number, NOT a mathematical constant.
inline constexpr double kMaxVramUtilizationFraction = 0.80;

/// Tier-select: the fraction of free VRAM the RESIDENT-tier result + its
/// per-call working set may occupy before tier-select declines Resident and falls
/// to HostRam. Strictly below kMaxVramUtilizationFraction (0.80) so the resident
/// f2/Vpair result PLUS the run_f2_blocks_resident working set (feeder outputs +
/// one chunk's slabs, already budgeted to 0.80 of the post-result residual) cannot
/// co-exceed free VRAM. Tunable policy number, NOT a mathematical constant.
inline constexpr double kResidentTierVramFraction = 0.70;

/// Tier-select: the fraction of free HOST RAM the HOST-tier result may occupy
/// before tier-select declines HostRam and falls to Disk. Below 1.0 so the host
/// F2BlockTensor (2·P²·n_block·8 bytes) plus the small pinned staging + OS headroom
/// stay within free RAM on a normal machine. Tunable policy number.
inline constexpr double kHostTierRamFraction = 0.60;

static_assert(kResidentTierVramFraction > 0.0 && kResidentTierVramFraction <= kMaxVramUtilizationFraction,
              "kResidentTierVramFraction must be in (0, kMaxVramUtilizationFraction].");
static_assert(kHostTierRamFraction > 0.0 && kHostTierRamFraction <= 1.0,
              "kHostTierRamFraction must lie in (0, 1].");

/// Streamed-path: fraction of the VRAM envelope reserved for the tile feeder;
/// the slabs+ring take the rest. Tunable policy number, parity-neutral §12.
inline constexpr double kStreamTileBudgetFraction = 0.25;

/// Default jackknife block size in centimorgans. ADMIXTOOLS 2's `blgsize`
/// default is 0.05 Morgans = 5 cM (architecture.md §9). The accessor
/// surface speaks cM; the block math stores Morgans (1 cM = 0.01 Morgans). The
/// conversion lives in exactly one place, next to `block_partition_rule.hpp`.
inline constexpr double kDefaultBlockSizeCm = 5.0;

/// Centimorgans per Morgan — the single conversion constant between the
/// cM-facing config accessor and the Morgan-based block rule.
inline constexpr double kCentimorgansPerMorgan = 100.0;

/// AT2 base-pair block-fallback window (bp). When a dataset ships NO genetic
/// linkage map — the .snp/.bim genetic-position column is ALL zero (common in
/// VCF/PLINK-derived modern data) — ADMIXTOOLS 2's `get_block_lengths` prints
/// "No genetic linkage map found! Defining blocks by base pair distance of
/// 2e+06" and partitions by a HARDCODED 2 Mb PHYSICAL-position window instead
/// of the (zero) Morgans column (admixtools 2.0.10 R/resampling.R; the value is
/// independent of `blgsize`'s Morgans default). `block_partition_rule`'s
/// `assign_blocks` reproduces this exactly: on an all-zero genetic map it walks
/// the physical bp position with THIS window. NOT the 0.05-Morgans (5 cM)
/// default — the bp fallback is a distinct, physical-distance rule.
inline constexpr double kBpFallbackWindow = 2.0e6;

/// Inclusive autosome chromosome-code range for the `autosomes_only` filter
/// ADMIXTOOLS 2's `extract_f2` default is `auto_only = TRUE` ("keep only
/// SNPs on chromosomes 1 to 22"), so AT2 parity = chromosomes 1..22 — the sex
/// chromosomes (X, Y) and MT/other non-autosomal codes are dropped. The exact
/// EIGENSTRAT codes those non-autosomal labels map to (X→23, Y→24, MT→90) are
/// single-homed as `kChromCodeX`/`kChromCodeY`/`kChromCodeMt` in
/// src/io/eigenstrat_format.hpp — `read_snp` EMITS them and this 1..22 range is
/// the complement that DROPS them, so the filter's correctness depends on the two
/// agreeing on those codes. Named here (not a bare 22) so the
/// single AT2 autosome definition lives in one place and the filter predicate
/// reads it.
/// (Verified against the AT2 extract_f2 reference: auto_only default TRUE = chr
/// 1-22. The real-AADR finding — chr 1-24 → 757 blocks, chr 1-23 → 756 —
/// is consistent: dropping chr 23 AND 24 here is the AT2-parity autosome set.)
inline constexpr int kAutosomeChromMin = 1;
inline constexpr int kAutosomeChromMax = 22;

/// The --mind filter "inactive" threshold: the max possible per-sample missing
/// fraction (1.0 = "never drop"). When FilterConfig::mind_max_missing equals this,
/// the conditional S-1 pre-pass keeps every sample (the drop decision is a no-op).
/// Single home for the field default AND the mind_prepass.cpp activity test so the
/// two cannot drift. Parity-neutral filter policy.
inline constexpr double kMindFilterInactiveThreshold = 1.0;

/// Disk-tier frozen default on-disk f2_blocks cache path (cwd-relative). The
/// LAST-RESORT value used only when both the `Config::disk_cache_path` knob and
/// the `STEPPE_F2_CACHE_PATH` env var are empty. Single-homed here (the doc on
/// `Config::disk_cache_path` already calls it the "frozen default") so the value
/// lives in one place and the consumer references it rather than re-spelling the
/// literal. Parity-neutral: it only chooses WHERE
/// the bytes land, never a reported number (§12). Value unchanged.
inline constexpr char kDefaultDiskCachePath[] = "./steppe_f2_blocks.cache";

// ---------------------------------------------------------------------------
// Precision — the typed precision knob (architecture.md §9, §12).
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
        /// stages including the f2 GEMMs (architecture.md §5 S2, §12).
        /// MEASURED 7–17× over native FP64 on real AADR at native-grade
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
        /// build() "fall back to native Fp64 or error").
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
    /// trap is not representable by this struct. Ignored for
    /// `Fp64` and `Tf32`.
    int mantissa_bits = kDefaultMantissaBits;

    // -----------------------------------------------------------------------
    // Named factories — sugar over the brace-init incantation, so a caller
    // writes `Precision::emulated_fp64()` instead of `{Kind::EmulatedFp64, 40}`.
    // Each returns byte-identical `{kind, mantissa_bits}` to the aggregate init
    // it replaces (parity-neutral §12). Static member functions do NOT break
    // aggregate-ness in C++20, so `Precision p{};` and `Precision{Kind::Tf32}`
    // keep compiling. `constexpr` (⇒ implicitly inline; required because
    // `Precision` lives in the header-only `steppe_api` INTERFACE target) also
    // lets the static_asserts below pin the factory outputs at compile time.
    // -----------------------------------------------------------------------

    /// Native FP64 — the validation oracle / fallback (§12). `mantissa_bits`
    /// carries the default and is ignored for `Fp64`.
    [[nodiscard]] static constexpr Precision fp64() {
        return Precision{Kind::Fp64, kDefaultMantissaBits};
    }

    /// Fixed-slice Ozaki emulated FP64 — the matmul-heavy default. `bits`
    /// defaults to `kDefaultMantissaBits` so the no-arg call equals the
    /// aggregate default (40) exactly.
    [[nodiscard]] static constexpr Precision emulated_fp64(int bits = kDefaultMantissaBits) {
        return Precision{Kind::EmulatedFp64, bits};
    }

    /// TF32 tensor-core — opt-in screening only (§12). `mantissa_bits` carries
    /// the default and is ignored for `Tf32`.
    [[nodiscard]] static constexpr Precision tf32() {
        return Precision{Kind::Tf32, kDefaultMantissaBits};
    }
};

// Compile-time pin: the named factories emit byte-identical {kind, mantissa_bits}
// to the aggregate defaults — `emulated_fp64()` must equal the 40-bit f2-GEMM
// default exactly, never a silent mode/mantissa change (architecture.md §12).
static_assert(Precision::emulated_fp64().kind == Precision::Kind::EmulatedFp64 &&
                  Precision::emulated_fp64().mantissa_bits == kDefaultMantissaBits,
              "Precision::emulated_fp64() must equal {EmulatedFp64, kDefaultMantissaBits}.");
static_assert(Precision::fp64().kind == Precision::Kind::Fp64 &&
                  Precision::fp64().mantissa_bits == kDefaultMantissaBits,
              "Precision::fp64() must equal {Fp64, kDefaultMantissaBits}.");
static_assert(Precision::tf32().kind == Precision::Kind::Tf32 &&
                  Precision::tf32().mantissa_bits == kDefaultMantissaBits,
              "Precision::tf32() must equal {Tf32, kDefaultMantissaBits}.");
// The struct stays a default-constructible aggregate (Precision{} == emulated_fp64()).
static_assert(Precision{}.kind == Precision::Kind::EmulatedFp64 &&
                  Precision{}.mantissa_bits == kDefaultMantissaBits,
              "Precision must stay a default-constructible aggregate equal to emulated_fp64().");

// ---------------------------------------------------------------------------
// DeviceConfig — resources, injected not globally discovered (architecture.md
// §9, §11.4). Promotes stray device-id / stream-count literals.
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

    /// FORWARD-RESERVED (NOT wired): the device path opens a SINGLE statistic stream
    /// unconditionally — this field gates no compute. Pinned to 1 because cuBLAS
    /// reproducibility does not hold across concurrent streams (architecture.md §12);
    /// it is a §12 trap that MUST stay 1, never a tunable throughput knob.
    std::size_t stream_count = 1;

    /// FORWARD-RESERVED (parked multi-GPU / S8 model-space search; NOT wired). No
    /// production consumer reads it; it gates no compute. Parity-neutral by
    /// construction — any S8 result is recomputed in EmulatedFp64/Fp64 before it is
    /// reported (§12).
    std::size_t search_streams = kDefaultSearchStreams;

    /// FORWARD-RESERVED (NOT wired): the device path uses the pool-backed
    /// (cudaMallocAsync) allocator UNCONDITIONALLY — this field never gates that
    /// choice and has no production consumer. Parity-neutral (architecture.md §7,
    /// §11.2).
    bool use_mem_pool = true;

    // -----------------------------------------------------------------------
    // MULTI-GPU OVERRIDE-KNOB BANNER (architecture.md §11.4).
    //
    // There are TWO distinct knob types in the capability-tier design and they
    // live in TWO distinct places — do not conflate them:
    //
    //   * OVERRIDE INTENT  → here, in `DeviceConfig`. The user-facing levers
    //     that say what the run MAY or PREFERS to do. This set is APPEND-ONLY as
    //     capability levers land: `devices` (the fixed combine order, §11.4),
    //     `enable_peer_access`, `prefer_p2p_combine`, and a future
    //     `enable_gds_ingest`. Every lever is parity-NEUTRAL — it moves
    //     bytes or changes observability only, never a reported number (§12), so
    //     the host-staged fixed-order combine and the device-resident P2P
    //     combine are bit-identical on both capability tiers (§11.4).
    //
    //   * DISCOVERED CAPABILITY + the WHICH-PATH tag → never here. The runtime
    //     probe result (canAccessPeer, free/total VRAM, emulated-FP64-honorable
    //     state) and the recorded "which path did this run actually take, and
    //     why did it degrade" tag are RUNTIME STATE, not intent: they live in
    //     `Resources` / the result metadata, never on `DeviceConfig` and never
    //     on the pure-numeric `F2BlockTensor`. A
    //     non-throwing tagged-degrade path (e.g. canAccessPeer == "no") records
    //     the fallback there; it is NOT an error (§11.4).
    // -----------------------------------------------------------------------

    /// Enable peer access opportunistically (cudaDeviceEnablePeerAccess when
    /// canAccessPeer) for single-node multi-GPU (architecture.md §11.4). This is
    /// the MAY-WE knob: whether the backend is permitted to call
    /// cudaDeviceEnablePeerAccess at all. DISTINCT from `prefer_p2p_combine`
    /// (which path to take WHEN peer access is available) — see below. The
    /// combine gate (the four-term §4 gate defined ONCE in `f2_blocks_multigpu.cpp`,
    /// "THE §4 COMBINE GATE", §8 single-source) ANDs this term in: `false` here forces
    /// the host-staged baseline (tagged `HostStaged`) even when the device CAN peer
    /// and `prefer_p2p_combine` is true, since the device-resident path would call the
    /// very `cudaDeviceEnablePeerAccess` this veto forbids.
    bool enable_peer_access = true;

    /// Prefer the device-resident P2P combine over the host-staged combine WHEN
    /// peer access is available (architecture.md §11.4). This is the WHICH-PATH
    /// knob and is DISTINCT
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
    /// the override-knob banner above). The multi-GPU combine reads this knob in
    /// its four-term §4 gate — defined ONCE in `f2_blocks_multigpu.cpp` ("THE §4
    /// COMBINE GATE", §8 single-source), which ANDs BOTH override-intent levers
    /// (`prefer_p2p_combine` WHICH-PATH AND `enable_peer_access` MAY-WE) with the
    /// discovered `can_access_peer` and the structural `G >= 2`. So a user who FORBIDS
    /// peer access (`enable_peer_access=false`) takes the host-staged baseline even
    /// when the device CAN peer and P2P is preferred — the device-resident path's
    /// `cudaDeviceEnablePeerAccess` is never reached against the veto.
    bool prefer_p2p_combine = true;

    /// Bit-stability INTENT for the statistic path (default ON). When true the
    /// run is held to the §12 reproducibility contract and `build()` enforces the
    /// constraints that make it hold (architecture.md §9 build()-validation list,
    /// §12):
    ///   * the statistic-bearing reductions run `run_to_run`-deterministic and
    ///     cuSOLVER runs in its scoped deterministic mode;
    ///   * `stream_count` is forced to 1 on the statistic path (cuBLAS
    ///     reproducibility does not hold across concurrent streams, §12);
    ///   * `precision == Precision::Kind::EmulatedFp64` requires the explicit
    ///     `cublasSetWorkspace` workspace (`kCublasWorkspaceBytes`), since
    ///     fixed-point emulation voids the run-to-run guarantee without an
    ///     adequate workspace (§12);
    ///   * the multi-GPU partials are combined in the fixed host-side device
    ///     order pinned by `devices` (§11.4) rather than a non-deterministic
    ///     AllReduce.
    /// This is OVERRIDE INTENT, not discovered state: it is the knob the §12
    /// stream_count/workspace/combine rules the parity-recompute path relies
    /// on are phrased against, and they are inexpressible without it. Set false
    /// only for throughput-only lanes whose results are recomputed in
    /// EmulatedFp64/Fp64 before any reported number (§12).
    bool deterministic = true;

    /// Force-tier OVERRIDE (default Auto = the select_output_tier policy). When set
    /// to Resident/HostRam/Disk it PINS the output tier regardless of free VRAM/RAM, so
    /// a test can exercise the Disk or HostRam stream at SMALL P (where Auto would pick
    /// Resident). It is the higher-precedence twin of the STEPPE_FORCE_TIER env var
    /// (config field wins). PARITY-NEUTRAL override intent (it moves bytes to a
    /// different tier, never a reported number, §12) — so it lives here, never on the
    /// numeric F2BlockTensor (override-knob banner above).
    enum class ForceTier { Auto, Resident, HostRam, Disk };
    ForceTier force_tier = ForceTier::Auto;

    /// Disk-tier on-disk f2_blocks cache path (TIER 2). Used only when the resolved
    /// tier is Disk; empty ⇒ the STEPPE_F2_CACHE_PATH env var, else the frozen default
    /// "./steppe_f2_blocks.cache" (cwd). The precompute-once/fit-many artifact (the
    /// on-disk f2_blocks cache; what ADMIXTOOLS 2 also keeps). Parity-neutral
    /// (it only chooses WHERE the bytes land, never a reported number, §12).
    std::string disk_cache_path;
};

// ---------------------------------------------------------------------------
// FilterConfig — on-the-fly QC thresholds (architecture.md §1, §5 S0').
// Promotes the absent-but-implied filter knobs. Defaults are
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
    /// whose missing fraction exceeds this. kMindFilterInactiveThreshold (1.0) ⇒
    /// keep every sample (the max possible missing fraction = "never drop").
    /// Decidable only from a streaming pre-pass over ALL SNPs (the conditional S-1
    /// pass), since per-sample missingness is not a single-tile quantity.
    double mind_max_missing = kMindFilterInactiveThreshold;

    // ----- Flag-gated filters (each defaults to a NO-OP so the parity /
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
    // prune.in (LD-pruned SNP id list) is READ here, never computed — steppe
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
