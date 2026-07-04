# `config.hpp` reference

## 1. Purpose

`include/steppe/config.hpp` is the single place the steppe library defines its
tunable settings. It holds three things:

1. **Named numeric constants** — every "magic number" that would otherwise be
   typed directly into the code lives here with a name and an explanation, so the
   rest of the codebase refers to the name instead of re-typing a bare number
   that could quietly drift out of sync.
2. **A precision policy** (`Precision`) — which flavor of floating-point math the
   heavy matrix-multiply stages should use.
3. **Configuration structs** (`DeviceConfig`, `FilterConfig`) and a small
   enum (`StrandMode`) — the settings a caller passes in to control which GPUs to
   use, how results are made reproducible, and which SNPs to keep or drop.

The header is intentionally free of any CUDA code. It only uses the C++ standard
library. That keeps it lightweight enough to be included everywhere — the core
library, the public API, the command-line tool, and the language bindings — without
forcing any of them to also pull in the GPU code.

Parity-neutral knobs only change *where* data lives or *how fast* something runs —
never a number that steppe reports back. Those knobs are safe to tune. A few knobs
are the opposite — changing them would change a reported result or break
reproducibility — and each of those is called out explicitly as frozen or "do not
change."

---

## 2. Named constants

### Precision and emulated-FP64 math

steppe runs its heavy matrix multiplications (the "f2" computations) using an
emulated form of double-precision arithmetic that is much faster than the GPU's
native double precision while staying about as accurate. These constants control
that.

| Constant | Value | What it's for |
|---|---|---|
| `kDefaultMantissaBits` | `40` | How many mantissa bits the emulated double-precision math keeps. 40 bits gives accuracy essentially as good as true double precision (worst-case error around 2.2e-11). Lowering it to 32 is faster but less accurate (worst-case error around 8.6e-9). Raising it to 48 is even more accurate than native double precision. These accuracy figures were measured on real data, not made up. |
| `kCublasWorkspaceBytes` | `64 MiB` (`64 × 1024 × 1024`) | The scratch-memory buffer size handed to the matrix-multiply library for the f2 computations. Giving it an explicit, fixed-size workspace is required for the emulated math to produce the exact same result every run. Formerly a bare "64 MiB" typed into the GPU code; now named and shared. |

The mantissa setting is always a *fixed* value. steppe deliberately does **not**
offer a "dynamic" mode that would pick the mantissa bit count automatically per
operation. On real-world data that dynamic approach was found to overshoot to
roughly 60 bits and collapse back to the speed of native double precision — all
cost, no benefit — so it was rejected and is simply not an option this file exposes.

### f-statistic sweep safety limits

When a user asks steppe to sweep over *every* combination of populations (for
example, every possible group of 4 out of 500), the number of combinations can be
astronomically large. These two constants keep such a sweep from accidentally
running for hours or exhausting memory.

| Constant | Value | What it's for |
|---|---|---|
| `kFstatMaxComb` | `100,000,000` (1e8) | The maximum number of combinations an all-combinations sweep will attempt before refusing to start. If a requested sweep would enumerate more than this, steppe stops up front — before doing any GPU work or allocating anything — unless the user explicitly passes `--sure`. This guards *compute time*, not just memory: every single combination is computed on the GPU in order to test it against the significance filter, so a billion-item sweep is hours of GPU work even if only a handful of results survive. The filter limits the output, never the amount of work. 100 million is a few minutes of GPU work at the measured rate. |
| `kFstatDefaultSweepTopK` | `1,000,000` (1e6) | How many results a bare sweep keeps when the user didn't specify a limit. Even though every combination is computed, steppe holds on to only the K most statistically significant ones (largest absolute z-score) in a fixed-size buffer on the GPU. Because only these K rows (about 40 MB at one million) ever get copied back to the host, a full 2.57-billion-item sweep can never blow up host memory no matter how many billions are computed. The user can override this with `--top-k`. |

### GPU kernel launch geometry

| Constant | Value | What it's for |
|---|---|---|
| `kCdivBlock` | `16` | The edge length of the square block of GPU threads used by the elementwise f2 kernels. It defines a 16×16 (256-thread) 2-D block that maps over the P×P output grid. This replaces a bare `dim3 block(16,16)`. The actual per-kernel launch dimensions live in a single launch-configuration helper; kernels never pick their own block size. |

### Numerical floors and divide-by-zero guards

These are tiny threshold values that keep the math well-behaved when a quantity is
at or near zero.

| Constant | Value | What it's for |
|---|---|---|
| `kRelFloor` | `1e-12` | A floor for relative-error comparisons. When a reference f2 value is smaller than this, the comparison switches to using absolute error instead, so that near-zero values don't produce a meaningless, blown-up relative error. It also serves as the divide-by-zero guard in those comparisons. Used by tests and accuracy checks. |
| `kAbsFloor` | `1e-300` | An absolute floor used to protect denominators in relative-error math (for example, dividing by `max(|reference|, kAbsFloor)`). This is roughly the smallest normal double-precision magnitude, so it is the smallest number the guards will treat as genuinely nonzero. |
| `kHetCorrDenomFloor` | `1.0` | A true mathematical floor. In the per-SNP heterozygosity correction, a term is divided by `max(N-1, 1)`; this constant is that `1`. It prevents a divide-by-zero when a population has only a single non-missing haploid sample (where N-1 would be 0). This same value is shared by the estimator used by both the CPU reference and the GPU path, so the two can never disagree. |

### Genetic map, jackknife blocks, and chromosomes

steppe estimates uncertainty using a block jackknife, which partitions SNPs into
blocks along the genome. These constants control how those blocks are defined and
which chromosomes count.

| Constant | Value | What it's for |
|---|---|---|
| `kDefaultBlockSizeCm` | `5.0` | The default jackknife block size, in centimorgans. This matches the parity default of 0.05 Morgans[^at2] (which is the same as 5 centimorgans). The config surface speaks in centimorgans, while the internal block math works in Morgans, and the conversion happens in exactly one place. |
| `kCentimorgansPerMorgan` | `100.0` | The conversion factor between the two units above — 100 centimorgans per Morgan. Kept as a single named constant so the centimorgan-facing config and the Morgan-based block rule always convert consistently. |
| `kBpFallbackWindow` | `2,000,000` (2e6) | The fallback block window, in base pairs, used when a dataset ships with no genetic linkage map (that is, the genetic-position column in the `.snp`/`.bim` file is all zeros, which is common for modern data derived from VCF or PLINK). In that case steppe partitions blocks by a hardcoded 2-megabase span of physical position instead. This exactly reproduces the reference behavior in the same situation[^at2]. It is a distinct physical-distance rule — not the 5-centimorgan genetic default. |
| `kAutosomeChromMin` | `1` | The lowest chromosome number counted as an autosome. |
| `kAutosomeChromMax` | `22` | The highest chromosome number counted as an autosome. Together with the min above, this defines "autosomes only" as chromosomes 1 through 22, which matches the parity default of keeping only chromosomes 1–22[^at2] and dropping the sex chromosomes (X, Y) and mitochondrial/other codes. Named here rather than typing a bare `22`, so the definition of "autosome" lives in exactly one place. |

### The `--mind` filter's off switch

| Constant | Value | What it's for |
|---|---|---|
| `kMindFilterInactiveThreshold` | `1.0` | The value that means "the per-sample missing-data filter is turned off." Since 1.0 is the largest possible missing fraction, setting the filter's threshold to 1.0 means no sample can ever exceed it, so nothing is dropped. This is both the default for the filter field and the value the pre-pass checks against to decide whether it can skip its work entirely — kept in one place so the two can't drift apart. |

### VRAM budgeting and the memory-tier system

steppe can hold its f2 results entirely in GPU memory (fastest), spill them to host
RAM, or spill them all the way to disk, depending on how big the problem is. The
constants here feed the logic that (a) predicts how much memory each option needs
and (b) decides which tier to use. Almost all of these are parity-neutral — they
change *where* bytes land or how memory is chunked, never any reported number.

#### Memory-footprint coefficients

These are plain integer counts. They describe how many buffers of a given size a
particular phase allocates, so the memory-planning code can compute a footprint from
a single source of truth instead of re-deriving the same numbers in several places
(which risks them silently drifting away from the real allocations).

| Constant | Value | What it's for |
|---|---|---|
| `kFeederRawBufsPerPop` | `3` | The GPU-input phase (the "feeder") holds 3 raw decoded buffers per population: the decoded genotype, variance, and count arrays. |
| `kFeederOutBufsPerPop` | `4` | The feeder also persists 4 output buffers per population. (Their sum with the raw buffers, 3 + 4 = 7, is the per-population memory coefficient for the all-in-GPU-memory tier.) |
| `kResidentTensorCount` | `2` | The number of large tensors held in GPU memory for the whole computation loop: the f2 tensor and its paired-variance tensor. This is the "2" in their combined footprint. |
| `kChunkInputStacks` | `4` | The number of input "stacks" in one batched matrix-multiply chunk: the gathered genotype and variance inputs (one each) plus the gathered scale input (two), totaling four. |
| `kChunkOutputStacks` | `4` | The number of output stacks in the same chunk: the two GEMM outputs (one each) plus the result buffer (two), totaling four. |
| `kStreamDeviceChunks` | `2` | How many GPU buffers the streaming (host-RAM or disk) path cycles through, so that one chunk's copy back to the host can finish while the next chunk is being computed. Deliberately two rather than three: a GPU buffer only needs to survive its own copy-to-host, because the slow disk/RAM write is absorbed by a separate host-side buffer ring. Keeping it at two keeps GPU memory use small. |

#### VRAM/RAM occupancy fractions

These say what fraction of available memory each stage is allowed to use before the
tier-selection logic backs off to the next tier. They are tuning numbers, not
mathematical constants.

| Constant | Value | What it's for |
|---|---|---|
| `kMaxVramUtilizationFraction` | `0.80` | The target fraction of free GPU memory the working set may occupy. This is the single home for both the up-front "will it fit?" check and the runtime chunk-sizing, so the two can never use different fractions. The fraction is applied to the free memory that *remains after* the large resident tensors and the matrix-multiply workspace have already been subtracted — so it is scaling the headroom for one temporary chunk, not the raw free memory. Using 0.80 of that already-reduced amount leaves at least a 20% margin for matrix-multiply scratch space and memory-allocator fragmentation. |
| `kResidentTierVramFraction` | `0.70` | The fraction of free GPU memory the all-in-GPU-memory ("resident") tier's result plus its working set may occupy before the tier-selector declines it and falls back to host RAM. Kept strictly below the 0.80 above so that the resident result plus its working set together cannot exceed free GPU memory. |
| `kHostTierRamFraction` | `0.60` | The fraction of free host RAM the host-RAM tier's result may occupy before the tier-selector declines it and falls back to disk. Below 1.0 so that the host result plus small staging buffers and operating-system overhead stay within free RAM on a normal machine. |
| `kStreamTileBudgetFraction` | `0.25` | In the streaming path, the fraction of the GPU memory budget reserved for the input feeder; the remaining 75% goes to the result slabs and the buffer ring. |

Two compile-time checks enforce sane relationships: `kResidentTierVramFraction`
must be greater than 0 and no larger than `kMaxVramUtilizationFraction`, and
`kHostTierRamFraction` must be greater than 0 and no larger than 1.

#### Fit-path memory budget

The second-phase model fit sizes its work against available GPU memory. These two
values are the fallback and safety-margin used in that calculation.

| Constant | Value | What it's for |
|---|---|---|
| `kFitBudgetFreeVramFallbackBytes` | `4 GiB` (`4 << 30`) | When the runtime can't determine how much GPU memory is free (the probe returns 0/unknown), the fit assumes this much is free — a conservative guess. |
| `kFitBudgetHeadroomBytes` | `512 MiB` (`512 << 20`) | A fixed amount of GPU memory the fit sets aside before dividing the rest by the per-model cost, so that a chunk never tries to claim all of the free memory. |

Both are tunable policy numbers that change the chunk size only, never a reported
result. They live here so that the fit path's memory levers sit next to the f2
path's memory policy rather than being scattered as bare numbers inside a function.

### Miscellaneous constants

| Constant | Value | What it's for |
|---|---|---|
| `kBlockGroupPadBase` | `2` | The base used when bucketing genome blocks by size in the batched f2 path. Blocks are grouped by rounding their size up to the next power of this base (i.e. the next power of two), and each group runs as one padded batched matrix-multiply. Using power-of-two buckets keeps the wasted padding within a group under a factor of 2, while keeping the number of batched calls small (proportional to the logarithm of the largest block). Measured on real data: 768 populations produced 10 groups with only 1.43× padding waste, versus 2.76× for a simpler design. The padding columns carry zeros, so they contribute nothing to the results. |
| `kGesvdjMaxDim` | `32` | The size threshold for choosing which singular-value-decomposition routine to use. If both dimensions of a matrix are 32 or smaller, steppe uses the faster Jacobi-based routine; otherwise it uses the QR-based routine. 32 is the size limit of the underlying batched Jacobi routine. **Do not change this value.** It is frozen for reproducibility: the reference results assert that a matrix with a dimension of 39 (greater than 32) took the QR path, and a small 2×5 matrix (both under 32) took the Jacobi path. You may rename the constant, but the value must stay 32. It is a routine-selection threshold, not a hardware warp size. |
| `kDefaultSearchStreams` | `4` | A reserved default for a not-yet-built multi-GPU model-space search feature. Nothing in production reads it today, so it controls no computation and is safe by construction. |
| `kInvalidDeviceId` | `-1` | The sentinel meaning "no device" for a CUDA device-id field — used for an empty, moved-from, or not-yet-resolved device id. Named so the "-1 means no device" convention is one symbol rather than a bare `-1` repeated at every device-id default. |
| `kDefaultDiskCachePath` | `"./steppe_f2_blocks.cache"` | The default file path for the on-disk f2 cache, relative to the current directory. This is the last-resort value, used only when both the `disk_cache_path` config field and the `STEPPE_F2_CACHE_PATH` environment variable are empty. Parity-neutral: it only chooses where bytes land. |

---

## 3. Precision

`Precision` is the typed knob that says which flavor of arithmetic the heavy
matrix-multiply stages run in. There are three modes, and steppe picks a mode based
on how numerically delicate an operation is, not on its size.

### `Precision::Kind` — the three modes

| Mode | What it is | When it's used |
|---|---|---|
| `Fp64` | Native double precision. | The gold-standard reference that everything else is validated against, and the fallback. It is also always used for the numerically delicate parts regardless of the selected mode: the small, cancellation-prone f2 numerator/divide and the ill-conditioned linear-solve and singular-value-decomposition steps. It produces bit-for-bit identical results run to run when run on a single stream. |
| `EmulatedFp64` | Emulated double precision built from fixed-size slices. | **The default** for all the matrix-multiply-heavy stages, including the f2 computations. Measured at 7 to 17 times faster than native double precision on real data, at essentially the same accuracy. It is accuracy-*approximate*, not bit-identical to native double precision and not fully standards-compliant on special values like infinities — which is exactly why native `Fp64` remains the reference oracle. |
| `Tf32` | TF32 tensor-core arithmetic (lower precision). | Opt-in, and only for quickly screening or ranking a space of models. Its results are flagged as approximate and are held to a loose tolerance. They are never bit-compared against the reference values[^at2], and are never reported as a final estimate/standard-error/z-score/p-value without being recomputed in one of the higher-precision modes first. |

A build-time subtlety worth knowing: the fixed-mantissa emulation is only actually
engaged when the GPU code is built with a specific capability enabled (on by
default). If that capability is compiled out, selecting `EmulatedFp64` is not
honorable — so instead of silently running the rejected dynamic-mantissa mode, the
GPU path quietly downgrades to native `Fp64` and logs a note. This header can't
enforce that itself, since the switch lives in the GPU layer, but the behavior is
worth being aware of.

### `Precision` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `Kind` | `EmulatedFp64` | Which of the three arithmetic modes to use for the matrix-multiply-heavy stages. |
| `mantissa_bits` | `int` | `40` (`kDefaultMantissaBits`) | How many mantissa bits the emulated mode keeps. Only meaningful when `kind` is `EmulatedFp64`; ignored for `Fp64` and `Tf32`. 40 is about native accuracy, 32 is faster and less accurate, 48 exceeds native. This is always a fixed cap — the rejected dynamic mode cannot be expressed by this struct at all. |

### Convenience factory functions

Instead of writing out the brace-initialization by hand, callers can use three
short factory functions. Each returns exactly the same `{kind, mantissa_bits}`
values as writing it out longhand.

- `Precision::fp64()` — native double precision.
- `Precision::emulated_fp64(bits = 40)` — emulated double precision; calling it
  with no argument gives exactly the default (40 bits).
- `Precision::tf32()` — TF32 screening mode.

A default-constructed `Precision{}` is identical to `Precision::emulated_fp64()`.
Compile-time checks guarantee these factories and the default stay exactly in sync,
so none of them can silently change mode or mantissa.

---

## 4. DeviceConfig

`DeviceConfig` bundles the resource and reproducibility settings a caller passes in.
It is designed to be handed in explicitly rather than discovered from global state.

A useful mental model for this struct: it holds the user's **intent** — what the run
*may* or *prefers* to do. It deliberately does **not** hold discovered runtime facts
(like whether two GPUs can actually talk to each other, or how much memory is
currently free) or the record of which path a run actually took. Those are runtime
state that lives elsewhere, in the results metadata.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `devices` | `vector<int>` | empty | The single source of truth for which GPUs to use and in what order. Empty means auto-detect every visible GPU in enumeration order. A non-empty list pins both the exact set of GPUs and their ordering — and that ordering is what fixes the deterministic order in which multi-GPU partial results get combined. A list of size 1 means single-GPU. There is no separate "count" field; the count is just the list's length. |
| `precision` | `Precision` | emulated FP64, 40-bit | The precision policy (see `Precision`). |
| `stream_count` | `size_t` | `1` | **Must stay 1.** The GPU path always uses a single statistics stream. This field is reserved and gates nothing. It is pinned to 1 because the matrix-multiply library's results are not reproducible across concurrent streams — it is a reproducibility trap, not a throughput dial. |
| `search_streams` | `size_t` | `4` (`kDefaultSearchStreams`) | Reserved for a not-yet-built multi-GPU model-search feature. Nothing reads it today. |
| `use_mem_pool` | `bool` | `true` | Reserved and not wired. The GPU path always uses the pool-backed asynchronous allocator regardless of this field. |
| `enable_peer_access` | `bool` | `true` | The "may we?" knob for multi-GPU: whether the backend is *permitted* to turn on direct GPU-to-GPU peer access at all. If false, the run is forced onto the host-staged combine path even if the hardware supports peer access — because the faster path would have to call the very peer-access function this vetoes. |
| `prefer_p2p_combine` | `bool` | `true` | The "which path?" knob for multi-GPU, and distinct from `enable_peer_access`. When peer access *is* available, this says to prefer the direct device-to-device combine (GPU 0 pulls each other GPU's partial result over a direct copy and sums them in the fixed device order) over gathering all partials to the host and summing there. Both combine paths sum in the same fixed order, so they produce bit-for-bit identical results to each other and to a single-GPU run — the choice only affects how bytes move, so it is parity-neutral. If peer access turns out to be unavailable, the backend quietly and safely falls back to the host-staged path. |
| `deterministic` | `bool` | `true` | The intent to hold the run to the reproducibility contract. When true, `build()` enforces the rules that make results bit-reproducible: deterministic reductions, deterministic mode for the linear-algebra library, `stream_count` forced to 1, the emulated mode required to use its explicit workspace, and multi-GPU partials combined in the fixed device order rather than a nondeterministic all-reduce. Set false only for throughput-only work whose results are recomputed in a higher-precision mode before anything is reported. |
| `force_tier` | `ForceTier` | `Auto` | Overrides the automatic memory-tier choice. `Auto` lets the policy decide; setting `Resident`, `HostRam`, or `Disk` pins that tier regardless of how much memory is free. This mainly lets tests exercise the disk or host-RAM path even on a small problem that would otherwise stay in GPU memory. It takes precedence over the `STEPPE_FORCE_TIER` environment variable. Parity-neutral — it moves bytes to a different tier, never a reported number. |
| `disk_cache_path` | `string` | empty | Where to put the on-disk f2 cache when the resolved tier is Disk. If empty, steppe uses the `STEPPE_F2_CACHE_PATH` environment variable, and if that too is empty, the frozen default `./steppe_f2_blocks.cache`. This is the "compute once, fit many times" cache artifact — the same kind of cache the reference implementation keeps[^at2]. Parity-neutral. |

### `DeviceConfig::ForceTier`

A small enum with four values used by the `force_tier` field:

- `Auto` — let the automatic policy pick the tier (the default).
- `Resident` — force everything to stay in GPU memory.
- `HostRam` — force the spill-to-host-RAM path.
- `Disk` — force the spill-to-disk path.

---

## 5. StrandMode

`StrandMode` controls what to do with strand-ambiguous SNPs — the palindromic ones
(an A/T or C/G pair) whose two possible strand readings are indistinguishable, which
makes them a classic source of corruption when merging data from different sources.

| Value | Meaning |
|---|---|
| `Drop` | **The default.** Drop all strand-ambiguous SNPs. This is the merge-safety choice and reproduces steppe's original, pre-flag behavior bit-for-bit. |
| `Keep` | Keep strand-ambiguous SNPs (as long as they are otherwise clean biallelic A/C/G/T SNPs). This reproduces the reference default behavior, which keeps ambiguous SNPs — so use this for an exact parity match[^at2] on a panel that still contains palindromes. |
| `Flip` | Not yet implemented. It is accepted as a documented token but currently behaves exactly like `Keep`: it does not drop palindromes, but it also performs no frequency-based strand correction. A future frequency-based reorientation pass will go here. |

Multiallelic SNPs are always dropped regardless of this setting; even `Keep` and
`Flip` still require clean biallelic A/C/G/T pairs.

---

## 6. FilterConfig

`FilterConfig` holds the on-the-fly quality-control filters applied while reading
genotypes. **Every default is a no-op** — out of the box, nothing is filtered, so
the standard path is untouched unless you explicitly ask for a filter.

### Continuously-valued thresholds

| Field | Type | Default | Meaning |
|---|---|---|---|
| `maf_min` | `double` | `0.0` | Minimum minor-allele frequency to keep a SNP. `0.0` means no filter. The frequency used is the *pooled folded* minor allele frequency — the minor-allele frequency computed by pooling reference and allele counts across all kept samples together (not a per-population frequency). |
| `geno_max_missing` | `double` | `1.0` | Maximum per-SNP missing-data fraction (the "geno" filter): drop a SNP whose missing fraction exceeds this. `1.0` means keep every SNP. The fraction is measured over the *individuals* axis — the fraction of kept individuals with missing data at that SNP (the PLINK `--geno` convention[^plink]). Note that the reference's similar `maxmiss` uses a different denominator[^at2] (fraction over populations). |
| `mind_max_missing` | `double` | `1.0` (`kMindFilterInactiveThreshold`) | Maximum per-sample missing-data fraction (the "mind" filter): drop a *sample* whose missing fraction across all SNPs exceeds this. `1.0` means keep every sample. This one requires a streaming pre-pass over all SNPs to decide, because per-sample missingness can't be computed from a single tile of data. |

### Flag-gated filters

Each of these defaults to off (a no-op) so the standard path is untouched unless the
flag is explicitly set.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `autosomes_only` | `bool` | `false` | Keep only SNPs on autosomes (chromosomes 1–22). `false` keeps every chromosome. The parity default is autosomes-only[^at2], so set this true to match it; it is off here so the default keeps the sex chromosomes too. |
| `drop_monomorphic` | `bool` | `false` | Drop SNPs with no variation (pooled minor-allele frequency of exactly 0). `false` keeps them. This is effectively the strict-positive boundary of `maf_min`, but kept as its own named flag. |
| `transversions_only` | `bool` | `false` | Keep only transversion SNPs (a purine↔pyrimidine change) and drop transitions (A↔G, C↔T). `false` keeps transitions too. |
| `strand_mode` | `StrandMode` | `Drop` | The strand-ambiguous-SNP policy (see `StrandMode`). |

### Include / exclude SNP lists

These let you constrain exactly which SNPs are used, by their IDs. An empty value
means no constraint. steppe *reads* an externally supplied LD-pruned SNP list but
never computes linkage disequilibrium itself.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `include_snp_ids` | `vector<string>` | empty | An explicit keep-set of SNP IDs (like `--extract`). If non-empty, keep only these IDs (intersected with whatever else passes the other filters). These are SNP IDs from the first column of the `.snp` file, not row indices. |
| `exclude_snp_ids` | `vector<string>` | empty | An explicit drop-set of SNP IDs (like `--exclude`). Any ID listed here is dropped even if it would otherwise pass. |
| `prune_in_path` | `string` | empty | Path to an external `prune.in` file — one SNP ID per line — listing LD-pruned SNPs to keep. It is read, never computed. When set, it acts as an additional keep-set that composes with `include_snp_ids`. |

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
[^plink]: **PLINK** — the PLINK / PACKEDPED (`.bed`/`.bim`/`.fam`) genotype format and toolset. Chang CC, Chow CC, Tellier LCAM, Vattikuti S, Purcell SM, Lee JJ. *Second-generation PLINK: rising to the challenge of larger and richer datasets.* GigaScience 2015;4:7.
