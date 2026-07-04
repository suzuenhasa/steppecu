# `backend.hpp` reference

## 1. Purpose

`src/device/backend.hpp` defines the single interface every compute operation in
steppe passes through: `ComputeBackend`. The core library never issues a matrix
multiply, singular-value decomposition, or Cholesky factorization itself — it
hands the work to a `ComputeBackend`, and one of two concrete implementations
carries it out:

- **`CudaBackend`** — the real GPU implementation (the deliverable), which runs
  everything as resident, batched device kernels.
- **`CpuBackend`** — a plain scalar implementation in native double precision.
  It is the *reference oracle*: the trusted answer the GPU results are
  continuously compared against, and the shared single source of the math so the
  two paths cannot quietly drift apart. It is a development/test tool, never a
  user-facing runtime.

The file also declares all the plain-data structs these methods take in and hand
back — genotype tiles going in, f2 matrices and fit results coming out — plus a
handful of named constants and two host helper functions.

Everything here is expressed once against the interface. The compute layer is
written a single time and never branches on "is this a GPU or a CPU"; picking a
backend picks the implementation.

---

## 2. The CUDA-free seam and the two backends

This header is deliberately free of any CUDA code, by contract. It only uses the
C++ standard library and steppe's own plain headers. That is what lets the core
library and the command-line tool include it and call into the GPU without
themselves having to pull in the CUDA toolkit — CUDA stays private to the device
layer, behind this one door. Every type that crosses this boundary is plain data:
ordinary host vectors and structs, never a GPU-specific type. A device result is
copied into a plain host vector before it is returned across the seam. (Some
methods instead return an *opaque handle* to data that stays on the GPU; those
handle types are themselves CUDA-free wrappers.)

### One backend instance per GPU

A single `ComputeBackend` instance is bound to exactly one GPU. When more than
one GPU is in play, there is one backend (and one set of per-GPU resources — its
own stream and math-library handles) per device. The interface is therefore
single-device on purpose: splitting work across several GPUs and combining their
partial results is orchestrated *above* this interface, not inside any one
method. Backends are move-only and owned through a smart pointer; they cannot be
copied.

### The "base throws" pattern for GPU-only methods

Many methods are GPU-only concepts (anything that leaves a result resident in GPU
memory, streams to disk, or runs a batched device kernel). For these, the base
class provides a default body that simply throws a clear error saying the
operation needs a CUDA backend. Only `CudaBackend` overrides them; `CpuBackend`
and any lightweight test stand-in inherit the throwing default and never call
them. This keeps the header compilable with no CUDA present and avoids forcing
the CPU oracle to implement device-only plumbing.

A few methods are the exception: their default body is a *real, correct*
CPU implementation, because the operation is pure integer/bit work that is
naturally expressible on the host. `detect_sample_ploidy_device` and
`transpose_to_canonical` are like this — the base body *is* the oracle the GPU
version is checked against, so a backend that doesn't override them still gets the
right answer.

### Capability queries instead of catch-the-exception

Because some methods throw a sentinel error when not implemented, the header adds
explicit yes/no capability queries — `provides_rank_sweep()` and
`provides_batched_fit()` — so callers can *ask* whether a real implementation
exists instead of calling the method and catching the sentinel. This matters:
catching the sentinel would also swallow a *genuine* numerical error thrown from
inside a real implementation. The base returns `false` for both; only
`CudaBackend` overrides the underlying method and so returns `true`.

---

## 3. Precision across the seam

Almost every compute method takes a `Precision` argument. It governs one thing
only: which flavor of arithmetic the heavy matrix-multiply sub-steps use. It does
**not** govern the numerically delicate steps.

- **Matrix-multiply-heavy work** (the f2 computations, covariance products)
  defaults to emulated double precision (`EmulatedFp64{40}`), which is measured at
  roughly 7–17× faster than native double on real data at essentially the same
  accuracy.
- **The numerically delicate work** — the small, cancellation-prone f2
  numerator and divide, and the ill-conditioned linear solves and singular-value
  decompositions — always runs in native double precision regardless of the
  `Precision` argument. Emulation is a matrix-multiply optimization; it does not
  help (and can hurt) a cancellation-sensitive reduction, so those steps are
  carved out to stay native.

`CpuBackend` ignores the `Precision` argument entirely and computes everything in
native double precision — it is the oracle, so it is unconditionally the
reference flavor.

There is a separate opt-in knob, `set_solve_precision`, that lets a caller
*promote* the normally-native solve stages (SVD/Cholesky/weight solve) to the
emulated tensor-core path — intended for the very high-throughput model search
where there are millions of tiny solves. The default is native, so a backend that
never calls it behaves exactly as the native reference. This is a
measurement/exploration seam today.

---

## 4. The f4 flatten order (a binding invariant)

Several structs store a per-block f4 matrix flattened into a one-dimensional
array. The order that flattening uses is fixed here, once, and must not change:
the `m = nl·nr` entries of a per-block f4 matrix are laid out **row-major** over
`(i, j)` as `k = j + nr·i`, where `i` runs over the left sources and `j` over the
right populations.

This is not an arbitrary choice — it is the parity order for the equivalent
quantities[^at2], and the covariance matrix, the jackknife, and the
weight solve all index into these flattened vectors assuming exactly this order.
Using the transposed order `i + nl·j` instead would silently mismatch the f4
vector against the covariance matrix and break parity. Any
code touching these flattened arrays must honor `k = j + nr·i`.

---

## 5. Genotype decode: input tile views and output contracts

These structs are the plain-data inputs and outputs of turning packed genotype
bytes into allele frequencies. None of them names a GPU type, so they cross the
seam by value; borrowed pointers are valid only for the duration of the call.

### `DecodeTileView` — the input tile

A non-owning view of one packed genotype tile plus its population partition. The
packing is individual-major: the bytes hold one record per gathered individual,
with individuals laid out contiguously by population, and each 2-bit code is a
single SNP for that individual (most-significant-bit first within a byte).

| Field | Meaning |
|---|---|
| `packed` | The packed genotype bytes, population-contiguous records. |
| `bytes_per_record` | Stride between one individual's record and the next. |
| `n_snp` | Number of SNPs in the tile (the column axis). |
| `n_individuals` | Total gathered individuals across all populations. |
| `pop_offsets` | Population segment boundaries over the individual axis (length P+1). |
| `n_pop` | Number of populations P. |
| `sample_ploidy` | Optional per-sample ploidy (2 diploid, 1 pseudo-haploid). |
| `ploidy` | Uniform fallback ploidy used only when `sample_ploidy` is null (default 2). |
| `detect_ploidy_on_device` | Ask the backend to derive per-sample ploidy itself. |

Ploidy handling deserves a note. The pseudo-haploid auto-detection[^at2]
works *per sample* — a sample that ever shows a heterozygous call is diploid,
otherwise it is pseudo-haploid — so mixed-ploidy populations (real for ancient
DNA) are handled correctly. `sample_ploidy`, when non-null, gives that per-sample
answer explicitly and overrides the scalar `ploidy`. When it is null, the scalar
`ploidy` applies uniformly (the legacy all-diploid path stays bit-identical).

`detect_ploidy_on_device` is a third option: when true *and* `sample_ploidy` is
null, the backend derives per-sample ploidy itself, in lockstep with the decode,
from the same bytes — moving that scan off the host critical path. It is ignored
when an explicit `sample_ploidy` is supplied (explicit always wins) and is false
by default. It is bit-identical to the host detector by construction, because it
is a literal port of the same scan over the same primitives.

### `DecodeResult` — the decode output

The allele-frequency contract as plain column-major `[P × M]` host arrays
(element for population `i`, SNP `s` lives at index `i + P·s`, the layout that
drops straight into the matrix views the f2 computation consumes).

| Field | Meaning |
|---|---|
| `q` | Reference-allele frequency in `[0, 1]`, zero where invalid. |
| `v` | Validity mask (1.0 valid / 0.0 missing). |
| `n` | Non-missing haploid count (ploidy × non-missing individuals). |
| `P` | Number of populations. |
| `M` | Number of SNPs. |

### `SnpMajorTileView`, `CanonicalTile`, `TileEncoding` — the transpose path

Some file formats store genotypes SNP-major (one record per SNP, individuals
interleaved within each SNP) rather than individual-major. `SnpMajorTileView` is
the non-owning input view of such a source together with the selected, reordered
list of individuals to gather. The backend transposes it into the canonical
individual-major layout that the rest of the pipeline already consumes, returned
as a `CanonicalTile` (a small plain struct, not a full I/O tile, so the header
stays independent of the I/O layer). `TileEncoding` is the native-code to
canonical-code map a SNP-major source needs before its codes re-pack; the first
supported format already uses the canonical convention, so its map is `Identity`,
and later formats add enumerators without touching the transpose body. The
transpose is pure 2-bit unpack, remap, and re-pack — no floating point — so it is
exactly bit-identical across backends.

A subtlety worth knowing: a SNP-major record can be padded to a minimum stride
larger than the individual count strictly requires. Only the explicitly selected
individual rows (each below the real individual count) are ever read, so those
padding bytes are never decoded as phantom individuals.

---

## 6. The f2 result types

### `F2Result` — one f2 matrix

The output of an f2 computation: the symmetric f2 matrix and the pairwise-valid
SNP counts, both column-major `[P × P]` (element `(i, j)` at `i + P·j`).

| Field | Meaning |
|---|---|
| `f2` | The bias-corrected f2 matrix. |
| `vpair` | Pairwise-valid SNP count: number of SNPs valid in both populations `i` and `j`. |
| `P` | Number of populations. |

**Diagonal convention (pinned).** `f2(i, i)` carries the full same-population
computation, *not* a forced zero. Because the two frequencies are identical, the
per-SNP term reduces to `−2 × (within-population heterozygosity correction)`,
generally nonzero. This is a within-population quantity, not a between-population
f2, and downstream statistics only read off-diagonal f2 — but both backends fill
the diagonal identically by construction, so that a full-matrix comparison
(diagonal included) can never reintroduce a difference between them.

**Vpair is retained, not discarded.** `vpair` is kept because it is the weight the
block jackknife needs later. The per-pair divide and the later weighting must
compose to the parity definition without double-normalizing[^at2]. The diagonal
`vpair(i, i)` is population `i`'s own valid-SNP count and is likewise filled, not
zeroed.

---

## 7. Fit-engine result types

These plain structs carry the intermediate and final results of the qpAdm model
fit across the seam. They all use the flatten order from section 4.

### `F4Blocks` — per-block f4 plus the jackknife point estimate

| Field | Meaning |
|---|---|
| `x_blocks` | `[m × n_block]` per-block f4 matrix, flattened `k = j + nr·i` per block. |
| `x_total` | `[m]` the weighted-jackknife point estimate of f4. |
| `x_loo` | `[m × n_block]` per-entry leave-one-out replicate values, carried so a re-fit needs no recompute. |
| `nl` | Rows of the f4 matrix (number of left sources). |
| `nr` | Columns of the f4 matrix (number of right populations, minus the reference). |
| `n_block` | The surviving block count. |
| `block_sizes` | `[n_block]` per-surviving-block SNP count — the jackknife weight. |

The block count is the *survivor* count for a reason. A jackknife block in which
any population pair has zero jointly-valid SNPs is a missing block; such
blocks are dropped entirely before the jackknife[^at2] (rather than imputing zero, which
would bias f4 toward zero and inflate variance). So the f4 block arrays are
compacted to the survivors, and the survivor SNP counts are carried here directly.
When there are no missing blocks — the common case — this equals the full block
list and the path is byte-identical.

### `JackknifeCov` — the covariance and its inverse

| Field | Meaning |
|---|---|
| `Q` | `[m × m]` the covariance matrix, unfudged (the reference convention). |
| `Qinv` | `[m × m]` inverse of the *fudged* covariance (diagonal nudged by `fudge × trace(Q)`). |
| `m` | Dimension. |
| `status` | Reports non-invertibility as a value, not a thrown exception. |

### `JackknifeDiag` — the diagonal-only variance

For the per-item statistics (f4/f3) that read only the diagonal of the covariance
and never invert it, this computes just the per-item variance without ever forming
the dense `m × m` matrix or its inverse. That is the production-scale shape: a
sweep of tens of thousands of items would make a dense matrix that needs tens of
gigabytes, so the diagonal path is linear in memory. `var[k]` is exactly the
`k`-th diagonal entry the full covariance would produce, so the goldens do not
move.

| Field | Meaning |
|---|---|
| `var` | `[m]` the per-item diagonal variance. |
| `m` | Dimension. |
| `status` | Always `Ok` (no inversion means no non-SPD failure). |

### `GlsWeights` — the weight fit

| Field | Meaning |
|---|---|
| `w` | `[nl]` normalized admixture weights (sum to 1). |
| `A` | `[nl × r]` refined left factor, column-major. |
| `B` | `[r × nr]` refined right factor, column-major. |
| `chisq` | The fit's chi-squared. |
| `r` | Rank. |
| `status` | Reports a degenerate solve as a value. |

### `RankSweep` — the rank test over all candidate ranks

The qpWave/qpAdm rank test over ranks `0 … rmax`. It carries both the per-rank
chi-squared/degrees-of-freedom/p-value and the nested "rankdrop" table[^at2]
(rows ordered by descending rank, each row compared to the
next), plus the chosen rank and the numerical rank of the covariance.

| Field | Meaning |
|---|---|
| `chisq`, `dof`, `p` | Per-candidate-rank chi-squared, degrees of freedom, p-value. |
| `rd_f4rank`, `rd_dof`, `rd_chisq`, `rd_p` | The nested table's rank, dof, chi-squared, p per row. |
| `rd_dofdiff`, `rd_chisqdiff`, `rd_p_nested` | Row-to-next-row differences; sentinel values mark the last (not-applicable) row. |
| `f4rank` | The smallest non-rejected rank. |
| `rank_Q` | Numerical rank of the covariance. |
| `svd_path` | Which SVD routine *would* be selected — observability only, not what executed. |
| `status` | Propagates degenerate-solve outcomes as values. |

`svd_path` is a reporting field only: it records which SVD routine the sizes would
select, while the routine actually executed is a deterministic on-device method at
all sizes. This is a documented pending seam, not a live dispatch.

### `PopDropRow` — leave-one-source-out feasibility

One row per pattern over the left sources: the full model plus each single-source
drop. Built by the host orchestrator (which re-runs the fit on the reduced source
set), not by a backend method — it is recorded here only to fix the contract shape.

| Field | Meaning |
|---|---|
| `pat` | Bit pattern over the sources (`"1"` = dropped). |
| `wt` | Number of sources dropped. |
| `dof`, `chisq`, `p`, `f4rank` | The reduced fit's statistics. |
| `weight` | Per-source weight for the surviving sources (not-a-number for a dropped slot). |
| `feasible` | Whether all surviving weights fall in `[0, 1]`. |
| `status` | Domain outcome as a value. |

---

## 8. Ratio block-jackknife types

There is one shared engine for two ratio statistics — the f4-ratio and the
normalized D statistic — because both are the same block-jackknife of a per-block
ratio, differing only in a few parameters. These structs feed and return from it.

### `RatioJackArray` — one input array descriptor

A CUDA-free descriptor for one per-(item, block) input array, so two callers with
different memory layouts can feed the same kernel with no repacking. The kernel
reads element `(item k, block b)` at `base + k·item_stride + b·block_stride`.

| Field | Meaning |
|---|---|
| `data` | Base pointer (host or device per the call); null means the array is absent. |
| `base` | Element offset of item 0, block 0. |
| `item_stride` | Step between items; `0` broadcasts the array across all items. |
| `block_stride` | Step between blocks. |

### `RatioBlockJackknife` — the output

One slot per item. The reported point estimate is the jackknife estimate (not the
plain block-sum total, which is measurably less accurate). The whole computation
is native double precision — it is a cancellation-sensitive statistic, never
emulated.

| Field | Meaning |
|---|---|
| `est` | `[N]` the jackknife point estimate per item. |
| `se` | `[N]` standard error. |
| `z` | `[N]` z-score (`est/se`). |
| `p` | `[N]` two-sided tail — filled only for the D-statistic path, empty for f4-ratio. |
| `N` | Item count. |
| `status` | Domain outcome as a value. |

---

## 9. qpfstats smoothing-solve type

### `QpfstatsSmooth`

The output of the joint f2 smoother, which reformulates a per-block regression
into one shared factorization and a single batched multi-column solve. On the GPU
this is one symmetric-product, one factorization, one product, and a pair of
triangular solves over all blocks at once — no per-block host loop.

| Field | Meaning |
|---|---|
| `b` | `[npairs × n_block]` column-major per-block smoothed coefficients. |
| `bglob` | `[npairs]` the global smoothed coefficients (the recentering target). |
| `recenter_shift` | `[npairs]` per-pair recenter shift; empty unless the fused entry computed it on-device. |
| `npairs` | The f-stat basis dimension. |
| `n_block` | The jackknife block count. |
| `status` | Reports a non-factorable matrix as a value (should not occur with a positive ridge). |

Blocks with a missing (not-a-number) combination row are handled by a rank-one
downdate of the shared factor before the solve; an all-missing block yields a zero
column (which must be zero, not dropped).

---

## 10. qpGraph types

### `QpGraphTopoArena` — a topology carried into the fit

All the flat integer and double arrays describing one admixture-graph topology,
uploaded once per topology. The device runs the same path-table fill, centered
weights, edge solve, and quadratic form per restart thread.

| Field | Meaning |
|---|---|
| `npop`, `nedge_norm`, `nadmix`, `npair`, `npath`, `base_leaf` | Topology sizes. |
| `pwts0` | `[nedge_norm × npop]` column-major, the parameter-independent weight table. |
| `pe_edge`, `pe_leaf`, `pe_path` | Path-edge table. |
| `pae_path`, `pae_admixedge` | Path-admix-edge table. |
| `cmb1`, `cmb2` | `[npair]` centered-column pair indices. |
| `constrained` | Whether drift edges are held non-negative. |
| `fudge` | The trace-scaled ridge added for numerical stability. |

### `QpGraphFleet` — the best-of-restarts fit for one topology

| Field | Meaning |
|---|---|
| `theta` | `[nadmix]` the best restart's mixture weights. |
| `theta_lo`, `theta_hi` | Per-weight min/max across restarts (the bracket). |
| `edge_length` | `[nedge_norm]` fitted drift edge lengths at the optimum. |
| `f3_fit` | `[npair]` the fitted f3 values. |
| `score` | The best (minimum) fit score. |
| `restart_spread` | Max-minus-min score across restarts (a convergence witness). |
| `status` | Domain outcome as a value. |

### `QpGraphFleetBatch` — best scores for many topologies at once

The generalization to a whole batch of candidate topologies fit in one launch; the
host does the global-best selection over these small arrays.

| Field | Meaning |
|---|---|
| `best_score` | `[G]` best score per topology. |
| `restart_spread` | `[G]` per-topology restart spread. |
| `status` | Domain outcome as a value. |

---

## 11. f-stat sweep types and the filter-mode constants

When a user sweeps over every combination of populations (say every group of 4 out
of 500), the backend enumerates, computes, filters, and compacts every combination
*on the device* and returns only the survivors — the full table is never brought
to the host.

### The two filter-mode constants

Named here so the host setter and the device reader cite the same names instead of
bare `0`/`1` literals. These are behavior-neutral mode tags, not frozen
reproducibility values.

| Constant | Value | Meaning |
|---|---|---|
| `kSweepFilterMinZ` | `0` | Fixed-threshold filter: keep items whose |z| is at least `min_z`. |
| `kSweepFilterTopK` | `1` | Rising-threshold top-K reservoir: keep the K most significant, with the threshold rising toward the K-th |z|. |

### `SweepConfig` — the request

| Field | Default | Meaning |
|---|---|---|
| `k` | `4` | Item arity (4 for quartets, 3 for triples). |
| `filter_mode` | `kSweepFilterMinZ` | Which of the two filter modes above. |
| `min_z` | `3.0` | The |z| threshold, and the floor for the rising-threshold mode. |
| `top_k` | `1,000,000` | The device reservoir cap — an upper bound on how many rows can come back. |
| `pop_subset` | empty | Optional subset of population indices; empty means all. |
| `sure` | `false` | Lift the maximum-combinations safety cap. |

Either filter mode returns at most `top_k` rows sorted by descending |z|,
independent of how many billions of combinations were computed — so the host can
never be flooded no matter the sweep size. In the fixed-threshold mode the device
reservoir still caps to `top_k` as a hard safety ceiling.

### `SweepSurvivors` — the result

Parallel arrays, one slot per survivor. Native double precision.

| Field | Meaning |
|---|---|
| `keys` | Survivor index tuples (up to 4 population indices; unused slots are 0). |
| `est` | Point estimate per survivor. |
| `se` | Diagonal jackknife standard error per survivor. |
| `z` | z-score per survivor. |
| `enumerated` | Total combinations the sweep would enumerate (echoed even when capped). |
| `capped` | Whether the safety cap refused the sweep (in which case no compute ran). |
| `status` | `Ok` unless capped. |

---

## 12. DATES types

These carry the results of the admixture-dating engine, which reduces an enormous
implicit object (all SNP pairs) to a small set of correlation moments using an FFT
trick, so the pairwise object is never actually formed.

### `DatesMoments` — the weighted-LD correlation moments

The per-(chromosome, distance-bin) sufficient statistics summed over every admixed
sample. Each moment is row-major `[n_chrom × n_bin]`. Chromosomes are indexed in
the same order as the caller's present-chromosome list; each bin spans a fixed
genetic distance.

| Field | Meaning |
|---|---|
| `n_chrom`, `n_bin` | Grid dimensions. |
| `s0` | Pair-count autocorrelation (the denominator). |
| `s1`, `s2` | Count-times-signal cross terms. |
| `s11`, `s22` | Signal-squared variance moments. |
| `s12` | Signal autocorrelation (the covariance numerator). |
| `status` | Domain outcome as a value. |

### `DatesExpFit` — one exponential-decay fit

The result of fitting a single decaying exponential to one windowed correlation
curve; the fit is run in a batch (the full-data fit plus one leave-one-chromosome
fit per chromosome).

| Field | Meaning |
|---|---|
| `date_gen` | The decay rate, in generations. |
| `error_sd` | Residual standard deviation of the fit. |
| `ok` | 1 only if a genuine decaying positive exponential was fit. |

---

## 13. Backend capability probe

### `BackendCapabilities`

A plain-data snapshot describing the one GPU a backend is bound to, plus whether it
can talk directly to the other visible GPUs. This is a *probe input* to an
out-of-band tag recorded elsewhere in the run record — it is deliberately never
attached to the numeric f2 result, so the numeric payload stays pure and a
bit-for-bit parity comparison sees only bits the math produced.

Every field is parity-neutral: each one drives a data-movement or observability
choice (which transport to combine partial results, which precision lane is
allowed), never the arithmetic. All fields value-initialize to a zero/false
"nothing probed yet / unknown" state, which is exactly what the base
`capabilities()` returns.

| Field | Meaning |
|---|---|
| `device_count` | Number of CUDA devices visible to the process (0 = unknown / no CUDA). |
| `compute_major`, `compute_minor` | Compute capability of the bound device ({0,0} = unknown). Observability, not a dispatch key — one build serves the target hardware. |
| `total_vram_bytes`, `free_vram_bytes` | Total and currently-free GPU memory on the bound device (0 = unknown). |
| `can_access_peer` | Whether the bound device can directly peer-access the other visible devices. |
| `emulated_fp64_honorable` | Whether emulated double precision is genuinely available on this build. |

`can_access_peer` is the switch between the fast device-to-device combine and the
host-staged fallback. It is true on the capable datacenter-class hardware and
expected to be false on consumer cards (where peer access is driver-disabled) —
false triggers a safe, tagged fall-back to the host-staged path, never a fault.
Both combine paths sum partial results in the same fixed device order, so the
reported numbers are bit-identical either way.

`emulated_fp64_honorable` being false means an emulated-double request must
degrade to native double (with a logged tag) rather than silently running a
different, rejected mode under the emulated label.

---

## 14. The `ComputeBackend` methods

Every compute operation is a method on `ComputeBackend`. They are grouped below by
what they do. Recall the conventions from section 2: unless noted, the base body
throws a sentinel for GPU-only methods, `CudaBackend` provides the batched device
implementation, and `CpuBackend` provides the native-double reference. Every
method that takes a `Precision` follows the policy in section 3.

### f2 computation

| Method | What it does |
|---|---|
| `compute_f2` | Compute the bias-corrected f2 matrix and pairwise-valid counts from the allele-frequency contract for one SNP block. Pure virtual — both backends implement it. |
| `compute_f2_blocks` | Compute the per-block f2 tensor over all SNPs plus retained pairwise counts, given a per-SNP block assignment. Pure virtual. |
| `compute_f2_blocks_device` | Same computation, but leave the result resident in GPU memory and hand back an opaque handle — the primary output, no forced copy to host. GPU-only. |
| `compute_f2_blocks_resident` | Like the device variant, tagged with a global block offset, for the direct device-to-device combine. GPU-only. |
| `compute_f2_blocks_into` | Copy each device's slice of the result directly into a shared pinned host buffer at its disjoint block offset — for concurrent race-free multi-GPU output. GPU-only. |
| `compute_f2_blocks_streamed` | Spill the per-block result block-by-block to host RAM or disk for out-of-core problems. Takes an optional trailing `redecode` descriptor (a `steppe::device::RedecodeSource*`, default null): when set, the streamed engine fills each per-chunk SNP-tile by re-decoding it on the device instead of copying from the dense host `Q`/`V`/`N` views (which are left null on that path, so the full `[P × M]` host input is never materialized); the null default preserves every existing caller, which keeps feeding a dense host buffer unchanged. Only the per-chunk tile *source* differs — the f2 GEMM, feeder, gather, assemble, ring, and spill are byte-for-byte the same — so the re-decode output is bit-identical to the dense-fed path. GPU-only. |

The device, resident, host-staged, and streamed variants all produce
bit-identical per-block numbers to `compute_f2_blocks`; the only difference is
*when and where* a result slab lands, never its bits.

### Genotype decode

| Method | What it does |
|---|---|
| `decode_af` | Decode a packed tile into the allele-frequency contract. Pure virtual — both backends implement it. |
| `detect_sample_ploidy_device` | Derive the per-sample ploidy vector for a tile. Base body is a real host implementation (the oracle); the GPU version runs a device prepass. |
| `transpose_to_canonical` | Turn a SNP-major source into the canonical individual-major tile — gather, encode, and re-pack in one pass. Base body is a real host implementation (the oracle); the GPU version is a device kernel. |
| `decode_af_compact_autosome` | Decode and, in the same pass, keep only autosome SNPs and stream-compact the result — leaving it resident, so the large copy back to host is gone. GPU-only. |
| `decode_af_compact_filter` | Decode and apply the full read-filter (allele class, minor-allele frequency, missingness, monomorphic, transversion, autosome, population coverage), resident and compacted. Uploads only a SNP-tile at a time so peak GPU memory stays bounded. GPU-only. |

Both compaction methods are integer-exact and preserve file order, so the kept set
and everything derived from it reproduce the host filter path exactly.

### f4 / f3 assembly

`assemble_f4` builds the per-block f4 matrix from resident f2 for one
target/reference choice (an `nl × nr` grid). `assemble_f4_quartets` and
`assemble_f3_triples` are the standalone forms whose columns are independent input
quartets or triples — the same four-slab (or three-slab) identity specialized to a
single tuple per column, adding no new math. Each comes in two overloads: one
reading device-resident f2 (overridden by `CudaBackend`) and one reading a host
tensor (overridden by `CpuBackend`, the oracle door). All are native double
precision.

### Jackknife, covariance, rank, and weights

| Method | What it does |
|---|---|
| `jackknife_cov` | The weighted block-jackknife covariance and the inverse of the fudged covariance. Weighted by per-block SNP count. Native double. |
| `jackknife_diag` | Just the diagonal variance, without ever forming the dense covariance — the production-scale shape that avoids the out-of-memory blow-up. Native double. |
| `rank_sweep` | The rank test over all candidate ranks, the nested rankdrop table, and the chosen rank. Native double. |
| `provides_rank_sweep` | Capability query: does this backend really implement `rank_sweep`? |
| `gls_weights` | The generalized-least-squares admixture weights via alternating refinement, then the constrained solve, then normalization. Native double. |
| `gls_weights_loo_batched` | All leave-one-block-out weight re-fits as one batched device solve (reusing the same inverse, for parity[^at2]). Native double. |
| `se_from_wmat` | The full leave-one-block-out standard-error reduction folded into one call so it stays on the device that produced the replicate matrix. Native double. |

### Model search

| Method | What it does |
|---|---|
| `fit_models_batched` | Fit many same-shape models against one resident f2 in a single batched device dispatch — the rotation primitive for large searches. Per-model domain failures are recorded in each result's status, never thrown, so a search of thousands does not abort on one degenerate model. Its base body delegates to the per-model default (see section 15); `CudaBackend` overrides it with the genuinely batched path. |
| `provides_batched_fit` | Capability query: does this backend implement the batched fit? |
| `batched_dispatch_count` | Observability counter: how many batched dispatches were issued (one per same-shape bucket, not one per model), used to prove the search really ran batched. |
| `set_solve_precision` | The opt-in promotion knob from section 3. Base is a no-op. |

### Ratio statistics (f4-ratio and D)

| Method | What it does |
|---|---|
| `dstat_block_reduce` | The per-(block, quadruple) numerator/denominator/count reduction for the normalized D statistic. Two overloads: host pointers, and reading resident decoded data. |
| `ratio_block_jackknife` | The one shared block-jackknife engine for both f4-ratio and D, batched over items, parameterized by weight, survivor mask, and centering mode. Native double. |
| `f4ratio_blocks_jackknife` | Fused assemble-then-ratio-jackknife for f4-ratio without copying the intermediate arrays to host. Two overloads (resident f2 for the GPU, host tensor for the oracle). |
| `dstat_blocks_jackknife` | Fused reduce-then-ratio-jackknife for the D statistic without copying the intermediates. Two overloads (resident decode result, and host pointers). |

### qpfstats

| Method | What it does |
|---|---|
| `qpfstats_smooth` | The shared-factor batched least-squares smoothing solve (section 9). |
| `qpfstats_blocks_smooth` | The fully fused reduce → block-jackknife → smooth → recenter path in one call that keeps everything resident, so only the small coefficient arrays come back. Two overloads (host pointers, and resident decode result). The jackknives and solve are native double; only the matrix-multiply sub-steps follow the precision argument. |

### qpGraph

| Method | What it does |
|---|---|
| `qpgraph_fit_fleet` | Run many projected-Newton restarts for one topology in a single launch — each restart is one device thread running its whole loop in-kernel, so there is no host objective evaluation per iteration. Returns only the best score, weights, and edges. |
| `qpgraph_fit_fleet_batch` | The generalization to a whole batch of candidate topologies fit in one launch, reading the same resident basis. |

### DATES

| Method | What it does |
|---|---|
| `dates_curve` | The FFT-based weighted-LD correlation engine, reducing the whole per-sample pipeline to the small correlation moments without ever materializing the pairwise object. The FFT and the weight/residual accumulation are native double. |
| `dates_repack` | Repack target genotypes onto the kept SNP axis — a bit-exact gather. Base body is a real host implementation (delegated into the core library so the header stays I/O-free). |
| `dates_fit` | The batched exponential-decay fit over the windowed curves. Base body is a real host implementation. The inner normal-equation accumulators are native double. |

### Lifecycle and probe

`ComputeBackend` has a virtual destructor and is non-copyable/non-movable. The
constructor takes no arguments. `capabilities()` returns the capability probe from
section 13; its base returns the all-unknown default so the CPU backend need not
override it.

---

## 15. Host helper functions

Two free functions live alongside the interface, both CUDA-free and declared here
so the host orchestrator can use them without naming the CUDA backend.

- **`core::qpadm::fit_models_batched_default`** — the per-model default body of
  the batched fit. It drives each model's assemble-then-fit chain through the
  backend's *own* device methods, so the GPU backend runs resident batched kernels
  and the CPU backend runs the scalar oracle, with no per-backend override needed.
  It is defined in the core library and only pulled in when the search actually
  runs, so the header does not force a dependency on the core qpAdm code.

- **`core::qpadm::model_in_small_path`** — the host gate that decides whether a
  model fits the size envelope the batched device path supports (bounded numbers
  of left sources, right populations, and rank). The search orchestrator uses it
  to split the model list: models that fit route to the batched device method,
  while the larger tail routes one dispatch per model. It delegates to the same
  single source-of-truth predicate the device path uses to size its per-thread
  arrays, so this gate can never drift wider than those arrays (a wider gate would
  overflow them).

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
