# `qpfstats.hpp` reference

## 1. Purpose

`include/steppe/qpfstats.hpp` is the public entry point for **qpfstats** — the
"joint f-statistic smoother." It declares one function, `run_qpfstats`, plus the
struct that holds its result, `QpfstatsResult`.

qpfstats takes a genotype dataset and a list of populations, computes every basic
f-statistic (f2, f3, f4) you can form from those populations, and then fits all of
them jointly so that they are mutually consistent. The output is a *smoothed* table
of per-block f2 values — the same shape and file format that steppe's f2 cache
uses — so downstream tools (qpAdm, f4, qpGraph) can consume the smoothed numbers
exactly as if they had read a normal f2 cache. The output is **not** a table of
estimate / standard-error / z-score / p-value rows; it is a tensor of f2 values.

The header is deliberately free of any CUDA/GPU code. It uses only the C++ standard
library plus a few steppe headers, and it only *forward-declares* the GPU resource
type. That keeps it lightweight enough to be included by the public API, the
command-line tool, and the language bindings without forcing any of them to pull in
the GPU layer. The actual per-SNP math and the smoothing solve run on the GPU
behind a single seam (described in section 6).

---

## 2. How qpfstats differs from the f4 / D paths

steppe has two families of f-statistic entry points, and it helps to know where
qpfstats sits.

- The **f4 / D** entry points (for example `run_f4`, or `run_qpdstat` reading an
  `--f2-dir`) read an already-computed f2 cache and report f4 or D statistics with
  their standard errors and significance. They consume a cache and emit a *report*.

- **qpfstats** reads the raw genotype triple (`.geno` / `.snp` / `.ind`) directly.
  It runs the SNPs through the *same* front-end as the f2-extraction and D-statistic
  paths — the file reader, the per-SNP allele-frequency decode (allele frequency,
  variance, and non-missing count per population), the block assignment from genetic
  position, and the tiled SNP streaming. It then drives the same genotype-f4
  numerator engine over the full set of f2/f3/f4 population combinations, fits them
  jointly, and emits a *smoothed f2 cache*.

So qpfstats is a **producer** of an f2 cache (like f2-extraction), not a consumer of
one (like the f4 report path). What makes it different from plain f2-extraction is
the joint fit: instead of computing each f2 in isolation, it regresses the whole
f2/f3/f4 family onto a common basis so the numbers are internally consistent, and
the resulting f2 tensor is the smoothed result of that fit.

This reproduces the `qpfstats` R ridge-regression path[^at2]. The
implementation is pinned to that reference so the numbers agree.

---

## 3. The smoothing algorithm

The computation follows the `qpfstats` smoothing in five steps[^at2]. Concrete counts
below are for a run over N = 9 populations.

### Step 1 — Enumerate the population combinations

Sort the N populations and enumerate every f2, f3, and f4 statistic they can form:

- **f2:** every unordered pair `(p1, p2)`, written as `(p1, p2, p1, p2)`. There are
  C(N, 2) of these — 36 for N = 9.
- **f3:** every triple, expanded three ways (the "3-rotation"), written as
  `(p1, p2, p1, p4)`. There are C(N, 3) × 3 of these — 252 for N = 9.
- **f4:** every group of four, expanded three ways, written as `(p1, p2, p3, p4)`.
  There are C(N, 4) × 3 of these — 378 for N = 9.

Total for N = 9: 36 + 252 + 378 = **666 combinations**.

### Step 2 — Compute the numerator per combination, per block

For each combination `(a, b, c, d)` and each jackknife block, average the product
`(a − b)(c − d)` of the four allele frequencies over the SNPs where **all four**
populations have data. This is exactly the genotype-f4 numerator that steppe's
D-statistic engine already computes; here only the numerator is kept (the
D-statistic denominator is ignored). The computation is batched over both axes at
once — the 666-combination axis and the block axis — and runs device-resident on
the GPU.

### Step 3 — Build the design matrix

Every combination maps to a row of coefficients over a fixed basis. The basis is the
set of off-diagonal population pairs `(i, j)` with `i < j` — there are C(N, 2) of
them (36 for N = 9), indexed as the row-major upper triangle. The coefficients are
the ±1/2 f4-identity coefficients (with a factor of 2 for a pure-f2 row, and an
overall divide-by-2). This is the "f-statistic vector-space basis": every f2, f3,
and f4 is expressed as a combination of these pairwise f2 building blocks.

### Step 4 — Solve the ridge regression

Assemble the numerator matrix (combinations × blocks) and a per-combination global
estimate obtained by a block jackknife over the numerators. Form the shared normal
matrix `A = xᵀx + ridge·I`, where **ridge = 1e-5** keeps the solve well-conditioned.

Then, for each block, solve `A · b = xᵀ · (that block's numerators)` for the pairwise
coefficients `b`. A block that is missing some combinations (their rows are NaN
because a population had no data in that block) is handled by *downdating* — the
missing rows' contribution is subtracted from `A` before the solve, so the block is
fit only from the combinations it actually has. A block with **no** usable
combinations yields `b = 0`. The same solve is run once globally to produce the
global coefficient vector.

### Step 5 — Scatter and recenter

Scatter each block's pairwise coefficients back into a symmetric P × P matrix (the
off-diagonal entries; the diagonal stays zero). Then recenter every block by
subtracting its own f2 estimate and adding back the global coefficients — the
recentering step[^at2]. The recentered, per-block, symmetric,
zero-diagonal tensor is the **smoothed f2 result**.

---

## 4. Parity pins

Several choices are fixed so that qpfstats reproduces the reference exactly[^at2]. These
are inherited from the D-statistic path and are proven against the reference.

- **Allele frequency.** Frequencies use the plain reference-count / allele-count / 2
  formula, forced to treat every sample as **diploid**. This is *not* the per-sample
  ploidy auto-detection that f2-extraction can do; it is the fixed diploid parity
  rule[^at2].
- **Jackknife blocks.** The block assignment is byte-for-byte identical to the
  reference block-length computation.
- **SNP mask.** The "all SNPs" rule applies per (combination, block): a SNP counts
  only where all four populations of that combination have data. There is **no**
  per-SNP missing-fraction filter, no minor-allele-frequency filter, and no
  drop-monomorphic filter. Autosomes-only filtering is **on**.

Every one of the three statistic families (f2, f3, f4) is always included — the full
basis, the parity default[^at2].

---

## 5. The output tensor and file format

The smoothed result is stored as steppe's per-block f2 tensor type
(`F2BlockTensor`), laid out as `f2[i + P·j + P·P·b]` for population indices `i`, `j`
and block index `b`, where `P` is the number of populations and there are `n_block`
blocks. Each block's slice is symmetric with a **zero diagonal** — the parity
convention[^at2], since the tensor is built purely from the off-diagonal pair basis.

The tensor drops straight into steppe's `write_f2_dir` writer, so the smoothed f2
can be saved to disk and later read by qpAdm, f4, or qpGraph like any ordinary
extracted f2 cache. To make that writer's missing-block detection work, the tensor's
per-pair "variance" slot is filled with each block's kept-SNP count replicated
across all pairs (so a block that contributed shows up as nonzero), and the per-block
SNP counts are carried alongside.

---

## 6. The CUDA-free contract

This header is standard C++ only, by design. The heavy work — the per-SNP numerator
reduction and the smoothing solve — runs on the GPU, but it is reached through a
single abstraction seam (the compute-backend held inside `device::Resources`). The
GPU kernels live in private `.cu` files compiled into the device library; this
header never sees them.

To keep the header CUDA-free, `device::Resources` is only **forward-declared** here
(the real definition lives in the device layer, and the `.cpp` implementation
includes it). The application and the language bindings reach the GPU only through
this seam, the same way the D-statistic entry point does. The final output is plain
host-side double-precision storage — no GPU handle leaks out.

If the smoothing solve hits a structural problem (for example a normal matrix that
turns out not to be invertible — which should not happen with a positive ridge
term), that is reported as a status **value** on the result, not thrown as an
exception. Genuine I/O faults (a missing or unreadable file) do throw and propagate
out.

---

## 7. `QpfstatsResult`

`QpfstatsResult` is the value returned by one `run_qpfstats` call. It bundles the
smoothed tensor, its labels, an outcome status, and a record of which arithmetic was
used.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `f2` | `F2BlockTensor` | — | The smoothed per-block f2 tensor (layout and conventions described in section 5). Its per-pair variance slot carries the replicated per-block kept-SNP counts, and its block-sizes carry the per-block SNP counts. |
| `pop_labels` | `vector<string>` | — | The population labels in index order along the P axis — sorted ascending, which is the order used for the tensor's dimension names[^at2]. |
| `status` | `Status` | `Status::Ok` | The per-call outcome. `Ok` means the result is populated. A structural domain problem (such as a non-invertible factor) is reported here as a value rather than thrown. |
| `precision_tag` | `Precision::Kind` | `Fp64` | Which arithmetic actually drove the smoothing solve, recorded for provenance and the run's metadata. By the landed precision policy the solve's Cholesky factor and triangular solve run in native double precision, while the matrix-multiply sub-steps run emulated-FP64; this tag records the engaged mode. |

---

## 8. `run_qpfstats`

`run_qpfstats` is the single entry point. It reads the genotype triple, runs the
full smoothing computation, and returns a `QpfstatsResult`. It is marked
`[[nodiscard]]` so the result cannot be accidentally ignored.

```cpp
QpfstatsResult run_qpfstats(const std::string& geno,
                            const std::string& snp,
                            const std::string& ind,
                            std::span<const std::string> pops,
                            double blgsize_morgans,
                            const Precision& precision,
                            device::Resources& resources);
```

| Parameter | Meaning |
|---|---|
| `geno`, `snp`, `ind` | Paths to the EIGENSTRAT / TGENO genotype triple (the genotype matrix, the SNP list, and the individual list). |
| `pops` | The set of population names to smooth over. The function **sorts them ascending** (the parity dimension-name order[^at2]) and reads **only** these populations from the dataset — not the whole file. |
| `blgsize_morgans` | The jackknife block size, in **Morgans**. The parity default is 0.05 Morgans (5 centimorgans)[^at2]. |
| `precision` | Governs **only** the matrix-multiply sub-steps of the solve (the `xᵀx` and `xᵀ·numerator` products). The Cholesky factor and the triangular solve stay in native double precision regardless — the numerically delicate carve-out. |
| `resources` | The GPU resources. The per-SNP numerator reduction and the smoothing solve are routed through this handle's first GPU backend, device-resident. |

Fixed behaviors, not parameters: ploidy is forced diploid, the "all SNPs" mask is
used, autosomes-only is on, and all of f2 / f3 / f4 are included (the full basis).
An I/O fault propagates out as an exception rather than being folded into the result
status.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
