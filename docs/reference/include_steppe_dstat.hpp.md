# `dstat.hpp` reference

## 1. Purpose

`include/steppe/dstat.hpp` is the public entry point for computing a
**normalized D statistic directly from genotype files**. It declares one result
struct (`DstatResult`) and one function (`run_dstat`).

There are two ways to get a D-type statistic out of steppe. One reads a
precomputed "f2" cache and reports the closely related **f4** number. The other —
the one this header exposes — reads the raw genotype triple (the `.geno`, `.snp`,
and `.ind` files) directly and reports the **normalized D**. `run_dstat` shares
the same decode front-end as the f2-extraction path (the file reader, the
per-SNP allele-frequency computation, the assignment of SNPs to jackknife blocks,
and the tile-by-tile streaming of SNPs), but then branches off into its own
per-SNP D kernel. It never touches the f2 cache at all.

The header is deliberately free of any CUDA code (see section 4), so it can be
included by the command-line tool and the language bindings without dragging in
the GPU layer.

---

## 2. The D statistic it computes

`run_dstat` computes the genotype-path D statistic[^at2] (the `qpdstat_geno`
behavior in its "all SNPs" mode, reporting D rather than f4). The output
has been pinned empirically against the reference values, agreeing
to roughly 15 significant digits on the per-block components.

### The per-SNP formula

For a quadruple of populations `(p1, p2, p3, p4)`, let `a`, `b`, `c`, `d` be the
reference-allele frequencies of those four populations at a single SNP. Two
quantities are formed at that SNP:

- **numerator** = `(a - b) * (c - d)`
- **denominator** = `(a + b - 2ab) * (c + d - 2cd)`

The two denominator factors are heterozygosity-style normalization terms; they
are what makes this the *normalized* D rather than the raw f4.

### Accumulation per block

SNPs are partitioned into jackknife blocks along the genome. Within each block,
the numerator and denominator are summed **only over the SNPs that are valid in
all four populations at once** (a SNP is skipped for a given quadruple if any of
the four populations has no data there). Each block then keeps the two means:

- `est_num[block] = (sum of numerator over valid SNPs) / (count of valid SNPs)`
- `est_den[block] = (sum of denominator over valid SNPs) / (count of valid SNPs)`

Note that the count of valid SNPs is per-block **and** per-quadruple — different
quadruples can survive at different SNPs, so they get different counts and
weights.

### From blocks to the reported numbers

The final D is `est_num / est_den`, computed as a **jackknife of the ratio**
across the blocks (the same shape as the f4-ratio jackknife, but with the
per-block, per-quadruple counts as weights and with no missing-data substitution
step). From that jackknife:

- `est` = the normalized D estimate.
- `se`  = the square root of the jackknife variance.
- `z`   = `est / se`.
- `p`   = a two-sided p-value: twice the upper tail of the standard normal at
  `|z|`.

The normalized D typically lands in a range of about ±0.06 — roughly ten times
larger than the f4 value from the f2 cache path — but the two paths produce
z-scores that track each other closely.

---

## 3. Parity pins

Three choices are frozen so that `run_dstat` matches the reference
exactly[^at2]. These were verified against the reference on real data.

### Allele frequency: forced diploid

For parity, each population's reference-allele frequency is the plain
ratio (reference-allele count) / (allele count) / 2, with **no** pseudo-haploid
adjustment[^at2]. To match that, `run_dstat` forces the diploid ploidy mode. It does
**not** use the automatic per-sample ploidy detection that the f2-extraction
path uses — that automatic detection can flip the sign of a near-zero D, which
would break parity.

### Blocks: identical to the reference

The way SNPs are assigned to jackknife blocks matches the reference
block-length routine byte-for-byte[^at2].

### SNP mask: all SNPs, per quadruple

In "all SNPs" mode the only per-SNP test is whether a SNP is valid (finite) for
a given block and quadruple. There is **no** maximum-missingness filter, **no**
minor-allele-frequency filter, and **no** dropping of monomorphic SNPs.
Autosomes-only filtering **is** on, which matches the parity default of
keeping only the autosomes[^at2].

---

## 4. The CUDA-free contract

This header contains only standard C++ — no CUDA. That is a deliberate contract,
and it is what lets the command-line app and the bindings include it freely.

The GPU still does the heavy lifting: the per-SNP D reduction runs on the GPU,
but it is reached only through a compute-backend seam. The actual GPU kernel
lives in a private CUDA source file inside the device library, not here. To keep
this header CUDA-free while still referring to GPU resources, the
`device::Resources` type is only **forward-declared** here; the implementation
`.cpp` file is what includes the real device header. Apps and bindings reach the
GPU exclusively by calling `run_dstat` — they never include CUDA themselves.

---

## 5. DstatResult

`DstatResult` is a table of results held as parallel arrays — one slot per input
quadruple, kept in the same order the quadruples were passed in. Reading across
the arrays at a given index gives you the full result for that one quadruple.

The struct is intentionally shaped to mirror the f4 result struct (the same
`p1..p4, est, se, z, p` columns), so that the existing f4 output emitter and the
existing binding marshalling code are reused verbatim. The `est`, `se`, `z`, and
`p` values already follow the reference D sign, Z, and p conventions[^at2].

| Field | Type | Meaning |
|---|---|---|
| `p1`, `p2`, `p3`, `p4` | `vector<int>` | The population-axis index of each population in the quadruple, echoed back so the emitter and bindings can label the rows. Each vector has length N (the number of quadruples). |
| `est` | `vector<double>` | The normalized D estimate (`est_num / est_den`) from the jackknife. |
| `se` | `vector<double>` | The standard error: the square root of the block-jackknife variance. |
| `z` | `vector<double>` | `est / se`. |
| `p` | `vector<double>` | The two-sided p-value: twice the upper tail of the standard normal at `|z|`. |
| `status` | `Status` | The per-call outcome; defaults to `Ok`. See below. |
| `precision_tag` | `Precision::Kind` | Which arithmetic produced the result; always `Fp64`. See below. |

### Degenerate quadruples never throw

A quadruple that has **no surviving blocks** (nothing was valid for it) is not an
error — it is recorded as a per-row NaN in that quadruple's slot, and the run
continues. A domain outcome like this is never turned into an exception; the
"record it and keep going" behavior is the contract. `status` stays `Ok` for a
populated result; the NaN sentinel rides on the individual degenerate row.

### Why `precision_tag` is always `Fp64`

The D reduction is numerically delicate (it involves cancellation-prone
differences), so it is always run in native double precision, accumulated on the
host in extended (long-double) precision. Because that carve-out always applies,
this tag is always `Fp64`.

---

## 6. run_dstat

```cpp
[[nodiscard]] DstatResult run_dstat(const std::string& geno,
                                    const std::string& snp,
                                    const std::string& ind,
                                    std::span<const std::string> pop_union,
                                    std::span<const std::array<int, 4>> quadruples,
                                    double blgsize_morgans,
                                    device::Resources& resources);
```

Computes the normalized D over the genotype triple and returns one row per
quadruple, in the input order.

| Parameter | Meaning |
|---|---|
| `geno`, `snp`, `ind` | Paths to the EIGENSTRAT/TGENO genotype triple. |
| `pop_union` | The set of population names referenced by the quadruples. Only these populations are decoded — not every individual in the file (a full file can hold tens of thousands). The **population axis** is exactly this set, sorted ascending by label. |
| `quadruples` | A span of `(p1, p2, p3, p4)` index tuples. Each index points into the sorted `pop_union` partition described above. The caller is responsible for resolving quadruple names against that same sorted order before calling. |
| `blgsize_morgans` | The jackknife block size, expressed in Morgans (the parity default is 0.05[^at2]). |
| `resources` | The GPU resources. The per-SNP D reduction is routed through the first GPU's backend (`resources.gpus[0].backend`), GPU-first and device-resident. |

Behavior that is fixed for this call, as covered in section 3: ploidy is forced
diploid, the "all SNPs" mask is used (no max-missingness, no minor-allele-
frequency, no monomorphic-dropping filters), and autosomes-only is on.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
