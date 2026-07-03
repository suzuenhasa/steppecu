# `decode_compact_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/decode_compact_kernel.cuh` declares the GPU launch wrappers for
the part of steppe that decides, entirely on the GPU, **which SNPs to keep** and
then **compacts the kept ones into a dense array** — without ever copying the data
back to the host to make that decision.

Older code did this filtering on the CPU: it decoded genotypes, walked the SNP
list, and pushed the ones that passed the filters into a growing list. That meant
moving a lot of data off the GPU just to throw most of it away. This header is the
device-resident replacement. The genotypes stay in GPU memory, one GPU thread per
SNP decides keep-or-drop, and a second pass copies the survivors into a tightly
packed result — all on the GPU.

The header declares only the launch wrappers (the C++ functions the rest of the
library calls). The actual GPU kernel bodies and their `<<< >>>` launch syntax live
in the matching `.cu` file. This split keeps the GPU launch mechanics out of the
header so that callers can include it without pulling in the kernel source.

The names in this header take a `cudaStream_t`, so the header is private to the GPU
backend — it is not part of steppe's public API and is not meant to be included by
the command-line tool, the language bindings, or the core library.

The correctness bar for everything here is exact reproduction of the old CPU path:
the set of SNPs the GPU keeps, and the order it keeps them in, must be
**bit-for-bit identical** to what the CPU loop would have produced. Several of the
comments below spell out exactly why each function achieves that.

---

## 2. The keep-and-compact pipeline

All three functions are pieces of one three-step pipeline. Understanding the
pipeline first makes each function easy to read.

**Step 1 — build a keep mask.** One of the two keep-mask functions runs one GPU
thread per SNP. Each thread evaluates the filter rules for its SNP and writes a
single byte: `1` to keep, `0` to drop. The output is an array of `M` bytes (one per
SNP), called the *keep flags*. A byte rather than a bit is used because it is what
the downstream compaction primitives expect, and its value can be read directly as
a boolean.

**Step 2 — turn flags into destination positions.** For the small one-dimensional
per-SNP arrays (chromosome number, genetic position), steppe hands the flags to a
standard GPU library primitive, CUB's `DeviceSelect::Flagged`, which compacts the
flagged elements into a dense output in one call. For the large two-dimensional
`P × M` tensors, that 1-D primitive does not apply, so steppe instead computes an
**exclusive prefix sum** of the keep flags. The exclusive prefix sum of SNP `s` is
simply *the number of kept SNPs strictly before `s`*. That number is exactly the
column index the kept SNP will occupy in the packed output. So the prefix sum turns
"is this SNP kept?" into "where does it go?".

**Step 3 — gather.** The gather function copies each kept SNP's column from the
source tensor to its computed destination column in the packed output.

A property that matters throughout: **file order is preserved.** Because the prefix
sum only ever counts upward as SNP index increases, kept SNPs land in the packed
output in the same relative order they had in the input file. This is the same order
the old CPU loop produced (it appended survivors as it scanned the file top to
bottom). It matters because a later stage assigns SNPs to jackknife blocks based on
their position along the genome, and that block assignment would come out wrong if
the kept order were scrambled.

The two keep-mask functions correspond to two different filtering situations,
called *regime A* and *regime B*.

---

## 3. Autosome keep mask (regime A)

`launch_autosome_keep_mask` builds the keep mask for the simplest filter: keep a SNP
only if it sits on an autosome — a chromosome whose number falls within an inclusive
range (chromosomes 1 through 22 in the default configuration, dropping the sex
chromosomes and mitochondrial or other codes).

Each thread handles one SNP `s` and writes
`d_flags[s] = (d_chrom[s] >= chrom_min && d_chrom[s] <= chrom_max) ? 1 : 0`.

The key property is that this test is **integer-exact**. The chromosome number is an
integer read straight from the `.snp` file, and the comparison is a plain integer
range check — there is no floating-point arithmetic and therefore no possibility of
a rounding difference between the GPU and the CPU. As a result, the set of SNPs the
GPU keeps is guaranteed bit-identical to the CPU reference, whose loop is the
equivalent of "if the chromosome is below 1 or above 22, skip it."

| Parameter | Type | Meaning |
|---|---|---|
| `d_chrom` | `const int*` | Per-SNP chromosome number, in GPU memory, length `M`. |
| `M` | `long` | Number of SNPs. |
| `chrom_min` | `int` | Lowest chromosome counted as an autosome (1 by default). |
| `chrom_max` | `int` | Highest chromosome counted as an autosome (22 by default). |
| `d_flags` | `std::uint8_t*` | Output keep-flag buffer, length `M`; `1` = keep, `0` = drop. Fed to CUB's `DeviceSelect::Flagged`. |
| `stream` | `cudaStream_t` | GPU stream to launch on. |

---

## 4. Full extract_f2 keep mask (regime B)

`launch_regimeb_keep_mask` builds the keep mask for the full quality-control filter —
the complete set of rules that steppe's f2 extraction applies, reproduced exactly on
the GPU. This is the elaborate case; regime A is the trivial one. Again it runs one
thread per SNP, each writing a single keep flag, but each thread now evaluates the
whole filter chain.

The decision has three parts, and each is faithful to the CPU reference:

**Pooled minor-allele frequency.** The thread first computes a per-SNP pooled summary
by summing, across all `P` populations, the allele-frequency estimate weighted by the
non-missing count — the sum of `Q · N` over populations, together with the total of
`N`. This yields the frequency obtained by pooling all populations together (as
opposed to any single population's frequency). This accumulation is done sequentially
over populations `0` through `P-1`, using the exact same shared summary routine the
CPU path uses. Doing it in that fixed sequential order makes the result independent
of how the compiler chooses to fuse multiply-and-add operations, so the pooled value
is bit-identical to the CPU loop's value rather than merely close to it.

**The pooled keep decision.** The pooled summary is then run through the same shared
keep-decision logic the CPU uses. That logic drops a SNP if it is multiallelic, if it
is strand-ambiguous (subject to the configured strand policy), if its pooled
minor-allele frequency is below the configured minimum, if its missing-data fraction
is too high, if it is monomorphic (a pooled frequency of exactly `0.0`, compared
exactly, not within a tolerance), if it is a transition when only transversions are
wanted, or if it is off the autosomes. A membership test that the CPU f2 path treats
as always-true is likewise always-true here, so it never removes anything.

**Population-coverage cutoff (`maxmiss`).** Separately from everything above, the SNP
is dropped if the fraction of populations with no data at all (a non-missing count of
`0` or less) *strictly exceeds* the `maxmiss` threshold. Both the "no data" test
(count `<= 0.0`) and the strict "greater than" comparison against `maxmiss` match the
CPU reference precisely. This population-axis cutoff is a distinct filter from the
per-individual missing-data threshold carried in the config, and it is passed as its
own argument.

A configuration subtlety: the caller forces the config's per-individual
missing-fraction threshold to `1.0` (effectively off) before calling this function,
because the population-coverage cutoff above is the missingness rule that applies in
this path. The `maxmiss` argument is the one that actually filters on coverage.

| Parameter | Type | Meaning |
|---|---|---|
| `d_Q` | `const double*` | Column-major `P × M` tensor of per-population allele-frequency estimates. Element for population `i`, SNP `s` lives at index `i + P·s`. |
| `d_N` | `const double*` | Column-major `P × M` tensor of per-population non-missing counts, same layout. |
| `P` | `int` | Number of populations. |
| `M` | `long` | Number of SNPs. |
| `d_ref` | `const char*` | Per-SNP reference-allele character from the `.snp` file, length `M`. |
| `d_alt` | `const char*` | Per-SNP alternate-allele character from the `.snp` file, length `M`. |
| `d_chrom` | `const int*` | Per-SNP chromosome number, length `M`. |
| `cfg` | `const steppe::FilterConfig&` | The filter thresholds (minor-allele-frequency minimum, strand policy, transversions-only, and so on). The per-individual missing threshold is forced to `1.0` by the caller. |
| `ploidy` | `double` | Ploidy used in the frequency and coverage math. |
| `total_indiv` | `double` | Total number of individuals, used in the coverage math. |
| `maxmiss` | `double` | Population-coverage cutoff: drop a SNP if the fraction of populations with no data strictly exceeds this. |
| `d_flags` | `std::uint8_t*` | Output keep-flag buffer, length `M`; `1` = keep, `0` = drop. |
| `stream` | `cudaStream_t` | GPU stream to launch on. |

---

## 5. Column gather and compaction

`launch_compact_columns_gather` performs step 3 of the pipeline for the large
tensors. Given a column-major `P × M` tensor and a keep mask, it copies each kept
SNP's whole column into a densely packed `P × M_kept` output tensor, in file order.

The destination for each column comes from `d_keep_idx`, the **exclusive prefix sum**
of the keep flags. For a kept SNP `s`, `d_keep_idx[s]` is the number of kept SNPs
before it, which is precisely the column it should occupy in the packed output. The
kernel runs one thread per (population, source SNP) pair; a thread copies its element
only when that SNP's flag is set, writing
`d_out[i + P·d_keep_idx[s]] = d_in[i + P·s]`.

Because the prefix sum increases monotonically with `s`, the kept columns land in the
output in their original relative order — the same subset, in the same order, the CPU
loop would have produced by appending survivors as it scanned. That preserved order
is what lets the later block-assignment stage reproduce the CPU block numbering
exactly.

The tensors are stored **column-major**: the value for population `i` at SNP `s` is
at flat index `i + P·s`, so a single SNP's data (all `P` populations) is a contiguous
run of `P` doubles — one column. The gather copies these columns intact.

**The input and output buffers must not overlap in memory.** The kernel reads from
`d_in` and writes to `d_out` without any ordering guarantee between the two, so
aliasing them would corrupt the result.

| Parameter | Type | Meaning |
|---|---|---|
| `d_in` | `const double*` | Source column-major `P × M` tensor. |
| `P` | `int` | Number of populations (column height). |
| `M` | `long` | Number of SNPs (source column count). |
| `d_flags` | `const std::uint8_t*` | Keep flags, length `M`; selects which columns are copied. |
| `d_keep_idx` | `const long*` | Exclusive prefix sum of the keep flags, length `M`; gives each kept SNP its destination column. |
| `d_out` | `double*` | Packed output tensor, `P × M_kept`. Must not alias `d_in`. |
| `stream` | `cudaStream_t` | GPU stream to launch on. |
