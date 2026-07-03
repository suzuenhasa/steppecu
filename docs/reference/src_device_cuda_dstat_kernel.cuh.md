# `dstat_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/dstat_kernel.cuh` declares exactly one function,
`launch_dstat_block_reduce`. It is the host-callable entry point for the GPU
stage that finishes a D-statistic (also written as qpDstat, or the ABBA-BABA
test). For every population quadruple and every genome block, that stage
produces the three running sums a later step turns into a D value and its
jackknife error bars.

The D-statistic is computed in two halves. The first half decodes genotypes into
allele frequencies; this function is the back end of the second half — the
per-SNP reduction that walks those frequencies and accumulates the sums. The
public, GPU-free contract for the whole computation lives in
`include/steppe/dstat.hpp`; the header here is one step below that, on the GPU
side.

This file contains no kernel code. The actual GPU kernel — its body and its
`<<<...>>>` launch — lives in the matching `.cu` file. Host orchestration code
calls the wrapper declared here and never sees the raw launch syntax.

Because the declaration names a CUDA type (`cudaStream_t`) and includes
`<cuda_runtime.h>`, this header is internal to the device library. It is the
private seam between the backend and the D-statistic kernel, not part of steppe's
CUDA-free public API. Section 7 explains that boundary.

---

## 2. What the kernel computes

A D-statistic compares four populations, labelled p1, p2, p3, and p4. At each SNP
the four populations each have an allele frequency; call them `a`, `b`, `c`, and
`d` respectively. The kernel forms three per-SNP quantities and adds them into
running totals:

| Per-SNP term | Formula |
|---|---|
| numerator contribution | `(a - b) * (c - d)` |
| denominator contribution | `(a + b - 2ab) * (c + d - 2cd)` |
| count | `1` |

Summed over the SNPs in a block, these become `numsum`, `densum`, and `cnt`. A
later stage divides the numerator total by the denominator total to get the
normalized D value for that block, and uses the per-block totals to compute a
block-jackknife standard error. The denominator is a heterozygosity-style
normalizer that keeps D within the range −1 to +1. This matches ADMIXTOOLS 2's
normalized-D definition.

All three sums accumulate in native double precision. The numerator and
denominator involve differences of nearly equal frequencies (`a - b`, `c - d`),
which lose accuracy if computed in a lower-precision or emulated format. This
reduction therefore deliberately stays in true double precision, rather than the
faster emulated arithmetic steppe uses elsewhere for large matrix multiplies.

---

## 3. How the work is split across GPU threads

The output is a grid of cells, one cell per (quadruple, block) pair. Each GPU
thread owns exactly one cell: quadruple `k`, block `b`. That thread walks the
block's SNP columns from `block_begin[b]` up to
`block_begin[b] + block_size[b]`, and for each qualifying SNP it reads the four
frequencies, forms the three per-SNP terms above, and accumulates them. When it
finishes the block it writes `numsum`, `densum`, and `cnt` for that one cell.

The SNP columns of a block are contiguous and in file order, so the walk is a
simple sequential scan with no gaps.

---

## 4. Which SNPs count: the finiteness mask

Not every SNP has a defined allele frequency in every population — a population
may have no genotype calls at a given SNP. Alongside the frequency array `Q`, the
decode step produces a companion validity array `V` that is `1` where the
frequency is defined and `0` where it is missing.

A SNP contributes to a cell only when `V == 1` for all four of p1..p4 at that
SNP. Any SNP missing in even one of the four populations is skipped for that
quadruple. This decision is made independently for each (block, quadruple) pair
rather than by pre-filtering to a single shared SNP set — so two different
quadruples can legitimately use different SNPs within the same block. This
reproduces ADMIXTOOLS 2's behavior of using every SNP that is present in all the
populations of a given comparison (its "all SNPs" mode).

---

## 5. Memory layouts and index formulas

Getting the layouts right is essential, because the input arrays and the output
arrays use different orderings.

- `Q` and `V` are **column-major** with shape [P × M] (P populations by M SNPs).
  Element (population `i`, SNP `s`) is at index `i + P*s`. All of one SNP's
  population values are adjacent in memory.
- The quadruple table `d_quad` is [4 × N] flattened: the four population indices
  for quadruple `k` are at `4*k`, `4*k+1`, `4*k+2`, and `4*k+3`. These are indices
  along the population axis (`0 .. P-1`).
- The outputs `d_numsum`, `d_densum`, and `d_cnt` are **row-major** with shape
  [N × n_block]. Cell (quadruple `k`, block `b`) is at `k*n_block + b`.

Note the asymmetry: the inputs are column-major, but the outputs are row-major.
The block boundaries are described by two parallel arrays — `d_block_begin` gives
each block's first SNP column, and `d_block_size` gives its SNP count. The block
sizes sum to M, and the blocks tile the SNPs contiguously in file order.

---

## 6. The `launch_dstat_block_reduce` parameters

| Parameter | Type | Meaning |
|---|---|---|
| `d_Q` | `const double*` | [P × M] column-major allele-frequency array from the decode step; element (i, s) at `i + P*s`. |
| `d_V` | `const double*` | [P × M] column-major validity mask paired with `d_Q`; `1` where the frequency is defined, `0` where missing. |
| `P` | `int` | Number of populations (the rows of Q/V), and the stride between consecutive SNP columns. |
| `M` | `long` | Number of SNPs (the columns of Q/V). Typed as `long` because a dataset can hold millions of SNPs — more than the `int` counts elsewhere need to represent. |
| `d_quad` | `const int*` | [4 × N] flattened table of population-index quadruples; quadruple `k` occupies `4*k .. 4*k+3`. |
| `N` | `int` | Number of quadruples. |
| `d_block_begin` | `const int*` | [n_block] the first SNP-column index of each block. |
| `d_block_size` | `const int*` | [n_block] the SNP count of each block; the sizes sum to M, and blocks are contiguous in file order. |
| `n_block` | `int` | Number of blocks. |
| `d_numsum` | `double*` | [N × n_block] row-major output: per-cell numerator sum. Cell (k, b) at `k*n_block + b`. |
| `d_densum` | `double*` | [N × n_block] row-major output: per-cell denominator sum. |
| `d_cnt` | `double*` | [N × n_block] row-major output: per-cell count of contributing SNPs. |
| `stream` | `cudaStream_t` | The CUDA stream the launch is queued on. |

All pointer arguments refer to memory that already lives on the GPU; the wrapper
launches the kernel and does not allocate or copy anything itself.

---

## 7. Why this header is device-private

The declaration takes a `cudaStream_t`, and the file includes
`<cuda_runtime.h>`. Any header that mentions a CUDA type can only be included by
code that is already being compiled with the CUDA toolchain. That places this
header on the device library's internal surface, not on the public API.

steppe keeps a firm line between two kinds of header:

- **Public, CUDA-free headers** that any caller can include. The D-statistic's
  public contract sits here, in `include/steppe/dstat.hpp`, and pulls in no GPU
  dependency.
- **Device-internal headers** like this one, which sit behind that line and are
  used only by the backend and the kernel's translation unit.

Keeping the kernel-launch declaration here — rather than in the public header —
is exactly what lets the public API stay free of any CUDA dependency. A caller
who only wants to request a D-statistic never has to pull in `<cuda_runtime.h>`.
