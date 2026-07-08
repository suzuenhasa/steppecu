# `fst_allpairs_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/fst_allpairs_kernel.cuh` declares the two GPU launch wrappers
behind `steppe fst --all-pairs` — the mode that fills a whole `P × P` population
matrix of Weir-Cockerham FST values in one sweep, rather than one pair at a time.
It is a launch-wrapper header: both functions name `cudaStream_t` and pull in
`<cuda_runtime.h>`, so the header stays private to the `steppe_device` target.
The kernel bodies and their `<<<...>>>` launches live in the paired
`fst_allpairs_kernel.cu`; this header is what the CUDA backend `#include`s to
reach them.

The all-pairs path is built around one idea: **decode the panel once, reuse it
for every pair.** The naive way to fill a matrix would re-scan the genotypes for
each of the `C(P, 2)` pairs. Instead the first kernel walks the genotypes a
single time and boils each population down to its per-SNP *sufficient statistics*
`{n, ac, het}`; the second kernel then reads only those tiny stats and forms
every pair's WC numerator and denominator. The genotype panel is touched once no
matter how many populations you throw at it.

Two contracts hold the numbers steady. The WC variance-component arithmetic is
shared with the single-pair FST kernel and the CPU oracle through
`core/internal/wc_fst.hpp`, so a matrix cell can never drift away from what the
single-pair kernel would report for that same pair. And the pair enumeration uses
the same closed-form `k = 2` unrank as the rest of steppe's sweeps
(`sweep_unrank.cuh`), so pair index `r` maps to the same `(i, j)` everywhere.

---

## 2. The two-stage design

The two declared functions are the two stages, called in order per SNP-tile.

**`launch_fst_suffstat_decode`** — the decode. One thread per `(pop p, SNP
s_local)` over the current SNP-tile `[s_lo, s_lo + tm)`. Each thread folds its
population's diploid genotype codes into a `WcPerPop` accumulator (the same
per-pop fold the single-pair `fst_wc_kernel` does), but instead of *finalizing* an
FST value it **stores** the three sufficient statistics `{n, ac, het}` into three
`P × tm` tensors, laid out pop-major (`buf[p*tm + s]`). This is the single-pair
kernel's per-pop loop generalized to all `P` populations and captured rather than
consumed — decoded once per tile and then reused across every one of the
`C(P, 2)` pairs.

**`launch_fst_allpairs_accumulate`** — the accumulate. **One CUDA block per pair**
`r` in the chunk `[pair0, pair0 + C)`. Each block maps its `r` to the two tile
population indices `(i, j)` via the `O(1)` `readv2_unrank_pair`, then its threads
stride-share this tile's SNPs, running the **shared** `wc_finalize` over the two
populations' sufficient-stat slices. The block tree-reduces its partial
`Σnum / Σden / Σvalid` in shared memory, and thread 0 **adds** that block total
into the persistent per-pair accumulator. Because each block owns a distinct pair
`r`, no atomics are needed — every accumulator cell has a single writer block.

---

## 3. Why block-per-pair, and why it accumulates across tiles

Blocking over the SNP axis (one block per pair, threads split across SNPs) is a
deliberate change from a v1-style one-thread-per-pair layout. With few
populations there are few pairs, and one-thread-per-pair would leave the launch
almost empty — a handful of busy threads and an idle GPU. Spreading each pair's
SNP work across a whole block's threads keeps the launch occupied even at low
pair counts, which is exactly the regime a small `P × P` matrix lives in.

The accumulate wrapper is called **once per SNP-tile**, and it `+=` into the
persistent per-pair accumulators (`d_pair_num`, `d_pair_den`, `d_pair_cnt`). That
is how the sweep reduces across SNP-tiles: each tile's decode fills the small
`P × tm` stat tensors, each tile's accumulate adds that tile's contribution onto
the running per-pair totals, and after the last tile the accumulators hold the
full-panel WC sums for every pair. The host later divides `num / den` per pair and
scatters the results into the `P × P` matrix.

---

## 4. The two entry points

### `launch_fst_suffstat_decode`

| Parameter | Type | Meaning |
|---|---|---|
| `d_packed` | `const std::uint8_t*` | The packed genotype panel on the device. |
| `bytes_per_record` | `std::size_t` | Stride (in bytes) of one SNP record in `d_packed`. |
| `d_pop_offsets` | `const std::size_t*` | Per-population sample offsets defining each pop's slice of a record. |
| `P` | `int` | Number of populations (the rows/cols of the eventual matrix). |
| `s_lo` | `long` | First global SNP index of the current tile. |
| `tm` | `long` | SNP count in the current tile (the tile width). |
| `d_n` | `double*` | `[P × tm]` pop-major output: per-`(pop, SNP)` allele count `n`. |
| `d_ac` | `double*` | `[P × tm]` pop-major output: per-`(pop, SNP)` allele-count sum `ac`. |
| `d_het` | `double*` | `[P × tm]` pop-major output: per-`(pop, SNP)` heterozygote count `het`. |
| `stream` | `cudaStream_t` | The CUDA stream the launch is queued on. |

### `launch_fst_allpairs_accumulate`

| Parameter | Type | Meaning |
|---|---|---|
| `d_n`, `d_ac`, `d_het` | `const double*` | The three `[P × tm]` sufficient-stat tensors produced by the decode for this tile. |
| `P` | `int` | Number of populations. |
| `tm` | `long` | The tile width (SNP count) these stats cover. |
| `s_lo` | `long` | First global SNP index of the tile — used to index `d_include`. |
| `d_include` | `const std::uint8_t*` | The **global** summary/inclusion mask, indexed by `s_lo + s`; or `nullptr` to count every SNP. |
| `pair0` | `long long` | First pair rank in this chunk of the `C(P, 2)` pairs. |
| `C` | `long long` | Number of pairs in the chunk (one block each). |
| `d_pair_num` | `double*` | Persistent per-pair WC numerator accumulator; `+=` across tiles. |
| `d_pair_den` | `double*` | Persistent per-pair WC denominator accumulator; `+=` across tiles. |
| `d_pair_cnt` | `long*` | Persistent per-pair count of contributing SNPs; `+=` across tiles. |
| `stream` | `cudaStream_t` | The CUDA stream the launch is queued on. |

All pointer arguments refer to memory that already lives on the GPU. The wrappers
launch their kernels and neither allocate nor copy host memory themselves — the
backend owns and sizes every buffer.

---

## 5. Parity and precision

The matrix values are gated to the single-pair FST path by construction, not by a
separate check: both call the **same** `wc_finalize` from
`core/internal/wc_fst.hpp`, which is also the CPU oracle's finalize. So a
`--all-pairs` matrix cell for pair `(i, j)` is the same number `steppe fst`
reports for that lone pair — there is no second WC implementation that could
drift.

The WC numerator and denominator are variance components built from small
differences, and the per-pair accumulators are long sums over SNPs and tiles, so
this arithmetic runs in **native FP64**. That matches the single-pair kernel and
the D-statistic-style reductions elsewhere in steppe: emulated-FP64 (Ozaki) is
steppe's default only for matmul-heavy work, and there is no GEMM here — just a
per-SNP fold and a sum. Any precision note about this file should say native FP64
for the fold and the reductions, and *why* (no matmul to emulate).

---

## 6. Contracts and invariants

- **Decode once, reuse per pair.** The genotype panel is scanned a single time
  per tile into `{n, ac, het}`; the accumulate stage never re-reads genotypes, so
  the panel is touched independent of `P`.
- **Pop-major stat layout.** The three stat tensors are `[P × tm]` with
  element `(p, s)` at `p*tm + s`. The accumulate stage slices them by population
  index.
- **Shared WC / single writer.** Both stages route through the shared
  `wc_finalize`, and each accumulate block owns one distinct pair rank `r`, so the
  per-pair accumulators need no atomics — one block writes each cell.
- **`+=` across tiles.** `launch_fst_allpairs_accumulate` adds into persistent
  accumulators and is called once per SNP-tile; the running totals are complete
  only after the final tile.
- **`d_include` is the global mask.** It is indexed by `s_lo + s`, not by the
  tile-local `s`; pass `nullptr` to include every SNP in the tile.
- **Device-private header.** Both declarations name `cudaStream_t` and the file
  includes `<cuda_runtime.h>`, so this header sits on the device library's
  internal surface, not on steppe's CUDA-free public API. It is the seam between
  the backend and the all-pairs kernel translation unit.
