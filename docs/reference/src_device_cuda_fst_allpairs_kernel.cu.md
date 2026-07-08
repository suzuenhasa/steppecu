# `fst_allpairs_kernel.cu` reference

## 1. Purpose

This file is the GPU engine behind `steppe fst --all-pairs`: instead of one
Weir & Cockerham 1984 FST between two populations, it fills in the whole `P × P`
symmetric matrix — every population against every other — in a single sweep of
the panel.

The naive way to build that matrix would be to run the single-pair FST `C(P,2)`
times, re-decoding the packed genotypes from scratch for each pair. That throws
away almost all the work: the sufficient statistics a pair needs (how many
called alleles each population has at a SNP, its allele count, its heterozygote
count) depend only on *one* population at a time, not on the pairing. So this
file splits the job in two:

1. **Decode once.** For each SNP-tile, unpack the 2-bit genotypes into a
   per-`(pop, SNP)` sufficient statistic `{n, ac, het}` — computed exactly once
   per population, never per pair.
2. **Fold every pair.** Enumerate all `C(P,2)` pairs on-device via the O(1)
   closed-form k=2 unrank, and for each pair run the WC finalize over the two
   populations' pre-decoded stats, accumulating a genome-wide `num`/`den` sum
   per pair.

Both stages reuse the *shared* primitives that the single-pair path and the CPU
oracle use, so the matrix can never quietly drift from the single-pair answer:

- `wc_accumulate` / `wc_finalize` (`core/internal/wc_fst.hpp`) — the actual WC
  variance-component math, in native FP64.
- `genotype_code` / `kCodesPerByte` (`core/internal/decode_af.hpp`) — the 2-bit
  unpack.
- `readv2_unrank_pair` (`sweep_unrank.cuh`) — the closed-form k=2 pair unrank.

The two kernels here are wrapped by thin host launchers; the `.cuh` companion
(`fst_allpairs_kernel.cuh`) declares those launchers and has its own short
reference doc.

---

## 2. Stage one: decode the sufficient statistics once

`fst_suffstat_decode_kernel` runs one thread per `(pop p, SNP s_local)` over the
current SNP-tile `[s_lo, s_lo + tm)`. The grid is the usual 2-D decode block
(`kDecodeBlockX × kDecodeBlockY`), tiled `tm` wide over SNPs and `P` tall over
populations, with the standard out-of-range guard (`p_l >= P || s >= tm`).

Each thread finds its SNP's byte and bit position inside the packed record
(`byte_in_record`, `pos_in_byte`), then walks *its* population's contiguous slice
of samples — `[pop_offsets[p], pop_offsets[p+1])` — folding every diploid code
into a `WcPerPop` accumulator with `wc_accumulate`. The result `{n, ac, het}` is
stored pop-major into three `P × tm` tensors at `off = p*tm + s`.

This is the single-pair `fst_wc_kernel`'s per-pop inner loop, but generalized to
all `P` populations and *stored* rather than immediately finalized. Decoded once
per tile, it is then reused across every one of the `C(P,2)` pairs — the whole
point of the design.

---

## 3. Stage two: one block per pair, fold across the tile

`fst_allpairs_accumulate_kernel` consumes those three tensors. It's launched with
**one CUDA block per pair** (`grid.x = C`, `kFstAccumBlock = 256` threads each)
over the pair chunk `[pair0, pair0 + C)`.

Each block maps its rank `r = pair0 + blockIdx.x` to the two population indices
`(i, j)` with `readv2_unrank_pair` — the closed-form k=2 unrank, no search, giving
`c[0] = i < c[1] = j`. From there the block's threads **stride-share this tile's
SNPs**: each thread walks `s = threadIdx.x, threadIdx.x + blockDim.x, ...`,
optionally skipping SNPs the global summary mask `include[s_lo + s]` zeroes out,
reads the two populations' pre-decoded `{n, ac, het}` slices, runs the shared
`wc_finalize`, and folds the valid sites into a private `(num, den, cnt)` partial.

Then a shared-memory tree reduction (`s_num` / `s_den` / `s_cnt`, sized to
`kFstAccumBlock`, a power of two) collapses the block's partials, and thread 0
**adds** the block total into the persistent per-pair `Σ` (`pair_num[r]`,
`pair_den[r]`, `pair_cnt[r]`). Because exactly one block owns each distinct `r`
within a launch, that `+=` needs **no atomic**, and the reduction is
deterministic. The kernel is called once per SNP-tile, and the `+=` is what
reduces the running WC num/den across all the tiles into a genome-wide sum.

---

## 4. Why one block per pair, not one thread per pair

The v1 design mapped **one thread** to each pair. That launched only `C` threads
total, and each of those threads serially looped all `M` SNPs. At low population
counts `C` is small (e.g. `P = 5` is just 10 pairs), so the launch was badly
occupancy-starved — a handful of threads each grinding through the whole genome.
That set steppe's roughly flat low-`K` wall.

Blocking over the SNP axis fixes it: `C` blocks × `kFstAccumBlock` threads keeps
the GPU busy even when `C` is small, and it shortens each thread's loop from `M`
to about `M / blockDim`. The correctness is unchanged — the block-tree reduction
lands on the same WC num/den — it's purely an occupancy restructuring.

---

## 5. Precision and parity

The WC arithmetic — `wc_accumulate` in the decode and `wc_finalize` in the
accumulate — is **native FP64**, straight out of `core/internal/wc_fst.hpp`,
the same functions the single-pair kernel and the CPU oracle call. There is no
matmul anywhere in this file, so the emulated-FP64 (Ozaki) default doesn't apply
here; native FP64 is the right tool, and sharing the exact WC functions is what
guarantees the matrix agrees with the single-pair path within FP64 round-off.

The block reduction's summation order is fixed and deterministic (a power-of-two
tree over `kFstAccumBlock`, then a single-writer `+=` per tile), so a given pair's
genome-wide `num`/`den` is bit-stable run to run.

---

## 6. The launch wrappers

Both host entries are thin and live at the bottom of the file:

- `launch_fst_suffstat_decode` early-returns on an empty problem
  (`P <= 0 || tm <= 0`), sizes the 2-D grid from `kDecodeBlockX/Y` with `cdiv`,
  launches the decode kernel, and runs `STEPPE_CUDA_CHECK_KERNEL()`.
- `launch_fst_allpairs_accumulate` early-returns on `C <= 0 || tm <= 0`, then
  launches `grid.x = C` blocks of `kFstAccumBlock` threads. Because it uses one
  block per pair, `grid.x` must stay under the CUDA grid-x cap; the caller chunks
  `C` down to `kFstPairChunkClamp` (2^30), which sits comfortably under
  `kMaxGridX` (2^31 − 1), and a `STEPPE_ASSERT` documents and guards that
  contract. The block dimension **must** equal `kFstAccumBlock` so the kernel's
  shared-memory reduction (sized to that constant) exactly covers the launched
  threads.

---

## 7. Edge cases and invariants at a glance

- **Decode once, reuse across all pairs.** The `{n, ac, het}` sufficient stats
  are computed per `(pop, SNP)` a single time per tile; every pair reads them
  (section 1–2).
- **Shared WC math** with the single-pair path and CPU oracle — the matrix can't
  drift from the single-pair answer (section 5).
- **Native FP64**, deterministic tree reduction; no matmul, so not the emulated
  default (section 5).
- **No atomics** — one block owns each pair rank `r`, so the per-pair `+=` has a
  single writer (section 3).
- **`include` mask is global**, indexed by `s_lo + s`; a null mask means keep
  every SNP (section 3).
- **Empty problem is a clean no-op** — both wrappers return before launching
  (section 6).
- **Grid-x cap is a hard contract** — the caller chunks `C` so `grid.x = C`
  stays under `kMaxGridX`; the assert enforces it (section 6).
