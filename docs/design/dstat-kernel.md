# The GPU genotype-path normalized-D per-SNP reduction (dstat_kernel)

> Source: `src/device/cuda/dstat_kernel.cu` — a CUDA TU PRIVATE to steppe_device. Cross-refs:
> architecture.md §4 (layering), §7 (the narrow launch-wrapper seam), §12 (parity / the
> cancellation carve-out), §13 (testing). The kernel body + `<<<>>>` live ONLY in the source
> TU; the backend reaches it through the narrow launch wrapper (`dstat_kernel.cuh`).

The GPU genotype-path NORMALIZED-D per-SNP reduction — qpDstat Part B (the S2 divergence;
`include/steppe/dstat.hpp`) AND the qpfstats genotype-f4 numerator reduce (the ~305k-combo
wall). NEVER the f2 GEMM, NEVER the f2 cache: a segmented reduction over each jackknife
block's SNP columns, accumulating the AT2 `qpdstat_geno` num/den per (combo, block) over
the `allsnps=TRUE` per-(block,combo) finiteness mask.

## STATISTIC (AT2 `qpdstat_geno`, `allsnps=TRUE`, `f4mode=FALSE`; pinned to the AT2 R golden)

For combo `(p1,p2,p3,p4)` at SNP `s` with `a=Q[p1,s]`, `b=Q[p2,s]`, `c=Q[p3,s]`,
`d=Q[p4,s]`,

```
  num = (a-b)*(c-d),  den = (a+b-2ab)*(c+d-2cd)
```

summed per block ONLY over SNPs valid in all 4 pops (`V==1` for `p1..p4`).

## THE PERFORMANCE FIX — SNP-TILED PAIRWISE-DIFFERENCE REUSE (the shared kernel)

The legacy mapping was one-thread-per-(combo,block), each thread re-reading its 4 pops'
Q/V down the block at stride P -> ZERO reuse across the ~305k combos -> ~11.6 TB of
uncoalesced global loads, memory-bound at ~8% of peak. Every numerator `(Q[a]-Q[b])·
(Q[c]-Q[d])` uses only `C(P,2)` distinct per-SNP differences. The tiled kernel: one
grid-block per jackknife block (`gridDim.x = n_block`) so the whole `[begin,begin+size)`
range is owned by one grid-block and accumulation stays ascending-s within the block (NO
cross-tile reduction); `gridDim.y = ceil(N/T)` combo-tiles, `T = blockDim.x` threads. Per
SNP the block's T threads COOPERATIVELY load `Q[P·s+i]`/`V[P·s+i]` for `i` in `[0,P)` into
shared memory ONCE (one coalesced contiguous-column read, reused by all T combos), then
fill the `C(P,2)` shared `diff[i<j]=Q[i]-Q[j]` and `het[i<j]=Q[i]+Q[j]-2·Q[i]·Q[j]` arrays;
each thread then reconstructs its combo's num/den from the shared pairwise tables and a
per-pop V-mask. One coalesced pass of Q+V (~191 MB each) per combo-tile -> compute-bound.

## GOLDEN-EXACT (the §12 cancellation carve-out; NATIVE FP64, never emulated)

- **PRODUCT identity:** `f1 = (p1<p2 ? +diff[pair(p1,p2)] : -diff[pair(p2,p1)])`. In
  IEEE-754 negation is exact and subtraction is symmetric so `-(Q[p2]-Q[p1]) ==
  Q[p1]-Q[p2]` bit-for-bit -> `f1·f2` is bit-identical to the legacy `(a-b)·(c-d)`. `den`
  uses the SYMMETRIC `het[ij]` -> direct lookup, bit-identical to `(a+b-2ab)·(c+d-2cd)`.
- **MASK identity:** `mask = V[p1]·V[p2]·V[p3]·V[p4] != 0` == the legacy 4-pop joint `V==1`
  test.
- **ORDER:** each (combo,block) sum is owned by ONE grid-block (`gridDim.x=b`) and one
  register, summed in ASCENDING s within `[begin,begin+size)` -> per-(combo,b) sums are
  BIT-IDENTICAL to the legacy kernel, not merely within rtol.

## HUGE-npop FALLBACK

When `2·C(P,2)+2·P` doubles no longer fit the 99 KB opt-in shared budget (`P > ~112` with
den), `launch_dstat_block_reduce` falls back to the RETAINED legacy per-(combo,block)
kernel (same TU, same wrapper — a runtime branch, NOT a forked path). steppe's resident
tier OOMs past ~250 pops anyway, so the fallback band is ~112..250.

## DESIGN-FOR-SCALE

Batched over BOTH the N-combo axis and the n_block axis; reads the resident Q/V `[P × M]`
in VRAM. The output `numsum/densum/cnt` are `[N × n_block]` (tiny).

This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). The kernel body +
`<<<>>>` live ONLY here; the backend reaches it through the narrow launch wrapper
(`dstat_kernel.cuh`), never includes this body (architecture.md §7).
