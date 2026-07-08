# `cuda_backend_fst_allpairs.cu` reference

## 1. Purpose

`src/device/cuda/cuda_backend_fst_allpairs.cu` is the GPU-backend translation unit
behind `steppe fst --all-pairs`: it computes the full population-by-population
Weir & Cockerham 1984 FST matrix in one pass over the panel. Where the single-pair
path answers "how differentiated are these two populations?", this method answers
it for **every** pair at once — all `C(P, 2) = P·(P−1)/2` of them — and hands back
the raw variance-component sums the caller finalizes into a `P × P` matrix.

It holds a single member function, `CudaBackend::fst_wc_all_pairs`, and it is a
CUDA TU private to `steppe_device`, mirroring `cuda_backend_fst.cu` (the single-pair
path) and `cuda_backend_pca.cu`. The two device kernels it drives live in
`fst_allpairs_kernel.cu` / `.cuh` and are documented there.

The whole design turns on one observation. A pairwise FST needs each population's
per-SNP sufficient statistics — sample size `n`, allele count `ac`, and
heterozygote count `het`. The naive all-pairs loop would re-decode the packed
genotypes for every pair, redoing the same `P` decodes `O(P²)` times. Instead this
method **decodes each population's `{n, ac, het}` exactly once per SNP-tile** and
then reuses that decoded window across every pair. Decode is `O(P)`; only the
cheap variance-component accumulation is `O(C(P,2))`.

---

## 2. What crosses PCIe, and what never does

The method is deliberately frugal about the bus. Going up to the GPU: the packed
genotype tile (individual-major, population-contiguous), the `P + 1` population
offsets, and — when present — an optional autosome summary mask. Coming back down:
just the three small `C(P,2)`-length pair vectors (numerator, denominator, valid
count).

Two larger things are engineered to **never** cross the bus, and never even fully
materialize on the GPU:

- **The `P × M × 3` sufficient-stat tensor is never materialized.** Only a
  `P × tileM` window of `{n, ac, het}` is resident at any moment — the same
  SNP-tiling idiom `extract_f2` and the PCA path use (see the `s_lo` sweep in
  section 4).
- **The `P²` combine stays on the GPU.** The per-pair sums are accumulated on the
  device across all tiles; the host only receives the finished vectors.

This is why a many-population run stays bounded in VRAM and light on the bus: the
expensive tensor is a sliding window, and the expensive reduction is device-resident.

---

## 3. The result struct and the maxcomb guard

The output is an `FstMatrix`: three parallel vectors of length `C(P,2)` in
pair-index order — `pair_num` (Σ WC numerator per pair), `pair_den` (Σ WC
denominator per pair), `pair_cnt` (valid-site count per pair) — plus `P`,
`enumerated` (= `C(P,2)`), a `capped` flag, a `status`, and a precision tag. The
method fills the raw sums; assembling them into the symmetric `P × P` matrix
(`FST = num/den` per off-diagonal cell) is the caller's job.

Before allocating anything, two guards fire:

- **Degenerate shape.** `M <= 0` or `P <= 0` returns `Status::InvalidConfig` with an
  empty matrix.
- **Maxcomb cap.** If the pair count exceeds `kFstatMaxComb` (10⁸, the shared
  f-stat sweep ceiling) and the caller did not pass `sure`, the method refuses:
  it sets `capped = true`, `status = InvalidConfig`, and returns without launching.
  This mirrors the f-stat sweep's runaway guard. The cap is on the **pair count
  only** — it does not bound the `C(P,2)·M` accumulation volume, which is the real
  work but is cheap per item.

`P < 2` (so `npairs == 0`) is not an error: the vectors are sized zero and the
method returns a well-formed, empty (diagonal-only) matrix with `Status::Ok`.

The precision tag is `Fp64` — this path runs in **native FP64** (the reduction
carve-out; section 6).

---

## 4. The tiled sweep

Once the guards pass, the method allocates the persistent per-pair accumulators
(`dPairNum`, `dPairDen`, `dPairCnt`, each length `C(P,2)`) and zeroes them, then
sweeps the SNP axis in tiles.

**Choosing the tile height.** The resident sufficient-stat window is
`3 · P · tileM` doubles. The method sizes `tileM` so that window fits within
roughly `free_vram / 4` (leaving room for the packed tile and the accumulators),
with a 1 GiB fallback if the capability query reports zero free VRAM. `tileM` is
clamped to `[1, M]`, and the environment variable `STEPPE_FST_TILE`, if set to a
positive integer, overrides the computed value (an escape hatch for tuning or
tests).

**The sweep itself.** For each tile `[s_lo, s_lo + tm)`:

1. **Decode once.** `launch_fst_suffstat_decode` fills the `P × tm` windows
   `dN`, `dAc`, `dHet` — one decode of every population's sufficient statistics for
   this tile.
2. **Accumulate every pair.** An inner loop over pair chunks calls
   `launch_fst_allpairs_accumulate`, which unranks each pair index into its
   `(i, j)` tile-population indices, runs the shared WC finalize over the two
   populations' decoded slices, and **adds** the block totals into the persistent
   per-pair accumulators. Because the accumulate uses `+=`, the reduction across
   SNP-tiles is the sum over successive tile launches — no per-tile intermediates.

**Pair chunking.** The pair axis is chunked at `kFstPairChunkClamp` (2³⁰) per
launch, so the flat pair index stays comfortably `int`-addressable and the launch
grid stays under the max grid dimension. This mirrors the sweep's integer clamp.
For any realistic `P` the chunk covers all pairs in one iteration; the loop exists
for the pathological large-`P` case.

The optional summary mask is indexed by **global** SNP position (`s_lo + s`) inside
the kernel, so tiling does not shift which SNPs the mask includes.

---

## 5. Contracts and invariants

- **Parity with the single-pair path is structural.** Both kernels share the WC
  variance-component arithmetic with the single-pair FST kernel and the CPU oracle
  through `core/internal/wc_fst.hpp`, so a matrix cell cannot numerically drift
  from what the single-pair path would produce for the same pair. The pair
  enumeration uses the same closed-form `k = 2` unrank as the f-stat sweep.
- **The `+=` accumulate is the cross-tile reduction.** Correctness depends on the
  accumulators being zeroed once up front (they are, via `cudaMemsetAsync`) and
  every tile adding into them; no tile overwrites another's contribution.
- **No atomics across pairs.** Each pair is owned by a distinct CUDA block in the
  accumulate kernel, so distinct pairs never contend for the same accumulator slot
  (details in the kernel doc).
- **Single-stream ordering.** All copies, memsets, and launches are enqueued on the
  backend's stream, and the method blocks on `cudaStreamSynchronize` before
  returning — so the host-side result vectors are populated by the time the call
  returns.

---

## 6. Precision policy

This path runs in **native FP64**, and that is the deliberate reduction carve-out,
not the house default. Steppe's default is emulated-FP64 (Ozaki), but that default
is matmul-only — it buys its accuracy inside GEMMs. There is no GEMM here: the WC
numerator and denominator are sums that subtract nearly equal quantities and are
sensitive to cancellation, which is exactly the case the native-FP64 carve-out
exists for. The decode, the per-pair finalize, and the cross-tile accumulation are
all plain `double`. The `precision_tag` on the result is `Fp64` to reflect this.

If you are writing or reading a precision note about this file: say native FP64,
say it is the cancellation-prone reduction carve-out, and do not describe it as the
emulated-FP64 default.

---

## 7. Edge cases at a glance

- **`M <= 0` or `P <= 0`** — `Status::InvalidConfig`, empty matrix, no launch.
- **`P < 2` (`npairs == 0`)** — clean, well-formed empty matrix, `Status::Ok`.
- **Pair count over `kFstatMaxComb` without `sure`** — `capped = true`,
  `Status::InvalidConfig`, no launch.
- **Zero free VRAM reported** — the tile budget falls back to 1 GiB rather than
  dividing by zero.
- **Empty packed tile** (`packed_bytes == 0`) — a 1-byte placeholder buffer is
  allocated so no zero-size device allocation occurs, and the upload is skipped.
- **No summary mask** — `d_include` is passed as `nullptr`; every SNP in the tile
  contributes.
- **`STEPPE_FST_TILE` set** — overrides the computed tile height (still clamped to
  `[1, M]`).
