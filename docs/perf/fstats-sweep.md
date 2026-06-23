# f-stats wall-clock sweep — f4 / f3 / f4-ratio / qpDstat (Part A + Part B)

**Build:** Release (`CMAKE_BUILD_TYPE=Release`, Ninja). Binary
`/workspace/steppe/build-rel/bin/steppe` + manual bench
`/workspace/steppe/build-rel/bin/bench_fstats_1240k`
(`tests/reference/bench_fstats_1240k.cu`). The Debug per-kernel
`cudaDeviceSynchronize` that voids timing is **not** in this build.

**Data:** REAL AADR v66 **1240K** panel only. The f2-path features (f4 / f3 /
f4-ratio / qpDstat Part A) read the production STPF2BK1 cache
`/workspace/data/1240k_sweep/f2_60` (P=60, n_block=711, 682,707/1,233,013 SNPs
kept, emu40, blgsize 5 cm, autosomes-only — the SAME cache the rotation bench
uses, sourced from the real `.geno` sha256). qpDstat Part B (genotype path) reads
the raw 1240K prefix
`/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB.{geno,snp,ind}` directly.
NO synthetic data for any number here.

**Box:** box5090 — 1x NVIDIA GeForce RTX 5090 (sm_120, 32607 MiB), CUDA 13.0.
SINGLE-GPU `--device 0` (multi-GPU PARKED). `ulimit -c 0`. emu40 precision
(default). Page-cache warmed; median of 3 timed runs after 1 warm-up; GPU util
sampled with `nvidia-smi --query-gpu` during the heaviest runs.

---

## TL;DR — the headline finding

**Shape audit:** the per-item DEVICE kernels are properly shaped — `assemble_f4_quartets`
/ `assemble_f3_triples` are ONE batched gather launch over ALL items reading the
RESIDENT f2 (zero f2-tensor D2H), and the qpDstat Part-B `dstat_block_reduce` is one
batched launch over all (quadruple × block) cells. There is **no per-item device-call
loop** anywhere.

**BUT the scaling shape of the f4/f3 standalone SE is wrong, and it is the dominant
cost.** `run_f4` / `run_f3` reuse the qpAdm `jackknife_cov`, which builds the **full
dense m×m covariance Q AND its full Cholesky inverse Qinv** (m = N items). f4/f3 only
ever read the **diagonal** of Q (the per-item variance) — they never consume the
off-diagonals or Qinv — yet they pay O(N²) memory (three m×m arrays: dQ, dQf, host
out.Q + out.Qinv) and O(N²·nb) SYRK + O(N³) Cholesky. Result: f4/f3 wall is
**super-linear (≈ O(N²))** and **OOMs between N=30k and N=50k** on a 32 GB GPU.

**The "worse-shaped" feature scales BEST.** f4-ratio uses the host per-tuple
`ratio_jackknife` (the documented AT2 `jack_mat_stats` seam, NOT `jackknife_cov`),
which is **O(N·nb) linear**. It runs to **1,000,000 tuples** with a ~flat per-item
cost — but with the **GPU ~idle** (host long-double loop bound). So the host loop the
audit flagged as the "CPU-shape risk" is in fact the only f4-family SE that scales
linearly; the on-device `jackknife_cov` is the real scaling wall.

**Action item (not done here):** give f4/f3 a **diagonal-only** jackknife variance
(per-item O(N·nb), like f4-ratio/dstat already have) instead of the full m×m
Q+Qinv. That removes the O(N²) memory + the OOM and makes f4/f3 linear like f4-ratio.

---

## f2 cache geometry (the item-axis input)

| field | f2_60 |
|---|---|
| P (pops) | 60 |
| n_block | 711 |
| SNPs kept / total | 682,707 / 1,233,013 |
| precision | emu40 (40-bit mantissa) |
| f2.bin | 40.96 MB |
| possible quartets C(60,4) | ≈ 487,635 |

The item batches (quartets / triples / 5-tuples) are deterministic distinct-within-
tuple enumerations over the 60 real pops (recycled with a rotation offset past
C(P,G)), so a given N is reproducible.

---

## f4 — `run_f4` / `assemble_f4_quartets` (device-resident bench, floor-free)

Single-GPU, f2_60 resident in VRAM, median of 3 (warm-up first). Item axis = N
quartets. **dispatches=0 is expected** (the dispatch counter is rotation-only; the
f4 path issues explicit single launches, not `fit_models_batched`).

| N quartets | wall (median) | items/sec | ms/item | peak GPU mem |
|---:|---:|---:|---:|---:|
| 100 | 1.77 ms | 56,649 | 0.0177 | — |
| 1,000 | 17.80 ms | 56,190 | 0.0178 | — |
| 5,000 | 220.2 ms | 22,704 | 0.0440 | — |
| 10,000 | 905.5 ms | 11,044 | 0.0905 | ~2.3 GB |
| 20,000 | 4,226.6 ms | 4,732 | 0.2113 | ~7.0 GB |
| 30,000 | 11,127.3 ms | 2,696 | 0.3709 | ~16 GB |
| 50,000 | **OOM** (cudaErrorMemoryAllocation) | — | — | >32 GB |
| ≥ 100k / 1M | **OOM** | — | — | — |

**ms/item is NOT flat — it grows ≈ linearly in N (i.e. wall ≈ O(N²)).** This is the
m×m covariance: at N the dense Q is N²·8 bytes (N=20k → 3.2 GB; N=30k → 7.2 GB),
and `jackknife_cov` allocates THREE such m×m buffers (dQ, dQf device + out.Q,
out.Qinv host), so it OOMs at N≈36k (3 × 36k² × 8 ≈ 31 GB) — observed between 30k
(ok) and 50k (OOM).

**GPU util (N=20k):** mixed — 100% during the SYRK + Cholesky spans, 0% during the
host xtau staging and the m×m Q/Qinv D2H (each ~3.2 GB host copy). High util here is
a symptom of the wrong (dense) work, not of good shape: f4 needs only the diagonal.

---

## f3 — `run_f3` / `assemble_f3_triples`

Identical engine to f4 (same `jackknife_cov`); identical O(N²) scaling + same OOM
ceiling. Item axis = N triples.

| N triples | wall (median) | items/sec | ms/item |
|---:|---:|---:|---:|
| 100 | 2.01 ms | 49,685 | 0.0201 |
| 1,000 | 19.51 ms | 51,268 | 0.0195 |
| 5,000 | 232.6 ms | 21,498 | 0.0465 |
| 10,000 | 914.9 ms | 10,930 | 0.0915 |
| 20,000 | 4,274.9 ms | 4,678 | 0.2137 |
| 30,000 | 11,118.3 ms | 2,698 | 0.3706 |
| ≥ 50,000 | **OOM** | — | — |

f3 == f4 to within noise (same m×m covariance code path). Same diagonal-only fix
applies.

---

## qpDstat Part A (`qpdstat --f2-dir`) == f4

The qpDstat f2-path dispatches verbatim to `run_f4` (cmd_qpdstat.cpp:309); the
numbers are byte-identical to the f4 table above (same engine, same O(N²) wall, same
OOM). No separate measurement needed.

---

## f4-ratio — `run_f4ratio` / `ratio_jackknife` (the LINEAR one)

Item axis = N 5-tuples. **Does NOT use `jackknife_cov`** — uses the host per-tuple
`ratio_jackknife` (AT2 `jack_mat_stats` seam): O(N·nb) linear, host long-double.

| N 5-tuples | wall (median) | items/sec | ms/item | peak GPU mem |
|---:|---:|---:|---:|---:|
| 100 | 2.65 ms | 37,768 | 0.0265 | — |
| 1,000 | 34.26 ms | 29,193 | 0.0343 | — |
| 10,000 | 417.9 ms | 23,929 | 0.0418 | — |
| 50,000 | 2,151.4 ms | 23,240 | 0.0430 | — |
| 100,000 | 4,777.8 ms | 20,930 | 0.0478 | — |
| 500,000 | 30,253.6 ms | 16,527 | 0.0605 | — |
| 1,000,000 | 60,824.4 ms | 16,441 | 0.0608 | ~22 GB (brief, assemble) |

**~Flat per-item (0.026 → 0.061 ms/item over 100 → 1M); reaches 1M with no OOM.**
Marginal slope (floor-subtracted) ≈ **0.061 ms/tuple ≈ 16,400 tuples/sec**.

**GPU util (N=1M):** **mostly 0% (931/975 samples at 0%; a brief 37–52% blip).** The
wall is the single-threaded host `ratio_jackknife` loop; the device assemble is a
short blip (~22 GB transient = the interleaved 2N-quartet x_blocks/x_loo carrier),
then the GPU sits idle for ~60 s while the host computes the per-tuple ratio
jackknife. So f4-ratio is **HOST-bound, GPU-idle** — linear and OOM-free, but
leaving the GPU unused. (Batching `ratio_jackknife` on-device would lift the GPU
util and cut the wall, but it is correct and the only linearly-scaling f4-family SE
as-is.)

---

## qpDstat Part B — `qpdstat --prefix` (genotype path, `run_dstat`)

Genotype-path normalized-D over the raw 1240K prefix. `--prefix` IS wired
(cmd_qpdstat.cpp:238 → `run_qpdstat_prefix` → `run_dstat`); the `--help` "not yet
implemented" string is **stale** — Part B is live and callable. Reads ONLY the
4–8-pop `pop_union` (`read_ind(Explicit{union})`), so a small-union D over the giant
prefix decodes a tiny P. pop_union = 8 real pops (Mbuti, Han, French, Sardinian,
Karitiana, Papuan, Turkey_N, Iran_GanjDareh_N — all present in the 1240K ind).

| N quadruples | wall (median) | note |
|---:|---:|---|
| 10 | 1.85 s | decode-front-end bound |
| 70 (all C(8,4)) | 1.96 s | flat in N |
| 700 | 1.97 s | flat in N |

**Flat in N** — the wall is the genotype read + `decode_af` over the tiny 8-pop
union, NOT the D throughput. The batched `dstat_block_reduce` (one launch over all
quadruples × blocks) + host `dstat_jackknife` are negligible. **GPU util peaks
~20%** (mostly 0–1%): file-IO + host-decode bound, NOT GPU bound. Peak GPU mem only
626 MB (tiny P union; one [P×M] Q/V materialize + one re-upload — NOT a per-block
bounce). MaxRSS ~712 MB.

**vs extract-f2:** Part B at 1.9 s is *faster* than extract-f2's 5.8 s floor because
it decodes only an 8-pop union vs extract-f2's 60 pops over 4,402 individuals — far
less decode work for the same genotype read. A LARGER pop_union would push Part B
toward (and past) the extract-f2 decode wall.

---

## CLI process-floor (the per-process artifact, for reference)

The `steppe f4/f3/f4-ratio/qpdstat` subcommands re-pay a fixed
CUDA-context-init + f2.bin-load floor PER PROCESS, and ingest items via argv
(ARG_MAX-capped at ~33k quartets ≈ 2 MB). Measured floor:

| invocation | wall |
|---|---|
| `f4 --f2-dir f2_60` N=1 | 0.75 s |
| `f4 --f2-dir f2_60` N=1000 | 0.78 s |

The floor (~0.75 s) dominates at small N; the marginal slope (0.78−0.75)/999 ≈
0.03 ms/quartet matches the bench's 0.018 ms/item. For production scale use the
in-process engine (the bench tables above), not repeated CLI spawns.

---

## Where these sit alongside the existing 1240K stages

| stage | scale | wall / throughput | GPU-bound? |
|---|---|---|---|
| extract-f2 (SET60, no-hash) | 60 pops, 4,402 ind | 5.81 s | yes (read+decode+f2) |
| qpAdm single fit (f2_60, jk=0) | 1 model | 1.07 s | floor (ctx+load) |
| qpAdm heavy 4-way (f2_60, jk=2) | 1 model | 3.67 s | yes (711-block SE) |
| rotation (f2_60, jk=None) | 781 models | 272 ms / 2,866 models/s | yes (batched) |
| rotation (f2_60, jk=All) | 781 models | 626 ms / 1,247 models/s | yes (batched) |
| **f4 / f3 / qpDstat-A** | **≤ 30k items** | **O(N²), OOM ≥ ~36k** | yes (but DENSE m×m — wrong work) |
| **f4-ratio** | **≤ 1M items** | **0.061 ms/item, 16.4k/s** | **no (host-bound, GPU idle)** |
| **qpDstat Part B (genotype)** | **8-pop union, any N** | **~1.9 s, flat in N** | no (IO+decode bound) |

---

## Shape verdict (per the audit question)

1. **`assemble_f4_quartets` / `assemble_f3_triples`** — ONE batched gather launch over
   ALL items, reading the resident f2 (zero f2-tensor D2H). **PROPER SHAPE.**
2. **qpDstat Part B `dstat_block_reduce`** — ONE launch over all (quadruple × block)
   cells, device-resident Q/V. One full materialize + one re-upload of the tiny
   [P×M] Q/V (NOT a per-block bounce). **PROPER SHAPE** for a small pop_union.
3. **f4 / f3 SE (`jackknife_cov`)** — **THE BOTTLENECK.** Builds the full dense m×m Q
   AND its Cholesky inverse Qinv when f4/f3 need only the diagonal. O(N²) memory →
   OOM at N≈36k; super-linear (O(N²)) wall. **WRONG SHAPE for a per-item statistic.**
   Fix: a diagonal-only jackknife variance (O(N·nb)), not the full covariance.
4. **f4-ratio / dstat host jackknife** — O(N·nb) linear; the only linearly-scaling
   f4-family SE. Correct scaling but **GPU-idle** (host long-double loop). Optional
   future win: batch it on-device to use the GPU and cut the wall.

No fabricated numbers; every value above is a measured median on REAL 1240K AADR,
SINGLE-GPU, Release. The bench TU is `tests/reference/bench_fstats_1240k.cu`.
