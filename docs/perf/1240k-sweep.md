# 1240K wall-clock sweep — extract-f2 + qpAdm fit + rotation

**Build:** Release (`CMAKE_BUILD_TYPE:STRING=Release`, confirmed in
`/workspace/steppe/build-rel/CMakeCache.txt`). Binary
`/workspace/steppe/build-rel/bin/steppe` (4.83 MB, `-DSTEPPE_BUILD_CLI=ON`,
Ninja). The Debug per-kernel `cudaDeviceSynchronize` that voids timing is **not**
in this build.

**Data:** REAL AADR v66 **1240K** panel
`/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB.{geno,snp,ind}` —
steppe-native TGENO. NOT the HO panel, NOT synthetic.

**Box:** box5090 — 2x NVIDIA GeForce RTX 5090 (sm_120, 32607 MiB each), CUDA 13.
`ulimit -c 0`, core dumps cleared (0 remaining).

**Device policy — SINGLE-GPU (`--device 0`) is the supported, measured path.**
MULTI-GPU IS PARKED: the 2nd GPU is *slower* for this workload — extract-f2's
f2_blocks get bounced (a 2nd ~7 GB D2H) and the qpAdm fit is a single-GPU compute
shape — so the supported path is one GPU. The numbers below are SINGLE-GPU. The
*original* sweep accidentally ran extract-f2/qpadm under `--device auto` (= both
GPUs); those numbers are re-reported here on `--device 0`. Multi-GPU is **not**
benchmarked here (root-caused, fix deferred).

**Status:** independently verified (this doc). Re-confirmed end to end on box5090,
Release, REAL 1240K, SINGLE-GPU: build type (`CMAKE_BUILD_TYPE:STRING=Release` in
`build-rel/CMakeCache.txt`), the REAL f2 geometry (f2_60: P=60, 711 blocks,
682,707/1,233,013 SNPs, sourced from the real `.geno` sha256), an extract-f2
`--device 0` spot re-time (43.2 s, within noise of 43.4 s), and the REAL batched
`run_qpadm_search` rotation throughput (2866 models/sec jk=0; 1247 models/sec full
LOO SE). The earlier "qpadm-rotate not implemented / ~1 model/sec" was a
CLI-scaffold + process-spawn artifact and is corrected below.

---

## Dataset geometry

| field | value |
|---|---|
| `.geno` | 7,117,276,654 bytes (6.7 GB), TGENO header `TGENO 23089 12...` |
| `.snp`  | 1,233,013 SNPs (`wc -l`) |
| `.ind`  | 23,089 individuals (`wc -l`) |
| pops with >=5 indiv | 926 distinct |
| ploidy | auto (`adjust_pseudohaploid`): mixed pseudo-haploid + diploid |

Haak anchors all present with >=5 individuals (verified): Mbuti=15, Han=46,
French=31, Sardinian=31, Turkey_N=68, Russia_Samara_EBA_Yamnaya=46,
Czechia_EBA_CordedWare=48, Israel_Natufian=17, Iran_GanjDareh_N=17,
Karitiana=16, Papuan=32, Serbia_IronGates_Mesolithic=73, Czechia_BellBeaker=61,
England_BellBeaker=28. A few Haak UP pops are too small under the >=5 rule
(Russia_Kostenki_UP=3, MA1 / Malta / UstIshim < 5) and are excluded.

Jump from HO (~600K SNPs) to 1240K (~1.23M SNPs): ~2x the SNP axis, and a
6.7 GB `.geno` that must be streamed and decoded on every extract-f2.

---

## extract-f2 — wall-clock x pop-set size (SINGLE-GPU)

`extract-f2 --prefix … --pops … --blgsize 0.05 --maxmiss 0 --auto-only --device 0`.
blgsize 0.05 -> 711–713 jackknife blocks. **SINGLE-GPU is the supported path.**

**CORRECTION (ROADMAP §6, the extract-f2 hash-bottleneck fix).** The original
~40–44 s wall below was NOT decode/IO — it was a **byte-at-a-time provenance
SHA-256** of the whole 6.7 GB `.geno` (stored as `geno_sha256` in `meta.json`),
which compressed at ~190 MB/s scalar and burned ~37 s of the ~41 s. The genuine
per-individual seek+gather + TGENO decode + f2 GEMM are each well under a second;
the table's flat ~40 s was the constant whole-file hash, NOT the (pop-set-scaling)
genuine work. That hash is now **OFF by default** (`--no-hash`); it is opt-in via
`--hash` (a bulk SHA-NI hash overlapped on a background thread with the GPU
pipeline). The original `--device auto`/`--device 0` numbers are kept below as the
historical record of the OLD (hashing-by-default) code.

| pop set | #indiv | SNPs kept (of 1,233,013) | blocks | wall (OLD, hash-by-default) | MaxRSS | f2.bin |
|---|---:|---:|---:|---:|---:|---:|
| SET5  |   206 | 1,045,408 | 713 | 40.02 s | 0.98 GB | 288 KB |
| SET15 |   596 |   691,342 | 711 | 40.61 s | 1.50 GB | 2.56 MB |
| SET30 | 1,907 |   704,508 | 711 | 41.66 s | 2.59 GB | 10.24 MB |
| SET60 | 4,402 |   682,707 | 711 | 43.97 s | 4.73 GB | 40.96 MB |

**Re-measured SINGLE-GPU (`--device 0`, REAL 1240K geno, page-cache warm), SET60
on the post-fix build:**

- **DEFAULT (no `--hash`):** two runs at **5.81 s / 5.88 s** (rc=0; P=60, 711
  blocks, **682,707 SNPs kept** — bit-matches the SET60 geometry above; f2.bin
  digest `a55433654f29…` is byte-identical to the OLD dir). vs the OLD
  hash-by-default 43.97 s, that is **~7.5x faster** — the ~38 s removed was
  exactly the provenance SHA, not decode/IO.
- **WITH `--hash`:** **7.75 s** (rc=0). The 6.7 GB hash runs on a background
  thread (bulk SHA-NI, ~5.5 s standalone — `sha256sum` is 5.54 s) overlapping the
  ~5.8 s GPU pipeline, so it adds only ~1.9 s, not ~38 s. The digest
  `geno_sha256 = ce9e13a990b26e29eadbb5630dfd2f706cff0aea7a5df59c8d7744346d7428b4`
  is **byte-for-byte equal to `sha256sum`** of the `.geno` — provenance preserved.

The genuine read+decode+f2 floor is ~5.8 s at SET60 (P=60, 4,402 individuals);
it is single-GPU (the 2nd GPU is parked, per the multi-GPU root cause, and would
cost a 2nd ~7 GB f2_blocks D2H). `meta.json` records `source_hash_computed`
(false on the no-hash default, with the `*_sha256` fields `""` by design) so a
consumer knows the absence is DELIBERATE.

**Scaling:** the genuine wall scales with the pop set (individuals gathered + the
P×P×n_block f2), no longer masked by the constant whole-file hash. SNPs-kept DROPS
as pops are added because `--maxmiss 0` requires zero missing across all selected
pops (joint completeness) — correct, intended behavior, not silent data loss.

---

## extract-f2 scaling to 700 pops (single-GPU, no-hash)

**Build/policy:** Release (`build-rel`, `CMAKE_BUILD_TYPE:STRING=Release`),
SINGLE-GPU (`--device 0`), **no `--hash`** (no-hash default — `meta.json` records
`source_hash_computed: false`), REAL 1240K, `--blgsize 0.05 --maxmiss 0
--auto-only`. The provenance-SHA bottleneck is FIXED (the 60-pop wall is now ~4.6 s,
was ~44 s with hashing).

Pop sets are **nested, deterministic file-encounter-order** lists of the 926
pops with >=5 individuals (each P-set is the first P such pops; strictly nested).
NOTE this differs from the §"extract-f2 — wall-clock" SET60 above, which is the
*largest*-60 pops (682,707 SNPs); the file-order 60-set here is 446,853 SNPs. The
collapse SHAPE is identical (and even steeper). #indiv per set is real
(`.ind` 3rd column).

| P | #indiv | SNPs kept (of 1,233,013) | blocks | f2.bin | wall (`--maxmiss 0`) | rc |
|---:|---:|---:|---:|---:|---:|---:|
|  60 |  1,353 | 446,853 | 711 |  40,956,508 B (41 MB) |  4.60 s | 0 |
| 120 |  2,785 | 147,472 | 710 | 163,586,904 B (164 MB) |  6.40 s | 0 |
| 250 |  4,876 |   8,557 | 645 | 645,002,644 B (645 MB) | 10.65 s | 0 |
| 500 |  9,544 |       3 |   3 |  12,000,076 B (12 MB) | 16.79 s | 0 |
| 700 | 14,057 |       0 |   — | — (no output, RC=2) | 22.95 s | 2 |

f2.bin obeys `P² · n_block · 16 + ~2.6 KB header` exactly (verified all rows;
e.g. 250²·645·16 = 645,000,000; 500²·3·16 = 12,000,000). `n_block` shrinks below
711 once whole jackknife blocks lose every SNP (645 at P=250, 3 at P=500).

**Headline 700-pop time: ~23 s — and it is NOT an f2 result.** Under `--maxmiss 0`
the 700-pop joint-completeness filter leaves **0 of 1,233,013 SNPs**, so the tool
exits **RC=2** (`every SNP was filtered out (0 of 1233013 kept) — relax the
filters`) after ~22.95 s. That ~23 s is essentially all read+decode of 14,057
individuals; the f2 GEMM never runs (0 SNPs). The maxmiss-0 700-pop "wall" is a
hard **filter-empty failure**, not a perf number.

**The `--maxmiss 0` SNP collapse is REAL and super-linear.** Each added pop is
another zero-missing constraint, so the surviving SNP set collapses:
446,853 (60) → 147,472 (120) → 8,557 (250) → 3 (500) → 0 (700). Consequence: at
high P the maxmiss-0 f2 is cheap-but-degenerate (or empty), so wall is dominated
by read+decode, not the GEMM.

**Where O(P²·SNPs) f2 starts to dominate — it doesn't, under maxmiss 0; it shows
as a VRAM wall under a relaxed filter.** Under maxmiss 0 the wall climbs
monotonically with #indiv (4.6 → 6.4 → 10.7 → 16.8 → 22.9 s as #indiv goes
1,353 → 14,057) while SNPs-kept (GEMM FLOPs) COLLAPSE to 3 then 0 — so it is
**decode/individual-bound**, not GEMM-bound. The true O(P²·SNPs) f2 cost only
appears when a full SNP set is retained, which requires relaxing maxmiss — and
then the limit is **memory, not time**:

| P | filter | SNPs kept | wall | peak VRAM (dev0) | result |
|---:|---|---:|---:|---:|---|
| 250 | `--maxmiss 0.5` | 1,104,231 | 17.1 s | ~16–31 GiB* | rc=0, f2.bin 713 MB (RESIDENT) |
| 500 | `--maxmiss 0.5` |   (full)  | 26.2 s | 17,544 MiB | OOM (RC=5) on the OLD RESIDENT-only path |
| 700 | `--maxmiss 0.5` |   (full)  | 37.1 s | 24,518 MiB | OOM (RC=5) on the OLD RESIDENT-only path |

**SUPERSEDED (ROADMAP §6, the M5 streamed-tier wiring).** The 500/700-pop OOMs
above were the **un-streamed RESIDENT path** — `extract-f2` called the
resident-only entry (`compute_f2_blocks_multigpu_device`), whose `7·P·M`-double
FP64 feeder does not fit a 32 GB 5090 once P passes ~250 at a full ~1.1 M-SNP set.
That entry's budget wall is real (`cuda_backend.cu`: at P=768/M=584k the raw
10.8 GB + feeder 17.9 GB + resident f2/Vpair 7.1 GB = 35.8 GB > 32 GB → OOM).

**With the adaptive tiered/streamed path wired in (`extract-f2` now calls the
UNIFIED `compute_f2_blocks_multigpu_tiered`), single-GPU extract-f2 COMPLETES 700
pops on 1240K.** The streamed SNP-tile INPUT path keeps the GPU peak independent of
M, so the resident feeder OOM no longer binds. Re-measured on box5090 (1x RTX 5090,
`--device 0`, Release, REAL 1240K, `--auto-top-k 700 --maxmiss 0.5`):

| P | filter | tier | SNPs kept | blocks | wall | peak VRAM (dev0) | f2.bin | result |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 700 | `--maxmiss 0.5` | **host** (auto-selected) | 1,133,249 | 713 | **57.6 s** | **25,368 MiB** | 5,589,922,916 B (5.59 GB) | **rc=0 — COMPLETES (was OOM)** |
| 700 | `--maxmiss 0.5` | **disk** (`--tier disk`) | 1,133,249 | 713 | 68.5 s | ~25 GiB (compute phase) | 5,589,922,916 B (5.59 GB) | rc=0 — COMPLETES |

The auto verdict picks **host** (result 5,630 MiB > the resident `7·P·M` feeder
envelope → stream); `--tier disk` pins the disk-staged variant. **The two 5.59 GB
`f2.bin` files are BYTE-FOR-BYTE IDENTICAL** (`cmp` clean, identical SHA-256
`714e6edc5461ef6251c061820593bbd07b647c8b2a1e85bf7cd07ce940828d05`) — the
streamed==resident guarantee (architecture.md §12) holds at 700-pop production
scale, not just the 9-pop golden. Peak VRAM ~25 GiB is INDEPENDENT of M (matches
the c65179f single-GPU evidence: full-autosome `derived_2500` M=584,131 completed
at P=1000/1500/2000/2500 single-GPU at a flat ~26 GB GPU peak where the resident
feeder OOMd at P>=768). So single-GPU extract-f2 no longer "tops out between 250
and 500 pops" — it completes 700 today and scales toward ~2500 via auto-select or
`--tier disk`; multi-GPU tiered sharding remains the follow-on (parked).

\*The 250-pop maxmiss-0.5 run succeeds (rc=0) but is already near the 32 GB wall;
peak VRAM was sampled at ~16 GiB by the original sweep and ~31 GiB on the verifier
re-run (finer sampling caught a higher transient allocation spike) — either way it
confirms P=250 with a full SNP set is the practical single-GPU ceiling.

**Peak VRAM vs 32 GiB.** Heaviest measured run = the 700-pop STREAMED (tier=host)
maxmiss-0.5 run at **25,368 MiB of 32,607 (~78 %)** — and crucially this peak is
INDEPENDENT of M (the SNP-tile input streaming holds the GPU footprint flat as M
and P grow). The OLD resident-only path's 24,518 MiB at the 700p decode was the
in-VRAM packed genotype matrix for 14,057 individuals × 1.23 M SNPs, after which
its `7·P·M` FP64 feeder OOMd; the streamed path never allocates that feeder.
Idle baseline 2 MiB. The maxmiss-0 sweep peaks well under 32 GB at every P (the SNP
collapse keeps the feeder buffers tiny); the OLD 32 GB feeder wall bound only the
resident path with a full SNP set at large P — and is now lifted by streaming.

*Independently verified on box5090 (1x RTX 5090, sm_120, `--device 0`), Release
(`build-rel`), REAL 1240K, no-hash. Spot re-runs reproduced exactly: P=250 maxmiss
0 (10.61 s, 8,557 SNPs, 645 blocks, f2.bin 645,002,644 B, `source_hash_computed:
false`), P=500 maxmiss 0 (16.71 s, 3 SNPs, f2.bin 12,000,076 B), P=700 maxmiss 0
(RC=2 filter-empty, 22.60 s, peak 24,518 MiB), P=250 maxmiss 0.5 (rc=0, 17.07 s,
1,104,231 SNPs), P=500 maxmiss 0.5 (RC=5 OOM on the OLD resident-only path,
26.29 s, 17,544 MiB). The only deviation from the sweep agent's report is the
P=250 maxmiss-0.5 peak-VRAM sample (noted above); all rc/SNP/blocks/f2.bin/wall
figures matched within noise. NOTE (ROADMAP §6): those P>=500 maxmiss-0.5 OOMs were
on the un-streamed RESIDENT entry; with the M5 tiered/streamed path now wired into
`extract-f2`, P=700 maxmiss-0.5 COMPLETES single-GPU (tier=host auto, 57.6 s, peak
25,368 MiB, f2.bin 5.59 GB; `--tier disk` byte-identical) — see the streamed-tier
table above.*

---

## qpAdm fit — wall-clock

`qpadm --f2-dir DIR --target … --left … --right …`, wrapped in `/usr/bin/time -v`.

| case | f2 dir | left / right | jackknife | **wall** | User | Sys | MaxRSS |
|---|---|---|---:|---:|---:|---:|---:|
| (a) single 2-way | f2_15 | 2 / 7 | 0 | **1.11 s** | 0.10 | 1.00 | 615 MB |
| (b) single 2-way | f2_60 | 2 / 6 | 0 | **1.07 s** | 0.10 | 0.97 | — |
| (c) heavy 4-way  | f2_60 | 4 / 12 | 2 | **3.67 s** | 2.69 | 0.98 | 632 MB |

**Verifier re-runs (f2_verify15, independent):**
- single 2-way, jackknife=0: **1.16 s** wall, User 0.11, Sys 1.05, MaxRSS 615 MB.
  Matches (a) within ~5%; RSS exact.
- heavy 4-way / 9-right, jackknife=2: **1.80 s** wall, **User 0.87 s**,
  Sys 0.93, MaxRSS 615 MB. Lighter than the sweep's (c) (4-way / **12**-right,
  more popdrop patterns) — the larger right set + extra popdrop subsets are what
  drive the sweep's User 2.69 s. The directional claim (jackknife=2 SE + popdrop
  is the GPU-compute-heavy path: User 0.10 -> 0.87 -> 2.69 s as the model grows)
  is confirmed; the absolute number is model-size dependent.

**Sensible results (parity sanity):** Czechia_EBA_CordedWare fits ~77/23
Yamnaya/Anatolia (z=95/29, feasible, status ok) — the sweep's 82/18 used a
different 7-outgroup right set; both are archaeologically correct (CordedWare is
~75–82% steppe). England_BellBeaker fits ~46% Yamnaya + 23% Anatolia + 36% WHG
with a small negative Iran term, p=0.36, feasible — textbook Bell Beaker.

**Floor:** single-fit wall is dominated by a ~1.0 s FIXED CUDA-context-init +
f2.bin-load floor (User is only ~0.10 s, Sys ~1.0 s). Pure fit compute is
~0.10 s with no SE. GPU work only becomes wall-visible with jackknife=2
(711-block SE + popdrop).

---

## Rotation — the batched engine EXISTS (correcting the original sweep)

**Correction.** The original sweep said "qpadm-rotate not implemented / rotation
runs at ~1 model/sec". That conflated two different things and was wrong about the
engine:

1. **The CLI subcommand `steppe qpadm-rotate` is a scaffold** — it parses and
   validates the flag set, then exits with
   `not yet implemented (M(cli-0) scaffold)`. True, but it is *only* the CLI
   wiring that is unbuilt.
2. **The batched rotation ENGINE is real and golden-gated.**
   `run_qpadm_search` (`src/core/qpadm/model_search.cpp:253`) dispatches to
   `CudaBackend::fit_models_batched` — f2 **RESIDENT in VRAM**,
   `cublasDgemmStridedBatched` + cuSOLVER `*Batched` + a model-batched fit
   kernel — and is gated against `golden_rot.json` (84 models) by
   `tests/reference/test_qpadm_rotation.cu` (PASSES single-GPU, batched
   dispatches=2).
3. **The "~1 model/sec" was a MEASUREMENT ARTIFACT**, not the engine's rate. The
   sweep looped the *CLI* `qpadm` subcommand once per model, so each model re-paid
   the full CUDA-context init + f2.bin load PER PROCESS (~0.85–1.0 s of
   process-spawn overhead; actual fit compute is a fraction of that). That rate
   measures process spawning, not rotation.

### REAL batched throughput — SINGLE-GPU, resident 1240K f2

Measured with `bench_rotation_1240k` (a manual bench TU — see "Bench vehicle"
below — NOT a ctest gate). It reads the REAL 1240K `f2_60` dir (P=60, 711 blocks,
682,707 SNPs) into VRAM **once** (`f2_device != null`), builds **781 REAL models**
(1 target + a 12-source pool, C(pool, 2..4), 6 outgroups, all over the SAME
resident f2), and times `run_qpadm_search` BATCHED on **device 0** (median of 3
timed iters after 1 warm-up):

| jackknife policy | wall (median) | **models/sec** | per-model | batched dispatches |
|---|---:|---:|---:|---:|
| `None` (point estimate)         | 272 ms | **2866** | 0.349 ms | 12 (<< 781) |
| `FeasibleOnly` (LOO SE on the 21 feasible survivors) | ~276 ms | **~2830** | 0.353 ms | 12 |
| `All` (full 711-block LOO SE, all 781) | 626 ms | **1247** | 0.802 ms | 12 |

All 781 returned `Ok`, 21 feasible, `precision_tag = EmulatedFp64`. **12 batched
dispatches for 781 models** proves the GPU-batched path (`fit_models_batched`),
NOT a per-model host loop. **Independently re-confirmed** by the verifier: jk=0
**2866 models/sec**, jk=2 (full LOO SE) **1247 models/sec** — both within noise of
the re-measure agent (2903 / 1235).

**So the REAL single-GPU batched rotation is ~2866 models/sec at the point
estimate (~2866x the bogus CLI-loop ~1/sec), and still ~1247 models/sec even with
the full 711-block LOO SE on every model** (~1247x). `FeasibleOnly` — the
production two-pass policy — is nearly free vs the point estimate because only the
21/781 feasible survivors pay the LOO SE.

**Shape, not f2 size, drives per-model compute.** Cross-checked on `f2_30`
(P=30): identical ~2900 models/sec at the same right/pool size — per-model compute
is MODEL-SHAPE-bound (nr, nl, jackknife policy), not f2-P-bound. A smaller right
set (f2_15, right=3) hits ~9000 models/sec.

### Bench vehicle

`tests/reference/bench_rotation_1240k.cu` (+ one `add_executable` in
`tests/CMakeLists.txt`, patterned off `bench_f2_multigpu`; links
`steppe::core/device/core_internal/api/warnings`, `CUDA_SEPARABLE_COMPILATION
ON`). **No product code changed.** REASON: the existing `test_qpadm_rotation.cu`
large-N/throughput path loads only the committed tiny fixture (`f2_rot.bin`,
9 pops) and cannot target a 1240K f2 dir. The new bench reads a real f2_blocks
dir's `f2.bin` (STPF2BK1, via `device/f2_disk_format.hpp` — the SAME on-disk
layout the CLI `read_f2_dir` consumes), uploads it RESIDENT to device 0, and times
the batched search. It is a MANUAL bench (no per-model golden; the accuracy gate
remains `test_qpadm_rotation`, which still PASSES single-GPU). Build confirmed in
`build-rel` (Release). LIMITS: throughput is for the model shapes shown; not a
ctest gate.

---

## Peak VRAM

Polled with `nvidia-smi --query-gpu=index,memory.used --format=csv -l 1`. These
were captured during the original `--device auto` runs (historical); the supported
single-GPU path uses one GPU's share. The point stands: peaks are tiny vs 32 GiB.

**extract-f2 (heaviest = SET60, 60 pops):** steady ~554 MiB during the `.geno`
decode; brief peak ~1750 MiB during the f2 compute/combine (the `-l 1` cadence
undersamples the sub-second f2-combine spike, so 1.75 GiB is the right ceiling).

**qpadm fit (heaviest = f2_60, 4-way, 12 right, jackknife=2):** ~668 MiB — a
single-GPU compute shape at these model sizes.

Both peaks (1.75 GiB extract, 0.67 GiB fit) are TINY vs 32 GiB per 5090. At
60 pops / 1.2M SNPs the box is nowhere near a VRAM wall — and the 781-model
resident rotation above adds only the f2 (40.96 MB) + small per-model batched
buffers, still far under the wall.

---

## Where the time goes

1. **extract-f2 was hash-bound, NOT decode-bound (CORRECTED, ROADMAP §6).** The
   old flat ~40–44 s was a **byte-at-a-time provenance SHA-256** of the whole
   6.7 GB `.geno` (~37 s of ~41 s at ~190 MB/s scalar) — a `meta.json` provenance
   value, not the genotype math. With that hash now OFF by default the SET60 wall
   is **~5.8 s** (`--hash` opt-in adds only ~1.9 s, overlapped on a background
   thread by bulk SHA-NI). The genuine per-individual seek+gather + TGENO decode +
   f2 GEMM are each well under a second; f2.bin grows 288 KB -> 41 MB as P^2 but
   adds little wall. SINGLE-GPU is the supported path; the 2nd GPU is parked and,
   per the multi-GPU root cause, would cost a 2nd ~7 GB f2_blocks D2H. (If you do
   want to speed it further, attack the genuine TGENO seek+gather+decode — NOT the
   hash, which is now opt-in and overlapped.)

2. **qpadm single-fit is floor-bound.** ~1.0 s fixed CUDA-context-init +
   f2.bin-load; pure fit compute is ~0.10 s (no SE). GPU compute only becomes
   visible with jackknife=2 SE + popdrop (User scales 0.10 -> 0.87 -> 2.69 s
   with model size). Fit compute scales with #left x #right and, for SE,
   x ~711 blocks x #popdrop patterns.

3. **Rotation is fast and real — only the CLI subcommand is a scaffold.** The
   batched engine `run_qpadm_search` runs ~2866 models/sec single-GPU at the point
   estimate (~1247 models/sec with full LOO SE) over a RESIDENT 1240K f2, paying
   the CUDA context + f2 load ONCE. The earlier ~1 model/sec was the CLI loop
   re-paying process spawn per model — a measurement artifact, not the engine.

---

## Limits / failures

- **NO failures, NO OOM** at full 1240K size in the original sweep set. The OLD
  resident-only extract-f2 DID OOM at P>=500 with a full ~1.1 M-SNP set (the
  `7·P·M` FP64 feeder exceeded 32 GB); **with the M5 tiered/streamed path wired in
  (ROADMAP §6), single-GPU extract-f2 now COMPLETES 700 pops** (tier=host
  auto-selected, 57.6 s, peak 25,368 MiB, f2.bin 5.59 GB; `--tier disk`
  byte-identical) and scales toward ~2500 (c65179f). Every fit and the 781-model
  batched rotation exited rc=0 with sensible results.
- **No silently-dropped coverage** beyond the intended `--maxmiss 0` joint-
  completeness filter (SNPs-kept correctly shrinks as pops are added).
- **Only the CLI `qpadm-rotate` subcommand is a scaffold** (M(cli-0)); the batched
  rotation ENGINE (`run_qpadm_search`) is real, golden-gated, and ~2866 models/sec
  single-GPU. The original "rotation not implemented / ~1 model/sec" was a
  CLI-scaffold + process-spawn artifact — corrected in this revision.
- **MULTI-GPU IS PARKED, not benchmarked.** The 2nd GPU is slower (extract-f2
  f2_blocks data-bounce, a 2nd ~7 GB D2H; the fit is a single-GPU compute shape).
  Root-caused, fix deferred. All numbers here are SINGLE-GPU (`--device 0`).
- VRAM peaks are tiny (<=1.75 GiB extract, 0.67 GiB fit vs 32 GiB) — abundant
  headroom for the large batched workloads the design targets.

---

*Independently verified on box5090 (1x RTX 5090, sm_120, `--device 0`), Release
build (`build-rel`), REAL AADR v66 1240K panel. SINGLE-GPU throughout; multi-GPU
parked and not benchmarked. The original sweep's rotation conclusion ("not
implemented / ~1 model/sec") is corrected: the batched `run_qpadm_search` engine
is real and golden-gated, the REAL single-GPU batched throughput is ~2866
models/sec (point estimate) / ~1247 models/sec (full LOO SE), and the ~1/sec was a
CLI process-spawn artifact. extract-f2 is re-reported single-GPU (43.2 s) as the
supported path.*

---

## qpadm-rotate scaling vs pool size (single-GPU)

**What this measures.** How the wired `qpadm-rotate` CLI (→ the batched
`run_qpadm_search` engine, `src/app/cmd_rotate.cpp:184`) scales as the **candidate
pool grows**, holding the target, the right set, and the subset sizes fixed. A
pool of `N` pops swept over subset sizes `[1,3]` enumerates

```
#models = N + C(N,2) + C(N,3)     (C(N,3) dominates at scale)
```

This was VERIFIED EXACT at every measured N (2625, 20875, 166750, 562625,
1333500 — matched bit-for-bit by the emitted CSV row counts).

**Fixed shape for the whole sweep.** Target = `England_BellBeaker` (28 ind);
right(6) = `Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana`
(all exact-label-matched, all >=5 ind); subsets `[1,3]`; `blgsize 0.05`;
`--maxmiss 0.5`; **713 jackknife blocks**; `--precision emu40` (default,
40-mantissa-bit emulated FP64). Pools are **nested name-sorted prefixes** of the
**919** candidate pops (the 926 pops with >=5 ind, minus target + 6 right), so
every smaller pool is a prefix of the larger — a clean monotone sweep.

**Two passes.** `--jackknife 0` = the raw point estimate (memory-light, scales to
the largest pools). `--jackknife 1` = the feasible-only LOO-SE pass-2 (much
hungrier — see the VRAM note). Per-model cost is **shape-bound** (n-left x n-right
x blocks), not pool-size-bound; pool size only sets the model *count*.

### Measured — jackknife 0 (point estimate)

REAL v66 1240K, single-GPU `--device 0`, Release. `models/sec = #models / wall`.

| pool N | #models | (N + C(N,2) + C(N,3)) | wall | models/sec | feasible |
|---:|---:|---|---:|---:|---:|
| 25  | 2,625      | 25 + 300 + 2,300       | 1.078 s | 2,436  | 440     |
| 50  | 20,875     | 50 + 1,225 + 19,600    | 1.201 s | 17,386 | 6,571   |
| 100 | 166,750    | 100 + 4,950 + 161,700  | 3.631 s | 45,922 | 47,386  |
| 150 | 562,625    | 150 + 11,175 + 551,300 | 9.816 s | 57,319 | 112,385 |
| 200 | 1,333,500  | 200 + 19,900 + 1,313,400 | 22.14 s | 60,226 | 241,922 |

### Measured — jackknife 1 (feasible-only SE pass-2)

**UPDATE (ROADMAP §6 — the jk1/jk2 SE pass-2 VRAM-budget fix, `cuda_backend.cu`
`fit_one_bucket`).** The OLD ~pool_50 jk1 OOM (the table's `—` rows below the line)
is **FIXED**: the SE pass-2 LOO arenas (`dLooS` m·nb + `dQinvS` m² + `dWmatS` nb·nl
+ `dSeS` nl doubles + `dSurv` int per model) coexist with the still-live pass-1
buffers in one flat RAII scope inside `fit_chunk`, but the per-chunk budgeter
counted ONLY pass-1, so jk≥1 over-committed and `cudaMalloc` OOM'd above ~pool_50.
The fix folds the pass-2 footprint into the EXISTING per-model budget term (JK-gated
to 0 for `None` so jk0 is bit-for-bit unchanged) — a smaller `B_max` under jk≥1
makes the EXISTING chunk loop tile BOTH passes. Chunk width moves no bits
(architecture.md §12; the LOO SE is per-model + chunk-independent), so **pool_50 jk1
is byte-identical pre/post the fix** (verified `cmp`/md5). jk1 now scales single-GPU
to **pool_200** (1.33M models). Re-measured below on box5090 (1× RTX 5090, 32 GB),
Release, REAL v66 1240K, `--device 0`, the `rot_pool200` resident f2 (P=207, 713
blocks, 1,123,319 SNPs):

| pool N | #models | wall | models/sec | peak VRAM | note (post-fix) |
|---:|---:|---:|---:|---:|---|
| 25 | 2,625   | 1.158 s | 2,267 | — | ok |
| 50 | 20,875  | 2.314 s | 9,022 | 29,148 MiB | ok; **byte-identical pre/post the fix** |
| 100 | 166,750 | 12.55 s | 13,287 | **30,014 MiB** | **COMPLETES (was OOM)** |
| 150 | 562,625 | 30.64 s | 18,361 | **30,170 MiB** | **COMPLETES (was OOM)** |
| 200 | 1,333,500 | 66.20 s | 20,144 | **30,062 MiB** | **COMPLETES (was OOM)** |

(Pre-fix the N≥60 jk1 rows OOM'd with `cudaErrorMemoryAllocation` at
`device_buffer.cuh:74` — re-confirmed on the pre-fix binary at pool_100 AND pool_200;
the post-fix peak sits ~2.5 GB below the 32,607 MiB ceiling at every pool, vs the
pre-fix OOM. GPU returned to 2 MiB after every run — no leak.)

**The OLD jk1 ceiling (superseded).** Before the fix the jk1 single-GPU wall landed
between N=50 and N=60 — a VRAM ceiling (32 GB), NOT the time wall. The SE pass-2
(LOO over survivors) allocated a second arena set while the pass-1 buffers were
still live, and the per-model VRAM budgeter in `cuda_backend.cu` `fit_one_bucket`
did **not** account for that pass-2 footprint. That budgeting bug is now fixed (see
the UPDATE above); the jk1 SE path scales to **pool_200** single-GPU, the same
ceiling as jk0. jk0 at every N=60/75/100/150/200 also succeeds.

### The rate does NOT hold flat — it RISES then CONVERGES

Contrary to the "rate is shape-bound so it holds across pool size" prior, the
small-pool rate is **fixed-per-dispatch-overhead-bound**: a few thousand models
cannot amortize the kernel-launch / H2D / cuSOLVER-batched-setup cost, so the rate
climbs steeply with N and **plateaus** in the large-N limit:

```
jk0:  N=25  -> 2,436 /s
      N=50  -> 17,386 /s
      N=100 -> 45,922 /s
      N=150 -> 57,319 /s      \  plateau: N=150 vs N=200
      N=200 -> 60,226 /s      /  differ only ~5% -> CONVERGED
```

**Converged jk0 rate ≈ 57,000–60,000 models/sec.** Extrapolate with the
*converged* rate (`#models / 60k`), NOT the small-pool rate — using a small-pool
rate would over-estimate the wall by >20x. The converged jk1 rate is lower
(9,022 /s at N=50) but is **moot for large pools — jk1 OOMs single-GPU before
they are reached.**

### f2-build cost (the f2 cache, separate from the rotation)

`extract-f2 --device 0 --blgsize 0.05 --maxmiss 0.5 --auto-only` over
{target + pool + 6 right}. **All tiers = RESIDENT** — the streamed host/disk tier
did NOT auto-engage even at 207 pops (still fits resident in 32 GB). Build time is
~linear in P (input-pass bound); `f2.bin` grows ~P^2 (16,384 B per block per
pop-pair slab).

| pool N | P = N+7 | SNPs kept | extract-f2 wall | f2.bin |
|---:|---:|---:|---:|---:|
| 25  | 32  | 1,089,721 | 3.80 s  | 11.7 MB  |
| 50  | 57  | 1,103,077 | 5.36 s  | 37.1 MB  |
| 100 | 107 | 1,118,938 | 8.53 s  | 130.6 MB |
| 150 | 157 | 1,118,608 | 11.34 s | 281.2 MB |
| 200 | 207 | 1,123,319 | 14.44 s | 488.8 MB |

The `f2.bin` pairs-quadratic model (`#pairs = P(P+1)/2`) reproduces the measured
sizes to <2.6% across the whole range (P^2 confirmed).

### Peak VRAM (single-GPU, 32,607 MiB ceiling)

- **jk0 pool_200** (largest, 1.33M models): **PEAK 31,578 MiB = 96.8%** of the
  32 GB ceiling. pool_200 is essentially the largest jk0 pool that fits resident:
  resident f2 (467 MB) + per-chunk arenas (`dX/dLoo/dXtau ≈ 3·B·m·nb`) brush the
  ceiling. The `fit_one_bucket` budgeter *does* chunk the bucket for jk0.
- **jk1 pool_50**: PEAK 29,148 MiB (byte-identical pre/post the SE-budget fix).
- **jk1 now scales to pool_200** (post the SE pass-2 budget fix): PEAK **30,014 /
  30,170 / 30,062 MiB** at pool_100 / 150 / 200 (all ~2.5 GB below the 32,607 MiB
  ceiling). The OLD **pool_60 jk1 OOM** (the uncounted pass-2 SE arenas) is FIXED —
  the budgeter now counts the pass-2 footprint and the existing chunk loop tiles
  both passes (see the jackknife-1 section above).
- **No VRAM leak** — both 5090s returned to 2 MiB after every run.

### Extrapolation to pool = 500 / 1000 / whole-set

Using the **measured converged jk0 rate** (bracket 57,319–60,226 /s; headline
figure quotes the N=200 plateau, 60,226 /s). The extract-f2 build time is the
linear-in-P fit `t ≈ 0.060·P + 1.9 s`; the `f2.bin` size is the P^2 model.

| pool N | #models | jk0 rotation wall | extract-f2 build | f2.bin / tier |
|---:|---:|---:|---:|---|
| 500 | 20,833,750 | **~5.8–6.1 min** | ~33 s | ~2.9 GB (resident) |
| 1000 | 166,667,500 | **~46–48 min** | ~63 s | ~11.5 GB (resident borderline → host/streamed tier) |
| whole-set (919) | 129,359,359 | **~36–38 min** | ~58 s | ~9.8 GB (resident borderline → host/streamed tier) |

(whole-set pool = 919 = the 926 pops with >=5 ind, minus target + 6 right;
`C(919,3)` dominates.)

**HONESTY — these are capability / stress projections, NOT a typical analysis.**
Real qpAdm rotations curate the pool to **dozens** of pops (subset `[1,3]` over a
hand-picked source set), which lands in the **seconds** regime (N=25 = 1.1 s,
N=50 = 1.2 s above). The 500/1000/whole-set numbers project what the *engine* can
chew (10^7–10^8 models, tens of minutes single-GPU) if you point it at the entire
panel — a throughput-ceiling stress test, not a research workflow. Both regimes
are real; quote the one that matches the question.

**Caveats carried forward.** (1) ~~jk1 (SE) OOMs single-GPU above ~pool_50~~ —
**FIXED** (ROADMAP §6): the chunk budgeter now counts the SE pass-2 footprint, so
jk1 scales to pool_200 single-GPU (peaks ~30 GB, ~2.5 GB under the ceiling), the
same envelope as jk0; the fix is parity-neutral (pool_50 jk1 byte-identical
pre/post). (2) The projections assume the f2 stays
resident; at pool≈1000 the ~11.5 GB `f2.bin` is near the point where the
host/streamed f2 tier engages, which adds I/O the resident-only measurements don't
include. (3) Single-GPU only; multi-GPU is PARKED.

---

*Independently verified (this doc, this section) on box5090 (2x RTX 5090 sm_120;
SINGLE-GPU `--device 0`, the 2nd GPU PARKED), Release build (`build-rel`,
`CMAKE_BUILD_TYPE:STRING=Release`, `-DSTEPPE_BUILD_CLI=ON`, Ninja), REAL AADR v66
1240K panel (1,233,013 SNPs / 23,089 ind / 926 pops >=5 ind — re-counted). The
`qpadm-rotate` CLI is wired to the real batched `run_qpadm_search` engine (NOT a
scaffold). Model-count formula `N + C(N,2) + C(N,3)` verified exact at every N.
Two pool points were RE-RUN fresh for this verification — pool_25 jk0 (1.047 s,
2,625 models, feasible 440) and pool_50 jk0 (1.364 s, 20,875 models, feasible
6,571) — reproducing the reported rate regime and bit-identical feasible counts.
The converged jk0 rate is ~57–60k models/sec; the small-pool rate is dispatch-
overhead-bound and must NOT be used for extrapolation. NO code changes were made.*
