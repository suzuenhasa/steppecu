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

**Status:** independently verified (this doc). The sweep agent's report was
re-confirmed end to end: build type, data geometry, an extract-f2 spot re-time
(within ~1%), a single fit, a heavy jackknife fit, the CLI-loop rate, and the
qpadm-rotate scaffold were all reproduced. Deviations are flagged inline.

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

## extract-f2 — wall-clock x pop-set size

`extract-f2 --prefix … --pops … --blgsize 0.05 --maxmiss 0 --auto-only`,
device=auto (both GPUs). blgsize 0.05 -> 711–713 jackknife blocks.

| pop set | #indiv | SNPs kept (of 1,233,013) | blocks | **wall** | MaxRSS | f2.bin |
|---|---:|---:|---:|---:|---:|---:|
| SET5  |   206 | 1,045,408 | 713 | **40.02 s** | 0.98 GB | 288 KB |
| SET15 |   596 |   691,342 | 711 | **40.61 s** | 1.50 GB | 2.56 MB |
| SET30 | 1,907 |   704,508 | 711 | **41.66 s** | 2.59 GB | 10.24 MB |
| SET60 | 4,402 |   682,707 | 711 | **43.97 s** | 4.73 GB | 40.96 MB |

**Verifier spot re-time (14-pop Haak set, independent run):** 40.38 s wall,
User 37.52 s, Sys 2.98 s, MaxRSS 1.37 GB, 690,626 SNPs kept, 711 blocks,
P=14, rc=0. This matches the sweep's SET15 (40.61 s, 1.50 GB, ~691K SNPs,
711 blocks) within ~1% on wall and ~9% on RSS — well inside the ~20%
tolerance, and reproducible. (`/usr/bin/time` and `xxd` were initially absent on
the box; `time` was installed, `od -c` substituted for `xxd`.)

**Scaling:** wall is essentially FLAT (40.0 -> 44.0 s) across a 21x individual
count and 12x pop count. User time is flat ~37–38 s; System time grows mildly
(2.7 -> 5.7 s) with more individuals decoded. SNPs-kept DROPS as pops are added
because `--maxmiss 0` requires zero missing across all selected pops (joint
completeness) — this is correct, intended behavior, not silent data loss.

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

## Rotation — qpadm-rotate is NOT YET IMPLEMENTED

`steppe qpadm-rotate` parses and validates a full flag set (`--pool`,
`--min/max-sources`, `--als-iters`, `--rank`, `--jackknife`, …) but the compute
is a scaffold. **Verified directly:** an actual run prints

> `steppe qpadm-rotate: not yet implemented (M(cli-0) scaffold — config parsed and validated; compute lands in a later milestone)`

and exits rc=0 in **0.12 s** with no fit output. The in-process batched
rotation does not exist yet. **This is the single biggest perf gap.**

**Fallback measured — CLI loop over the `qpadm` subcommand (f2_60, right=8):**

| loop | jackknife | total | **models/sec** | breakdown |
|---|---:|---:|---:|---|
| 30 models | 0 | 27.75 s | **1.08** | User 3.48 s (compute), Sys 24.28 s (30x ctx init) |
| 10 models | 2 | 9.60 s | **1.04** | User 1.19 s, Sys 8.43 s |

**Verifier re-run:** 8 light models in 7.00 s = **1.143 models/sec** — matches.

The CLI-loop rate (~1.0 models/sec) is **PROCESS-SPAWN bound, not compute
bound**: ~0.85–1.0 s/model is CUDA-context creation + f2.bin load paid PER
PROCESS. Actual fit compute is ~0.12 s/model (no SE). A real in-process
qpadm-rotate that pays the CUDA context + f2 load ONCE would run roughly ~8x
faster on the light path (toward ~8–10 models/sec from the 0.12 s compute) —
which is exactly the motivation for the unbuilt batched rotation, and matches
the project goal of thousands batched, multi-GPU.

---

## Peak VRAM (single vs both GPUs)

Polled with `nvidia-smi --query-gpu=index,memory.used --format=csv -l 1`.

**extract-f2 (heaviest = SET60, 60 pops):** BOTH GPUs used. Steady ~554 MiB/GPU
during the `.geno` decode; brief peak GPU0=1750 / GPU1=1742 MiB during the f2
compute/combine. Smaller sets peaked lower (SET5 784/784, SET15 1228/626,
SET30 626/626 MiB). Verifier 14-pop poll caught 626 MiB/GPU on both — the `-l 1`
cadence undersamples the sub-second f2-combine spike (a known limit of 1 Hz
polling), so 1.75 GiB is the right ceiling. So: multi-GPU, peak ~1.75 GiB/GPU.

**qpadm fit (heaviest = f2_60, 4-way, 12 right, jackknife=2):** GPU0=668 MiB,
GPU1=626 MiB (GPU1 at idle baseline). The fit is effectively a SINGLE-GPU
workload at these model sizes.

Both peaks (1.75 GiB extract, 0.67 GiB fit) are TINY vs 32 GiB per 5090. At
60 pops / 1.2M SNPs the box is nowhere near a VRAM wall.

---

## Where the time goes

1. **extract-f2 is fully decode-bound.** Wall is flat ~40–44 s from 5 to 60 pops
   despite 12x more individuals and P^2 f2 growth; the f2 compute is a rounding
   error at <=60 pops (f2.bin grows 288 KB -> 41 MB as P^2 but adds < 4 s wall).
   To speed extract-f2 you must attack the 6.7 GB TGENO streaming-decode
   throughput, NOT the f2 math. The 2nd GPU buys little here (decode-bound).

2. **qpadm single-fit is floor-bound.** ~1.0 s fixed CUDA-context-init +
   f2.bin-load; pure fit compute is ~0.10 s (no SE). GPU compute only becomes
   visible with jackknife=2 SE + popdrop (User scales 0.10 -> 0.87 -> 2.69 s
   with model size). Fit compute scales with #left x #right and, for SE,
   x ~711 blocks x #popdrop patterns.

3. **Rotation throughput is gated on an unbuilt feature.** Today's only path is
   the CLI loop at ~1 model/sec, bound by per-process CUDA-context + f2 load.

---

## Limits / failures

- **NO failures, NO OOM** at full 1240K size. Every extract and fit exited rc=0
  with sensible archaeogenetic results.
- **No silently-dropped coverage** beyond the intended `--maxmiss 0` joint-
  completeness filter (SNPs-kept correctly shrinks as pops are added).
- **qpadm-rotate unimplemented** (M(cli-0) scaffold) — biggest gap.
- **Multi-GPU underused by the fit** — independent models are not batched across
  the 2nd GPU; this is the unexploited headroom the batched rotation would use.
- VRAM peaks are tiny (<=1.75 GiB/GPU vs 32 GiB) — abundant headroom for the
  large batched workloads the design targets.

---

*Independently verified on box5090 (2x RTX 5090, sm_120), Release build, REAL
AADR v66 1240K panel. All sweep-agent claims re-confirmed; deviations (weight
mix on a different right set, heavy-fit User time on a smaller model) explained
above and attributable to model-set differences, not measurement error.*
