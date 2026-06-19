# ADMIXTOOLS 2 vs steppe — honest timing comparison (f2 precompute + qpAdm fit)

**Scope.** This compares the wall-clock of steppe's **f2 PRECOMPUTE** (the `extract_f2`
equivalent: all-pairs `f2_blocks[P×P×n_block]` FP64 leave-one-block-out jackknife tensor over
~584k SNPs) against ADMIXTOOLS 2 (`extract_f2` / `f2_from_geno`, Maier et al. 2023, eLife e85492)
and classic ADMIXTOOLS (Patterson). It also covers the qpAdm **FIT** phase. Steppe's fit phase is
**not built yet**, so no steppe fit number is claimed.

**Honesty posture.** Every number is tagged **[VERIFIED + source]** or **[ESTIMATE + assumptions]**.
The single biggest honesty flag is stated up front and carried throughout: **the AT2 side has NO
published extract_f2 wall-clock vs P**, so the AT2 number for this exact workload is an
**explicitly-labeled complexity-based ESTIMATE**, not a measurement. The steppe side is now
measured on both axes: **P=2000 = 15.1 s on 2× RTX PRO 6000 [VERIFIED]**, and after M5 out-of-core
streaming landed (`176a07d` tiered output + `c65179f` SNP-tile input), **full-autosome P=2500
COMPLETES on a single 32 GB RTX 5090 in ~51.5 s [VERIFIED, post-M5]** (76 GB result streamed, GPU
peak ~26 GB, parity bit-identical). The old ceiling — "P=2500 OOMs on every path" — was true only
**before M5** (see `docs/cleanup/m4.5/scaling-sweep.md`, now SUPERSEDED) and is no longer the state.

---

## 0. Provenance flags you must read before trusting any single number

1. **Two measured steppe anchors, two boxes — both repo-grounded.**
   (a) The committed full-autosome sweep on **2× RTX PRO 6000 Blackwell (95.6 GB/GPU, 169 GB host RAM)**
   gives **P=2000 = 15.1 s** but, *before M5*, OOM'd at P=2500 (the 75.7 GB result tensor + a 37.9 GB
   resident partial ≈ 114 GB ≫ 95.6 GB) — see `docs/cleanup/m4.5/scaling-sweep.md` (now **SUPERSEDED**).
   (b) **After M5 out-of-core streaming** (`176a07d` tiered output + `c65179f` SNP-tile input), the
   per-P single-5090 streamed sweep (P=512 ~3.6 s host / ~0.67 s device-resident; P=1000 ~10.4 s;
   P=1500 ~20.2 s; P=2000 ~34.0 s; **P=2500 ~51.5 s COMPLETES**) is the measured post-M5 result on a
   single 32 GB 5090 — the result is streamed out (76 GB) and GPU peak is bounded at ~26 GB, parity
   bit-identical. The pre-M5 "a single 32 GB card cannot hold the 75.7 GB P=2500 result" was correct
   only as a statement about the *resident* result; M5 streaming makes the on-device footprint
   `O(P·tile + P²)`, independent of M, so the full result never has to be co-resident.
   → Both anchors are now measured: **P=2000 = 15.1 s (PRO 6000)** and **P=2500 = 51.5 s (one 5090,
   streamed, post-M5)**.

2. **AT2 has no published `extract_f2` wall-clock-vs-P benchmark.** The AT2 paper's one performance
   figure (Appendix 1—figure 1a) benchmarks the **query/fit** side (x-axis = "number of f4-statistics"),
   not extract_f2 vs P. So the AT2 precompute time for this workload is **ESTIMATED**.

---

## 1. THE APPLES-TO-APPLES — f2 precompute (the `extract_f2` equivalent)

**Same workload both sides:** all-pairs f2 for P populations over ~584k SNPs, leave-one-block-out
jackknife with n_block≈757; output is the `f2_blocks[P×P×n_block]` FP64 tensor.

### Steppe — measured (repo-verified, 2× RTX PRO 6000, EmulatedFp64{40})

`docs/cleanup/m4.5/scaling-sweep.md:9-19`, median of 3 + warm-up, Release. G2 = device-resident 2-GPU.

| P    | result (2·P²·n_block·8) | single-GPU (G1) | 2-GPU device-resident | 2-GPU host-staged |
|-----:|------------------------:|----------------:|----------------------:|------------------:|
| 256  | 0.79 GB | 458.2 ms | **354.9 ms** | 463.2 ms |
| 512  | 3.18 GB | 1213.3 ms | **1045.2 ms** | 1478.9 ms |
| 768  | 7.14 GB | 2334.2 ms | **2117.0 ms** | 3086.9 ms |
| 1024 | 12.70 GB | 3802.4 ms | **3569.7 ms** | 5297.9 ms |
| 1536 | 28.58 GB | OOM (pre-M5) | 7843.2 ms | 11746.2 ms |
| 2000 | 48.45 GB | OOM (pre-M5) | **15099.5 ms** | 21828.6 ms |
| 2500 | 75.70 GB | (pre-M5 OOM; **completes post-M5 via streaming** — see below) | OOM (pre-M5) | OOM (pre-M5) |

**Post-M5 single-5090 streamed sweep [VERIFIED, post-M5]** (one 32 GB RTX 5090, M5 out-of-core,
`docs/cleanup/m5/00-results.md`): P=512 ~3.6 s, P=1000 ~10.4 s, P=1500 ~20.2 s, P=2000 ~34.0 s,
**P=2500 ~51.5 s** (76 GB result streamed out, GPU peak ~26 GB bounded, parity bit-identical).

**[VERIFIED, HIGH]** Steppe does P=2000 / 584k SNPs / all-pairs f2 in **15.1 s** (2-GPU device-resident,
PRO 6000). **P=2500 fits and completes in ~51.5 s on a single 32 GB 5090 via M5 streaming** — the
in-core sweep above OOM'd at P=2500 *before M5* because it held the whole 75.7 GB result resident;
M5 removes that wall. **Two defensible anchors:** P=2000 = 15.1 s (PRO 6000) and P=2500 = 51.5 s
(one 5090, streamed). The OOM column above is the pre-M5 in-core ceiling, kept for the record.

### AT2 `extract_f2` — for the same P / 584k SNPs / n_block≈757

**No published wall-clock exists for this.** What IS published:
- **The one real timing anchor [VERIFIED quote]:** AT2 GitHub issue #7 (Benjamin Peter's parallel patch)
  — the parallel `n_cores` version *"sped up calculations from ~a day to 20min"* **for his test case**.
  **No population / SNP / core count is stated**, so it is not directly comparable, but it establishes:
  (a) serial `extract_f2` at a large workload can be **~a day**; (b) it is embarrassingly parallel
  across pairs (≈72× from multicore here).
  Source: https://github.com/uqrmaie1/admixtools/issues/7
- **Compute complexity [VERIFIED]:** O(P²·M). The non-redundant f2 count is exactly **k²** for k
  populations (paper, Appendix 1—figure 1 caption: "k², (1/2)k³, (1/3)(k choose 4)" for f2/f3/f4).
  So extract_f2 is **quadratic in P, linear in SNPs** — identical scaling to steppe's P×P×n_block.

**ESTIMATE for the same workload (clearly an estimate, not a measurement):** anchoring to the
"~a day serial → 20 min parallel" issue-#7 point and the O(P²·M) shape, on a **typical multicore
server CPU (~16–64 cores)** and AT2's R/Rcpp path:
- P≈500 over ~584k SNPs: **~minutes to low tens of minutes**.
- P≈1000: **~tens of minutes to ~1 hour**.
- P≈2000–2500: **multiple hours**, increasingly dominated by the RAM/disk wall (§2), often forcing the
  chunked/disk-spill path that slows it further.

Confidence: **LOW on the absolute minutes** (no benchmark), **MEDIUM on the shape** (quadratic in P,
linear in SNPs, hours-not-seconds at thousands of pops — this is forced by O(P²·M) + the memory wall).

### The speedup factor — honestly bounded

Because the AT2 absolute time is an **ESTIMATE**, the ratio is an **order-of-magnitude band, not a
clean Nx**. The steppe side now has two measured anchors — **P=2000 = 15.1 s [VERIFIED, PRO 6000]**
and **P=2500 = 51.5 s [VERIFIED, post-M5, one 5090, streamed]**:

- **Defensible anchor (verified steppe vs estimated AT2):** at P≈2000 / 584k SNPs, steppe = 15.1 s.
  AT2 estimated at "multiple hours" (say ~1–6 h) → steppe is **~250× to ~1400× faster** for the
  precompute — an **order-of-magnitude band (≈10²–10³×), built on an ESTIMATED AT2 baseline.**
- At **P=2500 = 51.5 s [VERIFIED, post-M5]** the band is similar (AT2 multiple-hours / 51.5 s ≈
  10²–10³×). The *ratio* stays an estimate because the AT2 side has no measured number, **not**
  because the steppe number is uncertain.

**Do not quote a single "Nx" figure as fact.** The only honest statement is: **the GPU precompute is
plausibly 2–3 orders of magnitude faster wall-clock than AT2's CPU `extract_f2` at the thousands-of-pops
scale, but the AT2 baseline is an estimate (no measured AT2 number exists to pin the ratio exactly).**

---

## 2. RAM / DISK comparison

### AT2 `extract_f2` — RAM/disk-heavy by design [VERIFIED]

- **Default RAM:** keeps the **full allele-frequency matrix (all SNPs × all pops) in memory.**
  `maxmem` default = **8000 MB**; *"If the required amount of memory exceeds `maxmem`, allele frequency
  data will be split into blocks, and the computation will be performed separately on each block pair"*
  (docs note this is **not a precise cap**). **[VERIFIED, fetched 2026-06: extract_f2 reference page]**
  Source: https://uqrmaie1.github.io/admixtools/reference/extract_f2.html
- **Time↔RAM knob:** `cols_per_chunk` — *"Setting this to a positive integer makes the function slower,
  but requires less memory"* (splits the allele-freq matrix to disk). For very many pops, `afs_to_f2`
  exists explicitly *"for computing f2-statistics for a large number of populations ... too many to do
  everything in working memory."* **[VERIFIED]**
- **The authors' own memory wall [VERIFIED]:** the precompute-f2 approach *"only gives identical results
  in the absence of missing data, which limits its usefulness beyond a moderate number of populations"*
  (Appendix 1—figure 1 caption). I.e. AT2 itself flags that **thousands of pops** push past comfortable RAM.
- **Disk output [VERIFIED structure, ESTIMATED size]:** `extract_f2` writes the f2 blocks to `outdir`
  as **one file per population pair** (internal `write_f2` writes one pair at a time), later read by
  `f2_from_precomp`. File-count and total disk are both **O(P²)**. Size ≈ P²·n_block·8 bytes before
  compression; for n_block≈757: P=512 → ~1.6 GB, P=1000 → ~6 GB, P=2500 → ~38 GB (uncompressed,
  upper-triangle; AT2 stores per-pair files that compress, so realized disk is a constant-factor
  smaller but still grows as P²). **Note the specific "15,647 MB RAM / 313 MB disk" figure from early
  search snippets is UNVERIFIED — it is NOT in the fetched reference page; do not cite it.**

### Steppe — streams it / holds it resident on GPU [VERIFIED against repo]

- **Pre-M5:** steppe held the **full result tensor resident** (`2·P²·n_block·8`: f2 + Vpair). That
  resident footprint was the wall: P=2500 = **75.7 GB**, which OOM'd the 96 GB GPUs
  (`scaling-sweep.md:26-40`, now SUPERSEDED).
- **Post-M5 [VERIFIED]:** M5 out-of-core streaming (`176a07d` adaptive tiered output + `c65179f`
  SNP-tile input) **shipped**. The on-device footprint is now `O(P·tile + P²)`, independent of M, and
  the result goes to the fastest tier it FITS — VRAM-resident (small P) → host RAM → disk — auto-selected
  from runtime free VRAM/RAM. P=2500 completes on a single 32 GB 5090 with GPU peak bounded at ~26 GB
  (76 GB streamed out). Steppe now HAS the time-for-memory fallback (the disk tier is the streamable
  artifact); the pre-M5 "no such fallback yet" no longer holds.
- The pre-M5 host-materialization tail (whole-tensor D2H, ~4.15 GB/s) was the serial Amdahl tail that
  shrank the 2-GPU speedup 1.29×→1.07× as P grew (`why-d2h.md`, `architecture-audit.md`); device-resident
  output (`1f80c0c`) dissolved that tail for the in-VRAM case (~4.3× at P=512). **[VERIFIED]**

**The contrast.** AT2's constraint is **CPU RAM for the allele-freq matrix + O(P²) disk files** (it spills
to disk and slows down past a "moderate number of pops"). Steppe's *pre-M5* constraint was **GPU VRAM for
the resident result tensor** (it OOM'd past ~2000 pops on 96 GB). Both are O(P²) in storage; AT2 trades
time for RAM via on-disk chunking, and steppe now has the analogous fallback — **M5 adaptive tiered output
(VRAM → host RAM → disk) + SNP-tile input streaming**, which decouples the on-device footprint from both
M and the full result size.

---

## 3. THE FIT PHASE — qpAdm per-model + rotation

### Classic ADMIXTOOLS qpAdm — [VERIFIED, near-identical dataset]

CompPopGen 2019 workshop, classic command-line `qpAdm`/`qpWave` on EIGENSTRAT:
*"Running `qpWave` or `qpAdm` with this dataset takes 1-2 minutes, using a single core (no multithreading
is available)."* Dataset: **snps: 593124, indivs: 107, number of blocks: 711** — essentially identical
scale to steppe's AADR target (584,131 SNPs / 757 blocks). So **classic qpAdm ≈ 1–2 min per model,
single-threaded**, because each run re-reads the packed genotypes and recomputes f-stats.
**[VERIFIED, HIGH]** Source:
https://comppopgenworkshop2019.readthedocs.io/en/latest/contents/05_qpwave_qpadm/qpwave_qpadm.html

### AT2 `qpadm()` per-model — [VERIFIED qualitative; numeric ESTIMATE]

The two-step design is explicit (AT2 tutorial, fetched 2026-06): step 1 *"Computing f₂-statistics and
storing them on disk. This can be slow since it accesses the genotype data."*; step 2 *"Using
f₂-statistics to fit models. This is fast because f₂-statistics are very compact compared to genotype
data."* `qpadm_multi` *"runs multiple qpadm models, re-using f4-statistics where possible"* and supports
parallel evaluation. **No per-model wall-clock in seconds is published.**
- **ESTIMATE:** once f2 is precomputed, a qpadm fit reads a small f2 slice and does small linear algebra
  per block over ~757 blocks → **tens of milliseconds to a few seconds per model** on one core.
  Confidence: **MEDIUM** on the "sub-second-to-seconds" order (consistent with the documented "fast" /
  "orders of magnitude faster" + f4-reuse); **LOW** on any specific number.
Source: https://uqrmaie1.github.io/admixtools/articles/admixtools.html

### Rotation / model search — [scale VERIFIED, total ESTIMATE]

A rotating-outgroup screen runs many models. Reported scales: the Flegontov qpAdm-performance study ran
*"34,320 proximal and distal rotating models per simulation setup and replicate"* and a grand total of
*"27,511,200 individual qpAdm models"* (PMC10614728). A single real rotation is easily hundreds to tens
of thousands of models. **[VERIFIED scale]**
- **Classic:** rotation = #models × (1–2 min), serial → a 336-model screen ≈ 6–11 h; a 34,320-model
  screen ≈ weeks of single-core time. **[ESTIMATE built on the verified 1–2 min/model]**
- **AT2:** precompute f2 once, then run the whole rotation against RAM-resident f2 with f4-reuse +
  parallelism → minutes–hours (the f2 precompute is the dominant one-time cost). **[VERIFIED qualitative,
  no published wall-clock]**

### Steppe fit — NOT BUILT

The qpAdm fit / model-rotation engine is **explicitly not built** in steppe — it is **THE NEXT PHASE**
(Phase 2, S3–S8). **No steppe fit number is claimed.** Once built, reading device-resident `f2_blocks`
(or streamed tiles for large P), steppe's fit sits in the same "many models, each cheap" regime AT2
established — the per-model fit is small linear algebra on the compact tensor. Two honest framings stack:
(a) on the **precompute** the per-extract speedup is the decisive GPU advantage (matching AT2's own
framing — the f2 precompute is the expensive part); (b) the **model rotation** (thousands of INDEPENDENT
qpAdm models, no combine) is the embarrassingly-parallel, **multi-GPU-friendly** phase — that is where
multi-GPU genuinely shines, *not* on the precompute (see §4 / `architecture-audit.md`).

---

## 4. CAVEATS (what keeps this honest)

1. **The AT2 precompute time is an ESTIMATE, not a measurement.** No published `extract_f2`
   wall-clock-vs-P over ~584k SNPs exists. The only real anchor (issue #7 "~a day → 20 min") has **no
   workload metadata.** The AT2 band (minutes → hours) is reasoned from O(P²·M) + the memory wall.
2. **CPU cores assumed.** AT2 single-thread can be ~a day at scale; with `n_cores`/parallel it drops
   ~72× (issue #7). The "multiple hours" estimate assumes a typical 16–64-core server. Wall-clock swings
   directly with core count — a fair comparison must state the assumed CPU.
3. **Precision is not identical.** AT2 / classic are **native FP64**. Steppe is **emulated FP64**
   (Ozaki `EmulatedFp64{40}`), proven **bit-identical to a native-FP64 oracle** on real AADR (locked
   `test_f2_multigpu_parity`, `std::memcmp`) — FP64-*accurate*, but emulated, not native hardware FP64.
   That is the honest framing. **[VERIFIED: `docs/ROADMAP.md:16-18,71`]**
4. **`extract_f2` is precompute-once (amortized).** Both tools amortize it across the whole fit/rotation.
   The precompute speedup matters most when you re-extract often (new pop sets); for a single fixed
   pop set, it is paid once.
5. **Steppe's number is the PRECOMPUTE ONLY.** The fit phase is unbuilt; no end-to-end steppe pipeline
   number exists. Any speedup claim is scoped to the `extract_f2`-equivalent precompute.
6. **Hardware is not matched.** Steppe = 2× RTX PRO 6000 Blackwell (P=2000 = 15.1 s) or a single RTX
   5090 (post-M5 streamed sweep, P=2500 = 51.5 s). AT2 = whatever CPU you run it on. This is a
   CPU-vs-GPU cross-architecture comparison, not a same-hardware one — always state the box.
7. **The ratio, not the steppe number, is the estimate.** Both steppe anchors (P=2000 = 15.1 s,
   P=2500 = 51.5 s) are measured; the AT2 baseline is the part with no published benchmark, so the
   speedup is an order-of-magnitude band — **never quote a single Nx as fact.**

---

## 5. BOTTOM LINE

- **f2 precompute (the apples-to-apples):** For all-pairs f2 over ~584k SNPs with n_block≈757 at the
  thousands-of-pops scale, steppe's GPU precompute is **plausibly 2–3 orders of magnitude faster
  wall-clock (~10²–10³×) than AT2's CPU `extract_f2`** — but the AT2 baseline is an **ESTIMATE** (no
  measured AT2 number exists for this workload; the only anchor is issue #7's metadata-free "~a day →
  20 min"). The most defensible single comparison: **steppe P=2000 = 15.1 s [VERIFIED, 2× RTX PRO 6000]**
  (or **P=2500 = 51.5 s [VERIFIED, post-M5, one 5090, streamed]**) vs **AT2 ~1–6 h [ESTIMATE] →
  ~10²–10³×**. Do not present any single "Nx" as fact — the ratio is an estimate because the AT2 side is.
- **RAM/disk:** AT2 holds the full allele-freq matrix in CPU RAM (8 GB `maxmem` default, spills to
  O(P²) on-disk per-pair files past a "moderate number of pops") **[VERIFIED]**; steppe *pre-M5* held the
  full result tensor in GPU VRAM (75.7 GB @ P=2500 → OOM on 96 GB), and **M5 out-of-core (tiered output
  `176a07d` + SNP-tile input `c65179f`) removed that wall** — on-device footprint `O(P·tile + P²)`,
  result auto-tiered VRAM → host RAM → disk **[VERIFIED, post-M5]**. Both O(P²) storage; AT2 trades
  time-for-RAM via disk chunking, and steppe now has the analogous adaptive fallback.
- **Fit phase:** classic qpAdm = **1–2 min/model single-core [VERIFIED, 593k-SNP dataset]**; AT2 =
  sub-second-to-seconds/model from precomputed f2 **[ESTIMATE]**; a real rotation is hundreds to tens of
  thousands of models, so classic = hours-to-weeks, AT2 = minutes-hours. **Steppe's fit is NOT built — it
  is THE NEXT PHASE (Phase 2 qpAdm fit engine); the model rotation is multi-GPU's proper home. No steppe
  fit number is claimed.**

---

## Sources (all fetched / verified 2026-06)

- Maier et al. 2023, eLife e85492 — precompute-vs-fit split, complexity (k²/k³), Appendix 1—figure 1:
  https://elifesciences.org/articles/85492 · PDF:
  https://reich.hms.harvard.edu/sites/reich.hms.harvard.edu/files/inline-files/elife-85492-v2.pdf
- AT2 tutorial (two-step design verbatim; dim(f2_blocks)=7 7 708; count_snps=780009):
  https://uqrmaie1.github.io/admixtools/articles/admixtools.html
- `extract_f2` reference (maxmem=8000, not a precise cap; cols_per_chunk time↔RAM — fetched, verified):
  https://uqrmaie1.github.io/admixtools/reference/extract_f2.html
- `afs_to_f2` (too-many-pops, disk-spilled): https://uqrmaie1.github.io/admixtools/reference/afs_to_f2.html
- `qpadm_multi` (f4-reuse, parallel): https://uqrmaie1.github.io/admixtools/reference/qpadm_multi.html
- AT2 GitHub issue #7, parallel extract_f2, "~a day to 20min" (verbatim, no workload metadata):
  https://github.com/uqrmaie1/admixtools/issues/7
- CompPopGen 2019 — classic qpAdm "1-2 minutes, single core", 593124 SNPs/107 indivs/711 blocks (verbatim):
  https://comppopgenworkshop2019.readthedocs.io/en/latest/contents/05_qpwave_qpadm/qpwave_qpadm.html
- qpAdm rotation scale — "34,320 rotating models ... 27,511,200 individual qpAdm models":
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10614728/
- Steppe repo (VERIFIED): `docs/cleanup/m4.5/scaling-sweep.md` (measured 2× PRO 6000 sweep, P=2500 OOM
  **pre-M5, now SUPERSEDED**), `docs/cleanup/m5/00-results.md` (post-M5 single-5090 streamed sweep,
  P=2500 = 51.5 s, the headline result), `docs/cleanup/m4.5/why-d2h.md`,
  `docs/cleanup/m4.5/architecture-audit.md`, `docs/ROADMAP.md:16-18,71` (Phase-2 fit = next).

**Confidence summary:** AT2 algorithm / structure / O(P²) scaling / two-phase model / RAM-disk knobs =
**HIGH** (primary docs + paper). Classic qpAdm 1–2 min/model = **HIGH** (verbatim, matching dataset).
Single serial→parallel extract_f2 anecdote = **VERIFIED quote, no workload metadata**. AT2 absolute
extract_f2 wall-clock for P=500–2500 over ~584k SNPs = **ESTIMATE** (no benchmark; order-of-magnitude
band only). Steppe numbers = **VERIFIED** on both axes: PRO 6000 sweep P=2000 = 15.1 s (P=2500 OOM
*pre-M5*), and the post-M5 single-5090 streamed sweep P=2500 = 51.5 s (`docs/cleanup/m5/00-results.md`).
The AT2 *ratio* is the estimate — never quote a clean Nx.
