# steppe Feature Matrix — Smoke + Wall-Clock (real AADR)

**Every number in this document is a measured real-AADR run, golden-anchored where a committed golden exists, ZERO synthetic data** — not data, not throughput, not a fabricated or extrapolated number (per the project's `real-data-only-all-results` mandate). Each wall-clock is a `/usr/bin/time -v` Elapsed of one clean, sequential, single-GPU (`--device 0`) run on a Release build (`build-rel`). GPU util is `nvidia-smi -i 0 --query-gpu=utilization.gpu -lms 200`.

- **Branch / build:** `phase2-fit-engine` (== `main`); binary `/workspace/steppe/build-rel/bin/steppe`, Release, rebuilt 2026-06-24.
- **Box:** box5090 — 2x RTX 5090 (Blackwell sm_120, CUDA 13.0, driver 580), SINGLE-GPU `--device 0`. (rtxbox / box6000 were unreachable this pass; the 5090 is same-arch sm_120 and perf-representative — the 177s sweep and the DATES goldens were measured on it.)
- **No code changes** — this pass is smoke + measure + this doc only.
- **Verdict:** all CLI commands + all Python functions RAN OK (exit 0, sane non-empty output); every command with a committed golden MATCHED at its tier. No core command (qpadm / f4 / qpgraph / extract-f2) is broken. See [FLAGS](#flags) for the non-fatal findings.

---

## Per-feature table

### f2 precompute

| Feature | CLI command | Runs | Wall (real AADR) | GPU util | Golden-match | Python |
|---|---|---|---|---|---|---|
| f2 extract (genotype → STPF2BK1) | `extract-f2` | Y | 2.09 s (9-pop, 351539/584131 SNPs, 710 blocks) | host-decode/IO bound; f2 GEMM sub-200ms (small extract) | PASS — SNPs/blocks EXACT (351539/710 = golden_fit0); steppe-decode tier | Y (`extract_f2`; out=None == read_f2(out) bit-identical, maxabsdiff=0.0) |
| f2 load | (loader) | — | — | — | (loader, no golden) | Y (`read_f2`; f2_500 P=500/711 in 4.15 s, 5.76 GB RSS) |

### admixture

| Feature | CLI command | Runs | Wall (real AADR) | GPU util | Golden-match | Python |
|---|---|---|---|---|---|---|
| qpAdm fit | `qpadm` | Y | 0.92 s (fit0 9-pop, 1 model) | sub-200ms GLS solve (5x5 inverse + ALS over in-VRAM f2); 0% sampled = below sample period, not spin-wait | **PASS, TIGHT** — weights rtol ~8.5e-14 vs the 1e-6 target; chisq/p rtol ~1e-13. (vs directory-path convertf-PA golden: rtol ~5e-6 = documented TGENO-vs-convertf SNP-set delta) | Y (`qpadm`; ~1e-14 vs golden_fit0) |
| qpAdm rotation sweep | `qpadm-rotate` | Y | 24.24 s (nr=5, C(18,5)=**8568** models, SE-bearing CSV path), 3.29 GB RSS | **GPU-bound 100%** (98/121 samples; bookends are dir-load + write) | no per-model golden; n_models EXACT C(18,5)=8568; 353.5 models/s (SE path; ~2,866/s on lighter no-SE engine path) | Y (`qpadm_search`; model[0] == fit0 golden ~1e-13, alternates correctly rejected) |
| qpWave rank sweep | `qpwave` | Y | 0.91 s (9-pop, r=0,1) | sub-200ms; two small GLS solves over in-VRAM f2 | **PASS** — est_rank=1, rank0 chisq 1474.033 = golden_qpwave M1; per-rank chisq/p match steppe qpadm f2-object form | (QpWaveResult type exposed; covered via qpwave path) |

### descriptive f-stats

| Feature | CLI command | Runs | Wall (real AADR) | GPU util | Golden-match | Python |
|---|---|---|---|---|---|---|
| f4 (single quartet) | `f4` | Y | 0.73 s | sub-200ms (tiny work) | **PASS** — est rel ~8e-13 vs golden_fit0_f4_readf2; committed gate (60 quartets, rtol 1e-6) GREEN | Y (`f4`; sane on two real f2 dirs) |
| f3 (single triple) | `f3` | Y | 0.74 s | sub-200ms (tiny work) | **PASS** — est rel ~1.3e-13 vs golden_fit0_f3_readf2 | (F3Result type exposed) |
| f4-ratio | `f4-ratio` | Y | 0.72 s | sub-200ms (tiny work) | **PASS** — alpha rel ~2.3e-13 vs golden_fit0_f4ratio_readf2 | (F4RatioResult type exposed) |
| qpDstat (f2-path = f4) | `qpdstat --f2-dir` | Y | 0.73 s | sub-200ms (tiny work) | **PASS** — byte-identical to the f4 golden (f2-path qpdstat == f4) | Y (`qpdstat` / `dstat`) |
| qpDstat (genotype-path normalized-D) | `qpdstat --prefix` | Y | 1.25 s (raw TGENO; I/O/decode-bound) | sub-200ms GPU compute; wall is tile read | **PASS** — est rel ~1.3e-12 vs golden_fit0_dstat_geno (~0.0013 D); gate test_cli_dstat_geno (60 quadruples) GREEN | Y |
| f4 all-quartets SWEEP | `f4 --all-quartets` | Y | **7.52 s** representative C(175,4)=**37,752,925** quartets, top-1M survivors, ~5.0M q/s; VRAM 14.6 GB, RSS bounded 3.15 GB | **GPU 100%** in compute phase (on-device unrank→f4→jackknife→\|z\|-filter→CUB-compact) | no golden (discovery enumeration); count EXACT C(175,4); survivors sorted by \|z\| | (CLI sweep; not a Python facade) |

> **Full sweep, CITED from record (commit `2f6a050`, NOT re-run):** C(500,4) = **2,573,031,125** quartets in **~177 s (2:57)**, ~14.5M quartets/sec, GPU ~100%, peak host RSS ~3.1 GB (bounded), VRAM ~14.6 GB, top-K cut at \|z\|=140. The representative C(175,4) run above validates the same on-device path at seconds scale.

### smoothing (joint f-stats)

| Feature | CLI command | Runs | Wall (real AADR) | GPU util | Golden-match | Python |
|---|---|---|---|---|---|---|
| qpfstats (genotype-path joint smoother) | `qpfstats` | Y | 1.45 s (9-pop) | host-decode bound; GPU sub-200ms bursts. Cite 40-pop production qpfstats at 8.1 s | **PASS** — all 31995 entries (711 blocks x 45 pairs) vs golden_qpfstats_geno: maxrel 5.89e-12, 0 exceed rtol 1e-6 | Y (`qpfstats`; finite_frac=1.0, P=9/711) |

### graph fitting

| Feature | CLI command | Runs | Wall (real AADR) | GPU util | Golden-match | Python |
|---|---|---|---|---|---|---|
| qpGraph (fit one topology) | `qpgraph` | Y | 1.00 s (9-pop, numstart=10) | sub-200ms single fit; GPU-active signal shows on the search command | **PASS (gated)** — score 80.0674 == golden 80.06742; all 17 edges match; admix pSteppe→aCW 0.153484. (CLI rounds %.6g → displayed rel ~3e-7 is a DISPLAY artifact; tight parity confirmed by ctest + Python ~6e-9) | Y (`qpgraph`; score ~6e-9, weight ~5e-8) |
| qpGraph topology search | `qpgraph-search` | Y | 3.43 s (5 leaves, max-nadmix=1; fit_all 1092 ms, 1455.68 topo/s) | **GPU-ACTIVE 95–100%** (10/17 samples) during the batched fit | **PASS (gated)** — candidates EXACT **1590** (105 nadmix0 + 1485 nadmix1) vs at2_scores_5pop.csv; argmin nadmix=1; best score 0.365555 == golden 0.36555 | **Y** (`qpgraph_search` facade + `QpGraphSearchResult`; Python smoke 1590 candidates, best 0.3655547, argmin idx 316 / hash 10714990228868764576 == golden, `heuristic_recovered=True`) |

### dating

| Feature | CLI command | Runs | Wall (real AADR) | GPU util | Golden-match | Python |
|---|---|---|---|---|---|---|
| DATES admixture dating (cuFFT LD decay) | `dates` | Y | 2.51 s (PUR ← CEU+YRI; 600K-SNP HO) | **100%** FFT-autocorrelation bursts (2/13 samples); rest is host genotype decode | **PASS** (tier rtol 0.02) — date_gen 9.7335 vs golden 9.742 (rel 8.7e-4); SE 0.359 vs 0.317 (inside atol 0.10) | Y (`dates`; 9.73 gen, SE 0.36, literature-consistent) |

---

## FLAGS

No command failed to run and no golden mismatched at its acceptance tier this pass. The findings below are non-blocking.

1. **RESOLVED (8dcfa6a).** ~~`qpgraph_search` is NOT in the Python public facade (inventory gap).~~ Added a `qpgraph_search()` facade + `QpGraphSearchResult` accessor + `__all__` entries in `bindings/steppe/__init__.py`, mirroring `qpadm_search`; the already-computed per-candidate scored vector is now additively marshalled out of `module.cpp` (`cand_nadmix/hash/score/restart_spread`) so the facade exposes `.candidates` + `.argmin`. Engine/fit math UNCHANGED. Python smoke (real-AADR afprod=FALSE 9-pop f2, 5-pop search subset, device 0): 1590 candidates, best 0.3655547, argmin idx 316 / raw uint64 hash 10714990228868764576 == golden, `heuristic_recovered=True`.

2. **RESOLVED (8dcfa6a).** ~~STALE help text on `qpdstat`.~~ The `qpdstat` subcommand description AND the `--prefix` flag help in `src/app/cli_parse.cpp` no longer say "not yet implemented (Part B)"; they now describe the working genotype-path normalized-D (`--prefix PREFIX.{geno,snp,ind}`, allsnps=TRUE block-jackknife). Verified live: `qpdstat --help` "not yet implemented" count = 0.

3. **RESOLVED (8dcfa6a) — CMake arg aligned; constraint stands.** `qpdstat --prefix` remains TGENO-only by design (individual-major); pointing it at the convertf-PA prefix fails with exit 5 + a clear decode error (consistent with `aadr-tgeno-goldens-corrupt`). The stale `converted_pa/v66_HO_pa` reference in the `cli_dstat_geno` CMake comment was aligned to the raw TGENO prefix the gate derives internally; the `add_test` COMMAND already passed only the correct 3 args (binary, AADR root, golden dir). Gate `cli_dstat_geno` GREEN.

4. **qpgraph / qpgraph-search CLI rounds score/weights to `%.6g` in BOTH csv and json.** Full-precision score parity vs golden can't be read off the CLI (displayed rel ~3e-7 / 7e-7 is a DISPLAY artifact). Tight parity IS confirmed by the committed ctest gates + the matching edge weights + exact candidate counts (1590 = 105+1485) + argmin (nadmix=1). If a JSON consumer needs full-precision scores, a higher-precision score field is a follow-up.

5. **GPU-util "0%" on sub-second single-item commands is a sampler-resolution artifact, not a CPU fallback.** `extract-f2`, `qpadm`, `qpwave`, the four f2-path f-stats, and `qpfstats` showed 0% on the 200ms-cadence sampler purely because their per-item GPU compute window is below 200ms (confirmed by sub-second walls + exit 0 + correct golden output). The architecture has no CPU runtime — the fits ARE on the GPU. The GPU-saturating workloads (`qpadm-rotate` 100%, `f4 --all-quartets` 100%, `qpgraph-search` 95–100%, `dates` 100% FFT) confirm real device work; `extract_f2` showed 28% (decode/H2D-bound, tiny 4-pop).

---

## Notes

- **Data (all real, on box):** raw v66 HO TGENO `/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB`; committed STPF2BK1 dirs `/workspace/data/f2_500` (500 pops) and `/workspace/data/qpgraph_9pop_stpf2bk1`; the committed 9-pop fixture `tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin` (P=9, nb=710) re-staged into an STPF2BK1 dir (the way `test_cli_f4.cpp` does).
- **Why extract output (not f2_fit0_FINAL) feeds qpadm:** `f2_fit0_FINAL` is an AT2 RDS-layout dir (per-pop `*_f2.rds`), not steppe's STPF2BK1. The qpadm CLI requires STPF2BK1, so the qpadm smoke fit ran over the `extract-f2` output — making the extract→fit chain a true end-to-end smoke.
- **Goldens:** `tests/reference/goldens/at2/{csv/golden_qpfstats_geno.csv, golden_fit0*.csv/json, golden_qpwave*, golden_qpgraph_*, fixtures/at2_scores_5pop.csv}` and `tests/reference/goldens/dates/aadr_PUR_CEU_YRI/`.
- **TGENO-vs-convertf golden tiers:** the canonical steppe-decode (f2-object) tier matches at rtol ~1e-13; the directory-path convertf-PA tier (different 391333-SNP set vs steppe's 351539) is reached at rtol ~5e-6 (weights) / ~3e-3 (qpwave chisq). This is the DOCUMENTED, expected SNP-set delta baked into the golden JSON, not a parity failure (steppe decode is correct; AT2 v2.0.10 misreads raw TGENO — `aadr-tgeno-goldens-corrupt`).
