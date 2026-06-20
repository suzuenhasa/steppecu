# GROUP 8 (Comments) — Roll-up Summary

Tasks: 8.1 Restating code · 8.2 Stale comments · 8.3 Missing rationale · 8.4 Orphan TODO/FIXME/HACK.

## 1. Coverage

- **Units in scope:** 61 (scope = all)
- **Units with >=1 finding:** 36
- **Clean units (no Group 8 issue):** 25

Clean: include__steppe__fstats, block_partition_rule, f2_estimator, launch_config, log,
pchisq, views, f4_matrix, gls_solve, jackknife, model_search_core, device_buffer,
device_f2_blocks, f2_blocks_kernel, handles, shard_plan, stream_f2_blocks, vram_budget,
filter_decision, filter_plan, include_exclude, snp_filter, geno_reader, genotype_tile, ind_reader.

(Note: several "clean" units carry detailed read-only verification notes confirming all four
tasks were checked — e.g. shard_plan, vram_budget, geno_reader, snp_filter, device_f2_blocks.)

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|------|-----:|----:|----:|------:|
| 8.1 Restating code      | 0 | 0  | 1  | 1  |
| 8.2 Stale comments      | 0 | 16 | 26 | 42 |
| 8.3 Missing rationale   | 0 | 1  | 11 | 12 |
| 8.4 Orphan TODO/etc.    | 0 | 0  | 1  | 1  |
| **Total**               | **0** | **17** | **39** | **56** |

By severity (deduplicated finding bullets): **HIGH 0 · MED 17 · LOW 39 = 56**.
By task: 8.1 = 1 · **8.2 = 42 (dominant)** · 8.3 = 12 · 8.4 = 1.

8.2 MED breakdown (16): error, qpadm.hpp, f2_blocks_multigpu, f2_combine, f2_partials_validate,
qpadm_fit, backend_factory, cuda_backend, p2p_combine, resources, tier_select, mind_prepass (×1
each) + qpadm_fit_kernels (×2) + eigenstrat_format (×2). 8.2 LOW = 26. (16 + 26 = 42 for 8.2.)
8.3 MED = 1 (block_sink event-sync asymmetry). All other 8.3 (11) and 8.1/8.4 (1 each) are LOW.

## 3. Top findings (MED — no HIGH in this group)

The dominant defect class is **stale comments (8.2)**: comments describing behavior or APIs the
code no longer has, and cross-file line-number citations that have drifted.

Stale-behavior / stale-API (a maintainer would chase a phantom or wire to the wrong seam):

- [8.2][MED] src/core/fstats/f2_blocks_multigpu.hpp:45-47 — docstring claims the tiered path
  delegates to multi-GPU sharding for G>=2; the body has only `if (G < 1) throw` and always
  drives gpus[0]. Reader assumes sharding that does not exist.
- [8.2][MED] src/device/p2p_combine.hpp:6-7,10-11,25-28 — lead prose says the combine "SUMS the
  same fixed g-order onto a zero-initialized full tensor"; the impl is a disjoint raw-byte
  PLACEMENT (no memset, no place-add) — and the SAME header contradicts itself 40 lines down
  (.hpp:51-56). Pre-cleanup-B7 residue.
- [8.2][MED] src/core/fstats/f2_combine.hpp:6-11 — "PLACES + SUMS" / "Summing in a fixed order"
  narrative is the stale pre-B7 `+=` semantics; code is placement-only (`std::copy_n`). Same
  stale-summation pattern as p2p_combine.
- [8.2][MED] f2_partials_validate.hpp:23-27 — describes a `device_ids.size() == G` check the
  resident tier no longer threads (device_id is now per-handle in DevicePartial); sends a reader
  hunting a phantom guard.
- [8.2][MED] src/device/backend_factory.hpp:50-51 — `make_cuda_backend` doc says the shard +
  fixed-order combine is "the next workflow"; per ROADMAP §72 it is DONE (867a4bf, bit-identical).
- [8.2][MED] src/core/qpadm/qpadm_fit.hpp:32 — `run_impl` doc names a param `se_policy` that
  gates the S7 jackknife SE; no such param exists — the gate is `opts.jackknife` (JackknifePolicy).
- [8.2][MED] src/device/cuda/cuda_backend.cu:1502 — `large_svd_V` doc says `dXt` is "only used
  in the nr>=nl branch"; it is written and read in BOTH branches (Xt for nr>=nl, V for nl>nr).
  No live bug (both sized nr*nl), but understates the buffer's role.
- [8.2][MED] include/steppe/error.hpp:11-12 — "The three DOMAIN-OUTCOME values" but lists/has
  only TWO (RankDeficient, NonSpdCovariance). Count contradicts the enum.
- [8.2][MED] include/steppe/qpadm.hpp:17-22 — SCOPE banner says the header "freezes only the
  single-model M(fit-1) shapes"; it now also declares the rank-sweep, qpWave, search, and S8
  jackknife surface. Banner understates the declared API.
- [8.2][MED] src/io/eigenstrat_format.cpp:34-35 & :86 — two comments contradict the code:
  (a) "trailing-substring check" describes a strategy the code does not use (it does
  leading-token isolation + exact `==`); (b) "leave format set but counts 0" — line 87
  immediately sets `format = Unknown`.

Stale cross-file line-number citations (the recurring sub-pattern — see §4):

- [8.2][MED] src/device/cuda/qpadm_fit_kernels.cu:178,206,260,311,335 (+.cuh) — every small-LA
  helper's `cpu_backend.cpp:NNN` citation has rotted (seed_AB 626->761, opt_A 652->787,
  opt_B 715->850, als_weights 773->908, chisq_of 833->968); followers land mid-unrelated-function.
- [8.2][MED] src/device/cuda/qpadm_fit_kernels.cu:417,499,504 (+.cuh) — compute_loo_and_total
  553->675, xmat_from_total 598->733 likewise rotted.
- [8.2][MED] src/device/resources.hpp:14-15 — cites backend.hpp:193-202 for the per-device
  contract; that range is now DecodeTileView (actual block at backend.hpp:341).
- [8.2][MED] src/device/tier_select.hpp:47 — cites cuda_backend.cu:544-560 for "freed-raw-VRAM
  reuse"; that range is the pinned-DMA comment (actual reuse at :567, slab pre-size :577).

Missing rationale (8.3 — only one MED):

- [8.3][MED] src/device/cuda/block_sink.cu:89-94 vs 245-249 — HostRamSink and DiskSink handle a
  `cudaEventSynchronize(s.done)` failure ASYMMETRICALLY with no rationale: HostRam only WARNs and
  then memcpys a possibly-undrained slot (silent §12 parity-corruption risk), DiskSink fails fast.
  An intentional cross-tier deviation that should be documented or made consistent.

## 4. Cross-cutting patterns

1. **Stale cross-file line-number citations are the single largest sub-class (~12 findings).**
   `file.cpp:NNN` (and `:NN-MM` range) citations rot on every edit to the cited file. Worst
   offender: qpadm_fit_kernels (every small-LA helper citation into cpu_backend.cpp drifted ~130
   lines). Others: f2_from_blocks (backend.hpp:144->375), resources (193-202->341),
   tier_select (544-560->567), stream (267-323->281-336), f2_disk_format (off-by-one ×2 into
   fstats.hpp), backend (M(fit-2) vs M(fit-4) milestone slip), device_partial (.cu vs .cuh split).
   **Recommendation: cite by symbol/section name, not line number** (several units already do this
   correctly in sibling comments — the drift is the inconsistency).

2. **Pre-cleanup-B7 "sum/accumulate" residue in the combine path.** Both f2_combine.hpp and
   p2p_combine.hpp lead prose still describe a memset+`+=`-onto-zero accumulate; the code (and both
   files' own lower doc-comments) are disjoint PLACEMENT. The contradiction lives within single
   files — the lead narrative was not updated when the body was.

3. **"Next milestone / for now" framing that has since landed.** backend_factory ("next
   workflow"), cpu_backend ("later milestones add … here"), qpadm.hpp ("only M(fit-1) shapes"),
   nested_models (M(fit-1) host loop) all describe a past, smaller surface. Milestone-tense
   comments age poorly once the milestone ships.

4. **Doc-baked default values that drift from the runtime knob.** cuda_backend ("20 ALS iters" vs
   `opts.als_iterations`), qpadm_fit_kernels (illustrative nb = 701/702/708 across three sites),
   snp_reader (chrom codes 23/24/90 restated in prose vs single-homed constants), config
   (7-13x vs 7-17x speedup). Comments that hardcode a default/example go stale silently.

5. **8.1 (restating) and 8.4 (orphan TODO) are near-absent (1 each).** The codebase's comments are
   overwhelmingly rationale/contract prose, not mechanical narration; TODO/FIXME/HACK markers that
   exist (TODO(multigpu-host-bounce), check.cuh CAP-tier notes) are owner-tagged with measured data
   and design-doc cross-refs — only one (check.cuh:24 "TODO M4.5" on a parity claim) reads as
   genuinely orphaned. The comment debt is almost entirely **drift/staleness (8.2)**, not noise.

## 5. Headline

**Units in scope: 61 (36 with findings, 25 clean). Total findings: 56. HIGH: 0 (MED 10, LOW 46).**
Dominant class: stale comments (8.2 = 42 of 56), led by drifted cross-file line-number citations.
