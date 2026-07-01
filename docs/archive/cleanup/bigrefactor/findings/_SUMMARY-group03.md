# Group 3 — Dead / commented-out code — ROLLUP SUMMARY

Scope: all 61 in-scope units. Rollup of the `## Group 3` section appended to each
`findings/<slug>.md`. Tasks: 3.1 commented-out blocks kept "just in case";
3.2 unreachable code (`#if 0`, code after return/break); 3.3 unused symbols
(vars/params/includes/helpers); 3.4 computed but unread.

## 1. Coverage

| | count |
|---|---|
| Units in scope | 61 |
| Units clean (no Group 3 issues) | 43 |
| Units with findings | 18 |
| Total findings | 24 |

Units with findings: block_partition_rule, f2_blocks_multigpu, f2_estimator,
model_search, nested_models, ranktest, cpu_backend, block_sink, cuda_backend (2),
device_partial, f2_blocks_out (4), handles (2), qpadm_fit_kernels (2),
f2_disk_format, host_ram, tier_select, mind_prepass, geno_reader.

## 2. Counts by task and severity

All 24 findings are **LOW**. Zero HIGH, zero MED.

| Task | HIGH | MED | LOW | total |
|---|---|---|---|---|
| 3.1 commented-out / stale doc | 0 | 0 | 1 | 1 |
| 3.2 unreachable code | 0 | 0 | 1 | 1 |
| 3.3 unused symbols (incl. includes) | 0 | 0 | 16 | 16 |
| 3.4 computed but unread | 0 | 0 | 6 | 6 |
| **Total** | **0** | **0** | **24** | **24** |

3.3 dominates and is overwhelmingly **unused / redundant `#include`s** (10 of the 16):
`<cstdint>` (block_partition_rule.hpp:31, geno_reader.cpp:14),
`<cstddef>` (device_partial.hpp:8, f2_disk_format.hpp:16),
`<cstring>` (host_ram.cpp:12), `<utility>` (block_sink.cu:29),
`error.hpp` (model_search.cpp:19),
plus three on f2_blocks_out.cu (pinned_buffer.cuh:19, log.hpp:21, device_buffer.cuh:17),
and the umbrella/re-export `launch_config.hpp` (f2_estimator.hpp:35).
The remaining 3.3 are unreferenced helpers/accessors (tier_select, handles x2, cpu_backend)
and unused-but-void-cast params (qpadm_fit_kernels:1109).

## 3. Top findings

No HIGH or MED findings. The most consequential LOW (silent-status-discard, scale path):

- [3.4][LOW] src/device/cuda/cuda_backend.cu:1508 — `dInfo` in `large_svd_V` is passed
  to every `cusolverDnDgesvdj`/`Dgesvd` (1528,1538,1570,1579) but never copied back or
  checked: a non-converged Jacobi SVD / illegal gesvd param on the large (NRBIG) path
  passes silently into seed/ALS. Suggested: D2H+check `dInfo` once on the large path, or
  note the intentional discard.
- [3.4][LOW] src/device/cuda/cuda_backend.cu:2276,2279 — `solve_info` out-arg of
  `cusolverDnDpotrsBatched` is never read; per-column SPD solve status silently discarded
  (status gated only by the earlier `potrfBatched` `dInfo`). Suggested: check
  `solve_info != 0` (record-and-continue) or document the intentional discard.

Selected other LOW (hygiene / clearly safe):

- [3.4][LOW] src/core/fstats/f2_blocks_multigpu.cpp:358 — `out.P = P;` dead store; all
  three switch arms overwrite unconditionally before any read. Suggested: drop the line.
- [3.4][LOW] src/core/qpadm/ranktest.cpp:44 — `(void)m_full;` dead suppression; `m_full`
  IS read at line 71. Suggested: delete line 44.
- [3.3][LOW] src/device/tier_select.hpp:82 — `streamed_working_set_bytes` inline helper
  has zero call sites (its own doc admits it is unused by `select_output_tier`).
  Suggested: drop until a caller needs it, or wire into the bench/select it promises.
- [3.3][LOW] src/device/cpu/cpu_backend.cpp:745 — private `xmat_from_loo_block` never
  called (superseded by the S7 batched LOO path). Suggested: delete or `[[maybe_unused]]`.
- [3.2][LOW] src/device/cuda/f2_blocks_out.cu:160 — `return F2BlockTensor{};` after a
  switch covering all OutputTier enumerators is unreachable; intentional no-`default`
  fall-through to silence the warning. Suggested: leave as-is (noted for completeness).
- [3.1][LOW] src/core/qpadm/nested_models.hpp:10-12 — stale doc describes an n_block host
  loop that no longer exists (now the batched `gls_weights_loo_batched` seam). Suggested:
  update the comment to the batched-seam impl.
- [3.4][LOW] src/io/filter/mind_prepass.cpp:64 — redundant write loop re-writes the 0.0
  already set by `assign(n_ind, 0.0)`; self-documented as intentional convention.

## 4. Cross-cutting patterns

1. **Stray / transitive includes are the bulk of the noise (10 of 24).** A consistent IWIYU
   pass would clear `<cstdint>`/`<cstddef>`/`<cstring>`/`<utility>` strays and the three
   redundant direct includes on f2_blocks_out.cu in one sweep. Several are paired with a
   stale comment naming a symbol that is no longer used (`// Status`, "pin the D2H",
   "STEPPE_LOG_WARN (teardown)").
2. **Scaffold landed ahead of its consumer is not truly dead.** handles.hpp
   `CusolverMathModeScope::promoted()` (476) and `CusolverDnHandle::device_id()` (339),
   tier_select `streamed_working_set_bytes`, and the qpadm_fit_kernels `dLoo`/`d_se`
   void-cast params are deliberate seam/symmetry surfaces (the §6 fit-precision promotion
   seam, M4.5 multi-GPU). Flagged LOW for visibility, NOT removal — drop only if the
   matching consumer is abandoned.
3. **Required-but-unread cuSOLVER status out-args are a recurring real-but-minor gap.**
   `dInfo`/`solve_info` (cuda_backend.cu) are non-null API requirements whose returned
   convergence/SPD status is silently dropped on the large/per-column paths. Lowest-risk
   today (small bit-parity path is checked) but worth an explicit check or comment so the
   discard reads as intentional.
4. **No `#if 0` / parked code anywhere.** Across all 61 units, every flagged "comment block"
   on closer read was live design/parity rationale, not stashed-out source. The single 3.2
   item is an intentional defensive fall-through, not a defect.

## Headline numbers

Units in scope: 61 (43 clean, 18 with findings). Total findings: 24. HIGH: 0 (MED: 0; all 24 LOW).
