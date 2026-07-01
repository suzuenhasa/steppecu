# steppe — Portfolio-Readiness Review (SUMMARY)

Identify-only review of the steppe code surface (`src/ include/ bindings/`; `docs/`
excluded). 121 per-unit groups + 7 cross-cutting professionalism sweeps (X1–X7).
FP64-by-design, fixed-order reductions, single statistic stream, and the
parity-load-bearing `double` math are treated as intentional §12 design and were NOT
flagged. This report is prioritized to drive the fix campaign.

---

## 1. Headline counts

| Source            | HIGH | MED | LOW |
|-------------------|-----:|----:|----:|
| Per-unit (121)    |    1 |  32 | 200 |
| Cross-cutting (7) |   10 |  18 |  27 |
| **Total**         | **11** | **50** | **227** |

- **Units reviewed:** 121 per-unit + 7 cross-cutting = 128.
- **Verdict:** the *engineering* is staff-grade. The *presentation* is not yet
  portfolio-grade. There is exactly **one** real per-unit correctness/robustness bug
  (a cuFFT plan leak). Everything else is hygiene — and it clusters hard into one
  theme: **comment over-write + internal-process bleed.**

The cross-cutting sweeps are the make-or-break for a portfolio reviewer, so they come
first.

---

## 2. Cross-cutting findings (portfolio make-or-break)

Ordered by portfolio impact. The two dominant ones (X2 comments, X7 first-impression)
are the *same defect seen twice* and together they are THE headline of this review.

### X1 — Consistency (0 HIGH / 2 MED / 5 LOW) — **strong pass**
The codebase is exceptionally disciplined: every recurring op has a single documented
canonical home that is followed everywhere (RAII wrapper shape identical across 10
types; pImpl identical across 3; `STEPPE_CUDA_CHECK`/`CUBLAS_CHECK` universal; zero
`dynamic_cast`/`typeid` backend branching; 704 includes all quote-style root-relative,
zero `<>`/`../`). The only real drift:
- **[MED]** Post-launch kernel check done two ways — 14 sites open-code
  `STEPPE_CUDA_CHECK(cudaGetLastError())` instead of the documented
  `STEPPE_CUDA_CHECK_KERNEL()` (which adds the debug-sync async-fault attribution).
  `dates_kernel.cu` (9), `dstat_kernel.cu` (2), `qpgraph_fit_kernels.cu` (3).
- **[MED]** `qpgraph_fit_kernels.cu/.cuh` is the lone file using
  `namespace steppe::device::cuda` (38/40 device TUs use `steppe::device`).

### X3 — Structure / navigability (1 HIGH / 3 MED / 3 LOW) — **sound architecture, decomposition debt**
§4 layering genuinely HOLDS (io is a CUDA-free leaf, core host-pure, CUDA private to
`steppe_device`, public headers leak zero internals). The debt is god-files + a fat
seam + naming:
- **[HIGH]** `src/device/cuda/cuda_backend.cu:241-5648` — GOD-FILE. One `CudaBackend`
  class body, ~5400 LOC, ~30 overrides + private helpers, the largest file by 2.5x.
  Kernels ARE correctly factored out (zero `__global__` here) — this is pure host
  orchestration glue, but a new engineer cannot navigate one 5400-line class.
- **[MED]** `backend.hpp` — FAT INTERFACE: `ComputeBackend` = ~50 virtuals spanning
  every device op across all 9 tools; both backends must track all of it.
- **[MED]** `src/core/qpadm/` — MISNOMER/GRAB-BAG: holds qpadm + qpgraph fleet + f3/f4/
  f4ratio + fstat_sweep + model_search (8 subsystems under one tool's name).
- **[MED]** `src/access/` & `src/extract/` dirs contain only a CMakeLists; their source
  files physically live in `src/app/`.

### X2 — Comment signal-to-noise (6 HIGH / 5 MED / 4 LOW) — **THE dominant issue**
Comments are accurate and "why"-focused (a genuine strength: near-zero code-restating
comments) but written at **design-doc altitude and inlined into source**, then layered
with cross-ref chains and PR ticket IDs. Whole-codebase metrics: 830 comment lines
cross-reference `architecture.md`/`design §`/`ROADMAP §`; **42 files** carry internal
`cleanup X-N/B#` / `group-N` ticket archaeology; ~49 files open with a 25+ line header
essay; ~9,869 ALL-CAPS emphasis tokens; `cuda_backend.cu` is **31% comments**
(1749/5648). The 6 HIGH are design-doc essays guarding a few lines of code (the §4
COMBINE gate banner = 64 lines over 1 line of code; `p2p_combine.hpp`; `dstat_kernel.cu`
header; `decode_af.hpp` header; `cuda_backend.cu` Stream essay; `model_search.cpp` TODO
banner). The fix is **altitude/placement** (move to docs/, point from code) — not
deletion of meaning.

### X7 — Senior first-impression (3 HIGH / 4 MED / 3 LOW) — **same defect as X2, via voice**
The shipping source reads as a private engineering journal. The 3 HIGH:
- **[HIGH]** Internal cleanup-tracker IDs (`cleanup X-1/B1`, `X-8/B16`, …) footnoted in
  shipping comments across 25 files — including the **public** `include/steppe/config.hpp`
  and the **public seam** `backend.hpp`. An external reader cannot resolve them.
- **[HIGH]** `include/steppe/config.hpp:7-253` — the first public header a reviewer opens
  is saturated with private references (`ROADMAP §`, `M4`/`M5`, `group-5`, `B26`).
- **[HIGH]** `src/app/cli_parse.cpp:356` — STALE + self-contradicted: "only qpwave …
  remains a scaffold no-op" but qpwave is fully wired (and the file's own header line 13
  says so). The one true defect in X7.

One pass of "comment hygiene for an external reader" closes ~90% of X2+X7.

### X4 — Public-API ergonomics (0 HIGH / 2 MED / 5 LOW) — **senior-grade shop window**
Consistent result types, uniform `steppe <cmd>: <reason>` errors, typed/`__repr__`/`__all__`
Python facade. Two user-facing string warts to fix first:
- **[MED]** `bindings/module.cpp:1176` — `run_qpdstat` docstring says Part B "not yet
  implemented"; it IS shipped (`steppe.dstat`). A user is told a shipped feature is absent.
- **[MED]** Precision-token vocabulary split 3 ways across CLI/Python/emitted output
  (`emu40` vs `emulated_fp64` vs `emu`); `emu32` is unselectable from Python.

### X5 — Idiomatic modern C++20/CUDA (0 HIGH / 0 MED / 4 LOW) — **clean pass**
Zero C-casts/`typedef`/`NULL`/plain-enum/raw-new-delete-in-logic. Correct RAII
(move-only rule-of-5, `[[nodiscard]]`), `enum class`, span/optional/array, cub
primitives, overflow-safe `size_t` widening at scale. Only LOW: one hand-rolled copy
loop (`module.cpp:1103` → `std::copy_n`).

### X6 — Simplicity / over-engineering (0 HIGH / 2 MED / 3 LOW) — **NOT over-engineered**
The heavy machinery (50-virtual seam, PIMPL handles, `*_core` splits, tier/streaming)
is genuinely-earned GPU-shape + host-testability. Residue is dead/premature seams:
- **[MED]** Three dead `DeviceConfig` knobs (`stream_count`, `search_streams`,
  `use_mem_pool`) — documented, defaulted, asserted in tests, but NEVER read by any
  production path.
- **[MED]** `backend.hpp:1682 rank_test` — vestigial virtual, two full overrides, **zero
  callers** (the live path uses `rank_sweep`+`gls_weights`).

---

## 3. The HIGH per-unit bug (file:line)

There is exactly one, and it is a real resource-leak hazard (not a comment):

- **[HIGH] `src/device/cuda/cuda_backend.cu:1777-1922` (`dates_curve`)** — `cufftHandle
  plan_fwd/plan_inv` are RAW (non-RAII), created by `cufftPlanMany` and destroyed only at
  the function tail. Every `cufft_ok` between them THROWS on error — including the
  per-sample-loop `cufftExecD2Z`/`cufftExecZ2D` called `n_target` times. Any throw leaks
  BOTH plans (`cufftDestroy` never reached). This is the **one** place in the TU a CUDA
  resource is not RAII-wrapped (contrast the `GesvdjInfo` RAII added for exactly this
  hazard). Suggested: a move-only RAII type with a non-throwing `cufftDestroy` dtor,
  mirroring `GesvdjInfo`/`CublasHandle`.

---

## 4. Recurring per-unit patterns (the systemic fixes)

These MED/LOW findings repeat across many units — fixing the *pattern* is worth more
than fixing each site. Counts are per-unit findings by group tag.

### Pattern A — Stale / noisy comments (G8: 6 MED + 62 LOW, ~37 files) — **largest category**
This is the per-unit shadow of X2/X7. Two sub-shapes:
- **Stale comments describing removed implementations** (the highest-value MED cluster):
  - `nested_models.hpp:5-13,33` — 3 MED: header still documents the pre-M7 per-LOO-block
    `gls_weights_loo_batched` loop; the code now makes a single `se_from_wmat` call.
  - `f4ratio.cpp:13-38` — header documents a host-pure ratio-jackknife loop that was
    moved into `be.f4ratio_blocks_jackknife`.
  - `dates.cpp:29` — stale "reused elsewhere" comment on an unused include.
  - `qpgraph_model.hpp:39` — overstates cycle-detection coverage.
- **Hard-coded line-number cross-references + ticket IDs + changelog-style headers** —
  the bulk of the 62 LOW (e.g. `device_f2_blocks`, `p2p_combine`, `block_sink`,
  `f2_blocks_out`, `tier_select`, `snp_reader` verbose changelog headers).
- **Fix:** mechanical codebase-wide strip of `cleanup X-N/B#` + `group-N` tokens and
  hard line-number refs; sync the ~5 stale-implementation comments to current behavior.

### Pattern B — Duplicated parity-twin constants / literals (G5: 5 MED + 23 LOW)
Parity-load-bearing values expressed twice → silent-drift hazard between the
resident path and the CpuBackend oracle:
- `dstat.cpp:182` & `qpfstats.cpp:292` — oracle hardcodes `chr < 1 || chr > 22` while the
  resident path passes the named `kAutosomeChromMin/Max`. The comment asserts "BOTH
  produce the IDENTICAL kept SET" — exactly the invariant a drift would break.
- `qpadm_fit_kernels.cu:160-161,1034-1035` — `kTol=1e-15`/`kMaxSweeps=60` duplicated
  across the two Jacobi-SVD sweeps that are documented bit-identical.
- Unnamed magic literals: `cuda_backend.cu` int-clamp `0x40000000` ×3 + reservoir `160`;
  `dates.cpp:279` epsilon `1.0e-20`; `qpgraph_search.cpp:299` hill-climb cap `1000`.
- **Fix:** hoist to a single named `constexpr`; use the named constant on BOTH the
  resident and oracle paths.

### Pattern C — Copy-pasted boilerplate (G7: 8 MED + 29 LOW)
Verbatim duplication that will drift when a new case is added:
- `result_emit.cpp` — 4 MED: precision-tag ternary ×7; the 3 JSON-array lambdas defined
  twice; the rankdrop CSV/JSON loop body duplicated between `emit_csv`/`emit_qpwave_csv`.
- `bindings/module.cpp` — 3 MED: indices→names lambda ×4; the fixed-arity tuple-resolve
  loop; the device-fit prologue/epilogue (`ensure_resources`→upload→run→catch) ×9.
- `qpgraph_enumerate.cpp:247-279 vs 385-407` — the admix-1 construction body copy-pasted
  (the comment admits it "Mirrors enumerate_admix1's inner body").
- **Fix:** extract a single helper (`precision_str`, `names_of`, `resolve_tuple<N>`,
  `with_device_f2`, `admix1_children_of`) per cluster.

### Pattern D — Dead code / unused includes (G3: 4 MED + 36 LOW)
- `cpu_backend.cpp:1843-1855` — `xmat_from_loo_block` defined, never called.
- `cpu_backend.cpp:615` — `dates_curve` allocates a `res` vector never written/read.
- `cmd_extract_f2.cpp:353` — `extracted.precision_tag` (the ENGAGED precision) is
  computed-but-unread; meta.json records the *requested* precision even on a downgrade.
- Numerous unused/transitive includes (e.g. `run_config` uses `std::vector` with only a
  transitive `<vector>`; `dates.cpp:29` unused `block_partition_rule.hpp`).

### Pattern E — Real scale/overflow + robustness traps (G4/G10/G13/G18 — small but sharp)
The few that are genuine bugs at production scale (P~2500), not cosmetics:
- **`f3.cpp:69`** — `static_cast<int>(triples.size())` truncates the f3 batch m-axis; an
  all-triples sweep at P~2500 is C(2500,3) ≈ 2.6e9 > INT_MAX → silent wrap before the
  `N<=0` guard. A real int-overflow-at-scale bug.
- **`qpgraph_fit_kernels.cu:315-444`** — `D = t.nadmix` writes into fixed `[kMaxThetaDev=16]`
  stack arrays with NO guard that `nadmix <= 16`; neither host launcher rejects it →
  silent stack overrun (corruption, not crash) for >16 admix nodes.
- **`cuda_backend.cu` (G13)** — `cub::DeviceScan/Select/RadixSort` return codes never
  checked, while adjacent `cudaMemcpyAsync` IS checked — a failing CUB call is swallowed.
- **`device_decode_result.cu:52-57`** — `to_host_qvn` `cudaMemcpy`s from a null source for
  a regime-A handle (documented regime-B precondition not enforced at the boundary).

---

## 5. Recommended fix order (biggest portfolio impact first)

The portfolio thesis: *the engineering already passes; the presentation does not.* Fix in
this order.

**Phase 1 — Comment hygiene (closes ~90% of X2+X7; highest portfolio ROI, lowest risk)**
1. Mechanical codebase-wide strip of `cleanup X-N/B#` and `group-N` ticket tokens (42
   files). Keep the rationale, drop the label. **Do the PUBLIC surface first**
   (`include/steppe/*.hpp`, `backend.hpp`) — these must read self-contained.
2. Relocate the 6 X2-HIGH design-doc essays to `docs/` with a one-line pointer (the §4
   combine banner, `p2p_combine.hpp`, `dstat_kernel.cu`, `decode_af.hpp`, the
   `cuda_backend.cu` Stream essay, the `model_search.cpp` TODO banner).
3. Fix the stale/contradicted user-facing strings: `cli_parse.cpp:356` (qpwave "scaffold
   no-op"), `module.cpp:1176` (Part B "not yet implemented"), and the ~5 stale-
   implementation comments in Pattern A (`nested_models.hpp`, `f4ratio.cpp`).
4. Dial back ALL-CAPS density to the handful of genuine parity/precision invariants.

**Phase 2 — The one HIGH bug + the sharp scale/robustness traps (correctness)**
5. RAII-wrap the cuFFT plans in `dates_curve` (the only HIGH per-unit bug).
6. Pattern E: guard `f3.cpp:69` int-overflow; assert `nadmix <= kMaxThetaDev`; check the
   CUB return codes; enforce the `to_host_qvn` regime-B precondition.

**Phase 3 — Structure (the X3 decomposition debt — high reviewer-visible value)**
7. Split `cuda_backend.cu` into per-subsystem partial impls (f2 / decode+transpose / fit
   / sweep / jackknife / dates) behind one thin aggregate header (X3 HIGH).
8. Rename/split the misleading `src/core/qpadm/` grab-bag; align `access`/`extract`
   target dirs with their sources. (Segregate the fat seam is lower-priority — it
   naturally shrinks `cpu_backend.cpp` too, but it is the largest refactor.)

**Phase 4 — Systemic hygiene (the recurring patterns)**
9. Pattern B: hoist parity-twin constants to named `constexpr`, use them on BOTH oracle
   and resident paths (drift-proofs the parity gate).
10. Pattern C: extract the 5 boilerplate helpers (`precision_str`, `names_of`,
    `resolve_tuple<N>`, `with_device_f2`, `admix1_children_of`).
11. Pattern D: delete dead code/unused includes; fix the `precision_tag` engaged-vs-
    requested meta bug.
12. Polish: X1 kernel-check normalization (14 sites → `STEPPE_CUDA_CHECK_KERNEL()`),
    `qpgraph_fit_kernels` namespace re-alignment, X4 precision-token vocabulary, the dead
    `DeviceConfig` knobs / `rank_test` virtual (X6).

---

## 6. Honest bottom line

steppe is a genuinely senior/staff-grade GPU codebase: the §4 layering is real and
compiler-enforced, the RAII/idiom discipline is near-perfect (X5 pass), the consistency
is exemplary (X1 strong pass), it is not over-engineered (X6), and the public shop window
is clean (X4). It is held back from a portfolio-ready first impression by exactly two
things: **(1) comments written as a private engineering journal** (ticket IDs, milestone
codes, design-doc essays, ALL-CAPS — X2/X7, the dominant issue), and **(2) one 5400-line
god-file** (X3). There is a single real correctness bug (the cuFFT leak) and a short tail
of sharp scale/robustness traps. Phase 1 alone — pure comment hygiene, low-risk — moves
the needle the most for a hiring/portfolio reviewer.
