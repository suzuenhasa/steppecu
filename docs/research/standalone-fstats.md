# Standalone f-statistics on steppe — the CODE-VERIFIED build plan

**Synthesis of 4 lenses (steppe-machinery, near-stats, far-stats, integration), each load-bearing claim re-verified against source by the synthesizer.**

Repo: `/home/suzunik/steppe`, branch `phase2-fit-engine`. **HEAD is `25fa2ae`** (not the prompt's `a8cbd40`; the tree has advanced by one *workflow-script-only* commit — see Mismatch #1 below, which is the most important finding). Read-only survey; no code changed, no box runs.

---

## HEADLINE

- **Cheap-reuse (the f4 + f2_blocks + jackknife already exist, batched, device-resident, golden-gated):** **f4, f3 / outgroup-f3 / admixture-f3, f4-ratio**. SMALL–MEDIUM each. The only "new math" they need is a different f2-combine gather kernel; the SE engine and the entire CLI/IO scaffold are reused verbatim.
- **Cheap math but a NEW data dependency: D-statistic / qpDstat.** The *numerator* is f4 (free), but the heterozygosity *denominator* `(1/M)Σ(a+b−2ab)(c+d−2cd)` is **NOT recoverable from `F2BlockTensor`** — per-SNP allele frequencies are discarded after f2 precompute (VERIFIED: `include/steppe/fstats.hpp:47-78` stores only `f2`, `vpair`, `block_sizes`). D therefore needs a new per-SNP/per-block het reduction in the PRECOMPUTE, not a thin f4 wrapper. MEDIUM.
- **Need-new-machinery:** **qpfstats** (MEDIUM–HIGH; lands on the existing cuSOLVER/emulated-FP64 solve seam but needs the full-basis joint covariance + a shrinkage estimator), **qpGraph** (HIGH; needs a graph→f-stat path algebra + a general nonlinear optimizer steppe lacks), **DATES** (HIGH; needs per-SNP genotypes + a genetic-map LD-decay engine that the architecture *explicitly excludes by design*).
- **RECOMMENDED FIRST TARGET:** **f4** (single + batched all-quartets). It is one cell of the existing `assemble_f4` with `nl=nr=1`, its SE is `sqrt(diag(Q))` off the existing `jackknife_cov`, and **its golden already exists in-tree** (`tests/reference/goldens/at2/csv/golden_fit0_f4.csv` — see Reusable §G). Build the bindings module once alongside it.
- **COMMENT-VS-CODE MISMATCH DISCOVERED (the big one):** the HEAD commit `25fa2ae` message says *"M(cli-2) wire qpwave CLI to run_qpwave … delete dead scaffold"* — **but the commit added only `agentscripts/wire-qpwave-cli-cleanup.js` (67 lines) and touched NO C++.** The qpwave subcommand is **still a scaffold no-op**: `src/app/cli_parse.cpp:191` still routes to `run_not_yet_implemented`, which still exists at `:74`. The engine (`run_qpwave`) IS built and golden-gated; only the CLI binding is fake. **The commit message is false about the code.** (Details + 3 lesser mismatches in §5.)

---

## 1. VERIFIED inventory of reusable steppe machinery

Every row was opened and read by the synthesizer. "Callable" = standalone-callable today vs internal-only.

| Machinery | file:line (verified) | Callable? | Status |
|---|---|---|---|
| **A. f2 input tensor** `F2BlockTensor` (host) | `include/steppe/fstats.hpp:47-78` | reusable as any stat's input | DONE. Stores `f2`, `vpair`, `block_sizes`, `P`, `n_block` ONLY — **no per-SNP freqs** (the D-denominator gap). |
| device twin `DeviceF2Blocks` | `src/device/device_f2_blocks.hpp` (incl. `backend.hpp:41`) | reusable | DONE; resident, zero-D2H. |
| f2-dir reader `read_f2_dir` | `src/app/f2_dir_io.hpp:63` (returns `F2Dir{f2,pop_labels}`) | standalone-callable | DONE; reused by `cmd_qpadm`/`cmd_rotate`. |
| **B. f4 contraction** `ComputeBackend::assemble_f4` (virtual, 2 overloads) | `src/device/backend.hpp:604` (DeviceF2Blocks), `:618` (host F2BlockTensor); identity in contract `:597-602` | **INTERNAL-ONLY** | DONE + golden-gated. Reached ONLY via `run_qpadm_impl`/`run_qpwave_impl` (`qpadm_fit.cpp:265,284`). |
| f4 per-element core `f4_gather_elem` | `src/device/cuda/qpadm_fit_kernels.cu:514-527` | internal | DONE; carries `0.5*(f2(Li,R0)+f2(L0,Rj)−f2(L0,R0)−f2(Li,Rj))` verbatim. |
| f4 CUDA launcher (single) | `src/device/cuda/cuda_backend.cu:1383` | internal | DONE. |
| f4 CUDA launcher (model-batched) | `launch_assemble_f4_gather_models_batched`, decl `src/device/cuda/qpadm_fit_kernels.cuh:242` | internal | DONE — the qpDstat batching primitive. |
| f4 CPU oracle | `src/device/cpu/cpu_backend.cpp:427-496` | internal | DONE; survivor-block drop + LOO. |
| **C. f4 PODs** `F4Blocks` (x_blocks, x_total, x_loo) | `src/device/backend.hpp:96-127` | internal | DONE; the LOO replicates needed for SE are produced in the same call. |
| **D. block-jackknife covariance** `jackknife_cov` | `src/device/backend.hpp:634`; CPU `src/device/cpu/cpu_backend.cpp:503-560` | INTERNAL-ONLY | DONE; `Q = xtau·xtauᵀ/n_block`. `sqrt(diag(Q))` IS the f4/f3/D SE — reuse, no new jackknife. |
| jackknife CUDA cores | `src/device/cuda/qpadm_fit_kernels.cu:535,573` | internal | DONE; generic LOO/xtau, but only ever emits the m×m qpAdm Q (a scalar/vector SE driver is the small new part). |
| survivor-block drop | `survivor_blocks`, `src/device/cpu/cpu_backend.cpp:115-149` | internal | DONE; AT2 `read_f2(remove_na=TRUE)`. |
| **E. het correction** (the admixture-f3 bias term) | `het_correction` used at `src/device/cpu/cpu_backend.cpp:177`; f2 diagonal convention `backend.hpp:54-63` | resident, NOT a downstream consumer today | **PRESENT, not missing** — refutes "the admixture-f3 het term is new" (see Mismatch #4). |
| **F. rank sweep / nested / GLS / ALS** `run_rank_sweep`/`run_popdrop` | `src/core/qpadm/ranktest.hpp:37,57` | internal | DONE; qpWave/qpAdm only. Not needed for raw f-stats. |
| small linalg (LU/inv/Jacobi-SVD) | `src/core/internal/small_linalg.hpp` | internal | DONE; the qpfstats/qpGraph solve building block. |
| precision seam (cuSOLVER + emulated-FP64) | `set_solve_precision`, `src/device/backend.hpp:591` | internal | DONE (default native); the qpfstats promotion seam. |
| S8 batched many-small harness | `fit_models_batched` + default `src/device/backend.hpp:344`; `model_search.hpp` | internal | DONE; the structural template a qpDstat sweep clones. |
| **G. qpWave engine** `run_qpwave` (2 overloads) | `src/core/qpadm/qpadm_fit.cpp:315,321` (+ `_impl` `:279`) | engine standalone-callable; **CLI is a fake** | engine DONE + golden-gated (`tests/reference/test_qpwave_parity.cu`); CLI = scaffold no-op (Mismatch #1). |
| **H. CLI scaffold** subcommands wired | `src/app/cli_parse.cpp:163-300` (qpadm `:163`, qpwave `:173` FAKE, qpadm-rotate `:195`, extract-f2) | template | qpadm/rotate/extract-f2 DONE; qpwave FAKE. |
| name→index resolver `PopResolver` | `src/app/pop_resolver.hpp:44-74` (`resolve`/`label_at`) | standalone | DONE; reused by qpadm/rotate. |
| common flag helpers | `add_common_flags`/`add_output_flags`/`add_qpadm_option_flags`, `cli_parse.cpp:82,97,109` | template | DONE. |
| result emitter primitives | `fmt_double`/`json_double`/`status_str`/`model_feasible`, `src/app/result_emit.cpp:25-66` | template | DONE; CSV/TSV/JSON. |
| GPU plumbing boilerplate | `build_resources`→`upload_f2_blocks_to_device`→run→emit, `cmd_qpadm.cpp:107-117`, `cmd_rotate.cpp:174-185` | copy-paste | DONE; ~30 identical lines. |
| f4 golden CSV (already in-tree) | `tests/reference/goldens/at2/csv/golden_fit0_f4.csv` (cols `pop1..pop4,est,se,z,p`) | — | the literal standalone-f4 output shape. |
| **Python bindings** | — | **DOES NOT EXIST** | no `nanobind`/`pybind`/`NB_MODULE`, no `pyproject.toml`/`setup.py`/`.pyi`. `docs/research/interop-usecases.md` is a PLAN (M(py-1), unbuilt). |

**The one verified hard limit:** there is **NO standalone f4/f3/D/f4-ratio/qpDstat/qpfstats/qpGraph/DATES** anywhere. Grep over `src/ include/ tests/` for `qpdstat|abba|outgroup.?f3|f4.?ratio|qpgraph|dates|qpfstats|run_f4|run_f3|run_dstat` returns **zero implementation symbols**. The only f4 in the tree is `assemble_f4`, and it is matrix-shaped (shared `L0`/`R0` anchors, target prepended — `f4_gather_elem` `qpadm_fit_kernels.cu:518-521`) — a standalone arbitrary-quadruple `f4(A,B;C,D)` reuses the *math* verbatim but needs a **new free-index gather** (the existing one fixes the anchors).

---

## 2. Per-standalone: math, reuse, new work, effort

### f4(A,B;C,D) — **SMALL** (cheap-reuse) — RECOMMENDED FIRST
- **Math (AT2-verified, WebFetch):** `f4(A,B;C,D) = (1/M)Σ(a−b)(c−d) = ½[f2(A,D)+f2(B,C)−f2(A,C)−f2(B,D)]`. This is EXACTLY `assemble_f4`'s contract (`backend.hpp:597-602`). **Zero new math.**
- **Reuse:** `assemble_f4` (B), `jackknife_cov`→`sqrt(diag(Q))` (D), all CLI helpers (H), emitter (H). Golden exists (G).
- **New:** a free-index gather (unfreeze the L0/R0 pivots, or call the existing kernel with `nl=nr=1` and a remapped index set); a public `run_f4(span<quadruple>)`; quad enumerator (mirror `enumerate_pool_subsets`, `cmd_rotate.cpp:48`); an f4 emitter; the bindings module (one-time).

### f3(A;B,C) / outgroup-f3 / admixture-f3 — **SMALL–MEDIUM** (cheap-reuse)
- **Math:** `f3(A;B,C) = ½[f2(A,B)+f2(A,C)−f2(B,C)]`. outgroup-f3 = f3 with A=outgroup (semantic only). admixture-f3 adds the within-target het bias term.
- **Reuse:** jackknife (D); CLI/emitter (H); **the admixture-f3 het correction already exists as a resident quantity** (`het_correction` `cpu_backend.cpp:177`; the f2 diagonal carries `−2·hc_i`, `backend.hpp:54-63`). It is computed but never consumed downstream today — wiring it up is reuse, not new math (Mismatch #4).
- **New:** one 3-term f2-combine gather kernel (CPU+CUDA); entry; emitter; golden. (The new kernel is why it's a notch above f4.)

### D-statistic (ABBA-BABA) — **MEDIUM** (cheap math, NEW data dependency)
- **Math (AT2-verified, WebFetch):** `D(A,B;C,D) = f4(A,B;C,D) / [(1/M)Σ(a+b−2ab)(c+d−2cd)]`. Numerator IS f4 (free); the denominator is a per-SNP heterozygosity product and **NOT identical to f4** (confirmed by the AT2 vignette).
- **Reuse:** the entire f4 numerator path (B,C,D).
- **NEW (the real cost):** the het denominator needs per-SNP allele frequencies, which `F2BlockTensor` **discards** (VERIFIED `fstats.hpp:47-78` — only f2/vpair/block_sizes). So D requires a **new per-SNP/per-block het reduction** added to the PRECOMPUTE (retain a het sidecar in extract-f2, or recompute from genotypes) + a per-block ratio jackknife. **Do NOT schedule D as a thin f4 wrapper** — that was the one place the lenses agreed the docs undersell the cost.

### f4-ratio (admixture proportion α) — **SMALL–MEDIUM** (cheap-reuse)
- **Math:** `α = f4(O,A;X,C)/f4(O,A;B,C)` (two f4s sharing a structure).
- **Reuse:** two `assemble_f4` calls (B); the per-block `x_blocks`/`x_loo` arrays (C) for a ratio jackknife.
- **New:** the ratio-of-f4 point estimate + a jackknife-of-ratio (delta-method) SE composition; 5-pop arg parse; emitter; golden.

### qpDstat (batched D/f4 over many quadruples) — **MEDIUM** (highest ROI, cheap-reuse for f4 mode)
- **Math:** as f4/D, evaluated over an enumerated/filtered set of quadruples.
- **Reuse:** the batching primitive (`launch_assemble_f4_gather_models_batched`, `qpadm_fit_kernels.cuh:242`); the enumerator pattern (`enumerate_pool_subsets`, `cmd_rotate.cpp:48-78`); the S8 multi-GPU rotation dispatch (F). The hard parts (resident-f2 batched gather, multi-GPU shard, deterministic re-sort) are DONE.
- **New:** a free-index *batched* gather (reshape per-model→per-quadruple); enumerator filters (pop sets per slot); a large-output emitter; **and, for D mode, the same het denominator as the D-stat** (so qpDstat-f4 is MEDIUM but qpDstat-D inherits the precompute dependency). This is the highest-leverage new stat (competes with Dsuite).

### qpfstats (f-stat covariance smoothing) — **MEDIUM–HIGH** (need-new-machinery, on-seam)
- **Math:** a regularized/over-determined least-squares smoothing of the full f-stat system under heavy missingness.
- **Reuse:** the f2 tensor (A); jackknife (D); small linalg + the cuSOLVER/emulated-FP64 promotion seam (`set_solve_precision`, `backend.hpp:591`).
- **New:** the JOINT covariance across the full f-stat basis (steppe builds Q only per-model, m=nl·nr — `JackknifeCov`, `backend.hpp:129`) and the shrinkage/smoothing estimator. New statistics, no new IO. Schedule AFTER f3/f4 exist (it consumes them).

### qpGraph (admixture-graph fitting) — **HIGH** (need-new-machinery)
- **Math:** nonlinear optimization of edge lengths + admixture weights minimizing the weighted squared residual between graph-implied and observed f-stats.
- **Reuse:** observed f2/f3/f4 + jackknife covariance as fit targets/weights; the chisq scoring path (`cpu_backend.cpp` chisq) overlaps the objective.
- **New:** a graph topology → implied-f-stat path algebra; a **general constrained nonlinear optimizer** (steppe's ONLY optimizer is the qpAdm-specific bilinear ALS `opt_A`/`opt_B`, `cpu_backend.cpp:956-1007` — not reusable for a graph); optional topology search. Big-bet, deferred per ROADMAP.

### DATES (admixture dating via ancestry-LD decay) — **HIGH** (need-new-machinery, off-seam)
- **Math:** fit a weighted-LD decay curve `D_n = e^(−nd)·D_0` over genetic distance.
- **Reuse:** ONLY the genotype decode IO + the genetic map (`snp_reader.hpp` genpos; used today only for the 5cM jackknife block walk `block_partition_rule.hpp`).
- **New / blocker:** per-SNP-pair LD / decay machinery that does NOT exist (grep: nothing) and that the architecture **excludes by design** (`config.hpp`: LD is "READ, never computed… steppe does not compute LD itself"). Physical position is parsed-and-discarded (`snp_reader` has no physpos field). The f2_blocks data shape is wrong for it (per-pop freqs, no per-SNP-pair detail). Greenfield above the decode; a separate sub-project. Last.

---

## 3. Per-stat INTEGRATION cost (the cmd_+run_+cli+golden+binding pattern)

The CLI/test scaffold is a clean, verified template; the per-stat *integration* is small and patterned. Fixed boilerplate reused near-identically each time (all VERIFIED in §1.H):

1. **f2-dir read** — `read_f2_dir` (`f2_dir_io.hpp:63`), called identically by `cmd_qpadm.cpp:81` / `cmd_rotate.cpp:89`.
2. **name→index** — `PopResolver` (`pop_resolver.hpp:44-74`), at `cmd_qpadm.cpp:88` / `cmd_rotate.cpp:96`.
3. **flags** — `add_common_flags`/`add_output_flags` (`cli_parse.cpp:82,97`), called by every subcommand.
4. **emitter primitives** — `fmt_double`/`json_double`/`status_str` (`result_emit.cpp:25-66`).
5. **GPU plumbing** — `build_resources`→`upload_f2_blocks_to_device`→run→emit, ~30 identical lines (`cmd_qpadm.cpp:107-117`).
6. **subcommand reg** — one `Command` enum value (`cli_args.hpp:34-40`) + one CLI11 block; reuses existing `CliArgs` fields (`--f2-dir/--left/--right/--out/--format/--device/--precision`) — a pure stat needs **no new CliArgs fields** in the common case.
7. **golden test** — copy `tests/cli/test_cli_qpadm.cpp` (write STPF2BK1 fixture → spawn binary → parse CSV/JSON → assert vs golden) + ~6 lines in `tests/CMakeLists.txt` (the `:1141` `add_executable`/`target_link_libraries`/`add_dependencies(steppe_app)`/`add_test` pattern).

**Per-stat NEW code:** ~1 public `run_*` entry + (sometimes) 1 gather kernel + 1 result/emitter shape + 1 golden. **Split ≈ 60–70% reuse / 30–40% new** for f4/f3/f4-ratio; D-stat and qpDstat skew more-new (het normalizer / output volume).

**Bindings:** the nanobind module does not exist — building it (module + `QpAdmResult`/`F2BlockTensor` bindings + `to_dataframe()`, per `interop-usecases.md`) is a ONE-TIME cost, not per-stat. After that, each stat's binding is "bind the new result struct + a `to_dataframe`" — small, patterned. Build it alongside the first standalone stat (f4) per the build-sequence memo ("each stat with its own access surface").

---

## 4. RECOMMENDED ORDER

Cheap-reuse first (they reuse f4 + f2 + jackknife), new-machinery last:

0. **Wire the qpwave CLI** (Mismatch #1) — TINY; the engine + golden already exist, only `cli_parse.cpp:191` needs to call `run_qpwave` instead of `run_not_yet_implemented`. Free win, unblocks the scaffold template's honesty.
1. **f4** (single + batched all-quartets) + **build the nanobind module once** — SMALL; golden in-tree.
2. **f3 / outgroup-f3 / admixture-f3** — SMALL–MEDIUM (one new 3-term kernel; het term already resident).
3. **f4-ratio** — SMALL–MEDIUM (two f4s + ratio jackknife).
4. **qpDstat (batched)** — MEDIUM, highest ROI (reuse the rotation harness); f4-mode first, D-mode after step 5's denominator.
5. **D-statistic** — MEDIUM; lands in the PRECOMPUTE (new per-SNP het reduction), not the fit.
6. **qpfstats** — MEDIUM–HIGH (consumes f3/f4; on the cuSOLVER/emulated-FP64 seam).
7. **qpGraph** — HIGH (new path algebra + optimizer).
8. **DATES** — HIGH, last (separate sub-project; needs physpos + an LD engine the architecture currently excludes).

*(Sequencing nuance: D's precompute het-reduction (5) is a prerequisite for qpDstat's D-mode (4). Schedule qpDstat-f4 at 4, then 5, then qpDstat-D.)*

---

## 5. "VERIFIED, not assumed" + comment-vs-code mismatches

**Every load-bearing claim above was opened and read by the synthesizer** (not taken from a lens, doc, or memory). Specifically re-verified at HEAD `25fa2ae`: `assemble_f4` virtual + identity (`backend.hpp:604,618,597-602`); `f4_gather_elem` math (`qpadm_fit_kernels.cu:514-527`); CPU `assemble_f4` (`cpu_backend.cpp:427-496`); `jackknife_cov` (`backend.hpp:634`); `F4Blocks`/`F2BlockTensor` field lists (`backend.hpp:96`, `fstats.hpp:47-78`); `het_correction` (`cpu_backend.cpp:177`); reader/resolver/flags/emitter (`f2_dir_io.hpp:63`, `pop_resolver.hpp:44`, `cli_parse.cpp:82-109`, `result_emit.cpp:25`); batched kernel (`qpadm_fit_kernels.cuh:242`); `run_qpwave` (`qpadm_fit.cpp:315,321`); the zero-symbol grep for standalone stats; `git show --stat 25fa2ae`; the AT2 f2/f4/D definitions (WebFetch of the f-stats vignette, confirming D's denominator ≠ f4).

**Mismatches found:**

1. **(SEVERE — commit message lies)** HEAD commit `25fa2ae` message: *"M(cli-2) wire qpwave CLI to run_qpwave + CLI dedup + delete dead scaffold, golden_qpwave-gated."* **`git show --stat 25fa2ae` = ONE file changed: `agentscripts/wire-qpwave-cli-cleanup.js` (+67), no C++.** The qpwave subcommand is STILL a no-op: `cli_parse.cpp:191` calls `run_not_yet_implemented`, which STILL exists at `:74`. The engine `run_qpwave` IS built (`qpadm_fit.cpp:315`) and golden-gated (`test_qpwave_parity.cu`). **Truth: qpwave is "engine built + golden-gated, CLI is a fake scaffold."** The commit claiming otherwise is exactly the "trust nothing — open the code" case the mandate targets. (The far-stats/integration lenses correctly flagged the scaffold; none caught that the *latest commit message asserts it was already fixed when it was not*.)

2. **(MINOR)** `docs/architecture.md:208` lists a `qpdstat` CLI subcommand ("planned, P3"). It does NOT exist (grep empty). Correctly labeled "planned" — no contradiction, but confirms standalone-stats are unbuilt.

3. **(MINOR)** `snp_reader` documents/parses `<physpos>` but the `SnpTable` struct has no physpos field — physical position is **parsed-and-discarded**; only genpos is kept (for the coarse jackknife block walk). Harmless now; a load-bearing gap for any future DATES work.

4. **(REFRAME, not a code bug)** Some planning notes imply "the admixture-f3 het-bias term is new." It is NOT: the within-pop het correction is a resident, computed quantity (`het_correction`, `cpu_backend.cpp:177`; the f2 diagonal carries `−2·hc_i`, `backend.hpp:54-63`), today computed-but-not-consumed downstream. Wiring it into admixture-f3 is reuse.

**No other doc/code mismatch found.** The repo prose (ROADMAP §3 "no standalone entry point today", `docs/design/fit-engine.md:40`, `interop-usecases.md` as a plan) is ACCURATE on the *existence* claims — the f4 math lives internal-only, no standalone stat / no Python binding exists. The single false artifact is the commit message in #1.

**Sources:** AT2 f-statistics vignette (https://uqrmaie1.github.io/admixtools/articles/fstats.html — f2/f4/D definitions, D-denominator ≠ f4, WebFetch-confirmed); steppe source as cited file:line throughout.
