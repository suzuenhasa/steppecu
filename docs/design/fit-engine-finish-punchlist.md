# FIT-ENGINE BACKEND — FINISH PUNCH-LIST (step 1 of backend-first)

Read-only audit synthesis. Branch `phase2-fit-engine` @ `9b0db27` (main @ `0db07be`).
Scope: the Phase-2 qpAdm/qpWave fit engine (S3–S8 + the qpAdm/qpWave domain). Three
independent audit lenses (stage-coverage, AT2-feature-gaps, domain-correctness) reconciled
into one ranked list. Every claim cites a verified `file:line` / design ref / AT2 behavior.

---

## (1) HEADLINE

**The fit-engine backend is structurally complete but not yet contractually complete.** All
of S3–S8 is built and golden-gated on the GPU (assemble_f4, weighted jackknife_cov, the rank
sweep with both the small-Jacobi and the NRBIG cuSOLVER-gesvd paths, GLS opt_A/opt_B ALS +
the Σw=1 constrained solve, block-jackknife SE with the opt-in JackknifePolicy, and the S8
batched rotation). The big refactor is done. What is **missing is not new math — it is the
contract closure**: the domain-outcome taxonomy is incomplete and **untested**, a shipped
public entry (`run_qpwave`) has zero test coverage, and one AT2 correctness behavior
(missing-block NA handling) the design itself said to add "before the at-scale search" was
never added even though the at-scale search (M(fit-6)) shipped.

**FINISH-NOW count: 6 items** (3 are tight/cheap contract closures; 1 is a real correctness
gap on sparse real AADR; 2 are test/API-hygiene). None require new GPU kernels of substance.

The single most important one is the **domain-outcome test** (the M(fit-5) acceptance gate
that is named in `architecture.md:803` §13 and the §18 DoD `architecture.md:939`, but has no
test) — closely coupled to the **`ChisqUndefined` status that the spec mandates but the enum
omits**. After those plus the `run_qpwave` golden and the missing-block decision, the backend
can honestly be called "done" and ready for the CLI/bindings step.

The three lenses agreed strongly. Where they differed it was effort granularity, not verdict.

---

## (2) THE PUNCH-LIST

Classification: **FINISH-NOW** = a real backend gap to close before CLI/bindings · **DEFER**
= defensible deferral (additive later) · **NEW-FEATURE** = genuine step-3 scope (its own
statistic + CLI/bindings). Effort S/M/L. Parity risk = risk to AT2 numeric parity if shipped
as-is.

### FINISH-NOW (close before exposing the fit via CLI/bindings)

| # | Item | What it is + ref | Class | Effort | Parity risk | One-line approach |
|---|---|---|---|---|---|---|
| F1 | **Missing-block / NA handling (`est_to_loo_nafix`)** | AT2 excludes NA blocks from the LOO/jackknife; a `vpair[i,j,b]==0` makes that f4 entry NA and must NOT be imputed 0 (biases toward 0, inflates variance). NEVER added: `cpu_backend.cpp:682` explicitly states "With no missing blocks (M(fit-1), OQ-12)"; no NA mask in the LOO loops or kernels. Design OQ-12 (`fit-engine.md:681-685`) said "add NA handling **before the at-scale search**" — the at-scale search (M(fit-6)) already shipped, so this is **overdue by the design's own gate**. | **FINISH-NOW** | M–L | **high** (on sparse aDNA) | Implement the NA-aware LOO (skip `vpair==0` per-(entry,block)) **or** prove + hard-document that the resident global-intersection (maxmiss=0) f2 can never contain empty blocks and gate the CLI to that precondition. |
| F2 | **M(fit-5) domain-outcome TEST** | The M(fit-5) gate (`fit-engine.md:432`; spec §13 `architecture.md:803`; §18 DoD `architecture.md:939` — outcomes "returned as values and covered by a test") requires a deliberately collinear/rank-deficient/non-SPD model to return a **status VALUE, not a crash/NaN**. No such test exists: every `status` assertion in both `test_qpadm_parity.cu` and `test_qpadm_rotation.cu` checks `== Status::Ok`; grep for an expected `RankDeficient`/`NonSpdCovariance`/`collinear`/`degenerate` model in `tests/` = 0 hits. The machinery is built + wired; only the contract-required test is missing. | **FINISH-NOW** | S–M | low (test only) | Add a domain-outcome test: a collinear left set → `RankDeficient`; an indefinite-but-nonsingular Q fixture → `NonSpdCovariance`; a `dof≤0` model → `ChisqUndefined` (after F3). Assert status values, no crash/NaN. |
| F3 | **`ChisqUndefined` status value** | Spec §10 lists **three** domain outcomes incl. `STEPPE_ERR_CHISQ_UNDEFINED` "dof ≤ 0 or χ² not computable" (`architecture.md:676`); ROADMAP M(fit-5) names all three (`ROADMAP.md:92`); the design header documents it (`fit-engine.md:298`). But `error.hpp:21-41` defines only `RankDeficient`/`NonSpdCovariance` — `ChisqUndefined` is **absent** (already self-flagged HIGH in `docs/cleanup/include-error.md:43-47`). Live consequence: `qpadm_fit.cpp:98` calls `pchisq_upper(chisq, dof)` which returns **NaN** for `dof≤0` (`pchisq.hpp:89`) into `res.p` while `res.status` stays **Ok** — a rotation consumer filtering on `status==Ok` accepts a NaN-`p` model. | **FINISH-NOW** | S | low | Add the `ChisqUndefined` enumerator + a `dof≤0 ⇒ Status::ChisqUndefined` guard in `qpadm_fit.cpp:98` and the Cuda assemble path. One enumerator; §16 makes a later add a deliberate MINOR-bump churn. |
| F4 | **`run_qpwave` golden + test** | `run_qpwave` is a first-class public entry (4 overloads `qpadm.hpp:226-237`; impl `qpadm_fit.cpp:266-309` via `run_qpwave_impl`) with its own `QpWaveResult` (`qpadm.hpp:212-222`) — but **zero test coverage**: grep for `run_qpwave(` in `tests/` = only the comment label "M(fit-2) RANK TEST / qpWave" on `run_qpadm` paths, no actual call. qpWave's distinguishing semantic (no target prepend; `left[0]` is the reference row, `qpadm_fit.cpp:270-271`) is unvalidated. The design's own reference model #1 "qpWave 2-way feasible" (`fit-engine.md:548`, a [PROPOSAL]) was never pinned. Step 3 wants qpWave with its own CLI — exposing it untested is a real gap. | **FINISH-NOW** | S–M | med | Pin an AT2 `qpwave()` golden; add a test that calls `run_qpwave` directly and gates `est_rank`/rankdrop. Reuses the existing rank-sweep machinery. |
| F5 | **`opts.constrained` dead public flag** | `QpAdmOptions::constrained` exists (`qpadm.hpp:73`, documented "reserved") but is **never read** in any solve path (grep confirms the only "constrained" in the solve is the **Σw=1 equality-constraint** `solve_constrained_weights`, unconditional and unrelated to non-negativity). This is NOT AT2's `constrained=TRUE` (non-negative weights). A public reserved-but-ignored flag is an API trap once it lands in bindings. | **FINISH-NOW** (decision) | S | low | Decide: either implement AT2's non-negative-weights constrained solve, **or** remove/clearly mark the dead field before it ships in the public struct. Default is `false`, so the feature is low-priority; the dead public field is the hygiene issue. |
| F6 | **Determinism gate covers only a subset of result fields** | The G1==G2 bit-identical contract (`fit-engine.md:374`, §18) is tested (`test_qpadm_rotation.cu:452-467`) but the memcmp covers only `model_index/status/f4rank/weight/p/chisq/se`. It does NOT memcmp `z`, `dof`, `est_rank`, `rank_chisq/rank_dof`, or the `rankdrop_*` / `popdrop_*` arrays — all reported fields. A nondeterminism in any would pass. | **FINISH-NOW** | S | low | Widen the determinism memcmp to the full `QpAdmResult`. Cheap hardening. |

### DEFER (defensible — additive later, name as a known limitation)

| # | Item | What it is + ref | Class | Effort | Parity risk | One-line approach |
|---|---|---|---|---|---|---|
| D1 | **`boot=N` bootstrap SE** | Only the deterministic delete-1 block jackknife exists (`nested_models.cpp` codes the AT2 `!boot` branch only); no RNG seam, no `boot` field. AT2 `boot=N` is RNG-fragile (§12). Design listed it "(6, stretch, not first-milestone)" `fit-engine.md:551`; all goldens are `boot=FALSE` (`fit-engine.md:508`). | **DEFER** | M | n/a (no golden) | Track as an additive option; jackknife is the canonical deterministic SE and is fully gated. Name as not-supported at the CLI. |
| D2 | **`fudge_twice` / `getcov` / `return_f4`** | AT2 output-shaping/numeric toggles; none present (grep = 0). `getcov`/`return_f4` are reporting toggles trivially addable at the CLI/binding layer; `fudge_twice` is a niche double-ridge with no golden. | **DEFER** | S | low | Add at the CLI/binding layer when a consumer needs them; not backend-completeness blockers. |
| D3 | **maxmiss fit-side parity** | maxmiss is a **precompute** concern, not S3–S8. `FilterConfig::geno_max_missing` (`config.hpp`) already documents that steppe uses the PLINK per-individual denominator vs AT2's per-SNP-over-populations. The fit correctly consumes whatever missingness the f2 already reflects. | **DEFER** (precompute) | — | low | Close OQ-11 with a one-line "resident global-intersection f2 only" boundary statement (`fit-engine.md:676-679`) so the maxmiss/allsnps expectation is explicit. |
| D4 | **cuSOLVER emulated-FP64 SOLVE promotion seam** | The seam is real (`engage_solver_precision`/`CusolverMathModeScope`, design §1.4) but **degrades to native** because CUDA 13.0 / cuSOLVER 12.0 exposes no FP64-emulated mode (`fit-engine.md:267-271`, `qpadm_fit.cpp:73`). Default native; parity byte-identical. | **DEFER** (toolkit-blocked) | — | none | Correct as designed; nothing to finish until a newer cuSOLVER ships the mode. |
| D5 | **Multi-GPU rotation (S8 G≥2)** | `TODO(multigpu-host-bounce)` `model_search.cpp:165-166,280-282`; no-P2P 5090 host bounce caps the speedup (`fit-engine.md:340-353`). Run single-GPU. | **DEFER** (HW-blocked) | L | none (correctness) | Defensible; not a backend correctness gap. Revisit on P2P-capable hardware. |
| D6 | **OQ-11 tiered/streamed f2 at large P** | Resident-only first cut; a real ceiling at thousands of pops but a documented boundary (`fit-engine.md:679`). | **DEFER** | L | low | Close the boundary statement now (with D3); build the tiered path when P exceeds the resident envelope. |

### NEW-FEATURE (step 3 — its own statistic + CLI/bindings)

| # | Item | What it is + ref | Class | Effort | Parity risk | One-line approach |
|---|---|---|---|---|---|---|
| N1 | **`allsnps=TRUE` (per-quartet maximal SNP sets)** | AT2 selects different SNPs per f4-statistic for sparse data (the higher-power aDNA mode). steppe is global-intersection (maxmiss=0) only: the fit consumes one shared `DeviceF2Blocks`; `assemble_f4` is a pure 4-slab gather from that one tensor. **Structurally unreachable from the f2→f4 fit seam** — it requires re-deriving f4 per-quartet from the genotype/AF stream (a precompute-layer capability) with deterministic-reduction care. | **NEW-FEATURE** | L | n/a | Step-3 statistic with its own CLI/bindings + the deterministic-reduction guarantee. Name as a documented limitation at the current CLI so aDNA users do not assume AT2-equivalent power. |

### CONFIRMED-BUILT / N-A (do not re-flag)

- **Pseudo-haploid / ploidy** — honored entirely UPSTREAM (precompute: `decode_af`, `finalize_af(...ploidy)`); the fit consumes already-corrected f2 and is correctly ploidy-agnostic. Not a fit gap.
- **RankDeficient / NonSpdCovariance as VALUES** — detection is BUILT both backends (GPU Cholesky `potrf` non-SPD `cuda_backend.cu:1551-1559`; GLS/rank-sweep rank-deficient emit sites). Only the **test** (F2) and the `ChisqUndefined` sibling (F3) are missing. (Caveat: see the CPU-oracle note below.)
- **dofdiff / rankdrop / popdrop nested tables** — BUILT + golden-gated (`qpadm.hpp:146-153`, filled `qpadm_fit.cpp:164-182`, gated vs `golden_fit0`/`golden_fit1_NRBIG`).
- **S3–S8, JackknifePolicy{None,FeasibleOnly,All}, NRBIG large path, S8 batched rotation** — BUILT + golden-gated.

**One open correctness caveat worth a line in the F2 test (LOW, fold into F2):** the GPU
detects non-SPD via Cholesky `potrf` (`cuda_backend.cu:1551-1559`), but the CPU oracle inverts
Q via **LU with partial pivoting** (`small_linalg.hpp` `inverse`→`lu_factor`), flagging only a
zero pivot. An indefinite-but-nonsingular Q (true non-SPD) inverts cleanly under LU → oracle
returns `Ok` while the GPU returns `NonSpdCovariance` — the two backends can disagree on the
exact outcome M(fit-5) names, and it is currently untested. Either reconcile the oracle to a
Cholesky check or document the divergence and cover it in the F2 fixture.

---

## (3) RECOMMENDED FINISH SET (the minimal "backend done" bar)

To call the fit-engine backend **done and ready for the CLI step**, finish these, in order:

1. **F3 — `ChisqUndefined`** (S, ~one enumerator + a `dof≤0` guard). Stops the silent
   NaN-`p`-with-`status==Ok` bug; makes the taxonomy match the spec. Do this first because F2
   depends on it.
2. **F2 — the M(fit-5) domain-outcome test** (S–M). This is the named acceptance gate
   (`architecture.md:803`, §18 DoD `architecture.md:939`) and the single most load-bearing
   missing test — a CLI that crashes on a degenerate user model instead of returning a status
   is the canonical qpAdm UX failure. Fold the CPU-oracle non-SPD caveat into this fixture.
3. **F4 — the `run_qpwave` golden + direct test** (S–M). A shipped public entry must not go
   into bindings unproven; cheap, reuses the rank-sweep.
4. **F1 — the missing-block decision** (M–L). This is the one with real numeric-parity risk
   on sparse real AADR (the project's stated envelope). **Minimal acceptable closure:** if it
   can be proven that the resident global-intersection f2 for dense AADR never produces an
   empty block, document that precondition and hard-gate the CLI to maxmiss=0 — that converts
   a latent silent bias into an explicit, honest limitation. Full closure is the NA-aware LOO.
   Either is acceptable to ship the CLI; silently doing neither is not.
5. **F5 — the `opts.constrained` dead-flag decision** (S) and **F6 — widen the determinism
   memcmp** (S). API hygiene + a free determinism hardening; do them in the same pass.

**Rationale for what to defer:** `boot=N` (D1), `fudge_twice`/`getcov`/`return_f4` (D2), the
cuSOLVER emulated seam (D4), multi-GPU rotation (D5), and OQ-11 tiered f2 (D6) are all either
additive options with no golden/consumer demand, toolkit-blocked, or hardware-blocked — none
changes the parity of the path being exposed, and the jackknife is the canonical deterministic
SE. `allsnps=TRUE` (N1) is structurally a precompute/genotype-layer statistic the f2→f4 fit
seam cannot produce; it belongs to step 3 with its own CLI. Maxmiss fit-side (D3) is a
precompute concern. Defer all of these — but **name** allsnps and boot=N as known limitations
at the CLI boundary so users do not assume AT2-equivalent behavior.

---

## (4) THE BIG ADJUDICATIONS (explicit)

- **`allsnps=TRUE` — finish vs new-feature? → NEW-FEATURE (step 3), not a fit-backend finish
  item.** All three lenses agreed. It is structurally unreachable from the fit: the fit
  consumes one shared global-intersection `DeviceF2Blocks` and `assemble_f4` is a pure 4-slab
  gather; per-quartet maximal SNP sets require re-deriving f4 from the genotype/AF stream (a
  precompute capability) with a deterministic-reduction guarantee. Defensible to defer — but it
  must be a **documented limitation** at the CLI, because aDNA users expect the higher-power
  mode.

- **Bootstrap (`boot=N`) — finish vs defer? → DEFER (new-feature class).** The deterministic
  delete-1 jackknife is the canonical, parity-anchored SE and is fully built + gated; all
  goldens are `boot=FALSE`. Bootstrap adds an RNG-fragile (§12) path with no golden and no
  consumer demand. Honest backend completeness does not require it. Track as additive; name as
  not-supported.

- **Standalone qpWave — FINISH-NOW (test, not new code).** `run_qpwave` is already a
  first-class public entry (4 overloads, its own result type, listed in `architecture.md`); it
  is **not** missing — it is **untested** (zero test invocations). Because step 3 wants qpWave
  with its own CLI/bindings, the gap to close now is a golden + a direct test, not new math.

- **M(fit-5) taxonomy + test — FINISH-NOW, and the milestone's own gate is currently UNMET.**
  The status machinery is built and wired, which is why the milestone reads BUILT — but the
  acceptance gate (`fit-engine.md:432` / §13 `architecture.md:803` / §18 DoD
  `architecture.md:939`) requires the domain-outcome TEST, which does not exist, AND the
  taxonomy is missing `ChisqUndefined` (spec-named, enum-absent, with a live silent-NaN
  consequence at `qpadm_fit.cpp:98`). Both close cheaply (F2 + F3). This is the headline finish
  item.

- **NA / missing-block handling — FINISH-NOW (or an explicit hard-gate).** The design itself
  (OQ-12, `fit-engine.md:681-685`) conditioned this on "before the at-scale search," and the
  at-scale search shipped — so by the design's own gate it is **overdue**, not a clean
  deferral. It is genuine AT2 behavior (`est_to_loo_nafix`), not a nice-to-have. It is the only
  finish item with high parity risk on the real heterogeneous-coverage AADR the project
  targets. If the no-empty-block precondition can be proven for the dense global-intersection
  f2, documenting + hard-gating the CLI to maxmiss=0 is an acceptable minimal closure; shipping
  the at-scale search on sparse data with neither is not.

---

### Files cited (all verified read-only)
`include/steppe/error.hpp:11-13,21-41` · `include/steppe/qpadm.hpp:73,146-153,212-237` ·
`src/core/qpadm/qpadm_fit.cpp:62,73,80,98,160-182,266-309` ·
`src/core/qpadm/qpadm_fit.hpp:37` · `src/core/internal/pchisq.hpp:86-90` ·
`src/core/qpadm/model_search.cpp:165-166,280-282` ·
`src/device/cpu/cpu_backend.cpp:682` · `src/device/cuda/cuda_backend.cu:1551-1559` ·
`docs/design/fit-engine.md:267-271,298,374,432,508,548,551,676-679,681-685` ·
`docs/architecture.md:665-678,803,918,939` · `docs/ROADMAP.md:92` ·
`docs/cleanup/include-error.md:43-47` ·
`tests/reference/test_qpadm_parity.cu:273,281` ·
`tests/reference/test_qpadm_rotation.cu:452-467`.
