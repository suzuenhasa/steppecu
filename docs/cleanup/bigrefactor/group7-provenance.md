# Group-7 Duplication — Provenance & Canonical-Copy Report

Read-only git + workflow archaeology. Repo `/home/suzunik/steppe`, HEAD `ba1f0b9` (2026-06-21).
Every claim cites a real commit/file/function, verified via `git show <commit>^:` vs `git show <commit>:` symbol-count deltas (parent 0 → N proves birth, not a mere touch).

Two device-side duplication clusters fall under group-7:
- **D1 — `block_sink.cu`**: the `HostRamSink` (TIER 1) vs `DiskSink` (TIER 2) pinned-ring + background-writer twin. **ALREADY DEDUPED at HEAD** (`b1bd620`, collapsed into one `StagingRing` now living in `block_sink.cuh`, 16 refs; the old `struct HostRamSink`/`struct DiskSink` symbols return 0 in `block_sink.cu`).
- **D2 — `qpadm_fit_kernels.cu`**: the small / `_large` / `_models` kernel twins. **NOT done** — 100 `_large|_models` hits remain at HEAD. This is the live dedup that `agentscripts/fix-group7-device.js` (phase "D2 twin-collapse") targets.

---

## (1) HEADLINE TIMELINE — which copy entered first, when, and why

**D2 — `qpadm_fit_kernels.cu` (file born `3e78c79` M(fit-4), 2026-06-19 20:07):**

- The **small `dev_opt_A` / `dev_opt_B` / `dev_chisq_of` / `als_kernel` / `weights_chisq_kernel`** copies entered at **`3e78c79`** (2026-06-19 20:07, workflow `m-fit-gpu.js`, milestone **M(fit-4)** "the qpAdm fit ON THE GPU — the production CUDA-backend path") because that milestone built the first GPU fit as **one-thread-per-model kernels with compile-time-fixed per-thread local arrays** (`kQpMaxNl=5`, `kQpMaxNr=10`, `double Vfull[MAXNR*MAXNR]` register/local-resident — the fast path). The **`_large` twins** (`dev_opt_A_large`, `dev_opt_B_large`, `dev_chisq_of_large`, `als_large_kernel`, `weights_chisq_large_kernel`) entered later at **`01a94d5`** (2026-06-20 00:39, workflow `gpu-large-models.js`, "the qpAdm fit/rank-test runs ARBITRARY model sizes ON THE GPU via cuSOLVER + dynamic VRAM workspace") because a model with `nl>5`/`nr>10` (e.g. `nr=39`) **overflows the small kernel's fixed local arrays and is UNRUNNABLE**; the large twin routes SVD through cuSOLVER and sizes the ALS/weight/chisq working set in **runtime VRAM scratch, not per-thread local arrays**.

- The **single-model `assemble_f4_gather_kernel` / `f4_loo_total_kernel` / `f4_xtau_kernel`** copies entered at **`3e78c79`** (2026-06-19 20:07, `m-fit-gpu.js`, M(fit-4)) as the one-model-per-fit f4 assembly. The **`_models` twins** (`assemble_f4_gather_models_kernel`, `f4_loo_total_models_kernel`, `f4_xtau_models_kernel`) entered later at **`7852404`** (2026-06-20 04:53, workflow `m-fit-6-rotation.js`, milestone **M(fit-6) S8** "the qpAdm model-space ROTATION on the GPU(s) — batched + multi-GPU sharded") because the S8 rotation must fit+rank-test **thousands of candidate models BATCHED in one launch and sharded across both GPUs** — the `_models` kernels carry a per-model index/base offset so one grid-stride launch covers the whole batch (the single-model loop "is NOT acceptable" at that scale).

- The **`loo_large_batched_kernel`** (a 4th copy of the constrained weight-solve + ALS loop, inlining the `als_large` + `weights_chisq_large` math) entered LAST at **`8f485b7`** (2026-06-20 13:03, workflow `parallelize-large-loo-se.js`, "parallelize the large-model jackknife LOO SE") because the large path's LOO ran **single-threaded — ~701 leave-one-block-out refits looped on ONE CUDA thread (~371 s for NRBIG)**; the fix gave it a per-`(model,block)` parallel kernel on runtime-sized scratch, re-inlining the weight-solve math rather than sharing it.

**D1 — `block_sink.cu` (file born `176a07d`, 2026-06-19 14:30):**

- `HostRamSink` (TIER 1) and `DiskSink` (TIER 2) are **CO-BORN in the same commit `176a07d`** (2026-06-19 14:30, workflow `m5-tiered-streaming.js`, "adaptive tiered f2_blocks output — Resident/HostRam/Disk"). **Neither preceded the other** — the duplication is a within-commit copy-paste, not a first-then-later sequence. Within the file `HostRamSink` is physically defined first (≈line 42), `DiskSink` after (≈line 181). They were forked because the M5 brief mandated the *same* triple-buffered pinned-staging-ring **mechanism** for both tiers but did not mandate a *shared class*; the implementer wrote the plumbing twice, differing only in the writer's drain action (HostRam `memcpy` into the dst tensor vs Disk `pwrite` to the file region) and the throw tag. Already collapsed into one `StagingRing` by `b1bd620` (2026-06-21 06:23).

---

## (2) Provenance table — original vs duplicate vs canonical

| Duplicated region | Original copy (commit / date / why) | Duplicate copy (commit / date / why) | CANONICAL (kept body) + rationale |
|---|---|---|---|
| `dev_opt_A` / `dev_opt_A_large` | `3e78c79` 2026-06-19 M(fit-4) — register/local-resident fast path | `01a94d5` 2026-06-20 gpu-large-models — VRAM-scratch, cap-lift for nl>5/nr>10 | **small `dev_opt_A`** — entered first; one core takes a scratch pointer, small caller passes its own local arrays (preserve register-residency) |
| `dev_opt_B` / `dev_opt_B_large` | `3e78c79` 2026-06-19 M(fit-4) | `01a94d5` 2026-06-20 gpu-large-models | **small `dev_opt_B`** — same rationale |
| `dev_chisq_of` / `dev_chisq_of_large` | `3e78c79` 2026-06-19 M(fit-4) | `01a94d5` 2026-06-20 gpu-large-models | **small `dev_chisq_of`** — same rationale |
| `als_kernel` / `als_large_kernel` (the opt_A→opt_B ALS loop) | `3e78c79` 2026-06-19 M(fit-4) | `01a94d5` 2026-06-20 gpu-large-models | **small `als_kernel`** — entered first; core parameterized on scratch pointer |
| `weights_chisq_kernel` / `weights_chisq_large_kernel` (constrained weight solve) | `3e78c79` 2026-06-19 M(fit-4) | `01a94d5` 2026-06-20 gpu-large-models | **small `weights_chisq_kernel`** — entered first; SE/weights/chisq MUST stay bit-identical |
| `assemble_f4_gather_kernel` / `assemble_f4_gather_models_kernel` | `3e78c79` 2026-06-19 M(fit-4) — single-model | `7852404` 2026-06-20 M(fit-6) S8 — model-batched index/base | **single-model `assemble_f4_gather_kernel`** — entered first; single-model passes base=0, batched passes per-model slice |
| `f4_loo_total_kernel` / `f4_loo_total_models_kernel` | `3e78c79` 2026-06-19 M(fit-4) — single-model | `7852404` 2026-06-20 M(fit-6) S8 | **single-model `f4_loo_total_kernel`** — same rationale |
| `f4_xtau_kernel` / `f4_xtau_models_kernel` | `3e78c79` 2026-06-19 M(fit-4) — single-model | `7852404` 2026-06-20 M(fit-6) S8 | **single-model `f4_xtau_kernel`** — same rationale |
| Constrained weight-solve + ALS loop, 4th copy: `loo_large_batched_kernel` (inlines `als_large` + `weights_chisq_large`) | (math originates `3e78c79` small; `01a94d5` `_large`) | `8f485b7` 2026-06-20 parallelize-large-loo-se — per-(model,block) parallel LOO | **collapse into the shared weight-solve/ALS core** (root = small `3e78c79`); this inlined copy is the NEWEST and the easiest to drift — see gotcha §4 |
| `HostRamSink` (TIER 1) / `DiskSink` (TIER 2) ring + writer (D1) | `176a07d` 2026-06-19 m5-tiered-streaming — TIER 1, defined first in file | `176a07d` 2026-06-19 m5-tiered-streaming — TIER 2, same commit (co-born) | **`StagingRing`** (already done, `b1bd620`); template body = HostRamSink's structure, but the **fail-fast SEMANTICS are DiskSink's** (retrofitted onto HostRam by the HIGH fix `9dbc610`) |

Note: `7852404` (M(fit-6) S8) added the `_models` *kernels* but **reused** the existing templated `dev_opt_A` / `dev_chisq_of` / `dev_als_weights` helpers unchanged — so for those device helpers the only true twin is small-vs-`_large` (`3e78c79` vs `01a94d5`), not a three-way fork.

---

## (3) THE WHY — the design tension that produced the twins

Every D2 twin is a **deliberate copy-then-specialize**, and in every case the *first* copy is the simpler/smaller one with the later copy lifting a specific constraint:

- **Small fixed-local-array vs large VRAM-scratch (`3e78c79` → `01a94d5`).** The first GPU fit (M(fit-4)) was built as one-thread-per-model kernels with *compile-time-fixed* per-thread arrays (`kQpMaxNl=5`, `kQpMaxNr=10`) — maximally register/local-resident, fast, and bit-parity-checked against the CPU oracle. That design caps model size: anything with `nl>5`/`nr>10` overflows the local arrays and cannot launch. The cap-lift milestone (`gpu-large-models`) had to introduce cuSOLVER-routed SVD and **runtime-sized device workspace**, which is a structurally different kernel body (pointer + size args instead of fixed locals). The workflow brief explicitly weighed unify-vs-fork and chose to **keep the small fast-path for `nl<=5,nr<=10` (bit-parity preserved) and add the `_large` path for everything else** — sharing was *deliberately deferred* to avoid regressing the register-resident hot path.

- **Single-model vs model-batched (`3e78c79` → `7852404`).** M(fit-4) fit one model per call. The S8 rotation (M(fit-6)) is the production envelope: thousands of (often large) models, batched in one launch, sharded across both RTX 5090s. The `_models` kernels add a model-index/base-offset axis so a grid-stride launch processes the whole batch. Again a deliberate fork — the batched body is the single-model body with `base` generalized from 0 to a per-model slice; sharing was deferred to ship the rotation quickly on top of the cap-lift kernels.

- **Sequential vs parallel large-LOO (`01a94d5` → `8f485b7`).** The large path inherited a single-threaded LOO (~701 sequential refits, ~371 s). `parallelize-large-loo-se` added `loo_large_batched_kernel`, which **re-inlined** the `als_large` + `weights_chisq_large` math into a per-`(model,block)` parallel kernel rather than calling a shared helper — producing a 4th copy of the weight-solve/ALS concept. This is the clearest case of duplication driven by deadline + the kernel-locality constraint (inlining avoids cross-kernel device-function call overhead in the hot LOO loop).

- **HostRam vs Disk (`176a07d`, co-born).** The M5 tiered-output brief specified ONE pluggable sink interface (`begin/spill_block/finish`) with three impls (Resident/HostRam/Disk), and required tiers 1+2 to share the *same* pinned-ring + triple-buffered writer *mechanism* — but mandated the mechanism, not a shared class. The two sinks were copy-pasted in the creation commit, differing only in the drain action and throw tag. Not a first-then-later specialize: a **simultaneous** two-way copy.

---

## (4) Note for the in-progress `fix-group7-device` dedup (D2)

- **Canonical = the first-introduced copy in every pair = `3e78c79` M(fit-4).** Collapse each twin to ONE `__device__` core per concept **parameterized on the scratch pointer**; the small path passes pointers to its OWN local arrays (preserve register-residency, no perf regression); the `_large` (`01a94d5`) and `_models` (`7852404`) callers supply VRAM / per-model-slice scratch. This matches the `fix-group7-device.js` D2 spec verbatim ("ONE device core per concept parameterized on the scratch pointer; the small path passes pointers to its own local arrays … SE/weights/chisq MUST stay BIT-IDENTICAL").
  - `dev_opt_A/B`, `dev_chisq_of`, the constrained weight solve, and the ALS loop → small copy (`3e78c79`) canonical.
  - `assemble_f4_gather` / `f4_loo_total` / `f4_xtau` → single-model copy (`3e78c79`) canonical (single-model passes base=0).

- **GOTCHA 1 — the 4th weight-solve copy is the NEWEST.** `loo_large_batched_kernel` (`8f485b7`, 2026-06-20) **inlines** the `als_large` + `weights_chisq_large` math (file comment ≈line 885: "runs the EXACT als_large + weights_chisq_large"; the inlined ALS loop ≈line 947, the inlined weight chisq ≈lines 954-955). It is NOT a suffix-renamed helper, so a naive symbol-rename dedup will miss it. When collapsing the weight-solve/ALS core, this inlined block must be redirected to the same shared core or it will silently drift from the canonical math. There is **no `weights_chisq_models_kernel`** — the model-batched weight solve lives inlined here.

- **GOTCHA 2 — `_models` did not re-fork the device helpers.** `7852404` reused `dev_opt_A`/`dev_chisq_of`/`dev_als_weights` unchanged; do not expect three-way twins for those — only small-vs-`_large`.

- **GOTCHA 3 — D1 is already done.** `block_sink.cu` was deduped by `b1bd620` into `StagingRing` (now in `block_sink.cuh`); do not re-touch it. Preserve the `9dbc610` HIGH fail-fast: the unified ring uses **DiskSink's** fail-fast-on-bad-sync semantics (HostRam was retrofitted to match), so the canonical *semantics* for D1 are DiskSink's even though HostRamSink was the physical-first/structural template.

---

## Key files (absolute)

- `/home/suzunik/steppe/src/device/cuda/qpadm_fit_kernels.cu` — D2 twins (LIVE; 100 `_large|_models` hits at HEAD)
- `/home/suzunik/steppe/src/device/cuda/block_sink.cu` + `block_sink.cuh` — D1 (already `StagingRing`-deduped at HEAD)
- `/home/suzunik/steppe/agentscripts/m-fit-gpu.js` (`3e78c79`), `gpu-large-models.js` (`01a94d5`), `m-fit-6-rotation.js` (`7852404`), `parallelize-large-loo-se.js` (`8f485b7`), `m5-tiered-streaming.js` (`176a07d`), `fix-group7-device.js` (the dedup workflow)

### Verification anchors
HEAD `ba1f0b9`. Symbol-birth deltas: `dev_opt_A_large` 0→3 across `01a94d5^`→`01a94d5`; `assemble_f4_gather_models_kernel` 0→2 across `7852404^`→`7852404`; `loo_large_batched_kernel` 0→2 across `8f485b7^`→`8f485b7`; small `dev_opt_A` 0→4 across `3e78c79^`→`3e78c79`. `block_sink.cu` at HEAD: `StagingRing`=0 / `struct HostRamSink|struct DiskSink`=0 (moved to `.cuh`, 16 `StagingRing` refs). `qpadm_fit_kernels.cu` at HEAD: `_large|_models`=100.
