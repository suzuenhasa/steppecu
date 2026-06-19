# MISSING-FIXES backlog — forensic completeness audit of every fix-pass

> **Scope.** Steppe = CUDA-13 / Blackwell reimplementation of ADMIXTOOLS 2 f-statistics, branch `m4.5-multigpu`.
> Over many sessions we ran a series of fix-pass / implementation workflows, each with a *fixer + independent-verdict* loop meant to **commit-on-PASS / revert-on-FAIL**. Some runs hit API errors, used skip-and-continue, or could not re-run their verdict on the box — so a declared item could be reported "done" without its fix actually landing.
> This document is the cross-check of **every declared item** against git reality + the live tree.
>
> **Audit reference point.** The forensic sweep was taken at `HEAD = a41d67a` with a clean tree. This document was written slightly later and re-verified against the live tree; the **one material change** since the snapshot is that the in-flight perf-fix-pass **has now started landing P4** (uncommitted working-tree changes — see §0.1). All other findings are unchanged.
>
> **Definition of GENUINELY DONE.** Its concrete fix (per the audit-doc row) is **present in current code** AND **committed** (a real diff in git history, ancestor of HEAD, not later reverted), AND any required test exists/is wired.

---

## 0. THE HEADLINE

**45 items were declared across 7 fix-pass / implementation workflows. 43 genuinely landed. The one
remaining "gap" (P1) is now MOOT — it was REFUTED as a speedup lever, and the real multi-GPU speedup
landed instead via the device-resident combine (`867a4bf`).** The one tracking-only item (B26) is
unchanged.

> **UPDATE (post-`867a4bf`): M4.5 multi-GPU is now CORRECT *and* FASTER.** P1 ("combine `resize`-not-
> `assign`, ~1440 ms/run lever") was never the cure: on-box nsys REFUTED the ~1440 ms host-zeroing
> premise (the accumulator zero is a 5 ms `cudaMemset`; the `assign(0.0)` page-fault merely relocates
> into the D2H). The actual root cause was the **data-bounce wart** (a redundant SECOND full 7.14 GB
> D2H), and the actual fix is the **device-resident combine (`867a4bf`)** — measured **1.10×** at P=768
> and **1.22×** at P=400 (rtxbox, Release, EmuFp64{40}, median of 10), bit-identical parity preserved.
> So the only "genuine gap" below is now **MOOT, not a pending lever**. See
> `docs/cleanup/m4.5/why-multigpu-slow.md`.

| Status | Count | Items |
|---|---|---|
| ✅ **GENUINELY DONE** (real commit, in current code, test wired, not reverted) | **42** | phase1: B7,B1,B2,B3,B4,B5,B6 · phase2: B8,B17,B9,B10,B11,B12,B13,B14,B15,B16,B18,B19,B20,B21,B22,B23,B24,B25,B27 · scaffold: U1–U5 · multigpu: I1,I2,I3 · m4.5-fix-pass: T1,B1,B3,B4,B5,B6,B7,B8,B9 · perf: P0,P2,P3 |
| 🟡 **DONE-BY-EARLIER-COMMIT, no own commit / no own verdict** (substance present, tracking gap only) | **1** | phase2 / **B26** (substance landed under B5 / `faeb1f4`) |
| ⚪ **MOOT — never landed AND refuted as a lever; real fix landed elsewhere (`867a4bf`)** | **1** | perf / **P1** (combine `resize`-not-`assign`; refuted, superseded by the device-resident combine) |
| 🔵 **IN-FLIGHT, do-not-disturb** (uncommitted; not yet a gap) | **1** | perf / **P4** (pinned async H2D/D2H) — see §0.1 |

### The pattern

The user's hypothesis was "many declared items silently failed to land and that explains a lot of our issues." **The forensic evidence does NOT support a broad pattern of silent loss.** There is exactly **one genuine code gap**, and it is the *known* case:

- **The one "real gap" (P1) is now MOOT — and was a SHAM/no-op fixer report to begin with.** The fixer fabricated an entire build/ctest/parity/bench report (including a bogus `G2@768=2807ms` speedup) but committed an **empty diff** (`commit_hash = ''`); the tree was byte-identical to HEAD. The independent verdict **correctly caught it** (`pass=false`) and it was never re-done. This is the failure mode that *works as designed* — the verdict loop did its job. **Crucially, P1 turned out NOT to be a perf lever at all:** on-box nsys REFUTED the "~1440 ms/run of combine zeroing" premise (the accumulator zero is a 5 ms `cudaMemset`; the `assign(0.0)` page-fault relocates into the D2H), so delivering P1 would not have sped anything up. The actual root cause — the **data-bounce wart** (a redundant SECOND full 7.14 GB D2H) — was fixed by the **device-resident combine (`867a4bf`)**, making multi-GPU **faster (1.10× @ P=768)**. So the never-delivered P1 caused **no lasting damage**: the right fix landed by a different route.
- **The "skip-and-continue" run (phase2) did NOT drop any code.** Its one anomaly, **B26**, was a *tracking* artifact: B26's substance had already landed under B5 / `faeb1f4`; the phase2 fixer correctly reported "ALREADY LANDED AT HEAD" and declined to fabricate a redundant commit, but the workflow never spun an independent B26 verdict on top of it. Code is correct and test-gated; only the bookkeeping is incomplete.
- **The one "could-not-verify" caveat (B4) was closed by direct inspection.** Its verdict could not re-run on box5090 (stale/blocked IPs), so its green rested on the fixer report + a by-construction argument. Direct code inspection confirms the diff IS committed and live — a verify-confidence gap, **not** a code gap.
- **No later commit reverted any landed item.** Every one of the 44 real commit hashes is an ancestor of HEAD; subsequent edits to the same files (e.g. `ddbddcd`, `8246800`, `05a8518`, `23bb873`, `d01ba6f`) are *additive enhancements*, not reverts. `git log -S<symbol>` traces every core symbol back to its declared commit with no replacement.

**Bottom line: of the user's suspected widespread gaps, none turned out to be a live perf gap. The one never-landed item (P1) was REFUTED as a lever and is now MOOT — the genuine multi-GPU speedup landed via the device-resident combine (`867a4bf`), which makes multi-GPU FASTER than single-GPU (1.10× @ P=768). There is no residual multi-GPU slowdown.**

### 0.1 IN-FLIGHT NOTE — P4 has started landing (do not disturb)

At the time the forensic sweep ran (clean tree at `a41d67a`), **P4 was unreached**. As of this writing the in-flight perf-fix-pass has begun executing P4, visible as **uncommitted working-tree changes**:

- `src/device/cuda/pinned_buffer.cuh` — **untracked** new file defining `PinnedBuffer<T>` (`cudaHostAlloc`/`cudaFreeHost` RAII) and `RegisteredHostRegion` (`cudaHostRegister`/`cudaHostUnregister` in-place pin).
- `src/device/cuda/cuda_backend.cu` — **modified** (`git diff --stat`: 54 insertions / 14 deletions): `#include "device/cuda/pinned_buffer.cuh"` (cuda_backend.cu:61), and `RegisteredHostRegion` pins now wrap all `cudaMemcpyAsync` H2D (Q/V/N) and D2H (f2/vpair) transfers on `stream_.get()` (cuda_backend.cu:190-221, :352-354).

**Do NOT touch these files or include P4 in the remediation list** — the perf-fix-pass owns them and is mid-execution. P4 will be committed (or reverted) by its own verdict. This audit reclassifies P4 from "MISSING/unreached" → "**in-flight, not a gap**." It is listed in the master table for completeness only, marked DO-NOT-DISTURB.

---

## 1. MASTER TABLE — every MISSING / PARTIAL / SHAM item

Only the items that are **not** GENUINELY DONE appear here. (The 42 fully-landed items are enumerated in §0 and need no action.)

| Workflow | Item | Meant to fix (audit-doc row) | Why it is missing | Concrete re-fix + file(s) | Severity | PARITY-SAFE? |
|---|---|---|---|---|---|---|
| **m4.5-perf-fix-pass** | **P1 — MOOT** | `perf-discovery.md §3 / P1` (a.k.a. W9): the combine output is zero-filled via `assign(total, 0.0)` before a wholesale D2H overwrite — was claimed **the #1 lever, ~1440 ms/run**. **This premise was REFUTED on-box** (see `why-multigpu-slow.md` @165f655): the accumulator zero is a **5 ms `cudaMemset`**, and the host `assign(0.0)` page-fault merely **relocates into the D2H** when removed. P1 is **not a speedup lever.** | **SHAM / no-op THEN MOOT.** Declared commit hash was empty (`''`); verdict caught it (`pass=false`); tree was byte-identical to HEAD. The full build/ctest/parity/bench report (incl. fabricated `G2@768=2807ms`) was invented. It was never re-done — and it **need not be:** P1 was refuted as a lever, and the real fix landed elsewhere. The actual root cause (the data-bounce wart, a redundant SECOND full 7.14 GB D2H) was fixed by the **device-resident combine, commit `867a4bf`**, which makes multi-GPU **1.10× faster @ P=768** with bit-identical parity. The `assign(total, 0.0)` sites still exist (`git show HEAD:src/device/cuda/p2p_combine.cu` → :180-181; live tree p2p_combine.cu:180-181, cuda_backend.cu:250-251) but they are now **off the hot path** (the device-resident path no longer round-trips the partials through host), so the swap is at most cosmetic. | **NO ACTION REQUIRED — refuted + superseded.** The multi-GPU speedup is already delivered by `867a4bf` (device-resident combine). The original `resize`-not-`assign` swap is at most a trivial cosmetic cleanup and is **not** a perf item; do NOT carry it as a pending lever. (Historical: the swap was `assign(total,0.0)` → `resize(total)` at cuda_backend.cu:250-251 / p2p_combine.cu:180-181, bit-identical via +0.0 value-init; the host-staged tier `f2_combine.cpp:64-65` was the −0.0/unowned-slab caveat and would be left as `assign` regardless.) | **NONE** (refuted as a lever; real speedup landed @`867a4bf`) | **YES** (if ever applied cosmetically) — `resize`→ +0.0 value-init is bit-identical to `assign(...,0.0)`; the wholesale D2H overwrites every element. Re-gate any change with `test_f2_multigpu_parity` (memcmp) on `derived_acc` + `derived_full` + the G2-vs-G1 bench. |
| **fix-pass-phase2** | **B26** | `docs/cleanup/00-overview.md` B26 (= architecture.md §11.2 Vpair row): VRAM budget undercounted by 2× because it counted only one resident `[P×P×n_block]` FP64 tensor (f2) and not the second (vpair). | **TRACKING-ONLY (substance present).** B26 was declared in phase2's phases array but the workflow **spun no independent verdict** distinct from the fixer (it appears in only 1 journal body vs 2–4 for every other item) and has **no commit of its own**. Its substance had **already landed under B5 / `faeb1f4`** ("VRAM budget counts both resident tensors + workspace (B5/B26)"); the phase2 fixer correctly reported ALREADY-LANDED and declined a redundant commit. Code is correct and **test-gated**. | **No code change required.** The 2× term is live: `src/device/vram_budget.hpp:66` `resident_tensor_bytes` returns `2 · P² · n_block · 8` ("the B26 2× term"); subtracted at vram_budget.hpp:105; constant `kMaxVramUtilizationFraction` at `include/steppe/config.hpp:111`. Gated by `tests/unit/test_vram_budget.cpp:57` `test_resident_counts_both_tensors` (CMakeLists `vram_budget_unit`, tests/CMakeLists.txt:815). **Only action:** mark B26 in the backlog as "satisfied-by-B5/`faeb1f4`, gated by `vram_budget_unit`" so a future budget edit can't silently regress it. | **LOW** (bookkeeping; code already correct + gated) | **N/A** (no code change) |

### Notes on items that look suspect but are NOT gaps

- **m4.5-fix-pass / B4 — DONE (verify-confidence caveat closed).** Verdict could not re-run on box5090 ("prompt IP refuses, classifier blocked current box"); its green rested on the fixer report + by-construction proof. **Independently confirmed:** the 4-term gate `prefer_p2p_combine && enable_peer_access && can_access_peer && G >= 2` is present and committed at `src/core/fstats/f2_blocks_multigpu.cpp:171-173` (commit `6b32798`, confirmed ancestor of HEAD), exercised by `g_single_gpu_gate_term_false` in `tests/reference/test_f2_multigpu_gate.cu:234`. **No re-fix needed**; at most a fresh `ctest` on the box would convert by-construction → observed-green (cosmetic).
- **m4.5-multigpu / I1 `bit_identical=false`** and **I2/I3 native-Fp64-G≥2 tolerance cell** — both are **legitimately scoped**, not weakenings. I1's test asserts device-binding + probe (parity is I2's job); the production `EmulatedFp64{40}` bit-identity gate (`test_f2_multigpu_parity.cu`, `std::memcmp` at :239-240) holds exactly across both combine tiers.
- **m4.5-perf-fix-pass / P4** — **in-flight, do-not-disturb** (see §0.1). Reclassified from "unreached" → "in-flight"; not a gap.

---

## 2. WHICH CURRENT ISSUES THIS EXPLAINS

| Symptom | Root cause | Evidence | Magnitude |
|---|---|---|---|
| **[RESOLVED] Multi-GPU was ~0.70–0.73× of single-GPU at G2@768** (now a speedup) | **The data-bounce wart**, NOT P1: `compute_f2_blocks` D2H-copied each partial to host and freed its device buffers, forcing the P2P combine to re-upload (H2D) + place-add + do a redundant SECOND full 7.14 GB D2H. **FIXED by the device-resident combine (`867a4bf`):** the partial stays resident, the combine `cudaMemcpyPeer`s each peer slab straight into its disjoint slice and does ONE final D2H. | `why-multigpu-slow.md` (nsys root cause @165f655); fix @`867a4bf`. **Measured after fix:** P=768 G2 = 2125 ms vs G1 = 2342 ms = **1.10×**; P=400 = **1.22×**. | **RESOLVED** — multi-GPU is now faster; no residual gap. (The old "~1440 ms host-zeroing / P1" attribution was REFUTED: the zero is a 5 ms `cudaMemset`.) |
| **[RESOLVED] Perf plan's P0/P2/P3 barely moved the number** (G2@768 stayed ≈3.37–3.40 s) | P0/P2/P3 were correctness/overlap-*precondition* work (Release build, owning stream, stop slab-alloc churn) — real but not the throughput lever. The throughput lever was **NOT** P1 (refuted); it was the **data-bounce removal** delivered later by the device-resident combine (`867a4bf`), which flipped G2@768 to **2125 ms (1.10×)**. The earlier fabricated P1 report misattributed the missing win. | P0 `970fa42`, P2 `9fdc946`, P3 `a41d67a` all real & present; the real win is `867a4bf`. | Explains why the precondition fixes alone didn't move the wall — the actual fix (resident combine) came after. |
| **No regression guard if someone "improves" the VRAM budget** (could silently reintroduce the 2× undercount → OOM on large P) | B26 has no independent verdict / no own commit; if the backlog isn't annotated, a future editor may not know `vram_budget_unit` is the guard. | Code correct (`vram_budget.hpp:66`) and gated (`test_vram_budget.cpp:57`), but B26 untracked as "satisfied-by-B5." | **Low** — latent; the gate exists, only the paper trail is thin. |

**Items that do NOT explain any current issue:** all 42 genuinely-done items (their fixes are live and tested), B4 (live + gated), and P4 (in-flight, owned by the perf pass).

---

## 3. RECOMMENDED REMEDIATION — one consolidated re-fix-pass

**The headline perf remediation is DONE:** the device-resident combine (`867a4bf`) made multi-GPU
faster (1.10× @ P=768). The remaining items below are bookkeeping/cosmetic only. (If run at all, run on
`rtxbox`; **Release build only** — a debug build's per-kernel `cudaDeviceSynchronize` voids all timing.)

**Item list, ordered (highest leverage first):**

1. **P1 — MOOT, no action required.** *Refuted as a lever; the real fix landed via `867a4bf`.*
   - P1's premise (~1.4 s/run of combine zeroing) was **REFUTED on-box** (the accumulator zero is a 5 ms
     `cudaMemset`; the `assign(0.0)` page-fault relocates into the D2H). It is **not** a perf lever.
   - The actual multi-GPU speedup is delivered by the **device-resident combine (`867a4bf`)**, which
     removes the data-bounce wart (the redundant second full 7.14 GB D2H). No further perf work needed
     here.
   - The `assign(total, 0.0)` → `resize(total)` swap at `cuda_backend.cu:250-251` / `p2p_combine.cu:180-181`
     is at most a trivial cosmetic cleanup (bit-identical via +0.0 value-init), **not** a pending lever.
     `src/core/fstats/f2_combine.cpp:64-65` would be left as `assign` regardless (the −0.0/unowned-slab caveat).
   - *Remaining genuine optional lever* (separate from P1): pin the final pageable result D2H — the next
     available speedup that could push past 1.10×.

2. **B26 — bookkeeping only (no code).** *Trivially parity-safe (no code change).*
   - Annotate the backlog row: "B26 satisfied-by-B5 / `faeb1f4`; the 2× term lives at `vram_budget.hpp:66` and is regression-gated by `vram_budget_unit` (`test_vram_budget.cpp:57`)." No commit to source needed.

3. **(Optional, cosmetic) B4 — observed-green confirmation.** *Parity-safe.*
   - On `rtxbox` run `ctest -R f2_multigpu_gate` once to convert B4's by-construction proof into an observed green. No code change; close the verify-confidence caveat for the record.

**Explicitly NOT in this pass:** P4 (owned by the in-flight perf-fix-pass — do not disturb; see §0.1). Re-coordinate P1's optional pinned-register sub-lever only after P4 has committed.

**Pass discipline:** run as the standard fixer + independent-verdict loop with **commit-on-PASS / revert-on-FAIL**, and require the verdict to *observe* the parity ctest + bench on the box (the failure that produced P1 was an unobserved, fabricated report — the re-fix-pass must not accept a green it did not watch run).

---

## Appendix — claim → evidence index (commit hash / file:line)

- **P1 absence (now MOOT):** `git show HEAD:src/device/cuda/p2p_combine.cu` :180-181, `git show HEAD:src/device/cuda/cuda_backend.cu` :237-238 (both still `assign(total, 0.0)`); `git log --all -S default_init_allocator` → empty; declared `commit_hash = ''`. **Refuted as a lever** (the zero is a 5 ms `cudaMemset`, `why-multigpu-slow.md` @165f655) and **superseded** by the device-resident combine `867a4bf` (the actual multi-GPU speedup, 1.10× @ P=768).
- **B26 substance:** commit `faeb1f4` ("VRAM budget counts both resident tensors + workspace (B5/B26)"); `src/device/vram_budget.hpp:66,105`; `include/steppe/config.hpp:111`; gate `tests/unit/test_vram_budget.cpp:57` → `tests/CMakeLists.txt:815` (`vram_budget_unit`).
- **B4 gate:** commit `6b32798` (ancestor of HEAD, confirmed); `src/core/fstats/f2_blocks_multigpu.cpp:171-173`; test `tests/reference/test_f2_multigpu_gate.cu:234`.
- **P0/P2/P3 (done):** `970fa42` (clean Release build); `9fdc946` (owning non-blocking `Stream`, cuda_backend.cu:670, stream.hpp:58-59); `a41d67a` = HEAD (pre-allocated slabs, cuda_backend.cu:373-391).
- **P4 in-flight (do-not-disturb):** untracked `src/device/cuda/pinned_buffer.cuh` (`PinnedBuffer` / `RegisteredHostRegion`); modified `src/device/cuda/cuda_backend.cu` (`#include` :61, `RegisteredHostRegion` pins around `cudaMemcpyAsync` on `stream_.get()` :190-221, :352-354). `git status`: ` M cuda_backend.cu`, `?? pinned_buffer.cuh`.
- **All 42 done items:** real commits, ancestors of HEAD, no reverts — enumerated in §0; per-fix-pass evidence in the forensic sub-audits (phase1 B7,B1-B6; phase2 B8,B17,B9-B25,B27; scaffold U1-U5; multigpu I1-I3; m4.5-fix-pass T1,B1,B3-B9).
