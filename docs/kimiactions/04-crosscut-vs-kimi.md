# Cross-cutting review (X1–X7) vs Kimi review + action plans — reconciliation

Lead-engineer reconciliation of OUR consolidated cross-cutting review (`docs/release_cleanup/crosscutting/X1–X7`) against the Kimi external review (`docs/kimireview/ASSESSMENT.md` + `kimiwhole/*`) and the action plans it produced (`docs/kimiactions/01-open-worth-doing.md`, `02-ci-plan.md`, `03-low-polish.md`). All findings re-verified against the code at HEAD (`a2f9d64..HEAD` campaign landed; both reviews predate parts of it).

---

## 1. Verdict

The two reviews **strongly validate each other on substance and diverge cleanly on granularity.** Where Kimi and our X-review point at the same defect they agree almost perfectly — Kimi independently caught the dominant tells of X2 (comment manifestos), X3 (the cuda_backend.cu god-file + backend.hpp fat seam + cpu_backend.cpp size + access/extract split), and X7 (ticket/milestone archaeology, exact-line hits on `dstat_kernel.cu:1-48`, `f2_block_kernel.cu:1-54`, `model_search.cpp:167`, `check.cuh:25`). **23 of our X findings were independently corroborated by Kimi, 32 are X-only, and Kimi surfaced ~42 in-scope items our X-review missed** (the decode front-end dedup, the 4 SNP readers, the try/catch-as-capability smell, the int-width axis, and the entire docs/project-hygiene layer X-review scoped out). **But the headline is the gap, not the overlap:** the three campaign-DEFERRED cross-cutting HIGHs the task flagged — X2's six design-doc comment-essays, X3's 5,679-LOC cuda_backend.cu, and X7's ticket-IDs-in-public-source — are **caught by Kimi yet planned NOWHERE.** Kimi's own `09 §3` recommended "scrub milestone tokens," but `ASSESSMENT §3` dropped it before it reached `kimiactions`, and `ASSESSMENT §2` *over-credited* the campaign for the comment theme (`"handled by b3d975f"`) when b3d975f only fixed 3 unrelated stale headers. The result: an entire cross-cutting class — the portfolio "shop-window" professionalism layer — sits neither fixed nor planned. Our X-review's distinct value is precisely this layer Kimi de-prioritized into a one-line cleanup-plan row that never landed.

---

## 2. Overlap matrix

| X | Theme | Matching Kimi theme | Agreed (Kimi caught too) | X-only | Kimi-only (beyond X) | Net status |
|---|---|---|---|---|---|---|
| **X1** | Consistency / same-thing-N-ways | doc 02 "Top 10 Patterns" (2/3/5/6/7/9/10) | **0 / 7** | 7 | 6 | Kimi looked one granularity coarser; caught 0 of X1's idiom-level items but caught MORE on X1's own axis |
| **X2** | Comment signal-to-noise (essays, ticket archaeology) | doc 02 Pattern 1 + doc 09 §3 | **8 / 15** | 7 | 5 | Strong convergence; **entire theme = GAP** (over-credited to campaign, dropped from plans) |
| **X3** | Architecture / god-files / layering | doc 01 §1/§2/§4 + per-file `cuda_backend.md` etc. | **5 / 7** | 2 | 6 | Mostly DEFERRED-by-decision (§5); 2 untracked gaps (qpadm/ misnomer, module.cpp TU) |
| **X4** | Public-API cross-boundary consistency | doc 08 (library design) + doc 05 (CLI UX) | **1 / 7** | 6 | 12 | Complementary lenses; 3 real open gaps (2 MED), 1 fixed, 1 won't-fix |
| **X5** | Idiomatic C++20 / CUDA | doc 02 (C++) + doc 04 (CUDA) | **2 / 4** | 2 | 2 | Near-pass; only concrete fix (X5-1) already planned (03 C3) |
| **X6** | Simplicity / over-engineering (dead seams) | doc 01 (god-interface) + doc 08 (config hooks) | **0 / 5** | 5 | 3 | No overlap; 2 unplanned MED dead-seam gaps; X6 & Kimi *disagree* on the backend seam |
| **X7** | Senior first-impression / comment VOICE | doc 02 Pattern 1/4 + doc 09 §3 | **7 / 10** | 3 | 8 | Strong convergence; **comment-provenance scrub planned NOWHERE** |
| | **TOTAL** | | **23** | **32** | **42** | |

Reading: X2/X3/X7 = high agreement (the deferred HIGHs live here). X1/X6 = near-zero overlap (Kimi worked coarser / one level up). X4 = complementary (X4 = cross-boundary string consistency; Kimi-08 = struct/ABI/namespacing — additive, not redundant). X5 = effectively a joint pass.

---

## 3. GAPS — in our X-review, NOT in any action plan, NOT yet fixed  ⟵ KEY SECTION

Verified OPEN/DEFERRED at HEAD and absent from `01/02/03`. Ordered by ship-value (portfolio signal × low parity risk). Each tagged with the plan it should fold into.

### TIER 1 — the deferred cross-cutting HIGHs (caught by Kimi, planned nowhere)

**G1 [HIGH] — Internal ticket-IDs / milestone codes / "spike" archaeology in SHIPPING + PUBLIC source.** *(X7-F1/F2/F4/F5, X2-#10)*
→ **NEW item in `03-low-polish.md`: "Comment hygiene for an external reader"**, prioritized public-surface-first.
- Evidence at HEAD: ~252 ticket-token hits across ~40 files; `cleanup X-N/B-N` reaches the **public** seam `src/device/backend.hpp:56,74,581` and **public include** `include/steppe/config.hpp:4 sites (lines 7,10,33,66,91,147,249,312)`; milestone codes `M4×155 / M4.5×69 / M5×59`; "the spike" in `device_buffer.cuh:9`, `stream.hpp:5`, `handles.hpp:5`, `config.hpp:10,33,66`.
- Why: this is the single worst senior-first-impression signal — the public header reads as a private engineering journal. Campaign Group-8 (`30df7c5`) removed **zero** ticket tags. Kimi's `doc 09 §3` P2 row literally recommends this; it was dropped at `ASSESSMENT §3`. **Keep the rationale, strip the provenance tag.** Zero parity risk (comment-only).
- Pair with: a `02-ci-plan.md` regression grep-gate (next to the existing A6 arch-grep) that FAILS on any new `cleanup [A-Z]-?[0-9]` / `group-[0-9]` / `M[0-9]` token in a comment — note the planned clang-format runs with **comment-reflow OFF**, so it *entrenches* these rather than stripping them; the scrub must be a deliberate step.

**G2 [HIGH] — Six design-doc comment-essays guarding a few lines of code.** *(X2-#1,2,3,4,5,6)*
→ **NEW Cluster F in `01-open-worth-doing.md` ("source-comment altitude / first-impression"), or fold into the D doc-honesty cluster.** Relocate to `docs/` with a 1-line `// see docs/…` pointer.
- The six, all present at HEAD: `f2_blocks_multigpu.cpp:237` (64-line "§4 COMBINE GATE" banner over one `select_p2p_combine` predicate); `p2p_combine.hpp:1-33` header essay; `dstat_kernel.cu:1-48` (Kimi-cited exactly); `decode_af.hpp:1-58`; `cuda_backend.cu:~5585` (39-line Stream essay on one member); `model_search.cpp:168-192` (`TODO(multigpu-host-bounce)` perf-narrative banner, Kimi-cited at `09 §3`).
- Why: portfolio-facing, parity-neutral. Kimi caught the manifesto THEME (`02 Pattern 1`) but `ASSESSMENT §3` never promoted a relocation action. These are the prompt-named deferred cross-cutting HIGH.

**G3 [HIGH→decided-defer, but the accepted half is undone] — cuda_backend.cu god-file (5,679 LOC, 55 overrides).** *(X3-#1, X7-F7)*
→ **`01-open-worth-doing.md` DECISIONS note + do the accepted down-payment now.**
- The cross-TU split is **REJECTED by decision** (`ASSESSMENT §5.5`: a class can't be partial across TUs, kernels co-located deliberately). That is settled — but the **accepted half is still undone**: add the one-line "CudaBackend is intentionally a single seam TU" header note so the 5,679-LOC file reads as a *choice*. **NUANCE worth recording:** X3's own suggested decomposition (thin aggregate header + per-subsystem shared `.inc` includes, single class) sidesteps the exact "can't be partial" objection `ASSESSMENT` used to reject Kimi's proposal — it was never separately evaluated. Worth one sentence: "the `.inc`-include split is viable and was not weighed; parked."

### TIER 2 — open MED gaps, caught nowhere or planned nowhere

**G4 [MED] — 14 bare `STEPPE_CUDA_CHECK(cudaGetLastError())` post-launch sites instead of canonical `STEPPE_CUDA_CHECK_KERNEL()`.** *(X1-F1)*
→ **`03-low-polish.md` Cluster B (robustness), framed as a prerequisite/multiplier for the `02-ci-plan` §3-High one-shot compute-sanitizer pass.**
- Sites (verified, 4 files): `dates_kernel.cu` (×9), `dstat_kernel.cu` (×2), `qpgraph_fit_kernels.cu` (×3).
- Why highest-value of X1: `CHECK_KERNEL` adds the debug-only `cudaDeviceSynchronize` that attributes async kernel faults to the launch site under compute-sanitizer. Without it, the *already-planned* sanitizer pass loses launch-site attribution on exactly these dates/dstat/qpgraph kernels. Direct synergy — do this first.

**G5 [MED] — Stale "Part B, not yet implemented" docstring on a SHIPPED feature.** *(X4-1)*
→ **`01-open-worth-doing.md` Cluster D (doc-vs-as-built honesty)** or `03 Cluster C`.
- `bindings/module.cpp:1107` tells `help(steppe._core.run_qpdstat)` the genotype path is unimplemented — but it ships: `run_dstat` is bound (`module.cpp:647`) and exposed as `steppe.dstat` (`__init__.py:818`). A user-facing lie. String-only fix → point at `steppe.dstat(prefix,…)` / CLI `qpdstat --prefix`.

**G6 [MED] — Precision-token vocabulary splits three ways across CLI / Python / emit.** *(X4-2)*
→ **`03-low-polish.md` Cluster C, as the missing sibling of C1 (Precision factories).**
- CLI accepts `emu40|emu32|fp64|tf32` (`config_builder.cpp:269-279`); Python accepts `fp64/native, emulated_fp64/emu, tf32` (`module.cpp:170-178`) — **no `emu32`, so the 32-bit mantissa is UNSELECTABLE from Python**; emit token is `emu|tf32|fp64`. C1 fixes the C++ `Precision{Kind,40}` struct ergonomics, not the string tokens. Pick one accepted set + one emitted token; add `emu32` to the Python acceptor; keep `native` as a documented alias.

**G7 [MED] — `src/core/qpadm/` directory misnomer / grab-bag (28 files).** *(X3-#3)*
→ **NEW recorded decision in `03-low-polish.md` (or `01` as MED).** **Caught NOWHERE — Kimi missed it (its closest hit, `06:60`, is build-*target* granularity, not the dir name), no `ASSESSMENT` decision exists.**
- Verified: the dir named for qpAdm houses `qpgraph_*` (8 files), `f3.cpp/f4.cpp/f4ratio.cpp`, `fstat_sweep.cpp`, `model_search*`. Either rename `→ src/core/fit/` or split `qpgraph_* → src/core/qpgraph/` and `f3/f4/f4ratio/fstat_sweep → src/core/fstats/` (note: `src/core/fstats/` already exists for the f2-blocks family). **Even if the decision is DEFER, record it** so the misleading name reads as a choice. Pairs with Kimi `06:60` steppe_core sublibrary split.

**G8 [MED] — Dead DeviceConfig knobs: `stream_count` / `search_streams` / `use_mem_pool` (+ `kDefaultSearchStreams`).** *(X6-1)*
→ **`03-low-polish.md` Cluster A — same "declared-not-wired" pattern the plan already treats for `STEPPE_SANITIZER`/`STEPPE_NVTX`.**
- Verified 0 production consumers (`include/steppe/config.hpp:353,357,362,183`; `cudaMallocAsync` is unconditional, `use_mem_pool` never gates it). Kimi (`08:419`) names `stream_count` only to argue for a `validate()` hook — a *different* angle — never flags it dead. Either delete, or replace the live-sounding paragraph with a one-line "forward-reserved for parked multi-GPU/S8" note.

**G9 [MED] — Vestigial `rank_test` virtual (base throws, both backends override, only "caller" is an uncalled wrapper).** *(X6-2)*
→ **`03-low-polish.md` standalone dead-virtual removal, OR a concrete down-payment on `ASSESSMENT §5.2` "slim the backend.hpp seam (DEFER)".**
- `backend.hpp:1692` + overrides `cuda_backend.cu:4386`, `cpu_backend.cpp:1556` + dead wrapper `gls_solve.hpp:17` (zero callers). Live path uses `rank_sweep` + `gls_weights`. Delete the virtual, 2 overrides, and the wrapper — trims the seam without the full role-split. Kimi lists it in its virtual enumeration but never flags it dead.

### TIER 3 — open LOW gaps (small, zero parity risk)

**G10 [LOW] — `dates()` returns a bare untyped dict** (`__init__.py:855-879`, no `->` hint, no `DatesResult` class) — the one ergonomic outlier in the Python facade; the C++ `DatesResult` value type already exists (`dates.hpp`). *(X4-4)* → `03 Cluster C`, sibling to C1/C2/C3.

**G11 [LOW] — Device-layer convention re-align (one tiny commit, the two lone-deviator files).** *(X1-F2/F3/F4)* → `03-low-polish.md`.
- `qpgraph_fit_kernels.{cu,cuh}` use `namespace steppe::device::cuda` (43 other device TUs use `steppe::device`); `qpgraph_fit_kernels.cuh:55` spells raw `__host__ __device__` instead of `STEPPE_HD`; `qpadm_fit_kernels.cuh:37,53,69,357` name device inputs `f2`/`vpair` with no `d_` prefix alongside `int* d_keep`. Collapse into one cosmetic re-align.

**G12 [LOW] — bindings/module.cpp single 1,170-LOC TU.** *(X3-#6)* — **untracked split.** → `03 Cluster C`, next to C3 (which already edits module.cpp). Split per-tool `bind_qpadm.cpp`/`bind_dates.cpp`/… each via `register_*(m)`, module.cpp the thin assembler. Kimi reviewed module.cpp but flagged only the f2 loop / exception mapping, not the TU split.

**G13 [LOW] — Confusable sibling TUs `f2_block_kernel.{cu,cuh}` vs `f2_blocks_kernel.{cu,cuh}` (one `s`).** *(X7-F6)* → `03` XS rename (e.g. `f2_batched_kernel`). X7-unique; Kimi touches `f2_block_kernel.cu` for magic numbers but never flags the name trap.

**G14 [LOW] — Stale one-liner scrub** (bundle into G1's hygiene pass): `model_search.cpp:173-174` self-changelog ("The doc once said…WRONG"); `qpgraph_fit_kernels.cu:254` "unscaled q for now" (scaling completes at :264 — reword); `log.hpp:7,15,24,25` "eventually a spdlog"/"swap later"; `check.cuh:25,227,290` `TODO(M4.5)` ×3. *(X2-#11/12/13/14, X7-F9)*

**G15 [LOW] — ALL-CAPS / charged-vocabulary density** (`NEVER ×106`, `MUST ×31`, `LAW ×8`, `REJECTED ×11`) — reserve emphasis for genuine parity/precision invariants. *(X7-F8)* → `03` XS, pairs with G1.

**G16 [LOW] — `target_compile_features(cxx_std_20)` on 5 targets, omitted on `steppe_core`+`steppe_device`.** *(X1-F7)* → `03 Cluster A` build polish, 2-line CMake add so all 7 targets declare the standard the same way.

**G17 [LOW] — Optional**: add `requires std::is_trivially_copyable_v<T>` to the two unconstrained buffer templates (`device_buffer.cuh:42`, `pinned_buffer.cuh:73`). *(X5-2)* → `03 Cluster C` header-sugar batch; marginal by X5's own rating.

### NOT gaps — close explicitly (do NOT add to a plan)

- **X4-3 rotation `f4rank` column** — by-design, golden-locked. The X4 "fix" (emit `r.f4rank`) would MOVE the rotation output and **break `golden_rot.json`** (whose `res$f4rank` is NULL). Documented at `result_emit.cpp:303-308,368-372`. **Won't-fix.**
- **X1-F5 `precision_tag`** — does NOT reproduce; already uniform across all 11 result structs at the campaign base `a2f9d64`. The X1 read was stale. **Moot.**
- **X4-5 / X7-F3 cli_parse scaffold comments** — **FIXED** by campaign `30df7c5` (the "qpwave remains a scaffold no-op" line is gone — verified absent at HEAD). No action.
- **X6-3/4/5** (`set_solve_precision`, `TileEncoding` 1-value enum, 4-copy ploidy scan) — accept-as-is per X6's own verdict (documented forward-seams; ploidy = the parity-oracle independent-re-derivation doctrine). Not gaps.
- **X5-3/4** (index-for vs range-for; no `std::format`) — X5 itself rules non-actionable; at most a forward-style line in `NAMING-STYLE-STANDARD`, not a plan item.

---

## 4. Where Kimi caught MORE (real, in-scope, our X-review missed)

These are genuine X-review blind spots — most already routed into plans, a few still open:

1. **Duplicated genotype decode front-end** across dstat/qpfstats/dates/extract (Kimi `02 Pattern 3`) — a real parity-divergence risk; **planned (`01 C1`)**. X1 declared the codebase "exceptionally disciplined" and missed it entirely.
2. **Four near-identical ~100–150-LOC SNP-major tile readers** in `geno_reader.cpp` (Kimi `02 Pattern 9`) — copy-paste drift; **still unplanned** (`ASSESSMENT` didn't elevate it) = a real consistency gap Kimi caught and X-review missed.
3. **try/catch-as-capability-detection** error contract (`qpadm_fit.cpp:187`, `model_search.cpp:136`; Kimi `02 Pattern 7`) — a dead/bug-hiding catch; **planned (`01 B3`)**. X-review reached only the reason-field naming.
4. **`kPrimaryGpu`/`primary_backend()` redeclared in 5+ core TUs** (Kimi `02 Pattern 2`) — X1's own axis; `ASSESSMENT §2` declined on merit (TU-private convention). Settled, but Kimi surfaced it.
5. **backend.hpp is itself a 1,857-LOC god-INTERFACE** (56 virtuals; Kimi `02 Pattern 8` / `01 §2.1`) — X3 named only the `.cu`. `ASSESSMENT §5.2` defer + header-note (note still undone, see G3).
6. **core→io doc-vs-code layering lie** (`architecture.md:239` says only app wires io, but `src/core/CMakeLists.txt:153` links `steppe::io`; Kimi `01 §1.1`) — X3 *praised* core→io as fine and missed the doc lie. **Planned (`01 D1`).**
7. **Integer-width / span-vs-raw-pointer axis** + the `steppe::Index` proposal (Kimi `02 Pattern 10`) — narrowing fixed at H1/G4, blanket alias rejected `§4`. X-review never listed int-width.
8. **Library-design depth** (Kimi `08`): result-struct decomposition (`QpAdmResult` kitchen-sink → nested tables; `§5.2` defer), stable **C ABI** (`§5.1` defer to M(abi-1)), `f2_at` indexing helper (→ `03 C2`), Precision named factories (→ `03 C1`), `new F2Handle`/`f2_to_numpy` C-isms (→ `03 C3`), missing Doxygen/examples (→ `03 C4/C5`), `geno_max_missing` field name, `-1` sentinels vs `optional`, `QpAdmOptions::jackknife` ignored by `run_qpadm`.
9. **Config-validation hooks** (`DeviceConfig::validate()/is_valid`, Kimi `08 §7`) — an API-ergonomics angle on the *same* struct X6 flags for dead knobs (G8); complementary.
10. **The entire docs / project-hygiene layer X-review scoped OUT** (Kimi `09`): `architecture.md` 958-line stale monolith, drifted status docs (NEXT-STEPS/TODO/ROADMAP asserting "no CLI/no bindings"), docs/ tree clutter (cleanup/ 134 + release_cleanup/ 161 + kimireview/ 102 files), root traps (aadr/ data dir, .claude/.codex), missing CHANGELOG/NOTICE/ATTRIBUTION. Mostly **planned (`01 D1` + `03` portfolio)** — genuinely beyond the source-comment lens.
11. **Disagreement worth recording**: Kimi `01 §1` wants the `ComputeBackend` god-interface **role-split** (Precompute/Fit/Decode/Sweep/Dates/Qpfstats); X6 calls the fat interface **earned** (the price of the compiler-enforced CUDA-free core boundary). `ASSESSMENT §5.2` sided with X6 and DEFERRED Kimi's split to a header note.

---

## 5. Unified recommendation — what to ADD so the cross-cut work is not lost

The action plans are sound on the substantive engineering (decode dedup, try/catch contract, doc-sync, C ABI, factories) but **systematically omit the professionalism/first-impression layer** because `ASSESSMENT §2` over-credited the campaign for comments and `§3` dropped the milestone-scrub. Five concrete additions:

1. **`03-low-polish.md` → NEW item "Comment hygiene for an external reader" (LOW effort, HIGH signal).** Purge `cleanup X-N/B-N` (~252 hits / ~40 files), bare milestone codes (`M4×155`, `M4.5×69`, `M5×59`), and "spike" archaeology from shipping source — **public surface first**: `include/steppe/config.hpp` and `src/device/backend.hpp` must read self-contained, zero ROADMAP/milestone/ticket citations. Keep rationale, drop provenance. Bundle G14/G15. *(Discharges G1, G14, G15 — the X7 + X2-#10 deferred HIGH.)*

2. **`01-open-worth-doing.md` → NEW Cluster F "source-comment altitude"** (or extend D): relocate the six HIGH design-doc essays (G2) to `docs/` with `// see docs/…` pointers; add a short **DECISIONS** note recording the *deliberately deferred* structural debt (cuda_backend.cu single-seam TU per §5.5, backend.hpp role-segregation per §5.2, access/extract source relocation per §5.6, cpu_backend.cpp split) so the omissions read as choices — and add the **undone accepted header-note** to cuda_backend.cu (G3). *(Discharges G2, G3, and the §5 silent omissions.)*

3. **`02-ci-plan.md` → regression grep-gate** next to A6: FAIL on any new `cleanup [A-Z]-?[0-9]` / `group-[0-9]` / `M[0-9.]+` comment token. Flag that the planned clang-format runs comment-reflow OFF and therefore *entrenches* the essays — the scrub (item 1) must be deliberate, not a formatter side effect. *(Locks in G1.)*

4. **`03-low-polish.md` → fold the three open MEDs** the plans miss: G4 (14 → `STEPPE_CUDA_CHECK_KERNEL`, framed as the compute-sanitizer multiplier — do FIRST), G6 (precision-token unification + `emu32` to Python), G8 (dead DeviceConfig knobs, alongside the existing `STEPPE_SANITIZER`/`STEPPE_NVTX` treatment); plus G9 (dead `rank_test` virtual). And **record G7** (the `src/core/qpadm/` misnomer — caught by *neither* review, decided *nowhere*) even if the decision is DEFER.

5. **`01 Cluster D` / `03 Cluster C` → the two user-facing string fixes**: G5 (stale "Part B not yet implemented" docstring on shipped `steppe.dstat`) and G10 (`dates()` typed result). Then close the non-gaps explicitly in `ASSESSMENT`: X4-3 (golden-locked won't-fix), X1-F5 (moot), X5-3/4 + X6-3/4/5 (accept-as-is) — so the next reviewer sees they were weighed, not missed.

**Net:** the engineering-substance gaps are mostly already planned (Kimi's strength). The unplanned residue is almost entirely the **shop-window layer** — ticket-ID scrub, essay relocation, ALL-CAPS, confusable names — which is exactly the deferred cross-cutting HIGH the campaign punted and the `kimiactions` triage chain dropped. One LOW-risk comment-hygiene pass (public-surface-first) plus a CI grep-gate recovers the bulk of it at near-zero parity risk.
