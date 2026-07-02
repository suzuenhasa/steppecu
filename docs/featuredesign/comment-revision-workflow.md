# Workflow design — revise code comments to be release-ready

**Status: scoped, ready to run (pending scope sign-off).** A multi-agent workflow that rewrites
the **comments** across the source tree so they're fit for a public release: **no references to
internal documents** (`architecture.md §12`, `cli-bindings.md`, `ADR-0002`), no internal
process/milestone shorthand (`M(py-1)`, `the C8 TU split`, `the A3 plan`), and no dev-box names
(`box5090`, `rtxbox`, `vast.ai`) — while **preserving every load-bearing invariant** a comment
documents. Code is never touched.

## 1. The problem, quantified (327 source files)

| internal-reference pattern | hits | files |
|---|---:|---:|
| `§` section refs (`§12`, `§4`) | 2334 | 285 |
| design-doc citations (`architecture.md`, `cli-bindings.md`, …) | 1235 | 256 |
| milestone tags (`M(py-1)`, `M(fit-2)`, `M0`) | 121 | 39 |
| dev-box / vast names (`box5090`, `rtxbox`, `vast.ai`) | 60 | 40 |
| named goldens (`golden_fit0.json` + field refs) | 46 | 11 |
| `ADR-…` | 8 | 6 |
| plan/phase shorthand (`A3 plan`, `S0'`, `C8`) | ~14 | 10 |

Nearly **every file** cites an internal doc. Overall comment density is **37.6%** (`docs/commentdump/INDEX.txt`),
with a long tail of 80–92%-comment headers.

## 2. What "release-ready" means — the three rules

**A. STRIP** (remove the reference, keep the surrounding technical statement):
- section refs `§N`; design-doc filenames (`architecture.md`, `cli-bindings.md`, `run-sheet.md`, …);
- `ADR-NNNN`; milestone tags `M(...)`, `M0`; plan/phase shorthand (`A3 plan`, `S0'`, `C8 TU split`);
- dev-box / infra names (`box5090`, `rtxbox`, `vast.ai`, "the 2× RTX 5090 box");
- named internal fixtures where they leak structure (`golden_fit0.json` `reproduction.note`);
- decision-log / process chatter ("the reviewer flagged…", "we went with X because…").

**B. KEEP or REWRITE** (preserve the substance; drop only the citation):
- **Load-bearing invariants** — fixed reduction order, cancellation-sensitivity, precision policy,
  thread-safety, ownership/lifetime, alignment, ABI. These are the *reason the comment exists*.
- **Legitimate external references** — ADMIXTOOLS 2 / AT2 (the published method steppe
  reimplements — keep), `cuBLAS`/`cuSOLVER`/`cuFFT`/CUDA API names, IEEE-754, the RDS/XDR format,
  published algorithm names.
- **The qpAdm math vocab** (`nl`/`nr`/`r`/`m`/`nb`/`dof`/`chisq`/`Q`/`A`/`B`/`w`) — meaningful AT2
  method terms; keep the terms, drop any `§3.2` citation.

  Example: `// fixed order (architecture.md §12 cancellation carve-out) — do NOT reorder`
  → `// fixed reduction order — cancellation-sensitive; do not reorder`

**C. TIGHTEN (optional, conservative)** — trim only *clearly* redundant narration (comments that
restate the adjacent code) in the most over-commented files. **Not** an aggressive density
rewrite: a prior over-engineered comment-style pass produced *more* verbosity, so the rule here
is "remove citations and restated-code lines; leave the judgment/why intact." Recommend running
this as a **separate opt-in mode**, not bundled with the strip pass.

## 3. Hard guardrails

1. **COMMENTS ONLY.** No change to code, identifiers, string literals, includes, or logic. Every
   agent's diff must touch only comment text. In particular the protected math-vocab identifiers
   in *code* (`nl`, `nr`, `r`, `m`, …) are never renamed — safe by construction, but stated
   explicitly in the prompt.
2. **Preserve invariants.** Never delete a comment documenting an ordering / precision /
   concurrency / ownership / ABI invariant — rewrite it to drop only the internal citation.
3. **Golden gate.** Because the change is comments-only, the build **must** still compile and the
   parity goldens **must** be byte-identical. A differing golden ⇒ a code edit slipped in ⇒ reject.
   This is the objective proof of safety.
4. **No new elaborate style doc.** A terse rule set in each agent prompt (§2), never a long
   standard (which historically inflated comment volume).

## 4. Workflow shape

**Phase 0 — Inventory & bucket** (cheap, mostly the grep above + the comment dump). Produce the
worklist: the ~285 files with internal refs, each tagged with which patterns it carries + its
comment density, **grouped by module** (`src/device`, `src/core`, `src/app`, `bindings`, `tests`,
…) so revisers get consistent per-module context.

**Phase 1 — Revise** (fan-out, the bulk). Pipeline over the worklist, **batched by module**; one
agent per file (or a few closely-related files). Each agent rewrites comments per §2, editing in
place, and returns a structured result: `{patterns_removed, invariants_preserved[], flagged[]}`
where `flagged` = comments it judged risky to touch (kicked to human review). Prompt carries the
§2 rules + the §3 guardrails verbatim. No worktree isolation needed — agents edit disjoint files.

**Phase 2 — Verify** (adversarial + gate), per file as it finishes:
- **Critic pass** (adversarial): re-read the revised file — (a) `grep` the diff for any surviving
  internal ref (`§`, `.md`, `box5090`, `ADR-`, `M(`, …) → must be zero; (b) confirm each §3
  invariant class still documented; (c) confirm the diff changed **only** comment lines. Failures
  loop back to Phase 1.
- **Build + golden gate** (once per module batch, on the GPU box): compile + run the parity
  goldens; assert byte-identical. This catches any code that slipped past the diff check.

**Phase 3 — Synthesize.** Tree-wide residual-reference grep (target: 0 in code comments),
files-changed count, comment density before/after, and the consolidated `flagged[]` list of
invariant comments for human eyes.

## 5. Scale, batching, cost

~285 files → ~285 revise + ~285 critic agents + per-module build gates. This is a **large**
workflow (hundreds of agents, token-heavy). **Recommend staging by module** (do `src/device`,
verify + commit, then `src/core`, then `src/app`/`bindings`/`tests`) so it's reviewable and
committable in coherent chunks rather than one giant diff. Each module stage is an independent
Workflow run gated by its own goldens.

## 6. Risks & mitigations

| risk | mitigation |
|---|---|
| Over-stripping a load-bearing invariant comment | §2B "preserve WHY, drop citation" rule + adversarial critic + human review of the `flagged[]` list |
| An accidental code edit | comment-only diff check (Phase 2a) **and** the byte-identical golden gate (Phase 2b) |
| Inconsistent terminology across files | batch by module; shared terse rule set in every prompt |
| Verbosity backfire (the prior style-doc trap) | tightening is opt-in + conservative; no elaborate standard |
| Losing legitimate external refs (AT2, cuBLAS) | §2B explicitly keeps them; critic checks they survive |

## 7. Files needing extra care (protect-list)

The precision / reduction / jackknife / cancellation paths, where comments encode invariants that
must survive verbatim-in-substance: `src/device/cuda/dstat_kernel.cu`, the f2/f4 reduction +
`extract_f2` kernels, the emulated-FP64 (Ozaki) matmul path, and the jackknife/SE code. These get
the adversarial critic **and** are surfaced in `flagged[]` for a human pass before their module
commits.

## 8. Deliverable per stage
A module-scoped branch with a comments-only diff, green build + byte-identical goldens, and a
report (residual refs, density delta, flagged invariants). Merge per module after review.
