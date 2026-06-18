# steppe — RESUME (read this first after a compaction / new session)

**The one line that matters:** read `agentscripts/README.md` first — it's the workflow map
(where we are + every workflow's state/retrigger + what's still needed). Everything else is
reachable from there. The rest of this file is the fast re-orientation.

## Read these (don't re-derive)
1. **`agentscripts/README.md`** ← THE WORKFLOW MAP / master index.
2. `ls handoff-*.md` → read the latest (e.g. `handoff-ba37d95.md`) — cold-start handoff.
3. `docs/BOX-RUNBOOK.md` ← how to stand up / verify / update a GPU box (commands + REJECT criteria).
4. `docs/cleanup/m4.5/00-overview.md` ← the M4.5 cleanup backlog (B1–B9, L4).
   (Auto-loaded memories cover the boxes: `[[rtxbox]]`, `[[steppebox5090]]`, `[[steppe-project]]`, `[[steppe-dev-process]]`.)

Verify state: `cd /home/suzunik/steppe && git branch --show-current && git log --oneline -6`

## State (as of this writing)
- Branch **`m4.5-multigpu`**, ~29 commits ahead of `main` (**UNMERGED**). **M4.5 single-node multi-GPU DONE + cleaned, 36/36 ctest green.**
- **KEY OPEN FINDING:** multi-GPU is currently **SLOWER than single-GPU** (B1 fan-out alone isn't a speedup) — gated on **L4 (pool allocator)** + **B2 (P2P)**. Bench: `tests/reference/bench_f2_multigpu.cu`.
- steppe = the f2 **precompute** (`f2_blocks`); the **qpAdm fit engine (Phase 2, S3–S8) is NOT built** — so no admixture/population test (e.g. Yamnaya models) runs yet.

## Boxes
- **box5090** = vast 2× RTX 5090 (**UP**, flaky network): `ssh box5090`. Build/test here. **Long jobs (build/ctest/bench) → run DETACHED on the box + poll a `/tmp/*.log`** (RUNBOOK §7) — the network drops long ssh sessions. Already has the repo + data (`raw` + `derived_acc` + `derived_full`).
- **rtxbox** = 2× RTX PRO 6000 (**SPUN DOWN**, ephemeral): **update the `~/.ssh/config` alias to the new IP before use** (RUNBOOK §0). Needed for B2, the real multi-GPU speedup measurement, AT2 goldens, GDS.

## Build / test (nvcc is NOT on PATH — note the exports)
```
ssh box5090 'cd /workspace/steppe && export PATH=/usr/local/cuda/bin:$PATH && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && \
  cmake -S . -B build -GNinja && cmake --build build && ctest --test-dir build --output-on-failure'
```
Dev loop: edit locally → `rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh /home/suzunik/steppe/ box5090:/workspace/steppe/` → build/test on box.

## Workflows
Retrigger via `Workflow({scriptPath: "agentscripts/<name>.js"})`. Patterns:
- **impl** = Contracts/Design → per-file Implement → build agent → verify agent.
- **fix-pass** = fixer + independent verdict (template `fix-pass-phase2.js`).
- **audit** = per-unit auditor → adversarial critic → holistic capstone (read-only).
- **B2 is staged:** `agentscripts/m4.5-b2-p2p-fix-pass.js` (run on rtxbox).
- To set up a box: `scripts/box_bringup.sh <alias> [--build]` + `scripts/p2p_probe.cu`.

## Open threads (pick up here)
1. **nsys profile + L4 pool-allocator fix-pass** = the real speedup (box5090; re-run the bench to confirm).
2. **B2 P2P fix-pass** (rtxbox, staged).
3. **Merge `m4.5-multigpu` → `main`** (it's green + proven).
4. Then **M5** (out-of-core streaming) / **M7** (AT2 goldens) / Phase 2 fit engine.

## Rules
Nothing builds locally (rsync to the box). Commit only on green (ROADMAP §6 message + the
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer). VERIFY first,
then question the verification. DON'T re-probe a box that's already set up. Use workflows for
substantive work. No synthetic data for any accuracy claim (real AADR only).

---
*If you need raw session history (rare — the above usually suffices): grep `~/.claude/projects/-home-suzunik-steppe/*.jsonl` (that's how the milestone-build workflows were recovered).*
