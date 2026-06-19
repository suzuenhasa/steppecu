# steppe — RESUME (read this first after a compaction / new session)

**The one line that matters:** read `agentscripts/README.md` first — it's the workflow map
(where we are + every workflow's state/retrigger + what's still needed). Everything else is
reachable from there. The rest of this file is the fast re-orientation.

## Read these (don't re-derive)
1. **`agentscripts/README.md`** ← THE WORKFLOW MAP / master index.
2. `ls handoff-*.md` → read the latest (e.g. `handoff-ba37d95.md`) — cold-start handoff.
3. `docs/BOX-RUNBOOK.md` ← how to stand up / verify / update a GPU box (commands + REJECT criteria).
4. `docs/cleanup/m4.5/{why-d2h,why-multigpu-slow,architecture-audit,parallelism-check}.md` ← the honest perf story; `docs/research/at2-timing-comparison.md` ← vs ADMIXTOOLS 2.
   (Auto-loaded memories cover the boxes: `[[rtxbox]]`, `[[steppebox5090]]`, `[[steppe-project]]`, `[[steppe-dev-process]]`.)

Verify state: `cd /home/suzunik/steppe && git branch --show-current && git log --oneline -6`

## State (as of this writing)
- Branch **`m5-input-streaming`** @ `d0fdfe4` (**being MERGED to `main` now**). **Phase 1 PRECOMPUTE is substantially COMPLETE THROUGH M5** — M0–M4 + before-M4.5 cleanup + M4.5 multi-GPU are on `main`; this branch adds device-resident output (`1f80c0c`) + M5 streaming (tiered output `176a07d` + SNP-tile input `c65179f`). Parity `memcmp` bit-identical on real AADR throughout.
- **KEY LESSON — the precompute was HOST-RESULT-BOUND, not compute-bound.** Device-resident output (`1f80c0c`) keeps the result in VRAM (`DeviceF2Blocks`; host `F2BlockTensor` is opt-in `.to_host()`): P=512 ~673 ms device-resident vs ~2879 ms bulk-to-host = **~4.3×**. ~80% of the old wall was the bulk D2H — getting it OFF the CPU was the real win, **NOT multi-GPU** (`docs/cleanup/m4.5/why-d2h.md`).
- **M5 streaming DONE:** adaptive **tiered output** (VRAM → host RAM → disk, auto-selected) + **SNP-tile input streaming** (GPU footprint O(P·tile + P²), independent of M). **Full-autosome P=2500 (M=584131, n_block=757) completes on a SINGLE 32 GB 5090 in ~51.5 s** (76 GB result streamed, GPU peak ~26 GB bounded), parity bit-identical. One-5090 sweep: P=512 ~3.6 s, P=1000 ~10.4 s, P=1500 ~20.2 s, P=2000 ~34.0 s, P=2500 ~51.5 s.
- **HONEST multi-GPU story:** M4.5 multi-GPU is correct/bit-identical but on the precompute is only a *modest throughput layer* (and was SLOWER than single-GPU until the data-bounce was fixed; nsys ~22–74% overlap + a serial D2H tail). Multi-GPU's proper home is the **FIT / ROTATION phase** (thousands of independent models, no combine), not the precompute. (`why-multigpu-slow.md`, `parallelism-check.md`.)
- steppe = the f2 **precompute** (`f2_blocks`); the **qpAdm fit engine (Phase 2, S3–S8) is NOT built** — so no admixture/population test (e.g. Yamnaya models) runs yet.

## Boxes
- **box5090** = vast 2× RTX 5090 (**UP**, flaky network): `ssh box5090`. Build/test here. **Long jobs (build/ctest/bench) → run DETACHED on the box + poll a `/tmp/*.log`** (RUNBOOK §7) — the network drops long ssh sessions. Already has the repo + data (`raw` + `derived_acc` + `derived_full`). **The M5 single-GPU streamed sweep was measured here — full-autosome P=2500 in ~51.5 s on ONE 32 GB 5090, parity bit-identical.**
- **rtxbox** = 2× RTX PRO 6000 (**SPUN DOWN**, ephemeral): **update the `~/.ssh/config` alias to the new IP before use** (RUNBOOK §0). M4.5 multi-GPU + the nsys root-cause trace + the streamed P=2000=15.1 s figure were measured here. Still the box for AT2 goldens, GDS, the device-resident final-D2H pinning, and any `ncu`/nsys re-profiling.

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
- **B2 (`agentscripts/m4.5-b2-p2p-fix-pass.js`): REASSESS / likely-subsumed.** Its original motivation (the slow/bouncing combine) is now largely addressed by the device-resident combine (867a4bf) — re-evaluate before running rather than treating it as a pending speedup.
- To set up a box: `scripts/box_bringup.sh <alias> [--build]` + `scripts/p2p_probe.cu`.

## Open threads (pick up here)
Phase 1 precompute is substantially **complete through M5** (device-resident output + tiered/streaming). The real next work:
1. **Merge `m5-input-streaming` → `main`** (green + bit-identical; in progress).
2. **qpAdm FIT ENGINE (Phase 2, S3–S8)** — does not exist yet; this is the real next milestone. Reads `f2_blocks` (device-resident in-VRAM, or streamed tiles for large P), runs the GLS solve + rank test + the **model rotation** (many independent models = the embarrassingly-parallel, **multi-GPU-friendly** phase — multi-GPU's proper home).
3. **AT2 goldens** (install R + admixtools; pin `extract_f2`/`qpadm` goldens) — the Phase-2 parity gate · **M6** (multi-dataset merge).
- *Deferred / optional, NOT blockers:* multi-GPU block-sharding on the stream (throughput); pin the device-resident final-D2H on rtxbox; the **TurboQuant-L2 rotation screen** only AFTER the fit exists (`docs/research/turboquant-l2-experiment.md`); the L4 pool allocator is at most a few % (optional cleanup, not a speedup lever); the bench byte-traffic columns are observability-only (currently print 0.00).

## Rules
Nothing builds locally (rsync to the box). Commit only on green (ROADMAP §6 message + the
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer). VERIFY first,
then question the verification. DON'T re-probe a box that's already set up. Use workflows for
substantive work. No synthetic data for any accuracy claim (real AADR only).

---
*If you need raw session history (rare — the above usually suffices): grep `~/.claude/projects/-home-suzunik-steppe/*.jsonl` (that's how the milestone-build workflows were recovered).*
