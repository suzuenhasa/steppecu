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
- **`main`** @ `42cca57` — the whole chain is **MERGED**: device-resident output (`1f80c0c`) + M5 streaming (tiered output `176a07d` + SNP-tile input `c65179f`) + the perf/architecture investigations + ALL docs **including the canonical spec `architecture.md`** (now current with M5). **Phase-1 PRECOMPUTE is COMPLETE THROUGH M5** (M0–M4 + before-M4.5 cleanup + M4.5 multi-GPU were already on `main`). Parity `memcmp` bit-identical on real AADR throughout. **Now starting Phase 2 (the qpAdm fit engine) — design on branch `phase2-fit-engine`.**
- **KEY LESSON — the precompute was HOST-RESULT-BOUND, not compute-bound.** Device-resident output (`1f80c0c`) keeps the result in VRAM (`DeviceF2Blocks`; host `F2BlockTensor` is opt-in `.to_host()`): P=512 ~673 ms device-resident vs ~2879 ms bulk-to-host = **~4.3×**. ~80% of the old wall was the bulk D2H — getting it OFF the CPU was the real win, **NOT multi-GPU** (`docs/cleanup/m4.5/why-d2h.md`).
- **M5 streaming DONE:** adaptive **tiered output** (VRAM → host RAM → disk, auto-selected) + **SNP-tile input streaming** (GPU footprint O(P·tile + P²), independent of M). **Full-autosome P=2500 (M=584131, n_block=757) completes on a SINGLE 32 GB 5090 in ~51.5 s** (76 GB result streamed, GPU peak ~26 GB bounded), parity bit-identical. One-5090 sweep: P=512 ~3.6 s, P=1000 ~10.4 s, P=1500 ~20.2 s, P=2000 ~34.0 s, P=2500 ~51.5 s.
- **HONEST multi-GPU story (UPDATED — measured on real data):** M4.5 multi-GPU is correct/bit-identical but on the precompute is only a *modest throughput layer*. We expected the **FIT/ROTATION (S8)** to be multi-GPU's payoff — but **MEASURED on real AADR (P=600), the S8 rotation is host-bounce-capped on the consumer 5090s**: the one-time `f2` replication is ~8.72 GB / ~3.8 s through host (no GPU↔GPU P2P on GeForce), so G2/G1 only reached **~1.21× at 9086 real models** (no 1.5× crossover). **→ RUN THE FIT/ROTATION SINGLE-GPU on box5090.** Multi-GPU rotation is DEFERRED (`TODO(multigpu-host-bounce)`, commit 2a0c020); its real payoff needs P2P hardware (rtxbox) or the per-device-precompute fix. (`why-multigpu-slow.md`, `parallelism-check.md`.)
- steppe = the f2 **precompute** (`f2_blocks`); the **qpAdm fit engine (Phase 2, S3–S8) is NOT built** — so no admixture/population test (e.g. Yamnaya models) runs yet.

## Boxes
- **box5090** = vast 2× RTX 5090 (**UP**, flaky network): `ssh box5090`. Build/test here. **Long jobs (build/ctest/bench) → run DETACHED on the box + poll a `/tmp/*.log`** (RUNBOOK §7) — the network drops long ssh sessions. Already has the repo + data (`raw` + `derived_acc` + `derived_full`). **The M5 single-GPU streamed sweep was measured here — full-autosome P=2500 in ~51.5 s on ONE 32 GB 5090, parity bit-identical.** **RUN GUIDANCE: run the qpAdm fit / S8 rotation SINGLE-GPU on box5090** — the 5090s have no GPU↔GPU P2P, so multi-GPU rotation host-bounces the f2 (~3.8 s / 8.72 GB, only ~1.21× at 9086 real models); multi-GPU rotation is deferred (`TODO(multigpu-host-bounce)`).
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
Phase 1 precompute is **complete through M5** and merged to `main`. **The active work is now PHASE 2 — the qpAdm fit engine:**
1. **qpAdm FIT ENGINE (Phase 2, S3–S8) — ACTIVE, in design** (`agentscripts/fit-engine-design.js` → `docs/design/fit-engine.md`). Does not exist yet. Reads `f2_blocks` (device-resident in-VRAM, or streamed tiles for large P), runs S3/S4 (f3/f4 derivation + jackknife), S5 (rank test / SVD), S6 (GLS solve — Cholesky + weighted normal equations), S7 (block-jackknife SE), S8 (the **model rotation** — many independent models; expected to be multi-GPU's home, but MEASURED host-bounce-capped on the 5090s → **run SINGLE-GPU**, see the multi-GPU note above). Build order + the first-milestone contract come from the design.
2. **AT2 goldens** (install R + admixtools on a box; pin `extract_f2`/`qpadm` goldens per §12: R ver / RNGkind / AT2 ver / blgsize / boot / seed) — **the Phase-2 acceptance gate** (qpAdm `est/se/z/p` `memcmp`/tolerance vs AT2). Stand these up alongside the first fit slice.
3. **M6** (multi-dataset merge) · later precompute polish.
- *Deferred / optional, NOT blockers:* multi-GPU block-sharding on the stream (throughput); pin the device-resident final-D2H on rtxbox; the **TurboQuant-L2 rotation screen** only AFTER the fit exists (`docs/research/turboquant-l2-experiment.md`); the L4 pool allocator is at most a few % (optional cleanup, not a speedup lever); the bench byte-traffic columns are observability-only (currently print 0.00).

## Rules
Nothing builds locally (rsync to the box). Commit only on green (ROADMAP §6 message + the
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer). VERIFY first,
then question the verification. DON'T re-probe a box that's already set up. Use workflows for
substantive work. No synthetic data for any accuracy claim (real AADR only).

---
*If you need raw session history (rare — the above usually suffices): grep `~/.claude/projects/-home-suzunik-steppe/*.jsonl` (that's how the milestone-build workflows were recovered).*
