# agentscripts — workflow map & state (compaction-proof)

Saved [Workflow] scripts that build, audit, and fix `steppe`. **This file is the
single source of truth for: where we are, which workflows exist + their state, how
to retrigger one, and which workflows we still need to write.** Read it first after a
compaction. Each script is a self-contained JS [Workflow] rerunnable via
`Workflow({scriptPath: "agentscripts/<name>.js"})`.

---

## WHERE WE ARE (2026-06-18)
- **`main` @ `ac0ca6f`** = Phase-1 precompute **M0–M4 + the before-M4.5 cleanup (B1–B27)**, 24/24 green.
- **branch `m4.5-multigpu` @ `867a4bf`** (46 commits ahead of main, **being MERGED to `main` now**) = **M4.5 single-node multi-GPU DONE — correct, bit-identity-proven, AND now FASTER than single-GPU.** Block-aligned shard + host-staged fixed-order combine + opt-in `cudaMemcpyPeer` device-resident P2P combine. The production `EmulatedFp64{40}` path is `memcmp`-identical single-GPU vs G==2 on real AADR — proven on **both** the host-staged AND the device-resident P2P combine paths, on **both** datasets (`derived_acc` P=50 + `derived_full` P=768).
- **✅ KEY RESULT (the bench `tests/reference/bench_f2_multigpu.cu`, rtxbox 2×RTX PRO 6000 sm_120, Release, `EmuFp64{40}`, median of 10):** multi-GPU is now **FASTER than single-GPU** — P=768 G1=2342ms vs G2=2125ms = **1.10×**; P=400 = **1.22×**. (This REPLACES the old pre-fix "multi-GPU is slower (0.70× / 0.75–0.97×)" finding, which is now FALSE.) **Root cause of the old slowdown, found by nsys (`docs/cleanup/m4.5/why-multigpu-slow.md`): the DATA-BOUNCE wart** — `compute_f2_blocks` D2H-copied each device partial to host and freed its device buffers, forcing the P2P combine to re-upload (H2D) + place-add + a 2nd D2H (a redundant SECOND full 7.14 GB D->H copy). **FIXED @ `867a4bf` by the device-resident combine:** per-device compute leaves its partial RESIDENT (returns an opaque move-only `DevicePartial`), the combine allocates ONE root result, D2D-copies the root partial + `cudaMemcpyPeer`s each peer partial straight into its DISJOINT block slice, then ONE final D2H — deleting the re-upload, the accumulator memset, the place-add kernel, the staging buffers, and the per-peer sync. **Refuted (do NOT carry as open problems):** the `cudaMalloc`/`cudaFree` serialization theory (nsys measured **74% overlap**, not ~18%; the L4 pool allocator is at most a few % and is NOT the lever) and the "combine host-zeroing dominates" theory (the memset is 5ms; P1 was reverted). Single-GPU still OOMs at P≥700 on a 32 GB 5090, so there multi-GPU is *also* enabling.
- **Boxes:** `box5090` (vast 2×RTX 5090, 32 GB, CONSUMER tier, `can_access_peer=false`; `ssh box5090` = `-i ~/.ssh/id_vastai -p 43215 root@78.92.24.57`; **flaky network — long ssh drops, run long jobs DETACHED + poll a logfile**). `rtxbox` (2×RTX PRO 6000, 96 GB, P2P-capable; **currently SPUN DOWN — ephemeral, update the `~/.ssh/config` alias to the new IP before use**). See `[[rtxbox]]` / `[[steppebox5090]]` memories. nvcc not on PATH on either → `export PATH=/usr/local/cuda/bin:$PATH`.

## THE 3 WORKFLOW PATTERNS (the user's conventions — honor them)
- **Implementation** (milestones / new code): **Contracts/Design → per-file Implement (1 devoted agent per file/unit) → Build agent → Verify agent**. Reference: `steppe-m0-scaffold-f2.js`, `m4.5-scaffold.js`, `m4.5-multigpu.js`.
- **Fix-pass** (cleanup / B-items): **STRICTLY SEQUENTIAL, 2 agents per item — independent fixer + independent verdict** (fixer does the dev-loop; verdict adversarially re-checks the diff + re-runs build/ctest + commits-green/reverts). Reference + template: **`fix-pass-phase2.js`**. Halt-on-fail for dependent items (phase1); skip-and-continue for independent (phase2).
- **Audit** (review): per-unit **deep auditor → adversarial critic → holistic capstone**, READ-ONLY → `docs/cleanup/`. Reference: `per-file-cleanup-audit.js`, `m4.5-cleanup-audit.js`.
- **Research/verify**: parallel read-only fan-out.
- All builds/tests run on the box; nothing builds locally. Commit-green / revert; never `git add .` (leave `aadr/`, `build_run.sh`, `f2_emu_spike.cu`, `handoff-*.md` untracked).

---

## EXISTING WORKFLOWS (retrigger via `Workflow({scriptPath: "agentscripts/<name>.js"})`)

| Script | Pattern | What it does | State |
|---|---|---|---|
| `steppe-m0-scaffold-f2.js` | impl | M0: lift the f2 kernel into the architecture | ✅ done — **reference impl pattern** |
| `steppe-f2-realdata-loader.js` | impl | M1: real-AADR Q/V/N loader / decode | ✅ done |
| `steppe-aadr-realdata-test.js` | impl/test | real-data equivalence test wiring | ✅ done |
| `m4.5-scaffold.js` | impl | M4.5 capability-tier scaffold (STEPPE_CUDA_WARN, BackendCapabilities, prefer_p2p_combine, MathModeScope, device_id) | ✅ done (5 units, merged into branch) |
| `m4.5-multigpu.js` | impl | M4.5 algorithm (Resources, shard, host-staged combine, P2P combine, parity gate) | ✅ done (Design+I1–I3) |
| `fix-pass-phase1.js` | fix-pass | M0–M4 cleanup B7,B1–B6 (halt-on-fail) | ✅ done |
| **`fix-pass-phase2.js`** | fix-pass | M0–M4 cleanup B8–B27 (skip-continue) | ✅ done — **the fix-pass TEMPLATE** |
| `m4.5-fix-pass.js` | fix-pass | M4.5 cleanup T1,B1,B3–B9 on box5090 | ✅ done (36/36 green) |
| **`m4.5-b2-p2p-fix-pass.js`** | fix-pass | **B2 P2P transport rework (P1 grid-stride+fuse · P2 hoist peer-enable · P3 kill double-bounce · P4 streamed)** | ⚪ **REASSESS — likely subsumed.** Its original motivation (the slow/bouncing combine) is largely addressed by the device-resident combine @ `867a4bf`; the double-bounce is already killed. Re-scope before running rather than treating as a pending speedup |
| `m4.5-why-slow-investigation.js` | research | nsys "measure-first" root-cause probe of the multi-GPU slowdown → `docs/cleanup/m4.5/why-multigpu-slow.md` (the DATA-BOUNCE wart; 74% overlap, refuted the `cudaMalloc`-serialization theory) | ✅ done (root cause @ `165f655`) |
| `m4.5-device-resident-combine.js` | impl | the device-resident combine fix: partials stay resident (`DevicePartial`), root D2D + `cudaMemcpyPeer` into disjoint slices + one final D2H | ✅ done — **the speedup @ `867a4bf` (now 1.10×/1.22×)** |
| `per-file-cleanup-audit.js` | audit | M0–M4 28-unit audit → `docs/cleanup/` | ✅ done |
| `m4.5-cleanup-audit.js` | audit | M4.5 11-unit audit → `docs/cleanup/m4.5/` (8.4/10; the B1–B9 backlog) | ✅ done |
| `code-cleanup-review.js` | audit | earlier lens-based review | ⚪ superseded by per-file |
| `cleanup-audit-recovery.js`, `cleanup-finish.js` | audit | recovery passes after a server incident | ⚪ one-off, done |
| `research-gpu-dominant-pipeline.js` | research | GDS/nvCOMP/CUDA-graphs/async CPU-offload research | ✅ done (folded into TODO ⚡) |
| `verify-no-alternatives-gpu.js` | research | adversarial re-verify of "impossible on consumer 5090" (aikitoria P2P) | ✅ done |
| `verify-reviewer-feedback.js` | research | adversarial verify of a reviewer critique | ✅ done |
| `steppe-arch-redesign-gemm-multigpu-etl.js` | research | architecture redesign (GEMM/multi-GPU/ETL) | ✅ done |
| `steppe-precision-policy-revision.js` | research | precision-policy doc revision | ✅ done |
| `steppe-aadr-dataprep.js` | data | authored the AADR data-prep scripts | ✅ done (generator = `aadr/build_tgeno_matrix.py`) |
| `steppe-f2-emulated-fp64-spike.js` | research | the fixed-slice Ozaki precision/throughput spike | ✅ done |
| `cuda-qpadm-scaffold-prompt.js` | research | initial architecture/scaffold-prompt generation | ✅ done |

**Not a workflow but the matching bench tool:** `tests/reference/bench_f2_multigpu.cu` — single-GPU vs G==2 wall-clock SWEEP over P (repacks `derived_full` subsets; no data regen). Now `ITERS=10`, reports **median + p10/p90** with out-of-band timers. Build `cmake --build build --target bench_f2_multigpu`; run `./build/bin/bench_f2_multigpu [root] [P1 P2 ...]`. The byte-traffic columns are observability-only (currently print 0.00). Latest rtxbox result: P=768 = 1.10×, P=400 = 1.22× (post-`867a4bf`).

---

## WORKFLOWS WE STILL NEED TO AUTHOR (the forward map)

| Need | Pattern | What / why | Box | Priority |
|---|---|---|---|---|
| ~~**L4 pool-allocator fix-pass**~~ | fix-pass | ~~`cudaMallocAsync` pool so the fan-out threads stop serializing on `cudaMalloc`~~ | — | ⚪ **REFUTED — not the lever.** nsys measured 74% overlap; the allocator is at most a few %. The speedup came from the device-resident combine (`867a4bf`), not a pool. Demoted to **optional** cleanup, not a blocker |
| ~~**nsys measure-first profile**~~ | research/probe | ~~profile the G==2 run to classify the bottleneck~~ | — | ✅ **DONE** → `m4.5-why-slow-investigation.js` / `docs/cleanup/m4.5/why-multigpu-slow.md`. Found the DATA-BOUNCE wart (redundant 2nd 7.14 GB D2H), refuted the `cudaMalloc`-serialization theory |
| ~~**B2 P2P fix-pass**~~ | fix-pass | already written → `m4.5-b2-p2p-fix-pass.js` | rtxbox | ⚪ **REASSESS — likely subsumed** by the device-resident combine @ `867a4bf` |
| **qpAdm FIT engine** | impl | **Phase 2 — does not exist yet.** The actual next milestone after the M4.5 merge: the qpAdm fit/solve on top of the precomputed f2 blocks. | any | **HIGH — the next real work** |
| **AT2 goldens** | impl | install R + admixtools, generate pinned `extract_f2` goldens (R ver/RNGkind/AT2 ver/blgsize/boot/seed per §12), wire the AT2-parity gate (the real M7 acceptance gate, deferred since M1). | rtxbox/PRO | MED (M7) |
| **M5 out-of-core streaming** | impl | pinned double-buffer SNP-tile ingest (the big CPU-offload win; solves the more-SNPs OOM). Composes with B1/L4. | box5090 + rtxbox | MED (next milestone) |
| **M6 multi-dataset merge** | impl | S-2 intersection/union, polarity, drop ambiguous (no strand flip). | any | LATER |
| **M7 on-disk cache + FST** | impl | ADMIXTOOLS-compatible `f2_blocks` cache + FST + GDS/nvCOMP. Solves the more-populations OOM (don't hold all f2_blocks resident). | rtxbox/PRO | LATER |

**Non-workflow next steps:** merging `m4.5-multigpu` → `main` NOW (M4.5 complete — correct AND faster); prune merged feature branches. The perf rabbit-hole is **CLOSED**. After merge, the real next work is the **qpAdm FIT ENGINE (Phase 2, does not exist yet)**, then M5 out-of-core streaming, AT2 goldens (M7 gate), M6 merge. Remaining M4.5 perf is **optional only**: pin the final pageable result D2H (the next available speedup, could push past 1.10×).

## RETRIGGER MECHANICS
- **Setting up / refreshing a box** (new instance, or after a spin-up): follow **`docs/BOX-RUNBOOK.md`** (SSH alias → verify/reject → install tooling → nvcc-PATH → rsync → data → build → P2P probe), or run the automated `scripts/box_bringup.sh <ssh-alias> [--build]`. P2P check: `scripts/p2p_probe.cu`.
- Run a workflow: `Workflow({scriptPath: "/home/suzunik/steppe/agentscripts/<name>.js"})` (resume a paused one with `resumeFromRunId`). Watch with `/workflows`.
- **box5090 long jobs (build/ctest/bench):** the vast box drops long ssh connections — run **detached on the box** (`setsid bash -c "... > /tmp/run.log 2>&1; echo DONE >> /tmp/run.log" </dev/null &`) and poll `/tmp/run.log` in short ssh reads (a `tail`/`grep ALLDONE` loop), don't hold one long ssh.
- **rtxbox:** ephemeral + currently down — `ssh rtxbox` alias in `~/.ssh/config` is stale; update HostName/Port to the new instance, confirm `nvidia-smi -L` = 2×RTX PRO 6000, rsync, confirm `/workspace/data/aadr/{raw,derived_acc,derived_full}` (regen with `build_tgeno_matrix.py --auto-top N` if missing) before any rtxbox workflow.
