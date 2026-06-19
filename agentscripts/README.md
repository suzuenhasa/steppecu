# agentscripts — workflow map & state (compaction-proof)

Saved [Workflow] scripts that build, audit, and fix `steppe`. **This file is the
single source of truth for: where we are, which workflows exist + their state, how
to retrigger one, and which workflows we still need to write.** Read it first after a
compaction. Each script is a self-contained JS [Workflow] rerunnable via
`Workflow({scriptPath: "agentscripts/<name>.js"})`.

---

## WHERE WE ARE (2026-06-19)
- **PHASE-1 PRECOMPUTE IS SUBSTANTIALLY COMPLETE THROUGH M5.** `main` carries M0–M4 (3-GEMM f2 kernel + real-AADR loader + filters + per-block `f2_blocks` tensor) + the before-M4.5 cleanup (B1–B27) + M4.5 single-node multi-GPU (block-aligned shard, host-staged + device-resident P2P combine, bit-identical). **branch `m5-input-streaming` @ `~60b0332`** (the device-resident output + the full M5 streaming chain, **about to be MERGED to `main`**) finishes the precompute through M5.
- **DEVICE-RESIDENT OUTPUT (`1f80c0c`) — the real win.** The precompute now returns a **DEVICE-RESIDENT** handle (`DeviceF2Blocks`; the result stays in VRAM); the host `F2BlockTensor` is an opt-in `.to_host()`. **MEASURED: P=512 device-resident ~673ms vs ~2879ms bulk-to-host = ~4.3×.** THE KEY LESSON: the precompute was **HOST-RESULT-BOUND** — ~80% of the old wall was copying the 6.36 GB+ result to CPU; getting it OFF the CPU was the real win, **NOT multi-GPU**.
- **M5 OUT-OF-CORE STREAMING — DONE** (tiered output `176a07d` + SNP-tile input streaming `c65179f`). **(a) ADAPTIVE TIERED OUTPUT, not mandatory:** the result goes to the fastest tier it FITS — VRAM-resident (small P, keeps the 4.3×) → host RAM (big box) → disk (laptop), auto-selected from runtime free VRAM/RAM. **(b) SNP-tile INPUT streaming:** per-block decode, GPU footprint `O(P*tile + P²)` **INDEPENDENT of M** (no `7*P*M` feeder wall). **RESULT: full-autosome (M=584131, n_block=757) P=2500 COMPLETES on a SINGLE 32 GB RTX 5090 in ~51.5s** (76 GB result streamed, GPU peak ~26 GB bounded), parity memcmp **BIT-IDENTICAL**. Measured sweep (one 5090, streamed): P=512 ~3.6s, P=1000 ~10.4s, P=1500 ~20.2s, P=2000 ~34.0s, P=2500 ~51.5s.
- **THE HONEST MULTI-GPU STORY (corrects the older claims here):** M4.5 multi-GPU was built and bit-identity-proven, but the REAL perf wins came from **device-resident output + streaming (getting off the CPU), NOT multi-GPU per se**. On the precompute, multi-GPU is a modest throughput layer (and was *measured slower* than single-GPU until the data-bounce was fixed — nsys showed ~22–74% overlap, a serial D2H/host tail; see `docs/cleanup/m4.5/parallelism-check.md`, `why-d2h.md`, `why-multigpu-slow.md`, `architecture-audit.md`). Multi-GPU genuinely **SHINES on the FIT/ROTATION phase** (thousands of INDEPENDENT models, no combine) — that is its proper home, not the precompute.
- **vs ADMIXTOOLS 2** (`docs/research/at2-timing-comparison.md`, sourced + verified-vs-estimated): the f2 precompute is *plausibly* **~2–3 orders of magnitude** faster than AT2 `extract_f2` at thousands-of-pops scale — but AT2 published NO `extract_f2`-vs-P timing, so that ratio is an **ESTIMATE** (complexity `O(P²*M)` + the one "~a day → 20min" anecdote), anchored to steppe MEASURED P=2000=15.1s (PRO 6000) / P=2500=51.5s (one 5090). Do NOT state a single clean "N×" as fact — it is an estimate.
- **THE NEXT PHASE = the qpAdm FIT ENGINE (Phase 2, S3–S8), which DOES NOT EXIST YET.** It reads `f2_blocks` (device-resident for the in-VRAM case per `why-d2h.md`; streamed tiles for large P), runs the GLS solve + rank test + the MODEL ROTATION (the embarrassingly-parallel, multi-GPU-friendly phase). AT2 goldens are the validation gate.
- **Boxes:** `box5090` (vast 2×RTX 5090, 32 GB, CONSUMER tier, `can_access_peer=false`; `ssh box5090` = `-i ~/.ssh/id_vastai -p 43215 root@78.92.24.57`; **flaky network — long ssh drops, run long jobs DETACHED + poll a logfile**). `rtxbox` (2×RTX PRO 6000, 96 GB, P2P-capable; **ephemeral — update the `~/.ssh/config` alias to the new IP before use**). See `[[rtxbox]]` / `[[steppebox5090]]` memories. nvcc not on PATH on either → `export PATH=/usr/local/cuda/bin:$PATH`.

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
| **`m4.5-b2-p2p-fix-pass.js`** | fix-pass | **B2 P2P transport rework (P1 grid-stride+fuse · P2 hoist peer-enable · P3 kill double-bounce · P4 streamed)** | ⚪ **SUBSUMED.** Its motivation (the slow/bouncing combine) was addressed by the device-resident combine `867a4bf` + the device-resident output `1f80c0c`; the double-bounce is killed. Historical |
| `m4.5-perf-discovery.js` | research | DIAGNOSE why G==2 was slower than G==1; Release re-bench + nsys G1-vs-G2 → the parity-safe speedup plan `docs/cleanup/m4.5/perf-discovery.md` | ✅ done (the perf plan) |
| `m4.5-why-slow-investigation.js` | research | nsys "measure-first" root-cause probe of the multi-GPU slowdown → `docs/cleanup/m4.5/why-multigpu-slow.md` (the DATA-BOUNCE wart; 74% overlap, refuted the `cudaMalloc`-serialization theory) | ✅ done (root cause @ `165f655`) |
| `m4.5-device-resident-combine.js` | impl | the device-resident P2P combine fix: partials stay resident (`DevicePartial`), root D2D + `cudaMemcpyPeer` into disjoint slices + one final D2H | ✅ done @ `867a4bf` |
| `m4.5-perf-fix-pass.js` | fix-pass | make multi-GPU actually faster per the perf-discovery plan (P0 release-clean → P1 resize-not-zero → P2 per-device Stream → P3 buffer reuse → P4 pinned H2D); Release on rtxbox | ✅ done (P0 `970fa42`, P2 `9fdc946`, P3 `a41d67a`; P1 reverted as a sham, redone separately) |
| `m4.5-p1-redo.js` | fix-pass | REDO P1 (the combine zero-fill lever via a CUDA-free default-init allocator) — the one genuine gap the forensic audit caught (original P1 was a sham). Sham-hardened verdict | ✅ done |
| `m4.5-parallelism-check.js` | research | test "is it TRUE parallel multi-GPU?" — nsys per-device timeline on box5090 → `docs/cleanup/m4.5/parallelism-check.md` (only 22.7% GPU-overlap; the per-call pinning D2H tail serializes the two devices) | ✅ done (`52deaff`) |
| `m4.5-hoststaged-direct-d2h.js` | impl | host-staged combine speed fix: each device D2Hs its compact partial DIRECTLY into its disjoint block-slice of ONE pinned result (delete the separate partials buffer + the `copy_n`) | ✅ done (`a84b85b`) |
| `m4.5-persistent-pin-d2h.js` | impl | kill the per-call `cudaHostRegister`/`Unregister` D2H serialization (persistent per-backend pinned staging) so the two devices' D2Hs overlap | ✅ done (`94c6d8e`) |
| `m4.5-why-d2h.js` | research | the architectural question: WHY is there a D2H of `f2_blocks` at all — fundamental (host/disk consumer) or incidental (phase boundary + unbuilt fit)? → `docs/cleanup/m4.5/why-d2h.md` (incidental for the fits-in-VRAM case → keep it device-resident) | ✅ done |
| `m4.5-architecture-audit.js` | audit | ADVERSARIAL architecture audit of the multi-GPU precompute (data-layout/parallelism/algorithm/system-fit) → `docs/cleanup/m4.5/architecture-audit.md` (the full-resident-host model is wrong for the block-by-block consumer; the streaming/tiled levers) | ✅ done |
| `m4.5-scaling-sweep.js` | bench | at-scale OOM-tolerant 3-path P-sweep on rtxbox (768→2500) → `docs/cleanup/m4.5/scaling-sweep.md` | ⚪ done — **PRE-M5, now STALE** (it reports P=2500 OOMs everywhere; M5 streaming now completes P=2500; superseded by `docs/cleanup/m5/00-results.md`) |
| `m4.5-gpu-resident-output.js` | impl | get `f2_blocks` OFF the CPU: device-resident output handle, opt-in `.to_host()`, no forced D2H | ✅ done @ `1f80c0c` (~4.3× @P=512) |
| `m5-tiered-streaming.js` | impl | M5 adaptive tiered output (Resident/HostRam/Disk, auto-selected from free VRAM/RAM; block-axis triple-buffered spill) | ✅ done @ `176a07d` |
| `m5-input-streaming.js` | impl | M5 SNP-tile INPUT streaming: per-block-tile decode/upload, GPU footprint `O(P*tile + P²)` — removes the `7*P*M` feeder wall; full-autosome any-P on a 32 GB card | ✅ done @ `c65179f` |
| `turboquant-l2-research.js` | research | could a TurboQuant-like L2-resident quantization fit steppe (precision-critical) without breaking §12 parity → `docs/research/turboquant-l2-experiment.md` | ✅ done (`13e4633`; deferred until the fit exists) |
| `at2-timing-comparison.js` | research | how long the equivalent precompute/fit takes in ADMIXTOOLS 2 / qpAdm, sourced + verified-vs-estimated → `docs/research/at2-timing-comparison.md` | ✅ done (`60b0332`) |
| `fixpass-completeness-audit.js` | audit | forensic cross-check of every fix-pass's DECLARED items vs the actual code/git history → `docs/cleanup/missing-fixes.md` (caught the P1 sham) | ✅ done (`b4c2913`) |
| `m4.5-docs-refresh.js` | docs | refresh all docs to the post-`867a4bf` reality (multi-GPU correct + bit-identical; the refuted hypotheses) | ✅ done |
| `m5-docs-refresh.js` | docs | refresh all docs to the post-M5 reality (precompute complete through M5; the honest multi-GPU story; next = fit engine) + the new M5 results doc | ✅ done (`d0fdfe4`) |
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

**Not a workflow but the matching bench tool:** `tests/reference/bench_f2_multigpu.cu` — single-GPU vs G==2 wall-clock SWEEP over P (repacks `derived_*` subsets; no data regen), `ITERS=10`, reports **median + p10/p90** with out-of-band timers. The `--tiered` flag exercises the M5 device-resident / tiered-streaming path (used by the full-autosome sweep over `derived_2500`, M=584131, n_block=757). Build `cmake --build build-rel --target bench_f2_multigpu`; run `./build-rel/bin/bench_f2_multigpu [--tiered] [root] [P1 P2 ...]`. Byte-traffic columns are observability-only. **Headline result (one 5090, `--tiered`, full-autosome, streamed):** P=512 ~3.6s, P=1000 ~10.4s, P=1500 ~20.2s, P=2000 ~34.0s, **P=2500 ~51.5s COMPLETES** (76 GB result, GPU peak ~26 GB). Device-resident @P=512 ~673ms (~4.3× vs bulk-to-host). The earlier rtxbox multi-GPU bench (P=768 G2/G1=1.10×, P=400=1.22× post-`867a4bf`) was the M4.5 result before device-resident output / streaming reframed the win.

---

## WORKFLOWS WE STILL NEED TO AUTHOR (the forward map)

| Need | Pattern | What / why | Box | Priority |
|---|---|---|---|---|
| **qpAdm FIT engine** | impl | **Phase 2 (S3–S8) — does not exist yet. THE NEXT REAL WORK.** Reads `f2_blocks` (device-resident for the in-VRAM case per `why-d2h.md`; streamed tiles for large P), runs the GLS solve + rank test + the MODEL ROTATION (many independent qpAdm models = the embarrassingly-parallel, multi-GPU-friendly phase — multi-GPU's proper home, not the precompute). | any | **HIGH — the next phase** |
| **AT2 goldens** | impl | install R + admixtools, generate pinned `extract_f2` (and later `qpadm`) goldens (R ver/RNGkind/AT2 ver/blgsize/boot/seed per §12), wire the AT2-parity gate (the real acceptance gate for the fit; deferred since M1). | rtxbox/PRO | HIGH (the fit's validation gate) |
| **Multi-GPU block-sharding on the stream** | impl | shard the M5 block-stream across G devices for throughput (the precompute's modest multi-GPU layer, now bounded by the per-block working set not VRAM). Deferred / optional — single-GPU streaming already completes full-autosome P=2500. | rtxbox/PRO | LOW (optional throughput) |
| **M6 multi-dataset merge** | impl | S-2 intersection/union, polarity, drop ambiguous (no strand flip). | any | LATER |
| **M7 on-disk cache + FST** | impl | ADMIXTOOLS-compatible `f2_blocks` cache + FST + GDS/nvCOMP. The M5 disk tier already writes a streamable on-disk artifact; M7 hardens it into the persistent cache + FST format. | rtxbox/PRO | LATER |

**Non-workflow next steps:** merging `m5-input-streaming` → `main` NOW (the precompute is complete through M5 — device-resident output + tiered/SNP-tile streaming, full-autosome P=2500 runs on a 32 GB card); prune merged feature branches. The precompute perf rabbit-hole is **CLOSED** (the win was getting OFF the CPU, not multi-GPU). After merge, the real next work is the **qpAdm FIT ENGINE (Phase 2, does not exist yet)** with AT2 goldens as its validation gate, then M6 merge / M7 cache. Optional deferrals: multi-GPU block-sharding on the stream; the device-resident final-D2H pinning on rtxbox; the TurboQuant-L2 rotation screen (`docs/research/turboquant-l2-experiment.md`, *after* the fit exists).

## RETRIGGER MECHANICS
- **Setting up / refreshing a box** (new instance, or after a spin-up): follow **`docs/BOX-RUNBOOK.md`** (SSH alias → verify/reject → install tooling → nvcc-PATH → rsync → data → build → P2P probe), or run the automated `scripts/box_bringup.sh <ssh-alias> [--build]`. P2P check: `scripts/p2p_probe.cu`.
- Run a workflow: `Workflow({scriptPath: "/home/suzunik/steppe/agentscripts/<name>.js"})` (resume a paused one with `resumeFromRunId`). Watch with `/workflows`.
- **box5090 long jobs (build/ctest/bench):** the vast box drops long ssh connections — run **detached on the box** (`setsid bash -c "... > /tmp/run.log 2>&1; echo DONE >> /tmp/run.log" </dev/null &`) and poll `/tmp/run.log` in short ssh reads (a `tail`/`grep ALLDONE` loop), don't hold one long ssh.
- **rtxbox:** ephemeral — `ssh rtxbox` alias in `~/.ssh/config` may be stale; update HostName/Port to the new instance, confirm `nvidia-smi -L` = 2×RTX PRO 6000, rsync, confirm `/workspace/data/aadr/{raw,derived_acc,derived_full,derived_2500}` (`derived_2500` = full-autosome M=584131/n_block=757, the M5 sweep input; regen with `build_tgeno_matrix.py --auto-top N` if missing) before any rtxbox workflow.
