# steppe — re-orient prompt (copy/paste into a fresh session)

Paste the block below verbatim into a new session to bring the assistant up to speed.
It is deliberately **read-and-verify** (re-derive state from the canonical docs + git,
don't trust stale recall), so it won't rot as work continues. Keep this file updated when
the build phase changes.

---

```
Refresh yourself on the steppe project before doing anything. Work through this in order and DON'T start any task until you've reported back:

1. READ MEMORY: read /home/suzunik/.claude/projects/-home-suzunik-steppe/memory/MEMORY.md (the index) and open every memory file it lists — especially: steppe-project, steppe-dev-process, steppebox5090, rtxbox, build-the-real-gpu-implementation, gpu-first-architecture-not-cpu-shape, real-data-only-all-results, design-for-scale-not-smallest-model, refactor-process-rules, build-sequence-backend-first, cpu-is-test-only, **aadr-tgeno-goldens-corrupt** (the AT2-parity saga: v66 is TGENO, AT2 misreads it, the 4-fix parity chain, convertf pipeline), fit-phase2-progress, fit-precision-future-proof-emulated, perf-bench-release-build.

2. READ THE DOCS (in this order): docs/RESUME.md (the handoff card) → agentscripts/README.md (the workflow map) → docs/ROADMAP.md → docs/RUN-GUIDE.md → docs/architecture.md (skim; §5 stages, §12 PARITY LAW, §4 layering) → docs/design/fit-engine.md + docs/design/fit-engine-finish-punchlist.md → docs/design/cli-bindings.md (the step-2 CLI/bindings contract) → **the AT2-parity arc: docs/research/{tgeno-at2-support,block-partition-at2,f2-estimator-at2}.md + docs/studies/haak2015.md** → docs/research/{pycuda-cuda13-viability,interop-usecases}.md. Also `ls handoff-*.md` and read the latest.

3. VERIFY STATE WITH GIT (don't trust docs/memory blindly — verify against the repo): cd /home/suzunik/steppe && git branch --show-current && git rev-parse --short HEAD main && git log --oneline -15 && git status --short. Confirm whether main == phase2-fit-engine. If a memory/doc names a file/flag/commit, confirm it still exists before relying on it.

4. CHECK IN-FLIGHT WORK: look for the last running/finished workflow (agentscripts/*.js + its /tmp output or the subagents/workflows transcript) and whether its deliverable landed/committed.

5. RESTATE THE HARD RULES back to me so I know they're loaded: develop locally → rsync → build/test ONLY on the box (box5090, `ssh box5090`, nvcc not on PATH so prefix the CUDA env exports; RELEASE build-rel for anything perf; build the CLI with -DSTEPPE_BUILD_CLI=ON); NO SYNTHETIC DATA for ANY result incl. perf — only real AADR + the AT2 goldens, never synthetic/smoke tests; a task is DONE only when it's the proper GPU implementation (CPU-for-parity is just a step; "runs on GPU" ≠ "properly parallel"); steppe is GPU-only — the CpuBackend is the dev/test parity oracle ONLY, never a user runtime (no CPU wheel, no --device cpu); capable-path (PRO6000/P2P/CUDA13+) is the priority with the 5090 the tagged fallback; verify any CUDA/cuBLAS/cuSOLVER API claim against the CUDA 13.x docs; commit only on green (default ctest + STEPPE_THOROUGH where the CpuBackend oracle is touched) with the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and git identity suzuenhasa <suzu@enhasa.co>; NEVER `git add .` (leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md untracked); clear box core dumps (rm -f /var/lib/vastai_kaalia/data/core-*); use workflows for substantial work; verify-then-question.

6. THE BUILD SEQUENCE (memory build-sequence-backend-first): (1) finish the fit-engine backend — DONE (F1–F6); (1b) **FULL AT2 END-TO-END PARITY — DONE** (the 4-fix chain: blgsize=Morgans + monomorphic-drop → AT2 SNP-anchored block partition → per-sample pseudo-haploid `adjust_pseudohaploid`; steppe matches AT2 raw→f2→qpAdm on diploid+pseudo-haploid to the FP floor; Haak 2015 reproduced); (2) CLI + Python bindings — M(cli-0)/M(cli-1)`qpadm`/M(cli-4)`extract-f2` DONE (the study flow works); next M(cli-2) qpwave → M(cli-3) qpadm-rotate → M(py-1) nanobind + DLPack seam; (3) standalone f-stats (f4/D/f3/qpDstat, then qpfstats/DATES/qpGraph) each WITH its own CLI/bindings. LIKELY-IN-FLIGHT/just-done: the golden regen (regen-goldens-convertf.js, run wm9bhrf3m) — regenerating golden_rot/fit1_NRBIG/fitNA from the convertf-PA; check it committed (fit0/qpwave already correct).

Then give me a tight summary: branch/HEAD + whether main is in sync, what's DONE, what's in-flight, and the next step — and WAIT for my direction. Don't launch anything until I say go.
```
