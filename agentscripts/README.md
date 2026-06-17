# agentscripts — multi-agent workflow scripts

Saved [Workflow] scripts used to build, audit, and fix `steppe`. Each is a self-contained
JS workflow rerunnable via the Workflow tool: `Workflow({scriptPath: "agentscripts/<name>.js"})`.
They orchestrate subagents deterministically — **fan-out** for research/review, **strictly
sequential fixer→verdict** for code changes.

| script | what it did |
|---|---|
| `cuda-qpadm-scaffold-prompt.js` | initial architecture / scaffold-prompt generation |
| `steppe-aadr-dataprep.js` | AADR fetch + per-pop Q/V/N matrix prep |
| `steppe-f2-emulated-fp64-spike.js` | the f2 emulated-FP64 precision/throughput spike (fixed-slice Ozaki) |
| `steppe-precision-policy-revision.js` | precision-policy doc revision |
| `steppe-arch-redesign-gemm-multigpu-etl.js` | architecture redesign (GEMM reformulation / multi-GPU / ETL) |
| `verify-reviewer-feedback.js` | adversarial verify of an external reviewer critique |
| `research-gpu-dominant-pipeline.js` | GPU-dominant / CPU-offload research (GDS, nvCOMP, CUDA graphs, async) |
| `verify-no-alternatives-gpu.js` | adversarial re-verify of "impossible on consumer 5090" (aikitoria P2P, etc.) |
| `per-file-cleanup-audit.js` | **deep per-file adversarial audit** — 1 devoted agent + critic per unit → `docs/cleanup/` |
| `code-cleanup-review.js` | earlier lens-based cleanup review (superseded by per-file) |
| `cleanup-audit-recovery.js`, `cleanup-finish.js` | recovery/finish passes for the audit after a server-side incident |
| `fix-pass-phase1.js` | **Phase-1 cleanup fix pass** (B7,B1–B6) — sequential fixer+verdict, build+`ctest`-gated, commit-green / revert-halt |

**Patterns:**
- **Research / review** = parallel fan-out, **read-only** (safe to parallelize).
- **Code-fix** = STRICTLY SEQUENTIAL (fixes share files + one box build dir), **two agents per task** (independent fixer + independent verdict), **commit-green / revert-on-fail**, **retry-once-on-transient-500**. Phase 1 used halt-on-fail; Phase 2 (mostly-independent items) uses skip-and-continue.
- All builds/tests run on the remote GPU box (`/workspace/steppe`); nothing builds locally.
