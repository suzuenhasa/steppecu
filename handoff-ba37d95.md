# steppe — cold-start handoff @ `m4.5-multigpu` `ba37d95`

Everything a fresh session needs to resume. Canonical spec: `docs/architecture.md`; build order + status: `docs/ROADMAP.md`; living task list: `docs/TODO.md`; the per-file audit + master cleanup backlog: `docs/cleanup/00-overview.md` + `docs/cleanup/m4.5/00-overview.md`. **→ THE WORKFLOW MAP (where we are + every workflow's state/retrigger + the workflows still needed) is `agentscripts/README.md` — read it first.** **Read these before writing code.**

---

## 0. TL;DR
`steppe` = GPU/CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics (precompute-once / fit-many), senior-engineer quality bar. **Status: Phase 1 precompute M0–M4 + before-M4.5 cleanup (B1–B27) merged to `main` @ `ac0ca6f`; M4.5 single-node multi-GPU + its cleanup fix-pass DONE on branch `m4.5-multigpu` @ `ba37d95` (36/36 ctest green, bit-identity-proven on real AADR, NOT yet merged).** The M4.5 cleanup landed T1 (parity VRAM-gate), B1 (fan-out), B3–B9; **B2 (P2P transport) is STAGED for rtxbox** (`agentscripts/m4.5-b2-p2p-fix-pass.js`).
- **⚠ THE KEY OPEN FINDING (bench `tests/reference/bench_f2_multigpu.cu`, P-sweep on box5090): multi-GPU is currently SLOWER than single-GPU at every P (0.75–0.97×) — B1's fan-out alone is NOT a speedup.** Prime suspect: per-call `cudaMalloc`/`cudaFree` (device-wide-synchronizing + global driver lock, audit **L4**) serializes the two device-threads; + the host-staged combine's D2H round-trip. Single-GPU OOMs at P≥700 on a 32 GB 5090 (so there multi-GPU is *enabling*, not faster).
- **NEXT:** (a) **L4 pool-allocator fix-pass** (the speedup enabler — write it) + an **nsys profile** to confirm the `cudaMalloc` serialization (§11.3 measure-first gate); (b) **B2 P2P fix-pass on rtxbox** (staged) + measure the real speedup there (P2P + single-GPU P=768 fits); (c) merge `m4.5-multigpu` → main; then M5 (streaming) / M7 (AT2 goldens). **See `agentscripts/README.md` for the full forward map.**

## 1. Boxes
- **CAPABLE box = rtxbox (Verda): CURRENTLY SPUN DOWN** (was `ssh rtxbox` → `root@31.22.104.224`; ephemeral — the alias IP is now STALE, update `~/.ssh/config` to the new instance before use). 2× **RTX PRO 6000 Blackwell** (sm_120, 96 GB ea), CUDA 13.0.88, driver 580, $3.78/h. **GOTCHA: `nvcc` not on PATH → `export PATH=/usr/local/cuda/bin:$PATH`.** **Real stock-driver P2P confirmed: `cudaMemcpyPeer` byte-exact 55.6 GB/s, `can_access_peer==true`.** Needed for: B2 P2P fix-pass, the real multi-GPU speedup measurement, AT2 goldens, GDS. See [[rtxbox]].
- **BUDGET box = box5090 (vast 2× RTX 5090, UP):** `ssh box5090` = `-i ~/.ssh/id_vastai -p 43215 root@78.92.24.57`. 32 GB ea, CONSUMER tier (`can_access_peer=false` → host-staged combine; the graceful-degrade target). **Flaky network — long ssh drops; run long build/ctest/bench DETACHED on the box + poll a `/tmp/*.log` (see `agentscripts/README.md` retrigger mechanics).** See [[steppebox5090]].
- **DATA strategy:** vast→rtx path is SLOW (~1 MB/s). `raw` came from LOCAL (~10 MB/s); `derived_*` REGENERATED on the box from raw via `python3 /workspace/data/aadr/build_tgeno_matrix.py --geno raw/...geno --ind raw/...ind --out <dir> <flags>` (needs `apt install python3-numpy`). `derived_acc` = `--auto-top 50 --snp-cap 100000`; `derived_full` = `--auto-top 768`.
- **DISK/cores:** run tests with `ulimit -c 0` (asserts abort → core dumps).

## 2. Working rules (hard)
NOTHING builds/runs locally — author locally → rsync to `/workspace/steppe` → build/test on the box. NO synthetic data for any precision/accuracy claim (real AADR only). git identity `suzuenhasa <suzu@enhasa.co>`. Commit only on green, per-milestone, ROADMAP §6 message + `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Don't reuse the old MVP at `/home/suzunik/vahaduo/qpadm/steppe`.

## 3. Build & run (on rtxbox)
```
ssh rtxbox
cd /workspace/steppe
export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0
cmake -S . -B build -GNinja && cmake --build build
ctest --test-dir build --output-on-failure
```
Dev loop: edit locally → `rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh /home/suzunik/steppe/ rtxbox:/workspace/steppe/` → build/test on box → commit locally on green. **30/30 ctest green at `18b7801`** (~140 s; `filter_oracle` ~45 s = the CPU long-double parity oracle).

## 4. The workflow PATTERN (this is how milestones get built — the user insists on it)
- **Implementation** (milestone/new code) = the `steppe-m0-scaffold-f2` shape: **Contracts/Design** (1 agent fixes interfaces) → **Implement** (1 devoted agent PER FILE/unit) → **Build agent** (rsync+build+ctest fix-loop) → **independent Verify agent** (adversarial diff + commit-green/revert). STRICTLY SEQUENTIAL (shared files + one box build dir), halt-on-fail, retry-once-on-transient-null. The two M4.5 workflows (`agentscripts/m4.5-scaffold.js`, `m4.5-multigpu.js`) are the current reference.
- **Fix/cleanup** = `agentscripts/fix-pass-phase2.js` (per-item fixer + verdict).
- **Research/review** = parallel read-only fan-out.
- All builds/tests on the box; commit-green/revert; never `git add .` (leave `aadr/`, `build_run.sh`, `f2_emu_spike.cu`, `handoff-*.md` untracked).

## 5. State — M4.5 (branch `m4.5-multigpu` @ `18b7801`)
- **Scaffold** (`d81d2a1`→`182d7cb`): `STEPPE_CUDA_WARN` (non-throwing tagged degrade, `device/cuda/check.cuh`) · CUDA-free `BackendCapabilities` POD + `ComputeBackend::capabilities()` (`device/backend.hpp`) · `DeviceConfig::prefer_p2p_combine` + override-knob banner (`config.hpp`) · `CublasHandle` device-ordinal + `MathModeScope` (`handles.hpp`) · `CudaBackend(int device_id)` + `cudaSetDevice` + the real capability probe (`cuda_backend.cu`).
- **Algorithm** (`dd24941`→`18b7801`): `Resources`/`PerGpuResources` + `build_resources` (`include/steppe/resources.hpp` + `resources.cpp`, CUDA-free; one backend+probe per device, fixed `g=0..G-1` order) · **block-aligned shard** + per-device partials + **host-staged fixed-order combine** (`core::compute_f2_blocks_multigpu`; G==1 = exact single-GPU path; reuses `compute_f2_blocks` per device) · **opt-in `cudaMemcpyPeer` P2P device-combine** (`device/cuda/p2p_combine.cu`, gated on `can_access_peer && prefer_p2p_combine`, tagged degrade, `CombinePath` recorded out-of-band on `Resources`).
- **Parity (`test_f2_multigpu_parity`, THE gate):** production `EmulatedFp64{40}` → G==2 host-staged AND P2P **bit-identical** to single-GPU on `derived_acc` + `derived_full`; P2P==host-staged bit-identical. Native `Fp64` at G≥2 → oracle tol (rel ≤2.87e-15) = §12 `[UNCERTAIN]` `cublasGemmStridedBatchedEx` batchCount-sensitivity, correctly scoped (EmulatedFp64 is the production f2 default; native Fp64 is only the oracle). **NEVER NCCL AllReduce on a parity path.**
- **The science:** f2 = 3-GEMM; **fixed-slice Ozaki** emulated-FP64 (40-bit ≈ native at 7–17×); dynamic mantissa = the rejected trap; native FP64 = oracle + numerator/divide.

## 6. NEXT
1. **Merge `m4.5-multigpu` → `main`** (M4.5 is green + proven).
2. **"5090 after" — consumer graceful-degrade validation** on the budget vast 5090 (if still up): the parity test's tier-assertions are currently **PRO-specific** (assert `can_access_peer==true` + P2P-ran) — add a **tier branch** so on a consumer box it asserts `can_access_peer==false` → host-staged path + the "P2P unavailable" degrade tag. Validates the fallback the capability-tier design promises.
3. **M5** (out-of-core SNP-tile streaming — pinned double-buffer ingest, the big CPU-offload win; never deferred, genotype input ≫ VRAM) · **M6** (multi-dataset merge) · **M7** (on-disk cache + FST + **install R+admixtools on a box → AT2 goldens**, the real parity gate).
4. **Deferred:** L1–L19 (`docs/cleanup/00-overview.md`); B5/B6 at-scale; the ncu/nsys profile (the §11.3 measure-first gate); GDS for M5/M7 (real on PRO 6000, dead on consumer).

## 7. Resume / git
- `main` @ `ac0ca6f` (M0–M4 + cleanup + docs/handoff/agentscripts). `m4.5-multigpu` @ `18b7801` = M4.5 (above), branched off main, **unmerged**.
- Merged feature branches `m0-f2-scaffold`/`m1-decode-af`/`m2-filters`/`m3-block-partition`/`m4-perblock-f2` = prunable.
- Untracked strays (leave): `aadr/`, `build_run.sh`, `f2_emu_spike.cu`. This handoff is committed for durability.
- Memory: `~/.claude/projects/-home-suzunik-steppe/memory/` ([[steppe-project]], [[rtxbox]], [[steppebox5090]], [[steppe-dev-process]], [[steppe-old-mvp-deprecated]]).
