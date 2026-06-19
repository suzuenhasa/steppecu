export const meta = {
  name: 'm-fit-gpu',
  description: 'The qpAdm fit ON THE GPU — the production implementation (CUDA backend), reading f2_blocks RESIDENT in VRAM, batched-capable. Implements the qpAdm ComputeBackend virtuals (f4 gather from resident f2 / jackknife covariance Q / SVD+ALS rank fit / GLS weights / batched jackknife SE) on the GPU via cuSOLVER + cuBLAS + custom kernels — NOT the CpuBackend. Adds cuSOLVER (+ the search-stream pool) to PerGpuResources. Validated against the af6a8c2 AT2 golden (weights 0.5589/0.4411, p 0.327) RUNNING ON THE GPU. The CpuBackend (M(fit-1)) is only the test-side diff oracle, never run as the deliverable. Design -> implement on box5090 -> INDEPENDENT verdict (GPU result matches the golden + actually runs on the GPU resident). HALT-on-fail.',
  phases: [
    { title: 'Design', detail: 'freeze the CUDA-backend qpAdm impl: f4-gather/Q/SVD-ALS/GLS/SE kernels + cuSOLVER, resident + batched-capable' },
    { title: 'Implement', detail: 'the CUDA backend qpAdm fit on box5090, reading DeviceF2Blocks resident, to green parity vs the golden' },
    { title: 'Verify', detail: 'independent verdict: the GPU fit matches the af6a8c2 golden AND runs on the GPU (resident, no host round-trip); commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45'"
const QPTEST = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -55'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell reimplementation of ADMIXTOOLS 2. Phase-1 precompute (f2_blocks, DEVICE-RESIDENT DeviceF2Blocks) DONE. Phase 2 = the qpAdm fit engine. Branch phase2-fit-engine @ 9b55f09. M(fit-1) already implemented the qpAdm MATH and proved it BIT-EXACT vs the AT2 golden — but on the CpuBackend reference. THIS milestone implements the fit ON THE GPU — the production path.',
  'HARD RULE (the user, emphatic; memory build-the-real-gpu-implementation): deliver the PRODUCTION GPU implementation, NOT a CPU reference. The fit must run on the GPU (CUDA backend), reading f2_blocks RESIDENT in VRAM, batched-capable. The CpuBackend (M(fit-1)) stays ONLY as the test-side diff oracle for localization + the no-GPU fallback — it is NEVER the thing run/reported as the deliverable. Do NOT add CPU-first scaffolding.',
  'WHAT TO BUILD: the CUDA-backend implementation of the qpAdm ComputeBackend virtuals (added in M(fit-1), backend.hpp, batched-capable: F4Blocks carries the n_block axis) — on the GPU: (1) f4-gather: a kernel that builds the f4 matrix X[nl x nr x n_block] by gathering the model pop-subset from the RESIDENT f2 tensor (DeviceF2Blocks, VRAM) per block; (2) jackknife covariance Q: est_to_loo + xtau + Q = xtau*xtau^T via cuBLAS SYRK, on-device, regularize (fudge 1e-4) + invert (cuSOLVER potrf/potri or Cholesky-solve); (3) the rank fit: SVD of X (cuSOLVER gesvd / gesvdjBatched — X is small, 2x5) + the AT2 ALS opt_A/opt_B (20 iters, small GEMMs) on-device; (4) GLS weights: the constrained normal-equations solve (cuSOLVER potrf + solve); (5) SE: the n_block (~708) leave-one-block-out weight re-fits BATCHED on-device (strided/pointer-array batched solves, NOT a host loop) -> jackknife variance. Add cuSOLVER handle (+ the search-stream pool) to PerGpuResources/handles (OQ-10, currently absent). Reads DeviceF2Blocks resident; no per-model host round-trip.',
  'PRECISION (§12): the conditioning-driven policy — the f4 build + the covariance SYRK in EmulatedFp64{40} where well-conditioned; the cancellation-sensitive f-stat differences + the ill-conditioned SVD/Cholesky/GLS in NATIVE FP64 (the design says the rank-test SVD + the weight solve are ill-conditioned -> native FP64). Match the CpuBackend reference (which is bit-exact vs AT2) so the GPU result also matches the golden.',
  'GATE: the GPU (CUDA backend) qpAdm fit for the golden model (target=England_BellBeaker; left=Czechia_EBA_CordedWare,Turkey_N; right=Mbuti,Israel_Natufian,Iran_GanjDareh_N,Han,Papuan,Karitiana) produces weights matching the af6a8c2 golden: CordedWare 0.5589062489, Turkey_N 0.4410937511 (rtol ~1e-6), X/Q cross-checked, p 0.32682. The test (tests/reference/test_qpadm_parity) must exercise the CUDA backend / GPU path (assert it ran on the GPU, f2 resident) — extend it to run BOTH backends (CpuBackend diff oracle + the CUDA backend = the real path) or add a CUDA-path assertion.',
  'BOX = box5090 (vast 2x RTX 5090, UP, sm_120, CUDA 13). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists. RELEASE only. NOTHING builds locally. f2 golden fixture + AADR data on the box. The golden f2 fixture (tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin) is loaded into a DeviceF2Blocks for the GPU fit.',
  'KEY FILES: src/device/backend.hpp (the qpAdm virtuals — implement on CUDA), src/device/cuda/cuda_backend.cu (+ NEW src/device/cuda/qpadm_*.cu kernels: f4 gather, jackknife, the small-LA via cuSOLVER), src/device/cuda/handles.hpp + src/device/resources.hpp (ADD cuSOLVER handle + search streams to PerGpuResources), src/core/qpadm/* (the host-pure orchestrator — already dispatches through ComputeBackend; should now drive the CUDA backend), include/steppe/qpadm.hpp (run_qpadm reading DeviceF2Blocks), tests/reference/test_qpadm_parity.cu (extend to the GPU path), src/device/device_f2_blocks.hpp.',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); the qpAdm test (' + QPTEST + '). Iterate until the qpAdm parity test passes ON THE GPU (CUDA backend) vs the af6a8c2 golden. Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA architect. Produce the EXACT FROZEN contract for the qpAdm fit ON THE GPU (CUDA backend) — the production path, reading f2_blocks resident, batched-capable. READ-ONLY (read backend.hpp qpAdm virtuals + the CpuBackend reference impl from M(fit-1) + cuda_backend.cu/handles/resources + device_f2_blocks.hpp + the golden; do NOT edit, do NOT touch the box).', STD, '',
  'Specify with file:line + signatures: (1) the cuSOLVER handle + search-stream pool to ADD to PerGpuResources/handles (OQ-10) — RAII, per-device; (2) each CUDA-backend qpAdm virtual impl: the f4-gather kernel (build X[nl x nr x n_block] from the resident DeviceF2Blocks f2 tensor per block — the index gather), the on-device jackknife (est_to_loo + xtau + Q via cuBLAS SYRK + fudge + cuSOLVER invert), the rank fit (cuSOLVER gesvd on the small X + the ALS opt_A/opt_B small-GEMM loop), the GLS weight solve (cuSOLVER potrf+solve), and the BATCHED SE (the ~708 LOO weight re-fits as batched on-device solves, NOT a host loop); (3) the precision map (§12: EmulatedFp64 SYRK/f4 where well-conditioned, native FP64 SVD/Cholesky/GLS); (4) how the data stays resident (DeviceF2Blocks in VRAM; the model gather on-device; no host round-trip); (5) the parity-test extension to run the CUDA backend (assert GPU path + f2 resident) and match the af6a8c2 golden (reuse the CpuBackend ONLY as a diff oracle for localization). The implementer makes NO design decisions. GPU-first, batched-capable, production — the CpuBackend is not the deliverable.',
].join('\n'), { label: 'design:qpadm-gpu', phase: 'Design' })

phase('Implement')
const impl = await agent([
  'You are a senior CUDA engineer. Implement the qpAdm fit ON THE GPU (CUDA backend) per the FROZEN design and dev-loop on box5090 until the qpAdm parity test passes on the GPU path vs the af6a8c2 golden. EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(missing — derive from backend.hpp qpAdm virtuals + the CpuBackend reference; implement on CUDA via cuSOLVER/cuBLAS/kernels, resident, batched SE)') + '\n>>>', '', DEVLOOP, '',
  'Build the CUDA-backend qpAdm: the f4-gather + jackknife + SVD/ALS + GLS + batched-SE kernels/cuSOLVER calls + the cuSOLVER/search-stream additions to PerGpuResources, all reading DeviceF2Blocks RESIDENT. Build Release + ctest, run the qpAdm parity test on the GPU path. Iterate until: build clean (warnings-as-errors), ctest green, AND the qpAdm parity test passes ON THE CUDA BACKEND (GPU) vs the golden — weights 0.5589/0.4411 (rtol~1e-6), X/Q cross-checked, p 0.32682 — with the fit RUNNING ON THE GPU (f2 resident, no host round-trip). Report: git diff --stat; files added/changed; build/ctest; the qpAdm parity output (GPU path, steppe-vs-golden); confirmation it ran on the GPU (cuSOLVER/kernels, resident). Do NOT commit. If a quantity diverges on the GPU path, report which + by how much (use the CpuBackend bit-exact result to localize) — do NOT fabricate.',
].join('\n'), { label: 'impl:qpadm-gpu', phase: 'Implement' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','runs_on_gpu','resident_no_roundtrip','weights_match','X_match','Q_match','p_match','gpu_weights','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real diff implementing the CUDA-backend qpAdm fit (cuSOLVER+cuBLAS+kernels, cuSOLVER added to PerGpuResources) + Release build clean + ctest green + the qpAdm parity test passes ON THE GPU (CUDA backend) vs the af6a8c2 golden (weights rtol~1e-6, X/Q, p) + the fit genuinely runs on the GPU reading f2 resident (NOT the CpuBackend)' },
    diff_is_real: { type: 'boolean', description: 'you re-ran git diff: the CUDA-backend qpAdm impl (kernels + cuSOLVER) is genuinely present, not a stub or a CpuBackend passthrough' },
    runs_on_gpu: { type: 'boolean', description: 'the parity test exercises the CUDA backend / GPU path (you confirmed it ran GPU kernels/cuSOLVER, not the CpuBackend)' },
    resident_no_roundtrip: { type: 'boolean', description: 'the fit reads f2_blocks RESIDENT in VRAM (DeviceF2Blocks); no per-model host round-trip of f2' },
    weights_match: { type: 'boolean', description: 'GPU weights == golden 0.5589/0.4411 within rtol~1e-6' },
    X_match: { type: 'boolean' }, Q_match: { type: 'boolean' }, p_match: { type: 'boolean' },
    gpu_weights: { type: 'string', description: 'the YOU-measured GPU-path weights/se/z/p vs the golden' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'GPU-path steppe-vs-golden per quantity + confirmation it ran on the GPU resident; for FAIL exactly what diverged or where it fell back to CPU' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (maximally adversarial — the user is emphatic this must be the REAL GPU implementation, not a CPU reference; prior fix-passes have SHAMMED). The implementer reported:\n<<<\n' + (impl || '(impl died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM non-empty AND genuinely implements the CUDA-backend qpAdm fit (real kernels + cuSOLVER calls; cuSOLVER added to PerGpuResources/handles) reading DeviceF2Blocks resident — NOT a stub, NOT a CpuBackend passthrough, NOT CPU-first scaffolding presented as GPU. If it secretly runs on the CpuBackend => pass=false. (2) RE-RUN yourself: ' + BUILD + ' ; ' + QPTEST + ' . (3) PASS only if ALL: diff real (GPU kernels/cuSOLVER); Release build clean; ctest green; the qpAdm parity test passes ON THE CUDA BACKEND (GPU) vs the golden (weights rtol~1e-6 to 0.5589/0.4411, X/Q cross-checked, p 0.32682); AND you confirm the fit actually ran on the GPU reading f2 RESIDENT (inspect the test/code — it must dispatch to the CUDA backend, not CpuBackend; spot-check for GPU activity if feasible). \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-new/changed source+test files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (the qpAdm fit on the GPU + the measured vs-golden parity) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly what diverged or where it fell back to CPU.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:qpadm-gpu', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ qpAdm FIT ON GPU LANDED ' + verdict.commit_hash + ' — runs_on_gpu=' + verdict.runs_on_gpu + ' resident=' + verdict.resident_no_roundtrip + ' weights=' + verdict.weights_match + ': ' + verdict.gpu_weights)
else log('--- qpAdm GPU FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, impl, verdict }
