export const meta = {
  name: 'fix-group7-device',
  description: 'Pick up the DEFERRED group-7 (Duplication) DEVICE-SIDE items — the parity-load-bearing CUDA kernel copy-paste that the MED/LOW passes risk-budgeted out. Two careful units, each fix->build-repair->golden-gate(+thorough oracle + bit-identical SE/weights + no-perf-regression)->verdict->commit, HALT-on-fail (hot path; do NOT skip-and-continue). D1 block_sink.cu: HostRamSink/DiskSink copy-paste a ring-buffer staging writer (acquire_slot byte-identical, writer-loop skeleton identical, spill_block structurally identical, stop-and-join 4x, event-teardown identical) -> extract a shared StagingRing/CRTP base, ONE definition, both sinks behavior-identical. D2 qpadm_fit_kernels.cu: the small (fixed local-array) vs large (VRAM-scratch) kernel TWINS are line-for-line identical math (dev_opt_A/_large, dev_opt_B/_large, dev_chisq_of/_large, the single-model vs _models f4 gather/loo/xtau, the constrained weight solve x4, the ALS loop x4) -> collapse to ONE __device__ core per concept parameterized on the scratch pointer; the small path passes pointers to its OWN local arrays (PRESERVE register-residency / no perf regression); the SE/weights/chisq MUST stay BIT-IDENTICAL. REAL-AADR goldens only; CUDA-doc verification; capable-path priority + 5090 fallback; core dumps cleared per build.',
  phases: [ { title: 'D1 block_sink StagingRing' }, { title: 'D2 qpadm_fit_kernels twin-collapse' } ],
}

const R = '/home/suzunik/steppe'
const FIND = R + '/docs/cleanup/bigrefactor/findings'
const STDDOC = R + '/docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === DEFAULT CTEST (GPU-vs-golden, REAL AADR) === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
const THOROUGH = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
// qpadm_parity prints the NRBIG SE timing + the per-quantity vs-golden deltas — used to confirm bit-identity + no perf regression
const PARITYVERBOSE = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && ctest --test-dir build-rel -R qpadm_parity -V 2>&1 | grep -iE \"weight|chisq|se|TIME|PASS|FAIL|bit-ident|d\\|\" | tail -50; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) C++20 reimplementation of ADMIXTOOLS 2. Branch phase2-fit-engine == main @ facf73e (Phase A/B/C of the big-refactor all landed). This picks up the DEFERRED group-7 DEVICE-SIDE duplication. Standards: docs/architecture.md (§4 layering [CUDA PRIVATE to steppe_device], §12 precision/PARITY, §13 testing) + the NAMING-STYLE-STANDARD (' + STDDOC + ', esp §3.2 protected parity vocab = RENAME FORBIDDEN).',
  'PARITY IS NON-NEGOTIABLE (§12): these are the parity-load-bearing fit kernels + the streaming spill. The SE / weights / chisq / f4 / rank results MUST stay BIT-IDENTICAL through the refactor — this is pure dedup (one math body instead of two copies), NOT a numeric change. Verify bit-identity via the THOROUGH CpuBackend oracle (qpadm_parity asserts the GPU result == golden_fit0/golden_fit1_NRBIG to ~1e-11 and the oracle bit-checks) + qpadm_rotation (golden_rot) + the G1==G2 determinism gate.',
  'NO PERF REGRESSION: the small-path kernels use FIXED per-thread LOCAL arrays (register/local-memory resident, fast for nl<=5/nr<=10); the large-path uses runtime VRAM scratch (arbitrary size). When you collapse a twin to ONE __device__ core, the small path MUST pass pointers to its OWN local arrays into the core (so it keeps local-array residency) — do NOT force the small path onto VRAM scratch (that would regress the rotation/fit throughput we just built). The shared core takes a scratch POINTER; the caller decides local-vs-VRAM. Confirm qpadm_parity NRBIG SE timing + qpadm_rotation timing do not regress vs HEAD.',
  'REAL DATA ONLY: the gate is the REAL-AADR AT2 goldens golden_fit0/golden_fit1_NRBIG/golden_rot. NO synthetic data, EVER.',
  'RELEASE/NDEBUG PITFALL: gate build is RELEASE (-DNDEBUG, -Werror); STEPPE_ASSERT compiles out -> mark assert-only/extraction-dead params [[maybe_unused]]. A build-repair step also patches trivial -Werror.',
  'DOC-VERIFY: verify any CUDA / cuBLAS / cuSOLVER / C++-stdlib API claim (esp. cudaEvent*/cudaStream*/threading lifetime for block_sink, and __device__ inline/scratch semantics for the kernels) against the CUDA 13.x docs (ToolSearch select:WebSearch,WebFetch) and cite. HARDWARE: box5090 (2x RTX 5090, P2P DISABLED)=fallback; capable-path (PRO6000/P2P/CUDA13+) is priority where relevant (block_sink Disk/HostRam tiers feed the multi-GPU streaming — keep the tiering correct).',
  'BOX = box5090. ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists, RELEASE only. NOTHING builds locally. Core dumps cleared before+after every build.',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD at start (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build (' + BUILD + '); thorough (' + THOROUGH + '); parity-verbose (' + PARITYVERBOSE + '). Do NOT commit (the verdict commits). Do NOT use synthetic data. Iterate until build clean + ALL gates green + bit-identical + no perf regression.'

const ITEMS = [
  { id: 'D1', title: 'D1 block_sink StagingRing', files: 'src/device/cuda/block_sink.cu (+ block_sink.cuh)',
    fix: 'Extract the duplicated ring-buffer staging writer shared by HostRamSink + DiskSink into ONE definition (a StagingRing member or a CRTP base): [7.1][MED] acquire_slot (byte-identical), the writer_loop queue-pop skeleton (cv_work_.wait + stop/empty check + ready_.front/pop — identical; only the per-slot drain action differs: HostRam memcpy vs Disk pwrite), spill_block (structurally identical; only the byte count differs), the writer stop-and-join sequence (copy-pasted 4x across two finish()es + two dtors -> one stop_writer() helper), the event-destroy teardown loop (identical). Fold the [7.x][LOW]: the begin()-side ring setup dup, the (n_block<0?0:n_block) clamp recomputed 6x, the HostRam dst-base computed twice, give HostRam the slab_bytes_ member like Disk. PRESERVE the HIGH fix already in this file (9dbc610: HostRam fail-fast on a non-success event sync via the shared throwing helper) — the StagingRing must keep that fail-fast policy for BOTH tiers. The drain callback differs per tier; everything else is shared. Behavior-identical.',
    gate_extra: 'block_sink feeds the streaming output; f2_multigpu_parity + the f2 streaming tests exercise HostRam+Disk through it. Confirm those pass (the spill path is byte-exact).' },
  { id: 'D2', title: 'D2 qpadm_fit_kernels twin-collapse', files: 'src/device/cuda/qpadm_fit_kernels.cu (+ qpadm_fit_kernels.cuh)',
    fix: 'Collapse the small/large kernel TWINS to ONE __device__ core per concept, parameterized on the scratch POINTER (the small path passes pointers to its OWN local arrays to PRESERVE register-residency; the large path passes VRAM-scratch pointers): [7.1][MED] dev_opt_A/dev_opt_A_large + dev_opt_B/dev_opt_B_large (~84 dup lines, the ALS Kronecker GLS-ridge solve) -> one dev_opt_A/dev_opt_B taking scratch ptrs; dev_chisq_of/_large -> one taking the residual scratch ptr; assemble_f4_gather_kernel vs _models_kernel, f4_loo_total_kernel vs _models_kernel, f4_xtau_kernel vs _models_kernel -> one __device__ core taking a base offset (single-model passes base=0, model-batched passes the per-model slice); the constrained weight solve (appears 4x: dev_als_weights, weights_chisq_kernel, weights_chisq_large_kernel, loo_large_batched_kernel) -> one solve_constrained_weights(A,nl,r,RHS,wv,lu,y,piv,w_out)->bool taking caller scratch; the ALS opt_A->opt_B loop (4x) expressed once. Fold [7.1][LOW] dev_seed_ab GEMM tail vs seed_from_V_kernel -> one seed_ab_from_V; [7.4][LOW] the grid-stride launch-wrapper boilerplate (~11x: total/ceil-div/65535-clamp/launch/CHECK) -> a launch_grid_stride(total,block) helper using the existing kMaxGridDimX. The math + FP op order MUST be bit-identical to the current code (the comments at the twin sites assert they already are); this is mechanical dedup. NOTE the bounds: the small path local arrays are sized by kQpMaxNl/Nr/R (qpadm_bounds.hpp from H2) — keep them; the shared core must work for both the small (fixed) and large (runtime) m/t.',
    gate_extra: 'This is the parity-LOAD-BEARING fit hot path. REQUIRE: qpadm_parity (golden_fit0 9-pop small path + golden_fit1_NRBIG nr=39 large path) bit-identical weights/chisq/se vs HEAD; qpadm_rotation (golden_rot 84-model) green; the STEPPE_THOROUGH CpuBackend-oracle bit-identical; G1==G2 determinism intact; and NRBIG SE timing + rotation timing NOT regressed (the small path kept its local arrays). Run the parity-verbose gate to confirm the per-quantity deltas are unchanged from HEAD.' },
]

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item','pass','diff_real','bit_identical','no_perf_regression','build_clean','goldens_green','thorough_green','no_synthetic','commit_hash','note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if: the diff is a genuine behavior-preserving dedup (one body replacing two copies) + Release build clean + default ctest 39/39 green vs real goldens + STEPPE_THOROUGH qpadm green (oracle BIT-IDENTICAL) + the SE/weights/chisq are BIT-IDENTICAL to HEAD + NO perf regression (small path kept local arrays) + NO synthetic data + NO §3.2 protected-vocab rename' },
    diff_real: { type: 'boolean', description: 'a real dedup: one shared core/ring replacing the duplicated copies, not a stub/no-op' },
    bit_identical: { type: 'boolean', description: 'the SE/weights/chisq/f4/rank results are bit-identical to HEAD (you compared the per-quantity deltas; the oracle still matches to the same tolerance, not worse)' },
    no_perf_regression: { type: 'boolean', description: 'the small path still uses local arrays (not forced onto VRAM scratch); qpadm_parity NRBIG-SE + qpadm_rotation timings not regressed vs HEAD' },
    build_clean: { type: 'boolean' }, goldens_green: { type: 'boolean' }, thorough_green: { type: 'boolean' }, no_synthetic: { type: 'boolean' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed + the bit-identity evidence + the timing evidence; for FAIL exactly what diverged (and whether it is a parity break or a perf regression)' },
  },
}

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  const fixer = [
    'You are a senior CUDA engineer doing a careful DEVICE-SIDE dedup of steppe parity-critical kernels. Do NOT commit (the verdict commits). FIRST clean HEAD: ' + CLEAN + ' .', STD, '', DEVLOOP, '',
    'ITEM ' + it.id + ': ' + it.fix, '', 'PRIMARY FILES: ' + it.files, '', 'EXTRA GATE: ' + it.gate_extra,
    '', 'Read the [7.x] findings for this file in ' + FIND + ' for the exact line cites. Apply the dedup so the math/threading is BEHAVIOR-IDENTICAL (one body replacing the copies). For D2 the small path MUST keep its local arrays (pass pointers into the shared core) — no VRAM-scratch forcing, no perf regression. Verify any CUDA API claim vs the CUDA 13.x docs (cite). Build + run ALL gates (default ctest, thorough, parity-verbose) until clean + green + bit-identical + no regression. Report: every file:line changed; the dedup structure; PROOF of bit-identity (per-quantity weight/chisq/se deltas vs HEAD, the oracle still bit-matching); the timing (NRBIG SE + rotation) vs HEAD; the FULL gate output. Do NOT commit. If ANYTHING diverges in numbers or regresses timing, STOP and report exactly what — do NOT loosen a tier or fake it.',
  ].join('\n')
  const fix = await tryAgent(fixer, { label: 'fix:' + it.id, phase: it.title })
  if (fix === null) { ledger.push({ item: it.id, pass: false, note: 'fixer died' }); log('--- ' + it.id + ' fixer died — HALT'); break }

  // build-repair (trivial -Werror only)
  await tryAgent(['You are the BUILD-REPAIR step for ' + it.id + ' of steppe. The fixer accumulated edits (do NOT clean/revert). Reach a CLEAN Release build, patching ONLY trivial -Werror (unused param/var from the dedup, [[maybe_unused]] for STEPPE_ASSERT-only params). DO: ' + RSYNC + ' then ' + BUILD + ' . If a trivial -Werror fires, fix minimally + rebuild, LOOP up to 4x. Do NOT change dedup logic, do NOT revert. If it fails for a NON-trivial reason (a real type/logic error), STOP + report. Report final build + the trivial patches.', STD].join('\n'), { label: 'repair:' + it.id, phase: it.title })

  const verdict = await tryAgent([
    'You are the INDEPENDENT VERDICT for ' + it.id + ' of steppe (adversarial — this is parity-LOAD-BEARING device code; a dedup that shifts ANY bit fails). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
    'THE DEDUP: ' + it.fix, '', 'EXTRA GATE: ' + it.gate_extra,
    '', 'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — confirm a REAL behavior-preserving dedup (one shared core/ring replacing the duplicated copies), NOT a stub, NOT a numeric change; for D2 confirm the small path STILL uses local arrays (no VRAM-scratch forcing). (2) RE-RUN: ' + BUILD + ' ; ' + THOROUGH + ' ; ' + PARITYVERBOSE + ' . (3) PASS only if ALL: real dedup; Release build clean; default ctest 39/39 green vs real goldens; STEPPE_THOROUGH qpadm green with the oracle BIT-IDENTICAL; the weight/chisq/se per-quantity deltas UNCHANGED vs HEAD (bit-identical, not merely within-tier); NO perf regression; NO synthetic; NO §3.2 rename. ',
    'ON PASS: cd ' + R + ' && git add ONLY this item\'s changed files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (the device-side group-7 dedup + bit-identity + no-regression evidence; CUDA-doc cite if any) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.',
    'ON FAIL: ' + CLEAN + ' (leave repo green) + report exactly what diverged (parity break vs perf regression vs build). Return the structured verdict.',
  ].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:' + it.id, phase: it.title })
  if (verdict === null) { ledger.push({ item: it.id, pass: false, note: 'verdict died' }); log('--- ' + it.id + ' verdict died — HALT'); break }
  ledger.push(verdict)
  if (verdict.pass) log('+++ ' + it.id + ' committed ' + verdict.commit_hash + ' — bit_identical=' + verdict.bit_identical + ' no_perf_regression=' + verdict.no_perf_regression)
  else { log('--- ' + it.id + ' FAILED (' + verdict.note + ') — reverted; HALT (hot-path, no skip-continue)'); break }
}

const passed = ledger.filter(x => x.pass).length
log('group-7 device dedup: ' + passed + '/' + ITEMS.length + ' committed')
return { ledger, passed }
