export const meta = {
  name: 'parallelize-large-loo-se',
  description: 'Kill the obscene test time: PARALLELIZE the large-model (nr>10) jackknife LOO SE, which currently runs SINGLE-THREADED (~701 leave-one-block-out refits looped on ONE CUDA thread, each a heavy ~39-dim Jacobi SVD) — this is why qpadm_parity (the NRBIG nr=39 model) takes ~371 s and every ctest is ~13 min. The small-model path ALREADY parallelizes the LOO across B*nb threads; the large path does NOT (the many-thread LOO kernel is hard-capped to the small bound nl<=5/nr<=10, so nr>10 models fall back to the single-thread large path). The ~701 LOO refits are INDEPENDENT (embarrassingly parallel). FIX: give the large path a parallel LOO — one thread (or block) per (model,block) refit using a RUNTIME-SIZED device workspace (mirroring the cap-lift dynamic-VRAM-scratch approach), so the refits run concurrently. PARITY: the SE must be BIT-IDENTICAL to today (same per-refit math + the same deterministic fixed-order variance reduction — only the execution parallelism changes). GATE: qpadm_parity drops from ~371 s to fast (target minutes->seconds); the NRBIG se/z is bit-identical; ALL goldens still pass; build clean + ctest green. Single-GPU; GPU is the deliverable. Design -> GPU implement -> INDEPENDENT verdict. commit-green / revert+HALT.',
  phases: [
    { title: 'Design', detail: 'freeze HOW to parallelize the large LOO (per-(model,block) thread/block + dynamic device workspace, bit-identical SE); read-only' },
    { title: 'GPU', detail: 'implement the parallel large-LOO SE on box5090, dev-loop on a FAST targeted test (qpadm_parity, single-GPU); to green + the speedup' },
    { title: 'Verify', detail: 'independent verdict: NRBIG se/z bit-identical, qpadm_parity time slashed (measured), ALL goldens green, build/ctest green; commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === FULL CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -50'"
// FAST dev-loop test: qpadm_parity ALONE (the NRBIG SE is here), single-GPU (CUDA_VISIBLE_DEVICES=0 => the rotation G2 case auto-skips; we are not iterating on the rotation). Time it.
const FASTTEST = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && export CUDA_VISIBLE_DEVICES=0 && ulimit -c 0 && /usr/bin/time -v ctest --test-dir build-rel -R qpadm_parity --output-on-failure 2>&1 | tail -40'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120, RTX 5090) reimplementation of ADMIXTOOLS 2. Branch phase2-fit-engine (just after the jackknife-se-policy commit — pull the latest HEAD). The qpAdm fit + rotation are BUILT; SINGLE-GPU is the supported path (multi-GPU deferred).',
  'THE PROBLEM (root cause of ~13-min ctest / ~371 s qpadm_parity): the LARGE-model (nr>10) jackknife LOO SE runs SINGLE-THREADED. qpadm_fit_kernels.cu has a many-thread batched LOO kernel (one thread per (model,block) = B*nb threads) BUT it is hard-capped to the SMALL per-thread bound (nl<=5, nr<=10) because it uses fixed-size per-thread local arrays (the code comment: "the many-thread LOO batched kernel uses the SMALL bound ... the single-thread sweep can use a big bound while the batched LOO [stays small]"). So a model with nr>10 (e.g. the NRBIG fixture nl=2, nr=39, n_block=701) cannot use the parallel LOO and falls back to the single-thread LARGE path: ~701 leave-one-block-out refits run SEQUENTIALLY on ONE thread, each a ~39-dim Jacobi SVD + ALS + weight solve. That serial loop is the ~371 s.',
  'THE FIX: the ~701 (and B*nb) LOO refits are INDEPENDENT => parallelize the LARGE path LOO too. Mirror the cap-lift pattern (gpu-large-models: the single-model large fit moved its working arrays from fixed per-thread local to a RUNTIME-SIZED device workspace, large_dbl_scratch/large_int_scratch). Do the same for the LOO: a many-thread (or block-per-refit) large-LOO kernel where each (model,block) refit uses a slice of a runtime-sized device-workspace arena (sized B*nb * per-refit-scratch from nl/nr/m/r), so the refits run CONCURRENTLY instead of one-thread-serial. Occupancy will be limited by the large per-thread scratch, but ~701 parallel (even low-occupancy) >> 701 serial.',
  'PARITY (NON-NEGOTIABLE, §12): the SE must be BIT-IDENTICAL to today. The LOO refits are independent so their order does not affect each refit; keep each per-refit math identical to the current single-thread large path, and keep the FINAL variance reduction (qpadm_se_from_wmat-style) in its existing DETERMINISTIC fixed op order (no atomics) so the SE is bit-identical and G1==G2 stays bit-identical. The NRBIG golden (golden_fit1) + 9-pop (golden_fit0) + the rotation golden (golden_rot) MUST all still pass with IDENTICAL se/z. ONLY the parallelism changes, never the numbers.',
  'GATE: qpadm_parity (the NRBIG single-model SE) runs in FAST time (target minutes->seconds; report the before ~371 s vs after) WITH the NRBIG se/z BIT-IDENTICAL to the pre-fix values; ALL goldens pass (golden_fit0/golden_fit1/golden_rot); build clean (warnings-as-errors) + FULL ctest green; G1==G2 determinism intact. Single-GPU. NO loosened tiers, NO changed SE numbers.',
  'BOX = box5090 (vast 2x RTX 5090, UP, sm_120, CUDA 13). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists. RELEASE only. NOTHING builds locally. DEV-LOOP FAST: iterate on qpadm_parity ALONE with CUDA_VISIBLE_DEVICES=0 (skip the rotation/G2 during iteration); run the FULL ctest only to confirm green at the end. To iterate even faster while wiring the kernel, you MAY temporarily test a reduced-block NRBIG-like case for functional correctness, then confirm timing + parity on the REAL full NRBIG fixture before declaring done (no synthetic accuracy claims — the parity check is the REAL NRBIG golden).',
  'KEY FILES: src/device/cuda/qpadm_fit_kernels.cu/.cuh (the LOO/SE kernels — the many-thread small-bound LOO + the single-thread large path; ADD the parallel large-LOO using dynamic device workspace), src/device/cuda/cuda_backend.cu (the large-model fit path / where the single-thread large LOO is dispatched + the workspace sizing), tests/reference/test_qpadm_parity.cu (the NRBIG case — may add a timing print), docs/design/fit-engine.md (note the large-LOO is now parallel).',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); FAST iterate (' + FASTTEST + ' — qpadm_parity only, single-GPU, timed); FULL build+ctest (' + BUILD + ') to confirm green. Iterate until qpadm_parity is fast + se/z bit-identical + full ctest green. Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA architect. Freeze HOW to parallelize the large-model jackknife LOO SE (currently single-thread ~701 serial refits) so it runs concurrently, bit-identically. READ-ONLY (read qpadm_fit_kernels.cu/.cuh — the small-bound many-thread LOO kernel + the single-thread large path + the dynamic-workspace *_large kernels + the kQpMax* / MAXM/MAXT bounds + the variance-reduction kernel; cuda_backend.cu the large-fit dispatch + workspace sizing; test_qpadm_parity.cu the NRBIG case; do NOT edit, do NOT touch the box).', STD, '',
  'Specify with file:line + signatures: (1) the parallel large-LOO kernel — per-(model,block) thread (B*nb) OR block-per-refit, using a RUNTIME-SIZED device-workspace arena (sized from nl/nr/m/r * B*nb) for each refit\'s Jacobi-SVD + ALS + weight scratch (NOT fixed per-thread local arrays); how it reuses the existing per-refit math so the SE is bit-identical; (2) where the workspace is allocated + sized (extend large_dbl_scratch/the existing arena) + the VRAM budget; (3) how the dispatch chooses parallel-large-LOO for nr>small-bound vs the existing small-bound many-thread LOO for small models (keep the small path UNCHANGED — it is already parallel + golden); (4) the determinism plan (independent refits; the final variance reduction keeps its fixed op order, no atomics => bit-identical SE, G1==G2 intact); (5) the validation: NRBIG se/z bit-identical + the qpadm_parity timing drop. The implementer makes NO design decisions. GPU-first, single-GPU; bit-identical SE is the hard constraint.',
].join('\n'), { label: 'design:large-loo', phase: 'Design' })

phase('GPU')
const gpu = await agent([
  'You are a senior CUDA engineer. Implement the PARALLEL large-model LOO SE per the frozen design so qpadm_parity (NRBIG) goes from ~371 s to fast, with BIT-IDENTICAL se/z. EDIT locally + dev-loop on box5090 (FAST: qpadm_parity only, CUDA_VISIBLE_DEVICES=0); do NOT commit. The SE numbers must NOT change.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(missing — add a many-thread/block-per-refit large-LOO kernel using a runtime-sized device-workspace arena; dispatch it for nr>small-bound; keep the small-path LOO unchanged; bit-identical SE; deterministic fixed-order variance reduction)') + '\n>>>', '', DEVLOOP, '',
  'Build the parallel large-LOO SE. Iterate FAST on qpadm_parity (single-GPU, timed) until it is fast AND the NRBIG se/z is BIT-IDENTICAL to the pre-fix values (the golden_fit1 + the CpuBackend oracle); then run the FULL ctest to confirm ALL goldens green + G1==G2. Report: git diff --stat; the parallel-LOO kernel + workspace code; the BEFORE (~371 s) vs AFTER qpadm_parity time (measured); proof the NRBIG se/z is bit-identical (per-quantity |delta|); full ctest green; G1==G2 intact. Do NOT commit. Do NOT change the SE math/numbers; do NOT loosen tiers; do NOT fake the timing.',
].join('\n'), { label: 'impl:large-loo', phase: 'GPU' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','large_loo_parallel','se_bit_identical','qpadm_parity_fast','before_after_time','all_goldens_pass','determinism_intact','build_green','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: a real diff parallelizing the large-model LOO SE (many-thread/block-per-refit + runtime-sized device workspace, NOT the single-thread loop) + qpadm_parity (NRBIG) time SLASHED (you measured before~371s vs after) + the NRBIG se/z BIT-IDENTICAL to pre-fix + ALL goldens pass (golden_fit0/golden_fit1/golden_rot) + G1==G2 bit-identical + Release build clean + FULL ctest green. NO changed SE numbers, NO loosened tier.' },
    diff_is_real: { type: 'boolean', description: 'you re-read the diff: a genuine parallel large-LOO kernel (per-(model,block) refit on dynamic workspace), not a stub, not still the single-thread loop' },
    large_loo_parallel: { type: 'boolean', description: 'the large-model (nr>small-bound) LOO refits now run CONCURRENTLY (B*nb or block-per-refit), not serially on one thread' },
    se_bit_identical: { type: 'boolean', description: 'the NRBIG se/z (and all SE) are BIT-IDENTICAL to the pre-fix values — the parallelization changed only execution, not numbers (you verified)' },
    qpadm_parity_fast: { type: 'boolean', description: 'qpadm_parity now runs fast (you re-ran + timed it)' },
    before_after_time: { type: 'string', description: 'the YOU-measured qpadm_parity wall time before (~371s) vs after the fix' },
    all_goldens_pass: { type: 'boolean', description: 'golden_fit0 (9-pop) + golden_fit1 (NRBIG) + golden_rot (rotation) all pass (you re-ran)' },
    determinism_intact: { type: 'boolean', description: 'G1==G2 bit-identical still holds (you verified)' },
    build_green: { type: 'boolean', description: 'Release build clean (warnings-as-errors) + FULL ctest green (you re-ran)' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed + the before/after time + bit-identical-SE confirmation; for FAIL exactly what broke or where the SE numbers shifted' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (adversarial — the fix must SPEED UP without changing ANY SE number; the user is fed up with the obscene test time). The implementer reported:\n<<<\n' + (gpu || '(gpu phase died)') + '\n>>>\n\nThe frozen design:\n<<<\n' + (design || '(design died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM a real PARALLEL large-LOO (per-(model,block) refit on a runtime-sized device workspace), NOT the single-thread loop, and that the per-refit MATH + the deterministic variance reduction are UNCHANGED (only parallelism). (2) RE-RUN yourself: ' + BUILD + ' (full ctest) and time qpadm_parity. (3) PASS only if ALL: diff real + parallel; qpadm_parity time SLASHED vs ~371s; the NRBIG se/z BIT-IDENTICAL to pre-fix (spot-check the actual se/z values vs golden_fit1 / the prior values); ALL goldens pass; G1==G2 bit-identical; Release build clean; FULL ctest green. NO changed SE numbers, NO loosened tier. \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-changed source+test+doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (parallelize the large-model LOO SE: ~701 independent refits now concurrent on dynamic device workspace; qpadm_parity <before>->;<after>; SE bit-identical, all goldens pass) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly what broke or where the SE shifted.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:large-loo', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ LARGE-LOO PARALLELIZED ' + verdict.commit_hash + ' — se_bit_identical=' + verdict.se_bit_identical + ' goldens=' + verdict.all_goldens_pass + ' | qpadm_parity ' + verdict.before_after_time)
else log('--- LARGE-LOO FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, gpu, verdict }
