export const meta = {
  name: 'multigpu-rotation-scaling',
  description: 'Investigate + push the multi-GPU scaling of the S8 qpAdm rotation. Measured at landing: 2520 models = G1 ~4900 / G2 ~6600 models/sec (~1.35x, sub-2x — "fixed per-launch overhead"). PROFILE why it is sub-2x (sweep N over a wide range G1 vs G2; break down replication vs launch overhead vs load imbalance vs actual compute via nsys), then OPTIMIZE toward ~2x where the bottleneck is addressable (e.g. amortize launch overhead, balance the shard, overlap the one-time f2 replication, fewer/bigger batched dispatches), then VERIFY the improved scaling with parity preserved. HONEST: if the ceiling is fundamental at a given N, document the real scaling curve + the crossover N where multi-GPU approaches 2x — do not fabricate a speedup. Deliverable = a real optimization (committed) AND/OR a measured honest scaling characterization (doc). Profile -> Optimize -> INDEPENDENT verdict. commit-green / revert+HALT.',
  phases: [
    { title: 'Profile', detail: 'sweep N (G1 vs G2) + nsys breakdown: replication / launch overhead / imbalance / compute — WHY sub-2x, and does it improve with N?' },
    { title: 'Optimize', detail: 'apply the addressable bottleneck fix(es) to push toward ~2x; parity preserved' },
    { title: 'Verify', detail: 'independent verdict: re-measure the scaling, confirm the improvement (or document the honest ceiling) + parity intact; build/ctest green; commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45'"
const QPTEST = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -90'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120, 2x RTX 5090, 32607 MiB ea) reimplementation of ADMIXTOOLS 2. Branch phase2-fit-engine @ 7852404 (== main). The S8 qpAdm model-space ROTATION just landed: run_qpadm_search(span<QpAdmModel>) fits many models BATCHED (CudaBackend::fit_models_batched: same-shape bucketing -> cublasDgemmStridedBatched + cusolverDnDpotrfBatched/potrsBatched + model-batched kernels) and SHARDS the model list across Resources::gpus (plan_model_shards contiguous, jthread-per-device, one-time f2 replication per device ~22GB, results to pre-sized model_index slots => G1==G2 bit-identical). The rotation test is tests/reference/test_qpadm_rotation.cu (has a synthetic scale-N path).',
  'THE TASK: investigate + push multi-GPU scaling. MEASURED at landing: validated 84-model set noisy; synthetic 2520 models G1 ~4891 / G2 ~6591 models/sec = ~1.35x (sub-2x), attributed to "fixed per-launch overhead when sharding 2520 models". Both GPUs confirmed working. PROFILE the real cause and PUSH toward ~2x where addressable. This is the multi-GPU exploration the user asked for.',
  'PROFILE (be rigorous, measure don\'t guess): (1) sweep N = {256, 512, 1024, 2520, 5000, 10000, 20000, 50000} (synthetic-shape models is fine for the THROUGHPUT measurement — NO accuracy claim from synthetic; accuracy stays the real-AADR goldens) and report G1 vs G2 models/sec + the G2/G1 ratio at each N. Key question: does the ratio -> ~2x as N grows (i.e. overhead amortizes)? If so, find the crossover N. (2) BREAK DOWN the per-run wall (nsys or CUDA events): the one-time f2 replication (cudaMemcpyPeer / re-upload ~22GB/device), per-device launch/cuSOLVER-batched setup overhead, load IMBALANCE (contiguous shard may bucket unevenly by model shape => one device does more batched work), and the actual batched compute. Identify the DOMINANT sub-2x cause. (3) Watch for contention artifacts (an orphaned test process pegging a GPU corrupted a prior measurement — verify nvidia-smi is clean before timing; pin the timing to a quiet box).',
  'OPTIMIZE (only what the profile justifies): candidate fixes — amortize/reduce per-launch overhead (fewer, bigger batched dispatches; reuse cuSOLVER batched workspaces across buckets; persistent scratch), BALANCE the shard by estimated work (bucket models by shape so each device gets equal same-shape batched work, not just equal count), OVERLAP the one-time f2 replication with the first compute (or skip re-replication if f2 already resident per device), and cut redundant H2D/D2H per shard. Keep it GPU-batched + multi-GPU; do NOT regress to a host loop. Parity (the real-AADR goldens + G1==G2 bit-identical determinism) MUST be preserved.',
  'PRECISION/PARITY: unchanged unified policy (batched SYRK/GEMM EmulatedFp64{40}, SVD/Qinv/chi^2 native). The 84-model real-AADR rotation golden + the 9-pop + NRBIG single-model goldens MUST still pass; G1 vs G2 results MUST stay bit-identical + identically ordered (determinism gate).',
  'GATE: a real, measured improvement in G2/G1 scaling (toward ~2x) at production N WITH all goldens still passing and determinism preserved — OR, if the ceiling is fundamental, an HONEST documented scaling characterization (the N-sweep curve + the dominant-cause breakdown + the crossover N) committed to docs. Either way: build clean + full ctest green + parity intact. Do NOT fabricate a speedup; do NOT loosen tiers; do NOT break determinism.',
  'BOX = box5090 (vast 2x RTX 5090, UP, sm_120, CUDA 13, 2 DEVICES). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists. RELEASE only (perf MUST be Release). NOTHING builds locally. Verify nvidia-smi is quiet before any timing. Long timing/nsys runs DETACHED + poll /tmp logs (flaky network).',
  'KEY FILES: src/core/qpadm/model_search.cpp + model_search_core.{hpp,cpp} (the shard plan + fan-out — load balance lives here), src/device/cuda/cuda_backend.cu fit_models_batched (the batched dispatch + the per-bucket overhead + the f2 replication), src/device/resources.hpp (PerGpuResources / device_count), tests/reference/test_qpadm_rotation.cu (the scale-N throughput harness — extend the N sweep + the timing breakdown), docs/research/ (the scaling characterization doc), docs/design/fit-engine.md (the S8 throughput note).',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); the rotation test (' + QPTEST + '). Verify nvidia-smi quiet before timing. Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Profile')
const profile = await agent([
  'You are a senior CUDA performance engineer. PROFILE the multi-GPU scaling of the S8 qpAdm rotation on box5090 — measure WHY it is ~1.35x (sub-2x) at 2520 models and whether it amortizes toward 2x with N. EDIT locally (extend the scale-N harness/timing) + measure on the box; do NOT commit, do NOT optimize yet (profile first).', STD, '', DEVLOOP, '',
  'DO: confirm nvidia-smi is quiet; sweep N={256,512,1024,2520,5000,10000,20000,50000} measuring G1 vs G2 models/sec + the ratio (Release; warm runs; report variance); break down the per-run wall (f2 replication vs per-device launch/cuSOLVER-batched setup vs load imbalance vs batched compute) via CUDA events and/or nsys (DETACHED + poll). Determine: does G2/G1 -> ~2x as N grows (crossover N)? and the DOMINANT sub-2x cause. Report the full N-sweep table + the breakdown + a concrete, prioritized optimization recommendation (what to change, expected gain). Do NOT commit.',
].join('\n'), { label: 'profile:multigpu', phase: 'Profile' })

phase('Optimize')
const opt = await agent([
  'You are a senior CUDA engineer. Apply the OPTIMIZATION the profile justifies to push the S8 rotation multi-GPU scaling toward ~2x at production N — parity + determinism preserved. EDIT locally + dev-loop on box5090; do NOT commit. If the profile shows the ceiling is fundamental at small N but amortizes at large N (no code fix needed), instead make the change be an HONEST scaling-characterization doc + any cheap win, and SAY so.', STD, '',
  'THE PROFILE:\n<<<\n' + (profile || '(profile died — re-measure the N-sweep G1/G2 + the breakdown, then optimize the dominant addressable cause)') + '\n>>>', '', DEVLOOP, '',
  'Apply the justified fix (e.g. shape-balanced shard, reduced/amortized per-launch overhead, reused batched workspaces, overlapped/avoided redundant f2 replication, fewer bigger dispatches). Re-measure G1 vs G2 across N. Iterate until: build clean (warnings-as-errors), full ctest green, the real-AADR rotation golden + 9-pop + NRBIG still pass, G1==G2 still bit-identical, AND the G2/G1 ratio is measurably improved at production N (or, if fundamental, a clear honest characterization is written). Report: git diff --stat; the change; the before/after N-sweep G1/G2 throughput; parity + determinism confirmation. Do NOT commit. Do NOT fabricate a speedup or break determinism.',
].join('\n'), { label: 'optimize:multigpu', phase: 'Optimize' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','scaling_improved_or_characterized','parity_preserved','determinism_preserved','build_green','scaling_numbers','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: a real diff (an optimization that measurably improves G2/G1 scaling at production N, OR an honest documented scaling characterization + any cheap win) + Release build clean + full ctest green + ALL goldens still pass (84-model real-AADR rotation + 9-pop + NRBIG) + G1==G2 still bit-identical/ordered + the multi-GPU throughput numbers are real (measured by you, nvidia-smi quiet). NO fabricated speedup, NO loosened tier, NO broken determinism.' },
    diff_is_real: { type: 'boolean', description: 'you re-read the diff: a genuine optimization or a genuine measured-characterization doc (not a stub, not a fudged number)' },
    scaling_improved_or_characterized: { type: 'boolean', description: 'either G2/G1 measurably improved at production N, OR the honest scaling curve + dominant-cause + crossover N is documented (you re-measured)' },
    parity_preserved: { type: 'boolean', description: 'the real-AADR rotation golden + 9-pop + NRBIG goldens still pass (you re-ran)' },
    determinism_preserved: { type: 'boolean', description: 'G1 vs G2 results bit-identical + identically ordered (you re-verified)' },
    build_green: { type: 'boolean', description: 'Release build clean + full ctest green (you re-ran)' },
    scaling_numbers: { type: 'string', description: 'the YOU-measured before/after (or characterized) G1 vs G2 models/sec across N + the ratio + the dominant sub-2x cause' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed (optimization or characterization) + the real scaling numbers; for FAIL exactly what broke / regressed / where determinism or parity failed' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (adversarial — multi-GPU perf claims have been contaminated before by orphaned GPU processes; verify nvidia-smi is quiet and RE-MEASURE yourself; do not trust the reported numbers). The optimizer reported:\n<<<\n' + (opt || '(optimize died)') + '\n>>>\n\nThe profile:\n<<<\n' + (profile || '(profile died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM a genuine optimization or a genuine measured-characterization (not a fudged number, not a stub, not a regression to a host loop). (2) RE-RUN yourself with nvidia-smi VERIFIED QUIET first: ' + BUILD + ' ; ' + QPTEST + ' ; and re-measure the G1 vs G2 throughput across a couple of N. (3) PASS only if ALL: diff real; Release build clean; full ctest green; ALL goldens pass (84-model rotation + 9-pop + NRBIG); G1==G2 bit-identical/ordered; AND either a measured scaling improvement at production N or an honest documented characterization. NO fabricated speedup, NO broken determinism, NO loosened tier. \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-changed source+test+doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (the multi-GPU rotation scaling: the optimization or the honest characterization + the measured G1/G2 numbers) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly what broke or regressed.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:multigpu', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ MULTIGPU SCALING LANDED ' + verdict.commit_hash + ' — improved/characterized=' + verdict.scaling_improved_or_characterized + ' parity=' + verdict.parity_preserved + ' determinism=' + verdict.determinism_preserved + ' | ' + verdict.scaling_numbers)
else log('--- MULTIGPU SCALING FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { profile, opt, verdict }
