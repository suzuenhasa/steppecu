export const meta = {
  name: 'sweep-1240k-fix',
  description: 'CORRECT two errors in the 1240K wall-clock sweep (docs/perf/1240k-sweep.md, commit bc7c742): (1) extract-f2 was timed on BOTH GPUs only (--device defaults to auto=all) — re-time --device 0 (single) vs --device 0,1 (both) on the SAME 1240K pop set to see if multi-GPU helps or HURTS (the known m4.5 data-bounce wart where the 2nd GPU added a 7GB D2H and was SLOWER). (2) The rotation throughput was MEASURED WRONG: the sweep looped the CLI `qpadm` subcommand (~1 model/sec) which is PROCESS-SPAWN bound (re-pays CUDA ctx + f2 load per process) and concluded "rotation not implemented". FALSE — the batched rotation ENGINE run_qpadm_search (-> CudaBackend::fit_models_batched: cublasDgemmStridedBatched + cuSOLVER potrfBatched/potrsBatched + model-batched fit kernel, f2 resident, model list sharded across GPUs) EXISTS and is golden-gated (test_qpadm_rotation.cu vs golden_rot.json, 84 models); only the CLI subcommand qpadm-rotate is a scaffold. Measure the REAL in-process batched throughput: run run_qpadm_search over many real-pop models on a 1240K f2 (prefer the EXISTING test_qpadm_rotation large-N throughput-timing path; else a minimal Release bench TU that loads the 1240K f2 dir + builds N real-pop models + times run_qpadm_search), single-GPU vs both-GPU -> real models/sec. Then CORRECT docs/perf/1240k-sweep.md. RELEASE build only (perf-bench-release-build); REAL 1240K data only (no synthetic genotypes; synthetic MODEL combinations of real pops are fine). measure -> verify+correct-doc(+commit). HALT-on-fail; resumable on 529. box5090 2x RTX 5090.',
  phases: [ { title: 'Re-measure (GPU single-vs-both + real batched rotation)' }, { title: 'Verify + correct the perf doc' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -12 && grep -m1 CMAKE_BUILD_TYPE build-rel/CMakeCache.txt; " + CORECLEAN + "'"
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm on box5090 (2x RTX 5090, sm_120, 32 GiB each). This corrects two errors in the 1240K perf sweep (docs/perf/1240k-sweep.md @ bc7c742). RELEASE build only (build-rel, -DSTEPPE_BUILD_CLI=ON) — debug per-kernel cudaDeviceSynchronize voids timing (memory perf-bench-release-build). REAL 1240K data: ' + PREFIX + '.{geno,snp,ind} (TGENO, 1,233,013 SNPs, 23,089 ind, 6.7GB). nvcc -> ' + PATHENV + '. Clear core dumps. /usr/bin/time was absent on the box (apt-get install time, or use bash `time`/date math). od -c for the magic (xxd absent).',
  'ERROR 1 — extract-f2 device: --device defaults to auto = ALL GPUs, so the sweep timed BOTH. Re-time the SAME 1240K pop set with --device 0 (SINGLE GPU) vs --device 0,1 (BOTH) — the m4.5 multi-GPU path has a known data-bounce wart (a 2nd ~7GB D2H of f2_blocks) that made multi-GPU SLOWER (memory m45-multigpu-slowdown-rootcause, m45-d2h-incidental-fused-fit). Determine: is multi-GPU faster, equal, or SLOWER for extract-f2 at 1240K? Use a mid/large pop set (e.g. the 30- or 60-pop set; build it from the .ind, pops with >=5 indiv).',
  'ERROR 2 — rotation throughput: the sweep looped the CLI `qpadm` (~1 model/sec) which re-pays CUDA-ctx + f2-load PER PROCESS (spawn-bound) and wrongly concluded "rotation not implemented". The batched rotation ENGINE run_qpadm_search EXISTS (src/core/qpadm/model_search.cpp:253 -> CudaBackend::fit_models_batched; f2 resident in VRAM, model list sharded across GPUs, cublasDgemmStridedBatched + cuSOLVER *Batched) and is golden-gated (test_qpadm_rotation.cu vs golden_rot.json 84 models; the test even has a throughput-only large-N timing set). Only the CLI subcommand qpadm-rotate is a scaffold. MEASURE THE REAL in-process batched throughput on a 1240K f2: prefer the EXISTING test_qpadm_rotation large-N throughput path (read tests/reference/test_qpadm_rotation.cu — how it loads f2 + the large-N set; can it target a 1240K f2 dir / env var?); if it cannot point at 1240K, write a MINIMAL Release bench TU (tests/reference/bench_rotation_1240k.cu, add to tests/CMakeLists.txt, pattern off bench_f2_multigpu.cu) that loads a 1240K f2 dir (the same f2-dir reader the CLI uses) + builds N real-pop models (e.g. 1 target + a ~12-source pool -> C(pool,2..4) = a few hundred models) + times run_qpadm_search BATCHED. Report real models/sec, single-GPU (1 device) vs both-GPU (2 devices, model-sharded), + the per-model compute. REAL f2 (no synthetic genotypes); the model COMBINATIONS are real pops.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Re-measure (GPU single-vs-both + real batched rotation)')
const MEAS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','build_type','extract_f2_single_vs_both','rotation_real','rotation_vehicle','multigpu_verdict','notes'],
  properties: {
    done: { type: 'boolean' },
    build_type: { type: 'string', description: 'CMAKE_BUILD_TYPE confirmed Release' },
    extract_f2_single_vs_both: { type: 'string', description: 'same 1240K pop set: --device 0 (single) wall vs --device 0,1 (both) wall; which is faster + by how much' },
    rotation_real: { type: 'string', description: 'REAL batched run_qpadm_search throughput on a 1240K f2: models/sec single-GPU and both-GPU, #models, per-model compute; vs the bogus CLI-loop ~1/sec' },
    rotation_vehicle: { type: 'string', description: 'how rotation was timed: test_qpadm_rotation large-N path, or a new bench TU (name it); whether any code was added/committed' },
    multigpu_verdict: { type: 'string', description: 'does multi-GPU HELP or HURT for (a) extract-f2 and (b) the batched rotation? quantify; relate to the m4.5 data-bounce wart' },
    notes: { type: 'string', description: 'anything surprising; OOM/limits; whether the data-bounce wart is present' },
  },
}
const meas = await tryAgent([
  'You are a CUDA performance engineer CORRECTING the 1240K sweep. NO product code changes (a bench TU is OK if needed). Start: ' + RSYNC + ' then ' + BUILD + ' (CONFIRM Release).', STD, '',
  'TASK A: re-time extract-f2 on the SAME 1240K pop set with --device 0 vs --device 0,1 (wrap in time). TASK B: read tests/reference/test_qpadm_rotation.cu to find the batched large-N throughput path; measure REAL run_qpadm_search batched throughput on a 1240K f2 (single-GPU vs both-GPU) — use the existing test large-N path if it can target 1240K, else add a minimal bench TU + build it. Report models/sec (NOT the bogus CLI-loop ~1/sec) + the per-model compute. Clear cores. Return the structured measurement. If you add a bench TU, note it (the verify step will commit it).',
].join('\n'), { schema: MEAS_SCHEMA, label: 'remeasure:1240k', phase: 'Re-measure (GPU single-vs-both + real batched rotation)' })
if (!meas || !meas.done) { log('HALT: re-measure failed — ' + (meas ? meas.notes : 'agent died')); return { halted: true, meas } }
log('re-measured: f2 ' + String(meas.extract_f2_single_vs_both).slice(0,90) + ' | rot ' + String(meas.rotation_real).slice(0,90))

phase('Verify + correct the perf doc')
const verdict = await tryAgent([
  'You are the INDEPENDENT VERIFIER correcting the 1240K perf doc. The re-measure agent reported:\n<<<\n' + JSON.stringify(meas, null, 1) + '\n>>>', STD, '',
  'DO: (1) re-confirm Release build. (2) sanity-check the numbers: spot re-run ONE extract-f2 --device 0 and confirm it differs sensibly from --device 0,1; spot-confirm the batched rotation models/sec is FAR above the bogus ~1/sec (the engine pays CUDA ctx + f2 load ONCE). (3) confirm REAL 1240K f2 (not synthetic genotypes). (4) if a bench TU was added, confirm it builds + is reasonable.',
  'THEN CORRECT ' + R + '/docs/perf/1240k-sweep.md: (a) the rotation section — REPLACE the wrong "qpadm-rotate not implemented / ~1 model/sec" with: the batched engine run_qpadm_search EXISTS + is golden-gated (only the CLI subcommand is a scaffold), the CLI-loop ~1/sec was process-spawn overhead (a measurement artifact), and the REAL batched throughput is <models/sec single + both GPU>; (b) the extract-f2 section — add the --device 0 (single) vs --device 0,1 (both) comparison + the multi-GPU verdict (help/hurt + the data-bounce wart). Keep it honest + clearly marked Release/real-1240K. Then cd ' + R + ' && git add docs/perf/1240k-sweep.md + any added bench TU (+ tests/CMakeLists.txt) ONLY (NEVER git add dot; never aadr/), commit (perf: correct 1240K sweep — extract-f2 single-vs-both GPU + REAL batched run_qpadm_search rotation throughput; the CLI-loop rate was spawn-bound, not the engine) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Return the corrected headline numbers + the hash.',
].join('\n'), { label: 'verify:correct-doc', phase: 'Verify + correct the perf doc' })
if (verdict === null) { log('--- verify/doc died — HALT'); return { halted: true, meas } }
log('+++ corrected 1240K perf doc: ' + String(verdict).slice(0, 220))
return { meas, verdict }
