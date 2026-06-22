export const meta = {
  name: 'sweep-1240k-fix',
  description: 'Correct the ONE real error in the 1240K perf sweep (docs/perf/1240k-sweep.md @ bc7c742): the rotation throughput was measured WRONG. The sweep looped the CLI qpadm subcommand (~1 model/sec) which is PROCESS-SPAWN bound (re-pays CUDA ctx + f2 load per process) and wrongly concluded "rotation not implemented". The batched rotation ENGINE run_qpadm_search EXISTS (src/core/qpadm/model_search.cpp:253 -> CudaBackend::fit_models_batched: f2 resident in VRAM, cublasDgemmStridedBatched + cuSOLVER potrfBatched/potrsBatched + model-batched fit kernel) and is golden-gated (test_qpadm_rotation.cu vs golden_rot.json, 84 models; the test has a throughput-only large-N timing set). Only the CLI subcommand qpadm-rotate is a scaffold. Measure the REAL in-process batched throughput on a 1240K f2 via run_qpadm_search (prefer the existing test_qpadm_rotation large-N path; else a minimal Release bench TU) -> real models/sec. ALSO: the original extract-f2 numbers were taken on BOTH GPUs (--device defaults to auto=all); MULTI-GPU IS PARKED (memory multi-gpu-parked / m45-*) so re-report extract-f2 SINGLE-GPU (--device 0) as the supported path. SINGLE-GPU ONLY throughout (--device 0); do NOT benchmark multi-GPU (it is parked). RELEASE build (perf-bench-release-build); REAL 1240K data (no synthetic genotypes; model COMBINATIONS of real pops are fine). measure -> verify+correct-doc(+commit). HALT-on-fail; resumable on 529. box5090, ONE RTX 5090.',
  phases: [ { title: 'Re-measure single-GPU (extract-f2 + REAL batched rotation)' }, { title: 'Verify + correct the perf doc' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm on box5090 (2x RTX 5090 sm_120, 32 GiB ea). Correcting the 1240K perf sweep (docs/perf/1240k-sweep.md @ bc7c742).',
  'SINGLE-GPU ONLY: MULTI-GPU IS PARKED (memory multi-gpu-parked, m45-multigpu-slowdown-rootcause, m45-d2h-incidental-fused-fit) — the 2nd GPU is SLOWER (f2_blocks data-bounce, a 2nd ~7GB D2H); root-caused, fix deferred. DO NOT benchmark multi-GPU or compare single-vs-both. Run everything with --device 0 (single GPU). The original sweep accidentally used --device auto (=all GPUs); we re-report SINGLE-GPU as the supported path.',
  'RELEASE build only (build-rel, -DSTEPPE_BUILD_CLI=ON) — debug per-kernel cudaDeviceSynchronize voids timing (perf-bench-release-build). REAL 1240K: ' + PREFIX + '.{geno,snp,ind} (TGENO, 1,233,013 SNPs, 23,089 ind, 6.7GB). nvcc -> ' + PATHENV + '. Clear core dumps. /usr/bin/time may be absent (apt-get install time, or bash time/date math); od -c for magic (xxd absent).',
  'THE REAL ERROR (rotation): the sweep looped the CLI qpadm (~1 model/sec) = PROCESS-SPAWN bound (re-pays CUDA ctx + f2 load per process) and wrongly said "rotation not implemented". The batched ENGINE run_qpadm_search EXISTS (src/core/qpadm/model_search.cpp:253 -> CudaBackend::fit_models_batched; f2 RESIDENT in VRAM, cublasDgemmStridedBatched + cuSOLVER *Batched + model-batched fit kernel) and is golden-gated (test_qpadm_rotation.cu vs golden_rot.json, 84 models; it has a throughput-only large-N timing set). Only the CLI subcommand qpadm-rotate is a scaffold. Measure REAL batched throughput on a 1240K f2, SINGLE-GPU: prefer the existing test_qpadm_rotation large-N path (read the test — how it loads f2 + the large-N set; can it target a 1240K f2 dir / env var?); else add a MINIMAL Release bench TU (tests/reference/bench_rotation_1240k.cu, add to tests/CMakeLists.txt, pattern off bench_f2_multigpu.cu) that loads a 1240K f2 dir (same reader the CLI uses) + builds N real-pop models (1 target + ~12-source pool -> C(pool,2..4) a few hundred models) + times run_qpadm_search batched on ONE GPU. Report real models/sec + per-model compute vs the bogus CLI-loop ~1/sec.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Re-measure single-GPU (extract-f2 + REAL batched rotation)')
const MEAS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','build_type','extract_f2_singlegpu','rotation_real','rotation_vehicle','notes'],
  properties: {
    done: { type: 'boolean' },
    build_type: { type: 'string', description: 'CMAKE_BUILD_TYPE confirmed Release' },
    extract_f2_singlegpu: { type: 'string', description: 'extract-f2 --device 0 (SINGLE GPU) wall-clock on a representative 1240K pop set (e.g. 30 or 60 pops); note vs the original both-GPU number (parked path)' },
    rotation_real: { type: 'string', description: 'REAL batched run_qpadm_search throughput on a 1240K f2, SINGLE GPU: models/sec, #models, per-model compute; vs the bogus CLI-loop ~1/sec' },
    rotation_vehicle: { type: 'string', description: 'how rotation was timed: test_qpadm_rotation large-N path, or a new bench TU (name it); whether code was added' },
    notes: { type: 'string', description: 'anything surprising; limits; confirm SINGLE-GPU throughout (no multi-GPU benchmarking)' },
  },
}
const meas = await tryAgent([
  'You are a CUDA performance engineer correcting the 1240K sweep, SINGLE-GPU only (multi-GPU is PARKED — do not benchmark it). NO product code changes (a bench TU is OK). Start: ' + RSYNC + ' then ' + BUILD + ' (CONFIRM Release).', STD, '',
  'TASK A: re-time extract-f2 on a representative 1240K pop set with --device 0 (SINGLE GPU) — one honest supported-path number (the original sweep used auto=both, the parked slower path). TASK B: measure REAL run_qpadm_search BATCHED throughput on a 1240K f2, SINGLE GPU (--device 0) — read test_qpadm_rotation.cu for the large-N throughput path; use it if it can target 1240K, else add a minimal bench TU + build it. Report models/sec (NOT the bogus CLI-loop ~1/sec) + per-model compute. Clear cores. Return the structured measurement.',
].join('\n'), { schema: MEAS_SCHEMA, label: 'remeasure:1240k', phase: 'Re-measure single-GPU (extract-f2 + REAL batched rotation)' })
if (!meas || !meas.done) { log('HALT: re-measure failed — ' + (meas ? meas.notes : 'agent died')); return { halted: true, meas } }
log('re-measured single-GPU: f2 ' + String(meas.extract_f2_singlegpu).slice(0,80) + ' | rot ' + String(meas.rotation_real).slice(0,80))

phase('Verify + correct the perf doc')
const verdict = await tryAgent([
  'You are the INDEPENDENT VERIFIER correcting the 1240K perf doc (SINGLE-GPU; multi-GPU is PARKED). The re-measure agent reported:\n<<<\n' + JSON.stringify(meas, null, 1) + '\n>>>', STD, '',
  'DO: (1) re-confirm Release build + that everything was SINGLE-GPU (--device 0). (2) sanity: spot re-run the extract-f2 --device 0 once (stable?); spot-confirm the batched rotation models/sec is FAR above the bogus ~1/sec (engine pays CUDA ctx + f2 load ONCE). (3) confirm REAL 1240K f2. (4) if a bench TU was added, confirm it builds.',
  'THEN CORRECT ' + R + '/docs/perf/1240k-sweep.md: (a) ROTATION — replace the wrong "qpadm-rotate not implemented / ~1 model/sec" with: the batched engine run_qpadm_search EXISTS + is golden-gated (only the CLI subcommand is a scaffold), the CLI-loop ~1/sec was process-spawn overhead (a measurement artifact), the REAL single-GPU batched throughput is <models/sec>; (b) DEVICE — note the original extract-f2/qpadm numbers were taken on BOTH GPUs via --device auto, but MULTI-GPU IS PARKED (the 2nd GPU is slower — the f2_blocks data-bounce; memory multi-gpu-parked / m45-*), so report the SINGLE-GPU (--device 0) extract-f2 number as the supported path and state multi-GPU is parked + NOT benchmarked. Keep it honest, Release/real-1240K/single-GPU. Then cd ' + R + ' && git add docs/perf/1240k-sweep.md + any added bench TU (+ tests/CMakeLists.txt) ONLY (NEVER git add dot; never aadr/), commit (perf: correct 1240K sweep — REAL single-GPU batched run_qpadm_search rotation throughput; extract-f2 re-reported single-GPU; multi-GPU parked, not benchmarked) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Return the corrected headline numbers + the hash.',
].join('\n'), { label: 'verify:correct-doc', phase: 'Verify + correct the perf doc' })
if (verdict === null) { log('--- verify/doc died — HALT'); return { halted: true, meas } }
log('+++ corrected 1240K perf doc: ' + String(verdict).slice(0, 220))
return { meas, verdict }
