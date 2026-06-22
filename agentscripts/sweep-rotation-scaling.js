export const meta = {
  name: 'sweep-rotation-scaling',
  description: 'Measure how the steppe qpadm-rotate rotation scales with POOL SIZE (the model count is combinatorial), on the REAL v66 1240K, single-GPU, and extrapolate to 500 / 1000 / the whole population set. The now-wired qpadm-rotate CLI calls the batched run_qpadm_search (~2866 models/sec measured at the Olalde shape). A pool of N pops with --min-sources 1 --max-sources 3 yields N + C(N,2) + C(N,3) models. Phase 1 MEASURE: build f2 dirs for pools of ~25/50/100/150/200 real pops (from the 1240K .ind, >=5 ind; use the streamed tier as needed) and for EACH run qpadm-rotate (target = a real BA target e.g. England_BellBeaker; the pool; a fixed 6-pop right; --min-sources 1 --max-sources 3) at --jackknife 1 (the realistic feasible-only-SE policy) AND --jackknife 0 on one point -> record model count, wall-clock, models/sec, the extract-f2 build time, peak VRAM. Confirm the per-model rate HOLDS across pool size (shape-bound, not pool-size-bound, like f2 was). STOP raising the pool once a single rotation wall exceeds ~5-8 min (C(200,3)=1.3M models ~= 7min at 2866/s). Phase 2 EXTRAPOLATE+DOC: using the MEASURED rate, project the rotation time for pools of 500 (~20.8M models), 1000 (~167M), and the whole eligible set (count pops with >=5 ind, ~900; ~C(N,3) models) -- the model counts are exact combinatorics, the rate is measured; also give the extract-f2 build time + VRAM/tier at those pool sizes (the f2 dir is P^2*711*16B, streamed). Be honest: a whole-set 1-3-way rotation is a CAPABILITY/stress number (10^7-10^8 models, hours), NOT a typical analysis (real rotations curate pools to dozens); state both the measured curve and the projections. Append to docs/perf/1240k-sweep.md + commit. SINGLE-GPU (--device 0; multi-gpu parked); RELEASE; REAL v66; HALT-on-fail; resumable on 529.',
  phases: [ { title: 'Measure rotation rate vs pool size' }, { title: 'Extrapolate to 500/1000/whole + doc' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const ENSURE_BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -6 && grep -m1 CMAKE_BUILD_TYPE build-rel/CMakeCache.txt'"
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm on box5090 (2x RTX 5090, but SINGLE-GPU only — multi-gpu PARKED). The qpadm-rotate CLI is now wired (main @ 93e6c23) to the batched run_qpadm_search engine (~2866 models/sec measured at the Olalde shape, golden-gated vs golden_rot). CLI: steppe qpadm-rotate --f2-dir DIR --target T --pool p1,p2,.. --right r1,.. --min-sources 1 --max-sources 3 --jackknife 1 --format csv --device 0. Binary ' + BIN + '.',
  'GOAL: measure rotation scaling vs POOL SIZE on REAL v66 1240K (' + PREFIX + ', 1.23M SNPs) and extrapolate to 500/1000/whole-set. A pool of N pops with subsets [1,3] = N + C(N,2) + C(N,3) models (C(N,3) dominates: N=100->161700, 150->551300, 200->1313400, 500->20.7M, 1000->166M). Per-model cost is SHAPE-bound (nr,nl,right-count), NOT pool-size-bound (the f2 sweep showed f2_30==f2_60 models/sec) — so the rate should HOLD as the pool grows and time = model_count / rate.',
  'RELEASE build (build-rel, -DSTEPPE_BUILD_CLI=ON) — debug voids timing (perf-bench-release-build). REAL v66 1240K only (no synthetic). SINGLE-GPU --device 0. nvcc -> ' + PATHENV + '. Clear core dumps. /usr/bin/time may be absent (bash time/date). Build the pool f2 dirs with extract-f2 (streamed tier auto-engages for big P now); record the extract-f2 time too.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Measure rotation rate vs pool size')
const MEAS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','build_type','rate_table','rate_holds','f2_build','peak_vram','notes'],
  properties: {
    done: { type: 'boolean' },
    build_type: { type: 'string', description: 'CMAKE_BUILD_TYPE Release confirmed; single-GPU; qpadm-rotate present' },
    rate_table: { type: 'string', description: 'table: pool N (~25/50/100/150/200) -> #models (N+C(N,2)+C(N,3)), qpadm-rotate WALL (jackknife 1), models/sec; plus one jackknife 0 point; note #feasible' },
    rate_holds: { type: 'string', description: 'does models/sec stay ~constant across pool size (shape-bound)? the converged rate at jk1 and jk0' },
    f2_build: { type: 'string', description: 'extract-f2 build time + tier + f2.bin size at each pool size (the cache cost, separate from the rotation)' },
    peak_vram: { type: 'string', description: 'peak VRAM during the rotation at the largest pool; vs 32GB' },
    notes: { type: 'string', description: 'maxmiss used; the target/right used; any limit; single-GPU/real confirmed' },
  },
}
const meas = await tryAgent([
  'You are a CUDA perf engineer measuring qpadm-rotate scaling vs pool size on REAL v66 1240K, single-GPU. NO code changes. Start: ' + RSYNC + ' then ' + ENSURE_BUILD + ' (CONFIRM Release + that qpadm-rotate is wired, not the scaffold).', STD, '',
  'STEPS: (1) from the .ind build nested real pop sets of ~25, 50, 100, 150, 200 pops (>=5 ind), PLUS a fixed target England_BellBeaker + a fixed 6-pop right (Mbuti,Israel_Natufian,Iran_GanjDareh_N,Han,Papuan,Karitiana). (2) For each pool, extract-f2 (--device 0 --blgsize 0.05 --maxmiss 0.5 --auto-only; streamed tier auto for big P) over {target + pool + right}; record the build time + tier + f2.bin size. (3) Run qpadm-rotate --target England_BellBeaker --pool <the N pool pops> --right <the 6> --min-sources 1 --max-sources 3 --jackknife 1 --device 0, timed -> model count (verify == N+C(N,2)+C(N,3)), wall, models/sec, #feasible; also one --jackknife 0 run to get the raw point-estimate rate; poll nvidia-smi for peak VRAM on the largest. (4) STOP raising N once a single rotation wall exceeds ~5-8 min. Confirm the rate is ~constant across N. rm big f2 dirs + clear cores after. Report the structured measurement. REAL data, single-GPU, Release.',
].join('\n'), { schema: MEAS_SCHEMA, label: 'measure:rotation', phase: 'Measure rotation rate vs pool size' })
if (!meas || !meas.done) { log('HALT: rotation sweep failed — ' + (meas ? meas.notes : 'agent died')); return { halted: true, meas } }
log('rotation rate: ' + String(meas.rate_holds).slice(0,120))

phase('Extrapolate to 500/1000/whole + doc')
const verdict = await tryAgent([
  'You are the INDEPENDENT VERIFIER + doc author for the rotation-scaling sweep. The measure agent reported:\n<<<\n' + JSON.stringify(meas, null, 1) + '\n>>>', STD, '',
  'DO: (1) re-confirm Release + single-GPU + qpadm-rotate wired. (2) spot re-run ONE pool point to confirm models/sec is stable. (3) sanity: model count == N+C(N,2)+C(N,3); rate ~constant across N. (4) count the eligible pops in the v66 1240K .ind (>=5 ind) for the whole-set number.',
  'THEN APPEND to ' + R + '/docs/perf/1240k-sweep.md a section "qpadm-rotate scaling vs pool size (single-GPU)": the MEASURED table (N -> #models -> wall -> models/sec, jk1 + jk0), the converged rate, the f2-build cost, peak VRAM; then the EXTRAPOLATION to pool=500 (~20.8M models), 1000 (~167M), whole-set (the counted N, ~C(N,3) models) using the MEASURED rate -> the rotation time (+ the extract-f2 build time + the f2.bin size / tier at those pool sizes). Be HONEST: these whole-set numbers are a CAPABILITY/stress projection (10^7-10^8 models, hours), not a typical analysis — real rotations curate pools to dozens; state both. Mark REAL v66 1240K / single-GPU / Release / the measured rate. Then cd ' + R + ' && git add ONLY docs/perf/1240k-sweep.md (NEVER git add dot; never aadr/), commit (perf: qpadm-rotate scaling vs pool size on 1240K single-GPU + 500/1000/whole-set projection) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Return the headline: the measured models/sec, and the projected time for 500 / 1000 / whole-set pools, + the commit hash.',
].join('\n'), { label: 'extrapolate:doc', phase: 'Extrapolate to 500/1000/whole + doc' })
if (verdict === null) { log('--- extrapolate/doc died — HALT'); return { halted: true, meas } }
log('+++ rotation scaling doc: ' + String(verdict).slice(0, 240))
return { meas, verdict }
