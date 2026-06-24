export const meta = {
  name: 'production-perf-pass',
  description: 'PRODUCTION-SCALE perf pass on REAL AADR — measure wall-clock / throughput / GPU-util at production scale for the built tools, AND re-confirm each reproduces its already-pulled committed golden (correctness anchor). USER-MANDATED, NON-NEGOTIABLE: REAL AADR DATA ONLY — the raw v66 TGENO, the real f2 dirs (f2_500, f2_fit0_FINAL), the convertf-PA. ZERO SYNTHETIC anywhere — not data, not throughput, not a fabricated comparison. Every perf number is on real AADR; every correctness claim is vs a committed golden in tests/reference/goldens. The host-compute campaign just landed (qpfstats 59->8.1s etc.), so this is the clean-state production picture. MEASURE (fine-grained SINGLE-GPU: nvidia-smi -i 0 --query-gpu=utilization.gpu -lms 200 — NOT dmon -d 1, which ALIASES on the 2-GPU box) for each tool: golden reproduced Y/N (vs the pulled golden, the tier), the production-scale params, wall-clock (/usr/bin/time), GPU util%, throughput (items/sec where applicable), peak host RSS + VRAM. Tools + scales: (1) qpfstats pop-count sweep (9 -> 20 -> 40 pops, real TGENO) — golden = the 9-pop qpfstats-geno (rtol 1e-6); shows the post-fix scaling. (2) the all-quartets f-stat SWEEP on real f2 (f2_500, up to C(500,4)) — golden = the explicit-list f4 over a golden pop set == the f4 golden. (3) the qpAdm S8 ROTATION at scale (many models on real f2) — golden = the fit0 9-pop weights (rtol 1e-6). (4) DATES on the real PUR/CEU/YRI TGENO — golden = the DATES-reference date 9.742 (rtol 0.02). (5) qpGraph single-graph fit (the fleet w/ restarts) on the 9-pop golden topology — golden = the qpgraph score/weights; note the fleet-at-scale is the spike/topology-search envelope. Also a GOLDEN-PARITY ROLL-UP (full STEPPE_THOROUGH ctest = every golden passes = the correctness anchor). SINGLE-GPU --device 0; multi-GPU PARKED. Runs are SEQUENTIAL on the box (one clean-timed run at a time — no concurrent GPU contention). Synthesis writes docs/perf/production-pass.md (real-AADR numbers + golden-parity per tool) + commits. NO code changes (measurement + doc only). FAIL-PROTOCOL: if a golden does NOT reproduce at scale, that is a CORRECTNESS regression -> HALT + report (do NOT paper over it); if a tool cannot run at the stated scale (OOM/limit), report the real limit honestly (no synthetic fallback).',
  phases: [ { title: 'Sequential production-scale measure on real AADR (golden-anchored)' }, { title: 'Synthesize the production-pass perf doc + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -8'"
const TGENO = '/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB'
const F2_500 = '/workspace/data/f2_500'
const F2_FIT0 = '/workspace/data/aadr/f2_fit0_FINAL'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ 3422ba4. The host-compute audit campaign just COMPLETED (everything on-device; qpfstats 40-pop 59s->8.1s). This is a production-scale perf pass to record the clean-state picture on REAL AADR, golden-anchored.',
  'USER-MANDATED, NON-NEGOTIABLE (they have said this repeatedly + it is in memory real-data-only-all-results): REAL AADR DATA ONLY. ZERO SYNTHETIC — not data, not throughput, not a fabricated/extrapolated number. Every perf figure is a measured run on real AADR; every correctness figure is vs a committed golden (tests/reference/goldens). If something cannot run at scale, report the REAL limit (OOM/VRAM/time) honestly — NEVER substitute synthetic or a guess.',
  'REAL DATA: the raw v66 TGENO ' + TGENO + ' (steppe decodes TGENO); the real f2 dirs ' + F2_500 + ' (500 pops) + ' + F2_FIT0 + ' (the 9-pop fit0); the convertf-PA /workspace/data/aadr/converted_pa/v66_HO_pa. The committed goldens in tests/reference/goldens/at2 (+ dates/).',
  'MEASURE with FINE-GRAINED SINGLE-GPU util: nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader -lms 200 (NOT dmon -d 1 — it ALIASES on the 2-GPU box, proven this session). Wall via /usr/bin/time -v (Elapsed + Maximum resident). VRAM via nvidia-smi -i 0 --query-gpu=memory.used. Throughput = items/wall where applicable (combos/s, models/s, quartets/s).',
  'SINGLE-GPU --device 0 (multi-GPU PARKED). RELEASE build-rel. Box ' + SSH + '; nvcc -> ' + PATHENV + '. Runs SEQUENTIAL on the box (one clean-timed run at a time). NO code changes (measure + doc only).',
  'FAIL-PROTOCOL: a golden NOT reproducing at scale = a CORRECTNESS regression -> HALT + report (do NOT paper over). A tool that cannot reach the stated scale -> report the real limit honestly (no synthetic fallback).',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

const MEASURE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['tool','golden_reproduced','golden_detail','scale_points','realdata_only','notes'],
  properties: {
    tool: { type: 'string' },
    golden_reproduced: { type: 'boolean', description: 'the committed golden reproduced at its tier (correctness anchor)' },
    golden_detail: { type: 'string', description: 'which golden + the measured rtol/match' },
    scale_points: { type: 'string', description: 'per scale point: params (real-AADR data + size) | wall | GPU util% | throughput | peak RSS | VRAM — the measured production numbers' },
    realdata_only: { type: 'boolean', description: 'true = every number is on real AADR, no synthetic' },
    notes: { type: 'string', description: 'honest limits hit (OOM/VRAM/time) + the GPU-bound shape observation' },
  },
}

phase('Sequential production-scale measure on real AADR (golden-anchored)')
// build once up front so all measures use the same current binary
await tryAgent(['Prep: ' + RSYNC + ' then ' + BUILD + ' — confirm a clean Release build-rel on box5090 (the current main @ 3422ba4). Report the build status only (no measurement yet).', STD].join('\n'), { label: 'prep:build', phase: 'Sequential production-scale measure on real AADR (golden-anchored)' })

const tools = [
  { key: 'qpfstats-sweep', p: 'MEASURE qpfstats pop-count scaling on REAL AADR TGENO ' + TGENO + '. (a) GOLDEN: confirm the 9-pop qpfstats-geno golden reproduces (rtol 1e-6) — run the cli_qpfstats / the 9 fit0 pops. (b) SCALE: run steppe qpfstats --prefix <TGENO> --pops <list> --out-dir /tmp/... at 9, 20, 40 pops (real pops with good counts from the .ind), each with /usr/bin/time + nvidia-smi -i 0 -lms 200. Report per N: npop, npopcomb, individuals decoded, wall, GPU util%, peak RSS, VRAM. This shows the post-host-campaign scaling (40-pop was ~8.1s after the fixes).' },
  { key: 'fstat-sweep', p: 'MEASURE the all-quartets f-stat SWEEP on REAL f2 ' + F2_500 + '. (a) GOLDEN: confirm an explicit-list f4 over a golden pop set reproduces the f4 golden (rtol). (b) SCALE: steppe f4 --all-quartets --f2-dir ' + F2_500 + ' --top-k 1000000 --sure --shard-dir /tmp/sweep with /usr/bin/time + nvidia-smi -i 0 -lms 200 — report the quartet count (C(500,4)=2.57B), wall, quartets/s, GPU util%, peak RSS, VRAM. (The prior measure was ~177s/14.5M-q-s; re-confirm on the current build.)' },
  { key: 'qpadm-rotation', p: 'MEASURE the qpAdm S8 ROTATION at scale on REAL f2 ' + F2_FIT0 + ' (or ' + F2_500 + ' subset). (a) GOLDEN: confirm the fit0 9-pop qpAdm weights reproduce (rtol 1e-6). (b) SCALE: run a rotation/search over many models (a real left/right pop set giving hundreds-to-thousands of models) via the steppe CLI/search entry, /usr/bin/time + nvidia-smi -i 0 -lms 200 — report n_models, wall, models/s, GPU util%, peak RSS, VRAM. This is the production fit-engine envelope.' },
  { key: 'dates', p: 'MEASURE DATES on REAL AADR. (a) GOLDEN: steppe dates --prefix ' + TGENO + ' --target PUR --left CEU,YRI -> confirm date_gen ~9.742 (rtol 0.02) vs the DATES-reference golden. (b) the run IS the scale measure: /usr/bin/time + nvidia-smi -i 0 -lms 200 — report inds decoded, SNPs, wall, GPU util% (DATES is decode-bound — report honestly), peak RSS, VRAM. Note the decode vs cuFFT split if visible.' },
  { key: 'qpgraph', p: 'MEASURE qpGraph single-graph fit on the REAL 9-pop golden topology. (a) GOLDEN: steppe qpgraph --f2-dir <a steppe STPF2BK1 of the 9-pop> --graph <the golden topology> -> confirm the score/weights vs the qpgraph golden (rtol). (b) the fit wall + GPU util via /usr/bin/time + nvidia-smi -i 0 -lms 200. NOTE the single-graph fit is intrinsically small; the FLEET-at-scale (the topology-search envelope) was measured in the optimizer spike (1M fits, GPU 100%, 24ms, commit 932d108) — cite that as the qpGraph production-scale number, do NOT fabricate a new fleet run here.' },
]
const measures = []
for (const t of tools) {
  const r = await tryAgent(['You are the production-scale measurer for ' + t.key + ' (REAL AADR ONLY, golden-anchored, sequential on the box). ' + t.p, STD].join('\n'), { schema: MEASURE_SCHEMA, label: 'measure:' + t.key, phase: 'Sequential production-scale measure on real AADR (golden-anchored)' })
  measures.push(r)
  if (r) log('measured ' + t.key + ': golden=' + r.golden_reproduced + ' realdata=' + r.realdata_only)
}
const ok = measures.filter(Boolean)
// the golden-parity roll-up: the full ctest = every committed golden passes (the correctness anchor)
const rollup = await tryAgent(['GOLDEN-PARITY ROLL-UP: on box5090 run the FULL STEPPE_THOROUGH ctest (' + SSH + " 'cd /workspace/steppe && " + PATHENV + " && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -8') and report PASS/total — this is the correctness anchor that EVERY committed golden (f4/f3/f4-ratio/qpDstat/qpfstats/qpgraph/dates/qpadm) still reproduces on the current build. Report the count + any failure. REAL AADR (the goldens are real-AADR-derived).", STD].join('\n'), { label: 'measure:golden-rollup', phase: 'Sequential production-scale measure on real AADR (golden-anchored)' })
log('measures: ' + ok.length + '/' + tools.length + '; rollup: ' + String(rollup).slice(0,80))
if (ok.length === 0) { log('--- all measures died — HALT'); return { halted: true } }

phase('Synthesize the production-pass perf doc + commit')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['committed','headline_numbers','all_goldens_held','any_regression','realdata_confirmed','note'],
  properties: {
    committed: { type: 'string', description: 'the commit hash + doc path' },
    headline_numbers: { type: 'string', description: 'the key production numbers per tool (real AADR): qpfstats scaling, the sweep, the rotation, DATES, qpGraph' },
    all_goldens_held: { type: 'boolean', description: 'every committed golden reproduced (the ctest roll-up + per-tool)' },
    any_regression: { type: 'string', description: 'any golden that did NOT reproduce (a correctness regression) — empty if none' },
    realdata_confirmed: { type: 'boolean', description: 'every number is on real AADR, no synthetic' },
    note: { type: 'string' },
  },
}
const synth = await tryAgent([
  'You are synthesizing the production perf pass. The per-tool measures + the golden roll-up:\n<<<\n' + JSON.stringify({ measures: ok, golden_rollup: rollup }) + '\n>>>', STD, '',
  'WRITE docs/perf/production-pass.md: a per-tool table (tool | golden reproduced (which, tier) | production scale (real-AADR data + size) | wall | GPU util% | throughput | RSS | VRAM), the qpfstats post-campaign scaling, the sweep/rotation/DATES/qpGraph numbers, and the golden-parity roll-up (ctest count). Lead with: EVERY number is on REAL AADR, golden-anchored, no synthetic. Flag any correctness regression prominently. Then cd ' + R + ' && git add ONLY docs/perf/production-pass.md, commit (perf(production-pass): real-AADR production-scale numbers + golden parity — post host-compute campaign; qpfstats scaling, the all-quartets sweep, the qpAdm rotation, DATES, qpGraph; every figure real-data + golden-anchored) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. If ANY golden regressed, set any_regression + do NOT bury it. Return the structured synthesis.',
].join('\n'), { schema: SYNTH_SCHEMA, label: 'synth:perf', phase: 'Synthesize the production-pass perf doc + commit' })
if (synth === null) { log('--- synth died — HALT'); return { halted: true, measures: ok } }
log('PRODUCTION PASS: ' + synth.committed + ' — goldens-held=' + synth.all_goldens_held + ' regression=' + String(synth.any_regression).slice(0,80))
return { measures: ok, rollup, synth }
