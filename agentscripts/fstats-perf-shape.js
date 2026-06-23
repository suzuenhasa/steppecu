export const meta = {
  name: 'fstats-perf-shape',
  description: 'MEASURE the real wall-clock of the new f-stat features (f4, f3, f4-ratio, qpDstat --f2-dir [Part A], qpDstat --prefix [Part B genotype-path]) on REAL AADR at PRODUCTION scale (NOT the 9-pop golden — that is a parity spot-check), and AUDIT whether each uses PROPER GPU SHAPE (device-resident, batched, no host-side per-item loops). Also place the new numbers next to the existing stages (extract-f2 / qpadm / rotation) from docs/perf/1240k-sweep.md so the answer covers everything. THREE phases: (1) AUDIT the kernels read-only (is the contraction a single batched device launch over ALL quartets/triples, or a loop? is the jackknife on-device-batched or a host per-item loop? is the genotype-path D kernel device-resident SNP-tile-streamed? cite file:line — find ANY CPU-shape bottleneck) + survey the box data (raw TGENO + convertf-PA + any existing scaled f2 caches). (2) MEASURE wall-clock on box5090 (single RTX 5090, Release build-rel) at PRODUCTION scale: build/locate a large 1240K f2 cache (~100-300 pops), time f4/f3/f4-ratio/qpdstat-A over increasing batch sizes (report items/sec + wall-clock + nvidia-smi GPU util), and time qpDstat Part B (genotype path) on 1240K (genotype-read-bound, ~extract-f2 cost). SEQUENTIAL on the one GPU (NO parallel timing — contention corrupts numbers). REAL AADR only, no synthetic. (3) SYNTHESIZE: a wall-clock table (everything + the new features) + a definitive GPU-shape verdict (proper shape yes/no + where, with file:line + measured util evidence) + any underutilization-at-small-scale or CPU-bottleneck findings + recommendations. Write docs/perf/fstats-sweep.md. FAIL-PROTOCOL: read-only measurement, nothing to commit except the perf doc; NEVER git checkout/clean; if the box/data is unavailable, report it (do not fabricate numbers).',
  phases: [ { title: 'Audit GPU shape + survey box data' }, { title: 'Measure wall-clock at scale (sequential, Release)' }, { title: 'Synthesize the perf table + GPU-shape verdict' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const PULL = 'rsync -az -e ssh box5090:/workspace/steppe/docs/perf/ ' + R + '/docs/perf/'
const REL = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -8; " + CORECLEAN + "'"
const PA = '/workspace/data/aadr/converted_pa/v66_HO_pa'
const TGENO = '/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 3da3f53. The NEW standalone f-stats just landed: f4 (src/core/qpadm/f4.cpp run_f4 / assemble_f4_quartets), f3 (f3.cpp run_f3 / assemble_f3_triples), f4-ratio (f4ratio.cpp run_f4ratio / ratio_jackknife), qpDstat Part A (cmd_qpdstat.cpp --f2-dir = run_f4), qpDstat Part B (src/core/stats/dstat.cpp run_dstat + src/device/cuda/dstat_kernel.cu, the genotype-path normalized-D). The fit/contraction kernels are in src/device/cuda/*.cu + the CpuBackend oracle. jackknife_cov (src/device/backend.hpp ~:634).',
  'GOAL: (a) the REAL wall-clock of these new features at PRODUCTION scale on REAL AADR (NOT the 9-pop golden fixture — that is a tiny parity spot-check; production = large P + many quartets/triples, per the design-for-scale rule), and (b) a definitive answer to "are we using PROPER GPU SHAPE?" — device-resident, batched over all items in few launches, NO host-side per-item loops (the CPU-shape anti-pattern). Place the numbers next to the existing stages (extract-f2/qpadm/rotation) already in docs/perf/1240k-sweep.md.',
  'METHODOLOGY (HARD RULES): RELEASE build (build-rel; a debug build voids timing per memory perf-bench-release-build). SINGLE-GPU --device 0 (multi-gpu PARKED). REAL AADR only — NO synthetic data for ANY perf number (memory real-data-only). Timings SEQUENTIAL on the one GPU (parallel runs contend + corrupt numbers). Warm the GPU; report median of >=3 runs; sample GPU util (nvidia-smi dmon / --query-gpu=utilization.gpu during the run) to show the kernel is actually working the GPU. Box ' + SSH + '; nvcc -> ' + PATHENV + '. convertf-PA at ' + PA + '; raw TGENO at ' + TGENO + '.',
  'SHAPE AUDIT = read the actual kernels (file:line): is assemble_f4_quartets ONE batched device launch over ALL quartets (good) or a per-quartet loop (bad)? Is the f4/f3/f4ratio SE jackknife on-device-batched or a HOST per-item loop (the likely CPU-shape risk)? Is run_dstat genotype-path device-resident + SNP-tile-streamed (like extract-f2) or does it bounce per-block to host? Is the D kernel batched over quartets? Name any host-side per-item bottleneck that would cap scaling.',
  'NEVER git checkout/clean. Read-only measurement; the only artifact to write/commit is docs/perf/fstats-sweep.md. If the box or the AADR data is unreachable, REPORT it honestly — do NOT fabricate numbers.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Audit GPU shape + survey box data')
const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['shape_findings','cpu_shape_risks','box_data','existing_perf','plan'],
  properties: {
    shape_findings: { type: 'string', description: 'per feature (f4/f3/f4ratio/qpdstat-A/qpdstat-B): is the contraction/kernel a single batched device launch over all items, device-resident? cite file:line. Is the jackknife on-device or host?' },
    cpu_shape_risks: { type: 'string', description: 'any host-side per-item loop / D2H bounce / serialization that would cap GPU scaling (file:line); or "none found"' },
    box_data: { type: 'string', description: 'what is on box5090: raw TGENO, convertf-PA, any existing 1240K f2 caches (path + P + n_block); what must be (re)built for a production-scale measurement' },
    existing_perf: { type: 'string', description: 'the existing stage numbers from docs/perf/1240k-sweep.md (extract-f2, qpadm, rotation) to place alongside' },
    plan: { type: 'string', description: 'the exact measurement plan: which f2 cache (P, SNPs), the batch-size scale points for f4/f3/f4ratio/qpdstat-A, and the qpdstat-B genotype-path run' },
  },
}
const audit = await tryAgent([
  'You are auditing the GPU shape of the new f-stats + surveying the box for a production-scale perf run (read-only; NO code changes, NO timing yet). READ the kernels: src/core/qpadm/f4.cpp + f3.cpp + f4ratio.cpp (run_f4/run_f3/run_f4ratio + assemble_f4_quartets/assemble_f3_triples + ratio_jackknife — are they batched device launches? where is the jackknife computed — device or host?), src/device/cuda/*.cu (the f2 contraction + the gather + dstat_kernel.cu — device-resident? batched over items?), src/core/stats/dstat.cpp (run_dstat — does the genotype path reuse the extract-f2 decode front-end + SNP-tile stream device-resident, or bounce to host?), backend.hpp jackknife_cov. Determine PROPER-SHAPE vs CPU-SHAPE for each, file:line. Then survey box5090: ' + SSH + " 'ls -la " + PA + "* " + TGENO + "* /workspace/data/aadr/ /workspace/*.f2* /workspace/steppe/*f2* 2>/dev/null | head -40' and find any existing 1240K f2 cache (P, n_block). Read docs/perf/1240k-sweep.md for the existing extract-f2/qpadm/rotation numbers.", STD, '',
  'Return the structured audit: the per-feature shape findings (file:line), any CPU-shape risks, the box data inventory, the existing perf numbers, and the concrete measurement plan (which f2 cache at what P/SNPs, the batch-size scale points). Do NOT measure yet.',
].join('\n'), { schema: AUDIT_SCHEMA, label: 'audit:shape', phase: 'Audit GPU shape + survey box data' })
if (audit === null) { log('--- audit died — HALT'); return { halted: true } }
log('shape audit done; CPU-shape risks: ' + String(audit.cpu_shape_risks).slice(0,160))

phase('Measure wall-clock at scale (sequential, Release)')
const measure = await tryAgent([
  'You are measuring the new f-stats wall-clock at PRODUCTION scale on box5090 (single RTX 5090, Release). Audit + plan:\n<<<\n' + JSON.stringify(audit) + '\n>>>', STD, '', 'DEV: ' + RSYNC + ' then build Release: ' + REL + ' . Confirm bin/steppe built.', '',
  'MEASURE (SEQUENTIAL, single-GPU, REAL AADR, warm + median of >=3, sample GPU util): (1) get a PRODUCTION-scale 1240K f2 cache — use an existing one if present, else extract-f2 a ~100-300 pop set at 1240K (TIME this extract too as a data point). (2) Time f4 (run_f4 / `steppe f4`) over increasing quartet-batch sizes (e.g. 1k / 100k / 1M quartets, or all-quartets over the pop set) — report wall-clock + quartets/sec + GPU util%. (3) Time f3 over triples + f4-ratio over tuples + qpdstat --f2-dir (= f4) similarly. (4) Time qpDstat Part B (`steppe qpdstat --prefix ' + TGENO + ' --pops/--quadruples ...`) at 1240K over a batch of quadruples — report wall-clock (genotype-read-bound) + util + how it compares to extract-f2. Capture nvidia-smi util during the heaviest run to PROVE the GPU is working (or expose underutilization). Report ALL raw numbers (the table), the scaling behavior, and whether GPU util confirms proper shape or reveals a bottleneck. If a scale point OOMs or a CPU bottleneck shows, report it honestly.',
].join('\n'), { label: 'measure:perf', phase: 'Measure wall-clock at scale (sequential, Release)' })
if (measure === null) { log('--- measure died — HALT'); return { halted: true, audit } }

phase('Synthesize the perf table + GPU-shape verdict')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['proper_gpu_shape','wallclock_table','shape_verdict','bottlenecks','recommendations','doc_committed','note'],
  properties: {
    proper_gpu_shape: { type: 'boolean', description: 'true iff ALL the new features are device-resident + batched with no scaling-capping host-side per-item loop (small-scale underutilization of the tiny golden does NOT count against this — judge at production scale)' },
    wallclock_table: { type: 'string', description: 'the wall-clock table: each feature (+ extract-f2/qpadm/rotation) at its scale, wall-clock, throughput, GPU util%' },
    shape_verdict: { type: 'string', description: 'the definitive GPU-shape assessment per feature (proper vs cpu-shaped), with file:line + the measured util evidence' },
    bottlenecks: { type: 'string', description: 'any CPU-shape bottleneck / underutilization / OOM found (or none); the smallest scale where the GPU saturates' },
    recommendations: { type: 'string', description: 'perf follow-ups (if any) — e.g. on-device jackknife, larger batching, genotype-path streaming' },
    doc_committed: { type: 'string', description: 'the commit hash of docs/perf/fstats-sweep.md' },
    note: { type: 'string' },
  },
}
const synth = await tryAgent([
  'You are synthesizing the f-stats perf + GPU-shape report (adversarial — judge proper-shape honestly). Audit:\n<<<\n' + JSON.stringify(audit) + '\n>>>\nMeasurements:\n<<<\n' + measure + '\n>>>', STD, '',
  'WRITE docs/perf/fstats-sweep.md: the wall-clock table (the new features + extract-f2/qpadm/rotation from the existing doc), the per-feature GPU-shape verdict (file:line + measured util), the bottlenecks/underutilization, and recommendations. Be honest: if a feature is CPU-shaped or underutilizes the GPU even at scale, SAY SO. Then cd ' + R + ' && git add ONLY docs/perf/fstats-sweep.md (+ any small doc edit), commit (perf(f-stats): wall-clock at production scale + GPU-shape audit — f4/f3/f4-ratio/qpDstat A+B on real 1240K AADR) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Return the structured synthesis (proper_gpu_shape bool + the table + the verdict + bottlenecks + recommendations).',
].join('\n'), { schema: SYNTH_SCHEMA, label: 'synth:perf', phase: 'Synthesize the perf table + GPU-shape verdict' })
if (synth === null) { log('--- synth died — HALT'); return { halted: true, audit } }
log('PERF: proper_gpu_shape=' + synth.proper_gpu_shape + ' — ' + String(synth.note).slice(0,140))
return { audit, synth }
