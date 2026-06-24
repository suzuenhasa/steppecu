export const meta = {
  name: 'feature-smoke-walltime',
  description: 'SMOKE + WALL-CLOCK every steppe feature on REAL AADR — confirm all 12 CLI commands + the Python API RUN, capture each WALL-CLOCK + GPU util, and check the committed golden where one exists. USER-MANDATED: REAL AADR ONLY, ZERO SYNTHETIC (data, throughput, all of it). Runs SEQUENTIAL on the box (one clean-timed run at a time, no GPU contention). Bounded scales (representative real-AADR runs — NOT hours; the full C(500,4)=2.57B sweep`s 177s is already on record from the production pass 2f6a050, so run a REPRESENTATIVE sweep + CITE the full number, do not re-run 177s). COMMANDS (each: run -> exit 0 + sane output, /usr/bin/time wall, nvidia-smi -i 0 -lms 200 util, golden-match where applicable): extract-f2 (genotypes->f2 dir), qpadm (fit, golden_fit0 weights 1e-6), qpadm-rotate (the rotation, ~10660 models), qpwave (rank sweep), f4/f3/f4ratio (explicit, their readf2 goldens), qpdstat (--f2-dir f4-path golden + --prefix genotype-D golden), f4 --all-quartets (a REPRESENTATIVE sweep e.g. C(150-200,4) + cite the full 2.57B/177s), qpfstats (9-pop genotype golden 1e-6), qpgraph (single-graph golden score 80.0674), qpgraph-search (bounded enumeration, the global-best/argmin), dates (PUR/CEU/YRI golden ~9.74). PLUS a PYTHON API smoke: build/confirm the wheel-or-build, import steppe, exercise read_f2 + qpadm + f4 + extract_f2 + qpfstats + qpgraph + dates + qpadm_search on real data (confirm each returns sane + matches the CLI/golden where trivial). FLAG the known inventory gap (qpgraph_search not in the Python facade __all__) — report whether it is callable from Python or CLI-only. SINGLE-GPU --device 0 (multi-GPU PARKED). NO code changes (smoke + measure + doc). Synthesis writes docs/feature-matrix.md (a per-feature table: command | runs Y/N | wall | GPU util | golden-match | Python Y/N) + commits. FAIL-PROTOCOL: a command that does NOT run, or a golden that does NOT match, is a real finding -> report it prominently (do NOT paper over); do NOT HALT the whole pass for one failing command (record it + continue), but if a CORE command (qpadm/f4/qpgraph) is broken, flag it loud. NEVER synthetic fallback for a number.',
  phases: [ { title: 'Build + sequential CLI smoke/wall-clock (real AADR, all 12 commands)' }, { title: 'Python API smoke + synthesize the feature matrix + commit' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ eb17c87. The full tool suite is built + golden-gated (ctest passing). This pass SMOKE-TESTS + WALL-CLOCKS every feature on REAL AADR to confirm they all run + how fast.',
  'USER-MANDATED, NON-NEGOTIABLE (in memory real-data-only-all-results): REAL AADR DATA ONLY, ZERO SYNTHETIC — not data, not throughput, not a fabricated/extrapolated number. Every wall-clock is a measured real-AADR run; every golden-match is vs a committed golden (tests/reference/goldens). A command that cannot run -> report the real error; never substitute synthetic.',
  'REAL DATA: raw v66 TGENO ' + TGENO + ' (steppe decodes TGENO); f2 dirs ' + F2_500 + ' (500 pops) + ' + F2_FIT0 + ' (9-pop fit0); convertf-PA /workspace/data/aadr/converted_pa/v66_HO_pa; the committed goldens tests/reference/goldens/at2 + dates/. The RUN-SHEET docs/RUN-SHEET.md has canonical invocations for extract-f2/qpadm/qpadm-rotate/qpwave/the f-stats+sweep; use --help for qpfstats/qpgraph/qpgraph-search/dates.',
  'MEASURE per command: ran_ok (exit 0 + sane non-empty output), WALL via /usr/bin/time -v (Elapsed), GPU util via nvidia-smi -i 0 --query-gpu=utilization.gpu -lms 200 (NOT dmon — aliases on the 2-GPU box; a 100%-host-core can be a spin-wait), golden-match where a golden exists (the tier), peak VRAM where relevant. BOUNDED scales: representative real-AADR runs (seconds-scale where possible). The full C(500,4)=2.57B sweep is 177s ON RECORD (2f6a050) — run a REPRESENTATIVE sweep (e.g. --pops a 150-200 subset of f2_500, C(150-200,4)) + CITE the full number; do NOT re-run 177s.',
  'SINGLE-GPU --device 0 (multi-GPU PARKED). RELEASE build-rel. Runs SEQUENTIAL on the box (one clean-timed run at a time). NO code changes (smoke + measure + doc).',
  'FAIL-HANDLING: a command that does NOT run, or a golden that does NOT match, is a REAL finding -> record + report prominently (do NOT paper over, do NOT synthetic-fallback). Do NOT abort the whole pass for one failing command (record it + continue to the rest); a broken CORE command (qpadm/f4/qpgraph/extract-f2) is flagged loud.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

const BATCH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['batch','commands','notes'],
  properties: {
    batch: { type: 'string' },
    commands: { type: 'array', description: 'one entry per command tested', items: {
      type: 'object', additionalProperties: false, required: ['command','invocation','ran_ok','wall','gpu_util','golden_match','output_summary'],
      properties: {
        command: { type: 'string' },
        invocation: { type: 'string', description: 'the exact real-AADR invocation run' },
        ran_ok: { type: 'boolean', description: 'exit 0 + sane non-empty output' },
        wall: { type: 'string', description: 'the /usr/bin/time Elapsed wall-clock' },
        gpu_util: { type: 'string', description: 'the fine single-GPU util observation (or N/A + why)' },
        golden_match: { type: 'string', description: 'golden-match result + tier (or "no golden")' },
        output_summary: { type: 'string', description: 'the key result + any error' },
      },
    } },
    notes: { type: 'string' },
  },
}

phase('Build + sequential CLI smoke/wall-clock (real AADR, all 12 commands)')
await tryAgent(['Prep: ' + RSYNC + ' then ' + BUILD + ' — confirm a clean Release build-rel (CLI + Python) on box5090 at main @ eb17c87. Report the build status only.', STD].join('\n'), { label: 'prep:build', phase: 'Build + sequential CLI smoke/wall-clock (real AADR, all 12 commands)' })

const batches = [
  { key: 'core-admixture', p: 'SMOKE + WALL-CLOCK (sequential on the box, real AADR): (1) extract-f2 — a small real pop set (the 9 fit0 pops) ' + TGENO + ' -> a temp f2 dir; (2) qpadm — fit the fit0 9-pop model over ' + F2_FIT0 + ' (or the extracted dir), golden_fit0 weights rtol 1e-6; (3) qpadm-rotate — a real source pool over ' + F2_500 + ' (e.g. target England_BellBeaker, nr=5, a pool giving hundreds-thousands of models), report n_models + models/s; (4) qpwave — a real left set (rank sweep). Per command: invocation, ran_ok, wall (/usr/bin/time), GPU util (-i 0 -lms 200), golden-match, output summary.' },
  { key: 'descriptive-fstats', p: 'SMOKE + WALL-CLOCK (sequential, real AADR): (1) f4 — an explicit golden quartet over the 9-pop f2, golden_fit0_f4_readf2 rtol 1e-6; (2) f3 — an explicit golden triple, its readf2 golden; (3) f4ratio — an explicit 5-tuple, its golden; (4) qpdstat --f2-dir (f4-path, its golden) AND qpdstat --prefix (genotype-path normalized-D, golden_fit0_dstat_geno); (5) the SWEEP: f4 --all-quartets --f2-dir ' + F2_500 + ' --pops <a 150-200-pop subset> --top-k 1000000 --sure --shard-dir /tmp/sw -> report the C(P,4) count + wall + quartets/s + GPU util, and CITE the full C(500,4)=2.57B/177s from the production pass (do NOT re-run 177s). Per command: invocation, ran_ok, wall, GPU util, golden-match, summary.' },
  { key: 'new-tools', p: 'SMOKE + WALL-CLOCK (sequential, real AADR): (1) qpfstats --prefix ' + TGENO + ' --pops <the 9 fit0 pops> --out-dir /tmp/qpfs -> golden_qpfstats_geno rtol 1e-6 (cite the 40-pop 8.1s from the production pass); (2) qpgraph --f2-dir <a steppe STPF2BK1 of the 9-pop> --graph <the golden topology> -> score 80.0674246076 golden; (3) qpgraph-search --f2-dir <the 5-pop set> --pops <the 5 leaves> -> the exhaustive enumeration, report the candidate count + the global-best score + argmin + wall + GPU util; (4) dates --prefix ' + TGENO + ' --target PUR --left CEU,YRI -> date ~9.742 golden (rtol 0.02). Per command: invocation, ran_ok, wall, GPU util, golden-match, summary. Use --help to get exact flags.' },
]
const cli = []
for (const b of batches) {
  const r = await tryAgent(['You are the feature-smoke measurer for the ' + b.key + ' batch (REAL AADR, sequential on the box, golden-anchored). ' + b.p, STD].join('\n'), { schema: BATCH_SCHEMA, label: 'smoke:' + b.key, phase: 'Build + sequential CLI smoke/wall-clock (real AADR, all 12 commands)' })
  cli.push(r)
  if (r) log(b.key + ': ' + (r.commands ? r.commands.filter(c => c.ran_ok).length + '/' + r.commands.length + ' ran' : 'no data'))
}
const okcli = cli.filter(Boolean)

phase('Python API smoke + synthesize the feature matrix + commit')
const pysmoke = await tryAgent([
  'PYTHON API SMOKE (REAL AADR, on box5090). Build/confirm the steppe Python extension (STEPPE_BUILD_PYTHON build, or the wheel), then `import steppe` and exercise on real data: read_f2 (a committed f2 dir), qpadm (the fit0 model -> weights, vs the golden), f4 (an explicit quartet), extract_f2 (a small real pop subset -> handle, vs read_f2), qpfstats, qpgraph, dates, qpadm_search. For EACH: does it import+run+return sane, wall (rough), and does it match the CLI/golden where trivial. ALSO check the inventory gap: is qpgraph_search callable from Python (it is NOT in __all__) — report CLI-only vs callable. Return a structured summary (per-function ran_ok + wall + match) as the batch `python-api` (use the same shape: command=the python fn, invocation=the call, ran_ok, wall, gpu_util=N/A-ok, golden_match, output_summary).', STD,
].join('\n'), { schema: BATCH_SCHEMA, label: 'smoke:python', phase: 'Python API smoke + synthesize the feature matrix + commit' })

const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['committed','all_ran','failures','wall_table','python_ok','qpgraph_search_python','note'],
  properties: {
    committed: { type: 'string', description: 'commit hash + doc path' },
    all_ran: { type: 'boolean', description: 'every CLI command + Python fn ran on real AADR' },
    failures: { type: 'string', description: 'any command/fn that did NOT run or whose golden did NOT match (empty if all green)' },
    wall_table: { type: 'string', description: 'the per-feature wall-clock summary (command | wall | util | golden)' },
    python_ok: { type: 'boolean' },
    qpgraph_search_python: { type: 'string', description: 'is qpgraph_search callable from Python or CLI-only?' },
    note: { type: 'string' },
  },
}
const synth = await tryAgent([
  'Synthesize the feature smoke + wall-clock matrix. CLI batches + Python:\n<<<\n' + JSON.stringify({ cli: okcli, python: pysmoke }) + '\n>>>', STD, '',
  'WRITE docs/feature-matrix.md: a per-feature table (Feature | CLI command | runs Y/N | wall-clock (real AADR) | GPU util | golden-match | Python Y/N), grouped by category (f2 precompute / admixture / descriptive f-stats / smoothing / dating), the representative-sweep number + the cited full-2.57B/177s, and a FLAGS section. Lead with: every number REAL AADR, golden-anchored, no synthetic. PROMINENTLY flag any command that did not run or golden that did not match, and the qpgraph_search-not-in-Python-facade gap. Then cd ' + R + ' && git add ONLY docs/feature-matrix.md, commit (docs(feature-matrix): smoke + wall-clock every CLI command + the Python API on real AADR — runs + walls + golden-match per feature; all real-data) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Return the structured synthesis.',
].join('\n'), { schema: SYNTH_SCHEMA, label: 'synth:matrix', phase: 'Python API smoke + synthesize the feature matrix + commit' })
if (synth === null) { log('--- synth died — HALT'); return { halted: true, cli: okcli, python: pysmoke } }
log('FEATURE MATRIX: ' + synth.committed + ' — all_ran=' + synth.all_ran + ' failures=' + String(synth.failures).slice(0,100))
return { cli: okcli, python: pysmoke, synth }
