export const meta = {
  name: 'fstats-sweep-cli-walltime',
  description: 'Wire the (already-built, GPU-only, committed eda83c5) f-stat SWEEP into the CLI so it is user-callable, then WALL-CLOCK the full C(500,4)=2.57B-quartet sweep on real AADR f2_500, then update the docs. The GPU sweep core (run_f4_sweep / run_f3_sweep — on-device unrank + assemble_f4_quartets + diagonal jackknife + |z| filter + CUB compaction, PROVEN GPU-bound 84-99% util) EXISTS + is committed; the gap is it has NO CLI surface (steppe f4 --help shows only --pop1..4/--pops). WIRE: add the sweep flags to the f4/f3 (and qpdstat) commands — --all-quartets / --all-triples (sweep over the --f2-dir pop set), --min-z FLOAT (default 3.0), --top-k INT, --sure (override the maxcomb cap), --shard-dir PATH — dispatching to the EXISTING run_f4_sweep/run_f3_sweep. MINE the CLI flag/dispatch plumbing from wip/fstats-massive-overbuild (cli_parse.cpp + cmd_f4/f3.cpp sweep flags + sweep_dispatch.hpp + the config knobs cli_args.hpp/config_builder.cpp/run_config.hpp) BUT point it at the NEW GPU-only run_f4_sweep (NOT the rolled-back CPU host-enumeration). The EXPLICIT-list mode (--pop1..4/--pops) stays byte-identical -> goldens EXACT. Then on box5090: `time ./bin/steppe f4 --all-quartets --f2-dir /workspace/data/f2_500 --min-z 6 --sure --shard-dir /tmp/sweep_out` (single-GPU, real AADR) while sampling nvidia-smi -> report the WALL-CLOCK + quartets/sec + the GPU util (reconfirm GPU-bound). Then UPDATE docs (docs/perf/fstats-sweep.md + TODO + REORIENT) with the wall-clock + GPU-bound numbers + the CLI usage. NO golden regen; the existing f4/f3/f4ratio/qpdstat/dstat_geno goldens stay EXACT. SINGLE-GPU (multi-GPU PARKED). REAL AADR, no synthetic. FAIL-PROTOCOL: NEVER git checkout/clean; on failure git stash push -u + HALT.',
  phases: [ { title: 'Wire the sweep CLI + build + golden-gate + WALL-CLOCK the 2.57B sweep' }, { title: 'Verify + docs + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -20 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -25; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'
const F2_500 = '/workspace/data/f2_500'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimpl of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ eda83c5. The GPU-ONLY f-stat sweep is BUILT + COMMITTED (run_f4_sweep / run_f3_sweep in src/core/qpadm/... + steppe/fstat_sweep.hpp; on-device unrank + assemble_f4_quartets + diagonal jackknife + |z| filter + CUB DeviceSelect compaction; PROVEN GPU-bound 84-99% util on a real 2.57B-quartet run; tests/reference/test_fstat_sweep_parity.cu gates it). The GAP: it has NO CLI surface — steppe f4 --help shows only --pop1..4/--pops (explicit list). This task WIRES the CLI + WALL-CLOCKS it + updates docs. Do NOT change the sweep COMPUTE (it is done + GPU-bound); only add the CLI plumbing.',
  'WIRE: add sweep flags to cmd_f4.cpp / cmd_f3.cpp (and cmd_qpdstat.cpp if trivial): --all-quartets / --all-triples (enumerate C(P,k) over the --f2-dir pop set), --min-z FLOAT (default 3.0), --top-k INT, --sure (override the maxcomb cap), --shard-dir PATH. Build a SweepRequest from the flags + dispatch to the EXISTING run_f4_sweep/run_f3_sweep. MINE the flag-registration + dispatch shape from branch wip/fstats-massive-overbuild (git show wip/fstats-massive-overbuild:src/app/cli_parse.cpp / :src/app/sweep_dispatch.hpp / :src/app/cmd_f4.cpp / :src/core/config/cli_args.hpp etc.) BUT point the dispatch at the NEW GPU-only run_f4_sweep (NOT the rolled-back CPU host-enumeration path). Emit: stream the survivor rows (the sweep already shards / returns survivors) — for the CLI, write the shards (--shard-dir) and/or print a summary (survivor count, the kept rows to stdout/CSV if small).',
  'THE EXPLICIT-list mode (--pop1..4/--pops) MUST stay byte-identical — the f4/f3/f4ratio/qpdstat/dstat_geno goldens re-pass at 1e-6 EXACT, NO regen. If any golden shifts, that is a BUG.',
  'WALL-CLOCK: on box5090, single-GPU --device 0, after a clean build: `time ./build-rel/bin/steppe f4 --all-quartets --f2-dir ' + F2_500 + ' --min-z 6 --sure --shard-dir /tmp/sweep_out` (the full C(500,4)=2,573,031,125-quartet sweep; --min-z 6 keeps the output small but the WALL-CLOCK measures enumerating+computing+filtering ALL 2.57B). Capture the real wall-clock + compute quartets/sec = 2573031125 / seconds. SAMPLE nvidia-smi during the run (--query-gpu=utilization.gpu,power.draw,memory.used in a 1s loop, or dmon) to RECONFIRM GPU-bound. Report the wall-clock, quartets/sec, GPU util, peak VRAM, survivor count. REAL AADR, no synthetic.',
  'DOCS: update docs/perf/fstats-sweep.md (add the CLI usage + the measured wall-clock + quartets/sec + GPU util for the full 2.57B sweep), docs/TODO.md + docs/REORIENT-PROMPT.md (the sweep is now CLI-wired + the perf numbers).',
  'FAIL-PROTOCOL: NEVER git checkout -- . / git clean -fd. On ANY failure ' + STASH + ' "wip:sweep-cli-FAILED-<reason>" + HALT. NON-trivial blocker -> STOP + report.',
  'SINGLE-GPU --device 0 (multi-GPU PARKED). RELEASE build-rel. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Wire the sweep CLI + build + golden-gate + WALL-CLOCK the 2.57B sweep')
const fixer = await tryAgent([
  'You are wiring the (already-built, GPU-only) f-stat sweep into the CLI + wall-clocking it. Do NOT change the sweep compute. Do NOT commit. Do NOT git checkout/clean.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build (' + BUILD + ').', '',
  'STEP 1: read the NEW run_f4_sweep/run_f3_sweep signature (steppe/fstat_sweep.hpp + the .cpp) + SweepRequest; read the cmd_f4.cpp/cmd_f3.cpp current (explicit-list) structure; mine the CLI flag/dispatch plumbing from wip/fstats-massive-overbuild (git show). STEP 2: ADD the sweep flags (--all-quartets/--all-triples, --min-z [default 3.0], --top-k, --sure, --shard-dir) to cmd_f4/cmd_f3 (+ qpdstat if trivial), build a SweepRequest, dispatch to the NEW run_f4_sweep/run_f3_sweep; keep the explicit-list path byte-identical. STEP 3: ' + RSYNC + ' + ' + BUILD + ' — clean build + STEPPE_THOROUGH ctest green + the explicit goldens EXACT. STEP 4: WALL-CLOCK on box5090: `' + SSH + " 'cd /workspace/steppe && " + PATHENV + ' && rm -rf /tmp/sweep_out && (nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader -l 1 > /tmp/sweep_smi.log 2>&1 &) ; SMIPID=$! ; /usr/bin/time -v ./build-rel/bin/steppe f4 --all-quartets --f2-dir ' + F2_500 + " --min-z 6 --sure --shard-dir /tmp/sweep_out 2>/tmp/sweep_time.log ; kill $SMIPID 2>/dev/null; echo === TIME ===; grep -E \"wall clock|Maximum resident\" /tmp/sweep_time.log; echo === SMI util range ===; sort -t, -k1 /tmp/sweep_smi.log | tail -5'\"'\"' . Compute quartets/sec = 2573031125 / wall_seconds. Report the WALL-CLOCK, quartets/sec, GPU util range, peak VRAM/RSS, survivor count, + the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'wire+walltime', phase: 'Wire the sweep CLI + build + golden-gate + WALL-CLOCK the 2.57B sweep' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true } }

phase('Verify + docs + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','cli_wired','wall_clock','quartets_per_sec','gpu_util','goldens_exact','gpu_bound','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the sweep is CLI-callable (steppe f4 --all-quartets works), the explicit goldens are EXACT, the full 2.57B sweep wall-clocked + GPU-bound reconfirmed, ctest green, docs updated, single-GPU' },
    fail_severity: { type: 'string' },
    cli_wired: { type: 'boolean', description: 'steppe f4 --all-quartets (+ --min-z/--top-k/--sure/--shard-dir) works + dispatches to the GPU run_f4_sweep' },
    wall_clock: { type: 'string', description: 'the measured wall-clock of the full C(500,4)=2.57B sweep (real AADR f2_500)' },
    quartets_per_sec: { type: 'string', description: '2573031125 / wall_seconds' },
    gpu_util: { type: 'string', description: 'the nvidia-smi GPU util range during the timed sweep (reconfirm GPU-bound)' },
    goldens_exact: { type: 'boolean' }, gpu_bound: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the wall-clock + throughput + GPU util; the CLI usage; on FAIL the blocker' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the sweep CLI-wiring + wall-clock. The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — only CLI/config plumbing added (no sweep COMPUTE change); explicit-list path untouched. (2) ' + BUILD + ' — ctest green + explicit goldens EXACT (confirm no golden shifted). (3) confirm the wall-clock + GPU util the implementer reported are real (re-run the timed sweep if needed, or trust the captured /tmp/sweep_time.log + sweep_smi.log if consistent). (4) single-GPU, real AADR. PASS only if the CLI works + goldens exact + the wall-clock is captured + GPU-bound reconfirmed. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed CLI/config/doc/test files (NEVER git add dot; never aadr/ atlas_results/ handoff-*.md; NO golden files change), update docs/perf/fstats-sweep.md + TODO + REORIENT with the CLI usage + the measured wall-clock + quartets/sec + GPU util, commit (feat(cli): wire the GPU f-stat sweep into steppe f4/f3 --all-quartets/--all-triples [--min-z/--top-k/--sure/--shard-dir] -> the committed GPU-bound run_f4_sweep; wall-clock <X>s for C(500,4)=2.57B on real f2_500, <Y> quartets/s, GPU <util>%) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:sweep-cli-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:cli', phase: 'Verify + docs + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ sweep CLI ' + verdict.commit_hash + ' — WALL-CLOCK ' + verdict.wall_clock + ' (' + verdict.quartets_per_sec + '), GPU ' + verdict.gpu_util)
else log('--- sweep CLI FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
