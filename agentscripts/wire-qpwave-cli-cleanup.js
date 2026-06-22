export const meta = {
  name: 'wire-qpwave-cli-cleanup',
  description: 'M(cli-2): wire the qpwave CLI to the existing run_qpwave engine AND do the CLI consolidation/cleanup in the same pass (no code duplication). qpwave is the ONE remaining CLI scaffold (src/app/cli_parse.cpp:189 run_not_yet_implemented); the engine run_qpwave + golden_qpwave + test_qpwave_parity (engine gate) already exist. NO AT2 runs: golden_qpwave.json (the existing fixture) is the wiring gate. Discipline: design/verify -> implement -> verify(golden_qpwave through the CLI). WIRE (no dup): create cmd_qpwave.{hpp,cpp} mirroring cmd_qpadm.cpp (reuse the f2-dir loader read_f2_dir + PopResolver + the result emitter) — qpwave has NO target, left[0]=reference; call run_qpwave; emit the rank-sweep result (per-rank chisq/dof/p, matching golden_qpwave fields) via the SAME csv/tsv/json emitter helpers (extend result_emit with emit_qpwave reusing fmt/quote/etc — do NOT copy the formatter); route cli_parse qpwave subcommand to run_qpwave_command instead of run_not_yet_implemented. CLEANUP (same pass, behavior-preserving): (1) DELETE the now-dead run_not_yet_implemented (qpwave was its only caller); (2) DEDUP the per-subcommand flag definitions that are IDENTICAL across subcommands into shared helpers like the existing add_common_flags/add_output_flags/add_qpadm_option_flags — at minimum --f2-dir (qpadm/qpwave/rotate, 3x identical), --right (qpadm/qpwave/rotate, 3x identical), --target (qpadm/rotate, 2x identical); --left differs in help text (qpadm=sources vs qpwave=set/ref) so parametrize the help or leave it — keep it BEHAVIOR-PRESERVING (the same flags parse identically). GOLDEN-GATE: a new tests/cli/test_cli_qpwave.cpp (mirror test_cli_qpadm/test_cli_rotate) that runs `steppe qpwave` reproducing golden_qpwave at the existing tier; the existing cli_qpadm/cli_rotate/cli_extract_qpadm + all parity goldens MUST stay green + bit-identical (proves the flag dedup did not regress). design -> implement -> build-repair -> verdict + commit; HALT-on-fail; SINGLE-GPU; REAL data; no synthetic; no AT2.',
  phases: [ { title: 'Design qpwave wiring + CLI dedup (verify)' }, { title: 'Implement + build' }, { title: 'Verify golden_qpwave + no-regression + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm/qpWave. Branch phase2-fit-engine == main @ a8cbd40. qpWave = the rank/cladality test underlying qpAdm: given left + right pops (NO target, left[0]=reference), the minimum rank that relates them. The engine run_qpwave EXISTS + is golden-gated (test_qpwave_parity #22 vs golden_qpwave.json, both backends). The qpwave CLI subcommand is a SCAFFOLD (src/app/cli_parse.cpp:189 run_not_yet_implemented); its flags (--f2-dir --left --right + the shared qpadm-option/output/common flags) already parse.',
  'NO AT2: golden_qpwave.json is the EXISTING gate (no AT2 re-run). qpadm + qpadm-rotate are already wired (cmd_qpadm.cpp, cmd_rotate.cpp) — MIRROR that pattern for qpwave (reuse, do not duplicate).',
  'WIRE (no dup): cmd_qpwave.{hpp,cpp} mirroring cmd_qpadm.cpp — reuse read_f2_dir (the f2-dir loader), PopResolver (name->index), build_resources + upload, call steppe::run_qpwave (left[0]=ref, no target), emit the rank-sweep result via the SAME emitter (extend result_emit with emit_qpwave reusing the csv/tsv/json + fmt/quote helpers; match golden_qpwave fields — the per-rank rankdrop chisq/dof/p). Route cli_parse qpwave -> run_qpwave_command.',
  'CLEANUP (same pass, BEHAVIOR-PRESERVING): delete the now-dead run_not_yet_implemented (qpwave is its last caller); dedup the IDENTICAL per-subcommand flags into shared helpers (matching add_common_flags/add_output_flags/add_qpadm_option_flags) — --f2-dir (3x), --right (3x), --target (2x) are identical; --left help differs (qpadm sources / qpwave ref-set) so parametrize or leave. The flags must parse IDENTICALLY after the refactor (the cli_qpadm/cli_rotate/cli_extract goldens prove it).',
  'GOLDEN-GATE: new tests/cli/test_cli_qpwave.cpp (mirror test_cli_qpadm.cpp / test_cli_rotate.cpp) — `steppe qpwave` reproduces golden_qpwave at the existing tier. The existing cli_qpadm + cli_rotate + cli_extract_qpadm + qpwave_parity + all goldens MUST stay green + bit-identical.',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE -DSTEPPE_BUILD_CLI=ON; nothing builds locally; clear core dumps. NAMING-STYLE-STANDARD + the CLI/output schema (docs/design/cli-bindings.md). Box ' + SSH + '; nvcc -> ' + PATHENV + '; binary ' + BIN + '.',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+thorough-ctest (' + BUILD + '). Do NOT commit (the verdict commits). NO synthetic. SINGLE-GPU. NO AT2.'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Design qpwave wiring + CLI dedup (verify)')
const design = await tryAgent([
  'You are a senior engineer designing the qpwave CLI wiring + the CLI dedup cleanup (verify-before-implement; NO code changes this phase). READ: include/steppe/qpadm.hpp (run_qpwave signature + the qpwave result type/fields), src/app/cmd_qpadm.cpp (the pattern to mirror: f2-dir load -> resolve -> upload -> run -> emit) + cmd_qpadm.hpp, src/app/cmd_rotate.cpp (the other wired exemplar), src/app/result_emit.{hpp,cpp} (the emitter helpers to reuse + how qpadm/rotate emit), tests/reference/goldens/at2/golden_qpwave.json + scripts/golden_qpwave_generate.R (the EXACT left/right/params + the result fields qpwave outputs), tests/reference/test_qpwave_parity.cu (the engine gate) + tests/cli/test_cli_qpadm.cpp (the CLI golden e2e pattern), src/app/cli_parse.cpp (the qpwave scaffold + the flag helpers + the --f2-dir/--target/--right/--left dedup targets).', STD, '',
  'PRODUCE the spec: (1) cmd_qpwave wiring — the exact run_qpwave call (args: f2 dir, left, right, options; left[0]=ref, no target), the emitter (emit_qpwave reusing which helpers, the output fields to match golden_qpwave). (2) the CLI e2e golden test inputs (the golden_qpwave left/right/params) + which fields to compare at which tier. (3) the dedup plan: which flags become shared helpers (--f2-dir/--right/--target identical; --left help-text handling), confirming behavior-preserving. (4) confirm run_not_yet_implemented becomes dead + removable. Cite file:line. Return the spec (do NOT implement).',
].join('\n'), { label: 'design:qpwave', phase: 'Design qpwave wiring + CLI dedup (verify)' })
if (design === null) { log('--- design died — HALT'); return { halted: true } }

phase('Implement + build')
const fixer = await tryAgent([
  'You are wiring the qpwave CLI + the CLI dedup cleanup, per this spec:\n<<<\n' + design + '\n>>>\n\nDo NOT commit. Start clean: ' + CLEAN + '.', STD, '', DEVLOOP, '',
  'IMPLEMENT: cmd_qpwave.{hpp,cpp} (mirror cmd_qpadm, reuse the loader/resolver/emitter — NO duplicated compute/format), emit_qpwave in result_emit (reuse the helpers), route cli_parse qpwave -> run_qpwave_command, DELETE run_not_yet_implemented, dedup the identical flags (--f2-dir/--right/--target, and --left with parametrized help) into shared helpers (behavior-preserving). Add tests/cli/test_cli_qpwave.cpp (mirror test_cli_qpadm) reproducing golden_qpwave. Wire the new files into the CMake. Build + full STEPPE_THOROUGH ctest. SANITY (no commit): run `steppe qpwave --f2-dir <a small f2 dir> --left ... --right ...` -> sensible rank-sweep output; confirm cli_qpadm/cli_rotate still pass (dedup didn not regress). Report every file changed, the no-dup approach, the dedup, and the FULL ctest. Do NOT commit.',
].join('\n'), { label: 'implement:qpwave', phase: 'Implement + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true } }
await tryAgent(['BUILD-REPAIR for the qpwave wiring + CLI dedup. Accumulated edits (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON) + green ctest, patching only trivial -Werror / CMake wiring of the new cmd_qpwave + test_cli_qpwave. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement + build' })

phase('Verify golden_qpwave + no-regression + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','qpwave_through_cli','no_duplication','dedup_no_regression','dead_code_removed','goldens_green','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if steppe qpwave CLI reproduces golden_qpwave at tier (new cli_qpwave test); cmd_qpwave reuses the loader/resolver/emitter (no dup); the flag dedup is behavior-preserving (cli_qpadm/cli_rotate/cli_extract bit-identical green); run_not_yet_implemented removed; THOROUGH ctest green; Release build clean; single-GPU; no synthetic; no AT2' },
    qpwave_through_cli: { type: 'boolean', description: 'steppe qpwave reproduces golden_qpwave through the CLI (new test_cli_qpwave)' },
    no_duplication: { type: 'boolean', description: 'cmd_qpwave reused read_f2_dir/PopResolver/the emitter; emit_qpwave reused the csv/json helpers; no copied compute/format' },
    dedup_no_regression: { type: 'boolean', description: 'the flag dedup (--f2-dir/--right/--target/--left helpers) is behavior-preserving — cli_qpadm/cli_rotate/cli_extract_qpadm still pass bit-identical' },
    dead_code_removed: { type: 'boolean', description: 'run_not_yet_implemented deleted (qpwave was its last caller)' },
    goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the qpwave output + golden match; the dedup; the dead-code removal; for FAIL the exact issue' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the qpwave wiring + CLI dedup (adversarial). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff — confirm cmd_qpwave mirrors cmd_qpadm + REUSES the loader/resolver/emitter (NO dup compute/format), the qpwave subcommand routes to run_qpwave_command, run_not_yet_implemented is DELETED, and the flag dedup is behavior-preserving (--f2-dir/--right/--target/--left helpers, same parse). (2) ' + BUILD + ' — THOROUGH ctest green incl the new cli_qpwave + the existing cli_qpadm/cli_rotate/cli_extract_qpadm (bit-identical — proves no dedup regression) + qpwave_parity. (3) confirm `steppe qpwave` reproduces golden_qpwave through the CLI (the gate; NO AT2). PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/cmake files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (M(cli-2): wire qpwave CLI to run_qpwave [no dup] + CLI dedup [shared --f2-dir/--right/--target/--left helpers] + delete dead run_not_yet_implemented — golden-gated through the CLI vs golden_qpwave; full CLI surface now wired) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. ALSO update docs/RUN-SHEET.md (drop the qpwave "scaffold" warning) + docs/RESUME.md/TODO.md (M(cli-2) done; CLI surface complete). ',
  'ON FAIL: ' + CLEAN + ' and report the exact issue. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:qpwave', phase: 'Verify golden_qpwave + no-regression + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ qpwave WIRED + CLI cleaned ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- FAILED (' + verdict.note + ')')
return { verdict }
