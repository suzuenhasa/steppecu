export const meta = {
  name: 'qpdstat-A-f2path',
  description: 'STEP 3 stat #4 PART A (user-approved A+B plan): the `steppe qpdstat` CLI + binding, --f2-dir mode = batched f4 + sign + Z + p over quadruples. This IS exactly what admixtools-R qpdstat returns on the f2_data path (PROVEN byte-identical: qpdstat(f2dir,f4mode=TRUE)==f4mode=FALSE; the normalized-D denom is genotype-only). So this is full AT2-f2-path parity, not a shortcut; the D SIGN+Z+p (what gene-flow tests use) are all here. Part B (the `--prefix` genotype-path normalized-D MAGNITUDE) is the NEXT workflow. THIN: reuse run_f4 / assemble_f4_quartets (f4.cpp) verbatim for the batched f4 over quadruples + jackknife_cov diagonal SE + steppe::access + the CLI flag helpers + emitter + the M(py-1) nanobind module — only new = the `qpdstat` subcommand/entry wrapping run_f4 with the D-output convention (p1,p2,p3,p4,est,se,z,p) + quadruple input. Generate a fixture-matched golden via admixtools::qpdstat(read_f2(the 9-pop convertf-PA f2 == fixture), quadruples) -> golden_fit0_qpdstat_readf2.csv, and CONFIRM it == golden_fit0_f4_readf2.csv (documents the f2-path qpdstat=f4 equivalence). Gate cli_qpdstat + a pytest at est/se/z/p rtol 1e-6. NOTE in --help + meta.json that --f2-dir qpdstat reports f4 (the AT2 f2-path convention); the normalized-D magnitude needs --prefix (Part B). FAIL-PROTOCOL (user-mandated): NEVER git checkout/clean — on failure git stash push -u + HALT; verdict classifies severity. SINGLE-GPU; REAL data; commit-on-green.',
  phases: [ { title: 'Regen qpdstat-f2 golden + design (AT2)' }, { title: 'Implement qpdstat --f2-dir + CLI + binding + build' }, { title: 'Verify f2-path qpdstat parity + commit-or-stash' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const PULL = 'rsync -az -e ssh box5090:/workspace/steppe/tests/reference/goldens/at2/ ' + R + '/tests/reference/goldens/at2/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -30; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'
const PA = '/workspace/data/aadr/converted_pa/v66_HO_pa'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 9db62af. CLI: extract-f2/qpadm/qpwave/qpadm-rotate/f4/f3/f4-ratio all wired + golden-gated. M(py-1) bindings landed. f4.cpp (run_f4 / assemble_f4_quartets, the m-axis batched f4 + jackknife diagonal SE) is the template.',
  'GOAL (Part A of the qpDstat A+B plan): a `steppe qpdstat` CLI + steppe.qpdstat binding whose --f2-dir mode returns batched f4 + sign + Z + p over quadruples. VERIFIED FACT: admixtools-R qpdstat on the f2_data path == f4 (f4mode is a no-op without genotypes; qpdstat(f2dir,f4mode=T) byte-identical to f4mode=F). So this is full AT2-f2-path parity. The normalized-D MAGNITUDE (which needs per-SNP genotypes) is Part B (`--prefix`), a SEPARATE later workflow — this workflow does ONLY the --f2-dir path.',
  'THIN, NO DUP: reuse run_f4 / assemble_f4_quartets verbatim (the batched f4 + jackknife_cov diagonal SE); reuse steppe::access + the CLI flag helpers + the emitter + the nanobind module. New = the `qpdstat` subcommand + entry that wraps the f4 compute with the D-output convention (columns p1,p2,p3,p4,est,se,z,p) + quadruple input (single --pops A,B,C,D + batched). Add --help/meta wording: --f2-dir qpdstat reports f4 (AT2 f2-path convention); normalized-D needs --prefix (Part B).',
  'GOLDEN (no-AT2 lifted for GENERATION only): on box5090, admixtools::qpdstat(read_f2(the 9-pop convertf-PA f2 from ' + PA + ' at maxmiss=0 == the fixture f2_fit0_9pop.bin), the quadruples) -> golden_fit0_qpdstat_readf2.csv (est/se/z/p), and CONFIRM est==golden_fit0_f4_readf2.csv (the f2-path qpdstat=f4 equivalence). The 9 fit0 pops: England_BellBeaker, Czechia_EBA_CordedWare, Turkey_N, Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout/clean. On ANY failure ' + STASH + ' "wip:qpdstatA-FAILED-<reason>" + HALT. NON-trivial blocker -> STOP + report. Classify minor (move-on) vs bad (defer).',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally; §4 layering.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Regen qpdstat-f2 golden + design (AT2)')
const REGEN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['done','golden_path','equiv_check','design','notes'],
  properties: {
    done: { type: 'boolean' },
    golden_path: { type: 'string', description: 'golden_fit0_qpdstat_readf2.csv (path + sample row) pulled local' },
    equiv_check: { type: 'string', description: 'PROOF golden_fit0_qpdstat_readf2 est == golden_fit0_f4_readf2 (the f2-path qpdstat=f4 equivalence) to ~1e-12' },
    design: { type: 'string', description: 'the qpdstat --f2-dir entry plan (wrap run_f4 over quadruples + the D-output convention; CLI + binding) — file:line for the reuse' },
    notes: { type: 'string' },
  },
}
const regen = await tryAgent([
  'You are (a) generating the fixture-matched qpdstat-f2 golden via AT2 and (b) designing the `qpdstat --f2-dir` entry by reusing run_f4. NO steppe code changes this phase. (1) READ f4.cpp (run_f4 / assemble_f4_quartets) + cmd_f4.cpp + the f4 binding + cli_f4 + golden_fit0_f4_readf2.csv (the template + the values). (2) On box5090 (AT2/R, lifted no-AT2): admixtools::qpdstat(read_f2(the 9-pop convertf-PA f2 at maxmiss=0 == the fixture), the quadruples used by golden_fit0_f4_readf2.csv) -> golden_fit0_qpdstat_readf2.csv (est/se/z/p); CONFIRM est == golden_fit0_f4_readf2.csv to ~1e-12 (the f2-path qpdstat=f4 equivalence). Pull: ' + PULL + ' . (3) design the qpdstat --f2-dir entry (wrap run_f4/assemble_f4_quartets with the D-output convention; the `steppe qpdstat` CLI subcommand reusing the flag helpers; steppe.qpdstat binding). Cite file:line. Return the structured result; do NOT implement.', STD,
].join('\n'), { schema: REGEN_SCHEMA, label: 'regen+design:qpdstatA', phase: 'Regen qpdstat-f2 golden + design (AT2)' })
if (!regen || !regen.done) { log('--- qpdstat-A regen/design failed — HALT: ' + (regen ? regen.notes : 'agent died')); return { halted: true, regen } }
log('qpdstat-f2 golden == f4: ' + String(regen.equiv_check).slice(0,100))

phase('Implement qpdstat --f2-dir + CLI + binding + build')
const fixer = await tryAgent([
  'You are implementing `steppe qpdstat --f2-dir` (Part A) per this design (reuse run_f4, no dup):\n<<<\n' + regen.design + '\nGOLDEN: ' + regen.golden_path + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced). Work from HEAD.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT the `qpdstat` subcommand + entry (--f2-dir mode = batched f4 over quadruples via run_f4/assemble_f4_quartets, D-output convention p1..p4,est,se,z,p; single --pops A,B,C,D + batched), the steppe.qpdstat binding, --help/meta wording (f2-path reports f4; normalized-D needs --prefix = Part B, not yet implemented -> a clear message if --prefix is given), and the gate: cli_qpdstat + a pytest reproducing golden_fit0_qpdstat_readf2.csv at est/se/z/p rtol 1e-6. Build + full STEPPE_THOROUGH ctest. SANITY: cli_qpdstat reproduces the golden; existing goldens/cli/pytest stay green. Report files changed, the reuse, the gate, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:qpdstatA', phase: 'Implement qpdstat --f2-dir + CLI + binding + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, regen } }
await tryAgent(['BUILD-REPAIR for qpdstat Part A. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON) + green ctest, patching only trivial -Werror / CMake wiring. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement qpdstat --f2-dir + CLI + binding + build' })

phase('Verify f2-path qpdstat parity + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','qpdstat_parity','python_ok','equiv_to_f4','no_duplication','goldens_green','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if `steppe qpdstat --f2-dir` reproduces golden_fit0_qpdstat_readf2.csv (== f4) at est/se/z/p rtol 1e-6 (cli + pytest), --prefix gives a clear not-yet-implemented message, no dup (reuses run_f4), full ctest green, build clean, single-GPU, no synthetic' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (empty if pass)' },
    qpdstat_parity: { type: 'boolean' }, python_ok: { type: 'boolean' }, equiv_to_f4: { type: 'boolean', description: 'the qpdstat-f2 golden == the f4 golden (the proven equivalence)' },
    no_duplication: { type: 'boolean' }, goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for qpdstat Part A (adversarial). regen+design:\n<<<\n' + JSON.stringify(regen) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) confirm golden_fit0_qpdstat_readf2.csv == golden_fit0_f4_readf2.csv (the f2-path qpdstat=f4 equivalence). (2) git diff review: qpdstat --f2-dir reuses run_f4/assemble_f4_quartets (NO dup); --prefix gives a clear not-yet-impl message. (3) ' + BUILD + ' — STEPPE_THOROUGH ctest green; cli_qpdstat + pytest reproduce the golden at est/se/z/p rtol 1e-6. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new golden + new/changed source/test/cmake/binding/doc (NEVER git add dot; never aadr/ atlas_results/), commit (STEP3 qpDstat Part A: `steppe qpdstat --f2-dir` = batched f4+Z/p [AT2 f2-path parity, proven == f4] + steppe.qpdstat binding [reuse run_f4, no dup]; --prefix normalized-D = Part B pending) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (qpDstat Part A done; Part B = genotype-path normalized-D next). ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:qpdstatA-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:qpdstatA', phase: 'Verify f2-path qpdstat parity + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, regen } }
if (verdict.pass) log('+++ qpdstat Part A ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- qpdstat Part A FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
