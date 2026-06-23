export const meta = {
  name: 'f3-standalone',
  description: 'STEP 3 stat #2: standalone f3(C;A,B) — outgroup-f3 (shared drift) + admixture-f3 (the negative-f3 admixture test). f3(C;A,B) = mean_SNP (c-a)(c-b) = (f2(A,C)+f2(B,C)-f2(A,B))/2 from the UNBIASED f2 (steppe f2 already carries the het/bias correction — the pseudo-haploid fix — so f3 from it is the unbiased f3; verify whether any extra vpair/het term is needed for admixture-f3). Reuse the f2 linear combo + jackknife_cov + steppe::access + the CLI flag helpers + result emitter + the M(py-1) nanobind module — NO dup (mirror the just-landed f4: run_f4/f4.cpp 271e302). There is NO existing f3 golden, so GENERATE a fresh fixture-matched one (user chose Option 2; no-AT2 lifted for golden GENERATION only — the steppe runtime never needs AT2). Phase 1 REGEN+DESIGN: on box5090, admixtools::f3(read_f2(the 9-pop convertf-PA f2 object, maxmiss=0 == the fixture f2_fit0_9pop.bin), a sensible triple set [outgroup-f3 e.g. Mbuti;A,B over the panel + the admixture-f3 cases]) -> golden_fit0_f3_readf2.csv (est/se/z/p), PROVEN fixture-consistent (matches steppe f3-from-f2 / the four-slab triple identity to ~1e-6). Phase 2 IMPLEMENT run_f3 + `steppe f3` CLI + steppe.f3 binding. Phase 3 VERIFY: cli_f3 + a pytest reproduce the golden at est/se/z/p rtol 1e-6; full ctest green; commit. FAIL-PROTOCOL (user-mandated): NEVER git checkout/clean — on failure git stash push -u + HALT; verdict classifies severity (minor=move-on / bad=defer). SINGLE-GPU; REAL data; commit-on-green.',
  phases: [ { title: 'Regenerate the fixture-matched f3 golden + design (AT2)' }, { title: 'Implement run_f3 + CLI + binding + build' }, { title: 'Verify bit-tight f3 parity + commit-or-stash' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 9b9eda4. CLI complete; M(py-1) bindings landed; the just-landed standalone f4 (src/core/qpadm/f4.cpp, run_f4; cmd_f4.cpp; steppe.f4; cli_f4 #47; golden_fit0_f4_readf2.csv) is the EXACT template to mirror for f3 — copy its structure (reuse the f2 contraction + jackknife_cov + steppe::access + the CLI helpers + the emitter + the nanobind module).',
  'GOAL: standalone f3(C;A,B). MATH: f3(C;A,B) = mean_SNP (c-a)(c-b) = (f2(A,C)+f2(B,C)-f2(A,B))/2 computed PER jackknife block from the UNBIASED f2 (steppe f2 already has the het/bias correction from the pseudo-haploid fix — so f3 from it IS the unbiased f3; VERIFY whether admixture-f3 needs any extra vpair/het term, the research said it is already resident). z=est/se via the block-jackknife, p the 2-sided normal. Outgroup-f3 = f3(Outgroup;A,B) (shared drift); admixture-f3 = f3(Target;Src1,Src2) (negative => admixture). Same formula.',
  'GOLDEN (no existing f3 golden -> generate fresh, fixture-matched; no-AT2 lifted for GENERATION only): on box5090, admixtools::f3(read_f2(the 9-pop convertf-PA f2 from ' + PA + ' at maxmiss=0 == the committed fixture f2_fit0_9pop.bin), a sensible triple set) -> tests/reference/goldens/at2/csv/golden_fit0_f3_readf2.csv. The 9 fit0 pops: England_BellBeaker, Czechia_EBA_CordedWare, Turkey_N, Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana. PROVE the golden is fixture-consistent (matches steppe f3-from-the-fixture-f2 to ~1e-6) before trusting it.',
  'REUSE (verify file:line, no dup): the f4 template (f4.cpp/cmd_f4.cpp/the f4 binding/cli_f4); the f2 linear combo + jackknife_cov (backend.hpp:634); steppe::access; the CLI flag helpers (cli_parse.cpp); the result emitter; the nanobind module.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout/clean. On ANY failure ' + STASH + ' "wip:f3-FAILED-<reason>" + HALT. NON-trivial blocker -> STOP + report (verdict stashes). Classify minor (move-on) vs bad (defer).',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally; §4 layering (cmd/binding TUs CUDA-free).',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Regenerate the fixture-matched f3 golden + design (AT2)')
const REGEN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['done','golden_path','fixture_match_check','design','notes'],
  properties: {
    done: { type: 'boolean' },
    golden_path: { type: 'string', description: 'golden_fit0_f3_readf2.csv (path + sample row + the triple set chosen) pulled to local' },
    fixture_match_check: { type: 'string', description: 'PROOF the golden matches the read_f2 fixture: admixtools f3 est vs steppe f3-from-fixture-f2 (the triple identity) to ~1e-6' },
    design: { type: 'string', description: 'the run_f3 plan mirroring f4 (the triple identity + jackknife; whether admixture-f3 needs an extra vpair/het term; CLI + binding shape) — file:line' },
    notes: { type: 'string', description: 'the admixtools f3 convention (f3(pop1;pop2,pop3) arg order); any caveat' },
  },
}
const regen = await tryAgent([
  'You are (a) generating a fresh fixture-matched f3 golden via AT2 and (b) designing run_f3 by MIRRORING the just-landed f4. NO steppe code changes this phase. (1) READ the f4 template: src/core/qpadm/f4.cpp (run_f4 + assemble_f4_quartets + the jackknife reuse), src/app/cmd_f4.cpp, the f4 binding in the nanobind module, tests/cli/test_cli_f4* + the f4 pytest, golden_fit0_f4_readf2.csv (the schema/approach). (2) On box5090 (AT2/R, lifted no-AT2): read_f2 the 9-pop convertf-PA f2 at maxmiss=0 (== the fixture), run admixtools::f3(that_f2, a sensible triple set: outgroup-f3 with Mbuti as outgroup over the panel + the admixture-f3 target cases) -> golden_fit0_f3_readf2.csv (est/se/z/p). PROVE fixture-consistency (steppe f3-from-the-fixture-f2 via (f2(A,C)+f2(B,C)-f2(A,B))/2 jackknifed == the golden to ~1e-6). Pull: ' + PULL + ' . (3) design run_f3 (the triple identity reusing the f4 machinery + jackknife_cov; whether the admixture-f3 het is already in the f2 or needs vpair; the `steppe f3` CLI + steppe.f3 binding). Cite file:line. Return the structured result; do NOT implement.', STD,
].join('\n'), { schema: REGEN_SCHEMA, label: 'regen+design:f3', phase: 'Regenerate the fixture-matched f3 golden + design (AT2)' })
if (!regen || !regen.done) { log('--- f3 golden regen/design failed — HALT (defer): ' + (regen ? regen.notes : 'agent died')); return { halted: true, regen } }
log('f3 golden regenerated + fixture-matched: ' + String(regen.fixture_match_check).slice(0,120))

phase('Implement run_f3 + CLI + binding + build')
const fixer = await tryAgent([
  'You are implementing standalone f3 per this design (MIRROR the f4 template), gated against the fixture-matched golden:\n<<<\n' + regen.design + '\nGOLDEN: ' + regen.golden_path + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced). Work from HEAD.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT run_f3 (mirror run_f4 — the triple identity reusing the f2 contraction + jackknife_cov + steppe::access; no dup), the `steppe f3` CLI (reuse the flag helpers + emitter; --pops C,A,B + a batched/outgroup-f3 mode), the steppe.f3 binding (nanobind module), and the gate: a cli_f3 ctest + a pytest reproducing golden_fit0_f3_readf2.csv at est/se/z/p rtol 1e-6. Build + full STEPPE_THOROUGH ctest. SANITY: cli_f3 reproduces the golden; existing goldens/cli/pytest stay green. Report every file added/changed, the reuse, the gate, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:f3', phase: 'Implement run_f3 + CLI + binding + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, regen } }
await tryAgent(['BUILD-REPAIR for f3. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON) + green ctest, patching only trivial -Werror / CMake wiring. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement run_f3 + CLI + binding + build' })

phase('Verify bit-tight f3 parity + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','f3_parity','python_f3_ok','golden_fixture_matched','no_duplication','goldens_green','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the fresh golden is fixture-matched, `steppe f3` reproduces golden_fit0_f3_readf2.csv at est/se/z/p rtol 1e-6 (cli_f3 + pytest), no dup, full ctest green, build clean, single-GPU, no synthetic' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (empty if pass)' },
    f3_parity: { type: 'boolean' }, python_f3_ok: { type: 'boolean' }, golden_fixture_matched: { type: 'boolean' },
    no_duplication: { type: 'boolean' }, goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the f3 parity + fixture-match; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for f3 (adversarial). regen+design:\n<<<\n' + JSON.stringify(regen) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) confirm golden_fit0_f3_readf2.csv is fixture-matched (independently check est == steppe f3-from-fixture-f2 to ~1e-6). (2) git diff review: run_f3 reuses the f4/f2/jackknife machinery (no dup). (3) ' + BUILD + ' — STEPPE_THOROUGH ctest green; cli_f3 + pytest reproduce the golden at est/se/z/p rtol 1e-6. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new golden + new/changed source/test/cmake/binding/doc (NEVER git add dot; never aadr/ atlas_results/ handoff-*.md), commit (STEP3 f3: standalone f3(C;A,B) outgroup+admixture CLI `steppe f3` + steppe.f3 binding [reuse f4/f2/jackknife, no dup] + fixture-matched AT2 golden -> bit-tight parity) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (f3 done; next f4-ratio). ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:f3-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:f3', phase: 'Verify bit-tight f3 parity + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, regen } }
if (verdict.pass) log('+++ f3 ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- f3 FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
