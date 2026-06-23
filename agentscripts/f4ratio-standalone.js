export const meta = {
  name: 'f4ratio-standalone',
  description: 'STEP 3 stat #3: standalone f4-ratio (admixture proportion). alpha = f4(A,O ; X,C) / f4(A,O ; B,C) — the classic Patterson f4-ratio estimator of the mixture proportion (the exact AT2 qpf4ratio convention to be verified). Reuses the just-landed f4 machinery (run_f4 / assemble_f4_quartets, f4.cpp) to get the two f4 numerator/denominator PER jackknife block, then the ratio, then the SE = block-jackknife OF THE RATIO (Busing pseudo-values of the per-block alpha — NOT the components separately). Reuse jackknife_cov/the jackknife seam + steppe::access + the CLI flag helpers + emitter + the M(py-1) nanobind module — NO dup; only new math = the per-block ratio + its jackknife. Generate a fresh fixture-matched AT2 golden (Option 2; no-AT2 lifted for golden GENERATION only — runtime never needs AT2): admixtools::qpf4ratio(read_f2(the 9-pop convertf-PA f2 == the fixture f2_fit0_9pop.bin), the 5-pop tuples) -> golden_fit0_f4ratio_readf2.csv (alpha/se/z), PROVEN fixture-consistent. Then run_f4ratio + `steppe f4-ratio` CLI + steppe.f4ratio binding; gate cli_f4ratio + a pytest at rtol 1e-6. FAIL-PROTOCOL (user-mandated): NEVER git checkout/clean — on failure git stash push -u + HALT; verdict classifies severity (minor=move-on / bad=defer). SINGLE-GPU; REAL data; commit-on-green.',
  phases: [ { title: 'Regenerate the fixture-matched f4-ratio golden + design (AT2)' }, { title: 'Implement run_f4ratio + CLI + binding + build' }, { title: 'Verify bit-tight f4-ratio parity + commit-or-stash' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 56fe9fc. CLI complete; M(py-1) bindings; standalone f4 (f4.cpp run_f4 / assemble_f4_quartets, 271e302) + f3 (f3.cpp run_f3 / assemble_f3_triples, cfa0d9d) landed + golden-gated. They are the EXACT templates to mirror.',
  'GOAL: standalone f4-ratio = the admixture proportion alpha = f4(A,O;X,C)/f4(A,O;B,C) (verify the EXACT AT2 qpf4ratio pop-arg convention). The numerator + denominator are each an f4 (reuse the f4 per-block machinery). The estimate = ratio of the block-summed f4s; the SE = the block-jackknife OF THE RATIO statistic (compute alpha leaving out each block via Busing pseudo-values — NOT se(num)/se(den) separately). z=alpha/se. Reuse jackknife_cov / the jackknife seam.',
  'GOLDEN (fresh, fixture-matched; no-AT2 lifted for GENERATION only): on box5090, admixtools::qpf4ratio(read_f2(the 9-pop convertf-PA f2 from ' + PA + ' at maxmiss=0 == the fixture f2_fit0_9pop.bin), a sensible set of 5-pop tuples) -> tests/reference/goldens/at2/csv/golden_fit0_f4ratio_readf2.csv (alpha/se/z). The 9 fit0 pops: England_BellBeaker, Czechia_EBA_CordedWare, Turkey_N, Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana. PROVE fixture-consistency (steppe f4-ratio-from-the-fixture-f2 == the golden to ~1e-6) before trusting it.',
  'REUSE (verify file:line, no dup): f4.cpp (run_f4 / assemble_f4_quartets — the per-block f4); f3.cpp (the sibling template); jackknife_cov (backend.hpp:634) / the loo+total seam; steppe::access; the CLI flag helpers; the result emitter; the nanobind module. ONLY new = the per-block ratio + its jackknife.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout/clean. On ANY failure ' + STASH + ' "wip:f4ratio-FAILED-<reason>" + HALT. NON-trivial blocker -> STOP + report (verdict stashes). Classify minor (move-on) vs bad (defer).',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally; §4 layering (cmd/binding TUs CUDA-free).',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Regenerate the fixture-matched f4-ratio golden + design (AT2)')
const REGEN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['done','golden_path','fixture_match_check','convention','design','notes'],
  properties: {
    done: { type: 'boolean' },
    golden_path: { type: 'string', description: 'golden_fit0_f4ratio_readf2.csv (path + sample row + the 5-pop tuples) pulled to local' },
    fixture_match_check: { type: 'string', description: 'PROOF the golden matches the read_f2 fixture: AT2 alpha vs steppe f4-ratio-from-fixture-f2 to ~1e-6' },
    convention: { type: 'string', description: 'the EXACT AT2 qpf4ratio pop-arg convention (which of the 5 pops is num/den/target/refs) so the CLI matches' },
    design: { type: 'string', description: 'the run_f4ratio plan: reuse the f4 per-block + the ratio + the jackknife-of-the-ratio; the CLI + binding shape — file:line' },
    notes: { type: 'string', description: 'any caveat (ratio SE jackknife specifics; near-zero denominator handling)' },
  },
}
const regen = await tryAgent([
  'You are (a) generating a fresh fixture-matched f4-ratio golden via AT2 and (b) designing run_f4ratio by MIRRORING the f4/f3 templates. NO steppe code changes this phase. (1) READ f4.cpp (run_f4 / assemble_f4_quartets / the jackknife reuse) + f3.cpp + cmd_f4.cpp + the f4 binding + cli_f4 (the template). (2) Determine the EXACT admixtools qpf4ratio convention (web/the R docs: which pops are num/den). On box5090 (AT2/R, lifted no-AT2): read_f2 the 9-pop convertf-PA f2 at maxmiss=0 (== fixture), run admixtools::qpf4ratio(that_f2, a sensible set of 5-pop tuples, e.g. England_BellBeaker as the admixed target between CordedWare and Turkey_N with Mbuti/Han outgroups) -> golden_fit0_f4ratio_readf2.csv (alpha/se/z). PROVE fixture-consistency (steppe f4-ratio-from-the-fixture-f2 [ratio of the two block-jackknifed f4s] == the golden to ~1e-6). Pull: ' + PULL + ' . (3) design run_f4ratio (the per-block num/den f4 reusing assemble_f4_quartets, the ratio, the jackknife-OF-THE-RATIO for the SE; the `steppe f4-ratio` CLI + steppe.f4ratio binding). Cite file:line. Return the structured result; do NOT implement. If qpf4ratio cannot be made fixture-consistent, HALT + report (bad).', STD,
].join('\n'), { schema: REGEN_SCHEMA, label: 'regen+design:f4ratio', phase: 'Regenerate the fixture-matched f4-ratio golden + design (AT2)' })
if (!regen || !regen.done) { log('--- f4ratio golden regen/design failed — HALT (defer): ' + (regen ? regen.notes : 'agent died')); return { halted: true, regen } }
log('f4ratio golden regenerated + fixture-matched: ' + String(regen.fixture_match_check).slice(0,120))

phase('Implement run_f4ratio + CLI + binding + build')
const fixer = await tryAgent([
  'You are implementing standalone f4-ratio per this design (MIRROR f4/f3), gated against the fixture-matched golden:\n<<<\nconvention: ' + regen.convention + '\n' + regen.design + '\nGOLDEN: ' + regen.golden_path + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced). Work from HEAD.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT run_f4ratio (reuse the f4 per-block + jackknife_cov machinery; the per-block ratio + the jackknife-of-the-ratio for the SE; no dup), the `steppe f4-ratio` CLI (reuse the flag helpers + emitter; the 5-pop tuple input), the steppe.f4ratio binding, and the gate: a cli_f4ratio ctest + a pytest reproducing golden_fit0_f4ratio_readf2.csv at alpha/se/z rtol 1e-6. Build + full STEPPE_THOROUGH ctest. SANITY: cli_f4ratio reproduces the golden; existing goldens/cli/pytest stay green. Report every file added/changed, the reuse, the gate, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:f4ratio', phase: 'Implement run_f4ratio + CLI + binding + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, regen } }
await tryAgent(['BUILD-REPAIR for f4-ratio. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON) + green ctest, patching only trivial -Werror / CMake wiring. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement run_f4ratio + CLI + binding + build' })

phase('Verify bit-tight f4-ratio parity + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','f4ratio_parity','python_ok','golden_fixture_matched','no_duplication','goldens_green','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the golden is fixture-matched, `steppe f4-ratio` reproduces golden_fit0_f4ratio_readf2.csv at alpha/se/z rtol 1e-6 (cli + pytest), the SE is the jackknife-of-the-ratio (not components), no dup, full ctest green, build clean, single-GPU, no synthetic' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (empty if pass)' },
    f4ratio_parity: { type: 'boolean' }, python_ok: { type: 'boolean' }, golden_fixture_matched: { type: 'boolean' },
    no_duplication: { type: 'boolean' }, goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the f4-ratio parity + fixture-match + the SE-jackknife approach; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for f4-ratio (adversarial). regen+design:\n<<<\n' + JSON.stringify(regen) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) confirm golden_fit0_f4ratio_readf2.csv is fixture-matched (independently: alpha == steppe f4-ratio-from-fixture-f2 to ~1e-6). (2) git diff review: run_f4ratio reuses the f4/jackknife machinery (no dup); the SE is the jackknife OF THE RATIO (per-block alpha pseudo-values), NOT se(num)/se(den). (3) ' + BUILD + ' — STEPPE_THOROUGH ctest green; cli_f4ratio + pytest reproduce the golden at alpha/se/z rtol 1e-6. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new golden + new/changed source/test/cmake/binding/doc (NEVER git add dot; never aadr/ atlas_results/ handoff-*.md), commit (STEP3 f4-ratio: standalone admixture-proportion CLI `steppe f4-ratio` + steppe.f4ratio binding [reuse f4 + jackknife-of-the-ratio, no dup] + fixture-matched AT2 golden -> bit-tight parity) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (f4-ratio done; next D/qpDstat). ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:f4ratio-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:f4ratio', phase: 'Verify bit-tight f4-ratio parity + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, regen } }
if (verdict.pass) log('+++ f4-ratio ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- f4-ratio FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
