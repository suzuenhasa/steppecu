export const meta = {
  name: 'f4-standalone',
  description: 'STEP 3 stat #1 (v2, user chose Option 2 = regenerate fixture-matched AT2 goldens): standalone f4(A,B;C,D). The existing golden_fit0_f4.csv is qpadm-internal-allsnps-normalized (a CONSTANT 1.1132x off the read_f2/maxmiss=0 fixture f2_fit0_9pop.bin), so it CANNOT gate a standalone f4 over the fixture. FIX (no-AT2 LIFTED for golden GENERATION only, like every other steppe golden; the steppe runtime still never depends on AT2): Phase 1 REGEN a fixture-matched golden on box5090 via admixtools::f4(read_f2(the 9-pop convertf-PA f2 object), the SAME quartets as golden_fit0_f4.csv) -> golden_fit0_f4_readf2.csv (est/se/z/p matching the fixture normalization; the read_f2 f2 object == f2_fit0_9pop.bin, so it is consistent with the steppe fixture). CONFIRM the regenerated golden now matches steppe assemble_f4 to ~1e-6 (i.e. the 1.1132 offset is GONE) before trusting it. Phase 2 IMPLEMENT run_f4 (single + batched all-quartets; reuse assemble_f4 backend.hpp:604,618 + jackknife_cov :634 + steppe::access + the CLI flag helpers + result emitter + the M(py-1) nanobind module — NO dup compute) + a `steppe f4` CLI + a steppe.f4 binding. Phase 3 VERIFY: cli_f4 + a pytest reproduce golden_fit0_f4_readf2.csv at est/se/z/p rtol 1e-6 (bit-tight AT2 f4-table parity); full ctest green; commit. Discipline: design -> regen-golden -> implement -> verify. FAIL-PROTOCOL (user-mandated): NEVER git checkout/clean — on failure git stash push -u the attempt (reviewable) + HALT; the verdict classifies severity (minor=move-on / bad=defer). SINGLE-GPU; REAL data; commit-on-green.',
  phases: [ { title: 'Design + regenerate the fixture-matched f4 golden (AT2)' }, { title: 'Implement run_f4 + CLI + binding + build' }, { title: 'Verify bit-tight f4 parity + commit-or-stash' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 93b3217. CLI complete; M(py-1) nanobind bindings landed (steppe._core + bindings/steppe; the f2-dir reader + PopResolver are the shared steppe::access lib). The fit engine computes f4 INTERNALLY (assemble_f4) + the block-jackknife (jackknife_cov).',
  'THE GOLDEN PROBLEM (decided by the user = Option 2): tests/reference/goldens/at2/csv/golden_fit0_f4.csv is qpadm-internal f4 with the ALLSNPS/maximal-overlap normalization (its `weight` column = qpadm popdrop weights), a CONSTANT 1.1131993890x off the read_f2/maxmiss=0 fixture f2_fit0_9pop.bin (P=9, nb=710, 351539 SNPs). A standalone f4 over the fixture CANNOT match it at tolerance. FIX = regenerate a fixture-matched golden via admixtools::f4(read_f2(...)).',
  'NO-AT2 IS LIFTED FOR GOLDEN GENERATION ONLY (this is how every steppe golden was made; the steppe RUNTIME still never links/needs AT2). AT2/R + admixtools are on the box; convertf-PA at ' + PA + '. The 9 fit0 pops: England_BellBeaker, Czechia_EBA_CordedWare, Turkey_N, Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana (golden_fit0 = England_BellBeaker via CordedWare+Turkey_N, right = the other 6). The f4 golden = f4 over the quartets in golden_fit0_f4.csv, but computed from read_f2(the 9-pop maxmiss=0 f2) so it matches the fixture.',
  'REUSE (verify file:line, no dup): assemble_f4 + the identity (backend.hpp:597-602,604,618); jackknife_cov (:634); steppe::access (the shared f2-dir reader + PopResolver); the CLI flag helpers (cli_parse.cpp:72/87/126) + the cmd_qpadm.cpp pattern; the result emitter (result_emit.cpp); the M(py-1) nanobind module (add steppe.f4).',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER git checkout -- . / git clean -fd (they destroyed work before). On ANY failure PRESERVE via ' + STASH + ' "wip:f4-FAILED-<reason>" (tracked+untracked, restorable) + HALT. The fixer must NOT destroy; NON-trivial blocker -> STOP + report (the verdict stashes). Classify severity: minor (move-on) vs bad (defer to user).',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. NAMING-STYLE-STANDARD + §4 layering. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Design + regenerate the fixture-matched f4 golden (AT2)')
const REGEN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['done','golden_path','fixture_match_check','design','notes'],
  properties: {
    done: { type: 'boolean' },
    golden_path: { type: 'string', description: 'the regenerated fixture-matched f4 golden (path + a sample row: pop1..pop4,est,se,z,p) — pulled to local tests/reference/goldens/at2/' },
    fixture_match_check: { type: 'string', description: 'PROOF the new golden matches the read_f2/maxmiss=0 fixture (not allsnps): the new est vs steppe assemble_f4 (or vs the old golden / 1.1132) — the 1.1132 offset must be GONE (new golden ~= fixture f4 to ~1e-6)' },
    design: { type: 'string', description: 'the run_f4 implementation plan (reuse assemble_f4/jackknife_cov/access; the CLI + binding shape) — file:line for the reuse seams' },
    notes: { type: 'string', description: 'how the golden was regenerated (the R/admixtools call: read_f2 the 9-pop convertf-PA f2 + f4 over the quartets); any caveat' },
  },
}
const regen = await tryAgent([
  'You are (a) regenerating the fixture-matched f4 golden via AT2 and (b) designing run_f4. NO steppe code changes this phase. (1) READ how golden_fit0_f4.csv + the f2_fit0_9pop.bin fixture were generated (tests/reference/goldens/at2/scripts/golden_fit0*_generate.R + the fixup), confirm the 1.1132 allsnps-vs-read_f2 mismatch. (2) On box5090 (AT2/R, lifted no-AT2): build/read the read_f2 f2 OBJECT for the 9 fit0 pops from the convertf-PA ' + PA + ' at maxmiss=0 (the SAME f2 as the committed fixture), run admixtools::f4(that_f2, the quartets from golden_fit0_f4.csv) -> a NEW golden_fit0_f4_readf2.csv (est/se/z/p). PROVE it now matches the fixture (compute f4 from f2_fit0_9pop.bin directly, or vs steppe assemble_f4 — the 1.1132 offset must be GONE, agreement ~1e-6). Pull it to local: ' + PULL + ' . (3) design run_f4 (reuse assemble_f4/jackknife_cov/steppe::access; the CLI `f4` + the steppe.f4 binding; cite file:line).', STD, '',
  'Return the structured result (the regenerated golden + the fixture-match PROOF + the run_f4 design). Do NOT implement run_f4 yet. If the regen cannot be made fixture-consistent, HALT + report (bad).',
].join('\n'), { schema: REGEN_SCHEMA, label: 'regen+design:f4', phase: 'Design + regenerate the fixture-matched f4 golden (AT2)' })
if (!regen || !regen.done) { log('--- f4 golden regen/design failed — HALT (defer): ' + (regen ? regen.notes : 'agent died')); return { halted: true, regen } }
log('f4 golden regenerated + fixture-matched: ' + String(regen.fixture_match_check).slice(0,120))

phase('Implement run_f4 + CLI + binding + build')
const fixer = await tryAgent([
  'You are implementing standalone f4 per this design + against the fixture-matched golden:\n<<<\n' + regen.design + '\nGOLDEN: ' + regen.golden_path + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced). Work from HEAD (the regenerated golden is already pulled local).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT run_f4 (single quartet + batched all-quartets; reuse assemble_f4/jackknife_cov/steppe::access — no dup), the `steppe f4` CLI (reuse the flag helpers + emitter; --pops A,B,C,D + all-quartets-over-the-dir), the steppe.f4 binding (M(py-1) module), and the gate: a cli_f4 ctest + a pytest reproducing the NEW golden_fit0_f4_readf2.csv at est/se/z/p rtol 1e-6. Build + full STEPPE_THOROUGH ctest. SANITY: cli_f4 reproduces the fixture-matched golden (est/se/z/p rtol 1e-6); `steppe f4 --f2-dir <fit0 dir> --pops England_BellBeaker,Czechia_EBA_CordedWare,Han,Iran_GanjDareh_N` matches the golden row; existing goldens/cli/pytest stay green. Report every file added/changed, the reuse, the gate result, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:f4', phase: 'Implement run_f4 + CLI + binding + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, regen } }
await tryAgent(['BUILD-REPAIR for f4. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON) + green ctest, patching only trivial -Werror / CMake wiring. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement run_f4 + CLI + binding + build' })

phase('Verify bit-tight f4 parity + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','f4_parity','python_f4_ok','golden_fixture_matched','no_duplication','goldens_green','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the regenerated golden is fixture-matched (1.1132 offset gone), `steppe f4` reproduces golden_fit0_f4_readf2.csv at est/se/z/p rtol 1e-6 (cli_f4 + pytest), no dup, full ctest green, Release build clean, single-GPU, no synthetic' },
    fail_severity: { type: 'string', description: 'if pass=false: "minor" or "bad" (empty if pass)' },
    f4_parity: { type: 'boolean', description: 'steppe f4 == the fixture-matched AT2 golden at rtol 1e-6 (est+se+z+p)' },
    python_f4_ok: { type: 'boolean' }, golden_fixture_matched: { type: 'boolean', description: 'the regenerated golden matches the read_f2 fixture (not allsnps; the 1.1132 offset is gone)' },
    no_duplication: { type: 'boolean' }, goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the f4 est-vs-golden parity + the fixture-match proof; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for f4 (adversarial). The regen+design reported:\n<<<\n' + JSON.stringify(regen) + '\n>>>\nThe implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) confirm the regenerated golden_fit0_f4_readf2.csv is FIXTURE-MATCHED (not the allsnps one — independently check the est matches steppe assemble_f4 / the fixture f4 to ~1e-6, the 1.1132 offset gone). (2) git diff review: run_f4 reuses assemble_f4/jackknife_cov/access (no dup); the CLI + binding reuse. (3) ' + BUILD + ' — STEPPE_THOROUGH ctest green; cli_f4 + the pytest reproduce the fixture-matched golden at est/se/z/p rtol 1e-6. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new golden + the new/changed source/test/cmake/binding/doc files (NEVER git add dot; never aadr/ atlas_results/ handoff-*.md), commit with a ROADMAP §6 message (STEP3 f4: standalone f4 CLI `steppe f4` + steppe.f4 binding [reuse assemble_f4 + jackknife, no dup] + regenerated fixture-matched AT2 golden_fit0_f4_readf2 -> bit-tight f4-table parity) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (RESUME/TODO: f4 done; next f3). ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:f4-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:f4', phase: 'Verify bit-tight f4 parity + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, regen } }
if (verdict.pass) log('+++ f4 ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- f4 FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
