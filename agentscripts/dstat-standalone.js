export const meta = {
  name: 'dstat-standalone',
  description: 'STEP 3 stat #4: standalone D-statistic / qpDstat (the LAST core descriptive f-stat). D(A,B;C,D) = f4(A,B;C,D) / denom, denom = the ABBA+BABA / het normalization. THE QUESTION (the research flagged the f2 cache discards per-SNP freqs): can D be computed from the f2-cache-level data the way admixtools itself does it (admixtools f4(f4mode=FALSE)/qpdstat returns NORMALIZED D from f2_data — so AT2 MUST derive the denominator from f2-cache-level quantities, NOT raw genotypes)? VERIFY how AT2 does it (read the admixtools R source + the box DReichLab AdmixTools C /workspace/AdmixTools_src) + whether steppe f2 cache (f2 + vpair + block_sizes, or a BOUNDED addition) supports it. The DESIGN PHASE GATES: (a) feasible from the f2 cache (or a small bounded addition) -> proceed (regen fixture-matched golden + implement run_dstat + CLI `steppe qpdstat` + binding, gate rtol 1e-6, mirror f4/f3/f4ratio); (b) genuinely needs a genotype pass / a major f2-dir FORMAT change -> HALT + DEFER to the user (do NOT silently change the format / do a genotype pass without sign-off). steppe ALREADY has D sign+Z+p (== f4 Z/p; the denom cancels), so a fallback is qpDstat=batched-f4+Z/p (honestly labeled). FAIL-PROTOCOL (user-mandated): NEVER git checkout/clean — on failure git stash push -u + HALT; verdict classifies severity (minor=move-on / bad=defer). SINGLE-GPU; REAL data; commit-on-green.',
  phases: [ { title: 'Investigate AT2 D-denominator + design (GATE)' }, { title: 'Implement run_dstat + CLI + binding + build' }, { title: 'Verify bit-tight D parity + commit-or-stash' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ d59b604. CLI complete; M(py-1) bindings; standalone f4 (f4.cpp 271e302), f3 (f3.cpp cfa0d9d), f4-ratio (f4ratio.cpp d59b604) all landed + bit-tight AT2-parity (fixture-matched read_f2 goldens). ctest 51/51. They are the templates to mirror for D.',
  'GOAL: standalone D / qpDstat. D(A,B;C,D) = f4(A,B;C,D)/denom. The f4 NUMERATOR + its Z/p are FREE (steppe already has them — D sign+Z+p == f4 Z/p since the positive denom cancels in Z). The OPEN question is the NORMALIZED-D MAGNITUDE: the denom = sum over SNPs of the het/ABBA-BABA product, and steppe f2 cache (fstats.hpp stores f2 + vpair + block_sizes) discards per-SNP freqs. BUT admixtools f4(f4mode=FALSE)/qpdstat returns the normalized D FROM f2_data — so AT2 derives the denom from f2-cache-level quantities (the per-pop heterozygosities / vpair-like terms), NOT raw genotypes. FIND OUT EXACTLY HOW (verify, do not assume).',
  'THE GATE (design phase): determine feasibility. (a) FEASIBLE: D-denom derivable from steppe f2 cache (f2/vpair) or a SMALL bounded addition (e.g. a per-pop or per-pair het term computable in extract-f2 without storing per-SNP data) -> proceed. (b) STRUCTURAL: needs a genotype re-read pass or a major f2-dir FORMAT change -> HALT + DEFER (return feasible=false + the options; do NOT silently change the format). Either way report the TRUTH with file:line / the AT2 source.',
  'NO-AT2 LIFTED FOR GOLDEN GENERATION only. AT2/R + the DReichLab AdmixTools C source are on the box (/workspace/AdmixTools_src); convertf-PA at ' + PA + '. The 9 fit0 pops: England_BellBeaker, Czechia_EBA_CordedWare, Turkey_N, Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana. Golden (if feasible): admixtools D over the read_f2(9-pop maxmiss=0 == the fixture) quadruples -> golden_fit0_dstat_readf2.csv.',
  'REUSE (no dup): f4.cpp (run_f4 / assemble_f4_quartets for the numerator) + the jackknife seam (D jackknifes the per-block D = num_loo/den_loo, like the f4-ratio jackknife-of-the-ratio) + steppe::access + the CLI helpers + emitter + the nanobind module.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout/clean. On ANY failure ' + STASH + ' "wip:dstat-FAILED-<reason>" + HALT. NON-trivial/structural blocker -> STOP + report (defer). Classify minor vs bad.',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally; §4 layering.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Investigate AT2 D-denominator + design (GATE)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','at2_denom','steppe_support','golden_path','fixture_match_check','design','blocker_options','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff D-denom is derivable from the steppe f2 cache (f2/vpair) or a SMALL bounded addition (no genotype re-read, no major format change). false -> structural blocker, defer to user' },
    at2_denom: { type: 'string', description: 'EXACTLY how admixtools f4(f4mode=F)/qpdstat computes the D denominator (the formula + whether from f2_data or genotypes), cited from the admixtools R / the box AdmixTools C source (file:line)' },
    steppe_support: { type: 'string', description: 'whether steppe f2 cache (f2/vpair/block_sizes) or a bounded addition supports that denom — what exactly is/ist not available (file:line in fstats.hpp / the access lib)' },
    golden_path: { type: 'string', description: 'if feasible: golden_fit0_dstat_readf2.csv (path + sample row) pulled local; else empty' },
    fixture_match_check: { type: 'string', description: 'if feasible: PROOF the golden D matches steppe D-from-fixture to ~1e-6; else empty' },
    design: { type: 'string', description: 'if feasible: the run_dstat plan (num via assemble_f4_quartets + the denom seam + the per-block-D jackknife; CLI + binding); else empty' },
    blocker_options: { type: 'string', description: 'if NOT feasible: the options for the user (A: qpDstat=batched-f4+Z/p free; B: full-D via <the new machinery needed>; the effort/format-impact of each)' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are investigating the D-statistic denominator + designing run_dstat (verify-before-implement; NO steppe code changes). CRITICAL: VERIFY how admixtools ACTUALLY computes the normalized D denominator (it must, since f4(f4mode=FALSE)/qpdstat returns D from f2_data) — read the admixtools R source (web github uqrmaie1/admixtools, the f4/qpdstat denom) AND the DReichLab AdmixTools C on the box (/workspace/AdmixTools_src, the qpDstat/dofstats denom). Then read steppe fstats.hpp (what the f2 cache stores: f2/vpair/block_sizes) + the access lib + f4.cpp to see what is available. DETERMINE feasibility: is the D denom derivable from steppe f2-cache-level data (f2/vpair) or a SMALL bounded extract-f2 addition (NOT a genotype re-read, NOT a major format change)?', STD, '',
  'IF FEASIBLE: regen the fixture-matched golden (admixtools D over read_f2(9-pop maxmiss=0 == fixture) quadruples -> golden_fit0_dstat_readf2.csv; PROVE it matches steppe D-from-fixture to ~1e-6; pull ' + PULL + ') + produce the run_dstat design (num via assemble_f4_quartets + the denom + the per-block-D jackknife). IF NOT FEASIBLE (structural — needs a genotype pass or a format bump): set feasible=false + give blocker_options (A: qpDstat=batched-f4+Z/p free; B: full-D via the specific new machinery + its cost). Cite file:line / the AT2 source. Return the structured result; do NOT implement.',
].join('\n'), { schema: DESIGN_SCHEMA, label: 'design:dstat', phase: 'Investigate AT2 D-denominator + design (GATE)' })
if (design === null) { log('--- dstat design died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- D STRUCTURAL (defer): ' + design.at2_denom + ' || OPTIONS: ' + design.blocker_options); return { halted: true, deferred: true, design } }
log('D feasible from f2-cache: ' + String(design.fixture_match_check).slice(0,120))

phase('Implement run_dstat + CLI + binding + build')
const fixer = await tryAgent([
  'You are implementing standalone D / qpDstat per this design (MIRROR f4/f3/f4ratio), gated against the fixture-matched golden:\n<<<\nAT2 denom: ' + design.at2_denom + '\nsteppe support: ' + design.steppe_support + '\n' + design.design + '\nGOLDEN: ' + design.golden_path + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced). Work from HEAD.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT run_dstat (numerator via assemble_f4_quartets + the verified denom from the f2-cache/the bounded addition + the per-block-D jackknife for the SE; no dup), the `steppe qpdstat` CLI (reuse the flag helpers + emitter; single quadruple + batched over a quadruple list), the steppe.dstat/qpdstat binding, and the gate: a cli_dstat ctest + a pytest reproducing golden_fit0_dstat_readf2.csv at D/se/z rtol 1e-6. Build + full STEPPE_THOROUGH ctest. SANITY: cli_dstat reproduces the golden; existing goldens/cli/pytest stay green. Report every file added/changed, the reuse, the gate, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:dstat', phase: 'Implement run_dstat + CLI + binding + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for D/qpDstat. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON) + green ctest, patching only trivial -Werror / CMake wiring. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement run_dstat + CLI + binding + build' })

phase('Verify bit-tight D parity + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','dstat_parity','python_ok','golden_fixture_matched','no_duplication','goldens_green','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the golden is fixture-matched, `steppe qpdstat` reproduces golden_fit0_dstat_readf2.csv at D/se/z rtol 1e-6 (cli + pytest), the SE is the per-block-D jackknife, no dup, full ctest green, build clean, single-GPU, no synthetic' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (empty if pass)' },
    dstat_parity: { type: 'boolean' }, python_ok: { type: 'boolean' }, golden_fixture_matched: { type: 'boolean' },
    no_duplication: { type: 'boolean' }, goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the D parity + the denom approach; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for D/qpDstat (adversarial). design:\n<<<\n' + JSON.stringify(design) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) confirm golden_fit0_dstat_readf2.csv is fixture-matched (independently: D == steppe D-from-fixture to ~1e-6, using the verified AT2 denom). (2) git diff review: run_dstat reuses the f4/jackknife machinery (no dup); the denom matches the AT2 formula; the SE is the per-block-D jackknife. (3) ' + BUILD + ' — STEPPE_THOROUGH ctest green; cli_dstat + pytest reproduce the golden at D/se/z rtol 1e-6. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new golden + new/changed source/test/cmake/binding/doc (NEVER git add dot; never aadr/ atlas_results/ handoff-*.md), commit (STEP3 D/qpDstat: standalone D-statistic CLI `steppe qpdstat` + steppe.qpdstat binding [reuse f4 + the AT2 f2-cache denom + per-block-D jackknife, no dup] + fixture-matched AT2 golden -> bit-tight parity) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (D done; the cheap+core f-stat tier f4/f3/f4-ratio/D COMPLETE; qpfstats/qpGraph/DATES deferred). ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:dstat-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:dstat', phase: 'Verify bit-tight D parity + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ D/qpDstat ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- D/qpDstat FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { design, verdict }
