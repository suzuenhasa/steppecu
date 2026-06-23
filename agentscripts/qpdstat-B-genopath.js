export const meta = {
  name: 'qpdstat-B-genopath',
  description: 'STEP 3 stat #4 PART B (decision (i), D-specific genotype path): `steppe qpdstat --prefix <geno>` -> the full NORMALIZED-D magnitude + se + z + p. The normalized-D denom Sum_snp het(p1,p2)*het(p3,p4) needs per-SNP genotypes (VERIFIED genotype-only in AT2 too). So this is a NEW genotype-reading stat path that REUSES the extract-f2 decode front-end (io read + decode_af + assign_blocks + SNP-tile streaming) but DIVERGES at S2 into a per-SNP D kernel (num=(a-b)(c-d), denom=(a+b-2ab)(c+d-2cd), accumulated per block, per quadruple) + a num/den block-jackknife (D=sum(num)/sum(den); per-block loo_num/loo_den, like the f4-ratio jackknife-of-the-ratio); it NEVER touches the f2 cache. Plus a SMALL clearly-shared DATES down-payment: surface per-SNP genpos (snp_reader.hpp already retains genpos_morgans) through the genotype-stat path (D ignores it; DATES needs it). Decision (i): D-specific, separable kernel — do NOT build a speculative kernel framework. THE RISK = genotype-path SNP-FILTERING PARITY (which SNPs each quadruple uses: per-quadruple maximal overlap vs allsnps; the same class of trap as the f2 allsnps 1.1132x issue). The DESIGN PHASE GATES: verify AT2 genotype-path qpdstat (cpp_fstats.cpp dstatnum/dstatden + the SNP-overlap handling; DReichLab qpDstat.c) + match steppe decode filtering to it, generate the genotype-path golden, PROVE parity achievable; if the filtering CANNOT be matched to a tight tier -> HALT + DEFER. Ref docs/research/dates-genotype-stat-seam.md. FAIL-PROTOCOL (user-mandated): NEVER git checkout/clean — on failure git stash push -u + HALT; verdict classifies severity (minor=move-on / bad=defer). SINGLE-GPU; REAL data; commit-on-green.',
  phases: [ { title: 'Verify AT2 geno-path D + SNP-filtering parity + golden (GATE)' }, { title: 'Implement run_dstat (geno path) + CLI + binding + build' }, { title: 'Verify normalized-D parity + commit-or-stash' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ de38b7e. CLI: extract-f2/qpadm/qpwave/qpadm-rotate/f4/f3/f4-ratio/qpdstat(--f2-dir, Part A a565ca0) all wired + golden-gated. M(py-1) bindings landed. ctest 52/52.',
  'GOAL (Part B, decision (i)): `steppe qpdstat --prefix <geno>` -> the NORMALIZED-D magnitude + se + z + p, a NEW genotype-reading stat path. D = mean_snp(num)/mean_snp(den), num=(a-b)(c-d), den=(a+b-2ab)(c+d-2cd) (a,b,c,d = per-SNP allele freqs of p1,p2,p3,p4); SE = block-jackknife of D (per-block loo_num/loo_den, the f4-ratio jackknife-of-the-ratio shape). Replace the current --prefix Part-B sentinel (cmd_qpdstat.cpp) with the real path.',
  'ARCHITECTURE (from docs/research/dates-genotype-stat-seam.md): REUSE the extract-f2 decode FRONT-END (the io reader + decode_af [per-SNP Q/V/N + per-sample ploidy] + assign_blocks [from genpos] + SNP-tile streaming); DIVERGE at S2 into the per-SNP D kernel (per block, per quadruple) + the num/den jackknife; NEVER read/write the f2 cache. SINGLE-GPU device-resident; SNP-tile streamed (memory-bounded like extract-f2). DATES down-payment: surface per-SNP genpos (snp_reader.hpp:41 genpos_morgans already retained) through the genotype-stat path (D ignores it; comment it as the DATES seam).',
  'THE PARITY RISK (gate it): genotype-path qpdstat SNP filtering. admixtools qpdstat_geno + DReichLab qpDstat.c pick, per quadruple, a SNP set (per-quadruple maximal overlap when allsnps=FALSE, or the global set when allsnps=TRUE). steppe decode must use the MATCHING filtering for bit-tight parity (this is the same class of trap as the f2 allsnps 1.1132x normalization). VERIFY the exact AT2 behavior + match it; if it cannot be matched to a tight tier, HALT + DEFER (report the achievable tier + why).',
  'GOLDEN (no-AT2 lifted for GENERATION only): on box5090, subset the convertf-PA ' + PA + ' to the 9 fit0 pops (England_BellBeaker, Czechia_EBA_CordedWare, Turkey_N, Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana), run admixtools::qpdstat(<that geno prefix>, the quadruples from golden_fit0_qpdstat_readf2.csv, f4mode=FALSE) -> golden_fit0_dstat_geno.csv (D/se/z, the NORMALIZED D, distinct from the f2-path f4 golden), with the SNP-filtering matching steppe decode. The genotypes that produced f2_fit0_9pop.bin are this 9-pop convertf-PA subset.',
  'REUSE (verify file:line, no dup): the decode front-end (decode_af + the io reader + assign_blocks + the SNP-tile streaming used by extract-f2 / compute_f2_blocks); the jackknife framework (the f4ratio.cpp ratio_jackknife / jackknife_cov seam); cmd_qpdstat.cpp (the Part-A command, add the --prefix branch); the binding + emitter; steppe::access (pop resolution). New = the per-SNP D kernel + the num/den jackknife + the genpos surfacing.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout/clean. On ANY failure ' + STASH + ' "wip:qpdstatB-FAILED-<reason>" + HALT. NON-trivial/structural blocker (esp. the SNP-filtering parity) -> STOP + report (defer). Classify minor vs bad.',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally; §4 layering (cmd/binding TUs CUDA-free; the kernel in a .cu).',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Verify AT2 geno-path D + SNP-filtering parity + golden (GATE)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','at2_snp_filtering','steppe_match_plan','golden_path','parity_check','design','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff the genotype-path D SNP-filtering can be matched between steppe decode and AT2 qpdstat_geno to a tight tier (rtol <= ~1e-4, ideally 1e-6). false -> structural parity gap, defer' },
    at2_snp_filtering: { type: 'string', description: 'EXACTLY how AT2 genotype-path qpdstat picks the per-quadruple SNP set (allsnps default, maxmiss, per-quadruple overlap), cited (cpp_fstats.cpp / qpDstat.c file:line) + the num/den formulas confirmed' },
    steppe_match_plan: { type: 'string', description: 'how steppe decode (maxmiss / per-quadruple overlap) will match it; the achievable tier' },
    golden_path: { type: 'string', description: 'if feasible: golden_fit0_dstat_geno.csv (path + sample row, the NORMALIZED D distinct from the f4 golden) pulled local; else empty' },
    parity_check: { type: 'string', description: 'if feasible: PROOF D-from-the-9pop-genotypes (a from-genotypes recompute) matches the AT2 golden to the achievable tier; else empty' },
    design: { type: 'string', description: 'if feasible: the run_dstat plan (decode front-end reuse + the D kernel + the num/den jackknife + the genpos surfacing + the CLI --prefix branch + binding); else empty' },
    blocker: { type: 'string', description: 'if NOT feasible: the SNP-filtering parity gap + the options for the user' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are verifying the AT2 genotype-path D + the SNP-filtering parity, and designing run_dstat (verify-before-implement; NO steppe code changes). (1) VERIFY how admixtools genotype-path qpdstat computes the normalized D: the num/den formulas (cpp_fstats.cpp dstatnum/dstatden) AND the per-quadruple SNP-set selection (allsnps default? maxmiss? per-quadruple maximal overlap?) — read the admixtools R qpdstat_geno/f4blockdat_from_geno + the DReichLab /workspace/AdmixTools_src/src/qpDstat.c. (2) Read the steppe decode front-end (decode_af, the io reader, assign_blocks, the SNP-tile streaming in the f2 path) + cmd_qpdstat.cpp (the Part-A command) + f4ratio.cpp (the jackknife-of-the-ratio) + snp_reader.hpp (genpos_morgans). (3) DETERMINE whether steppe decode can MATCH AT2 genotype-path qpdstat SNP-filtering to a tight tier.', STD, '',
  'IF FEASIBLE: on box5090 subset the convertf-PA to the 9 fit0 pops, run admixtools::qpdstat(that_geno, the quadruples, f4mode=FALSE) -> golden_fit0_dstat_geno.csv (the NORMALIZED D, distinct from the f4 golden); PROVE a from-genotypes recompute of D matches it to the achievable tier; pull ' + PULL + ' ; produce the run_dstat design (decode reuse + D kernel + num/den jackknife + genpos surfacing + the --prefix CLI branch + binding). IF NOT FEASIBLE (the SNP-filtering parity cannot be matched): set feasible=false + blocker (the gap + options). Cite file:line. Return the structured result; do NOT implement.',
].join('\n'), { schema: DESIGN_SCHEMA, label: 'design:qpdstatB', phase: 'Verify AT2 geno-path D + SNP-filtering parity + golden (GATE)' })
if (design === null) { log('--- qpdstatB design died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- qpDstat Part B STRUCTURAL (defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('Part B feasible — geno-path D parity: ' + String(design.parity_check).slice(0,120))

phase('Implement run_dstat (geno path) + CLI + binding + build')
const fixer = await tryAgent([
  'You are implementing qpDstat Part B (the genotype-path normalized-D) per this design:\n<<<\nAT2 filtering: ' + design.at2_snp_filtering + '\nmatch plan: ' + design.steppe_match_plan + '\n' + design.design + '\nGOLDEN: ' + design.golden_path + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced). Work from HEAD.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT run_dstat (the genotype path: reuse the decode front-end [decode_af + io + assign_blocks + SNP-tile streaming], the per-SNP D kernel [num+denom per block per quadruple], the num/den block-jackknife; surface per-SNP genpos as the DATES down-payment; no dup of decode/jackknife), wire `steppe qpdstat --prefix <geno>` (replace the Part-B sentinel; --f2-dir Part A unchanged), the steppe.qpdstat --prefix binding, and the gate: a cli_dstat_geno ctest + a pytest reproducing golden_fit0_dstat_geno.csv at D/se/z the achievable tier (target rtol 1e-6). Build + full STEPPE_THOROUGH ctest. SANITY: cli_dstat_geno reproduces the normalized-D golden; Part A (--f2-dir) + all existing goldens/cli/pytest stay green. Report every file added/changed, the reuse, the gate result, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:qpdstatB', phase: 'Implement run_dstat (geno path) + CLI + binding + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for qpDstat Part B (the genotype-path D). Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON) + green ctest, patching only trivial -Werror / CMake wiring of the new kernel + cmd branch + tests. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement run_dstat (geno path) + CLI + binding + build' })

phase('Verify normalized-D parity + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','normalized_d_parity','python_ok','reuses_decode_frontend','genpos_surfaced','no_f2_cache','no_duplication','partA_intact','goldens_green','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if `steppe qpdstat --prefix` reproduces golden_fit0_dstat_geno.csv (the NORMALIZED D) at D/se/z the achievable tier (target rtol 1e-6) via cli + pytest, reuses the decode front-end (no dup), never touches the f2 cache, Part A intact, full ctest green, build clean, single-GPU, no synthetic' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (empty if pass)' },
    normalized_d_parity: { type: 'boolean' }, python_ok: { type: 'boolean' },
    reuses_decode_frontend: { type: 'boolean', description: 'run_dstat reuses decode_af + io + assign_blocks (no copied decode)' },
    genpos_surfaced: { type: 'boolean', description: 'per-SNP genpos surfaced through the genotype-stat path (the DATES down-payment)' },
    no_f2_cache: { type: 'boolean', description: 'the genotype path never reads/writes the f2 cache' },
    no_duplication: { type: 'boolean' }, partA_intact: { type: 'boolean', description: 'qpdstat --f2-dir (Part A) unchanged + still green' },
    goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the normalized-D parity tier + the SNP-filtering match + the seam established; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for qpDstat Part B (adversarial). design:\n<<<\n' + JSON.stringify(design) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) confirm golden_fit0_dstat_geno.csv is the NORMALIZED D (distinct from the f4/f2-path golden) + fixture-consistent (independently recompute D from the 9-pop genotypes to the achievable tier). (2) git diff review: run_dstat REUSES the decode front-end (no copied decode), NEVER reads the f2 cache, the num/den jackknife matches AT2, genpos surfaced, Part A (--f2-dir) untouched. (3) ' + BUILD + ' — STEPPE_THOROUGH ctest green; cli_dstat_geno + pytest reproduce the normalized-D golden at D/se/z the tier; Part A still green. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new golden + new/changed source/test/cmake/binding/doc (NEVER git add dot; never aadr/ atlas_results/ handoff-*.md), commit (STEP3 qpDstat Part B: `steppe qpdstat --prefix` = the genotype-path NORMALIZED-D magnitude [decode front-end reuse + per-SNP D kernel + num/den jackknife, no f2 cache] + the genotype-stat seam + per-SNP genpos down-payment for DATES; gated vs AT2 genotype-path qpdstat) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (qpDstat A+B COMPLETE; the genotype-stat seam established; DATES reuses it; qpfstats/qpGraph remain deferred). ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:qpdstatB-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:qpdstatB', phase: 'Verify normalized-D parity + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ qpDstat Part B ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- qpDstat Part B FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { design, verdict }
