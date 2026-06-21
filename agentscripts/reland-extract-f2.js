export const meta = {
  name: 'reland-extract-f2',
  description: 'Re-land M(cli-4) extract-f2 PROPERLY against the CORRECT golden (the recovered code is committed at HEAD; the prior verdict reverted it only because it gated against the CORRUPT golden — AT2 v2.0.10 misreads v66 TGENO). Phase 1: regenerate the CORRECT golden_fit0 + the f2_fit0_9pop.bin fixture from the convertf-converted PACKEDANCESTRYMAP (already on the box at /workspace/data/aadr/converted_pa/v66_HO_pa) via AT2 extract_f2 — this is the proven-correct yardstick (391333 SNPs, weights ~[CW 0.869, Turkey_N 0.131], NOT the corrupt 500848/[0.559,0.441]). Phase 2: re-gate — make test_cli_extract_qpadm HARD-gate that steppe extract-f2 (on the raw v66 TGENO) -> qpadm REPRODUCES the corrected golden_fit0 within an HONEST tier (SNP count exact 391333; feasibility/f4rank match; weights within tier; and assert it is NOT the corrupt golden), fixing the fixer-demoted-to-diagnostic issue; keep qpadm_parity (#17) green against the CORRECTED fixture+golden (tight, same-f2 fit parity). Build + full ctest + adversarial verdict -> commit. HONEST: there is a residual ~0.3% steppe-vs-AT2 precompute gap (719 vs 710 jackknife blocks -> small weight/p diff) — the DECODE + SNP set are exact; surface the block-partition residual as a tracked follow-up, do NOT hide it. GPU-only; real AADR only; commit on green.',
  phases: [ { title: 'Regen golden_fit0 (convertf-PA)' }, { title: 'Re-gate extract-f2 + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const PULL = 'rsync -az -e ssh box5090:/workspace/steppe/tests/reference/goldens/at2/ ' + R + '/tests/reference/goldens/at2/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main. The recovered M(cli-4) extract-f2 code is COMMITTED at HEAD (cmd_extract_f2.{hpp,cpp}, f2_dir_writer.{hpp,cpp} [STPF2BK1 writer w/ real vpair], the CLI wiring, tests/cli/test_cli_extract_qpadm.cpp). It BUILDS clean (45/45 ctest on the box) and extract-f2 is PROVEN CORRECT.',
  'THE CONTEXT (memory aadr-tgeno-goldens-corrupt): AADR v66 .geno is TGENO; admixtools R v2.0.10 does NOT support it and silently misreads it, so the committed golden_fit0 + f2_fit0_9pop.bin fixture are CORRUPT (500848 SNPs / weights [0.559,0.441]). steppe decode is CORRECT. PROVEN via convertf v8621 (on the box): TGENO -> PACKEDANCESTRYMAP at /workspace/data/aadr/converted_pa/v66_HO_pa ; AT2 extract_f2 on THAT gives 391333 SNPs / weights ~[CordedWare 0.869, Turkey_N 0.131] — matching steppe extract-f2 (391333 / ~[0.866,0.134]) to the SNP digit. The reproducible artifacts are on the box: convertf /workspace/AdmixTools_src/src/convertf, par /workspace/data/aadr/convertf_tgeno_to_pa.par, Rscript /workspace/data/aadr/at2_on_converted.R.',
  'THE JOB: re-land extract-f2 against the CORRECT golden. (1) Regenerate the corrected golden_fit0.json + the f2_fit0_9pop.bin fixture from the convertf-PA (AT2 extract_f2 + qpadm on /workspace/data/aadr/converted_pa/v66_HO_pa, EXACT golden_fit0 model/params — READ tests/reference/goldens/at2/scripts/golden_fit0_generate.R for the pops/left/right/target/blgsize/maxmiss; the generator already does read_f2->writeBin for the fixture, just point pref at the convertf-PA prefix). (2) Re-gate the extract-f2 test to HARD-assert reproduction of the corrected golden, and keep qpadm_parity green against the corrected fixture+golden.',
  'REAL DATA ONLY (no synthetic). GPU-only. HONEST about the residual ~0.3% steppe-vs-AT2 precompute gap (719 vs 710 jackknife blocks from the block-partition rule -> small weight/p diff; the DECODE + 391333-SNP set are EXACT) — gate extract-f2 within a justified tier (SNP count exact; feasibility + f4rank match; weights rtol ~5e-3 to cover the block-partition residual; p loose) AND assert it is decisively NOT the corrupt golden (500848/[0.559]); record the block-partition residual as a tracked follow-up (do NOT widen tiers to hide a real bug, but do NOT block on the known small block-partition diff). RELEASE/-Werror/-DSTEPPE_BUILD_CLI=ON; core dumps cleared per build. ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . NOTHING builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

// PHASE 1 — regenerate the corrected golden_fit0 + fixture from convertf-PA (on the box), pull to local.
phase('Regen golden_fit0 (convertf-PA)')
const REGEN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['done','snps','weights','p','golden_pulled','note'],
  properties: {
    done: { type: 'boolean' }, snps: { type: 'integer', description: 'SNP count from AT2-on-convertf-PA (expect ~391333)' },
    weights: { type: 'string', description: 'the corrected weights [CordedWare, Turkey_N] (expect ~[0.869,0.131])' },
    p: { type: 'string' },
    golden_pulled: { type: 'boolean', description: 'the corrected golden_fit0.json + f2_fit0_9pop.bin fixture regenerated AND pulled to local ' + R + '/tests/reference/goldens/at2/' },
    note: { type: 'string', description: 'block count + the steppe-vs-AT2 residual + any issue' },
  },
}
const regen = await tryAgent([
  'You are regenerating the CORRECT golden_fit0 for steppe from the convertf-converted PACKEDANCESTRYMAP (AT2 v2.0.10 cannot read the raw v66 TGENO — that is why the current golden is corrupt). On the box (' + SSH + '), READ tests/reference/goldens/at2/scripts/golden_fit0_generate.R for the EXACT pops/left/right/target/blgsize/maxmiss, then run that SAME generation but with pref = /workspace/data/aadr/converted_pa/v66_HO_pa (the convertf-PA, which AT2 reads correctly): extract_f2 -> qpadm -> write the corrected golden_fit0.json AND the f2_fit0_9pop.bin fixture (the generator already does read_f2 -> writeBin; just retarget pref). Capture the corrected weights/se/z/p/chisq/dof/rankdrop/popdrop + SNP count + block count. Then rsync the corrected golden_fit0.json + fixtures/f2_fit0_9pop.bin back to local: ' + PULL, STD, '',
  'Sanity: the corrected golden MUST be ~391333 SNPs, weights ~[CordedWare 0.869, Turkey_N 0.131] (NOT 500848 / [0.559,0.441] = the corrupt one). If it comes out corrupt-looking, STOP and report (something is wrong with the convertf-PA path). Return the structured result incl. whether the corrected golden + fixture were pulled to local. Note the block count (expect ~710 from AT2) for the residual discussion. REAL DATA ONLY; detach long R steps + poll.',
].join('\n'), { schema: REGEN_SCHEMA, label: 'regen:golden_fit0', phase: 'Regen golden_fit0 (convertf-PA)' })

if (!regen || !regen.done || !regen.golden_pulled) {
  log('HALT: corrected golden_fit0 regen/pull failed — ' + (regen ? regen.note : 'agent died') + '. extract-f2 stays committed (7e79a79) but un-re-gated.')
  return { halted: true, reason: 'golden regen failed', regen }
}
log('corrected golden_fit0: ' + regen.snps + ' SNPs, weights ' + regen.weights + ' (was corrupt 500848/[0.559,0.441]) — pulled to local')

// PHASE 2 — re-gate extract-f2 + qpadm_parity against the corrected golden, build, verdict, commit.
phase('Re-gate extract-f2 + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','golden_corrected','extract_reproduces_golden','parity_green','no_synthetic','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: the corrected golden_fit0 is committed (391333 SNPs, ~[0.869,0.131]); the extract-f2 test now HARD-gates reproduction of the corrected golden (not demoted to diagnostic, not the corrupt golden); qpadm_parity green against the corrected fixture+golden; full ctest green; Release build clean; NO synthetic' },
    golden_corrected: { type: 'boolean', description: 'golden_fit0.json + f2_fit0 fixture are the corrected convertf-PA ones (not the corrupt TGENO-misread)' },
    extract_reproduces_golden: { type: 'boolean', description: 'steppe extract-f2 on raw v66 TGENO -> qpadm reproduces the corrected golden within the justified tier (SNP exact, weights rtol ~5e-3, feasibility/f4rank match) AND is decisively NOT the corrupt 500848/[0.559]' },
    parity_green: { type: 'boolean' }, no_synthetic: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed + the corrected golden numbers + the e2e extract-f2 result + the residual block-partition gap (719 vs 710); for FAIL exactly what blocked it' },
  },
}
const fix = await tryAgent([
  'You are re-gating the (committed, recovered) M(cli-4) extract-f2 against the now-CORRECTED golden_fit0 for steppe. Do NOT commit (the verdict commits). Start clean at HEAD: ' + CLEAN + ' (HEAD already has the recovered extract-f2 code + the corrected golden_fit0/fixture pulled in phase 1 — confirm `git status` shows the corrected golden + fixture as modified).', STD, '',
  'THE CORRECTED GOLDEN (phase 1): ' + regen.snps + ' SNPs, weights ' + regen.weights + ', p ' + regen.p + '. ' + regen.note, '',
  'DO: (1) Update tests/cli/test_cli_extract_qpadm.cpp so it HARD-GATES (fails ctest on mismatch, not a non-gating diagnostic) that steppe `extract-f2` on the raw v66 TGENO (/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB) for golden_fit0 model -> `qpadm` reproduces the CORRECTED golden_fit0 within the justified tier (SNP count EXACT 391333; f4rank + feasibility match; weights rtol ~5e-3 [covers the known ~0.3% block-partition residual]; p loose) AND assert it is decisively NOT the corrupt 500848-SNP/[0.559,0.441] result. (2) Confirm qpadm_parity (#17, the CpuBackend/GPU fit on the f2_fit0 fixture) passes against the CORRECTED fixture+golden (tight — same-f2 fit parity). (3) Build + full ctest on the box (' + BUILD + '). Report: the corrected golden values, the e2e extract-f2 result, the residual block-partition gap (steppe 719 vs AT2 710 blocks), the FULL ctest. Do NOT commit. NO synthetic data; do NOT widen tiers to mask a real divergence (the ~0.3% block-partition residual is the only allowed slack and must be NOTED as a follow-up).',
].join('\n'), { label: 'fix:reland', phase: 'Re-gate extract-f2 + commit' })
if (fix === null) { log('--- reland fixer died — HALT'); return { halted: true, reason: 'fixer died', regen } }

const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT re-landing extract-f2 for steppe (adversarial). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager status — confirm the corrected golden_fit0.json + f2_fit0_9pop.bin fixture (391333 SNPs / ~[0.869,0.131], NOT the corrupt 500848/[0.559]) and the test now HARD-gates the golden reproduction (search the test for the assertion — it must FAIL ctest on mismatch, not just print a diagnostic). (2) RE-RUN ' + BUILD + ' . (3) PASS only if ALL: corrected golden committed; extract-f2 e2e reproduces it within the justified tier AND is NOT the corrupt golden; qpadm_parity green vs corrected fixture; full ctest green; Release build clean; NO synthetic. ',
  'ON PASS: cd ' + R + ' && git add ONLY the corrected golden_fit0.json + tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin + the changed test/source (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (re-land M(cli-4) extract-f2 against the corrected convertf-PA golden_fit0; the corrupt-TGENO golden replaced; note the residual block-partition gap as a follow-up) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.',
  'ON FAIL: ' + CLEAN + ' (note: HEAD already has the recovered extract-f2 code committed at 7e79a79 — a clean does NOT lose it; it only drops the uncommitted golden/test edits) and report exactly what blocked it. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:reland', phase: 'Re-gate extract-f2 + commit' })

if (verdict === null) { log('--- reland verdict died — HALT (extract-f2 code stays committed at 7e79a79)'); return { halted: true, reason: 'verdict died', regen } }
if (verdict.pass) log('+++ extract-f2 RE-LANDED ' + verdict.commit_hash + ' against the corrected golden — ' + verdict.note)
else log('--- reland FAILED (' + verdict.note + ') — extract-f2 code stays committed at 7e79a79; golden/gate not finalized')
return { regen, verdict }
