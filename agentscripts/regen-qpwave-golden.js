export const meta = {
  name: 'regen-qpwave-golden',
  description: 'Regenerate the LAST corrupt golden — golden_qpwave — from the convertf-PA (the other 4 goldens are now correct; qpwave was pinned in 6481dfa on the RAW v66 TGENO that AT2 MISREADS, dataset field still "v66.p1_HO.aadr.patch.PUB" / geno_sha256 7af8c2f5 != the convertf-PA e588406, so it is the same corrupt TGENO-misread; its test passes only green-on-garbage). steppe now has full AT2 parity (main @ cb4d19c), so the regenerated qpwave golden will be correct AND steppe run_qpwave reproduces it. Phase 1 REGEN (box, R/AT2 on /workspace/data/aadr/converted_pa/v66_HO_pa): READ tests/reference/goldens/at2/scripts/golden_qpwave_generate.R for the EXACT left/right/fudge(1e-4)/seed(42)/blgsize(0.05)/params, re-run with pref=convertf-PA (extract_f2 + qpwave(), adjust_pseudohaploid default TRUE) -> regenerate golden_qpwave.json (+ any f2 fixture it uses; if it shares the fit0 9-pop f2, reuse the already-corrected one), pull to local. Phase 2 RE-GATE: build + STEPPE_THOROUGH ctest -> steppe run_qpwave (both backends) must reproduce the corrected golden_qpwave at the existing tier; commit. regen -> verdict(build+validate+commit); HALT-on-fail; resumable on 529. REAL data; do NOT touch the other 4 goldens; NO synthetic; NO tier-widening to mask a miss.',
  phases: [ { title: 'Regen qpwave (convertf-PA)' }, { title: 'Re-gate + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const PULL = 'rsync -az -e ssh box5090:/workspace/steppe/tests/reference/goldens/at2/ ' + R + '/tests/reference/goldens/at2/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -50; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'
const PA = '/workspace/data/aadr/converted_pa/v66_HO_pa'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm/qpWave. Branch phase2-fit-engine == main @ cb4d19c, FULL AT2 parity (4-fix chain). The goldens golden_fit0/rot/fit1_NRBIG/fitNA are now CORRECT (regenerated from the convertf-PA). golden_qpwave is the ONE remaining CORRUPT golden — pinned (6481dfa) by extract_f2+qpwave on the RAW v66 TGENO which admixtools R v2.0.10 MISREADS (memory aadr-tgeno-goldens-corrupt). Regenerate it from the convertf-PACKEDANCESTRYMAP that AT2 reads correctly.',
  'THE FIX: AT2 extract_f2 + qpwave on ' + PA + '.{geno,snp,ind} (convertf v8621 conversion of the v66 TGENO). READ tests/reference/goldens/at2/scripts/golden_qpwave_generate.R for the EXACT left/right pop lists, fudge (1e-4), set.seed(42), blgsize (0.05 Morgans), maxmiss, the f2-dir-path canonical form (same caveat as golden_fit0.json). Re-run with pref=' + PA + ' (NOT the raw TGENO), adjust_pseudohaploid default TRUE -> regenerate golden_qpwave.json (the qpwave rank/chisq/p/loglik numbers) + any f2 fixture it uses (if it reads the fit0 9-pop f2 dir, that fixture f2_fit0_9pop.bin is ALREADY corrected — reuse it; if qpwave has its own f2 fixture, regenerate it). Pull corrected to local: ' + PULL,
  'VALIDATION: steppe run_qpwave (the C++ entry, both CpuBackend + CudaBackend) reads the corrected f2 and must reproduce the corrected golden_qpwave at the EXISTING tier (test_qpwave_parity.cu). Do NOT touch golden_fit0/rot/fit1_NRBIG/fitNA. Do NOT widen the qpwave tier to mask a mismatch — steppe parity is proven, so it should match; if it does not, report the exact residual.',
  'REAL DATA ONLY; GPU-only for steppe; AT2 R for the golden. Box ' + SSH + '; nvcc -> ' + PATHENV + '; RELEASE -DSTEPPE_BUILD_CLI=ON; nothing builds locally; core dumps cleared per build; detach long R/extract_f2 + poll (529-flaky). Standards: the existing golden-test pattern.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Regen qpwave (convertf-PA)')
const REGEN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['done','qpwave_numbers','reused_or_regen_fixture','pulled','note'],
  properties: {
    done: { type: 'boolean' },
    qpwave_numbers: { type: 'string', description: 'the regenerated golden_qpwave key numbers (rank/chisq/dof/p per the rank-sequence) vs the old corrupt values' },
    reused_or_regen_fixture: { type: 'string', description: 'whether qpwave reuses the corrected fit0 9-pop f2 fixture or got its own regenerated f2 fixture' },
    pulled: { type: 'boolean', description: 'corrected golden_qpwave.json (+ fixture if new) pulled to local' },
    note: { type: 'string' },
  },
}
const regen = await tryAgent([
  'You are regenerating the last corrupt steppe golden (golden_qpwave) from the convertf-PA. On the box (' + SSH + '), READ tests/reference/goldens/at2/scripts/golden_qpwave_generate.R for the EXACT params, then re-run with pref=' + PA + ' (extract_f2 blgsize 0.05 + qpwave() fudge 1e-4 set.seed(42) adjust_pseudohaploid TRUE) -> regenerate golden_qpwave.json + any f2 fixture it needs (reuse the corrected f2_fit0_9pop.bin if qpwave shares the 9-pop f2 dir). Then PULL to local: ' + PULL, STD, '',
  'Sanity: the regenerated qpwave numbers must differ from the committed corrupt ones (different SNP set than the TGENO-misread) + be statistically sensible. Detach long extract_f2 + poll. Return the structured result. Do NOT touch the other 4 goldens. REAL DATA; no synthetic.',
].join('\n'), { schema: REGEN_SCHEMA, label: 'regen:qpwave', phase: 'Regen qpwave (convertf-PA)' })
if (!regen || !regen.done || !regen.pulled) { log('HALT: qpwave regen/pull failed — ' + (regen ? regen.note : 'agent died')); return { halted: true, regen } }
log('qpwave golden regenerated from convertf-PA + pulled')

phase('Re-gate + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','qpwave_corrected','steppe_reproduces','thorough_green','others_untouched','no_synthetic','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if golden_qpwave is the corrected convertf-PA one (not the corrupt TGENO-misread), steppe run_qpwave (both backends) reproduces it at the existing tier, STEPPE_THOROUGH ctest green, the other 4 goldens untouched, build clean, no synthetic, no tier-widening to mask a miss' },
    qpwave_corrected: { type: 'boolean' }, steppe_reproduces: { type: 'boolean' },
    thorough_green: { type: 'boolean' }, others_untouched: { type: 'boolean', description: 'golden_fit0/rot/fit1_NRBIG/fitNA NOT touched' },
    no_synthetic: { type: 'boolean' }, build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'corrected qpwave numbers + steppe-vs-corrected (both backends); for FAIL the exact residual' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the qpwave golden regen (adversarial). The regen agent reported: qpwave=' + regen.qpwave_numbers + ' | fixture=' + regen.reused_or_regen_fixture + ' | ' + regen.note, STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager status + diff — confirm golden_qpwave.json (+ fixture if new) is the corrected convertf-PA one (dataset now the convertf-PA, differs from the corrupt TGENO-misread); the OTHER 4 goldens UNTOUCHED; test_qpwave_parity.cu not tier-widened to mask. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest green; steppe run_qpwave (Cpu + Cuda) reproduces the corrected golden_qpwave at the existing tier. (3) PASS only if: qpwave corrected, steppe reproduces (both backends), thorough ctest green, other 4 untouched, build clean, no synthetic, no tier-masking. ',
  'ON PASS: cd ' + R + ' && git add ONLY golden_qpwave.json (+ its fixture if regenerated) + any test file changed (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (regenerate golden_qpwave from convertf-PA: the LAST corrupt TGENO-misread golden replaced; steppe run_qpwave reproduces at tier; ALL 5 goldens now correct = complete AT2-parity suite) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.',
  'ON FAIL: ' + CLEAN + ' and report the exact residual. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:qpwave', phase: 'Re-gate + commit' })
if (verdict === null) { log('--- qpwave verdict died — HALT'); return { halted: true, regen } }
if (verdict.pass) log('+++ ALL 5 GOLDENS CORRECT ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- FAILED (' + verdict.note + ')')
return { verdict }
