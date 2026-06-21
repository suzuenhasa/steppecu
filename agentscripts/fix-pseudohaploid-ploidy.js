export const meta = {
  name: 'fix-pseudohaploid-ploidy',
  description: 'Fix the steppe pseudo-haploid f2 bug (diagnosis: docs/research/f2-estimator-at2.md). steppe HARD-CODES ploidy=2 for every sample (cmd_extract_f2.cpp:60 -> decode_af.hpp:142-156 finalize_af N=2n), so the het bias-correction denominator is 2n-1; AT2 (adjust_pseudohaploid=TRUE, cpp_readgeno.cpp:137-168 / R io.R:155) AUTO-DETECTS pseudo-haploid PER SAMPLE and uses n-1. Result: f2 EXACT on diploid moderns but up to 230% wrong on pseudo-haploid ancient pairs (15.6% median on the all-ancient set) — it only cancels in qpAdm f4 contrasts. FIX: per-sample pseudo-haploid auto-detection (a sample with NO heterozygous calls -> ploidy 1) feeding finalize_af N (allele count + het denominator); PER-SAMPLE not per-pop (mixed-ploidy pops: Turkey_N/Serbia/Yamnaya/Karitiana). Default AUTO (match AT2); a --ploidy auto|1|2 override. No f2 formula change. BLAST RADIUS: the M1 decode oracle (build_tgeno_matrix.py / decode_equivalence) likely encodes ploidy=2 too -> update it to the same per-sample detection so it stays valid AND becomes AT2-correct; the AT2-fixture fit goldens are unaffected (read, not recomputed); the extract-f2 e2e tightens. VALIDATION (real data, no AT2 re-run beyond the saved/compare): steppe f2 on the all-pseudo-haploid-ancient set now matches AT2 (the 15.6%/230% gap -> ~0 / FP floor); modern-diploid STAYS exact; the Haak qpAdm tightens further; decode_equivalence + all goldens green; build clean. fixer -> build-repair -> adversarial verdict -> commit; HALT-on-fail. GPU-only; REAL data.',
  phases: [ { title: 'Implement per-sample pseudo-haploid' }, { title: 'Validate f2 parity across regimes + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'
const RAW = '/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB'
const PA = '/workspace/data/aadr/converted_pa/v66_HO_pa'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; parity target = ADMIXTOOLS (DReichLab C + admixtools R 2.0.10). Branch phase2-fit-engine == main @ d39740e. THE DIAGNOSIS (docs/research/f2-estimator-at2.md — READ IT): steppe hard-codes ploidy=2 -> the het bias-correction denominator (q(1-q)/max(N-1,1)) uses N=2n for ALL samples; AT2 adjust_pseudohaploid=TRUE auto-detects pseudo-haploid per sample (no het calls => haploid) and uses N=n. f2 EXACT on diploid, up to 230% off on pseudo-haploid pairs (15.6% median on the all-ancient set), cancels to +0.29% in Haak qpAdm f4 contrasts. This is a real systematic bug for pseudo-haploid (the dominant aDNA type).',
  'THE FIX: (1) per-SAMPLE pseudo-haploid auto-detection — a sample is pseudo-haploid (ploidy 1) iff it has NO heterozygous genotype calls (match AT2 cpp_readgeno.cpp:137-168; verify the exact rule: all SNPs vs a prefix; AT2 scans for any het). Detect in steppe decode (geno_reader / decode path), per sample, store per-sample ploidy. (2) feed per-sample ploidy into decode_af finalize_af N (the allele count + the het denominator: diploid N=2*nonmissing, pseudo-haploid N=1*nonmissing). PER-SAMPLE, not per-pop (mixed-ploidy pops exist). (3) cmd_extract_f2.cpp:60 — remove the kPloidy=2 hardcode; default AUTO-detect (match AT2); add --ploidy auto|1|2 override. Do NOT change the f2 FORMULA (Σp^2-2Σpq+Σq^2 + the correction); only N.',
  'BLAST RADIUS: the M1 decode oracle (the on-box build_tgeno_matrix.py + the decode_equivalence test tests/.../decode*) likely also assumes ploidy=2. If decode_equivalence checks N (it likely does — N is a decode output), UPDATE build_tgeno_matrix.py to the SAME per-sample pseudo-haploid detection so decode_equivalence stays valid AND becomes AT2-correct (do NOT just relax the test). The AT2-generated FIT goldens/fixtures (golden_*.json, fixtures/*.bin) are READ, never recomputed -> unaffected by this decode change -> the qpadm_parity/rotation/qpwave fit tests stay green. The extract-f2 e2e (test_cli_extract_qpadm) should TIGHTEN.',
  'VALIDATION (real data; the parity proof): after the fix, on the box compute steppe f2 vs AT2 f2 (convertf-PA) on (1) the ALL-PSEUDO-HAPLOID-ANCIENT set (Russia_Samara_EBA_Yamnaya, Czechia_EBA_CordedWare, Turkey_N, Serbia_IronGates_Mesolithic, Israel_Natufian, Iran_GanjDareh_N) -> the prior 15.6% median / 230% max gap must COLLAPSE to ~0 (FP/emulated floor); (2) the MODERN-DIPLOID set (Han, French, Sardinian, Papuan, Karitiana, Mbuti) must STAY exact (0%); (3) the Haak qpAdm models tighten toward the saved AT2 reference (docs/studies/haak2015-at2-reference.md). decode_equivalence + the full ctest green. (Comparing steppe f2 vs the existing AT2 f2 dirs on the box is fine; do not need a fresh AT2 re-run if the f2 dirs exist, else a quick AT2 extract_f2 on PA for the ancient set is OK.)',
  'REAL DATA ONLY; GPU-only. Box ' + SSH + '; nvcc -> ' + PATHENV + '; RELEASE -DSTEPPE_BUILD_CLI=ON; nothing builds locally; core dumps cleared per build. Standards: NAMING-STYLE-STANDARD + architecture (decode_af / the io->decode seam, §4 layering). Verify the AT2 detection rule against the C/R source before coding.',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+ctest (' + BUILD + '). Do NOT commit (the verdict commits). NO synthetic. MEASURE the f2 diffs on the box.'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Implement per-sample pseudo-haploid')
const fixer = [
  'You are a senior engineer fixing the steppe pseudo-haploid f2 bug (a core decode-path correctness fix). Do NOT commit. Start clean: ' + CLEAN + '. READ docs/research/f2-estimator-at2.md (the diagnosis) + decode_af.hpp (finalize_af) + the geno decode path + cmd_extract_f2.cpp.', STD, '', DEVLOOP, '',
  'STEP 1 verify AT2 detection rule: read the AT2 C cpp_readgeno.cpp:137-168 (on the box /workspace/AdmixTools_src/src or the admixtools R src) — confirm EXACTLY when AT2 marks a sample pseudo-haploid (any het call across which SNPs? the exact scan). STEP 2 implement per-SAMPLE pseudo-haploid auto-detection in steppe (scan each sample for any het genotype -> ploidy 1 else 2), store per-sample ploidy, and feed it into finalize_af N (per-sample: diploid contributes 2 alleles, pseudo-haploid 1; the het correction denominator uses the resulting N). PER-SAMPLE not per-pop. STEP 3 cmd_extract_f2.cpp:60: default AUTO (match AT2), add --ploidy auto|1|2. STEP 4 the M1 oracle: check if decode_equivalence / build_tgeno_matrix.py assumes ploidy=2 in N; if so update the oracle to the SAME per-sample detection so the test stays valid + AT2-correct (do NOT relax it). Do NOT change the f2 formula. Build + full ctest. Report every file changed, the detection rule, and the FULL ctest. Do NOT commit. If decode_equivalence breaks in a way the oracle update cannot fix, STOP + report.',
].join('\n')
const fix = await tryAgent(fixer, { label: 'fix:pseudohaploid', phase: 'Implement per-sample pseudo-haploid' })
if (fix === null) { log('--- fixer died — HALT'); return { halted: true } }

await tryAgent(['BUILD-REPAIR for the pseudo-haploid ploidy fix. Accumulated edits (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON), patching only trivial -Werror. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement per-sample pseudo-haploid' })

phase('Validate f2 parity across regimes + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','pseudohaploid_collapsed','modern_still_exact','haak_tightened','decode_equiv_green','goldens_green','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: per-sample pseudo-haploid auto-detect implemented (not per-pop); steppe f2 on the ALL-PSEUDO-HAPLOID-ANCIENT set now matches AT2 (the prior 15.6%/230% gap collapsed to ~0/FP floor); MODERN-DIPLOID stays exact; Haak qpAdm tightened; decode_equivalence green (oracle updated, not relaxed); full ctest green; Release build clean; no synthetic' },
    pseudohaploid_collapsed: { type: 'boolean', description: 'steppe-vs-AT2 f2 on the ancient pseudo-haploid set dropped from ~15.6% median/230% max to ~0 (FP floor)' },
    modern_still_exact: { type: 'boolean', description: 'modern-diploid f2 still 0% vs AT2 (the fix did not regress diploid)' },
    haak_tightened: { type: 'boolean', description: 'the 3 Haak qpAdm models moved closer to the saved AT2 reference (weights/SE)' },
    decode_equiv_green: { type: 'boolean', description: 'decode_equivalence stays green via the oracle update (not by relaxing the test)' },
    goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the detection rule; the per-regime f2 diff before/after; the Haak tightening; how decode_equivalence/the oracle was kept valid; for FAIL the exact residual' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the pseudo-haploid f2 fix (adversarial; core decode change). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff — confirm per-SAMPLE pseudo-haploid auto-detection (NOT per-pop; a sample with no het -> ploidy1), fed into finalize_af N, default AUTO, no f2-formula change, and the M1 oracle updated (not the test relaxed). (2) ' + BUILD + ' (decode_equivalence + all goldens MUST stay green). (3) THE PARITY PROOF on the box: compute steppe f2 vs AT2 f2 (convertf-PA) on the ALL-PSEUDO-HAPLOID-ANCIENT set (Yamnaya/CordedWare/Turkey_N/Serbia_IronGates/Natufian/Iran_N) -> confirm the prior 15.6% median / 230% max gap COLLAPSED to ~0 (FP floor); on the MODERN-DIPLOID set (Han/French/Sardinian/Papuan/Karitiana/Mbuti) -> confirm STILL exact (0%); re-run the Haak qpAdm 3 models -> confirm tighter vs the saved AT2 reference. PASS only if all those hold + ctest green + build clean. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/oracle/doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (per-sample pseudo-haploid auto-detection = adjust_pseudohaploid parity: f2 on pseudo-haploid now matches AT2, was up to 230% off; modern unaffected; M1 oracle updated) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Append the corrected per-regime f2-parity + Haak numbers to docs/studies/haak2015.md.',
  'ON FAIL: ' + CLEAN + ' and report the exact residual (which regime still off? decode_equivalence?). Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:pseudohaploid', phase: 'Validate f2 parity across regimes + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ PSEUDO-HAPLOID f2 PARITY ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- FAILED (' + verdict.note + ')')
return { verdict }
