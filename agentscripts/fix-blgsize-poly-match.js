export const meta = {
  name: 'fix-blgsize-poly-match',
  description: 'Make steppe extract-f2 COMPLETELY match AT2 on the Haak v66 data (precompute parity), then re-run STEPPE ONLY (do NOT re-run AT2 — compare to the already-saved AT2 reference in docs/studies/haak2015-at2-reference.md). Two precompute-parity fixes the Haak study surfaced: (1) --blgsize UNITS — AT2 blgsize=0.05 is MORGANS (=5 cM); steppe read 0.05 as cM -> 48184 jackknife blocks vs AT2 709 -> wrong SE/p (weights were block-insensitive so they already matched). Make steppe --blgsize accept MORGANS like AT2 (so --blgsize 0.05 -> ~709 blocks). (2) MONOMORPHIC filtering — AT2 extract_f2 drops monomorphic SNPs from the f2 (poly_only) -> 264544 of 290750; steppe kept all 290750 -> small weight/chisq residual. Make steppe extract-f2 drop monomorphic SNPs from the f2 like AT2 -> 264544 SNPs. Then re-run ONLY steppe extract-f2 (--blgsize 0.05, the 15-pop Haak union) + qpadm the 3 Haak models on the GPU and confirm a COMPLETE match to the SAVED AT2 numbers (CordedWare 0.7423/se 0.01247/p 0.0110/709 blocks/264544 SNPs; BellBeaker 0.5269/p 6.2e-8; Sardinian WHG 0.1657/EEF 0.7173/Yam 0.1171). fixer -> build-repair -> verdict -> commit; HALT-on-fail. GPU-only; REAL data; existing goldens (fixtures) must stay green; commit on green.',
  phases: [ { title: 'Fix blgsize+poly' }, { title: 'Verify steppe-only match + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'
const RAW = '/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main. The CLI works: `steppe extract-f2 --prefix <p> --pops <names> --out <dir> [--blgsize B --maxmiss 0 --auto-only]` then `steppe qpadm --f2-dir <dir> --target T --left a,b --right r0,..`. Binary /workspace/steppe/build-rel/bin/steppe.',
  'THE TWO PARITY FIXES (from the Haak 2015 study, docs/studies/haak2015.md + haak2015-at2-reference.md): steppe extract-f2 matched AT2 on WEIGHTS but not on block count / SNP count / SE-p. Root causes: (1) --blgsize UNITS: AT2 blgsize=0.05 = MORGANS (=5 cM); steppe interpreted --blgsize 0.05 as cM -> 48184 blocks vs AT2 709. (2) MONOMORPHIC: AT2 drops monomorphic SNPs from f2 (poly_only) -> 264544 of 290750; steppe kept 290750.',
  'THE SAVED AT2 REFERENCE (do NOT re-run AT2 — compare against these): blgsize 0.05 Morgans, maxmiss 0, 264544 polymorphic SNPs, 709 jackknife blocks. CordedWare(Czechia_EBA_CordedWare = Russia_Samara_EBA_Yamnaya + Turkey_N): Yamnaya 0.7423258 (se 0.01247259, z 59.52), Turkey_N 0.2576742; chisq 18.22086, dof 7, p 0.01101227. BellBeaker(Czechia_BellBeaker): Yamnaya 0.526931 (se 0.01280085), Turkey_N 0.473069; chisq 46.76779, p 6.20e-08. Sardinian(= Serbia_IronGates_Mesolithic + Turkey_N + Russia_Samara_EBA_Yamnaya): WHG 0.1656654 (se 0.02939), Turkey_N 0.7172783 (se 0.01592), Yamnaya 0.1170563 (se 0.02398); chisq 116.5725, dof 6, p 8.54e-23. Right set (9): Mbuti, Russia_UstIshim_IUP, Russia_Kostenki_UP, Russia_Malta_UP, Han, Papuan, Karitiana, Iran_GanjDareh_N, Israel_Natufian. Union = those + the 3 targets + the 3 sources (15 pops; same as docs/studies/haak2015.md).',
  'REAL DATA ONLY; GPU-only. Box ' + SSH + '; nvcc -> ' + PATHENV + '; RELEASE -DSTEPPE_BUILD_CLI=ON; nothing builds locally; core dumps cleared per build. Standards: NAMING-STYLE-STANDARD + architecture §-rules. CUDA/API claims doc-verified. The EXISTING goldens/fixtures are AT2-generated (709 blocks / 264544 SNPs baked in) so the qpadm-on-fixture tests are UNAFFECTED by these extract-f2 changes and must stay green; the extract-f2 e2e test (test_cli_extract_qpadm) should now match the corrected golden_fit0 TIGHTER (the block-partition residual shrinks).',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD at start (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+ctest (' + BUILD + '). Do NOT commit (the verdict commits). NO synthetic. Do NOT re-run AT2 (use the saved reference).'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Fix blgsize+poly')
const fixer = [
  'You are a senior engineer fixing TWO precompute-parity items in steppe extract-f2 so it COMPLETELY matches AT2. Do NOT commit (the verdict commits). Start clean: ' + CLEAN, STD, '', DEVLOOP, '',
  'FIX 1 — --blgsize UNITS: trace --blgsize from the CLI (src/app/cmd_extract_f2.cpp / cli_parse / config_builder) to core::domain::block_partition_rule assign_blocks (which takes block_size in some unit). Determine the unit bug: AT2 blgsize is MORGANS (0.05 = 5 cM); steppe currently treats --blgsize as cM (or passes 0.05 straight to a cM field) -> 48184 blocks. FIX so steppe --blgsize is in MORGANS like AT2 (the field convention): --blgsize 0.05 must yield ~709 blocks on the Haak union (matching AT2). If the core assign_blocks takes cM, convert Morgans->cM (x100) at the CLI/config seam; verify the existing kDefaultBlockSizeCm/block rule semantics + that the existing block_partition unit test + the qpadm goldens still pass. Document the unit clearly in --blgsize help.',
  'FIX 2 — MONOMORPHIC f2 filter: AT2 extract_f2 drops SNPs monomorphic across the analysis pop set from the f2 (poly_only="f2") -> 264544 of 290750 on the Haak union. steppe keeps all 290750. Add monomorphic-SNP dropping to steppe extract-f2 / the precompute so the f2 is computed on the polymorphic subset, matching AT2 (-> 264544 SNPs on the Haak union). Check FilterConfig.drop_monomorphic / the decode_af / f2 path — wire it so extract-f2 drops monomorphic by default (AT2 parity) OR via the right default. Verify it does not break the f2 math (a monomorphic SNP contributes 0 to f2 differences anyway — dropping changes the SNP count + the per-block counts + thus SE, matching AT2).',
  '', 'Build + full ctest (existing goldens/fixtures MUST stay green — they read AT2 fixtures, unaffected). Then SANITY (no commit): rebuild the binary and run `steppe extract-f2 --prefix ' + RAW + ' --pops <15 Haak union> --out /workspace/data/haak/steppe_f2_fixed --blgsize 0.05 --maxmiss 0 --auto-only` and report the block count + SNP count (TARGET: ~709 blocks, 264544 SNPs, matching AT2). Report every file changed, the unit fix, the monomorphic fix, the new block/SNP counts, and the FULL ctest. Do NOT commit. If you cannot hit ~709 blocks / 264544 SNPs, report exactly why.',
].join('\n')
const fix = await tryAgent(fixer, { label: 'fix:blgsize-poly', phase: 'Fix blgsize+poly' })
if (fix === null) { log('--- fixer died — HALT'); return { halted: true } }

await tryAgent(['BUILD-REPAIR for the blgsize+poly fix. Accumulated edits in the tree (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON), patching only trivial -Werror (unused/[[maybe_unused]]). DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP+report. Report final build + patches.', STD].join('\n'), { label: 'repair', phase: 'Fix blgsize+poly' })

phase('Verify steppe-only match + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','blocks_match','snps_match','weights_match','sep_match','goldens_green','no_synthetic','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if steppe extract-f2 (--blgsize 0.05 Morgans, monomorphic dropped) on the Haak union now COMPLETELY matches the SAVED AT2 reference: ~709 blocks, 264544 SNPs, weights tight (rtol ~1e-3 or better), SE/p close; existing goldens green; Release build clean; NO AT2 re-run; NO synthetic' },
    blocks_match: { type: 'boolean', description: 'steppe block count ~= AT2 709 (was 48184)' },
    snps_match: { type: 'boolean', description: 'steppe SNP count = 264544 (was 290750)' },
    weights_match: { type: 'boolean', description: 'all 3 Haak models weights match AT2 tight (CW 0.7423, BB 0.5269, Sardinian 0.166/0.717/0.117)' },
    sep_match: { type: 'boolean', description: 'SE/p now close to AT2 (CW p~0.011, etc.) — the blgsize fix worked' },
    goldens_green: { type: 'boolean' }, no_synthetic: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the new steppe block/SNP counts + per-model steppe-vs-AT2 (weights/se/p); any residual (e.g. 709-vs-710 boundary nuance); for FAIL what is still off' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the blgsize+monomorphic parity fix (adversarial). Do NOT re-run AT2 — compare steppe to the SAVED AT2 reference numbers in STD. The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff — confirm the --blgsize-Morgans + monomorphic-drop fixes are real + standard-conformant. (2) ' + BUILD + ' (existing goldens MUST stay green). (3) RUN STEPPE ONLY: `LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe extract-f2 --prefix ' + RAW + ' --pops <15 Haak union from STD> --out /workspace/data/haak/steppe_f2_fixed --blgsize 0.05 --maxmiss 0 --auto-only` -> confirm ~709 blocks + 264544 SNPs; then `steppe qpadm` the 3 Haak models -> compare weights/se/p to the SAVED AT2 reference. PASS only if: blocks ~709, SNPs 264544, weights tight vs AT2, SE/p close vs AT2, goldens green, build clean. (A small residual e.g. 709 vs 710 from the assign_blocks boundary rule is acceptable IF SE/p are now close — note it; a still-48184 block count or 290750 SNPs is a FAIL.) ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (extract-f2 blgsize=Morgans + monomorphic-drop -> COMPLETE AT2 parity on Haak; steppe now 709 blocks/264544 SNPs; per-model steppe-vs-AT2) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Also append the corrected steppe numbers to docs/studies/haak2015.md.',
  'ON FAIL: ' + CLEAN + ' and report exactly what is still off (block count? SNP count? which model?). Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:blgsize-poly', phase: 'Verify steppe-only match + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ COMPLETE AT2 parity ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- FAILED (' + verdict.note + ')')
return { verdict }
