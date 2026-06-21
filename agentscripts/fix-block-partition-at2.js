export const meta = {
  name: 'fix-block-partition-at2',
  description: 'Reconcile steppe assign_blocks to the AT2/original-ADMIXTOOLS block-jackknife convention (bit-tight parity) per the verified spec docs/research/block-partition-at2.md. THE CHANGE (src/core/domain/block_partition_rule.cpp ~:52-80): replace the fixed-grid floor-bin loop (anchor = grid line k*blgsize) with the AT2 SNP-ANCHORED walk — carry double fpos=-1e20 + int prev_chrom=-1; open a new block when (c != prev_chrom || (pos - fpos) >= block_size_morgans); re-anchor fpos = pos ONLY when a new block opens; drop block_of()/prev_local_bin/the s==0 special case/the negative-position branch; keep the guards (:35-46) + block_ranges. This drives steppe 718 -> AT2 709 on the Haak union. CONTAINED per the spec: touches assign_blocks + its M3 unit test (re-derive the expected count by MEASURING on the box, do NOT guess) + correctness-neutral sizing literals (test_vram_budget 757s, test_launch_config:179, test_filter_oracle:450-451, bench_f2_multigpu:11) + docs; touches NO AT2-generated golden/fixture (they are read, never recomputed). VALIDATION (the parity proof): steppe extract-f2 on the Haak union must now emit 709 blocks with block_sizes ELEMENT-WISE == AT2 block_lengths (from /workspace/data/haak/at2_f2), AND the Haak qpAdm SE/p tighten toward AT2. fixer -> build-repair -> adversarial verdict -> commit; HALT-on-fail. REAL data; GPU-only; existing goldens stay green.',
  phases: [ { title: 'Reconcile assign_blocks' }, { title: 'Validate AT2 block parity + commit' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main. THE PARITY TARGET is ADMIXTOOLS (original DReichLab C + admixtools R 2.0.10). This reconciles steppe block-jackknife partition (assign_blocks) to the AT2 convention so steppe gets 709 blocks (not 718) on the Haak v66 union and the qpAdm SE/p match AT2 bit-tight. The full verified spec is docs/research/block-partition-at2.md — READ IT.',
  'THE EXACT CHANGE (from the spec): src/core/domain/block_partition_rule.cpp (~:52-80) assign_blocks — replace the fixed-grid floor-bin (block_of(pos)=floor(pos/blgsize), anchored at grid line k*blgsize) with the AT2 SNP-ANCHORED walk: carry `double fpos = -1e20;` and `int prev_chrom = -1;` (or per the existing chrom field); iterate SNPs in order; OPEN A NEW BLOCK when `c != prev_chrom || (pos - fpos) >= block_size_morgans` (>= is INCLUSIVE, matching AT2); on a new block, set `fpos = pos` (re-anchor to the actual opening SNP genpos) and increment the block id; assign the SNP to the current block. Drop block_of()/prev_local_bin, the s==0 special case, and the negative-position branch. KEEP the input guards (:35-46: reject block_size<=0 etc.) and the block_ranges() helper (hpp). Match AT2 C setblocks (qpsubs.c:1698-1759): fpos=-1e20 sentinel makes the first SNP open block 0; per-chromosome reset; trailing short block kept as-is.',
  'CONTAINED (per the spec) — also update: (a) the M3 unit test tests/unit/test_block_partition.cpp — the expected block count (kExpectedNBlock=757) is steppe OLD floor-bin count; under the AT2 walk it changes. MEASURE the new count by running the actual rule on the real v66 .snp on the box (do NOT guess); update the test to the MEASURED AT2-convention count; the block_of/negative-bin test cases (~:99,:132-143) are now moot — remove or adapt. (b) correctness-neutral sizing literals that hardcode 757: test_vram_budget.cpp, test_launch_config.cpp:179, test_filter_oracle.cu:450-451, bench_f2_multigpu.cu:11 — update to the new count where they assert it. (c) docs that cite 757/719 (ROADMAP/TODO/studies) — update. Touch NO AT2-generated golden or fixture (golden_*.json, fixtures/*.bin) — they are READ, never recomputed by assign_blocks, so they stay bit-exact.',
  'VALIDATION (the parity proof, real data): after the change, steppe extract-f2 on the Haak 15-pop union (' + RAW + ', --blgsize 0.05 --maxmiss 0 --auto-only) must emit 709 jackknife blocks with block_sizes ELEMENT-WISE EQUAL to AT2 block_lengths (AT2 ref at /workspace/data/haak/at2_f2 — read its block_lengths from the .rds / cache), AND the 3 Haak qpAdm models SE/p must tighten toward the saved AT2 reference (CW p~0.011, etc., docs/studies/haak2015-at2-reference.md). The existing goldens (qpadm_parity/rotation/qpwave/cli_extract — AT2 fixtures) MUST stay green.',
  'REAL DATA ONLY; GPU-only. Box ' + SSH + '; nvcc -> ' + PATHENV + '; RELEASE -DSTEPPE_BUILD_CLI=ON; nothing builds locally; core dumps cleared per build. Standards: NAMING-STYLE-STANDARD + architecture §-rules; this is a §-specified domain rule so keep it clean + documented. Do NOT re-run AT2 (use the saved block_lengths/reference).',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+ctest (' + BUILD + '). Do NOT commit (the verdict commits). NO synthetic. MEASURE the new block count on the box (do not guess).'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Reconcile assign_blocks')
const fixer = [
  'You are a senior engineer reconciling steppe assign_blocks to the AT2 block-jackknife convention (a §-specified core domain rule — be precise). Do NOT commit. Start clean: ' + CLEAN + '. READ docs/research/block-partition-at2.md (the verified spec) + the current src/core/domain/block_partition_rule.{hpp,cpp}.', STD, '', DEVLOOP, '',
  'IMPLEMENT the AT2 SNP-anchored walk in assign_blocks exactly as the spec states. Then MEASURE the new block count: on the box, run the built steppe (extract-f2 over a pop set on the real v66 .snp, or a tiny harness) to get the new full-v66 / test-input block count under the AT2 walk; update tests/unit/test_block_partition.cpp kExpectedNBlock + the moot block_of/negative cases + the sizing literals (vram_budget/launch_config/filter_oracle/bench) to the MEASURED count. Update docs citing 757/719. Build + full ctest (existing AT2-fixture goldens MUST stay green). SANITY: steppe extract-f2 on the Haak 15-pop union -> report the block count (TARGET 709) + spot-check block_sizes vs AT2 block_lengths. Report every file changed, the new measured count, the Haak block count, and the FULL ctest. Do NOT commit. If you cannot hit 709 on Haak, report exactly the per-chrom divergence.',
].join('\n')
const fix = await tryAgent(fixer, { label: 'fix:assign_blocks', phase: 'Reconcile assign_blocks' })
if (fix === null) { log('--- fixer died — HALT'); return { halted: true } }

await tryAgent(['BUILD-REPAIR for the assign_blocks AT2 reconciliation. Accumulated edits in the tree (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON), patching only trivial -Werror. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + patches.', STD].join('\n'), { label: 'repair', phase: 'Reconcile assign_blocks' })

phase('Validate AT2 block parity + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','haak_709','elementwise_match','sep_tightened','goldens_green','m3_measured','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: steppe extract-f2 on the Haak union now emits 709 blocks with block_sizes ELEMENT-WISE == AT2 block_lengths; the Haak qpAdm SE/p tightened to AT2; existing goldens green; the M3 test count was MEASURED (not guessed) + updated; Release build clean; no AT2 re-run; no synthetic' },
    haak_709: { type: 'boolean', description: 'steppe Haak union block count == 709 (was 718)' },
    elementwise_match: { type: 'boolean', description: 'steppe block_sizes element-wise equal to AT2 block_lengths on the Haak union (bit-tight block parity)' },
    sep_tightened: { type: 'boolean', description: 'the 3 Haak qpAdm models SE/p now match the saved AT2 reference tightly (CW p~0.011 etc.)' },
    goldens_green: { type: 'boolean' }, m3_measured: { type: 'boolean', description: 'the M3 block-count test was re-derived by MEASURING on the box, not guessed' },
    build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the new M3 count + Haak block count + element-wise match result + the tightened Haak SE/p vs AT2; for FAIL the exact residual' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the assign_blocks AT2 reconciliation (adversarial; this is a core domain rule). Do NOT re-run AT2 — use the saved AT2 block_lengths (/workspace/data/haak/at2_f2) + the saved Haak reference (docs/studies/haak2015-at2-reference.md). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff — confirm assign_blocks is the AT2 SNP-anchored walk (fpos re-anchor, >= cut, per-chrom reset) per the spec, clean + standard-conformant; the M3 count was MEASURED on the box (not guessed); no golden/fixture .bin/.json touched. (2) ' + BUILD + ' (existing goldens MUST stay green). (3) THE PARITY PROOF: run `steppe extract-f2` on the Haak 15-pop union -> confirm 709 blocks AND block_sizes ELEMENT-WISE == AT2 block_lengths (read AT2 block_lengths from /workspace/data/haak/at2_f2 .rds / cache); then `steppe qpadm` the 3 Haak models -> confirm SE/p now match the saved AT2 reference tightly (CW se~0.0125/p~0.011, BB p~6.2e-8, Sardinian). PASS only if: Haak 709 + element-wise block match + SE/p tightened + goldens green + M3 measured + build clean. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (assign_blocks -> AT2 SNP-anchored block convention: bit-tight block parity, Haak 709 element-wise == AT2, SE/p match; M3 count re-measured) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Append the now-complete steppe==AT2 Haak result to docs/studies/haak2015.md.',
  'ON FAIL: ' + CLEAN + ' and report the exact residual (block count? which chrom? which model SE/p?). Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:assign_blocks', phase: 'Validate AT2 block parity + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ AT2 BLOCK PARITY ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- FAILED (' + verdict.note + ')')
return { verdict }
