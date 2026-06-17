export const meta = {
  name: 'fix-pass-phase2',
  description: 'Phase-2 cleanup fixes B9-B27 (resumed after pause; B17 and B8 already committed) — STRICTLY SEQUENTIAL, two agents per task (independent fixer + verdict). Each: clean-HEAD -> fix -> rsync+build+ctest on the 5090 -> verdict (objective gate) -> commit on PASS / revert on FAIL. SKIP-and-continue (items mostly independent). Retry once on transient API error.',
  phases: [
    { title: 'B9' }, { title: 'B10' }, { title: 'B11' }, { title: 'B12' },
    { title: 'B13' }, { title: 'B14' }, { title: 'B15' }, { title: 'B16' }, { title: 'B18' }, { title: 'B19' },
    { title: 'B20' }, { title: 'B21' }, { title: 'B22' }, { title: 'B23' }, { title: 'B24' }, { title: 'B25' },
    { title: 'B26' }, { title: 'B27' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh -i ~/.ssh/id_vastai -p 43215 -o BatchMode=yes root@78.92.24.57'
const RSYNC = `rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e "ssh -i ~/.ssh/id_vastai -p 43215 -o BatchMode=yes" ${R}/ root@78.92.24.57:/workspace/steppe/`
const BUILD = `${SSH} 'cmake -S /workspace/steppe -B /workspace/steppe/build -GNinja >/tmp/cfg.log 2>&1 && cmake --build /workspace/steppe/build 2>&1 | tail -20 && echo === CTEST === && ctest --test-dir /workspace/steppe/build --output-on-failure 2>&1 | tail -30'`
const CLEAN = `cd ${R} && git checkout -- . && git clean -fd src tests include`

const STD = `steppe = CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics, branch m4-perblock-f2 (Phase-1 fixes B7,B1-B6 already landed). Standards: ${R}/docs/architecture.md (section 2 DRY/separation/RAII/fail-fast, section 4 layering [io leaf; CUDA PRIVATE to steppe_device; core CUDA-free], section 7 CUDA idioms, section 8 DRY single-home, section 9 config, section 11.2 VRAM budget, section 12 precision/parity, section 13 testing) and ${R}/docs/ROADMAP.md sections 4/5/6. STEPPE_DEBUG_ONLY / STEPPE_ASSERT now EXIST (core/internal/host_device.hpp, from B7); cdiv/grid_for live in core/internal/launch_config.hpp; STEPPE_LOG_WARN in the log home. The detailed finding for each fix is in the per-unit review under ${R}/docs/cleanup/ and the master backlog ${R}/docs/cleanup/00-overview.md — READ the relevant one(s) before editing.`

const DEVLOOP = `DEV LOOP (nothing builds locally): FIRST ensure a clean tree at HEAD so a prior skipped item cannot contaminate this fix — run: ${CLEAN} . Then edit locally, (1) rsync to the box: ${RSYNC} ; (2) build+test on the box: ${BUILD} . If you add a NEW test file, wire it into ${R}/tests/CMakeLists.txt first. Build MUST be clean (warnings-as-errors) and ALL ctest green (the 13 pre-existing tests must still pass — no regression).`

const ITEMS = [
  { id: 'B9', title: 'B9', files: 'config.hpp',
    fix: 'Add bool deterministic = true to DeviceConfig (matches the section-9 spec; gates the section-12 stream_count/workspace/combine rules M4.5 relies on). Doc-comment its meaning.',
    test: 'Build + ctest green (field add; assert default true in a config check if one exists, else build is the gate).' },
  { id: 'B10', title: 'B10', files: 'decode_af.hpp, snp_filter.cpp, new tests/unit/test_decode.cpp',
    fix: 'Fold ploidy>0 into finalize_af validity (ploidy<=0 with an>0 currently yields NaN/Inf Q with v=1.0 — degrade to masked-out {0,0,0}); reconcile with snp_filter clamp-to-1 + validate ploidy in {1,2}.',
    test: 'test_decode.cpp asserts ploidy=0 -> masked {0,0,0}, ploidy=1 -> haploid (1x), ploidy=2 -> diploid (2x).' },
  { id: 'B11', title: 'B11', files: 'f2_from_blocks.cpp, new tests/unit/test_f2_from_blocks.cpp',
    fix: 'Add validate_qvn(Q,V,N) + validate_partition(block_id,M,n_block) (debug STEPPE_ASSERT, CUDA-free): P/M agree, block_id.size()==M (incl null .data()), n_block<=M, ids non-decreasing. Add a CUDA-free MockBackend unit test (the seam was designed GPU-free-testable and has none).',
    test: 'test_f2_from_blocks.cpp drives a MockBackend with valid + malformed inputs (assert/throw on malformed).' },
  { id: 'B12', title: 'B12', files: 'cuda_backend.cu, f2_block_kernel.cu',
    fix: 'Add a P<=0 || M<=0 early-return guard to GPU compute_f2 (sibling-consistency: compute_f2_blocks/decode_af/CPU all guard; GPU compute_f2 throws from deep in cuBLAS instead of returning empty).',
    test: 'compute_f2 with P=0/M=0 returns an empty F2Result cleanly (no cuBLAS throw); build+ctest green.' },
  { id: 'B13', title: 'B13', files: 'block_partition_rule.cpp',
    fix: 'Guard assign_blocks: if (!(block_size_morgans > 0.0)) return empty — rejects 0 / negative / NaN (closes the float->int UB; the ConfigBuilder::build() that would catch it does not exist yet).',
    test: 'block_partition_unit cases with block_size 0 / negative / NaN -> empty / defined behavior (no UB).' },
  { id: 'B14', title: 'B14', files: 'snp_reader.cpp, new tests/unit/test_snp_reader.cpp',
    fix: 'Replace the extraction-failure fall-through (silent SNP-axis misalignment) with a token-count-based column decision; parse genpos with std::from_chars (locale-free, rejects NaN/Inf); fail-fast with the line number on a malformed record.',
    test: 'test_snp_reader.cpp: malformed record -> throws with line number; well-formed -> correct chrom + Morgan genpos.' },
  { id: 'B15', title: 'B15', files: 'snp_reader.cpp',
    fix: 'Replace std::stoi in chrom_code with std::from_chars; route overflow/garbage to the negative-sentinel path (makes the documented runtime_error-only contract true; stoi throws today).',
    test: 'chrom_code on garbage/overflow -> negative sentinel (no uncaught throw); in test_snp_reader.cpp.' },
  { id: 'B16', title: 'B16', files: 'eigenstrat_format.hpp, snp_reader.cpp, config.hpp',
    fix: 'Promote bare 23/24/90 to kChromCodeX/kChromCodeY/kChromCodeMt in eigenstrat_format.hpp; reference from snp_reader chrom_code; cross-link the config.hpp autosome-cutoff comment (autosome-filter correctness depends on these exact codes).',
    test: 'Build + an assertion chrom_code maps X->kChromCodeX etc.; ctest green.' },
  { id: 'B18', title: 'B18', files: 'geno_reader.cpp, eigenstrat_format.cpp',
    fix: 'tile.packed.resize can throw bad_alloc/length_error (neither runtime_error -> violates the unit contract): bound the tile up front + throw runtime_error, or relax the doc. Harden the header v*10+digit decimal parse against silent wrap; guard the streamoff offset multiply.',
    test: 'Oversized tile request -> runtime_error (or the documented exception); malformed header digits handled. In test_geno_reader.cpp.' },
  { id: 'B19', title: 'B19', files: 'include_exclude.cpp, test_filters.cpp',
    fix: 'After the prune.in getline loop add: if (in.bad() || (in.fail() && !in.eof())) throw — a directory opens but read-fails on Linux, currently a silently-empty keep-set.',
    test: 'Add the (currently zero) prune.in file-branch tests: directory / unreadable -> throws; valid file -> correct set.' },
  { id: 'B20', title: 'B20', files: 'snp_filter.hpp, snp_filter.cpp, filter_decision.hpp, test_filters.cpp',
    fix: 'Validate pop_individuals.size()==P (unguarded numerator can yield a negative missing_frac that spuriously passes geno), ploidy in {1,2}, non-null q/n, ref/alt size>=M; fix the is_monomorphic double-fold (shared with filter_decision); extract a shared snp_keep_decision(...) primitive (M4.5 fusion prep); add the missing HOST unit tests (only GPU/.cu coverage today — a section-13 violation).',
    test: 'Host test_filters cases for the new guards + the extracted snp_keep_decision primitive.' },
  { id: 'B21', title: 'B21', files: 'mind_prepass.hpp, mind_prepass.cpp, test_filters.cpp',
    fix: 'Fix the n_snp==0 + active header/code contradiction (header says drop, code keeps all) — make the doc match the code; kill the bare 4u x2 (use io::kCodesPerByte, already in scope).',
    test: 'Add the n_snp==0 + active test case; ctest green.' },
  { id: 'B22', title: 'B22', files: 'cuda_backend.cu, f2_block_kernel.cu',
    fix: 'Add an M <= INT_MAX guard before static_cast<int>(M) feeds cublasGemmEx int k (the M0 whole-matrix path; MatView::M is long). Or move M0 to cublasGemmEx_64.',
    test: 'A guard (M>INT_MAX -> typed error, not silent overflow); build+ctest green.' },
  { id: 'B23', title: 'B23', files: 'device_buffer.cuh',
    fix: 'Add a checked-multiply guard for n*sizeof(T) overflow (typed error, <limits>) before cudaMalloc; document bytes() exact under the invariant for the section-11.2 budget.',
    test: 'A host unit test constructing DeviceBuffer<T> with n near SIZE_MAX/sizeof(T) -> throws (no silent wrap); build+ctest green.' },
  { id: 'B24', title: 'B24', files: 'f2_blocks_kernel.cu',
    fix: 'Early-return / STEPPE_ASSERT on n_in_group <= 0 (a zero grid.z is an invalid launch); document s_pad >= 1 in launch_gather_group / run_f2_gemms_group / launch_assemble_blocks_group. M4.5 sharding can hand a device an empty SNP shard.',
    test: 'Covered by the B25 gather test + a guard assertion; build+ctest green.' },
  { id: 'B25', title: 'B25', files: 'new tests/reference/test_f2_blocks_gather.cu',
    fix: 'Add a focused gather/scatter unit test: a synthetic 2-3 block, REORDERED block_ids_in_group case asserting (i) pad columns (c>=sz) are exactly 0, (ii) gathered columns equal the feeder columns, (iii) the scatter lands block id slab at the right [PxP] offset (the M4 index math is only transitively covered today).',
    test: 'The new test itself (runnable on the box, no real data). Wire into CMakeLists.' },
  { id: 'B26', title: 'B26', files: 'fstats.hpp, architecture.md (section 11.2), cuda_backend.cu (verify only)',
    fix: 'B5 already made the cuda_backend RUNTIME budget count both f2+vpair — VERIFY that first. Finish the parts B5 did NOT: change fstats.hpp:19 to count BOTH FP64 tensors (2*P^2*n_block*8), and add a Vpair row (P^2*B*8) to the architecture.md section-11.2 budget table so the documented build() budget reserves for both.',
    test: 'Build + the existing vram_budget_unit still green; confirm fstats.hpp + architecture section-11.2 now agree with the runtime budget. (No new test needed if B5 covered the runtime path — say so.)' },
  { id: 'B27', title: 'B27', files: 'decode_af.hpp, decode_af_kernel.cu, cpu_backend.cpp',
    fix: 'Add accumulate_genotype(int code, int64_t& ac, int64_t& an) (STEPPE_HD) to decode_af.hpp; both decode loops (decode_af_kernel/cuda + cpu_backend) call it — completes the oracle-equals-GPU divergence-prevention story before M4.5 filter-fusion touches the inner step.',
    test: 'The existing decode_equivalence (GPU vs CPU vs numpy oracle) stays bit-exact (it exercises both loops); build+ctest green.' },
]

const fixPrompt = (it) =>
  `You are a senior CUDA/C++ engineer applying ONE fix to steppe (branch m4-perblock-f2). Do NOT commit (the independent verdict agent commits).\n\n${STD}\n\nFIX ${it.id}: ${it.fix}\nPRIMARY FILES: ${it.files}\nREQUIRED TEST (objective verdict gate): ${it.test}\n\nApply the fix per the architecture standards; update any doc-comments the fix makes stale. Where a claim depends on CUDA/cuBLAS or C++-stdlib behavior, cite the docs. Add the required test and wire it into tests/CMakeLists.txt if new.\n\n${DEVLOOP}\n\nReturn a thorough report: (1) every file changed + what changed; (2) the test added + what it asserts; (3) the FULL build result; (4) the FULL ctest result (paste the summary lines). If you CANNOT reach a clean green build + green required-test, do NOT pretend success — report exactly what blocked it.`

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item', 'pass', 'regression', 'commit_hash', 'note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true only if build clean + ALL ctest green + the diff genuinely addresses the finding + the required objective test is present and green + no regression' },
    regression: { type: 'boolean' },
    commit_hash: { type: 'string', description: 'short hash if committed on PASS, else empty' },
    note: { type: 'string', description: 'one-line rationale; for FAIL, the exact reason + whether it is a design call needing human input' },
  },
}

const verdictPrompt = (it, fixReport) =>
  `You are the INDEPENDENT VERDICT for fix ${it.id} of steppe (you did NOT write the fix — be adversarial). The fixer reported:\n<<<\n${fixReport}\n>>>\n\n${STD}\n\nThe finding being fixed: ${it.fix}\nThe objective gate: ${it.test}\n\nDO: (1) inspect the actual uncommitted changes — run: cd ${R} && git --no-pager diff --stat && git --no-pager diff ; (2) judge PASS only if ALL hold — build clean (warnings-as-errors), ALL ctest green (NO regression vs the 13 prior tests), the diff GENUINELY addresses ${it.id} (not a sham or comment-only), and the REQUIRED objective test is present and green. If the fixer report is inconsistent with the diff, re-run the box build/ctest yourself with: ${BUILD}\n\nON PASS: cd ${R} and git add ONLY the specific changed/new source+test+CMake+doc files for this fix (NEVER git add dot — leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md untracked), then commit with a ROADMAP section-6 message (what+why; the test added; the box build/run commands; end with the trailer line: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>). Capture the short hash via git rev-parse --short HEAD.\nON FAIL: revert the working tree so the repo stays green — run: ${CLEAN} — and report the exact reason (and whether it is a design call needing human input).\n\nReturn the structured verdict.`

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(`${opts.label}: transient null — retrying once`); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  log(`=== ${it.id}: fixing ===`)
  const fix = await tryAgent(fixPrompt(it), { label: `fix:${it.id}`, phase: it.title })
  if (fix === null) { ledger.push({ item: it.id, pass: false, regression: false, commit_hash: '', note: 'fix-agent terminal API error after retry — SKIPPED' }); continue }
  const v = await tryAgent(verdictPrompt(it, fix), { schema: VERDICT_SCHEMA, label: `verdict:${it.id}`, phase: it.title })
  if (v === null) { ledger.push({ item: it.id, pass: false, regression: false, commit_hash: '', note: 'verdict-agent terminal API error after retry — SKIPPED' }); continue }
  ledger.push(v)
  if (v.pass) log(`+++ ${it.id} committed ${v.commit_hash}`)
  else log(`--- ${it.id} FAILED verdict (${v.note}) — reverted; skipping to next`)
}
return { ledger }