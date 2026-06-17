export const meta = {
  name: 'fix-pass-phase1',
  description: 'Phase-1 cleanup fixes B7,B1,B2,B3,B4,B5,B6 — STRICTLY SEQUENTIAL, two agents per task (independent fixer + independent verdict). Each: fix -> rsync+build+ctest on the 5090 box -> verdict (objective test gate) -> commit on PASS / revert+HALT on FAIL. Retry once on transient API error. B5/B6 at-scale validation deferred to a PRO-6000 session.',
  phases: [
    { title: 'B7 homes' }, { title: 'B1 workspace' }, { title: 'B2 emu-guard' },
    { title: 'B3 block_ranges' }, { title: 'B4 diagonal' }, { title: 'B5 vram' }, { title: 'B6 grid' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh -i ~/.ssh/id_vastai -p 43215 -o BatchMode=yes root@78.92.24.57'
const RSYNC = `rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e "ssh -i ~/.ssh/id_vastai -p 43215 -o BatchMode=yes" ${R}/ root@78.92.24.57:/workspace/steppe/`
const BUILD = `${SSH} 'cmake -S /workspace/steppe -B /workspace/steppe/build -GNinja >/tmp/cfg.log 2>&1 && cmake --build /workspace/steppe/build 2>&1 | tail -20 && echo === CTEST === && ctest --test-dir /workspace/steppe/build --output-on-failure 2>&1 | tail -30'`

const STD = `steppe = CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics, branch m4-perblock-f2. Standards: ${R}/docs/architecture.md (section 2 DRY/separation/RAII/fail-fast, section 4 layering [io leaf; CUDA PRIVATE to steppe_device; core CUDA-free], section 7 CUDA idioms, section 8 DRY single-home, section 9 config, section 11.2 VRAM budget, section 12 precision/determinism/parity, section 13 testing) and ${R}/docs/ROADMAP.md sections 4/5/6. The detailed finding for each fix is in the per-unit review under ${R}/docs/cleanup/ and the master backlog ${R}/docs/cleanup/00-overview.md — READ the relevant one(s) before editing.`

const DEVLOOP = `DEV LOOP (nothing builds locally): after editing locally, (1) rsync to the box with: ${RSYNC} ; (2) build+test on the box with: ${BUILD} . If you add a NEW test file, wire it into ${R}/tests/CMakeLists.txt first (the reconfigure step picks it up). The build MUST be clean (warnings-as-errors) and ALL ctest green (the existing 8 tests must still pass — no regression).`

const ITEMS = [
  { id: 'B7', title: 'B7 homes',
    files: 'NEW core/internal/launch_config.hpp + core/internal/host_device.hpp + internal/log.hpp; EDIT f2_estimator.hpp, decode_af.hpp, decode_af_kernel.cu, check.cuh, stream.hpp, handles.hpp, device_buffer.cuh, config.hpp(comment)',
    fix: 'Create the real single-source homes the spec cites but that do not exist (X-4/B7): (a) core/internal/launch_config.hpp — move cdiv (both overloads) + grid_for out of f2_estimator.hpp + a warp-justified 32x8 decode block constant; DELETE decode_af_kernel.cu cdiv_l/cdiv_i and route it through core::cdiv with explicit dims (NOT grid_for square default); (b) core/internal/host_device.hpp — ONE STEPPE_HD plus a STEPPE_DEBUG_ONLY / STEPPE_ASSERT facility; (c) internal/log.hpp — ONE teardown-warning sink (STEPPE_LOG_WARN) replacing the 3 duplicated fprintf macros in device_buffer.cuh/stream.hpp/handles.hpp plus the check.cuh NDEBUG gate. Fix the dangling citations in config.hpp:49-50 and f2_estimator.hpp:117. PURE REFACTOR — no behavior change.',
    test: 'No behavior change, so the gate is: build clean + ALL existing ctest green, AND a grep for cdiv_l/cdiv_i across src/ returns nothing (the duplicate cdiv is gone), AND only one STEPPE_HD definition remains.' },
  { id: 'B1', title: 'B1 workspace',
    files: 'handles.hpp, cuda_backend.cu, f2_block_kernel.cu, f2_blocks_kernel.cu',
    fix: 'Fix the cuBLAS workspace-reset determinism void (X-1/B1): cublasSetStream resets the workspace to the default pool (cite the cuBLAS 13 docs), so the per-call cublasSetStream in BOTH GEMM routines discards the section-12 emulated-FP64 reproducibility workspace. Give CublasHandle a set_stream() that RE-APPLIES the owned workspace after cublasSetStream; bind stream+workspace once in the ctor; DROP the per-call cublasSetStream from run_f2_gemms AND run_f2_gemms_group (each passes nullptr today). Keep the single statistic stream (section 12).',
    test: 'Add a NEW determinism test (tests/reference/, .cu): compute f2 (and f2_blocks) TWICE on the same real-AADR input at EmulatedFp64{40} and assert the two outputs are BIT-IDENTICAL run-to-run (exact equality) — the workspace must persist for this to hold. Plus all existing ctest green. IF you cannot make bit-identical hold, this is a design call: report FAIL with the exact reason rather than committing.' },
  { id: 'B2', title: 'B2 emu-guard',
    files: 'f2_block_kernel.cu, cuda_backend.cu, config.hpp, test_f2_blocks_equivalence.cu',
    fix: 'Close the EmulatedFp64 dynamic-mantissa trap (X-6/B2): when STEPPE_HAVE_EMU_TUNING=OFF, engage_f2_precision still sets CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH but the FIXED-slice pin is compiled out, so cuBLAS uses its ~60-bit DYNAMIC default while still reporting EmulatedFp64{40}. Route the honorability decision through ONE predicate driving BOTH math-mode and compute-type; when tuning is OFF + EmulatedFp64 requested, fall back to native Fp64 with a logged capability tag OR throw INVALID_CONFIG — never silently run dynamic.',
    test: 'Build the box with -DSTEPPE_HAVE_EMU_TUNING=0 and confirm the EmulatedFp64 path is OBSERVABLY refused/downgraded (logged tag or throw), NOT silently dynamic. Also promote the M4 blocks-test silent-fallback skip to a FAIL when EmulatedFp64 is the default. Default build (=1) all ctest green.' },
  { id: 'B3', title: 'B3 block_ranges',
    files: 'block_partition_rule.hpp, block_partition_rule.cpp, cuda_backend.cu, cpu_backend.cpp, f2_from_blocks.cpp',
    fix: 'Single-home + validate the block-range scan (X-3/B3): add block_ranges(std::span<const int> block_id, long M, int n_block) returning std::vector<BlockRange> to block_partition_rule.{hpp,cpp} (host-pure, CUDA-free), validating 0<=id<n_block AND non-decreasing ONCE; both backends (cuda_backend.cu, cpu_backend.cpp) call it and DELETE their hand-duplicated scans; closes the OOB write/read.',
    test: 'A host unit test for block_ranges (tests/unit/, valid + malformed: empty, single-block, out-of-range id, non-monotonic), wired into CMakeLists, plus the existing f2_blocks_equivalence test still green (it exercises the path).' },
  { id: 'B4', title: 'B4 diagonal',
    files: 'cpu_backend.cpp, backend.hpp, fstats.hpp, test_f2_equivalence.cu, test_f2_blocks_equivalence.cu',
    fix: 'Fix the F2Result M0 diagonal divergence (X-2/B4): CpuBackend::compute_f2 walks j=i+1 (diagonal left 0) while the GPU and compute_f2_blocks fill the diagonal. Make CpuBackend::compute_f2 loop j=i to match; pin the diagonal convention on F2Result::f2 (backend.hpp/fstats.hpp); fix the 3 parroting comments; wire the PRODUCTION CpuBackend::compute_f2 into test_f2_equivalence.cu (it uses an inline oracle today) and diff the FULL matrix INCLUDING the diagonal.',
    test: 'test_f2_equivalence.cu diffs the full matrix (diagonal included) GPU-vs-production-CpuBackend and passes at the tight tier; all ctest green.' },
  { id: 'B5', title: 'B5 vram',
    files: 'config.hpp, cuda_backend.cu',
    fix: 'Fix the VRAM budget (X-5/B5 + B26): add an inline constexpr double kMaxVramUtilizationFraction = 0.80 to config.hpp (doc-comment reconciling it to architecture.md section 11.1 60-70 percent), replace the 0.80 literal in cuda_backend.cu; CRITICAL (B26): the budget must count BOTH the resident f2_blocks AND the resident vpair tensor (both P x P x n_block, 8 bytes) — it currently counts one, so it under-budgets by ~2x; also subtract the cuBLAS workspace (kCublasWorkspaceBytes) before applying the fraction; clamp the budget/max_blocks math in size_t before the int narrowing.',
    test: 'A host unit test of the budget helper (given free-VRAM, P, n_block: the computed max_blocks accounts for 2x tensors + workspace and never exceeds the fraction) plus build+ctest green. NOTE: at-scale validation (large-P near a 96GB ceiling) is DEFERRED to a PRO-6000 session — logic + unit-test green on the 5090 is PASS here; say so in your report.' },
  { id: 'B6', title: 'B6 grid',
    files: 'cuda_backend.cu, f2_block_kernel.cu, f2_blocks_kernel.cu',
    fix: 'Fix the grid-dimension launch failures (X-7/B6): add kMaxGridZ=65535 (config or launch_config.hpp from B7); the M4 grid.z = n_in_group in BOTH the gather and scatter launches bypasses grid_for, so clamp/tile it; re-orient the f2 feeder so the SNP count rides grid.x (matching the safe decode launcher); fold the decode grid.y clamp into grid_for (exists after B7).',
    test: 'A host unit test of the clamp/grid_for logic (grid dims stay <=65535; re-orientation puts the large extent on grid.x) plus build+ctest green. NOTE: the real >1.05M-SNP trigger needs a dataset we do not have, so that at-scale confirmation is DEFERRED (PRO-6000 / synthetic large-M); clamp-logic unit-test + build+ctest on the 5090 is PASS here.' },
]

const fixPrompt = (it) =>
  `You are a senior CUDA/C++ engineer applying ONE fix to steppe (branch m4-perblock-f2), devoting your full attention to it. Do NOT commit (the independent verdict agent commits).\n\n${STD}\n\nFIX ${it.id}: ${it.fix}\nPRIMARY FILES: ${it.files}\nREQUIRED TEST (objective verdict gate): ${it.test}\n\nApply the fix following the architecture standards; update any doc-comments / architecture.md notes the fix makes stale (keep docs accurate). Where a claim depends on CUDA/cuBLAS behavior, cite the official docs. Add the required test and wire it into tests/CMakeLists.txt if new.\n\n${DEVLOOP}\n\nReturn a thorough report: (1) every file changed + what changed; (2) the new test added + what it asserts; (3) the FULL build result (clean? warnings?); (4) the FULL ctest result (paste the summary lines); (5) for B5/B6 explicitly note the at-scale deferral. If you CANNOT reach a clean green build + green required-test, do NOT pretend success — report exactly what blocked it.`

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item', 'pass', 'regression', 'commit_hash', 'note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true only if build clean + ALL ctest green + the diff genuinely addresses the finding + the required objective test is present and green + no regression' },
    regression: { type: 'boolean', description: 'true if any previously-green test now fails or behavior regressed' },
    commit_hash: { type: 'string', description: 'short hash if committed on PASS, else empty' },
    note: { type: 'string', description: 'one-line rationale; for FAIL, the exact reason + whether it is a design call needing human input' },
  },
}

const verdictPrompt = (it, fixReport) =>
  `You are the INDEPENDENT VERDICT for fix ${it.id} of steppe (you did NOT write the fix — be adversarial). The fixer reported:\n<<<\n${fixReport}\n>>>\n\n${STD}\n\nThe finding being fixed: ${it.fix}\nThe objective gate: ${it.test}\n\nDO: (1) inspect the actual uncommitted changes — run: cd ${R} && git --no-pager diff --stat && git --no-pager diff ; (2) judge PASS only if ALL hold — build was clean (warnings-as-errors), ALL ctest green (NO regression vs the prior 8 tests), the diff GENUINELY addresses ${it.id} (not a sham or comment-only), and the REQUIRED objective test is present and green. If the fixer report is inconsistent with the diff, re-run the box build/ctest yourself with: ${BUILD}\nFor B5/B6: logic + unit-test green on the 5090, with at-scale validation deferred to a PRO-6000 session, IS a PASS (note it).\n\nON PASS: cd ${R} and git add ONLY the specific changed/new source+test+CMake files for this fix (NEVER git add dot — leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md untracked), then commit with a ROADMAP section-6 message (what+why; the test added; the exact box build/run commands; end with the trailer line: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>). Capture the short hash via git rev-parse --short HEAD.\nON FAIL: revert the working tree so the repo stays green — run: cd ${R} && git checkout -- . && git clean -fd src tests include — and report the exact reason (and whether it is a design call needing human input).\n\nReturn the structured verdict.`

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(`${opts.label}: transient null — retrying once`); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const ledger = []
for (let i = 0; i < ITEMS.length; i++) {
  const it = ITEMS[i]
  phase(it.title)
  log(`=== ${it.id}: fixing ===`)
  const fix = await tryAgent(fixPrompt(it), { label: `fix:${it.id}`, phase: it.title })
  if (fix === null) { ledger.push({ item: it.id, pass: false, regression: false, commit_hash: '', note: 'fix-agent terminal API error after retry — HALT' }); break }
  const v = await tryAgent(verdictPrompt(it, fix), { schema: VERDICT_SCHEMA, label: `verdict:${it.id}`, phase: it.title })
  if (v === null) { ledger.push({ item: it.id, pass: false, regression: false, commit_hash: '', note: 'verdict-agent terminal API error after retry — HALT' }); break }
  ledger.push(v)
  if (!v.pass) { log(`!!! ${it.id} FAILED verdict (${v.note}) — repo reverted; HALTING phase 1`); break }
  log(`+++ ${it.id} committed ${v.commit_hash}`)
}
return { ledger }