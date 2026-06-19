export const meta = {
  name: 'm4.5-perf-fix-pass',
  description: 'Make M4.5 multi-GPU actually FASTER (the perf-discovery plan docs/cleanup/m4.5/perf-discovery.md). SAME pattern as fix-pass-phase2: STRICTLY SEQUENTIAL, two agents per item (independent fixer + verdict). RELEASE build (build-rel, NDEBUG) — perf numbers are only meaningful in Release. Each: clean-HEAD -> fix -> Release build + parity + bench on rtxbox -> verdict (parity memcmp BIT-IDENTICAL + no regression + the diff is sound + bench G==2 not worse) -> commit-green / revert. P0 (Release warning-clean) is CRITICAL: halt if it fails (the rest build Release). P1-P4 skip-and-continue. Retry once on transient.',
  phases: [
    { title: 'P0 release-clean' },
    { title: 'P1 combine resize-not-zero' },
    { title: 'P2 real per-device Stream' },
    { title: 'P3 buffer reuse / pool' },
    { title: 'P4 pinned async H2D' },
    { title: 'Final bench' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh rtxbox'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ rtxbox:/workspace/steppe/'
// RELEASE build (NDEBUG => the per-kernel debug cudaDeviceSynchronize is gone) in a dedicated dir; sm_120 (project default arch).
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -20 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -40'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ./build-rel/bin/test_f2_multigpu_parity 2>&1 | tail -45'"
const BENCH = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/bench_f2_multigpu /workspace/data/aadr 200 400 768 2>&1 | grep -vE \"P2P combine unavailable\"'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimpl of ADMIXTOOLS 2 f-statistics. Branch m4.5-multigpu. M4.5 multi-GPU is DONE + bit-identity-proven, but MEASURED SLOWER than single-GPU (G2/G1 = 1.04x@P200 -> 0.70x@P768, Release). This fix-pass makes it FASTER per the MEASURED plan in docs/cleanup/m4.5/perf-discovery.md (READ IT — the ranked root causes, the nsys numbers, the parity-safe levers). The fan-out (std::jthread per device, f2_blocks_multigpu_core.cpp:109) is correct but starved.',
  'BOX = rtxbox (2x RTX PRO 6000 Blackwell, sm_120, CUDA 13, 96 GB ea, REAL P2P can_access_peer=true). ssh rtxbox; nvcc not on PATH -> ' + PATHENV + ' . Data /workspace/data/aadr/{raw, derived_acc, derived_full P=768}. NOTHING builds locally.',
  'BUILD RELEASE ONLY (build-rel, -DCMAKE_BUILD_TYPE=Release => NDEBUG): perf numbers are ONLY meaningful in Release (the debug build device-syncs after every kernel — voids timing; see the perf-bench-release-build memory). The default `build` dir is untouched.',
  'THE PARITY LAW (the line you must not cross): test_f2_multigpu_parity asserts the production EmulatedFp64{40} multi-GPU combine is memcmp-BIT-IDENTICAL to single-GPU (host-staged AND P2P), on derived_acc + derived_full P=768. EVERY fix MUST keep it bit-identical (parity does NOT depend on build type). A perf fix that builds + benches faster but breaks the parity memcmp is an AUTOMATIC FAIL. NEVER change the fixed g=0..G-1 combine order, NEVER NCCL AllReduce, NEVER reduce mantissa/precision. These are DATA-MOVEMENT / RESOURCE / SCHEDULING changes only — not arithmetic.',
  'STANDARDS: architecture.md §7 (CUDA idioms: RAII Stream/Event, async, pinned, narrow launch wrappers), §11.4 (SPMG), §12 (parity + the §12 single-stream-PER-DEVICE determinism rule: each device runs its statistic path on ONE stream — a real non-blocking per-device stream is fine and still deterministic; keep B1`s CublasHandle (stream,workspace) re-application). RAII for all device memory/handles. Cite CUDA docs (streams/events/overlap, pinned cudaMallocHost/HostRegister, cudaMallocAsync pool) for load-bearing claims.',
].join('\n')

const DEVLOOP = 'DEV LOOP (nothing builds locally): FIRST clean the tree at HEAD: ' + CLEAN + ' . Then edit locally; (1) rsync: ' + RSYNC + ' ; (2) RELEASE build + ctest: ' + BUILD + ' ; (3) parity: ' + PARITY + ' ; (4) bench: ' + BENCH + ' . Wire any new file into CMake first. Build MUST be clean (warnings-as-errors) in RELEASE, ctest green (no regression — note: assert/death tests are debug-only and inactive in Release; that is NOT a regression), parity BIT-IDENTICAL, and the bench G==2 must NOT regress (ideally improve).'

const ITEMS = [
  { id: 'P0', title: 'P0 release-clean', critical: true,
    files: 'src/core/internal/launch_config.hpp (grid_for unused param ~:114), the 2 death-test TUs with an unused `child_aborts` under NDEBUG (grep tests/ for child_aborts)',
    fix: 'Make the RELEASE build (-DCMAKE_BUILD_TYPE=Release => NDEBUG) clean under warnings-as-errors WITHOUT cache hacks. In Release the debug asserts/death-tests vanish, leaving 3 -Werror warnings: unused-parameter in launch_config.hpp grid_for (the param is only used in an NDEBUG-gated assert) and unused-function `child_aborts` in two death-test TUs. Fix idiomatically: mark the genuinely-conditionally-used param/function `[[maybe_unused]]` (or guard the death-test helper so it is not defined when its death-test is inactive). Do NOT weaken warnings; make the code honest under both build types.',
    test: 'A clean RELEASE build (build-rel): cmake -DCMAKE_BUILD_TYPE=Release configure + build with ZERO warnings under -Werror (CXX) / --Werror all-warnings (CUDA); ctest --test-dir build-rel green (debug-only death tests inactive is expected, not a regression); parity bit-identical. This is the baseline the rest build on.' },

  { id: 'P1', title: 'P1 combine resize-not-zero',
    files: 'src/device/cuda/p2p_combine.cu (~:180-181 the accumulator zeroing), src/core/fstats/f2_combine.cpp (~:64-65 assign(total,0.0))',
    fix: 'THE #1 MEASURED COST (~1440 ms/run, perf-discovery P1/W9). The combine allocates the full [P^2*n_block] f2+vpair output and ZERO-INITS it (assign(total,0.0) host / cudaMemset device) then OVERWRITES every element by placing the disjoint per-device partials (block-aligned shard => each block owned by exactly ONE device => the placement covers 100% of the tensor; B7 already made the host combine a contiguous std::copy_n). The zero-init is therefore REDUNDANT. Replace assign(total,0.0) with resize(total) (no zero-fill) on the host side, and drop the cudaMemset on the P2P device accumulator, ONLY AFTER proving full coverage. CAVEAT (yes-if-careful): if ANY output element is not written by the placement, resize leaves garbage and parity FAILS — so the parity memcmp is the guard; if coverage is not 100% (e.g. a gap block), keep the zero-init for the uncovered region only. Verify every block 0..n_block-1 is assigned to exactly one shard.',
    test: 'Parity BIT-IDENTICAL (the memcmp PROVES full coverage — garbage from a missed element would fail it) on derived_acc + derived_full; bench G==2@768 drops by ~1.4 s vs the P0 baseline; build+ctest green.' },

  { id: 'P2', title: 'P2 real per-device Stream',
    files: 'src/device/cuda/cuda_backend.cu (stream_ = nullptr ~:596; the ctor; all cudaMemcpyAsync/launch/sync sites), src/device/cuda/stream.hpp (the RAII Stream wrapper to instantiate)',
    fix: 'THE OVERLAP PRECONDITION (perf-discovery P2/F1; only 18% GPU overlap measured). CudaBackend uses the NULL legacy default stream (stream_ = nullptr), so the two fan-out device-threads implicitly serialize and their GEMMs do not overlap. Give each CudaBackend a REAL owning non-blocking Stream (the stream.hpp RAII wrapper, cudaStreamNonBlocking) created in the ctor; bind cuBLAS to it (keep B1`s CublasHandle.set_stream re-applying the workspace) and route ALL launches + cudaMemcpyAsync + the trailing cudaStreamSynchronize through it. This is the long-standing "default-stream debt" (audit Theme-1). §12: ONE stream PER DEVICE on the statistic path is REQUIRED for determinism — a single non-blocking per-device stream satisfies it (do NOT add multiple statistic streams). Bit-identity is unaffected (stream choice does not change the math).',
    test: 'Parity BIT-IDENTICAL; bench G==2 improves (the two devices now overlap — re-profile or just the wall-clock); build+ctest green; the §12 single-stream-per-device determinism (f2_determinism test) still green.' },

  { id: 'P3', title: 'P3 buffer reuse / pool',
    files: 'src/device/cuda/cuda_backend.cu (per-chunk DeviceBuffer allocs in compute_f2_blocks ~:355-388), src/device/cuda/device_buffer.cuh',
    fix: 'STOP THE PER-CHUNK ALLOC CHURN (perf-discovery P3/L4; 645 cudaMalloc + 648 cudaFree, global driver lock that serializes the two fan-out threads). The bucket loop allocates+frees dQ_raw/dV_raw/dN_raw/dIds/dQg/dVg/dSg/dGg/dVpairg/dRg EVERY chunk. Pre-allocate ONCE at MAX chunk size (the largest padded bucket the budget allows) OUTSIDE the loop and REUSE across chunks (or a cudaMallocAsync pool, release-threshold UINT64_MAX). cudaMalloc/cudaFree are device-wide-synchronizing + take the driver lock; eliminating per-chunk churn lets the two devices stop serializing. Parity-safe (same buffers, same math — only WHEN they are allocated changes). Respect the §11.2 VRAM budget (the max-size pre-alloc must still fit).',
    test: 'Parity BIT-IDENTICAL; nsys/the bench shows the cudaMalloc/cudaFree call count collapse (from ~645 to O(1) per device) and G==2 improves (esp. with P2 streams); build+ctest green; vram_budget_unit still green.' },

  { id: 'P4', title: 'P4 pinned async H2D',
    files: 'src/device/cuda/cuda_backend.cu (the Q/V/N H2D uploads), a small RAII PinnedBuffer (new src/device/cuda/pinned_buffer.cuh or extend device_buffer.cuh)',
    fix: 'PINNED STAGING so async H2D actually overlaps (perf-discovery P4/L2; ~44% pageable cudaMemcpyAsync is effectively blocking). Stage the per-device Q/V/N (and result D2H) host buffers through PINNED memory (cudaMallocHost, or cudaHostRegister an existing buffer) so cudaMemcpyAsync on the per-device stream (P2) genuinely overlaps copy with compute. Pin only the needed slots (mind RLIMIT_MEMLOCK; fallback to pageable+warn if pinning fails). RAII wrapper, dtor never throws. Parity-safe (data movement only).',
    test: 'Parity BIT-IDENTICAL; bench G==2 improves (copies overlap compute); build+ctest green; pinning failure degrades gracefully (no crash).' },
]

const fixPrompt = (it) =>
  'You are a senior CUDA/C++ engineer applying ONE perf fix to steppe M4.5 (branch m4.5-multigpu). RELEASE build only. Do NOT commit (the independent verdict agent commits).\n\n' + STD + '\n\nFIX ' + it.id + ': ' + it.fix + '\nPRIMARY FILES: ' + it.files + '\nOBJECTIVE GATE: ' + it.test + '\n\nApply the fix per the perf-discovery plan + the architecture §7/§11.4/§12 standards; update stale doc-comments (e.g. the false "per-device default streams overlap" comment for P2). Cite CUDA docs for load-bearing claims.\n\n' + DEVLOOP + '\n\nCRITICAL: keep test_f2_multigpu_parity BIT-IDENTICAL (run ' + PARITY + ') and do not regress the bench (run ' + BENCH + '). Return a thorough report: (1) files changed + what; (2) the FULL Release build result; (3) the parity result (paste the EmuFp64 G==2 == single-GPU bit-identical lines); (4) the bench table (G1/G2/speedup at 200/400/768) vs the prior numbers. If you cannot reach clean Release build + green ctest + bit-identical parity, do NOT pretend — report exactly what blocked it (and whether parity itself broke).'

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item', 'pass', 'regression', 'parity_bit_identical', 'g2_768_ms', 'commit_hash', 'note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if Release build clean (warnings-as-errors) + ctest green (no regression; debug-only death tests inactive is OK) + the diff genuinely implements the item (not a sham/weakened) + parity memcmp BIT-IDENTICAL + the bench G==2 did NOT regress' },
    regression: { type: 'boolean' },
    parity_bit_identical: { type: 'boolean', description: 'true if test_f2_multigpu_parity EmuFp64 G==2 (host-staged AND P2P) == single-GPU is memcmp-bit-identical' },
    g2_768_ms: { type: 'number', description: 'the bench G==2 wall-clock at P=768 (ms) after this fix — to track the speedup progression' },
    commit_hash: { type: 'string', description: 'short hash if committed on PASS, else empty' },
    note: { type: 'string', description: 'one-line rationale incl. the G1/G2 speedup at P=768 now; for FAIL, the exact reason + whether parity broke' },
  },
}

const verdictPrompt = (it, fixReport) =>
  'You are the INDEPENDENT VERDICT for perf fix ' + it.id + ' of steppe M4.5 (you did NOT write it — be adversarial, ESPECIALLY about parity bit-identity and that the perf change is real, not a sham). The fixer reported:\n<<<\n' + fixReport + '\n>>>\n\n' + STD + '\n\nThe finding: ' + it.fix + '\nThe gate: ' + it.test + '\n\nDO: (1) inspect the uncommitted changes: cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff ; (2) judge PASS only if ALL hold: the diff GENUINELY implements ' + it.id + ' (not a sham/comment-only/weakened), RELEASE build clean (warnings-as-errors), ctest green (no regression vs the Release baseline; debug-only death tests being inactive in Release is NOT a regression), parity test_f2_multigpu_parity BIT-IDENTICAL (EmuFp64 G==2 host-staged AND P2P == single-GPU, memcmp — read the output), and the bench G==2 did NOT regress (record g2_768_ms). If the report looks inconsistent, RE-RUN yourself: ' + BUILD + ' ; ' + PARITY + ' ; ' + BENCH + ' . A fix that benches faster but breaks the parity memcmp is an AUTOMATIC FAIL. Reject any change to the fixed g=0..G-1 combine order / NCCL AllReduce / precision.\n\nON PASS: cd ' + R + ' and git add ONLY the specific changed/new files for this fix (NEVER git add dot — leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md, build-rel untracked), commit with a ROADMAP §6 message (what+why; the measured before/after G==2; the box build/run cmds) ending with: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.\nON FAIL: revert: ' + CLEAN + ' — report the exact reason (and whether parity broke).\n\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retrying once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  log('=== ' + it.id + ': fixing ===')
  const fix = await tryAgent(fixPrompt(it), { label: 'fix:' + it.id, phase: it.title })
  if (fix === null) { ledger.push({ item: it.id, pass: false, regression: false, parity_bit_identical: false, g2_768_ms: 0, commit_hash: '', note: 'fix-agent terminal API error — ' + (it.critical ? 'HALT' : 'SKIPPED') }); if (it.critical) break; else continue }
  const v = await tryAgent(verdictPrompt(it, fix), { schema: VERDICT_SCHEMA, label: 'verdict:' + it.id, phase: it.title })
  if (v === null) { ledger.push({ item: it.id, pass: false, regression: false, parity_bit_identical: false, g2_768_ms: 0, commit_hash: '', note: 'verdict-agent terminal API error — ' + (it.critical ? 'HALT' : 'SKIPPED') }); if (it.critical) break; else continue }
  ledger.push(v)
  if (v.pass) log('+++ ' + it.id + ' committed ' + v.commit_hash + ' (parity=' + v.parity_bit_identical + ', G2@768=' + v.g2_768_ms + 'ms) — ' + v.note)
  else { log('--- ' + it.id + ' FAILED (' + v.note + ') — reverted'); if (it.critical) { log('!!! ' + it.id + ' CRITICAL — HALT'); break } }
}

phase('Final bench')
log('=== final Release bench (cumulative speedup) ===')
const finalBench = await tryAgent([
  'Run the FINAL Release bench to report the cumulative multi-GPU speedup after the perf fix-pass, and confirm parity still holds. No edits.', STD, '',
  'On rtxbox: (1) ensure a clean Release build: ' + BUILD + ' ; (2) parity: ' + PARITY + ' ; (3) FULL bench sweep: ' + SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ./build-rel/bin/bench_f2_multigpu /workspace/data/aadr 200 400 600 768 2>&1 | grep -vE \"P2P combine unavailable\"' .",
  'Return: the final G1/G2/speedup table at 200/400/600/768, vs the PRE-fix-pass baseline (1.04x@200 -> 0.70x@768), the parity result, and a one-line verdict: is multi-GPU now FASTER than single-GPU, and by how much at P=768?',
].join('\n'), { label: 'final-bench', phase: 'Final bench' })

return { ledger, finalBench }
