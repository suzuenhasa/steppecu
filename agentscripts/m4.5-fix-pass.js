export const meta = {
  name: 'm4.5-fix-pass',
  description: 'M4.5 cleanup fix-pass (the audit docs/cleanup/m4.5/00-overview.md before-M5 backlog) — SAME pattern as fix-pass-phase2: STRICTLY SEQUENTIAL, two agents per item (independent fixer + verdict). Each: clean-HEAD -> fix -> rsync+build+ctest on box5090 (2x RTX 5090) -> verdict (objective gate incl. the locked f2_multigpu_parity bit-identity) -> commit on PASS / revert on FAIL. T1 (the parity-gate VRAM fix) is CRITICAL: halt if it fails (it makes the gate green on the 32GB 5090). The rest skip-and-continue. B2 (P2P transport) is DEFERRED to a PRO-6000 session (P2P path cannot execute on a stock consumer 5090). Retry once on transient API error.',
  phases: [
    { title: 'T1 parity-VRAM-gate' },
    { title: 'B1 fan-out' },
    { title: 'B3 peer-access gate' },
    { title: 'B4 gate doc 3-term' },
    { title: 'B5 validate_partials' },
    { title: 'B6 drop block_sizes' },
    { title: 'B7 host-combine copy_n' },
    { title: 'B8 drop probe backend' },
    { title: 'B9 host tests' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build -GNinja >/tmp/cfg.log 2>&1 && cmake --build build 2>&1 | tail -20 && echo === CTEST === && ctest --test-dir build --output-on-failure 2>&1 | tail -45'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ./build/bin/test_f2_multigpu_parity 2>&1 | tail -40'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics. Branch m4.5-multigpu. M4.5 single-node multi-GPU is committed + bit-identity-parity-proven; THIS is the cleanup fix-pass from the audit (docs/cleanup/m4.5/00-overview.md + the per-unit reviews docs/cleanup/m4.5/<unit>.md — READ the relevant one(s) before editing). The audit verdict: 8.4/10 — correct + parity-locked + exemplary layering, but the multi-GPU speedup is unrealized (devices run serially) and the P2P combine models the anti-patterns it should exemplify. Every fix here MUST preserve the proven §12 bit-identity.',
  'BOX = box5090 (vast.ai 2x RTX 5090, sm_120, CUDA 13, driver 580; 32 GB ea — CONSUMER tier, P2P driver-DISABLED so can_access_peer==false and the combine runs HOST-STAGED here). Build/test ONLY on the box; NOTHING builds locally. Author files locally under ' + R + '. Data on the box: /workspace/data/aadr/{raw, derived_acc P=50, derived_full P=768}.',
  'THE PARITY GATE (the line you must not cross): tests/reference/test_f2_multigpu_parity.cu asserts the production EmulatedFp64{40} multi-GPU combine is memcmp-BIT-IDENTICAL to single-GPU. After T1 lands it runs GREEN on box5090 (derived_acc bit-identical + host-staged degrade validated; derived_full VRAM-skipped on 32GB). EVERY fix must keep it bit-identical — a fix that builds + passes other tests but breaks the f2_multigpu_parity bit-identity is an automatic FAIL. NEVER move a parity-critical reduction onto NCCL AllReduce or change the fixed g=0..G-1 accumulator order.',
  'CONSUMER-TIER REALITY: on this stock 5090 can_access_peer==false, so the P2P device-combine code path is NEVER executed (it degrades to host-staged). Therefore P2P-transport changes cannot be runtime-validated here — that is why B2 (the P2P streaming/double-bounce rework) is DEFERRED to a PRO-6000 session and is NOT in this pass. The items here are all testable on the 5090 via the host-staged path + host-pure logic (incl. B1, the actual speedup).',
  'STANDARDS: architecture.md §2 (DRY/single-source, RAII, fail-fast, testability), §4 (layering: CUDA PRIVATE to steppe_device; core/api/backend.hpp CUDA-FREE), §7 (CUDA idioms incl. grid-stride/streams/events/narrow launch wrappers), §8 (DRY single-home), §9 (Resources/PerGpuResources), §11.4 (SPMG combine), §12 (parity/determinism), §13 (testing). STEPPE_LOG_WARN/STEPPE_DEBUG_ONLY/STEPPE_ASSERT exist (core/internal/{log,host_device}.hpp); core::cdiv/grid_for in launch_config.hpp; STEPPE_CUDA_WARN (non-throwing) in device/cuda/check.cuh. NO synthetic data for accuracy (real AADR). Cite CUDA/cuBLAS/C++ docs where load-bearing.',
].join('\n')

const DEVLOOP = 'DEV LOOP (nothing builds locally): FIRST ensure a clean tree at HEAD so a prior skipped item cannot contaminate this fix — run: ' + CLEAN + ' . Then edit locally; (1) rsync to the box: ' + RSYNC + ' ; (2) build+test on the box: ' + BUILD + ' . If you add a NEW file (source/test), wire it into the CMake (top-level src/*/CMakeLists.txt and/or tests/CMakeLists.txt) first. Build MUST be clean (warnings-as-errors) and ALL ctest green (the 30 pre-existing tests must still pass — no regression), and f2_multigpu_parity MUST stay bit-identical.'

// B2 (P2P streaming/double-bounce/grid-stride) is intentionally ABSENT — deferred to a PRO-6000 session (can_access_peer==false here ⇒ P2P path not executable for validation).
const ITEMS = [
  { id: 'T1', title: 'T1 parity-VRAM-gate', critical: true,
    files: 'tests/reference/test_f2_multigpu_parity.cu',
    fix: 'Make the parity gate GREEN on the 32GB consumer 5090 + harden it (audit test-f2_multigpu_parity F-BUG-2 / F-COV-1 / L8). (a) VRAM-GATE THE SCALE DATASET: before running a dataset, probe free VRAM (cudaMemGetInfo) and estimate the test peak need (the test builds multiple concurrent Resources/backends per dataset); if it will not fit, SKIP that dataset with an EXPLICIT line "[skip] <name> P=.. : insufficient VRAM (need ~X MiB, free ~Y MiB)" — never a silent skip, never an OOM. On box5090 derived_full (P=768) is skipped-with-reason; derived_acc (P=50) runs bit-identical; on a 96GB PRO box both run. (b) F-COV-1: ASSERT EmulatedFp64 ACTUALLY ENGAGED (caps.emulated_fp64_honorable) before trusting the EmuFp64 bit-identical assertions — close the silent-native-fallback blind spot. (c) L8: select the consumer-vs-PRO tier branch by the can_access_peer BICONDITIONAL, not a "PRO"/"GeForce" device-name match.',
    test: 'On box5090: ./build/bin/test_f2_multigpu_parity exits 0 — derived_acc EmuFp64 G==2 host-staged == single-GPU BIT-IDENTICAL + the no-peer tagged-degrade asserted; derived_full cleanly [skip]-with-VRAM-reason (no OOM); ctest f2_multigpu_parity GREEN.' },

  { id: 'B1', title: 'B1 fan-out',
    files: 'src/core/fstats/f2_blocks_multigpu.cpp, src/core/CMakeLists.txt',
    fix: 'THE SPEEDUP (audit X2/B1, HIGH, PARITY-SAFE). f2_blocks_multigpu.cpp:130-154 issues each device compute_f2_blocks STRICTLY SEQUENTIALLY (each blocks on its trailing cudaStreamSynchronize before g+1 is issued) ⇒ wall-clock Σ not max. Fan the G compute_f2_blocks calls out across G HOST THREADS (one per device), each writing its OWN pre-sized partials[g] slot; JOIN before the combine; keep the combine fixed g=0..G-1. No shared mutable state (distinct slots; each backend guard_devices its own device). Capture each worker exception via std::exception_ptr and rethrow the first on join (an escaped exception calls std::terminate). Link Threads::Threads on steppe_core (it does not today). PARITY-SAFE: the combine reads partials[g] in fixed g order AFTER the join barrier; GEMM bits are independent of wall-clock slot.',
    test: 'f2_multigpu_parity stays BIT-IDENTICAL on box5090 (the locked memcmp; G==2 host-staged == single-GPU EmuFp64); build clean; ALL 30 ctest green. (Speedup is measured separately after the pass.)' },

  { id: 'B3', title: 'B3 peer-access gate',
    files: 'src/core/fstats/f2_blocks_multigpu.cpp, include/steppe/config.hpp (doc)',
    fix: 'Wire the DEAD enable_peer_access knob (audit X5/B3, MED). enable_peer_access is read by NO code (grep: only config.hpp:260 + doc) yet the P2P path calls cudaDeviceEnablePeerAccess; the combine gate (f2_blocks_multigpu.cpp:171) is only prefer_p2p_combine && can_access_peer. Widen the gate to prefer_p2p_combine && enable_peer_access && can_access_peer so the documented MAY-WE knob is honored and the enable is reached only with permission. (On box5090 can_access_peer==false ⇒ host-staged regardless, so no behavior change there — but the gate logic becomes correct + testable.)',
    test: 'A check that enable_peer_access=false forces last_combine_path==HostStaged even if can_access_peer were true (assert the gate logic); build+ctest green; f2_multigpu_parity bit-identical.' },

  { id: 'B4', title: 'B4 gate doc 3-term',
    files: 'src/core/fstats/f2_blocks_multigpu.cpp + the 5 doc sites (resources.hpp, p2p_combine.hpp, f2_blocks_multigpu.cpp comments)',
    fix: 'Make the gate match its 5-times-documented 3-term contract (audit X6/B4, MED). The gate is documented "… && G >= 2" in 5 files but the shipped gate is 2-term (G>=2 is enforced structurally by the if (G==1) return). Add the dead-true && G>=2 to the gate so code == doc (changes NO reached path), and collapse the 5 doc restatements to ONE authoritative home + cross-refs (§8 single-home).',
    test: 'build+ctest green; f2_multigpu_parity bit-identical (no reached path changes).' },

  { id: 'B5', title: 'B5 validate_partials',
    files: 'new core header (e.g. src/core/fstats/f2_partials_validate.hpp) + src/core/fstats/f2_combine.cpp + src/device/cuda/p2p_combine.cu',
    fix: 'Single-home validate_partials + close the short-partial OOB gap (audit X7/B5, MED). validate_partials is duplicated byte-for-byte between f2_combine.cpp and p2p_combine.cu (they MUST reject identically or parity-neutrality breaks). Hoist ONE CUDA-free validate_f2_partials(partials, shards, P, n_block_full) into a shared core header (it names only F2BlockTensor/DeviceShard); both tiers call it; the P2P side adds only the device_ids.size() check. ADD the currently-missing part.f2.size()==P*P*n_block check (the shared short-partial OOB gap). Remove the duplicated dead n_block_full<0?0: ternaries.',
    test: 'build+ctest green; f2_multigpu_parity bit-identical; (optional) a host unit test that validate_f2_partials throws on malformed/short partials.' },

  { id: 'B6', title: 'B6 drop block_sizes',
    files: 'src/device/shard_plan.hpp, src/device/shard_plan.cpp, src/core/fstats/f2_blocks_multigpu.cpp',
    fix: 'Drop the redundant block_sizes parameter; derive sizes from ranges (audit X1/B6, MED). plan_block_shards is handed BOTH ranges and a block_sizes that is literally ranges[b].size() narrowed to int. Remove the block_sizes parameter; derive sizes from ranges[b].size() (already long) inside the planner. One move collapses: the caller redundant vector+loop+long->int narrowing (removes a latent INT_MAX truncation), the planner unchecked parallel-array contract, its sign hole, and half its cast scatter. The shard plan must be byte-identical.',
    test: 'build+ctest green; f2_multigpu_parity bit-identical (plan unchanged); shard plan still correct (block_partition_aadr_consistency + any shard test green).' },

  { id: 'B7', title: 'B7 host-combine copy_n',
    files: 'src/core/fstats/f2_combine.cpp',
    fix: 'Host combine: std::copy_n placement, perf + removes the latent -0.0 flip (audit W7/B7, MED, STRICTLY SAFER). The host-staged combine sums each compact partial onto a 0.0-init full tensor with a scalar += triple loop. Because shards are DISJOINT (each block owned by one device) this is PLACEMENT not accumulation — replace the inner += with std::copy_n of the contiguous owned run (memcpy-grade). This ALSO removes the latent -0.0 bit-flip: (+0.0)+(-0.0)=+0.0 changes the bit pattern, while single-GPU computes the slab directly; std::copy is strictly MORE faithful. Correct the wrong "x+0.0==x for all finite x" doc.',
    test: 'build+ctest green; f2_multigpu_parity BIT-IDENTICAL (std::copy is strictly safer than +=, never worse); (optional) a host unit test of the placement + the -0.0 case.' },

  { id: 'B8', title: 'B8 drop probe backend',
    files: 'src/device/resources.cpp, src/device/backend_factory.hpp, src/device/cuda/cuda_backend.cu',
    fix: 'Drop the throwaway device-0 probe backend + add §9 ordinal validation (audit W8/B8, MED, HIGH-perf cold-start). resolve_device_order builds a THROWAWAY device-0 CudaBackend (cuBLAS create + 64 MiB workspace alloc + a discarded full peer-scan probe) just to read device_count, then probes device 0 AGAIN as gpus[0], and leaks a cudaSetDevice(0) ambient side effect. Replace it with a CUDA-free visible_device_count() factory query (one count query serves both); REJECT duplicate / out-of-range device ordinals in DeviceConfig (the §9 fail-fast that is missing); remove the leaked ambient cudaSetDevice.',
    test: 'build+ctest green; resources_build still green; f2_multigpu_parity bit-identical; a check that a duplicate/out-of-range ordinal in DeviceConfig::devices throws (fail-fast).' },

  { id: 'B9', title: 'B9 host tests',
    files: 'new tests/unit/test_shard_plan.cpp, tests/unit/test_f2_combine.cpp, tests/unit/test_f2_blocks_multigpu.cpp (+ tests/CMakeLists.txt) + a fake ComputeBackend',
    fix: 'GPU-free host unit tests for the host-pure logic (audit B9, MED). The planner / combine / orchestrator are host-pure but exercised ONLY through the one GPU .cu. Add GPU-FREE host tests: (1) shard_plan — tiling/skew/edges (n_block<G, empty shard, uneven); (2) f2_combine — placement + fixed g-order + every validate throw + the -0.0 case (post-B7); (3) f2_blocks_multigpu — the gate predicate (post-B3/B4) + sub-view/local-id math + empty/n_block<G, driven by a FAKE ComputeBackend (extract the orchestrator core if needed). These are the fast inner-loop gate.',
    test: 'the new host tests compile + run GREEN (no GPU/no real data needed); wired into tests/CMakeLists.txt; ALL ctest green; f2_multigpu_parity bit-identical.' },
]

const fixPrompt = (it) =>
  'You are a senior CUDA/C++ engineer applying ONE cleanup fix to steppe M4.5 (branch m4.5-multigpu). Do NOT commit (the independent verdict agent commits).\n\n' + STD + '\n\nFIX ' + it.id + ': ' + it.fix + '\nPRIMARY FILES: ' + it.files + '\nREQUIRED TEST / OBJECTIVE GATE: ' + it.test + '\n\nApply the fix per the architecture standards + the relevant docs/cleanup/m4.5/ review; update any doc-comments the fix makes stale. Where a claim depends on CUDA/cuBLAS or C++-stdlib behavior, cite the docs. Add the required test and wire it into the CMake if new.\n\n' + DEVLOOP + '\n\nCRITICAL: this fix MUST keep test_f2_multigpu_parity BIT-IDENTICAL (run it directly to confirm: ' + PARITY + '). Return a thorough report: (1) every file changed + what changed; (2) the test added + what it asserts; (3) the FULL build result; (4) the FULL ctest result (paste the summary + the f2_multigpu_parity line) + confirmation the parity bit-identity held. If you CANNOT reach a clean green build + green ctest + bit-identical parity, do NOT pretend success — report exactly what blocked it (and whether the parity bit-identity itself broke — a real correctness problem).'

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item', 'pass', 'regression', 'parity_bit_identical', 'green_count', 'commit_hash', 'note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if build clean (warnings-as-errors) + ALL ctest green + the diff genuinely addresses the finding (not a sham/comment-only/weakened test) + NO regression + f2_multigpu_parity GREEN and BIT-IDENTICAL' },
    regression: { type: 'boolean' },
    parity_bit_identical: { type: 'boolean', description: 'true if test_f2_multigpu_parity is green AND its EmuFp64 G==2 host-staged == single-GPU assertion is bit-identical (memcmp, not tolerance). A fix that breaks this is an automatic FAIL.' },
    green_count: { type: 'integer', description: 'number of ctest tests passing (>=30 expected after T1)' },
    commit_hash: { type: 'string', description: 'short hash if committed on PASS, else empty' },
    note: { type: 'string', description: 'one-line rationale; for FAIL, the exact reason + whether the parity bit-identity itself broke (a real correctness/design problem needing human input)' },
  },
}

const verdictPrompt = (it, fixReport) =>
  'You are the INDEPENDENT VERDICT for cleanup fix ' + it.id + ' of steppe M4.5 (you did NOT write the fix — be adversarial, ESPECIALLY about the parity bit-identity). The fixer reported:\n<<<\n' + fixReport + '\n>>>\n\n' + STD + '\n\nThe finding being fixed: ' + it.fix + '\nThe objective gate: ' + it.test + '\n\nDO: (1) inspect the actual uncommitted changes — run: cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff ; (2) judge PASS only if ALL hold — build clean (warnings-as-errors), ALL ctest green (NO regression vs the 30 prior tests), the diff GENUINELY addresses ' + it.id + ' (not a sham/comment-only/test-weakened), the required objective test present + green, AND test_f2_multigpu_parity is GREEN and BIT-IDENTICAL (read its output: the EmuFp64 G==2 host-staged == single-GPU line must say bit-identical, not tolerance; the no-peer tagged-degrade asserted). If the fixer report is inconsistent with the diff, RE-RUN the box build/ctest yourself: ' + BUILD + ' and the parity binary: ' + PARITY + ' . A fix that builds + passes other tests but breaks the f2_multigpu_parity bit-identity is an automatic FAIL. NEVER accept an NCCL-AllReduce / reordered-accumulator change on the combine.\n\nON PASS: cd ' + R + ' and git add ONLY the specific changed/new source+test+CMake+doc files for this fix (NEVER git add dot — leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md untracked), then commit with a ROADMAP §6 message (what+why; the test added; the box build/run commands; end with the trailer line: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>). Capture the short hash via git rev-parse --short HEAD.\nON FAIL: revert the working tree so the repo stays green — run: ' + CLEAN + ' — and report the exact reason (and whether the parity bit-identity itself broke = a real correctness/design problem needing human input).\n\nReturn the structured verdict.'

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
  if (fix === null) {
    ledger.push({ item: it.id, pass: false, regression: false, parity_bit_identical: false, green_count: 0, commit_hash: '', note: 'fix-agent terminal API error after retry — SKIPPED' })
    if (it.critical) { log('!!! critical ' + it.id + ' fix-agent died — HALT'); break }
    continue
  }
  const v = await tryAgent(verdictPrompt(it, fix), { schema: VERDICT_SCHEMA, label: 'verdict:' + it.id, phase: it.title })
  if (v === null) {
    ledger.push({ item: it.id, pass: false, regression: false, parity_bit_identical: false, green_count: 0, commit_hash: '', note: 'verdict-agent terminal API error after retry — SKIPPED' })
    if (it.critical) { log('!!! critical ' + it.id + ' verdict-agent died — HALT'); break }
    continue
  }
  ledger.push(v)
  if (v.pass) { log('+++ ' + it.id + ' committed ' + v.commit_hash + ' (' + v.green_count + ' green, parity_bit_identical=' + v.parity_bit_identical + ')') }
  else {
    log('--- ' + it.id + ' FAILED verdict (' + v.note + ') — reverted')
    if (it.critical) { log('!!! ' + it.id + ' is CRITICAL (the parity gate must be green for the rest) — HALT'); break }
    log('    (skip-and-continue: ' + it.id + ' is independent)')
  }
}
return { ledger }
