export const meta = {
  name: 'm4.5-b2-p2p-fix-pass',
  description: 'STAGED for a PRO-6000 (rtxbox) session — the B2 P2P-transport rework deferred from the 5090 fix-pass (the P2P path only executes when can_access_peer==true). Same pattern as fix-pass-phase2: STRICTLY SEQUENTIAL, two agents per item (independent fixer + verdict), build+ctest on rtxbox, commit-green/revert. Every fix re-gated by the locked f2_multigpu_parity bit-identity WITH THE P2P PATH ACTUALLY EXERCISED (P2P == host-staged == single-GPU, EmuFp64, on derived_acc + derived_full P=768). Fixes the data-bounce + sequential-transfer + grid-stride + casting warts in device/cuda/p2p_combine.cu. Skip-and-continue. Retry once on transient API error.',
  phases: [
    { title: 'P1 grid-stride+fuse' },
    { title: 'P2 hoist peer-enable' },
    { title: 'P3 kill double-bounce' },
    { title: 'P4 streamed P2P' },
  ],
}

// =====================================================================================
// PRE-FLIGHT (do these BEFORE triggering — the main loop, not the workflow, handles them):
//   1. rtxbox is a Verda rental and EPHEMERAL — the spun-down instance's IP is stale.
//      Update the `rtxbox` entry in ~/.ssh/config to the NEW instance (HostName/Port),
//      key ~/.ssh/id_vastai. Confirm `ssh rtxbox 'nvidia-smi -L'` shows 2x RTX PRO 6000.
//   2. nvcc is NOT on PATH on rtxbox -> the BUILD const exports /usr/local/cuda/bin.
//   3. rsync the repo + confirm data: `ssh rtxbox 'ls /workspace/data/aadr'` must have
//      raw + derived_acc + derived_full (regen derived_full via build_tgeno_matrix.py
//      --auto-top 768 if missing; see the rtxbox memory).
//   4. RUN THIS ONLY AFTER the 5090 fix-pass (m4.5-fix-pass.js) has landed B1/B3-B9 —
//      this pass builds on that cleaned-up code (esp. B3's enable_peer_access gate).
// =====================================================================================

const R = '/home/suzunik/steppe'
const SSH = 'ssh rtxbox'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh ' + R + '/ rtxbox:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build -GNinja >/tmp/cfg.log 2>&1 && cmake --build build 2>&1 | tail -20 && echo === CTEST === && ctest --test-dir build --output-on-failure 2>&1 | tail -45'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ./build/bin/test_f2_multigpu_parity 2>&1 | tail -55'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics. Branch m4.5-multigpu. M4.5 multi-GPU is committed + bit-identity-proven; the 5090 fix-pass landed the 5090-validatable cleanup (B1 fan-out, B3-B9). THIS is the DEFERRED B2: the P2P device-resident combine (device/cuda/p2p_combine.{cu,hpp}) transport rework — the named "prime perf target" the audit scored 7.5/10 because it models the §7 anti-patterns it should exemplify. READ docs/cleanup/m4.5/device-cuda-p2p_combine.md + docs/cleanup/m4.5/00-overview.md (X3/X4 + the W1-W9 table + L1/L2) before editing.',
  'BOX = rtxbox (2x RTX PRO 6000 Blackwell, sm_120, CUDA 13, driver 580; 96 GB ea). CRITICAL: this is the box where can_access_peer==true and the P2P device-combine path ACTUALLY EXECUTES (MEASURED cudaMemcpyPeer byte-exact 55.6 GB/s) — that is WHY B2 must run here and not on the 5090. Build/test ONLY on the box; NOTHING builds locally. Author files locally under ' + R + '. Data: /workspace/data/aadr/{raw, derived_acc P=50, derived_full P=768}; 96 GB fits both at P2P.',
  'THE PARITY GATE (the line you must not cross): tests/reference/test_f2_multigpu_parity.cu. On rtxbox it runs the P2P path FOR REAL and asserts P2P device-combine == host-staged == single-GPU, memcmp-BIT-IDENTICAL, for EmulatedFp64{40} on derived_acc AND derived_full (P=768, fits in 96 GB). EVERY fix here must keep that bit-identical with the P2P path exercised (resG2p.last_combine_path == P2pDeviceResident). A fix that builds + passes other tests but breaks the P2P bit-identity, or that silently degrades to host-staged (so P2P is no longer exercised), is an automatic FAIL. NEVER NCCL AllReduce; NEVER reorder the fixed g=0..G-1 accumulator adds (§12) — only the COPIES may overlap.',
  'STANDARDS: architecture.md §2 (DRY/RAII/fail-fast), §4 (CUDA PRIVATE to steppe_device), §7 (CUDA idioms: GRID-STRIDE loops, STREAMS/EVENTS for overlap, narrow void launch_* wrappers, async + PINNED staging, RAII), §11.4 (SPMG combine — device-resident cudaMemcpyPeer in fixed g order), §12 (parity/determinism). STEPPE_CUDA_WARN (non-throwing) in check.cuh; core::cdiv/grid_for in launch_config.hpp. Cite the CUDA docs (grid-stride loops, streams/events, cudaMemcpyPeer/Async, peer access) for every device-behavior claim.',
].join('\n')

const DEVLOOP = 'DEV LOOP (nothing builds locally): FIRST clean the tree at HEAD: ' + CLEAN + ' . Then edit locally; (1) rsync: ' + RSYNC + ' ; (2) build+test: ' + BUILD + ' . Wire any new file into CMake first. Build MUST be clean (warnings-as-errors), ALL ctest green (no regression), and f2_multigpu_parity BIT-IDENTICAL WITH THE P2P PATH EXERCISED (run it: ' + PARITY + ' — confirm the [PASS] P2P device-resident == single-GPU lines AND resG2p.last_combine_path == P2pDeviceResident).'

const ITEMS = [
  { id: 'P1', title: 'P1 grid-stride+fuse',
    files: 'src/device/cuda/p2p_combine.cu',
    fix: 'Grid-stride the place-add kernel + fuse f2+vpair behind a narrow wrapper (audit W4+W5; p2p_combine review). place_add_f2_kernel is one-element-per-thread (k = blockIdx.x*blockDim.x + threadIdx.x, NO grid-stride) guarded by a debug-only grid-extent STEPPE_ASSERT that is a no-op under NDEBUG (a RELEASE-only silent-truncation trap). Rewrite it as a GRID-STRIDE loop (k += gridDim.x*blockDim.x) so a fixed grid covers any element count safely. Then FUSE the f2 and vpair place-adds into ONE launch behind a narrow void launch_place_add(acc_f2, acc_vp, src_f2, src_vp, off, n, stream) wrapper (§7 idiom; matches the host baseline single interleaved loop). PARITY-SAFE: each element written exactly once with the same += operands/order; disjoint buffers.',
    test: 'On rtxbox: f2_multigpu_parity P2P device-resident == single-GPU BIT-IDENTICAL (EmuFp64, derived_acc + derived_full), resG2p path == P2pDeviceResident; build+ctest green.' },

  { id: 'P2', title: 'P2 hoist peer-enable',
    files: 'src/device/cuda/p2p_combine.cu, src/device/resources.cpp, src/device/p2p_combine.hpp',
    fix: 'Hoist cudaDeviceEnablePeerAccess to once-per-(root,peer) + delete the sticky-scrub (audit W6/X4). Today cudaDeviceEnablePeerAccess is called INSIDE the per-partial loop (p2p_combine.cu ~265) every iteration (cudaErrorPeerAccessAlreadyEnabled spam). Establish peer access ONCE per (root,peer) pair — ideally in build_resources at Resources construction, gated on DeviceConfig::enable_peer_access (composing with the 5090 pass B3 gate so the MAY-WE knob is honored) — and remove the per-iteration enable + any cudaGetLastError sticky-error scrub. Keep it NON-throwing (STEPPE_CUDA_WARN: AlreadyEnabled is expected). PARITY-NEUTRAL (transport setup only).',
    test: 'On rtxbox: peer access enabled once (not per-partial — verify no repeated AlreadyEnabled warns in the combine loop); f2_multigpu_parity P2P bit-identical; enable_peer_access=false forces host-staged (degrade tag); build+ctest green.' },

  { id: 'P3', title: 'P3 kill double-bounce',
    files: 'src/device/cuda/cuda_backend.cu, src/device/backend.hpp (the seam), src/device/cuda/p2p_combine.cu',
    fix: 'THE DATA-BOUNCE FIX (audit X3/W2 + L1, HIGH, yes-if-CAREFUL). Today a peer-owned partial does a host->peer->root DOUBLE-BOUNCE: compute_f2_blocks returns a HOST tensor + frees its device buffers, so the P2P combine re-uploads host->peer (H2D) then cudaMemcpyPeer peer->root — two copies of the same bytes + a per-iteration peer malloc/free, for data that just left the device. Fix: let the per-device partial stay DEVICE-RESIDENT (add a device-resident return path / handle to the compute_f2_blocks seam, or an opt-in that retains the device buffer) so the P2P combine does a genuine peer->root cudaMemcpyPeer pull with NO host round-trip. CRITICAL FOR PARITY: the retained device partial must be BYTE-EXACT identical to today host tensor, and the per-block GEMM batch shape MUST be UNPERTURBED (the EmuFp64 bit-identity depends on the grouped strided-batched batchCount; do not change it). The host-staged tier keeps its (inherent, sanctioned) D2H gather unchanged — only the P2P tier goes device-resident. Layering: keep backend.hpp CUDA-free (a device-resident handle must not leak a CUDA type across the seam — use an opaque/owned-by-device mechanism).',
    test: 'On rtxbox: f2_multigpu_parity P2P device-resident == host-staged == single-GPU BIT-IDENTICAL (EmuFp64, derived_acc + derived_full P=768); resG2p path == P2pDeviceResident; the P2P path no longer does a host->peer H2D for peer partials (verify in the diff / a quick nsys that cudaMemcpyPeer is peer->root with no preceding H2D of the same bytes); build+ctest green, no regression.' },

  { id: 'P4', title: 'P4 streamed P2P',
    files: 'src/device/cuda/p2p_combine.cu (+ a PinnedBuffer wrapper if needed: src/device/cuda/device_buffer.cuh or new pinned_buffer.cuh)',
    fix: 'Streamed P2P: overlap the copies, drop the full-device fence (audit W3+W9/X4). Today every peer pull is followed by a full cudaDeviceSynchronize() (the heaviest fence) and everything runs on the NULL stream (cross-device-serializes). Replace with: a K-deep staging RING (so partial g+1 copy does not overwrite g still being place-added — the current single reused dStage WAR hazard), per-pull CUDA STREAMS + EVENTS that OVERLAP the cross-device COPIES, and the place-adds SERIALIZED on ONE accumulator stream in FIXED g=0..G-1 order. Use PINNED host staging for any async H2D (needs a small RAII PinnedBuffer wrapper if none exists yet). yes-if-CAREFUL: overlap the transport ONLY; the accumulator adds stay in the literal fixed g order (§12) — never reorder them. (If P3 landed, the device-resident pulls make this cleaner; if P3 was skipped, stream the host-staged-upload version.)',
    test: 'On rtxbox: f2_multigpu_parity P2P bit-identical (EmuFp64, derived_acc + derived_full); resG2p path == P2pDeviceResident; (optional) an nsys note that the peer copies overlap and the full cudaDeviceSynchronize is gone; build+ctest green.' },
]

const fixPrompt = (it) =>
  'You are a senior CUDA/C++ engineer applying ONE P2P-transport fix to steppe M4.5 (branch m4.5-multigpu) on the PRO-6000 box. Do NOT commit (the independent verdict agent commits).\n\n' + STD + '\n\nFIX ' + it.id + ': ' + it.fix + '\nPRIMARY FILES: ' + it.files + '\nREQUIRED TEST / OBJECTIVE GATE: ' + it.test + '\n\nApply the fix per the architecture standards + docs/cleanup/m4.5/device-cuda-p2p_combine.md; update stale doc-comments. Cite the CUDA docs (grid-stride, streams/events, cudaMemcpyPeer/Async, peer access) for device-behavior claims.\n\n' + DEVLOOP + '\n\nCRITICAL: this fix MUST keep test_f2_multigpu_parity BIT-IDENTICAL WITH THE P2P PATH ACTUALLY EXERCISED on rtxbox (run it directly: ' + PARITY + '). Return a thorough report: (1) every file changed + what changed; (2) the FULL build result; (3) the FULL ctest result + the f2_multigpu_parity output (paste the P2P device-resident == single-GPU lines AND resG2p.last_combine_path). If you CANNOT reach clean+green+P2P-bit-identical, do NOT pretend success — report exactly what blocked it (and whether the P2P bit-identity itself broke, or the path silently degraded to host-staged = both real failures).'

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item', 'pass', 'regression', 'p2p_exercised', 'p2p_bit_identical', 'green_count', 'commit_hash', 'note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if build clean + ALL ctest green + diff genuinely addresses the finding (not a sham/weakened test) + NO regression + the P2P path is STILL EXERCISED (not silently degraded) + P2P device-resident == single-GPU BIT-IDENTICAL' },
    regression: { type: 'boolean' },
    p2p_exercised: { type: 'boolean', description: 'true if resG2p.last_combine_path == P2pDeviceResident (the P2P path actually ran — a fix that drops to host-staged so the path is no longer tested is a FAIL)' },
    p2p_bit_identical: { type: 'boolean', description: 'true if P2P device-resident == host-staged == single-GPU is memcmp-bit-identical (EmuFp64)' },
    green_count: { type: 'integer' },
    commit_hash: { type: 'string', description: 'short hash if committed on PASS, else empty' },
    note: { type: 'string', description: 'one-line rationale; for FAIL, the exact reason + whether parity/exercise broke (a real correctness problem)' },
  },
}

const verdictPrompt = (it, fixReport) =>
  'You are the INDEPENDENT VERDICT for P2P fix ' + it.id + ' of steppe M4.5 (you did NOT write the fix — be adversarial, ESPECIALLY about the P2P bit-identity AND that the P2P path is still actually exercised). The fixer reported:\n<<<\n' + fixReport + '\n>>>\n\n' + STD + '\n\nThe finding: ' + it.fix + '\nThe objective gate: ' + it.test + '\n\nDO: (1) inspect the actual uncommitted changes: cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff ; (2) judge PASS only if ALL hold — build clean (warnings-as-errors), ALL ctest green (NO regression), the diff GENUINELY addresses ' + it.id + ' (not a sham/comment-only/weakened), AND test_f2_multigpu_parity shows P2P device-resident == host-staged == single-GPU BIT-IDENTICAL (EmuFp64) AND resG2p.last_combine_path == P2pDeviceResident (the P2P path STILL RAN — a fix that quietly drops to host-staged so the P2P path is no longer exercised is a FAIL even if every assertion passes). RE-RUN the box build/ctest + the parity binary yourself if the report looks inconsistent: ' + BUILD + ' ; ' + PARITY + ' . NEVER accept NCCL-AllReduce / reordered-accumulator changes; for P3/P4 confirm only the COPIES overlap, not the accumulator add order.\n\nON PASS: cd ' + R + ' and git add ONLY the specific changed/new files for this fix (NEVER git add dot — leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md untracked), commit with a ROADMAP §6 message (what+why; the box build/run commands) ending with: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash via git rev-parse --short HEAD.\nON FAIL: revert: ' + CLEAN + ' — report the exact reason (and whether P2P parity/exercise broke).\n\nReturn the structured verdict.'

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
  if (fix === null) { ledger.push({ item: it.id, pass: false, regression: false, p2p_exercised: false, p2p_bit_identical: false, green_count: 0, commit_hash: '', note: 'fix-agent terminal API error after retry — SKIPPED' }); continue }
  const v = await tryAgent(verdictPrompt(it, fix), { schema: VERDICT_SCHEMA, label: 'verdict:' + it.id, phase: it.title })
  if (v === null) { ledger.push({ item: it.id, pass: false, regression: false, p2p_exercised: false, p2p_bit_identical: false, green_count: 0, commit_hash: '', note: 'verdict-agent terminal API error after retry — SKIPPED' }); continue }
  ledger.push(v)
  if (v.pass) log('+++ ' + it.id + ' committed ' + v.commit_hash + ' (' + v.green_count + ' green, p2p_exercised=' + v.p2p_exercised + ', bit_identical=' + v.p2p_bit_identical + ')')
  else log('--- ' + it.id + ' FAILED verdict (' + v.note + ') — reverted; skip-and-continue')
}
return { ledger }
