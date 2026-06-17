export const meta = {
  name: 'm4.5-scaffold',
  description: 'M4.5 capability-tier SCAFFOLD pass (makes CudaBackend multi-GPU-READY; the sharding/combine algorithm is the next workflow). 5 per-file units, STRICTLY SEQUENTIAL, 3 agents per unit (devoted implement -> build -> independent adversarial verify+commit). Build+ctest-gated on rtxbox (2x RTX PRO 6000). Commit-green / revert. HALT on first FAIL (unit 5 depends on 1/2/4). Retry once on transient API null.',
  phases: [
    { title: 'U1 check.cuh' },
    { title: 'U2 backend.hpp' },
    { title: 'U3 config.hpp' },
    { title: 'U4 handles.hpp' },
    { title: 'U5 cuda_backend.cu' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh rtxbox'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh ' + R + '/ rtxbox:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build -GNinja >/tmp/cfg.log 2>&1 && cmake --build build 2>&1 | tail -25 && echo === CTEST === && ctest --test-dir build --output-on-failure 2>&1 | tail -35'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics. Branch m4.5-multigpu. This is the M4.5 capability-tier SCAFFOLD pass: make the single-device CudaBackend MULTI-GPU-READY (capability probe + non-throwing tagged-degrade + per-device knobs + device_id threading) WITHOUT yet sharding or combining — that algorithm is the NEXT workflow. Do not implement sharding/combine here.',
  'BOX = rtxbox (2x RTX PRO 6000 Blackwell, sm_120, CUDA 13.0.88, driver 580; 96 GB ea). Build/test ONLY on the box; NOTHING builds locally. Author files locally under ' + R + '.',
  'READ FIRST (authoritative): ' + R + '/docs/architecture.md sections 9 (DeviceConfig/Resources/PerGpuResources, build() validation), 11.4 (SPMG multi-GPU, host-side fixed-order combine, capability-tiered P2P), 12 (parity/determinism), 7 (CUDA idioms), 10 (error/log taxonomy); ' + R + '/docs/cleanup/00-overview.md section (2) "CAPABILITY-TIER COHERENCE — the ONE unified design" + the master backlog; ' + R + '/docs/TODO.md M4.5 section + the capability-tier table. READ the relevant per-unit review under ' + R + '/docs/cleanup/ before editing your file.',
  'CAPABILITY-TIER LAW (architecture 11.4, re-verified workflow wxz1fiiln): real PCIe P2P (cudaMemcpyPeer device-resident combine) works on the STOCK driver on RTX PRO 6000 / datacenter Blackwell — MEASURED ON THIS BOX: cudaDeviceCanAccessPeer=1 both directions, byte-exact, 55.6 GB/s. On consumer GeForce 5090 P2P is driver-DISABLED (only via the aikitoria open-gpu-kernel-modules patch — dev-only, NEVER production). So the probe MUST use a NON-THROWING tagged-degrade path: canAccessPeer="no" and cudaErrorPeerAccessAlreadyEnabled are EXPECTED, not faults. Every capability lever is parity-NEUTRAL (data-movement/observability only); section-12 parity holds identically on both tiers.',
  'STANDARDS: RAII; strict layering (CUDA PRIVATE to steppe_device; core/api AND backend.hpp stay CUDA-FREE); DRY single-home; fail-fast; NO magic numbers; correct/current comments. STEPPE_LOG_WARN lives in core/internal/log.hpp (from B7); STEPPE_CUDA_CHECK in device/cuda/check.cuh; deterministic field exists on DeviceConfig (B9). Cite cuBLAS/CUDA/C++-stdlib docs where behavior is load-bearing. NO synthetic data for any accuracy claim (real AADR only).',
].join('\n')

const ITEMS = [
  { id: 'U1', title: 'U1 check.cuh', files: 'src/device/cuda/check.cuh (and src/core/internal/log.hpp only if the warn-sink needs a tweak)',
    fix: 'Add a NON-THROWING STEPPE_CUDA_WARN(expr) alongside the throwing STEPPE_CUDA_CHECK: on a non-cudaSuccess status it emits exactly ONE STEPPE_LOG_WARN line (file/line/error name+string via cudaGetErrorName/String) and CONTINUES — it does NOT throw, and it yields the cudaError_t so the caller can branch. This is the device-cuda-check CAP-1/CAP-2 finding: cudaDeviceCanAccessPeer returning "no" and cudaDeviceEnablePeerAccess returning cudaErrorPeerAccessAlreadyEnabled are EXPECTED capability-degrade conditions, NOT faults — routing them through the throwing checker turns a graceful degrade into a hard failure. Keep it a thin macro/inline consistent with check.cuh style + std::source_location like the existing checker. Also fold check.cuh-s open-coded NDEBUG gate into the shared facility if trivial. READ docs/cleanup/device-cuda-check*.md first.',
    test: 'A unit test (extend tests for cuda check, or add tests/unit/test_cuda_check.cu wired into tests/CMakeLists.txt): STEPPE_CUDA_WARN(cudaErrorPeerAccessAlreadyEnabled) does NOT throw, logs, and yields the status; STEPPE_CUDA_CHECK on the same status DOES throw (CudaError). Runnable on the box, no GPU work / no real data needed.' },

  { id: 'U2', title: 'U2 backend.hpp', files: 'src/device/backend.hpp',
    fix: 'Add a CUDA-FREE POD struct BackendCapabilities (architecture 9 / overview (2).1): { int device_count=0; int compute_major=0, compute_minor=0; std::size_t total_vram_bytes=0, free_vram_bytes=0; bool can_access_peer=false; bool emulated_fp64_honorable=false; } with doc-comments. Add a virtual method "virtual BackendCapabilities capabilities() const" to ComputeBackend WITH a default base implementation returning a value-initialized (unknown) BackendCapabilities, so CpuBackend need not override and backend.hpp compiles standalone. Document the per-device-instance contract: one backend instance is bound to ONE device; capabilities() reports THAT device + whether GPU0 can peer-access the other visible devices; the which-path/capability TAG is recorded out-of-band in Resources/a result envelope, NEVER on F2BlockTensor (keep it pure numeric). backend.hpp MUST stay CUDA-FREE (no CUDA headers). READ docs/cleanup/device-backend*.md.',
    test: 'Build gate + a host assertion (in config_unit or a tiny new host test) that a default-constructed BackendCapabilities is a POD/aggregate with the documented zero/false defaults. The real probe values are asserted in U5.' },

  { id: 'U3', title: 'U3 config.hpp', files: 'include/steppe/config.hpp',
    fix: 'Add "bool prefer_p2p_combine = true;" to DeviceConfig — DISTINCT from enable_peer_access (overview (2).3): enable_peer_access = MAY we call cudaDeviceEnablePeerAccess at all; prefer_p2p_combine = WHEN peer access is available, prefer the device-resident cudaMemcpyPeer combine over the host-staged combine (both bit-identical, section-11.4). Doc-comment citing 11.4. Add an OVERRIDE-KNOB BANNER comment above the multi-GPU knobs documenting the two knob types: override-INTENT lives here in DeviceConfig (devices, enable_peer_access, prefer_p2p_combine, future enable_gds_ingest); DISCOVERED capability + which-path tag lives in Resources/result metadata, NEVER here. Keep config.hpp host-only/CUDA-free. READ docs/cleanup/include-config*.md.',
    test: 'config_unit asserts prefer_p2p_combine defaults to true (and enable_peer_access stays true); build+ctest green.' },

  { id: 'U4', title: 'U4 handles.hpp', files: 'src/device/cuda/handles.hpp',
    fix: 'Two additions (L10 + overview (2).1 handles 2.3/11.x). (a) Record the device ORDINAL on CublasHandle: store the device_id the handle was created on; in debug, record-and-ASSERT it matches the current device on use (STEPPE_DEBUG_ONLY/STEPPE_ASSERT from host_device.hpp). NEVER call cudaSetDevice inside the wrapper — that is the caller/Resources responsibility. (b) Add a MathModeScope RAII (L10): ctor captures the current cuBLAS math mode (cublasGetMathMode) and applies a requested mode; dtor restores the captured mode; non-copyable, move-only optional, dtor NEVER throws (logs via STEPPE_LOG_WARN on nonzero status, consistent with the existing teardown pattern). For M4.5 Fp64-parity-recompute coexisting with EmulatedFp64 on one handle. READ docs/cleanup/device-cuda-handles*.md.',
    test: 'A test (extend an existing device test or add a small one, wired into CMake) asserting MathModeScope restores the prior math mode after scope exit, and (debug) the recorded ordinal matches the creation device. Runnable on the box.' },

  { id: 'U5', title: 'U5 cuda_backend.cu', files: 'src/device/cuda/cuda_backend.cu (and the make_cuda_backend declaration + any test/spike call sites if the ctor signature changes; src/device/cuda/f2_block_kernel.cu/.cuh only for the emulation_honorable predicate)',
    fix: 'Three COUPLED changes in CudaBackend (F19/F20 + X-6; READ docs/cleanup/device-cuda-cuda_backend*.md). (1) DEVICE_ID THREADING: thread an int device_id into the CudaBackend ctor (default from DeviceConfig::devices[0], or 0 when devices is empty) and cudaSetDevice(device_id) in the ctor (record-and-set) + guard each compute entry; update make_cuda_backend() and ALL call sites (tests, spike) so SINGLE-GPU behavior is unchanged (device 0) — ZERO regression on the 24 tests. (2) CAPABILITIES PROBE: implement ComputeBackend::capabilities() for CudaBackend — cudaGetDeviceProperties (compute_major/minor), cudaMemGetInfo (total+free; total_b is ALREADY captured around cuda_backend.cu:249-250, stop discarding it), cudaGetDeviceCount, and can_access_peer via cudaDeviceCanAccessPeer across the visible devices ROUTED THROUGH THE NON-THROWING STEPPE_CUDA_WARN (U1) (canAccessPeer="no" is expected, never a throw), and emulated_fp64_honorable from the existing emulation_honorable predicate. (3) EMU-HONORABLE TAGGED DEGRADE: verify/refine the X-6/B2 path so that when EmulatedFp64 is requested but NOT honorable it degrades to native Fp64 with a one-shot STEPPE_LOG_WARN capability tag (route through the probe + the warn primitive), never silently running dynamic. DO NOT shard or combine — that is the next workflow.',
    test: 'On rtxbox (2x RTX PRO 6000): a reference/device test asserting capabilities() reports compute_major>=12, total_vram_bytes>0, device_count==2, and can_access_peer==true (the REAL stock-driver P2P this box has — measured 55.6 GB/s); construct CudaBackend on device 0 (and device 1 since count>=2) with no error. ALL 24 pre-existing tests still green (no regression from the ctor signature change). Wire any new test into tests/CMakeLists.txt.' },
]

const implementPrompt = (it) => [
  'You are a senior CUDA/C++ engineer applying ONE devoted change to steppe for the M4.5 capability-tier scaffold. The git tree is CLEAN at HEAD; make ONLY your unit-s edits. Do NOT build, do NOT commit (a build agent builds, an independent verify agent commits).',
  STD, '',
  'UNIT ' + it.id + ' — your devoted scope: ' + it.files,
  'CHANGE: ' + it.fix,
  'OBJECTIVE TEST (the verify gate): ' + it.test, '',
  'Author the change + the required test locally per the architecture standards; wire any new test into tests/CMakeLists.txt. Update any doc-comments the change makes stale. Where a claim depends on CUDA/cuBLAS or C++-stdlib behavior, cite the docs. Return a thorough report: (1) every file changed + exactly what changed + why; (2) the test added + what it asserts; (3) any cross-file consequence the build agent must know (e.g. a changed signature and the call sites you updated). Do NOT build, do NOT commit.',
].join('\n')

const buildPrompt = (it, implReport) => [
  'You are the BUILD agent for M4.5 scaffold unit ' + it.id + '. An implement agent authored changes to: ' + it.files + '.',
  STD, '',
  'IMPLEMENT REPORT:\n<<<\n' + implReport + '\n>>>', '',
  'DO, in order: (1) rsync local -> box:  ' + RSYNC,
  '(2) build + full ctest on the box:  ' + BUILD,
  '(3) FIX-LOOP: on any COMPILE error or TEST failure, edit the offending LOCAL files (' + R + '/...) minimally — preserving the unit-s intent — then re-rsync and rebuild, up to 6 iterations. Common issues after a ctor-signature change: make_cuda_backend + test/spike call sites, include paths, warnings-as-errors, a new test not wired into tests/CMakeLists.txt.',
  'The bar: build CLEAN (warnings-as-errors) AND all ctest GREEN — there are 24 pre-existing tests that MUST still pass (no regression); the unit may add 1+ new tests.',
  'Report: the FULL final build result, the FULL ctest summary lines (paste "X/Y tests passed" and the per-test lines), the exact green count, and EVERY fix you had to make. Do NOT commit. If you cannot reach clean+green in 6 iterations, say so and give the exact blocking error.',
].join('\n')

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item', 'pass', 'regression', 'green_count', 'commit_hash', 'note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if build clean (warnings-as-errors) + ALL ctest green + diff genuinely addresses the finding (not a sham/comment-only/test-weakened) + the objective test present and green + NO regression vs the 24 prior tests' },
    regression: { type: 'boolean', description: 'true if any of the 24 prior tests broke' },
    green_count: { type: 'integer', description: 'number of ctest tests passing (>=24 expected)' },
    commit_hash: { type: 'string', description: 'short hash if committed on PASS, else empty' },
    note: { type: 'string', description: 'one-line rationale; for FAIL, the exact reason + whether it is a design call needing human input' },
  },
}

const verifyPrompt = (it, implReport, buildReport) => [
  'You are the INDEPENDENT VERIFY agent for M4.5 scaffold unit ' + it.id + ' (you did NOT write or build it — be ADVERSARIAL).',
  STD, '',
  'The finding being implemented: ' + it.fix,
  'The objective gate: ' + it.test, '',
  'IMPLEMENT REPORT:\n<<<\n' + implReport + '\n>>>',
  'BUILD REPORT:\n<<<\n' + buildReport + '\n>>>', '',
  'DO: (1) inspect the ACTUAL uncommitted changes:  cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff',
  '(2) judge PASS only if ALL hold: the diff GENUINELY addresses ' + it.id + ' (not a sham, not comment-only, not a weakened/removed test), build CLEAN (warnings-as-errors), ALL ctest GREEN with NO regression vs the 24 prior tests, and the REQUIRED objective test is present and green. If the reports are inconsistent with the diff, RE-RUN the box build/ctest yourself:  ' + BUILD,
  'ON PASS: cd ' + R + ' and git add ONLY the specific changed/new source+test+CMake+doc files for THIS unit (NEVER git add dot — leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md, and any /tmp artifacts untracked). Commit with a ROADMAP section-6 style message (what + why; the test added; the box build/run commands) ending with the trailer line exactly:  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>  . Capture the short hash:  git rev-parse --short HEAD .',
  'ON FAIL: revert so the repo stays green:  ' + CLEAN + '  — and report the exact reason (and whether it is a design call needing human input).',
  'Return the structured verdict.',
].join('\n')

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retrying once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  log('=== ' + it.id + ': implement ===')
  const impl = await tryAgent(implementPrompt(it), { label: 'impl:' + it.id, phase: it.title })
  if (impl === null) { ledger.push({ item: it.id, pass: false, regression: false, green_count: 0, commit_hash: '', note: 'implement-agent terminal API error after retry — HALT' }); break }

  log('=== ' + it.id + ': build ===')
  const build = await tryAgent(buildPrompt(it, impl), { label: 'build:' + it.id, phase: it.title })
  if (build === null) { ledger.push({ item: it.id, pass: false, regression: false, green_count: 0, commit_hash: '', note: 'build-agent terminal API error after retry — HALT' }); break }

  log('=== ' + it.id + ': verify ===')
  const v = await tryAgent(verifyPrompt(it, impl, build), { schema: VERDICT_SCHEMA, label: 'verify:' + it.id, phase: it.title })
  if (v === null) { ledger.push({ item: it.id, pass: false, regression: false, green_count: 0, commit_hash: '', note: 'verify-agent terminal API error after retry — HALT' }); break }
  ledger.push(v)

  if (v.pass) { log('+++ ' + it.id + ' committed ' + v.commit_hash + ' (' + v.green_count + ' green)') }
  else { log('--- ' + it.id + ' FAILED verify (' + v.note + ') — reverted; HALT (downstream units depend on it)'); break }
}
return { ledger }
