export const meta = {
  name: 'm4.5-multigpu',
  description: 'M4.5 single-node multi-GPU ALGORITHM (on top of the committed capability-tier scaffold): Resources/PerGpuResources + block-aligned SNP shard across G devices + host-staged fixed-order combine (portable baseline) + optional cudaMemcpyPeer P2P device-combine (rtxbox real-P2P fast-path), gated bit-identical to single-GPU (section-12 parity). Design phase first, then per-file Implement -> build -> independent verify+commit, STRICTLY SEQUENTIAL, ctest-gated on rtxbox (2x RTX PRO 6000), commit-green/revert, HALT on first FAIL, retry-once on transient null.',
  phases: [
    { title: 'Design' },
    { title: 'I1 Resources' },
    { title: 'I2 orchestrator+host-combine+parity' },
    { title: 'I3 P2P device-combine' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh rtxbox'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh ' + R + '/ rtxbox:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build -GNinja >/tmp/cfg.log 2>&1 && cmake --build build 2>&1 | tail -25 && echo === CTEST === && ctest --test-dir build --output-on-failure 2>&1 | tail -40'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics. Branch m4.5-multigpu. The M4.5 capability-tier SCAFFOLD is ALREADY COMMITTED (28/28 ctest green): non-throwing STEPPE_CUDA_WARN (device/cuda/check.cuh), CUDA-free BackendCapabilities POD + virtual ComputeBackend::capabilities() (device/backend.hpp), DeviceConfig::prefer_p2p_combine + enable_peer_access + deterministic (include/steppe/config.hpp), CublasHandle device-ordinal + MathModeScope (device/cuda/handles.hpp), and CudaBackend(int device_id) ctor + cudaSetDevice binding + the real capabilities() probe (device/cuda/cuda_backend.cu). THIS WORKFLOW builds the ALGORITHM on top: shard SNP work across G devices, combine the per-device partials, bit-identical to single-GPU.',
  'BOX = rtxbox (2x RTX PRO 6000 Blackwell, sm_120, CUDA 13.0.88, driver 580; 96 GB ea). MEASURED on this box: cudaDeviceCanAccessPeer=1 both directions, byte-exact cudaMemcpyPeer at 55.6 GB/s (the real stock-driver P2P). Build/test ONLY on the box; NOTHING builds locally. Author files locally under ' + R + '. Data on the box: /workspace/data/aadr/{raw, derived_acc P=50 M=100000, derived_full P=768 M=584131}.',
  'READ FIRST (authoritative): ' + R + '/docs/architecture.md sections 11.4 (SPMG multi-GPU: tile sharding + host-side fixed-order combine in g=0..G-1 order; capability-tiered P2P device-resident combine via cudaMemcpyPeer; NCCL/cudaMemcpy broadcast only — NEVER NCCL AllReduce), 12 (PARITY: bit-identical across G AND to single-GPU; fixed device order; NEVER a parity-critical sum on NCCL AllReduce), 9 (DeviceConfig/Resources/PerGpuResources), 7, 10; ' + R + '/docs/cleanup/00-overview.md section (2). Also READ the committed contract: ' + R + '/src/device/backend.hpp (ComputeBackend per-device-instance contract lines ~193-202, BackendCapabilities, compute_f2_blocks signature) and ' + R + '/include/steppe/fstats.hpp (F2BlockTensor layout i + P*j + P*P*b).',
  'PARITY LAW (architecture 12, the GATE): the multi-GPU combined f2_blocks + Vpair MUST be BIT-IDENTICAL to the single-GPU reference (same bits, not just within tolerance), AND identical across G. Achieve this by BLOCK-ALIGNED sharding: assign whole CONTIGUOUS BLOCK ranges to devices (block_id is non-decreasing so each block s SNP columns are contiguous; use core block_ranges from block_partition_rule). Each block is then computed ENTIRELY on one device -> its f2/Vpair bits equal the single-GPU result for that block (same SNP columns, same sm_120 cuBLAS, same fixed-slice emulated-FP64, single stream). The host-side fixed-order combine places/sums the per-device partials in g=0..G-1 (DeviceConfig::devices) order; summing exact zeros for non-owned blocks is exact. The cudaMemcpyPeer P2P device-combine moves the SAME bytes and sums the SAME fixed order on-device -> bit-identical to host-staged. NEVER NCCL AllReduce (its order varies with G).',
  'CAPABILITY-TIER FORK: the P2P device-combine is OPT-IN, gated on capabilities().can_access_peer (true on rtxbox) AND DeviceConfig::prefer_p2p_combine; on no-peer-access (consumer GeForce) it degrades to the host-staged combine with a STEPPE_LOG_WARN tag ("P2P combine unavailable (no peer access) -> host-staged fixed-order combine") via the non-throwing STEPPE_CUDA_WARN. The host-staged fixed-order combine is the PORTABLE PARITY BASELINE and the only path on the budget box. Both bit-identical (parity-neutral).',
  'STANDARDS: RAII; strict layering (CUDA PRIVATE to steppe_device; core/api/backend.hpp CUDA-FREE); DRY single-home; fail-fast; NO magic numbers; correct comments. Reuse the existing single-device CudaBackend::compute_f2_blocks per device (do NOT reimplement the per-block GEMM). Cite CUDA/cuBLAS docs where load-bearing. NO synthetic data for accuracy claims (real AADR: derived_acc fast gate + derived_full scale gate). Do NOT commit (verify agent commits); implement agents author only (build agent builds).',
].join('\n')

// ---- Design phase: one agent reads the real code and produces the precise design + signatures ----
const designPrompt = [
  'You are the lead architect for steppe M4.5 multi-GPU. Produce the PRECISE implementation design that the per-file implement agents will build against verbatim. Do NOT write production code or build — design only (you MAY read any file and may write a short scratch note).',
  STD, '',
  'READ the real code: src/device/backend.hpp (ComputeBackend, BackendCapabilities, compute_f2_blocks), include/steppe/fstats.hpp (F2BlockTensor exact fields + layout + block_sizes), src/device/cuda/cuda_backend.cu (the CudaBackend(device_id) ctor, capabilities(), compute_f2_blocks impl), src/core/cpu/cpu_backend.cpp (the oracle), src/core/domain/block_partition_rule.{hpp,cpp} (block_ranges from B3), src/core/fstats/f2_from_blocks.cpp (the CUDA-free orchestration seam), include/steppe/config.hpp (DeviceConfig).',
  'DECIDE and SPECIFY exactly (with rationale + verbatim signatures):',
  '1. Resources / PerGpuResources: the exact struct(s) and their HOME (a new header? device layer vs include/steppe?). For M4.5 precompute, PerGpuResources minimally needs device_id + a per-device CudaBackend (constructed via the device_id ctor) + its probed BackendCapabilities. Resources = the G of them. Honor architecture section-9 shape but keep M4.5-precompute-focused (cuSOLVER/NCCL fields may be stubbed/omitted with a documented "later" note). State whether Resources is CUDA-free or device-layer (it holds unique_ptr<ComputeBackend>, which is CUDA-free; the BUILDER that news CudaBackends is device-layer).',
  '2. The SHARD: confirm block-aligned sharding via block_ranges; specify how device g gets its block range [b0,b1) and the corresponding SNP column sub-views of Q/V/N (and a local block_id), and whether each device produces a FULL-shape partial (non-owned blocks zero) or a COMPACT partial placed at the global block offset. Pick the one that is (a) provably bit-identical to single-GPU and (b) simplest; justify against the F2BlockTensor layout.',
  '3. The HOST-STAGED fixed-order combine: the exact algorithm (gather each device partial to host, place/sum in g=0..G-1 order, reference precision) and where it lives (CUDA-free if possible).',
  '4. The P2P device-combine: the exact cudaMemcpyPeer plan (GPU0 pulls each peer partial, sums on-device in g=0..G-1 order), the gate (capabilities().can_access_peer && DeviceConfig::prefer_p2p_combine), the tagged-degrade fallback, and where it lives (device layer).',
  '5. The public entry point: the function signature the API/f2_from_blocks seam calls to run multi-GPU precompute (taking Resources + full Q/V/N + block partition + Precision -> combined F2BlockTensor), and how single-GPU (G==1) stays the exact current path.',
  '6. The PARITY TEST plan: assert the multi-GPU combined F2BlockTensor is BIT-IDENTICAL (memcmp / exact ==) to the single-GPU reference on derived_acc (fast gate) AND derived_full (scale gate), across G=1 vs G=2, and host-staged == P2P on rtxbox. Name the test file + how it loads the data + the exact assertions.',
  'Return: the design as a precise spec with verbatim C++ signatures for Resources/PerGpuResources, the builder, the combine functions, and the public entry point — enough that each implement agent matches them exactly. Flag any place the existing compute_f2_blocks impl makes block-aligned bit-identity subtle (e.g. padding/bucketing in the grouped GEMM) and how the design preserves it.',
].join('\n')

const ITEMS = [
  { id: 'I1', title: 'I1 Resources', files: 'the Resources/PerGpuResources home chosen by the design (likely a new header + a device-layer builder .cu), wired into CMake',
    fix: 'Implement Resources / PerGpuResources EXACTLY per the design: the struct(s) + a builder that, given a DeviceConfig, constructs one CudaBackend per device in DeviceConfig::devices (via the committed CudaBackend(int device_id) ctor), probes each via capabilities(), and assembles Resources (RAII, move-only ownership of the backends). Single-GPU (devices=={0} or empty) yields exactly one PerGpuResources. Keep the CUDA-free/device-layer split the design specified. Do NOT shard or combine here.',
    test: 'A test (wired into CMake): build Resources for DeviceConfig{devices={0}} -> 1 backend bound to device 0; and for {0,1} on this 2-GPU box -> 2 backends bound to devices 0 and 1, each capabilities() reporting compute_major>=12 and (for the {0,1} case) can_access_peer==true. No real f2 compute needed.' },

  { id: 'I2', title: 'I2 orchestrator+host-combine+parity', files: 'the multi-GPU orchestrator (device layer, e.g. src/device/cuda/multigpu_f2.cu/.cuh) + the public entry point + its CUDA-free declaration + the new parity test; wired into CMake',
    fix: 'Implement the BLOCK-ALIGNED shard + per-device partials (reusing each PerGpuResources backend.compute_f2_blocks over its block range and SNP sub-views) + the HOST-STAGED fixed-order combine (place/sum partials in g=0..G-1 order, reference precision) EXACTLY per the design -> a combined F2BlockTensor. Provide the public entry point; single-GPU (G==1) must be the exact current path (no combine overhead). This is the PORTABLE PARITY BASELINE (no P2P yet). DO NOT reimplement the per-block GEMM — reuse compute_f2_blocks.',
    test: 'The PARITY GATE (test_f2_multigpu_parity.cu, wired into CMake): on derived_acc (P=50, fast) AND derived_full (P=768, scale), assert the multi-GPU (devices={0,1}) combined f2_blocks AND Vpair are BIT-IDENTICAL (exact memcmp/==, not tolerance) to the single-GPU (devices={0}) reference. Must run green on rtxbox. This is THE gate — if not bit-identical, FAIL (do not relax to tolerance).' },

  { id: 'I3', title: 'I3 P2P device-combine', files: 'the orchestrator (add the cudaMemcpyPeer device-resident combine path) + the parity test (extend); CMake if needed',
    fix: 'Add the OPT-IN P2P device-combine per the design: gated on capabilities().can_access_peer && DeviceConfig::prefer_p2p_combine, GPU0 pulls each peer partial via cudaMemcpyPeer (byte-exact DMA) and sums on-device in the SAME fixed g=0..G-1 order; on no-peer-access or prefer_p2p_combine=false, degrade to the host-staged combine with a one-shot STEPPE_LOG_WARN tag via the non-throwing STEPPE_CUDA_WARN. Record which path ran out-of-band (Resources/result metadata, NEVER on F2BlockTensor). Never NCCL AllReduce.',
    test: 'Extend the parity test: on rtxbox (can_access_peer==true), the P2P device-combine result is BIT-IDENTICAL to the host-staged combine AND to the single-GPU reference (derived_acc + derived_full). Also assert prefer_p2p_combine=false forces the host-staged path (same bits). Green on rtxbox.' },
]

const implementPrompt = (it, design) => [
  'You are a senior CUDA/C++ engineer implementing ONE devoted unit of steppe M4.5 multi-GPU. The git tree is CLEAN at HEAD; make ONLY this unit s edits. Do NOT build, do NOT commit (a build agent builds, an independent verify agent commits).',
  STD, '',
  '== THE DESIGN (build against this verbatim) ==', design, '',
  'UNIT ' + it.id + ' — your devoted scope: ' + it.files,
  'IMPLEMENT: ' + it.fix,
  'OBJECTIVE TEST (the verify gate): ' + it.test, '',
  'Author the change + the required test locally per the architecture standards and the design; wire new files/tests into the CMake (top-level + tests/CMakeLists.txt). Cite CUDA/cuBLAS docs where behavior is load-bearing. Report: (1) every file added/changed + what + why; (2) the test + what it asserts (and that it asserts BIT-IDENTITY where required, not tolerance); (3) cross-file consequences the build agent must know. Do NOT build, do NOT commit.',
].join('\n')

const buildPrompt = (it, implReport) => [
  'You are the BUILD agent for M4.5 multi-GPU unit ' + it.id + '. An implement agent authored: ' + it.files + '.',
  STD, '',
  'IMPLEMENT REPORT:\n<<<\n' + implReport + '\n>>>', '',
  'DO: (1) rsync local -> box:  ' + RSYNC,
  '(2) build + full ctest on the box:  ' + BUILD,
  '(3) FIX-LOOP on any compile/test error: edit the offending LOCAL files minimally (preserve the unit/design intent), re-rsync, rebuild — up to 6 iterations. Common: CMake wiring of new .cu/.cuh + test targets, multi-backend construction, the cudaMemcpyPeer enable/disable, include paths, warnings-as-errors.',
  'Bar: build CLEAN (warnings-as-errors) AND all ctest GREEN — the 28 pre-existing tests MUST still pass (no regression); this unit adds 1+ tests. The parity test (when present) MUST pass with BIT-IDENTITY (exact), not tolerance — if it only passes under tolerance, that is a FAIL, report it.',
  'Report: FULL final build result, FULL ctest summary (paste X/Y + the per-test lines, esp. the parity test), the green count, and EVERY fix you made. Do NOT commit. If you cannot reach clean+green+bit-identical in 6 iterations, say so with the exact blocker (and whether bit-identity itself failed — a real numerical/design problem, not a build problem).',
].join('\n')

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item', 'pass', 'regression', 'bit_identical', 'green_count', 'commit_hash', 'note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if build clean + ALL ctest green + diff genuinely implements the unit/design (not a sham/weakened test) + NO regression vs the 28 prior tests + (for units with the parity test) BIT-IDENTITY holds (exact, not tolerance)' },
    regression: { type: 'boolean' },
    bit_identical: { type: 'boolean', description: 'true if the multi-GPU == single-GPU parity is exact/bit-identical (memcmp), false if only tolerance or not-yet-applicable to this unit' },
    green_count: { type: 'integer', description: 'ctest tests passing (>=28 expected, growing per unit)' },
    commit_hash: { type: 'string', description: 'short hash if committed on PASS, else empty' },
    note: { type: 'string', description: 'one-line rationale; for FAIL, the exact reason + whether a design call needing human input (esp. if bit-identity failed)' },
  },
}

const verifyPrompt = (it, implReport, buildReport) => [
  'You are the INDEPENDENT VERIFY agent for M4.5 multi-GPU unit ' + it.id + ' (you did NOT write or build it — be ADVERSARIAL, especially about the BIT-IDENTICAL parity claim).',
  STD, '',
  'The unit: ' + it.fix,
  'The objective gate: ' + it.test, '',
  'IMPLEMENT REPORT:\n<<<\n' + implReport + '\n>>>',
  'BUILD REPORT:\n<<<\n' + buildReport + '\n>>>', '',
  'DO: (1) inspect the ACTUAL uncommitted changes:  cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff',
  '(2) judge PASS only if ALL hold: the diff GENUINELY implements ' + it.id + ' per the design (not a sham, not comment-only, not a test weakened to tolerance where bit-identity is required, not NCCL AllReduce on a parity path), build CLEAN (warnings-as-errors), ALL ctest GREEN with NO regression vs the 28 prior tests, the required test present and green, and where a parity test exists it asserts BIT-IDENTITY (exact memcmp/==) and passes. If anything is inconsistent, RE-RUN the box build/ctest yourself:  ' + BUILD + '  — and for the parity test, confirm it truly compares exact bits (read the test source).',
  'ON PASS: cd ' + R + ' and git add ONLY the specific changed/new source+test+CMake+doc files for THIS unit (NEVER git add dot — leave aadr/, build_run.sh, f2_emu_spike.cu, handoff-*.md, /tmp artifacts untracked). Commit with a ROADMAP section-6 message (what+why; the test; the box build/run commands) ending exactly with:  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>  . Capture short hash:  git rev-parse --short HEAD .',
  'ON FAIL: revert so the tree stays green:  ' + CLEAN + '  — report the exact reason, and CLEARLY flag if BIT-IDENTITY itself failed (a real numerical/design problem needing human input, NOT a build nit).',
  'Return the structured verdict.',
].join('\n')

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retrying once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

phase('Design')
log('=== Design: reading the real code + producing the precise spec ===')
let design = await tryAgent(designPrompt, { label: 'design', phase: 'Design' })
if (design === null) { return { ledger: [{ item: 'Design', pass: false, note: 'design-agent terminal API error after retry — HALT' }] } }

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  log('=== ' + it.id + ': implement ===')
  const impl = await tryAgent(implementPrompt(it, design), { label: 'impl:' + it.id, phase: it.title })
  if (impl === null) { ledger.push({ item: it.id, pass: false, regression: false, bit_identical: false, green_count: 0, commit_hash: '', note: 'implement-agent terminal API error after retry — HALT' }); break }

  log('=== ' + it.id + ': build ===')
  const build = await tryAgent(buildPrompt(it, impl), { label: 'build:' + it.id, phase: it.title })
  if (build === null) { ledger.push({ item: it.id, pass: false, regression: false, bit_identical: false, green_count: 0, commit_hash: '', note: 'build-agent terminal API error after retry — HALT' }); break }

  log('=== ' + it.id + ': verify ===')
  const v = await tryAgent(verifyPrompt(it, impl, build), { schema: VERDICT_SCHEMA, label: 'verify:' + it.id, phase: it.title })
  if (v === null) { ledger.push({ item: it.id, pass: false, regression: false, bit_identical: false, green_count: 0, commit_hash: '', note: 'verify-agent terminal API error after retry — HALT' }); break }
  ledger.push(v)

  if (v.pass) { log('+++ ' + it.id + ' committed ' + v.commit_hash + ' (' + v.green_count + ' green, bit_identical=' + v.bit_identical + ')') }
  else { log('--- ' + it.id + ' FAILED verify (' + v.note + ') — reverted; HALT'); break }
}
return { design_summary: (design || '').slice(0, 1200), ledger }
