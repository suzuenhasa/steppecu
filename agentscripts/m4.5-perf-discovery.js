export const meta = {
  name: 'm4.5-perf-discovery',
  description: 'DIAGNOSE why M4.5 multi-GPU (G==2) is slower than single-GPU (G==1) in the bench, and produce the parity-safe speedup plan. READ/MEASURE only (no source edits) — measures on rtxbox (2x RTX PRO 6000), analyzes the code, writes docs/cleanup/m4.5/perf-discovery.md. Seeded with the prior bench + nsys findings so it does not re-derive.',
  phases: [
    { title: 'Measure', detail: 'rtxbox: RELEASE re-bench (the debug-sync hypothesis) + nsys G1-vs-G2' },
    { title: 'Analyze', detail: 'parallel read-only: allocation / synchronization / data-movement / fan-out concurrency levers' },
    { title: 'Synthesize', detail: 'the prioritized parity-safe speedup plan -> docs/cleanup/m4.5/perf-discovery.md' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh rtxbox'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ rtxbox:/workspace/steppe/'

const CONTEXT = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimpl of ADMIXTOOLS 2 f-statistics, branch m4.5-multigpu. M4.5 single-node multi-GPU is DONE + bit-identity-proven (36/36 ctest green). The OPEN QUESTION: the bench tests/reference/bench_f2_multigpu.cu shows G==2 (2-GPU sharded, host-staged OR P2P combine) is SLOWER than G==1 (single-GPU) at every P. Goal: find WHY and the PARITY-SAFE fix. Read docs/cleanup/m4.5/00-overview.md (the audit: B1 fan-out, B2 P2P, L4 pool allocator) + agentscripts/README.md (the workflow map / the finding) + docs/BOX-RUNBOOK.md (box ops).',
  'BOX = rtxbox (2x RTX PRO 6000 Blackwell, sm_120, CUDA 13, 96 GB ea, REAL P2P can_access_peer=true). ssh rtxbox . nvcc not on PATH -> ' + PATHENV + ' . Data at /workspace/data/aadr/{raw, derived_acc, derived_full P=768}. rtxbox is stable for long jobs, but if an ssh hangs use the detached-run+poll trick (docs/BOX-RUNBOOK §7).',
  'PRIOR FINDINGS (do NOT re-derive — confirm/extend):',
  '  * Bench on rtxbox (DEFAULT build, EmulatedFp64{40}, min-of-2): P=200 G1=401ms/G2=398ms(1.01x); P=400 947/1140(0.83x); P=600 1713/2232(0.77x); P=768 2494/3467(0.72x). G==2 is the P2P device-combine (cudaMemcpyPeer ran).',
  '  * nsys of that run (CUDA API time share): cudaDeviceSynchronize 42.1% (2.53s, 186 calls) · cudaMemcpyAsync 32.8% (PAGEABLE host mem -> effectively blocking) · cudaMemcpy 11.4% · cudaFree 8.2% (648 calls) · cudaMalloc 3.0% (645 calls) · cudaMallocAsync 2.2% · cudaMemcpyPeer 0.0% (the P2P combine — negligible, OFF the critical path as architecture.md §11.4 predicts).',
  '  * CONCLUSION SO FAR: B2 (the P2P combine rework) is NOT the speedup lever (cudaMemcpyPeer is 0.0%). The cost is synchronization (42%) + unpinned data movement (~44%) + per-chunk alloc churn (~11%).',
  '  * THE LEADING HYPOTHESIS TO TEST FIRST: the only explicit cudaDeviceSynchronize in src/ is check.cuh:201 — the DEBUG post-launch kernel check (STEPPE_CUDA_CHECK_KERNEL), active because the build has NO CMAKE_BUILD_TYPE (NDEBUG undefined). So there is a FULL DEVICE SYNC AFTER EVERY KERNEL LAUNCH, which serializes the two device streams and forfeits all fan-out overlap. A RELEASE build (-DCMAKE_BUILD_TYPE=Release => NDEBUG) should drop it. If so, the "multi-GPU is slower" finding is largely a DEBUG-BUILD ARTIFACT and a Release re-bench may show the real fan-out speedup.',
  'PARITY LAW: any proposed fix must preserve the proven §12 bit-identity (test_f2_multigpu_parity memcmp). Bit-identity must NOT depend on build type.',
].join('\n')

phase('Measure')

const m1 = await agent([
  'You are a CUDA performance engineer. MEASURE on rtxbox — do NOT edit any source. Test THE hypothesis: is the multi-GPU slowdown a DEBUG-BUILD artifact (the per-kernel cudaDeviceSynchronize from STEPPE_CUDA_CHECK_KERNEL, active because the default build has no NDEBUG)?',
  CONTEXT, '',
  'STEPS (run on rtxbox; rsync first if needed: ' + RSYNC + '):',
  '1. Build a RELEASE tree in a SEPARATE dir so the default build is untouched:  ' + SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release && cmake --build build-rel 2>&1 | tail -5'  . Confirm NDEBUG is defined in Release (grep the compile flags / CMakeCache, or verify STEPPE_CUDA_CHECK_KERNEL compiles out the sync). If a long step risks an ssh hang, run it detached + poll a /tmp log (docs/BOX-RUNBOOK §7).",
  '2. PARITY in Release: run ./build-rel/bin/test_f2_multigpu_parity and confirm the EmulatedFp64 G==2 == single-GPU is STILL memcmp-BIT-IDENTICAL (bit-identity must not depend on build type). Report PASS/FAIL.',
  '3. RELEASE re-bench: ./build-rel/bin/bench_f2_multigpu /workspace/data/aadr 200 400 600 768 (filter the P2P-unavailable warns). Report the G1/G2/speedup table.',
  '4. Compare to the DEFAULT-build numbers (above). Did Release change the picture — does G==2 now beat or approach G==1? By how much per P? Quantify the debug-sync cost.',
  'Return: the Release bench table, the parity result, and a crisp verdict: how much of the slowdown was the debug per-kernel sync, and whether multi-GPU is faster/slower in Release.',
].join('\n'), { label: 'release-rebench', phase: 'Measure' })

const m2 = await agent([
  'You are a CUDA performance engineer. MEASURE on rtxbox — no source edits. Profile the RELEASE build to find the real bottleneck and whether the two devices actually overlap.',
  CONTEXT, '',
  'PRIOR STEP (the Release re-bench) reported:\n<<<\n' + (m1 || '(missing)') + '\n>>>', '',
  'STEPS (Release build-rel from step M1; nsys at /usr/local/cuda/bin/nsys):',
  '1. nsys-profile a focused run and read --stats: prefer profiling a SINGLE-P run for G==1 and a SINGLE-P run for G==2 SEPARATELY if you can (e.g. add a temporary tiny driver, OR profile the bench at one P and reason about the G1 vs G2 segments). Capture the CUDA API Summary + the GPU Kernel Summary + (key) whether the two devices kernels OVERLAP in time (gputrace / the per-device timeline) in G==2.',
  '2. Quantify in RELEASE: the API time share (is cudaDeviceSynchronize gone? what dominates now — cudaMemcpyAsync/pageable copies? cudaMalloc/Free churn?), and the GPU-busy overlap factor for G==2 (do device 0 and device 1 run concurrently, or serialize?).',
  'Return: the Release nsys breakdown, the device-overlap verdict (concurrent vs serial in G==2), and which remaining costs (pinned staging? pool allocator? stream/event overlap?) gate the fan-out from delivering ~2x. Cite the numbers.',
].join('\n'), { label: 'nsys-release', phase: 'Measure' })

phase('Analyze')
const LENSES = [
  { key: 'allocation', focus: 'Per-chunk DeviceBuffer alloc churn (cuda_backend.cu compute_f2_blocks ~:306-386 allocates dQ_raw/dV_raw/dN_raw/dIds/dQg/dVg/dSg/dGg/dVpairg/dRg PER CHUNK in the bucket loop; cudaMalloc/cudaFree are device-synchronizing + take a global driver lock that serializes the two fan-out threads). Quantify how much, and the parity-safe fix (L4: cudaMallocAsync pool / pre-allocated reused buffers, release-threshold MAX; per-device).' },
  { key: 'synchronization', focus: 'The sync structure: the debug post-launch cudaDeviceSynchronize (check.cuh; M1 quantifies its release-vs-debug cost), the per-chunk cudaStreamSynchronize (cuda_backend.cu:333/386/397), the fan-out join barrier, the p2p_combine per-partial cudaDeviceSynchronize (:283). Which syncs serialize the two devices / kill pipelining, and the parity-safe fix (events not device-syncs; stream-scoped not device-scoped).' },
  { key: 'data-movement', focus: 'The H2D/D2H: Q/V/N uploaded per device on PAGEABLE host memory (cudaMemcpyAsync on pageable => blocking, ~44% of API time); the per-device sub-view uploads doubled across G; the partial D2H + the host-staged gather. The parity-safe fix (L2: pinned host staging so async H2D overlaps; eliminate redundant copies). Does the multi-GPU re-upload more total bytes than single-GPU?' },
  { key: 'fan-out-concurrency', focus: 'core::compute_f2_blocks_multigpu (f2_blocks_multigpu.cpp): the std::jthread-per-device fan-out — does it ACTUALLY let the two devices run concurrently, or is there a shared lock / shared default stream / a serializing op (cudaMalloc/cudaSetDevice/cuBLAS handle) that prevents overlap? What is the theoretical best (max not sum) and what blocks it?' },
]
const analyses = await parallel(LENSES.map((L) => () => agent([
  'You are a CUDA performance engineer doing a READ-ONLY analysis of ONE lens of the M4.5 multi-GPU slowdown: ' + L.key + '. No source edits — analyze the code + the measured data and return findings + parity-safe fixes.',
  CONTEXT, '',
  'RELEASE re-bench (M1):\n<<<\n' + (m1 || '(missing)') + '\n>>>',
  'RELEASE nsys (M2):\n<<<\n' + (m2 || '(missing)') + '\n>>>', '',
  'YOUR LENS: ' + L.focus, '',
  'Read the actual code (src/device/cuda/cuda_backend.cu, f2_blocks_kernel.cu, check.cuh, device_buffer.cuh, stream.hpp; src/core/fstats/f2_blocks_multigpu.cpp, f2_combine.cpp; src/device/p2p_combine.* ). For each finding: location, the cost (tie to the measured numbers where possible), the concrete PARITY-SAFE fix, expected impact on the G1-vs-G2 gap, effort (S/M/L), and whether it maps to an audit item (L4/L2/L3/B1/B2). Flag anything that would BREAK §12 bit-identity as rejected-for-parity. Return your lens findings (no file write).',
].join('\n'), { label: 'analyze:' + L.key, phase: 'Analyze' }))).then(a => a.filter(Boolean))

phase('Synthesize')
const plan = await agent([
  'You are the lead. Synthesize the M4.5 multi-GPU PERF DISCOVERY into the parity-safe speedup plan and WRITE it to ' + R + '/docs/cleanup/m4.5/perf-discovery.md (Write tool).',
  CONTEXT, '',
  'RELEASE re-bench (M1):\n<<<\n' + (m1 || '') + '\n>>>',
  'RELEASE nsys (M2):\n<<<\n' + (m2 || '') + '\n>>>',
  'LENS ANALYSES:\n<<<\n' + analyses.join('\n\n---\n\n') + '\n>>>', '',
  'The doc must contain: (1) THE HEADLINE — is multi-GPU actually slower, and how much of the original finding was the debug-build per-kernel sync (Release vs default numbers); (2) the real bottleneck ranking in Release (with the nsys numbers) + the device-overlap verdict; (3) a PRIORITIZED, PARITY-SAFE fix plan (each item: the fix, the measured/expected impact on the G1-vs-G2 gap, effort, audit-item mapping L2/L3/L4/etc., and PARITY-SAFE/rejected-for-parity); (4) explicitly: is B2 (P2P combine) still worth doing, and is it for cleanliness or speed?; (5) the recommended NEXT workflow to author (e.g. a fix-pass for the top levers) + whether to just set CMAKE_BUILD_TYPE=Release as the bench/perf default. Tag every claim to a measured number where possible.',
  'Return a 5-8 line executive summary (the headline + the top 3 levers + the B2 verdict + the recommended next step).',
].join('\n'), { label: 'synthesize-plan', phase: 'Synthesize' })

return { m1, m2, analyses, plan }
