export const meta = {
  name: 'm4.5-parallelism-check',
  description: 'Test the hypothesis that the M4.5 multi-GPU was NOT constructed as true parallel multi-GPU — that something runs SEQUENTIALLY / the work is not really split / the parallelism is at the wrong place. The numbers support it: a truly parallel 2-GPU run should be ~50% of single-GPU wall, but device-resident is 91% and host-staged is 140% (SLOWER). Two lenses: (1) READ the construction for serialization; (2) MEASURE on the LIVE box5090 with nsys — per-device GPU timeline: do the two GPUs kernels overlap in wall-clock, is work split 50/50, where is the serial time. Then a verdict + docs/cleanup/m4.5/parallelism-check.md.',
  phases: [
    { title: 'Check', detail: 'parallel: read construction for serialization + nsys per-device timeline on box5090' },
    { title: 'Verdict', detail: 'is it truly parallel? the serial bottleneck / wrong-place construction -> doc' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const NSYS_CHECK = SSH + " 'ls -la /workspace/steppe/build-rel/bin/bench_f2_multigpu; " + PATHENV + " && (which nsys || ls /usr/local/cuda/bin/nsys)'"
const NSYS_PROF = SSH + " '" + PATHENV + " && cd /workspace/steppe && nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true -o /tmp/par512 ./build-rel/bin/bench_f2_multigpu /workspace/data/aadr 512 2>&1 | tail -20'"
const NSYS_STATS = SSH + " '" + PATHENV + " && nsys stats --report cuda_gpu_trace --report cuda_gpu_kern_sum /tmp/par512.nsys-rep 2>&1 | tail -140'"

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell reimpl of ADMIXTOOLS 2 f-statistics. M4.5 = single-node multi-GPU f2 precompute: a block-aligned SNP/block shard across G devices, a per-device CONCURRENT fan-out (one std::jthread per device, each driving its own CudaBackend on its own device + stream), then a fixed-order combine (host-staged, or device-resident P2P on capable boxes).',
  'THE HYPOTHESIS TO TEST (the user is skeptical, and the numbers back them): a truly parallel 2-GPU run should take ~50% of the single-GPU wall (each GPU does half the work concurrently). MEASURED instead: device-resident G2 = 91% of G1 wall (1.10x), host-staged G2 = 140% of G1 wall (0.71x — SLOWER than one GPU). On box5090 (consumer, no P2P -> host-staged only): P=256 G1 883ms vs G2host 1063ms; P=512 G1 2653ms vs G2host 3733ms. This is NOT healthy parallelism. SO: is the multi-GPU actually running the two GPUs CONCURRENTLY, is the work actually SPLIT in half, or is something SEQUENTIAL / constructed at the wrong place? Find the truth with code evidence + an nsys timeline — do NOT hand-wave PCIe or the D2H tail.',
  'BOX = box5090 (2x RTX 5090 sm_120, 32GB ea, CONSUMER no-P2P, CUDA 13.0.88), LIVE. ' + SSH + ' (alias); nvcc/nsys not on PATH -> ' + PATHENV + ' . build-rel exists with the current bench (./build-rel/bin/bench_f2_multigpu); nsys CUDA-tracing works on this consumer box (it is the ncu PERF COUNTERS that are restricted, NOT nsys cuda/nvtx/osrt tracing). Single-GPU OOMs ~P700 so use P<=512 for clean G1-vs-G2. NOTHING builds locally.',
  'KEY FILES (cite file:line): src/core/fstats/f2_blocks_multigpu_core.cpp (compute_multigpu_partials = THE fan-out: jthread per device + the join barrier; genuinely concurrent or serialized?), src/core/fstats/f2_blocks_multigpu.cpp (the gate + where setup sits relative to the timed path), src/device/shard_plan.cpp (is the work split ~50/50 by SNP/block count?), src/device/resources.cpp and resources.hpp (Resources/build_resources — is per-device backend construction, cuBLAS handle creation, workspace alloc, peer-enable done ONCE/amortized or per-call / serially?), src/device/cuda/cuda_backend.cu (per-device compute_f2_blocks/_resident/_into: blocking cudaStreamSynchronize, per-call cudaMalloc on the driver global lock (audit L4), cuBLAS create per-call vs once, the H2D/feeder/GEMM/D2H sequence), src/device/backend_factory.hpp, tests/reference/bench_f2_multigpu.cu (what is INSIDE the timed region — is Resources build / cuBLAS init / warm-up amortized, or is per-run setup being timed?). Prior context: why-multigpu-slow.md claimed 74% overlap on rtxbox (VERIFY on the 5090; different box, may not hold), architecture-audit.md (Flaw 2 feeder peak, Flaw 3 serial D2H tail), scaling-sweep.md.',
].join('\n')

phase('Check')
const construction = await agent([
  'You are a senior CUDA/HPC engineer. READ the M4.5 multi-GPU construction and find EVERY place it could run sequentially instead of in parallel, split work unevenly, or pay serial per-call setup. READ-ONLY; cite file:line. Be adversarial — assume it is mis-constructed and try to prove it.', STD, '',
  'Answer concretely: (1) is the fan-out genuinely concurrent — one jthread per device launched before any joins, each on its own device + stream + handle, no shared mutex, no lazy global init serializing them? Or is there a hidden serialization: a mutex, a shared cuBLAS handle, a global cudaSetDevice race forcing order, the per-call cudaMalloc/cudaFree driver global lock that blocks the other thread, or a blocking call issued before the second thread even starts? (2) is the WORK actually split ~50/50 (shard_plan balances by SNP/block count — does each device do ~half the GEMM FLOPs)? (3) is per-device SETUP (cuBLAS handle create, workspace cudaMalloc, peer-enable, Resources/backend construction) done ONCE and amortized, or paid PER CALL / inside the timed region / serially across devices? (4) is the parallelism at the RIGHT place — host threads each blocking on their own device (so the only concurrency is two blocked host threads) and is the bench timing one-time setup (Resources build) that belongs outside the loop? (5) does anything force the two devices to run their COMPUTE back-to-back rather than simultaneously? Return a serialization findings list (each: file:line + is-it-serial + does-it-block-the-other-device + does-it-dominate-at-small-P), whether the work is balanced, and your read on whether the COMPUTE (not the combine/D2H) is genuinely concurrent.',
].join('\n'), { label: 'check:construction', phase: 'Check' })

const MEASURE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['method','p_profiled','g1_wall_ms','g2_wall_ms','dev0_busy_ms','dev1_busy_ms','gpu_union_wall_ms','compute_overlap_pct','work_split_ratio','serial_breakdown','two_gpus_run_concurrently','is_truly_parallel','evidence','note'],
  properties: {
    method: { type: 'string', description: 'how overlap was computed from the nsys cuda_gpu_trace per-device timeline' },
    p_profiled: { type: 'number' },
    g1_wall_ms: { type: 'number' }, g2_wall_ms: { type: 'number' },
    dev0_busy_ms: { type: 'number', description: 'GPU-active time on device 0 during the G2 run (kernels+memcpy)' },
    dev1_busy_ms: { type: 'number' },
    gpu_union_wall_ms: { type: 'number', description: 'wall span from first GPU op to last across BOTH devices in the G2 run' },
    compute_overlap_pct: { type: 'number', description: 'of the two devices kernel windows, what % overlaps in wall-clock (0=back-to-back/sequential, 100=fully concurrent)' },
    work_split_ratio: { type: 'string', description: 'dev0:dev1 GPU-busy ratio — split ~50/50 or lopsided?' },
    serial_breakdown: { type: 'string', description: 'where the NON-overlapped (serial) time goes: per-device setup / cuBLAS init / H2D / GEMM / combine / D2H, with ms' },
    two_gpus_run_concurrently: { type: 'boolean', description: 'do device0 and device1 kernels actually execute in overlapping wall-clock windows at all?' },
    is_truly_parallel: { type: 'boolean', description: 'overall: genuinely parallel multi-GPU (concurrent compute + balanced split) or effectively sequential / mis-constructed?' },
    evidence: { type: 'string', description: 'raw nsys numbers: per-device kernel sums, timeline spans, the overlap computation' },
    note: { type: 'string' },
  },
}
const measure = await agent([
  'You are a CUDA performance engineer. MEASURE on the LIVE box5090 whether the two GPUs actually run CONCURRENTLY in a multi-GPU (G==2) f2 run — the decisive test of true parallelism. Produce a per-device GPU timeline with nsys; do NOT theorize.', STD, '',
  'PLAN (adapt): (1) confirm bench + nsys: ' + NSYS_CHECK + ' . (2) Profile a G2 run at P=512 (fits both G1 and G2 on 32GB): ' + NSYS_PROF + ' . (3) Pull the per-device timeline: ' + NSYS_STATS + ' . The cuda_gpu_trace rows carry Device + Start + Duration; GROUP BY device and find, within ONE G2 iteration, whether device-0 kernels and device-1 kernels occupy OVERLAPPING wall-clock windows (concurrent) or run BACK-TO-BACK (sequential). Compute per-device GPU-busy time, the union wall span, the compute-overlap %, and the dev0:dev1 work ratio.',
  'If the bench mixes G1/G2res/G2host in one process making the trace ambiguous, isolate the G2 window by timestamp (G2host is the last cell), or add a tiny NVTX range / a minimal standalone two-device driver if needed (rebuild only if necessary; then revert the edit and leave the tree clean). Classify the serial (non-overlapped) time into setup / cuBLAS / H2D / GEMM / combine / D2H.',
  'THE DECISIVE QUESTION: do device 0 and device 1 execute their COMPUTE kernels at the same wall-clock time (true parallel) or one-after-another (sequential)? Answer it from the timeline. Return the structured verdict with the raw nsys numbers as evidence. Leave git clean (cd ' + R + ' && git status).',
].join('\n'), { schema: MEASURE_SCHEMA, label: 'check:nsys-timeline', phase: 'Check' })

phase('Verdict')
const verdict = await agent([
  'You are the lead architect. Answer the user: is the M4.5 multi-GPU genuinely TRUE PARALLEL multi-GPU, or mis-constructed / partly sequential / parallelized at the wrong place? Use the code read + the MEASURED nsys timeline (numbers are ground truth). WRITE the answer to ' + R + '/docs/cleanup/m4.5/parallelism-check.md (Write tool). READ-ONLY on code.', STD, '',
  'THE CONSTRUCTION READ:\n<<<\n' + (construction || '(died)') + '\n>>>', '',
  'THE NSYS MEASUREMENT (ground truth):\n<<<\n' + JSON.stringify(measure, null, 2) + '\n>>>', '',
  'Where the code read and the measurement disagree, the MEASUREMENT wins. The doc must answer: (1) THE VERDICT — do the two GPUs actually run concurrently, is the work split 50/50, and is it therefore true parallel multi-GPU? Blunt yes/no with the measured overlap % + split ratio; (2) if NOT (or only weakly) parallel, the EXACT sequential bottleneck / mis-construction — cite file:line + the measured serial ms (per-call setup serialized across devices? the driver alloc lock? the combine/D2H serial and dominating? work imbalance? parallelism bolted on at the wrong layer?); (3) is the parallelism in the WRONG PLACE — what the construction SHOULD be (amortize all per-device setup out of the hot path; ensure the two devices compute truly concurrently; overlap combine/D2H with compute; or — per why-d2h — the multi-GPU-vs-D2H framing being the wrong axis); (4) reconcile the why-multigpu-slow.md 74%-overlap (rtxbox) vs the 5090 measurement; (5) the honest bottom line: was multi-GPU set up properly, and if not, the corrected construction. Tag every claim to file:line or a measured number.',
  'Return a tight 8-12 line executive summary directly answering is-it-true-parallel-or-built-wrong: the measured overlap + split, the precise serial bottleneck, and the corrected construction.',
].join('\n'), { label: 'verdict:parallelism', phase: 'Verdict' })

return { construction, measure, verdict }
