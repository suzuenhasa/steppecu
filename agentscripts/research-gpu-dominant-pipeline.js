export const meta = {
  name: 'research-gpu-dominant-pipeline',
  description: 'Research how to keep steppe\'s precompute pipeline GPU-dominant and minimize host-CPU dependence (GPUDirect Storage/cuFile, nvCOMP GPU decompression, async transfer, CUDA graphs, on-GPU orchestration), grounded in our code + vast.ai reality',
  phases: [
    { title: 'Research', detail: 'parallel: one agent per CPU-offload avenue — GDS/cuFile, nvCOMP, async-transfer, launch/CUDA-graphs, on-GPU orchestration' },
    { title: 'Synthesize', detail: 'CPU-minimal end-to-end architecture + prioritized roadmap (quick wins vs research bets) + vast.ai feasibility + milestone mapping' },
  ],
}

const CONTEXT = `
PROJECT: \`steppe\` — a CUDA-13/Blackwell (sm_120) GPU reimplementation of ADMIXTOOLS 2 f-statistics ("precompute-once / fit-many"), run on RENTED vast.ai Docker containers (2x RTX 5090, CUDA 13.0). The f2 GEMM compute is already GPU-bound and fast (~0.3 s for all 757 jackknife blocks at P=768 via grouped strided-batched cuBLAS Ozaki-40 fixed-slice FP64 emulation).

THE PROBLEM TO SOLVE: end-to-end runs (and the future Phase-2 fit engine) have HOST-CPU dependencies that punish the pipeline on a weak or contended host CPU — we just observed a box with a slower CPU inflate the test suite ~1.6x. We want the pipeline to stay GPU-DOMINANT so a bad host CPU doesn't bottleneck it. The host-bound parts to attack (EXCLUDING the CPU long-double reference ORACLE used only in tests — that one is understood and acceptable):
  1. Disk -> host -> GPU INGEST: reading the packed 2-bit genotype .geno (~4 GB), host staging, the H2D copy. Bandwidth-bound; touches host I/O, host RAM bandwidth, and PCIe.
  2. Kernel / cuBLAS LAUNCH OVERHEAD (every launch is CPU-issued) — especially Phase 2's many tiny solves (jackknife, qpWave SVD, qpAdm GLS, model-space search).
  3. Host ORCHESTRATION: SNP->block binning, on-the-fly filters, the f2_from_blocks driver (small in the precompute, larger in Phase 2).

GROUND YOUR RECOMMENDATIONS IN OUR DESIGN — read these:
  /home/suzunik/steppe/docs/architecture.md : §11.1 (out-of-core SNP-tile streaming — pinned double-buffer, resident f2_blocks accumulator), §11.2 (VRAM budget), §11.3 (profiling: Nsight Systems before Nsight Compute), §11.4 (single-node multi-GPU), §5 (S0 format decode + the io leaf), §2 & §7 (reformulate into tensor ops; RAII; CUDA idioms; async pools; streams/graphs).
  Code: /home/suzunik/steppe/src/io/ (geno_reader, the genotype front-end), /home/suzunik/steppe/src/device/cuda/ (decode_af_kernel.cu, cuda_backend.cu, device_buffer.cuh, stream.hpp), /home/suzunik/steppe/src/core/fstats/f2_from_blocks.cpp.

HARD REALITY CHECK: we run on RENTED vast.ai Docker containers. Any solution that needs host kernel modules, MOFED/InfiniBand stack, special /dev device nodes, root-on-the-HOST (not just in-container), a specific host filesystem, or a non-default driver MUST be assessed for whether it actually works in that environment — and you must state the FALLBACK if it doesn't. Genotypes are already 2-bit packed (4 SNPs/byte, EIGENSTRAT/TGENO). Derived matrices (Q/V/N .f64) are large (e.g. ~3.4 GB each at P=4266). CUDA 13.0, driver 580, RTX 5090 consumer cards (NOT datacenter — relevant for some features).

USE WEB RESEARCH: load WebSearch/WebFetch via ToolSearch and consult CURRENT (2025-2026) NVIDIA documentation, cuFile/GDS docs, nvCOMP docs, CUDA programming guide, and credible benchmarks. Cite sources. Be concrete and HONEST about consumer-GPU + rented-container feasibility — do not over-promise.`

const RESEARCH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension','what_it_is','applies_to_steppe','requirements_constraints','vast_container_feasible','expected_benefit','integration_points','risks','recommendation','sources'],
  properties: {
    dimension: { type: 'string' },
    what_it_is: { type: 'string', description: 'concise description of the technology/technique' },
    applies_to_steppe: { type: 'string', description: 'which of OUR host-bound bottlenecks (ingest / launch-overhead / orchestration) it addresses, and how' },
    requirements_constraints: { type: 'string', description: 'hardware/driver/filesystem/kernel-module/library requirements' },
    vast_container_feasible: { type: 'string', description: 'HONEST assessment: does it work inside a rented vast.ai Docker container on a consumer RTX 5090? what is the fallback if not?' },
    expected_benefit: { type: 'string', description: 'qualitative + any quantitative figures from sources; what host CPU/PCIe/IO work it removes' },
    integration_points: { type: 'array', items: { type: 'string' }, description: 'which steppe files/layers/milestones (M5/M4.5/Phase2/new) it touches' },
    risks: { type: 'array', items: { type: 'string' } },
    recommendation: { type: 'string', description: 'adopt-now / M5 / M4.5 / Phase2 / research-bet / not-worth-it — with one-line why' },
    sources: { type: 'array', items: { type: 'string' }, description: 'URLs or doc titles consulted' },
  },
}

const DIMS = [
  { key: 'gds-cufile', title: 'GPUDirect Storage (cuFile API)',
    focus: 'GPUDirect Storage (GDS) and the cuFile API: DMA data NVMe-disk -> GPU memory bypassing the host CPU bounce buffer. Research what it is, the requirements (cuFile lib, nvidia-fs kernel module, GDS-capable filesystem, NVMe, whether MOFED is needed), and CRITICALLY whether it works on a consumer RTX 5090 inside a rented vast.ai Docker container (kernel-module access, /dev/nvidia-fs, compatibility/POSIX fallback mode "cuFile compat"). Assess benefit for our ~4 GB .geno ingest + the large .f64 derived matrices, and how it would slot into io/geno_reader -> a DeviceBuffer read path. State the fallback (cuFile POSIX compat / regular pread) if true GDS is unavailable on vast.' },
  { key: 'nvcomp', title: 'GPU (de)compression (nvCOMP) + storage format',
    focus: 'nvCOMP and GPU-side (de)compression: store genotype/derived data compressed on disk, transfer the smaller compressed bytes, and DECOMPRESS ON THE GPU (zero host decompress work, less disk+PCIe bandwidth). Research nvCOMP codecs (LZ4, Snappy, GDeflate, ANS, Bitcomp, Cascaded) circa 2025-2026, their GPU throughput, and applicability to (a) already-2-bit-packed genotypes (is there exploitable redundancy beyond 2-bit packing? e.g. run-length / low-entropy genotype columns) and (b) the large Q/V/N .f64 and the M7 on-disk f2_blocks cache. Give the bandwidth math: does (smaller read + GPU decompress) beat (raw read + host-light copy)? Integration: a compressed on-disk format for io + the M7 cache. Feasibility on vast (nvCOMP is a userspace lib — easy).' },
  { key: 'async-transfer', title: 'CPU-light host->device transfer (no GDS)',
    focus: 'The PRAGMATIC, definitely-works-on-vast baseline for making host->device transfer CPU-minimal WITHOUT GDS: page-locked (pinned) host staging buffers, cudaMemcpyAsync on a dedicated copy stream overlapped with the compute stream (double/triple buffering), cudaMallocAsync memory pools, mmap + O_DIRECT / large sequential reads, and reading directly into pinned buffers. Quantify how little CPU an optimized async-copy pipeline actually uses (the CPU just issues DMA-triggering reads + memcpy-to-pinned; the DMA engine does the transfer). This is essentially the M5 streaming design — research best practices and pitfalls (pinned-memory cost, page-cache, NUMA) so M5 is CPU-light by construction. This is the SAFE high-priority path; rank it against GDS.' },
  { key: 'launch-graphs', title: 'Launch-overhead & host-sync minimization (CUDA Graphs)',
    focus: 'Minimizing CPU-issued launch overhead and host syncs: CUDA Graphs (capture a launch sequence once, replay with a single host call — huge for Phase 2 where each model fit is the same small kernel sequence run thousands of times), cudaGraphExecUpdate for parameter-only changes, reducing cudaStreamSynchronize/cudaDeviceSynchronize, fusing/batching kernels (we already use grouped strided-batched GEMMs), persistent kernels, and device-side launch (dynamic parallelism). Assess where launch overhead actually bites in steppe: the precompute (few large batched launches — probably fine) vs the Phase-2 fit engine (many tiny SVD/Cholesky/GEMM solves — likely launch-bound). Map to cuda_backend.cu + the future fit engine + architecture §11.3 graph-capture note.' },
  { key: 'on-gpu-orchestration', title: 'On-GPU orchestration / keep-resident + measure-first',
    focus: 'Reducing the host CPU to "kick off + copy out the tiny result" by keeping work GPU-RESIDENT: can SNP->block binning, the on-the-fly filters (currently host predicate-then-drop in io/filter), and the f2_from_blocks orchestration be done on the GPU as kernel passes / device-side reductions, so the CPU never touches per-tile or per-block data? Research device-side scan/segmented-reduction for block binning, GPU filter-mask application fused into the decode tile sweep, and keeping Q/V/N + f2_blocks resident across the whole stream. ALSO research the measure-first discipline (architecture §11.3): how to use Nsight Systems / NVTX to EMPIRICALLY confirm where host time actually goes (GPU-idle gaps = launch/host-bound; un-overlapped copies = ingest-bound) BEFORE optimizing — so we attack the real bottleneck, not a guessed one. Be honest about which orchestration is too tiny to be worth moving to the GPU.' },
]

phase('Research')
const findings = (await parallel(DIMS.map(d => () =>
  agent(
    `You are a senior GPU systems engineer researching ONE avenue to keep steppe GPU-dominant.\n${CONTEXT}\n\nYOUR AVENUE: ${d.title}\n${d.focus}\n\nFirst read the listed architecture sections + code to ground in OUR pipeline; then use WebSearch/WebFetch (load via ToolSearch) for current NVIDIA docs + benchmarks. Return the structured findings per the schema — concrete, honest about vast.ai consumer-GPU container feasibility, with sources.`,
    { schema: RESEARCH_SCHEMA, label: `research:${d.key}`, phase: 'Research' }
  )
))).filter(Boolean)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['cpu_minimal_architecture','quick_wins','research_bets','not_worth_it','vast_ai_feasibility','milestone_mapping','open_questions','summary'],
  properties: {
    cpu_minimal_architecture: { type: 'string', description: 'the end-to-end CPU-light pipeline: NVMe -> GPU -> [decompress] -> decode -> block-bin -> batched GEMM -> resident f2_blocks; state exactly what the CPU still must do' },
    quick_wins: { type: 'array', items: { type: 'string' }, description: 'low-effort, high-value, definitely-works-on-vast items (e.g. pinned+async double-buffer, CUDA graphs)' },
    research_bets: { type: 'array', items: { type: 'string' }, description: 'higher-ceiling but uncertain/feasibility-gated items (e.g. GDS/cuFile, nvCOMP) with the gating condition' },
    not_worth_it: { type: 'array', items: { type: 'string' }, description: 'things that do not pay off and why' },
    vast_ai_feasibility: { type: 'string', description: 'reality check: what is actually usable inside a rented vast.ai container on consumer RTX 5090, and what is not' },
    milestone_mapping: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['item','milestone'], properties: { item: { type: 'string' }, milestone: { type: 'string' } } } },
    open_questions: { type: 'array', items: { type: 'string' }, description: 'things to confirm empirically (e.g. via an Nsight profile or a small spike) before committing' },
    summary: { type: 'string' },
  },
}

const synthesis = await agent(
  `You are the synthesis lead. Below are research findings on five avenues to keep steppe GPU-dominant (minimize host-CPU dependence in ingest, launch overhead, and orchestration). ${CONTEXT}\n\nFINDINGS (JSON):\n${JSON.stringify(findings, null, 1)}\n\nProduce a decisive synthesis per the schema: (1) a CPU-minimal end-to-end architecture and exactly what the CPU is still responsible for; (2) prioritized quick-wins vs research-bets vs not-worth-it; (3) an HONEST vast.ai-container feasibility verdict (what works on a rented consumer-5090 container, what does not, with fallbacks); (4) map each adopted item to a steppe milestone (M5 streaming / M4.5 multi-GPU / Phase 2 fit engine / a new milestone); (5) the open questions to confirm empirically (Nsight profile / small spike) before committing. Be decisive and grounded — favor measure-first and definitely-works-on-vast paths over speculative ones.`,
  { schema: SYNTH_SCHEMA, label: 'synthesize', phase: 'Synthesize' }
)

return { findings, synthesis }