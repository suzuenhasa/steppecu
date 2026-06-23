export const meta = {
  name: 'new-tools-research',
  description: 'RESEARCH ONLY (no code changes): GPU-shape-first designs for the three remaining new-machinery tools — qpfstats, qpGraph, DATES — fanned out one agent each IN PARALLEL, plus a shared code-inventory of what steppe ALREADY has to reuse. THE HARD MANDATE (from the f-stat-sweep disaster): every design must be GPU-FIRST and GPU-BOUND — each agent MUST explicitly identify where the tool could go CPU-bound (the analogue of the sweep host-enumeration that pegged 1 core at GPU-0%) and design it OUT (on-device enumeration/compute/filter/solve/fit, batched, emulated-FP64, CUB/cuSOLVER, only small results crossing the host seam). REFORMULATE THE ALGEBRA where the textbook CPU formulation does not map to the GPU (the user: "reformulate the algebra where necessary, that is fine"). VERIFY everything (hard-verify mandate): the published method (web — the real qpfstats/qpGraph/DATES algorithms + papers) AND the steppe code (file:line) — do not assert a seam exists/is reusable without checking. Each tool agent produces: the algebra (+ any GPU reformulation), the GPU-BOUND design, the CPU-bound failure-mode + how it is prevented, the REUSE inventory (which existing seams, file:line), the genuinely-NEW machinery, the effort, the risks/open-questions. The inventory agent maps ALL reusable seams. Synthesis writes one design doc per tool + the inventory + a combined roadmap. NO IMPLEMENTATION. MULTI-GPU PARKED (design single-GPU; note where multi-GPU would later shard). EmulatedFp64 first. CUDA 13+ (verify APIs against the docs). REAL-AADR-only when any validation is proposed.',
  phases: [ { title: 'Parallel research: inventory + qpfstats + qpGraph + DATES' }, { title: 'Synthesize the per-tool GPU-shape design docs + roadmap' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine @ 4c54a06. BUILT: the qpAdm/qpWave fit engine + the S8 rotation; the standalone f-stats f4/f3/f4-ratio/qpDstat (both f2- and genotype-path) + a GPU-only all-quartets SWEEP (on-device unrank+compute+filter+CUB-compact, bounded top-K, GPU-bound — 2.57B quartets in 177s); Python bindings (M(py-1)). REMAINING new-machinery tools: qpfstats, qpGraph, DATES.',
  'THE HARD LESSON (the whole point of this research): we just shipped a CPU-BOUND f-stat sweep by accident (host enumeration/filter/IO pegged 1 CPU core at GPU-0%), had to roll it back, and rebuild it GPU-bound (on-device unrank + CUB compaction). NEVER AGAIN. Every design here must be GPU-FIRST and provably GPU-BOUND: identify the CPU-bound failure mode (host loops over items/SNPs/pairs/iterations, host-side solves/sorts/fits, big host materialization) and design it ON-DEVICE (batched kernels, cuSOLVER/cuBLAS/CUB, emulated-FP64, only small results to host). If a step seems to need the CPU, REFORMULATE the algebra so it does not.',
  'PRECISION: EmulatedFp64{40} default for matmul-heavy work, native-FP64 fallback + cancellation carve-out (fit-engine.md, commit 8d4f22f). MULTI-GPU PARKED (design single-GPU --device 0; the 5090s have no P2P; note where a future multi-GPU shard would go but do NOT design for it now). CUDA 13.x stack — VERIFY any CUB/cuBLAS/cuSOLVER API against the CUDA 13.x docs (the box has CCCL 3.x). REAL AADR only for any proposed validation; no synthetic.',
  'HARD-VERIFY (the user mandate): cite the published method (paper/admixtools source, web) AND the steppe code (file:line). Do NOT claim a seam is reusable, a formula is correct, or an API exists without checking it. FLAG anything speculative. The box (' + SSH + ') has the admixtools R + the DReichLab C source if you need to read the reference implementation.',
  'NO IMPLEMENTATION, NO code changes. Output is research docs only.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

const TOOL_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['tool','algebra','gpu_bound_design','cpu_bound_risk_and_prevention','reuse_inventory','new_machinery','effort','risks','sources'],
  properties: {
    tool: { type: 'string' },
    algebra: { type: 'string', description: 'the method math + any GPU reformulation needed (cite the paper/admixtools source)' },
    gpu_bound_design: { type: 'string', description: 'the GPU-FIRST, GPU-BOUND design: which kernels/cuSOLVER/cuBLAS/CUB, what is batched on-device, what (small) results cross the host seam' },
    cpu_bound_risk_and_prevention: { type: 'string', description: 'EXPLICITLY: where this tool could go CPU-bound (the sweep-disaster analogue) + exactly how the design prevents it (on-device, reformulated algebra, etc.)' },
    reuse_inventory: { type: 'string', description: 'the EXISTING steppe seams reused (the solve seam, the f2 cache, the genotype decode front-end, the genotype-stat seam, the jackknife, small_linalg, the rotation engine, the precision seam, the CLI/binding scaffold), file:line — VERIFIED' },
    new_machinery: { type: 'string', description: 'the genuinely-new code/kernels/algorithms needed (e.g. a nonlinear optimizer, an FFT/distance-binning kernel, a graph algebra)' },
    effort: { type: 'string', description: 'rough effort + the split reuse-vs-new' },
    risks: { type: 'string', description: 'open questions / parity oracles / the hardest part' },
    sources: { type: 'string', description: 'the paper/admixtools-source + steppe file:line consulted' },
  },
}
const INV_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['reusable_seams','gaps','notes'],
  properties: {
    reusable_seams: { type: 'string', description: 'the full map of reusable steppe seams for the new tools (the solve/cuSOLVER/emulated-FP64 seam, small_linalg, the f2 cache F2BlockTensor, the genotype decode front-end decode_af/read_tile/assign_blocks, the genotype-stat seam run_dstat, the jackknife jackknife_cov/jackknife_diag, the rotation engine run_qpadm_search/fit_models_batched, the CUB usage, the CLI/binding scaffold) — each with file:line + what it does + what it is reusable for' },
    gaps: { type: 'string', description: 'what steppe LACKS that the new tools need (a general nonlinear optimizer, an FFT, a genetic-map/LD engine, per-SNP-pair machinery)' },
    notes: { type: 'string' },
  },
}

phase('Parallel research: inventory + qpfstats + qpGraph + DATES')
const tools = [
  { key: 'qpfstats', p: 'TOOL: qpfstats (admixtools f-stat covariance smoothing / joint full-basis estimation). RESEARCH the algebra (the joint estimation of ALL f2/f3/f4 from a shrunk/smoothed full covariance — read the admixtools qpfstats method + paper) + a GPU-BOUND design on the EXISTING cuSOLVER/emulated-FP64 solve seam + small_linalg. It is "on-seam" (the research/standalone-fstats.md flagged it MEDIUM-HIGH). Verify the solve seam (set_solve_precision, backend.hpp; small_linalg.hpp) is reusable. Identify the CPU-bound risk (a host-side big-covariance solve / shrinkage loop) + keep it on-device (cuSOLVER, batched).' },
  { key: 'qpGraph', p: 'TOOL: qpGraph (admixture-graph fitting). RESEARCH the algebra (graph topology -> expected f-statistics via path algebra; fit = minimize the residual between observed + graph-predicted f-stats over edge weights + admixture proportions, under the block-jackknife covariance) + THE OPTIMIZER (steppe has NO general nonlinear/constrained optimizer — what algorithm: L-BFGS-B / projected gradient / a custom solver, and how to run it GPU-BOUND or at least with the heavy f-stat evaluations on-device). The CPU-bound risk is large here (an iterative host optimizer calling a host objective) — design the objective (graph->f-stat + the GLS residual) ON-DEVICE + consider many-graph/many-restart batching on the GPU. Reuse the f2 cache + the GLS/jackknife + small_linalg. This is HIGH.' },
  { key: 'DATES', p: 'TOOL: DATES (admixture dating via ancestry-LD/covariance decay vs genetic distance). RESEARCH the algebra (the weighted ancestry covariance between SNP PAIRS as a function of genetic distance d, decaying ~A·e^(-t·d)+c -> fit -> t generations; read the DATES/ALDER method + paper) + a GPU-BOUND design for the HARD part: the per-SNP-PAIR-by-genetic-distance accumulation (naive O(M^2) ~ 10^12 pairs) — design it GPU-bound via FFT (the ALDER/Loh trick) OR distance-binned on-device reduction, NOT a host O(M^2) loop. Plus a GPU (or tiny-host) nonlinear exponential-decay FITTER. REUSE the genotype-stat seam from qpDstat Part B (decode_af + read_tile + assign_blocks + the on-device per-SNP path) — qpDstat-B is the explicit on-ramp (docs/research/dates-genotype-stat-seam.md); per-SNP genpos is the down-payment. The CPU-bound risk is THE central design problem here. Verify the genotype-stat seam (run_dstat, src/core/stats/dstat.cpp) + decode_af. Note: the architecture doc excluded an LD engine by design, so this is genuinely new infrastructure.' },
]
const research = await parallel([
  () => tryAgent(['You are the CODE-INVENTORY agent: map EVERYTHING in steppe that the new tools (qpfstats/qpGraph/DATES) can REUSE, and what is missing. READ the actual code (file:line): the solve/precision seam (backend.hpp set_solve_precision/engage_*; the cuSOLVER usage in cuda_backend.cu), src/core/internal/small_linalg.hpp, the f2 cache (include/steppe/fstats.hpp F2BlockTensor + the f2-dir reader), the genotype decode front-end (decode_af, the io readers read_tile/read_snp/read_ind, assign_blocks), the genotype-stat seam (src/core/stats/dstat.cpp run_dstat), the jackknife (jackknife_cov + jackknife_diag), the rotation engine (src/core/qpadm/model_search.cpp run_qpadm_search/fit_models_batched), the CUB usage (the sweep), the CLI/binding scaffold. Produce the reusable-seams map + the gaps (no general optimizer, no FFT, no LD engine). Cite file:line. NO code changes.', STD].join('\n'), { schema: INV_SCHEMA, label: 'research:inventory', phase: 'Parallel research: inventory + qpfstats + qpGraph + DATES' }),
  ...tools.map(t => () => tryAgent(['You are the research agent for ' + t.key + ' (GPU-shape-first design; NO code changes). ' + t.p, STD].join('\n'), { schema: TOOL_SCHEMA, label: 'research:' + t.key, phase: 'Parallel research: inventory + qpfstats + qpGraph + DATES' })),
])
const ok = research.filter(Boolean)
log('research returned: ' + ok.length + '/4')
if (ok.length === 0) { log('--- all research died — HALT'); return { halted: true } }

phase('Synthesize the per-tool GPU-shape design docs + roadmap')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['docs_committed','roadmap','gpu_shape_summary','recommended_order','open_decisions','note'],
  properties: {
    docs_committed: { type: 'string', description: 'the commit hash + the doc paths written (code-inventory + qpfstats/qpgraph/dates designs + roadmap)' },
    roadmap: { type: 'string', description: 'the combined new-tools roadmap (qpfstats/qpGraph/DATES) with effort + dependencies' },
    gpu_shape_summary: { type: 'string', description: 'per tool: is a fully GPU-bound design feasible, and the key CPU-bound risk each one had to design out' },
    recommended_order: { type: 'string', description: 'the recommended build order + why (after the wheel)' },
    open_decisions: { type: 'string', description: 'what needs a user decision before implementing each (e.g. the qpGraph optimizer choice, the DATES FFT-vs-binning)' },
    note: { type: 'string' },
  },
}
const synth = await tryAgent([
  'You are synthesizing the new-tools research into design docs. Inventory + the 3 tool studies:\n<<<\n' + JSON.stringify(ok) + '\n>>>', STD, '',
  'WRITE: docs/research/code-inventory-for-new-tools.md (the reusable-seams map + gaps), docs/research/qpfstats-gpu-design.md, docs/research/qpgraph-gpu-design.md, docs/research/dates-gpu-design.md (each: algebra + GPU-bound design + the CPU-bound-risk-and-prevention + reuse + new machinery + effort + risks), and docs/research/new-tools-roadmap.md (the combined order + dependencies + open decisions). Each doc MUST foreground the GPU-bound design + the explicit CPU-bound-failure-mode-and-prevention (the lesson). Then cd ' + R + ' && git add ONLY those docs/research/*.md, commit (research(new-tools): GPU-shape-first designs for qpfstats/qpGraph/DATES + code inventory + roadmap — every design GPU-bound with the CPU-bound failure-mode designed out) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Return the structured synthesis.',
].join('\n'), { schema: SYNTH_SCHEMA, label: 'synth:new-tools', phase: 'Synthesize the per-tool GPU-shape design docs + roadmap' })
if (synth === null) { log('--- synth died — HALT'); return { halted: true, research: ok } }
log('NEW-TOOLS research: ' + synth.docs_committed + ' — order: ' + String(synth.recommended_order).slice(0,140))
return { research: ok, synth }
