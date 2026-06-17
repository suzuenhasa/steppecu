export const meta = {
  name: 'code-cleanup-review',
  description: 'Read-only review of steppe M0–M4 against architecture.md standards: magic numbers, function decomposition, readability/casting, CUDA tech-debt, folder/layer org & conformance — produces a prioritized cleanup backlog (no edits)',
  phases: [
    { title: 'Review', detail: 'parallel: one agent per lens — magic-numbers, decomposition, readability, cuda-debt, org/conformance' },
    { title: 'Backlog', detail: 'consolidate into a prioritized, deduped cleanup backlog split pre-M4.5 vs later + conformance gaps + folder-reorg proposal' },
  ],
}

const CONTEXT = `
PROJECT: \`steppe\` — CUDA-13/Blackwell GPU reimplementation of ADMIXTOOLS 2 f-statistics. Codebase: M0–M4 complete, on local branch m4-perblock-f2 at /home/suzunik/steppe. We are CLEANING UP M0–M4 to a high standard BEFORE building the next milestone (M4.5 multi-GPU) — to set a good standard for future work and pay down debt while the surface is small.

THIS IS A READ-ONLY REVIEW. DO NOT edit any file. Produce a concrete, prioritized cleanup backlog. For EACH finding give: file + location (function / approximate line), the issue, why it matters (cite the architecture principle/section it violates), the CONCRETE fix, severity (high/med/low), effort (S/M/L), and whether it should be fixed BEFORE M4.5 (true/false). Be concrete and useful — not nitpicking for its own sake; every finding should make the code clearly better or prevent a future bug.

THE STANDARD to review against (READ these first):
- /home/suzunik/steppe/docs/architecture.md : §2 (engineering principles — DRY/single-source, strict separation of concerns, RAII everywhere, no global mutable state, reformulate-into-tensor-ops, fail-fast, testability, correctness-before-speed), §4 (repository layout + the dependency-direction rule: app→api→core→device, io is an isolated leaf, CUDA is PRIVATE to steppe_device — core must not include a CUDA header), §7 (CUDA idioms — RAII owning wrappers, STEPPE_CUDA_CHECK + post-launch checks, launch-config helpers, async mem pools, streams/graphs, host/device separation, narrow void launch_xxx wrappers), §8 (DRY single-home table), §11.1/§11.2 (out-of-core streaming + VRAM budget), §12 (determinism/precision).
- /home/suzunik/steppe/docs/ROADMAP.md : §4 (the magic-number→config inventory — "no literal may survive except true mathematical constants"), §5 (cross-cutting standards), §6 (definition of done).

SCOPE — review the M0–M4 source under /home/suzunik/steppe/:
  include/steppe/ (config.hpp, error.hpp, fstats.hpp);
  src/core/ (internal/{views,f2_estimator,decode_af}.hpp, domain/block_partition_rule.{hpp,cpp}, fstats/f2_from_blocks.{hpp,cpp});
  src/device/ (backend.hpp, cpu/cpu_backend.cpp, cuda/{check.cuh,device_buffer.cuh,stream.hpp,handles.hpp,decode_af_kernel.{cu,cuh},f2_block_kernel.{cu,cuh},f2_blocks_kernel.{cu,cuh},cuda_backend.cu});
  src/io/ (eigenstrat_format, {geno,snp,ind}_reader, genotype_tile, filter/*);
  the CMakeLists.txt files. (Tests are lower priority — note only egregious issues.)

KNOWN SEED ITEMS (already flagged by the user from an M4 read — CONFIRM each with exact file:location, fold into your lens, then find MORE of the same kind):
- VRAM-budget literal \`0.80 * free_b\` should be a named constexpr (e.g. kMaxVramUtilizationFraction) in config.hpp.
- \`compute_f2_blocks\` is a >100-line monolith (block layout + bucket sorting + VRAM budget calc + chunked kernel launches) — extract testable private helpers.
- Repeated inline \`static_cast<std::size_t>(...)\` casting noise (e.g. \`static_cast<std::size_t>(n_block < 0 ? 0 : n_block)\`) — sanitize+cast inputs ONCE into named consts at function top, use thereafter.
- Default-stream debt: cuda backend uses \`stream_ = nullptr\`, so \`cudaMemcpyAsync\` is effectively SYNCHRONOUS wrt host — a dedicated RAII Stream must land before the out-of-core overlap (M5).`

const FINDING = {
  type: 'object',
  additionalProperties: false,
  required: ['file','location','issue','why_it_matters','fix','severity','effort','before_m45'],
  properties: {
    file: { type: 'string' },
    location: { type: 'string', description: 'function and/or approximate line' },
    issue: { type: 'string' },
    why_it_matters: { type: 'string', description: 'cite the architecture.md/ROADMAP principle violated' },
    fix: { type: 'string', description: 'the concrete change' },
    severity: { type: 'string', enum: ['high','med','low'] },
    effort: { type: 'string', enum: ['S','M','L'] },
    before_m45: { type: 'boolean', description: 'should this be fixed before starting M4.5?' },
  },
}
const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens','findings','notes'],
  properties: {
    lens: { type: 'string' },
    findings: { type: 'array', items: FINDING },
    notes: { type: 'string', description: 'any overall observation for this lens (good patterns to keep, or gaps)' },
  },
}

const LENSES = [
  { key: 'magic-numbers', title: 'Magic numbers & hardcoded values → config',
    focus: 'Find hardcoded literals / magic numbers that should be named constexpr (config.hpp) or named locals. Seed: the 0.80 VRAM fraction. Scan ALL M0–M4 source for surviving literals — block/tile sizes, thresholds, byte counts, launch geometry (grid/block dims), tolerances, buffer sizes — anything that is NOT a true mathematical constant (per ROADMAP §4 the only allowed bare literals are true math constants like the 2 in a²−2ab+b²). For each, name the production home (a config.hpp constant, or a named const at top of function).' },
  { key: 'decomposition', title: 'Function decomposition / monolithic functions',
    focus: 'Find over-long functions that mix concerns. Seed: compute_f2_blocks (>100 lines: block layout + bucket sorting + VRAM budget + chunked launches). Propose extraction into named, single-purpose private helpers, and say which deserve their own unit test. Apply across cuda_backend.cu, the kernels, the io readers, and any host orchestration. Cite §2 (separation of concerns) + §13 (testability).' },
  { key: 'readability', title: 'Readability & idiom hygiene',
    focus: 'Readability/idiom issues that obscure logic: repeated inline casts (sanitize-once pattern), const-correctness, unclear names, [[nodiscard]]/noexcept consistency, comment density that does not match the surrounding code, narrow launch wrappers (host never sees <<<>>>), and span/view usage vs raw pointers (§7). Seed: the static_cast<std::size_t> noise. Be concrete; avoid pure style bikeshedding.' },
  { key: 'cuda-debt', title: 'CUDA correctness & technical debt (§7/§12)',
    focus: 'CUDA correctness + debt vs §7/§12: the DEFAULT-STREAM debt (cudaMemcpyAsync on the null stream is synchronous wrt host — must pay before M5 overlap; flag every use), RAII completeness (every cudaMalloc/handle/workspace owned; nothing outside the allocation allowlist; move-construct AND move-assign on owning types), stream-ordering correctness with cudaMallocAsync, explicit cublasSetWorkspace for emulated-FP64 determinism (§12), post-launch STEPPE_CUDA_CHECK_KERNEL after every launch, and any place native-FP64 vs Ozaki precision could silently drift. Flag debt that will bite M4.5/M5 specifically.' },
  { key: 'org-conformance', title: 'Folder/layer organization & architecture conformance',
    focus: 'Folder/layer org + conformance vs §4/§8/§2. Does the file/folder layout match §4? Any LAYERING VIOLATION (a CUDA header leaking into core; io depending on core/device; core issuing a GEMM/CUDA call directly instead of via ComputeBackend)? Any DRY violation / duplicated logic across the M1–M4 additions (is the decode / f2 / filter / block logic truly single-sourced per §8, or copy-pasted)? Further folder-org opportunities (e.g. should filter/ or the kernels be grouped differently)? Does the code uphold the §2 principles (no global mutable state, injected resources, fail-fast validation)? Propose concrete moves/splits, but only where they improve the layering — not churn for its own sake.' },
]

phase('Review')
const reviews = (await parallel(LENSES.map(l => () =>
  agent(
    `You are a meticulous senior C++/CUDA reviewer. ${CONTEXT}\n\nYOUR LENS: ${l.title}\n${l.focus}\n\nRead the standard (architecture.md §§ + ROADMAP §4/§5) and the in-scope source, then return findings per the schema. Confirm the seed items relevant to your lens with exact file:location and find MORE of the same class. READ-ONLY — propose fixes, do not edit.`,
    { schema: REVIEW_SCHEMA, label: `review:${l.key}`, phase: 'Review' }
  )
))).filter(Boolean)

phase('Backlog')
const BACKLOG_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['before_m45','later','conformance_gaps','folder_reorg','good_patterns','summary'],
  properties: {
    before_m45: { type: 'array', description: 'prioritized cleanup items to do BEFORE M4.5', items: {
      type: 'object', additionalProperties: false, required: ['item','file','fix','severity','effort'],
      properties: { item: { type: 'string' }, file: { type: 'string' }, fix: { type: 'string' }, severity: { type: 'string' }, effort: { type: 'string' } } } },
    later: { type: 'array', description: 'cleanup items that can wait', items: {
      type: 'object', additionalProperties: false, required: ['item','file','fix'],
      properties: { item: { type: 'string' }, file: { type: 'string' }, fix: { type: 'string' } } } },
    conformance_gaps: { type: 'array', items: { type: 'string' }, description: 'genuine architecture-conformance violations (layering/DRY/RAII) — high priority' },
    folder_reorg: { type: 'string', description: 'any proposed folder/structure change, or "none needed"' },
    good_patterns: { type: 'array', items: { type: 'string' }, description: 'patterns worth keeping/standardizing across future milestones' },
    summary: { type: 'string' },
  },
}
const backlog = await agent(
  `You are the cleanup lead. Below are five lens reviews of the steppe M0–M4 codebase against its architecture standards. ${CONTEXT}\n\nREVIEWS (JSON):\n${JSON.stringify(reviews, null, 1)}\n\nConsolidate into ONE prioritized cleanup backlog per the schema: dedup overlapping findings, rank by severity×effort, split into BEFORE-M4.5 (debt that sets the standard / will bite multi-GPU+streaming — e.g. the default-stream debt, the 0.80 magic number, the compute_f2_blocks decomposition) vs LATER, surface any genuine architecture-conformance GAPS (layering/DRY/RAII — high priority), give a folder-reorg verdict, and note good patterns worth standardizing. Be decisive and concrete — this backlog feeds docs/TODO.md and a controlled fix pass.`,
  { schema: BACKLOG_SCHEMA, label: 'consolidate-backlog', phase: 'Backlog' }
)

return { reviews, backlog }