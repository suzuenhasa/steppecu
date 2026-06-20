export const meta = {
  name: 'refactor-review-groups',
  description: 'Big-refactor review, the remaining groups (continues the Group 4 pilot). Runs the groups passed in `args` (an array of group numbers) SEQUENTIALLY — each group fans out one read-only agent PER IN-SCOPE UNIT in parallel, and each agent APPENDS a "## Group N — <name>" section to the EXISTING docs/cleanup/bigrefactor/findings/<slug>.md (read -> append -> write back; NEVER alter prior groups\' sections). The ONLY new file per group is findings/_SUMMARY-groupNN.md. Scope tiers: all=every unit; device=src/device/** units; kernel=units with .cu/.cuh. Groups run sequentially because they append to the same per-file .md (concurrent appends would corrupt). Identify-only, file:line + severity + task id; honest "No Group N issues found" is valid. steppe FP64-by-design context applied throughout (no double-vs-float noise). Read-only, NO box, NO build.',
  phases: [
    { title: 'Scope', detail: 'reconstruct the 61 units from the existing findings/<slug>.md files (slug + Files: + layer/is_cuda)' },
    { title: 'Groups', detail: 'for each requested group (sequential): parallel per-unit append to findings/<slug>.md + a _SUMMARY-groupNN.md' },
  ],
}

const R = '/home/suzunik/steppe'
const FIND = R + '/docs/cleanup/bigrefactor/findings'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics + qpAdm. C++20 + CUDA. Layers: include/steppe (public), src/core (host-pure: domain/fstats/qpadm/internal/io), src/device (backend.hpp seam + cpu/ + cuda/ kernels + device headers), src/io (readers/filters). Hot path = reformulated dense LA (cuBLAS/cuSOLVER GEMM/SYRK) + custom kernels in src/device/cuda/*.cu; CpuBackend is the parity oracle.',
  'FP64 CONTEXT (avoid noise): steppe is FP64-BY-DESIGN — native FP64 + Ozaki emulated FP64 (mantissa 40), bit/tolerance-validated vs AT2 goldens (§12 PARITY LAW). `double` literals/math are INTENTIONAL and parity-load-bearing; NEVER flag "use float / 1.0f / sqrtf / tanks consumer-GPU perf". Single statistic stream + deterministic reductions are an intentional §12 parity requirement, not a perf bug. SCALE: P up to ~2500, M up to ~584131, n_block up to ~757; the resident f2 tensor P*P*n_block can exceed 2^31 elements (so int-index-overflow-before-widening at scale IS a real bug).',
  'OUTPUT RULES: IDENTIFY only — do NOT edit source, do NOT propose full rewrites (a one-line "Suggested:" direction is OK). Each finding: `- [TASK][SEVERITY] file:line — description. Suggested: <direction>.` SEVERITY = HIGH (real bug / wrong result / overflow / UB / race), MED (latent / scale-dependent / missing guarantee), LOW (hygiene). Cite REAL line numbers you read — NO fabrication, NO padding. "No Group N issues found." is a valid, valuable result. Apply the §12/FP64 context so intentional design choices are NOT flagged.',
].join('\n')

// scope tier: 'all' (every unit) | 'device' (src/device/** units) | 'kernel' (units containing .cu/.cuh)
const GROUP_DEFS = {
  2:  { name: 'Deprecated / removed APIs & platform support', scope: 'all', tasks: '2.1 Dropped archs (Maxwell/Pascal/Volta removed in CUDA 13.0; sm_50/60/70 build flags or CMake arch lists fail; min sm_75). 2.2 Texture/surface REFERENCES (texture<...>, cudaBindTexture*) removed in CUDA 12 — hard error; port to texture objects. 2.3 Non-_sync warp intrinsics (also Group 18). 2.4 cudaThreadSynchronize -> cudaDeviceSynchronize.' },
  3:  { name: 'Dead / commented-out code', scope: 'all', tasks: '3.1 Commented-out blocks kept "just in case". 3.2 Unreachable code (#if 0; code after return/break). 3.3 Unused symbols (vars, params, includes, helpers). 3.4 Computed but unread (assigned, never read).' },
  5:  { name: 'Hardcoded values / magic numbers', scope: 'all', tasks: '5.1 Unnamed literals (0.001f, 1024, 0.5) -> named constants. 5.2 Hardcoded sizes/bounds that should be params/derived. 5.3 Duplicated constants (same value in multiple places, e.g. block dim in launch AND as a shared-mem array size — DRIFT is a correctness bug). 5.4 Hardcoded paths/IDs/device ids. 5.5 Ambiguous 32 (warp size vs other) — name it.' },
  6:  { name: 'Naming', scope: 'all', tasks: '6.1 Cryptic names (single-letter/opaque outside tight loop counters; tmp, data2, arr, flag). 6.2 Misleading names (count that is an index, list that is a map). 6.3 Inconsistent conventions in one file (nElements vs num_elements vs n). 6.4 Nonstandard abbreviations.' },
  7:  { name: 'Duplication', scope: 'all', tasks: '7.1 Copy-pasted blocks differing by a constant — extract fn/template. 7.2 Repeated (esp. loop-invariant) expressions — compute once. 7.3 Repeated sizeof/casts — hoist/template. 7.4 Collapsible boilerplate (a macro/helper would fold it).' },
  8:  { name: 'Comments', scope: 'all', tasks: '8.1 Restating code (i++; // increment i). 8.2 Stale comments (describe behavior the code no longer has). 8.3 Missing rationale where needed (non-obvious constants, workarounds, intentional deviations). 8.4 Orphan TODO/FIXME/HACK (no context/owner).' },
  9:  { name: 'Constants & configuration', scope: 'all', tasks: '9.1 Should-be-const/constexpr left mutable. 9.2 Tangled config (tunable knobs buried in logic vs surfaced at file top / config struct). 9.3 Positional booleans foo(true,false,true) — use named flags/enums.' },
  10: { name: 'Initialization', scope: 'all', tasks: '10.1 Late/distant: declared far from first use, or uninitialized-then-assigned. 10.2 Zero-init assumptions: missing init relying on zero-init that does not hold.' },
  11: { name: 'Qualifiers & const-correctness', scope: 'kernel', tasks: '11.1 const __restrict__ on read-only kernel pointers (missed opt + missing aliasing guarantee). 11.2 Inconsistent __host__/__device__/__global__ qualifiers. 11.3 Host/device helper duplicated instead of __host__ __device__. 11.4 Large by-value structs as kernel params (param-space limit) — mark __grid_constant__ or pass by pointer.' },
  12: { name: 'Launch config & indexing', scope: 'kernel', tasks: '12.1 Block dim not a multiple of 32 (warp size). 12.2 Grid dim hardcoded vs (n+block-1)/block. 12.3 Missing grid-stride loop in a one-elem-per-thread kernel (breaks when input exceeds the grid). 12.4 Launch config baked-in vs derived (cudaOccupancyMaxPotentialBlockSize). 12.5 Compute-cap / device-property assumptions hardcoded vs queried.' },
  13: { name: 'Error handling', scope: 'device', tasks: '13.1 Unchecked cuda* API return (wrap in CHECK with file/line). 13.2 Unchecked launches — need cudaGetLastError() (launch error) AND a later sync/check (async error), BOTH. 13.3 Inconsistent checking (some guarded, some not). 13.4 Error-swallowing homegrown macro that hides errors.' },
  14: { name: 'Memory: allocation & lifetime', scope: 'device', tasks: '14.1 Alloc/free mismatch (cudaMalloc freed with free, or new[]/malloc with cudaFree). 14.2 Stream-ordered alloc: plain cudaMalloc/cudaFree on hot paths sync across ALL streams — prefer cudaMallocAsync/FreeAsync + pools. 14.3 Async/sync free pairing (cudaMallocAsync ptr freed with plain cudaFree or vice versa). 14.4 Free before async work on the buffer completes (use-after-free across streams). 14.5 Missing frees on error paths.' },
  15: { name: 'Memory: transfers', scope: 'device', tasks: '15.1 cudaMemcpy (H<->D) inside a loop that should be hoisted/batched/kept-resident. 15.2 Direction enum not matching the actual transfer. 15.3 Pageable host memory for frequent transfers where pinned ~doubles bandwidth.' },
  16: { name: 'RAII: ownership & wrapper hygiene', scope: 'device', tasks: '16.1 Wrap EVERY resource (streams, events, graphs/graph-execs, texture/surface objects, memory pools, CUDA arrays, pinned host memory, library handles cuBLAS/cuSOLVER *Create/*Destroy) — not just device memory; all leak identically. 16.2 Move-only + null-on-move: copy deleted, move resets moved-from to a null handle (copyable freeing wrapper = double-free; un-nulled moved-from = same). 16.3 Rule of five for a freeing destructor. 16.4 Single clear ownership (unique_ptr-like); raw kernel pointers are non-owning views, never freed; do not pass owning wrappers by value. 16.5 Don\'t reinvent the wrapper (thrust::device_vector / unique_ptr+CUDA deleter) unless genuinely needed.' },
  17: { name: 'RAII: lifetime & deleter pitfalls (CUDA-specific)', scope: 'device', tasks: '17.1 Non-throwing destructor (a failed cudaFree/cudaStreamDestroy in a dtor can\'t throw during unwinding — log/swallow; guard teardown-order: context may be gone at exit). 17.2 Deleter matches allocator (cudaFree<->cudaMalloc, cudaFreeHost<->cudaMallocHost/HostAlloc, cudaFreeAsync<->cudaMallocAsync, cudaFreeArray<->cudaMallocArray). 17.3 unique_ptr<T[]> on cudaMalloc (default delete[] is UB) — custom deleter. 17.4 RAII vs async lifetime: frees at scope exit != async work finished (use-after-free) — sync first or tie to stream. 17.5 Multi-GPU device-correct free: deleter must cudaSetDevice to the alloc device (record + restore).' },
  18: { name: 'Correctness traps (wrong numbers, not crashes)', scope: 'kernel', tasks: '18.1 Divergent __syncthreads() (reached by only some threads — in a divergent if / after early return) — UB. 18.2 Missing __syncthreads(): shared mem written then read with no barrier (RAW; also WAR). 18.3 Warp-synchronous assumptions (independent thread scheduling is universal on Turing+; needs __syncwarp()/_sync). 18.4 Non-_sync warp intrinsics (__shfl/__ballot/__any/__all) — use _sync variants with explicit mask. 18.5 Missing bounds guard (no if(idx<n)) — OOB, often silent. 18.6 Cross-thread read without barrier/atomic. 18.7 Order-dependent float reduction assuming a fixed thread-execution order. (NOTE steppe context: §12 REQUIRES fixed-order deterministic reductions — flag a reduction that ASSUMES order without enforcing it, but the deliberate fixed-order ones are correct.)' },
  19: { name: 'Performance: debug leftovers', scope: 'kernel', tasks: '19.1 Stray cudaDeviceSynchronize() left from debugging (serializes everything). NOTE: STEPPE_CUDA_CHECK_KERNEL\'s debug-only sync (gated by NDEBUG) is intentional, not a leftover. 19.2 Leftover printf / #if 0 in kernels. 19.3 Redundant __syncthreads() that no longer guards anything.' },
  20: { name: 'Performance: memory access', scope: 'kernel', tasks: '20.1 Uncoalesced access (consecutive threads not hitting consecutive global addresses). 20.2 Shared-memory bank conflicts (serializing access patterns). 20.3 Re-reading the same global value instead of caching in a register/shared.' },
  21: { name: 'Performance: occupancy & registers', scope: 'kernel', tasks: '21.1 Warp divergence: heavy branching serializing a warp (perf angle; correctness in G18). 21.2 Excessive shared memory per block cratering occupancy. 21.3 Register spills (monolithic kernels spilling to local) — candidate for splitting. 21.4 Missing register hints where they help: #pragma unroll on tight loops, __launch_bounds__, __forceinline__ (deliberately, not reflexively).' },
  22: { name: 'Performance: compute & launch', scope: 'kernel', tasks: '22.1 Atomics where a proper reduction/scan would be far cheaper. 22.2 Integer div/mod in loops (expensive on GPU; precompute / shifts-masks for power-of-two strides). 22.3 Loop-invariant work / repeated index recomputation that should be hoisted. 22.4 Launch overhead: many small/repeated launches dominated by per-launch cost — fuse or capture into a CUDA graph (only where the profiler confirms it).' },
}

// args plumbing proved unreliable for scriptPath launches, so use a hardcoded BATCH
// (prefer args if it actually arrives as a list/number; else the BATCH default).
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
let RUN = (Array.isArray(A) ? A : (typeof A === 'number' ? [A] : [])).map(Number).filter(n => GROUP_DEFS[n])
if (RUN.length === 0) RUN = [5, 6, 7, 8, 9, 10]   // BATCH 1b (2/3/4 done; resume 5-10); batch 2 = [11,12,13,14,15,16,17,18,19,20,21,22]

phase('Scope')
const UNIT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['units'],
  properties: { units: { type: 'array', items: {
    type: 'object', additionalProperties: false, required: ['slug','files','is_cuda','layer'],
    properties: {
      slug: { type: 'string' }, files: { type: 'array', items: { type: 'string' } },
      is_cuda: { type: 'boolean', description: 'the unit contains a .cu or .cuh' },
      layer: { type: 'string', description: 'device | core | io | include (from the file paths)' },
    } } } },
}
const scope = await agent([
  'You are reconstructing the review-unit list for the steppe big-refactor review. The Group 4 pilot already created one findings/<slug>.md per unit. Use those as the canonical unit list so appends hit the RIGHT files.', STD, '',
  'DO: `ls ' + FIND + '/*.md` (EXCLUDE _SUMMARY-*.md). Each filename minus ".md" IS the unit slug. For each, read its "Files:" header line to get the unit\'s file paths. Set is_cuda = any file ends .cu/.cuh; layer = device if any path under src/device, else core (src/core), io (src/io), or include (include/). Return the structured unit list (every findings/<slug>.md, excluding the summary).',
].join('\n'), { schema: UNIT_SCHEMA, label: 'scope:units', phase: 'Scope' })
const units = (scope && Array.isArray(scope.units)) ? scope.units : []
log('Scope: ' + units.length + ' units; running groups [' + RUN.join(', ') + ']')

for (const g of RUN) {
  const def = GROUP_DEFS[g]
  const gg = (g < 10 ? '0' : '') + g
  const inscope = units.filter(u =>
    def.scope === 'all' ? true :
    def.scope === 'device' ? (u.layer === 'device') :
    def.scope === 'kernel' ? (u.is_cuda === true) : true)
  log('Group ' + g + ' (' + def.name + '): ' + inscope.length + ' units [scope=' + def.scope + ']')
  phase('Group ' + g)
  const res = await parallel(inscope.map(u => () => agent([
    'You are a meticulous CUDA/C++ reviewer doing the GROUP ' + g + ' (' + def.name + ') pass on ONE steppe review unit. READ-ONLY analysis, then APPEND your findings to the unit\'s existing findings file.', STD, '',
    'GROUP ' + g + ' — ' + def.name + '. Tasks: ' + def.tasks, '',
    'THE UNIT: slug=' + u.slug + '; files: ' + (u.files || []).join(', ') + ' (is_cuda=' + u.is_cuda + ', layer=' + u.layer + ').',
    'DO: (1) Read every file in the unit IN FULL. (2) Check each Group ' + g + ' task with the §12/FP64/scale context. (3) Read the EXISTING file ' + FIND + '/' + u.slug + '.md (it exists from the Group 4 pilot and may already have other groups\' sections). (4) APPEND a new section to it — Write back the FULL existing content PLUS, at the end, exactly:\n\n## Group ' + g + ' — ' + def.name + '\n\n<your findings as `- [TASK][SEVERITY] file:line — desc. Suggested: <dir>.` lines, OR the single line `No Group ' + g + ' issues found.`>\n\n' +
    'CRITICAL: do NOT remove or alter any OTHER group\'s section. Preserve every existing "## Group X" section for X != ' + g + '. IDEMPOTENCY: if a "## Group ' + g + '" section ALREADY exists in the file (you are a re-run after an interruption), REPLACE that one section in place with your fresh findings — do NOT append a duplicate. Otherwise ADD your "## Group ' + g + '" section at the end. If the file does not exist, create it with `# Review findings — ' + u.slug + '` + `Files: ...` + your section. Cite REAL line numbers; no fabrication. Return a 1-line summary: slug + #HIGH/#MED/#LOW (or "clean").',
  ].join('\n'), { label: 'g' + g + ':' + u.slug, phase: 'Group ' + g })))

  await agent([
    'You are rolling up the GROUP ' + g + ' (' + def.name + ') review of steppe. Per-unit agents appended a "## Group ' + g + '" section to each findings/<slug>.md and returned 1-line summaries.', STD, '',
    'GROUP ' + g + ' tasks: ' + def.tasks, '',
    'PER-UNIT SUMMARIES (' + inscope.length + ' in-scope units, scope=' + def.scope + '):\n' + res.filter(Boolean).map((r,i)=>'- '+(inscope[i]?inscope[i].slug:i)+': '+String(r).slice(0,180)).join('\n'),
    '', 'DO: read the "## Group ' + g + '" sections across ' + FIND + '/*.md and WRITE ' + FIND + '/_SUMMARY-group' + gg + '.md with: (1) coverage (units in scope, clean vs with-findings); (2) counts by task + severity; (3) top findings (HIGH first) with file:line; (4) any cross-cutting pattern. Use Write. Return the headline numbers (units, total findings, #HIGH).',
  ].join('\n'), { label: 'sum:g' + g, phase: 'Group ' + g })
}

log('refactor-review groups [' + RUN.join(', ') + '] complete')
return { ran: RUN, units: units.length }
