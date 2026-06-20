export const meta = {
  name: 'refactor-review-group4',
  description: 'PILOT of the big-refactor review (docs/cleanup/bigrefactor/checklistreview.md). Group 4 — Type & numeric (tasks 4.1-4.7) over ALL ~55 code units (each .cu paired with its .cuh/_impl.cuh, each .cpp with its .hpp, lone headers standalone). Scope -> fan out ONE read-only agent PER UNIT in parallel -> each writes its findings to docs/cleanup/bigrefactor/findings/<slug>.md as a "## Group 4" section (file:line + severity + task id; identify-only, NO fixes) -> a summary agent rolls up findings/_SUMMARY-group04.md. CRITICAL CONTEXT: steppe is FP64-BY-DESIGN (Ozaki emulated FP64 + native FP64 parity oracle, §12) so double literals/math are INTENTIONAL, not a perf bug — do NOT flag them; flag only GENUINE type/numeric bugs (byte-vs-element confusion, unsigned countdown, signed/unsigned mismatch, int-index overflow at P~2500/M~584k scale, wrong index width). Read-only, NO box, NO build. This pilot validates the per-file .md format + signal quality before the other 20 groups fan out.',
  phases: [
    { title: 'Scope', detail: 'pair the code files into review units (cu+cuh, cpp+hpp, lone headers); return the unit list' },
    { title: 'Analyze', detail: 'one read-only agent per unit, parallel: check Group 4 tasks, write findings/<slug>.md' },
    { title: 'Summary', detail: 'roll up findings/_SUMMARY-group04.md (counts by task/severity, the notable units)' },
  ],
}

const R = '/home/suzunik/steppe'
const FIND = R + '/docs/cleanup/bigrefactor/findings'

const STD = [
  'PROJECT: steppe = a GPU/CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2 f-statistics + qpAdm. C++20 + CUDA. Layered: include/steppe (public), src/core (host-pure: domain/fstats/qpadm/internal/io), src/device (backend.hpp seam + cpu/ + cuda/ kernels), src/io (readers/filters). The hot path is reformulated dense LA (cuBLAS/cuSOLVER GEMM/SYRK) + a few custom kernels; the CpuBackend is the parity oracle.',
  'CRITICAL FP64 CONTEXT (do NOT produce noise): steppe is FP64-BY-DESIGN. The f2/f4/qpAdm math runs in native FP64 or Ozaki fixed-slice EMULATED FP64 (mantissa 40), validated bit/tolerance vs ADMIXTOOLS 2 goldens (§12 PARITY LAW). So `double` literals (1.0, 0.5), `sqrt`/`exp`/`pchisq` in double, and FP64 math are INTENTIONAL and correctness-load-bearing — NEVER flag them as "should be 1.0f / use sqrtf / tanks consumer-GPU perf" (that would BREAK parity). Task 4.1 (float-vs-double) is therefore mostly N/A here — only flag a genuinely WRONG narrowing (a float temp that loses precision in a parity-critical path) if you find one, and say why.',
  'SCALE CONTEXT (for 4.2/4.6 index width + overflow): P (population count) up to ~2500, M (SNP count) up to ~584131, n_block up to ~757. The resident f2 tensor is P*P*n_block doubles (up to ~10^10 elements => indices CAN exceed 2^31). Genotype tiles are M*P-ish. So a global index/offset into f2_blocks / the genotype matrix that is computed in 32-bit `int` before widening IS a real overflow bug (flag it: the product P*P*n_block or M*P in int overflows). Thread-local indices that stay small may remain int.',
  'GROUP 4 — TYPE & NUMERIC (the tasks to check, verbatim from the checklist): 4.1 float-vs-double literals/math (see FP64 context — mostly N/A; flag only genuine wrong narrowing). 4.2 Index width: `int` for global indices/offsets into arrays > 2^31 elements (needs size_t/int64_t) — REAL here at scale (f2 tensor, genotype matrix). 4.3 Allocation sizing: cudaMalloc/DeviceBuffer/new missing the `* sizeof(T)` (byte-count vs element-count confusion). 4.4 Unsigned countdown: `for (unsigned i=n-1; i>=0; --i)` never terminates. 4.5 Signed/unsigned compares in loop bounds. 4.6 Int overflow before widening: index arithmetic that overflows `int` BEFORE assignment to a wider type (e.g. `long idx = i*P + j` where i*P overflows int). 4.7 Host/device pointer typing: raw `T*` with no host-vs-device type distinction (a candidate for a thin wrapper so the wrong space can\'t be passed).',
  'OUTPUT RULES: IDENTIFY issues only — do NOT edit code, do NOT propose full rewrites (a one-line "suggested direction" is OK). Each finding: the task id (4.x), file:line, severity (HIGH = real bug / overflow / wrong result; MED = latent/scale-dependent risk; LOW = hygiene), a one-line description, and the suggested direction. If a unit has NO Group-4 issues, say so explicitly ("No Group 4 issues found"). Be PRECISE and cite real line numbers you read — no fabricated findings, no padding. Honest "clean" is a valid + valuable result.',
].join('\n')

phase('Scope')
const UNIT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['units'],
  properties: { units: { type: 'array', items: {
    type: 'object', additionalProperties: false, required: ['slug','files','is_cuda'],
    properties: {
      slug: { type: 'string', description: 'a filesystem-safe slug for the unit, e.g. src__device__cuda__block_sink' },
      files: { type: 'array', items: { type: 'string' }, description: 'the absolute paths in this unit (the .cu + its .cuh/_impl.cuh, or the .cpp + its .hpp, or a lone header)' },
      is_cuda: { type: 'boolean', description: 'true if the unit contains .cu/.cuh (device code) — informs later CUDA-specific groups, harmless here' },
    } } } },
}
const scope = await agent([
  'You are scoping the review units for the steppe codebase. List ALL code files and PAIR them into review units: each .cu with its matching .cuh AND any *_impl.cuh of the same stem (e.g. device_f2_blocks.cu + device_f2_blocks_impl.cuh) AND the closely-related device .hpp if obvious; each .cpp with its same-stem .hpp; a header with no .cpp is its own unit. Goal: a .cu and its .cuh are reviewed TOGETHER (one unit), per the user.', STD, '',
  'DO: `find ' + R + '/src ' + R + '/include -type f \\( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.hpp" \\) | sort`, then group into units by stem/dir. Every file must appear in exactly one unit. Return the structured unit list (slug = path-with-slashes-as-double-underscore of the primary file, no extension; files = the absolute paths; is_cuda = contains .cu/.cuh).',
].join('\n'), { schema: UNIT_SCHEMA, label: 'scope:units', phase: 'Scope' })

const units = (scope && Array.isArray(scope.units)) ? scope.units : []
log('Group 4 scope: ' + units.length + ' units')

phase('Analyze')
const results = await parallel(units.map(u => () => agent([
  'You are a meticulous CUDA/C++ reviewer doing the GROUP 4 (Type & numeric) pass on ONE steppe review unit. READ-ONLY analysis; then WRITE your findings file.', STD, '',
  'THE UNIT: slug=' + u.slug + '; files: ' + (u.files || []).join(', ') + ' (is_cuda=' + u.is_cuda + ').',
  'DO: Read every file in the unit IN FULL. Check each Group 4 task (4.1-4.7) with the FP64 + SCALE context above. Then WRITE the file ' + FIND + '/' + u.slug + '.md with EXACTLY this structure:\n' +
  '```\n# Review findings — ' + u.slug + '\n\nFiles: <the unit files>\n\n## Group 4 — Type & numeric\n\n<for each finding: a line `- [4.x][SEVERITY] file:line — description. Suggested: <direction>.`  OR the single line `No Group 4 issues found.`>\n```\n' +
  'Use the Write tool to create that .md. Cite REAL line numbers from what you read. Do NOT fabricate. Do NOT edit any source file. Return a 1-2 line summary: the unit slug + the count of HIGH/MED/LOW Group-4 findings (or "clean").',
].join('\n'), { label: 'g4:' + u.slug, phase: 'Analyze' })))

phase('Summary')
const summary = await agent([
  'You are rolling up the Group 4 (Type & numeric) review of steppe. The per-unit agents each wrote ' + FIND + '/<slug>.md with a "## Group 4" section + returned a 1-2 line summary.', STD, '',
  'THE PER-UNIT SUMMARIES:\n' + results.filter(Boolean).map((r,i)=>'- '+(units[i]?units[i].slug:i)+': '+String(r).slice(0,200)).join('\n'),
  '', 'DO: glob ' + FIND + '/*.md, read the Group 4 sections, and WRITE ' + FIND + '/_SUMMARY-group04.md with: (1) total units reviewed + how many clean vs with-findings; (2) counts by task (4.1-4.7) and by severity (HIGH/MED/LOW); (3) the top ~10 most important findings (the HIGH ones first) with file:line; (4) any cross-cutting pattern (e.g. a recurring int-index-at-scale risk). Use Write. Return the headline numbers (units, total findings, #HIGH) so the human can judge the pilot quality at a glance.',
].join('\n'), { label: 'summary:group4', phase: 'Summary' })

log('Group 4 pilot complete')
return { units: units.length, summary, results }
