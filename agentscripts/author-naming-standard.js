export const meta = {
  name: 'author-naming-standard',
  description: 'Author the steppe NAMING / STYLE STANDARD doc that Phase C (the 480 LOW fixes, dominated by Group 6 naming = 102 findings + the constants/config/init items) will be held to — the user requires ONE agreed standard before any LOW cleanup, so the vague-variable/naming judgment calls are consistent, not ad-hoc. Read-only: 3 parallel lenses mine the codebase de-facto conventions (naming of types/functions/members/constants; file/namespace/layering/macro idioms; the FP64/precision/CUDA-specific naming) + the Group 6/5/9/3/10 review findings (what actually varies), then a synthesis writes docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md = the codified conventions + concrete rules + a do/dont table, DESCRIPTIVE of the existing best practice (not a new style imposed). NO box, NO edits, NO source changes — this is a doc for the user to APPROVE before Phase C runs.',
  phases: [
    { title: 'Mine', detail: '3 parallel read-only lenses: de-facto naming conventions / file+namespace+macro idioms / the review findings on naming+constants+config+init' },
    { title: 'Author', detail: 'synthesize NAMING-STYLE-STANDARD.md — codified conventions + concrete rules + do/dont, for user approval' },
  ],
}

const R = '/home/suzunik/steppe'
const OUT = R + '/docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md'
const FIND = R + '/docs/cleanup/bigrefactor/findings'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) C++20 reimplementation of ADMIXTOOLS 2. Layers (architecture.md §4): include/steppe (public API), src/core (host-pure: domain/fstats/qpadm/internal/io), src/device (backend.hpp seam + cpu/ + cuda/ kernels + device headers), src/io (readers/filters). FP64-by-design (Ozaki emulated + native, §12). The big-refactor REVIEW is done; the per-file findings are ' + FIND + '/<unit>.md (each has "## Group N" sections). Phase A (HIGH) + Phase B (MED) fixes are committed.',
  'THE GOAL: author ONE NAMING / STYLE STANDARD that Phase C (the 480 LOW fixes) will be applied against. The standard must be DESCRIPTIVE of steppe\'s EXISTING de-facto conventions (codify what the code already does well, so the cleanup is CONSISTENT with the codebase) and prescriptive only where the review found real inconsistency. It is NOT a license to restyle the whole codebase — it is the rulebook for resolving the Group 6 naming (102 findings), Group 5 constants, Group 9 config, Group 10 init, Group 3 dead-code LOW items consistently.',
  'NO box, NO source edits, read-only. The deliverable is the doc for the USER to approve before any LOW fix runs.',
].join('\n')

phase('Mine')
const LENSES = [
  { key: 'naming-conventions', focus: 'Mine the DE-FACTO naming conventions actually used across steppe by reading representative files in each layer (include/steppe/config.hpp + qpadm.hpp, src/core/internal/*.hpp, src/core/qpadm/*.cpp, src/device/backend.hpp, src/device/cuda/cuda_backend.cu + qpadm_fit_kernels.cu + handles.hpp, src/io/*.cpp). Extract the OBSERVED conventions: types (PascalCase? e.g. DeviceF2Blocks, QpAdmModel), functions/methods (snake_case? run_qpadm, assemble_f4), local variables, struct/class members (trailing underscore? blas_, stream_), constants (kPascalCase? kDefaultMantissaBits, kQpMaxNl), enums (enum class Kind?), namespaces (steppe::core::qpadm, steppe::device), template params, file names (snake_case.cu/.cuh/.hpp), kernel names (_kernel suffix? launch_ wrappers?), the STEPPE_ macro prefix, __host__ __device__ usage. Report each convention WITH 2-3 real examples (file:line). Note where a convention is DOMINANT vs MIXED.' },
  { key: 'structure-idioms', focus: 'Mine the FILE / NAMESPACE / LAYERING / MACRO / DOC idioms: the file header-comment style (the // path + purpose + LAYERING note block at the top of each file), include ordering, the namespace nesting + closing-brace comments (}  // namespace steppe::device), the .cu<->.cuh pairing + launch_ wrapper seam, the CUDA-PRIVATE-to-steppe_device rule (§4), the core-is-CUDA-free rule, the RAII wrapper naming (CublasHandle, CusolverDnHandle, MathModeScope), the doc-comment convention (/// for API, the §-section cross-refs to architecture.md), the const/constexpr/[[nodiscard]]/noexcept usage patterns. Read backend.hpp, handles.hpp, a couple .cu/.cuh pairs, config.hpp. Report each idiom with real examples (file:line).' },
  { key: 'review-inconsistencies', focus: 'Mine what the REVIEW flagged as inconsistent/vague — read the Group 6 (Naming) + Group 5 (constants) + Group 9 (config) + Group 10 (init) + Group 3 (dead code) sections across ' + FIND + '/*.md and the _SUMMARY-group06/05/09/10/03.md. Categorize the LOW findings: cryptic/single-letter names (and which are legit loop counters vs real problems), misleading names, inconsistent conventions in one file (nElements vs num_elements vs n), nonstandard abbreviations, unnamed literals, positional booleans, late/distant init. Quantify the big buckets and give representative file:line examples. This tells the standard WHICH rules actually need stating (the real friction points), so the standard is targeted, not generic.' },
]
const mined = await parallel(LENSES.map(L => () => agent([
  'You are mining steppe conventions for the naming/style standard, READ-ONLY: ' + L.focus, '', STD,
  '', 'Use grep + read the actual files (no box, no edits). Report the observed conventions/findings WITH real file:line examples, and flag DOMINANT vs MIXED. Be concrete and honest; this feeds the standard.',
].join('\n'), { label: 'mine:' + L.key, phase: 'Mine' })))

phase('Author')
const doc = await agent([
  'You are the lead author of the steppe NAMING / STYLE STANDARD. Synthesize the 3 mining reports into ONE clear, codebase-faithful standard doc.', '', STD,
  '', 'THE MINING REPORTS:\n' + mined.filter(Boolean).map((m,i)=>'### '+LENSES[i].key+'\n'+m).join('\n\n'),
  '', 'WRITE ' + OUT + ' with: (1) Purpose + scope (this governs the Phase-C LOW cleanup; descriptive of steppe\'s existing conventions, not a new style). (2) THE CONVENTIONS — a section each for: types, functions/methods, variables/params, struct/class members, constants/constexpr, enums, namespaces, files, CUDA kernels + launch wrappers + .cu/.cuh pairing, macros (STEPPE_ prefix), doc-comments + architecture.md §-cross-refs, const/[[nodiscard]]/noexcept — each stated as a RULE with a real in-repo example and the rationale, marking which were already dominant. (3) FP64/PRECISION + CUDA-SPECIFIC naming rules (e.g. don\'t rename double->float; kWarpSize not bare 32; device vs host pointer naming). (4) A DO / DON\'T quick-reference table for the common Group-6 friction points (cryptic names, the legit-loop-counter exception, misleading names, abbreviations, unnamed literals, positional booleans). (5) WHAT NOT TO TOUCH — explicit non-goals so the LOW pass does not churn the whole codebase (e.g. do not mass-rename established public API; do not restyle correct existing names; tight-loop i/j/k counters are FINE; parity-load-bearing names stay). (6) A short "how Phase C applies this" note. Use the Write tool. Then return a 1-screen summary of the key rules + the non-goals so the user can approve at a glance.',
].join('\n'), { label: 'author:standard', phase: 'Author' })

log('naming-standard authored: ' + String(doc).slice(0, 200))
return { mined, doc }
