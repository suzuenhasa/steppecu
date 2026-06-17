export const meta = {
  name: 'per-file-cleanup-audit',
  description: 'Exhaustive per-unit cleanup audit of steppe M0-M4: one deep agent devoted to each .hpp+.cpp unit writes docs/cleanup/<unit>.md, an adversarial critic re-verifies+extends it, then a holistic capstone. No issue cap; verify against CUDA docs then question the verification; 9.5-10/10 bar.',
  phases: [
    { title: 'Audit', detail: '28 units, one deep agent devoted per unit -> docs/cleanup/<unit>.md (exhaustive, all categories)' },
    { title: 'Verify', detail: 'adversarial critic per unit: re-verify each finding, reject false positives, hunt what was missed, finalize the md' },
    { title: 'Holistic', detail: 'cross-file layering/DRY/conformance + capability-tier coherence + master backlog -> docs/cleanup/00-overview.md' },
  ],
}

const R = '/home/suzunik/steppe'
const CL = R + '/docs/cleanup'

const STANDARD = `steppe = a CUDA-13 / Blackwell (sm_120) GPU reimplementation of ADMIXTOOLS 2 f-statistics; M0-M4 complete on branch m4-perblock-f2. The quality bar is a demanding senior engineer's 9.5-10/10.
READ FIRST (the standard you judge against): ${R}/docs/architecture.md — §2 (engineering principles: DRY/single-source, separation of concerns, RAII everywhere, no global mutable state, reformulate-into-tensor-ops, fail-fast, testability), §4 (repo layout + the dependency-direction rule: app->api->core->device, io is an isolated leaf, CUDA PRIVATE to steppe_device, core must not include a CUDA header), §7 (CUDA idioms: RAII owning wrappers, STEPPE_CUDA_CHECK + post-launch checks, launch-config helpers, async pools, streams/graphs, host/device separation, narrow void launch_xxx wrappers, span/mdspan views not raw pointers), §8 (DRY single-home table), §9 (typed immutable config / injected resources), §11 (§11.1 streaming, §11.2 VRAM budget, §11.4 multi-GPU), §12 (precision/determinism/parity — Ozaki fixed-slice, native-FP64 oracle, host-side fixed-order combine), §13 (testing). Also ${R}/docs/ROADMAP.md §4 (magic-number->config inventory: "no literal may survive except true mathematical constants"), §5 (cross-cutting standards), §6 (definition of done). Also ${R}/docs/TODO.md — especially the "Keeping it GPU-dominant" CAPABILITY-TIER section: the capable path (full-host RTX PRO 6000 sm_120 96GB: GDS, full ncu, official P2P) vs the budget fallback (vast consumer 5090), every capability runtime-detected + degrade with an EXPLICIT logged tag; parity holds on both. Read your unit's directly-related files for context (the interface it implements, the headers it includes, the kernels it calls).`

const MANDATE = `QUALITY MANDATE — read carefully; the user has been burned by shallow reviews that stop at "2-3 issues" and produce a useless <50-line summary. Do NOT do that.
- BE EXHAUSTIVE. There is NO cap on findings. List EVERY substantiated issue, however small, grouped by category. The review LENGTH MUST SCALE TO THE FILE: a substantive .cu / backend may warrant dozens of findings and a long document; a trivial header may warrant fewer — but then you must EXPLICITLY justify why it is already near-perfect (don't pad, but don't stop early). A short summary-style review is a FAILED review.
- COVER ALL OF THESE CATEGORIES for the unit (skip one only by explicitly saying "N/A because ..."): (1) correctness & bugs; (2) edge cases & failure modes — empty/zero/negative/overflow/degenerate/misaligned inputs, integer width/wraparound, error paths; (3) numerical/precision vs §12 (cancellation, accumulation order, native-FP64-vs-Ozaki, determinism); (4) CUDA idioms / RAII / stream & async semantics / launch config / occupancy vs §7 (CUDA units especially); (5) magic numbers & hardcoded values vs §4; (6) decomposition / single-responsibility / function size vs §2; (7) readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density matching surrounding code; (8) performance; (9) layering / API / ABI vs §4; (10) testability vs §13; (11) CAPABILITY TIERS — where does THIS unit touch the PRO-6000-capable vs budget-5090 path, and what should it add for a runtime-detected, explicitly-tagged fallback (e.g. a capability probe, a tagged log line, a config hook)?
- VERIFY, DON'T ASSUME — THEN QUESTION THE VERIFICATION. Read the actual code line by line. Where ANY claim depends on the documented behavior of CUDA / cuBLAS / cuSOLVER / CCCL / Thrust / the C++ standard library, LOOK IT UP in the official docs (WebFetch/WebSearch) and CITE the source — do not assert from memory. After you state an issue, adversarially question your own claim: is it actually wrong, or is there a legitimate reason it is written this way? Maintain a "Considered & rejected" section listing candidate issues you checked and dismissed WITH the reason — that is evidence of depth and guards against false positives.
- For EACH finding give: location (function + line range), the issue, why it matters (cite arch §/ROADMAP §/CUDA doc), the concrete fix, severity (high/med/low), effort (S/M/L), before-M4.5? (yes/no).
- END the document with: a precise SCORE /10 with rationale, an itemized "What it takes to reach 10/10", and "Good patterns to keep".
DOCUMENT STRUCTURE (use this markdown skeleton): a title + "## Role & layering", "## Score: X/10 — <verdict>", "## Findings" with a "### " subsection per category above, "## Considered & rejected", "## What it takes to reach 10/10", "## Good patterns to keep".
READ-ONLY ON SOURCE: you produce a review document only; do NOT edit any source file. Your ONLY write is the markdown review.`

const U = (name, files, cuda) => ({ name, files, cuda: !!cuda })
const UNITS = [
  U('include-config', [`${R}/include/steppe/config.hpp`], false),
  U('include-error', [`${R}/include/steppe/error.hpp`], false),
  U('include-fstats', [`${R}/include/steppe/fstats.hpp`], false),
  U('core-internal-views', [`${R}/src/core/internal/views.hpp`], false),
  U('core-internal-f2_estimator', [`${R}/src/core/internal/f2_estimator.hpp`], false),
  U('core-internal-decode_af', [`${R}/src/core/internal/decode_af.hpp`], false),
  U('core-domain-block_partition_rule', [`${R}/src/core/domain/block_partition_rule.hpp`, `${R}/src/core/domain/block_partition_rule.cpp`], false),
  U('core-fstats-f2_from_blocks', [`${R}/src/core/fstats/f2_from_blocks.hpp`, `${R}/src/core/fstats/f2_from_blocks.cpp`], false),
  U('device-backend', [`${R}/src/device/backend.hpp`], false),
  U('device-cpu-cpu_backend', [`${R}/src/device/cpu/cpu_backend.cpp`], false),
  U('device-cuda-check', [`${R}/src/device/cuda/check.cuh`], true),
  U('device-cuda-device_buffer', [`${R}/src/device/cuda/device_buffer.cuh`], true),
  U('device-cuda-stream', [`${R}/src/device/cuda/stream.hpp`], true),
  U('device-cuda-handles', [`${R}/src/device/cuda/handles.hpp`], true),
  U('device-cuda-decode_af_kernel', [`${R}/src/device/cuda/decode_af_kernel.cuh`, `${R}/src/device/cuda/decode_af_kernel.cu`], true),
  U('device-cuda-f2_block_kernel', [`${R}/src/device/cuda/f2_block_kernel.cuh`, `${R}/src/device/cuda/f2_block_kernel.cu`], true),
  U('device-cuda-f2_blocks_kernel', [`${R}/src/device/cuda/f2_blocks_kernel.cuh`, `${R}/src/device/cuda/f2_blocks_kernel.cu`], true),
  U('device-cuda-cuda_backend', [`${R}/src/device/cuda/cuda_backend.cu`], true),
  U('io-eigenstrat_format', [`${R}/src/io/eigenstrat_format.hpp`, `${R}/src/io/eigenstrat_format.cpp`], false),
  U('io-geno_reader', [`${R}/src/io/geno_reader.hpp`, `${R}/src/io/geno_reader.cpp`], false),
  U('io-snp_reader', [`${R}/src/io/snp_reader.hpp`, `${R}/src/io/snp_reader.cpp`], false),
  U('io-ind_reader', [`${R}/src/io/ind_reader.hpp`, `${R}/src/io/ind_reader.cpp`], false),
  U('io-genotype_tile', [`${R}/src/io/genotype_tile.hpp`], false),
  U('io-filter-filter_decision', [`${R}/src/io/filter/filter_decision.hpp`], false),
  U('io-filter-snp_filter', [`${R}/src/io/filter/snp_filter.hpp`, `${R}/src/io/filter/snp_filter.cpp`], false),
  U('io-filter-mind_prepass', [`${R}/src/io/filter/mind_prepass.hpp`, `${R}/src/io/filter/mind_prepass.cpp`], false),
  U('io-filter-include_exclude', [`${R}/src/io/filter/include_exclude.hpp`, `${R}/src/io/filter/include_exclude.cpp`], false),
  U('io-filter-filter_plan', [`${R}/src/io/filter/filter_plan.hpp`], false),
]

const docsLine = (u) => u.cuda
  ? `This is a CUDA translation unit — research the official NVIDIA docs (CUDA C++ Programming Guide, cuBLAS, cuSOLVER, CCCL/Thrust, Runtime API) via WebFetch/WebSearch and CITE them for EVERY claim about device behavior, stream/async semantics, launch limits, RAII of CUDA handles, or precision/emulation.`
  : `Where any claim depends on documented C++ standard-library or CUDA-host-API behavior, look it up and cite it (WebFetch/WebSearch); do not assert from memory.`

phase('Audit')
const audited = await pipeline(UNITS,
  (u) => agent(
    `You are a meticulous senior CUDA/C++ reviewer devoting your ENTIRE attention to ONE unit of steppe: ${u.name}\nFiles (read them FULLY, plus their directly-related context files): ${u.files.join(', ')}\n\n${STANDARD}\n\n${docsLine(u)}\n\n${MANDATE}\n\nWrite your COMPLETE, exhaustive review to ${CL}/${u.name}.md using the Write tool. Then return ONLY one line: "${u.name}.md | score X/10 | N findings".`,
    { label: `audit:${u.name}`, phase: 'Audit' }
  ),
  (prev, u) => agent(
    `You are the ADVERSARIAL second pass devoted to ONE unit: ${u.name}\nFiles: ${u.files.join(', ')}\nThe first-pass review is at ${CL}/${u.name}.md (auditor reported: ${prev}). Read BOTH the source file(s) and that draft.\n\n${STANDARD}\n\n${docsLine(u)}\n\nYour job, in order: (1) RE-VERIFY every existing finding against the actual code and the official docs — DELETE or DOWNGRADE false positives (move them to "Considered & rejected" with the reason), CONFIRM the real ones, fix any wrong fixes. (2) The known failure mode is STOPPING EARLY — assume the auditor missed issues; hunt EXHAUSTIVELY across ALL categories in the mandate until you genuinely cannot find more. (3) Question your OWN additions the same way. ${MANDATE}\n\nThen OVERWRITE ${CL}/${u.name}.md with the FINAL, complete, verified review (Write tool). Return ONLY one line: "${u.name}.md | final score X/10 | +M added / -K rejected".`,
    { label: `verify:${u.name}`, phase: 'Verify' }
  )
)

phase('Holistic')
const overview = await agent(
  `You are the HOLISTIC reviewer for the whole steppe M0-M4 codebase. The per-unit reviews are finalized in ${CL}/*.md — list that directory and READ every per-unit review, then read the source tree layout and ${R}/docs/architecture.md (§2/§4/§8) + ${R}/docs/TODO.md (capability tiers).\n\n${STANDARD}\n\nProduce the WHOLE-CODEBASE review and WRITE it to ${CL}/00-overview.md with: (1) CROSS-FILE issues no single-file review can see — layering violations, DRY duplication across files, naming/contract inconsistencies, the §2/§4/§8 conformance gaps; (2) CAPABILITY-TIER COHERENCE — is the runtime-probe + explicitly-tagged-fallback pattern (PRO 6000 capable vs budget 5090) applied consistently across files? specify the ONE unified design (a capability probe + tagged results in DeviceConfig/Resources) and which files must change; (3) a PRIORITIZED MASTER backlog deduped across all units, split BEFORE-M4.5 vs LATER, each item -> file(s) + concrete fix + severity; (4) per-area scores and an OVERALL codebase score /10 with the concrete gap to 9.5+/10. Per-unit final scores to fold in: ${JSON.stringify(audited)}\n\nReturn a 4-6 line executive summary (overall score + the top 5 before-M4.5 items).`,
  { label: 'holistic-overview', phase: 'Holistic' }
)

return { audited, overview }