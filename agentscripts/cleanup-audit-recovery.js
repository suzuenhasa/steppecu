export const meta = {
  name: 'cleanup-audit-recovery',
  description: 'Recover the 3 units a transient 500-wave downgraded (geno_reader, snp_filter re-deepened; f2_blocks_kernel finalized) with the SAME deep audit+critic prompts, then refresh docs/cleanup/00-overview.md to all 28 units.',
  phases: [
    { title: 'Re-audit', detail: 'fresh deep audit + adversarial critic for the 3 affected units' },
    { title: 'Holistic', detail: 'refresh 00-overview.md across all 28 unit reviews' },
  ],
}

const R = '/home/suzunik/steppe'
const CL = R + '/docs/cleanup'

const STANDARD = `steppe = a CUDA-13 / Blackwell (sm_120) GPU reimplementation of ADMIXTOOLS 2 f-statistics; M0-M4 complete on branch m4-perblock-f2. The quality bar is a demanding senior engineer's 9.5-10/10.
READ FIRST (the standard you judge against): ${R}/docs/architecture.md — §2 (engineering principles: DRY/single-source, separation of concerns, RAII everywhere, no global mutable state, reformulate-into-tensor-ops, fail-fast, testability), §4 (repo layout + dependency-direction rule: app->api->core->device, io is an isolated leaf, CUDA PRIVATE to steppe_device, core must not include a CUDA header), §7 (CUDA idioms: RAII owning wrappers, STEPPE_CUDA_CHECK + post-launch checks, launch-config helpers, async pools, streams/graphs, host/device separation, narrow void launch_xxx wrappers, span/mdspan views not raw pointers), §8 (DRY single-home table), §9 (typed immutable config / injected resources), §11 (§11.1 streaming, §11.2 VRAM budget, §11.4 multi-GPU), §12 (precision/determinism/parity), §13 (testing). Also ${R}/docs/ROADMAP.md §4 (magic-number->config inventory: "no literal may survive except true mathematical constants"), §5, §6. Also ${R}/docs/TODO.md — especially the "Keeping it GPU-dominant" CAPABILITY-TIER section (capable full-host RTX PRO 6000 path vs budget consumer-5090 fallback, runtime-detected + explicitly-tagged). Read your unit's directly-related files for context.`

const MANDATE = `QUALITY MANDATE — the user has been burned by shallow reviews that stop at 2-3 issues / a useless <50-line summary. Do NOT do that.
- BE EXHAUSTIVE. NO cap on findings. List EVERY substantiated issue, grouped by category. Length scales to the file (a substantive .cu warrants many findings + a long doc). A short summary-style review is a FAILED review.
- COVER ALL CATEGORIES (skip one only by explicitly saying "N/A because ..."): (1) correctness & bugs; (2) edge cases & failure modes (empty/zero/negative/overflow/degenerate/misaligned inputs, integer width, error paths); (3) numerical/precision vs §12; (4) CUDA idioms / RAII / stream & async semantics / launch config / occupancy vs §7; (5) magic numbers vs §4; (6) decomposition / single-responsibility / function size vs §2; (7) readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density; (8) performance; (9) layering / API / ABI vs §4; (10) testability vs §13; (11) CAPABILITY TIERS — where does this unit touch the PRO-6000-capable vs budget-5090 path, what to add for runtime-detected, explicitly-tagged fallback.
- VERIFY, DON'T ASSUME — THEN QUESTION THE VERIFICATION. Read the code line by line. Where any claim depends on documented CUDA/cuBLAS/cuSOLVER/CCCL/C++-stdlib behavior, LOOK IT UP (WebFetch/WebSearch) and CITE it. After asserting an issue, adversarially question your own claim; keep a "Considered & rejected" section listing candidates you checked and dismissed WITH the reason.
- For EACH finding: location (function + line range), the issue, why it matters (cite arch §/ROADMAP §/CUDA doc), the concrete fix, severity (high/med/low), effort (S/M/L), before-M4.5? (yes/no).
- END with: a precise SCORE /10 + rationale, "What it takes to reach 10/10", "Good patterns to keep".
DOCUMENT STRUCTURE: title + "## Role & layering", "## Score: X/10 — <verdict>", "## Findings" (a "### " subsection per category), "## Considered & rejected", "## What it takes to reach 10/10", "## Good patterns to keep".
READ-ONLY ON SOURCE: produce a review document only; do NOT edit any source file. Your ONLY write is the markdown review.`

const UNITS = [
  { name: 'io-geno_reader', files: [`${R}/src/io/geno_reader.hpp`, `${R}/src/io/geno_reader.cpp`], cuda: false },
  { name: 'io-filter-snp_filter', files: [`${R}/src/io/filter/snp_filter.hpp`, `${R}/src/io/filter/snp_filter.cpp`], cuda: false },
  { name: 'device-cuda-f2_blocks_kernel', files: [`${R}/src/device/cuda/f2_blocks_kernel.cuh`, `${R}/src/device/cuda/f2_blocks_kernel.cu`], cuda: true },
]
const docsLine = (u) => u.cuda
  ? `This is a CUDA translation unit — research the official NVIDIA docs (CUDA C++ Programming Guide, cuBLAS, cuSOLVER, CCCL, Runtime API) via WebFetch/WebSearch and CITE them for EVERY claim about device behavior, stream/async semantics, launch limits, RAII of CUDA handles, or precision/emulation.`
  : `Where any claim depends on documented C++ standard-library or CUDA-host-API behavior, look it up and cite it (WebFetch/WebSearch); do not assert from memory.`

phase('Re-audit')
const redone = await pipeline(UNITS,
  (u) => agent(
    `You are a meticulous senior CUDA/C++ reviewer devoting your ENTIRE attention to ONE unit of steppe: ${u.name}\nFiles (read FULLY + their context): ${u.files.join(', ')}\n\n${STANDARD}\n\n${docsLine(u)}\n\n${MANDATE}\n\nWrite your COMPLETE, exhaustive review to ${CL}/${u.name}.md using the Write tool (overwrite any existing draft). Then return ONLY: "${u.name}.md | score X/10 | N findings".`,
    { label: `audit:${u.name}`, phase: 'Re-audit' }
  ),
  (prev, u) => agent(
    `You are the ADVERSARIAL second pass devoted to ONE unit: ${u.name}\nFiles: ${u.files.join(', ')}\nThe first-pass review is at ${CL}/${u.name}.md (auditor reported: ${prev}). Read BOTH the source and that draft.\n\n${STANDARD}\n\n${docsLine(u)}\n\nIn order: (1) RE-VERIFY every finding against the code + official docs — DELETE/DOWNGRADE false positives (to "Considered & rejected" with the reason), CONFIRM the real ones; (2) the known failure mode is STOPPING EARLY — assume the auditor missed issues; hunt EXHAUSTIVELY across ALL mandate categories until you genuinely cannot find more; (3) question your OWN additions too. ${MANDATE}\n\nOVERWRITE ${CL}/${u.name}.md with the FINAL complete review (Write). Return ONLY: "${u.name}.md | final score X/10 | +M added / -K rejected".`,
    { label: `verify:${u.name}`, phase: 'Re-audit' }
  )
)

phase('Holistic')
const overview = await agent(
  `You are the HOLISTIC reviewer for the whole steppe M0-M4 codebase. ALL 28 per-unit reviews are finalized in ${CL}/*.md — list that directory and READ every per-unit review (there are now 28, including device-cuda-f2_blocks_kernel which was just added), then read the source tree layout + ${R}/docs/architecture.md (§2/§4/§8) + ${R}/docs/TODO.md (capability tiers).\n\n${STANDARD}\n\nWrite the WHOLE-CODEBASE review to ${CL}/00-overview.md (OVERWRITE the existing 27-unit version) with: (1) CROSS-FILE issues no single-file review can see — layering, DRY duplication across files, contract inconsistencies, §2/§4/§8 gaps; (2) CAPABILITY-TIER COHERENCE — the ONE unified design (capability probe at Resources + non-throwing tagged-degrade path + override-knobs in DeviceConfig vs discovered-state in Resources) and which files change; (3) a PRIORITIZED MASTER backlog deduped across ALL 28 units, BEFORE-M4.5 vs LATER, each item -> file(s) + concrete fix + severity (fold in any device-cuda-f2_blocks_kernel-specific items now that its review exists); (4) per-area + OVERALL codebase score /10 + the concrete gap to 9.5+. Return a 4-6 line executive summary (overall score + top 5 before-M4.5 items).`,
  { label: 'holistic-overview', phase: 'Holistic' }
)

return { redone, overview }