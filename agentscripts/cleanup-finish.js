export const meta = {
  name: 'cleanup-finish',
  description: 'Finish the LAST 2 cleanup items only: geno_reader critic-finalize (its audit draft exists) + holistic refresh of 00-overview.md to all 28 units. Touches NOTHING else — the other 27 reviews are done/committed.',
  phases: [
    { title: 'geno_reader', detail: 'adversarial critic-finalize of the geno_reader audit draft' },
    { title: 'Holistic', detail: 'refresh 00-overview.md across all 28 unit reviews' },
  ],
}

const R = '/home/suzunik/steppe'
const CL = R + '/docs/cleanup'

const STANDARD = `steppe = a CUDA-13 / Blackwell (sm_120) GPU reimplementation of ADMIXTOOLS 2 f-statistics; M0-M4 complete on branch m4-perblock-f2. Bar: a demanding senior engineer's 9.5-10/10.
READ FIRST: ${R}/docs/architecture.md (§2 engineering principles, §4 layout+layering [io is an isolated leaf; CUDA PRIVATE to steppe_device], §7 CUDA idioms+RAII, §8 DRY single-home, §9 config, §11 memory/perf, §12 precision/parity, §13 testing); ${R}/docs/ROADMAP.md §4 (magic-number->config: "no literal may survive except true math constants") §5 §6; ${R}/docs/TODO.md (the "Keeping it GPU-dominant" capability-tier section: capable full-host RTX PRO 6000 vs budget consumer-5090, runtime-detected + explicitly-tagged fallback). Read the unit's directly-related files for context.`

const MANDATE = `QUALITY MANDATE — exhaustive, not shallow.
- NO cap on findings; list every substantiated issue grouped by category; length scales to the file.
- COVER ALL CATEGORIES (skip one only with explicit "N/A because ..."): (1) correctness & bugs; (2) edge cases/failure modes (empty/zero/negative/overflow/degenerate/misaligned, integer width, error paths); (3) numerical/precision vs §12; (4) CUDA idioms/RAII/stream-async/launch-config vs §7 [if applicable]; (5) magic numbers vs §4; (6) decomposition/SRP vs §2; (7) readability/naming/const/[[nodiscard]]/noexcept/comments; (8) performance; (9) layering/API vs §4; (10) testability vs §13; (11) capability tiers (PRO 6000 vs 5090 + tagged fallback).
- VERIFY then QUESTION the verification: read line by line; where a claim depends on documented C++-stdlib / CUDA-host-API behavior, look it up (WebFetch/WebSearch) and CITE it; keep a "## Considered & rejected" section.
- Each finding: location (function + line range), issue, why-it-matters (cite §/doc), concrete fix, severity (high/med/low), effort (S/M/L), before-M4.5? (yes/no).
- END with "## Score: X/10" + rationale, "## What it takes to reach 10/10", "## Good patterns to keep".
- READ-ONLY on source; your only write is the markdown review.`

phase('geno_reader')
const g = await agent(
  `You are the ADVERSARIAL critic finalizing ONE cleanup review: the unit io-geno_reader.\nFiles (read fully + context): ${R}/src/io/geno_reader.hpp, ${R}/src/io/geno_reader.cpp (and ${R}/src/io/eigenstrat_format.hpp which it uses).\nA fresh audit DRAFT is at ${CL}/io-geno_reader.md — read it; it is your starting point.\n\n${STANDARD}\n\nWhere a claim depends on documented C++ standard-library / POSIX / CUDA-host behavior, look it up (WebFetch/WebSearch) and CITE it.\n\nIn order: (1) RE-VERIFY every finding in the draft against the actual code + cited docs — DELETE/DOWNGRADE false positives (to "## Considered & rejected" with the reason), CONFIRM the real ones; (2) the failure mode is STOPPING EARLY — assume the draft missed issues; hunt EXHAUSTIVELY across ALL mandate categories (esp. the parser robustness: integer-width/overflow on the offset multiply + allocation, oversized-file vs header-inconsistent, out-of-range row gather, EIGENSTRAT vs TGENO branch, error paths) until you genuinely cannot find more; (3) question your own additions. ${MANDATE}\n\nOVERWRITE ${CL}/io-geno_reader.md with the FINAL complete review (Write). Return ONLY: "io-geno_reader.md | final score X/10 | N findings".`,
  { label: 'verify:io-geno_reader', phase: 'geno_reader' }
)

phase('Holistic')
const o = await agent(
  `You are the HOLISTIC reviewer for the whole steppe M0-M4 codebase. ALL 28 per-unit reviews are now finalized in ${CL}/*.md — list that directory and READ every per-unit review (28 of them, including device-cuda-f2_blocks_kernel and the freshly-finalized io-geno_reader + io-filter-snp_filter), then read the source tree layout + ${R}/docs/architecture.md (§2/§4/§8) + ${R}/docs/TODO.md (capability tiers).\n\n${STANDARD}\n\nWrite the WHOLE-CODEBASE review to ${CL}/00-overview.md (OVERWRITE the existing 27-unit version) with: (1) CROSS-FILE issues no single-file review can see — layering, DRY duplication across files, contract inconsistencies, §2/§4/§8 gaps; (2) CAPABILITY-TIER COHERENCE — the ONE unified design (capability probe at Resources + non-throwing tagged-degrade path distinct from STEPPE_CUDA_CHECK + override-knobs in DeviceConfig vs discovered-state in Resources) and which files change; (3) a PRIORITIZED MASTER backlog deduped across ALL 28 units, BEFORE-M4.5 vs LATER, each item -> file(s) + concrete fix + severity (ensure device-cuda-f2_blocks_kernel-specific items are folded in now that its review exists); (4) per-area + OVERALL codebase score /10 + the concrete gap to 9.5+. Return a 4-6 line executive summary (overall score + top 5 before-M4.5 items).`,
  { label: 'holistic-overview', phase: 'Holistic' }
)

return { g, o }