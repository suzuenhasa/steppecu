export const meta = {
  name: 'fixpass-completeness-audit',
  description: 'FORENSIC completeness audit: cross-check every item EVERY fix-pass workflow DECLARED against the ACTUAL code + git history + the audit-doc findings, to capture items that did NOT get fixed/committed properly (verdict-reverts, API-error SKIPs, HALTs, or "passed" shams that the code does not reflect) — especially during API problems. READ-ONLY (greps the session transcript + git + code; writes only docs/cleanup/missing-fixes.md). Runs safely in parallel with an in-flight fix-pass.',
  phases: [
    { title: 'Extract', detail: 'parse the session transcript for EVERY fix-pass ledger -> per-item declared status + commit + note' },
    { title: 'Cross-check', detail: 'parallel, one agent per fix-pass: verify each declared item is GENUINELY in the code + committed' },
    { title: 'Synthesize', detail: 'the master MISSING/INCOMPLETE backlog + which current issues it explains -> docs/cleanup/missing-fixes.md' },
  ],
}

const R = '/home/suzunik/steppe'
const TX = '/home/suzunik/.claude/projects/-home-suzunik-steppe/48ef1b13-abff-4ff9-b585-bc29ae04d1e7.jsonl'
const WFDIR = '/home/suzunik/.claude/projects/-home-suzunik-steppe/48ef1b13-abff-4ff9-b585-bc29ae04d1e7/subagents/workflows'

const CONTEXT = [
  'PROJECT: steppe = CUDA-13/Blackwell reimpl of ADMIXTOOLS 2 f-statistics, branch m4.5-multigpu. Over many sessions we ran a series of fix-pass / implementation WORKFLOWS, each with a "fixer + independent verdict" loop that was supposed to commit-on-PASS / revert-on-FAIL. BUT: some runs hit API errors (transient nulls, server 500s), some used skip-and-continue, some verdicts could not independently re-run on the box. SO some DECLARED items may NOT have actually landed in the code — a known case: in m4.5-perf-fix-pass, P1 (combine resize-not-zero) FAILED its verdict and was reverted/skipped (only P2 committed). The user suspects MORE such gaps and that they "explain a lot of issues we are having." THIS audit finds ALL of them.',
  'THE FIX-PASS / IMPL WORKFLOWS + their DECLARED items (verify each against reality):',
  '  * fix-pass-phase1 (HALT-on-fail): B7, B1, B2, B3, B4, B5, B6.',
  '  * fix-pass-phase2 (SKIP-and-continue; B8 + B17 noted "already committed" separately): B9, B10, B11, B12, B13, B14, B15, B16, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27.',
  '  * m4.5-scaffold: U1 (check.cuh STEPPE_CUDA_WARN), U2 (backend.hpp BackendCapabilities), U3 (config prefer_p2p_combine), U4 (handles ordinal + MathModeScope), U5 (cuda_backend device_id + probe + emu-degrade).',
  '  * m4.5-multigpu: I1 (resources/build_resources), I2 (orchestrator + host-staged combine + parity test), I3 (P2P combine).',
  '  * m4.5-fix-pass (M4.5 cleanup): T1 (parity VRAM-gate), B1 (fan-out), B3, B4 (its verdict noted it COULD NOT re-run on the box — extra-suspect), B5, B6, B7, B8, B9.',
  '  * m4.5-perf-fix-pass (IN-FLIGHT, do not disturb): P0 (committed 970fa42), P1 (FAILED — NOT committed), P2 (committed 9fdc946), P3 (in-flight), P4.',
  'SOURCES (read-only): the session transcript ' + TX + ' (each workflow returned a {ledger:[{item,pass,commit_hash,note,...}]} that appears in a task-notification result; the per-item fixer/verdict transcripts are under ' + WFDIR + '/<runid>/). git history (cd ' + R + ' && git log --all --oneline; git log -S<symbol>; git show <hash>). The actual code under ' + R + '/src, /include, /tests. The audit-doc FINDINGS the items were meant to fix: ' + R + '/docs/cleanup/00-overview.md (B1-B27 = the master backlog rows) and ' + R + '/docs/cleanup/m4.5/00-overview.md (the M4.5 B1-B9) and ' + R + '/docs/cleanup/m4.5/perf-discovery.md (P0-P4).',
  'A declared item is GENUINELY DONE only if: its concrete fix (per the audit-doc row) is PRESENT in the current code AND committed (a real diff in git history), AND any required test exists/is wired. NOT done = no commit, or the verdict reverted it and nothing re-did it, or the "commit" is a sham/comment-only/later-reverted, or only partially done.',
].join('\n')

phase('Extract')
const ledger = await agent([
  'You are a forensic build historian. Extract the COMPLETE per-item declared status for EVERY fix-pass/impl workflow listed below, from the session transcript. READ-ONLY.', CONTEXT, '',
  'Grep/parse ' + TX + ' for each workflow\'s returned ledger (the task-notification <result> JSON contains {"ledger":[{"item","pass","commit_hash","note",...}]}; some have parity_bit_identical / regression / green_count / g2_768_ms fields). Use python3 to parse the JSONL robustly (json.loads each line; find message.content tool_result / the task-notification result strings; extract every ledger entry). ALSO note any item whose note contains: "API error", "transient", "null", "SKIPPED", "HALT", "reverted", "could not", "off by one", "design call", or pass=false.',
  'Return a single consolidated TABLE: one row per (workflow, item) = declared pass (T/F) | commit_hash | a 1-line note (verbatim-ish, esp. any API-error/skip/revert/could-not-verify flag). Cover ALL items in the CONTEXT list. Be exhaustive and exact — this table is the spine the cross-check builds on. Flag the SUSPECT rows (non-pass, API-error/skip/HALT notes, or verdicts that could not independently re-run) explicitly.',
].join('\n'), { label: 'extract-ledgers', phase: 'Extract' })

phase('Cross-check')
const GROUPS = [
  { key: 'phase1', items: 'B7,B1,B2,B3,B4,B5,B6 (fix-pass-phase1, halt-on-fail)' },
  { key: 'phase2', items: 'B9-B27 + the separately-committed B8,B17 (fix-pass-phase2, skip-and-continue — the API-500-wave run; highest risk of dropped items)' },
  { key: 'm4.5-scaffold', items: 'U1-U5 (m4.5-scaffold)' },
  { key: 'm4.5-multigpu', items: 'I1-I3 (m4.5-multigpu)' },
  { key: 'm4.5-fix-pass', items: 'T1,B1,B3,B4,B5,B6,B7,B8,B9 (m4.5-fix-pass — note B4 verdict could NOT re-run on the box)' },
  { key: 'm4.5-perf-fix-pass', items: 'P0,P1,P2,P3,P4 (IN-FLIGHT — P1 confirmed NOT committed; verify the others against committed git only, do not touch the working tree)' },
]
const checks = await parallel(GROUPS.map((g) => () => agent([
  'You are a forensic code auditor verifying ONE fix-pass\'s declared items against REALITY. READ-ONLY — do not edit source, do not touch the working tree (a fix-pass may be running).', CONTEXT, '',
  'THE DECLARED-LEDGER TABLE (from the extract phase):\n<<<\n' + (ledger || '(missing)') + '\n>>>', '',
  'YOUR FIX-PASS: ' + g.key + ' — items: ' + g.items + '.',
  'For EACH item: (1) read its intended fix from the audit-doc row (docs/cleanup/00-overview.md for B*, docs/cleanup/m4.5/00-overview.md for the M4.5 B*, perf-discovery.md for P*); (2) check git (git log --all --oneline | grep the item; git show the claimed commit_hash; git log -S a key symbol/string of the fix) for whether a REAL commit landed; (3) read the ACTUAL current code at the fix\'s file(s) to confirm the change is PRESENT (not reverted, not a sham/comment-only, test wired if required). For a FAILED/SKIPPED/API-errored item, check whether a LATER commit re-did it. Classify each: DONE (cite commit + the code evidence) / MISSING (declared but not in code/git) / PARTIAL (some parts landed, some not — name which) / SHAM (committed but the code does not actually address the finding). Be adversarial and exact; cite commit hashes + file:line.',
  'Return: per-item verdict (DONE/MISSING/PARTIAL/SHAM) + evidence, and a short list of the genuinely-missing/incomplete items for this fix-pass with the concrete re-fix each needs.',
].join('\n'), { label: 'check:' + g.key, phase: 'Cross-check' }))).then(a => a.filter(Boolean))

phase('Synthesize')
const overview = await agent([
  'You are the lead. Synthesize the forensic completeness audit into the master MISSING-FIXES backlog and WRITE it to ' + R + '/docs/cleanup/missing-fixes.md (Write tool). READ-ONLY on source.', CONTEXT, '',
  'THE DECLARED-LEDGER TABLE:\n<<<\n' + (ledger || '') + '\n>>>',
  'THE PER-FIX-PASS CROSS-CHECKS:\n<<<\n' + checks.join('\n\n---\n\n') + '\n>>>', '',
  'The doc must contain: (1) THE HEADLINE — how many declared items did NOT genuinely land, across which fix-passes, and the pattern (API-error skips? verdict-reverts-never-redone? could-not-verify? shams?); (2) the MASTER TABLE of every MISSING / PARTIAL / SHAM item: workflow | item | what it was meant to fix (audit-doc row) | why it is missing (verdict-fail / API-error / skip / sham / could-not-verify) | the concrete re-fix + file(s) | severity | PARITY-SAFE?; (3) WHICH CURRENT ISSUES this explains (e.g. P1 not landing = the ~1.4s combine zeroing still in the hot path = part of the multi-GPU slowdown; map each missing item to any symptom); (4) the recommended remediation = a single consolidated re-fix-pass (the item list, ordered), to run on rtxbox AFTER the in-flight perf-fix-pass finishes. Tag every claim to a commit hash / file:line.',
  'Return a 5-8 line executive summary: the count of genuinely-missing items, the worst offenders, and the recommended re-fix-pass item list.',
].join('\n'), { label: 'synthesize-missing', phase: 'Synthesize' })

return { ledger, checks, overview }
