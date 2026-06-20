export const meta = {
  name: 'fix-med-groups',
  description: 'PHASE B of the big-refactor fix: the ~176 MED findings, GROUP-BATCHED (one build+test+verdict+commit per group; ~14 groups, sequential — groups touch overlapping files so they cannot race). Per group: clean HEAD -> a planner reads THIS group\'s [g.x][MED] findings across docs/cleanup/bigrefactor/findings/*.md vs CURRENT HEAD (skip anything Phase A / a prior group already resolved) -> ONE fixer agent PER TASK (g.1, g.2, ...) in SEQUENCE accumulating edits (no clean between, no race) -> build + the REAL-AADR golden ctest (default GPU-vs-golden + STEPPE_THOROUGH CpuBackend oracle) + an adversarial verdict -> commit the whole group GREEN / revert on fail (skip-and-continue). REAL DATA ONLY (the goldens golden_fit0/golden_fit1/golden_rot); NO synthetic data. Fixers VERIFY CUDA/cuBLAS/cuSOLVER API claims against the CUDA 13.x docs. Capable-path (PRO6000/P2P/CUDA13+) priority + 5090 tagged fallback. Core dumps cleared before+after every build. commit between groups.',
  phases: [
    { title: 'G5 magic-numbers' }, { title: 'G7 duplication' }, { title: 'G8 comments' }, { title: 'G9 constants' },
    { title: 'G10 init' }, { title: 'G12 launch-config' }, { title: 'G13 error-handling' }, { title: 'G14 mem-alloc' },
    { title: 'G15 mem-transfer' }, { title: 'G16 RAII-ownership' }, { title: 'G17 RAII-lifetime' },
    { title: 'G20 perf-memaccess' }, { title: 'G21 perf-occupancy' }, { title: 'G22 perf-compute' },
  ],
}

const R = '/home/suzunik/steppe'
const FIND = R + '/docs/cleanup/bigrefactor/findings'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === DEFAULT CTEST (GPU-vs-golden, REAL AADR) === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
const THOROUGH = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2. Branch phase2-fit-engine (3 HIGH fixes already landed: 9dbc610 block_sink fail-fast, 3beff6d kQpMax single-source incl. the new src/core/qpadm/qpadm_bounds.hpp, ed6cc44 cpu_backend als_ridge_solve). This is PHASE B: the MED findings, group-batched. The per-file findings are docs/cleanup/bigrefactor/findings/<unit>.md (each has a "## Group N" section). Standards: docs/architecture.md (§2 DRY/RAII/fail-fast, §4 layering [io leaf; CUDA PRIVATE to steppe_device; core CUDA-free], §8 DRY single-home, §9 config, §12 precision/PARITY, §13 testing).',
  'REAL DATA ONLY (memory real-data-only-all-results): the gate is the REAL-AADR AT2 goldens — golden_fit0 (9-pop) / golden_fit1_NRBIG (nr=39) / golden_rot (84-model) under tests/reference/goldens/at2/. NO synthetic data, EVER. The default ctest validates the GPU path vs the goldens; STEPPE_THOROUGH=1 additionally runs the CpuBackend oracle vs the same goldens (REQUIRED to catch CpuBackend regressions, since the default ctest is GPU-only).',
  'DOC-VERIFY (memory refactor-process-rules): VERIFY any CUDA/cuBLAS/cuSOLVER/C++-stdlib API-behavior claim against the official CUDA 13.x docs (ToolSearch select:WebSearch,WebFetch) and cite it. Do not fix API semantics from memory.',
  'HARDWARE (memory refactor-process-rules + steppebox5090): box5090 = 2x RTX 5090 (consumer, P2P DISABLED) = the FALLBACK/baseline, keep it green. If a fix touches P2P / multi-GPU / pro-GPU / CUDA-13+ capability (e.g. p2p_combine.cu, cudaMemcpyPeer, multi-GPU device-correct free in RAII groups 16/17, memory-pool/async-alloc in group 14), treat the CAPABLE path (RTX PRO 6000 / stock-driver P2P / newest CUDA-13 feature) as the PRIORITY and the 5090 limit as the explicitly-tagged graceful-degrade — NEVER hardcode to the 5090.',
  'SCOPE DISCIPLINE: fix ONLY the [g.x][MED] findings of the CURRENT group (and any [g.x][LOW] that folds in for free WITHOUT extra risk). Do NOT touch other groups\' findings, do NOT do speculative rewrites. MED findings that a prior phase already resolved (e.g. the [9.2][MED] kQpMax cross-ref resolved by H2, the [13.1][MED] made moot by H1, the [7.2]/[7.4] folded by H3) — SKIP them, note "already resolved". Behavior-preserving where possible; the golden gate catches any parity break.',
  'BOX = box5090. ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists, RELEASE only. NOTHING builds locally. Core dumps cleared before+after every build.',
].join('\n')

const DEVLOOP = 'DEV LOOP (nothing builds locally): the tree is at a CLEAN HEAD at group start. Task-fixers ACCUMULATE edits on the working tree (do NOT clean between tasks of the same group). Edit locally; the group verdict does the rsync (' + RSYNC + ') + build (' + BUILD + ') + thorough (' + THOROUGH + '). Do NOT commit (the verdict commits the whole group). Do NOT use synthetic data.'

const GROUPS = [
  { g: 5,  name: 'Hardcoded values / magic numbers', scope: 'all' },
  { g: 7,  name: 'Duplication', scope: 'all' },
  { g: 8,  name: 'Comments', scope: 'all' },
  { g: 9,  name: 'Constants & configuration', scope: 'all' },
  { g: 10, name: 'Initialization', scope: 'all' },
  { g: 12, name: 'Launch config & indexing', scope: 'kernel' },
  { g: 13, name: 'Error handling', scope: 'device' },
  { g: 14, name: 'Memory: allocation & lifetime', scope: 'device' },
  { g: 15, name: 'Memory: transfers', scope: 'device' },
  { g: 16, name: 'RAII: ownership & wrapper hygiene', scope: 'device' },
  { g: 17, name: 'RAII: lifetime & deleter pitfalls', scope: 'device' },
  { g: 20, name: 'Performance: memory access', scope: 'kernel' },
  { g: 21, name: 'Performance: occupancy & registers', scope: 'kernel' },
  { g: 22, name: 'Performance: compute & launch', scope: 'kernel' },
]

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['tasks','total_med','note'],
  properties: {
    total_med: { type: 'integer', description: 'count of ACTIONABLE MED findings for this group at current HEAD (excluding already-resolved / cross-ref)' },
    tasks: { type: 'array', description: 'one entry per task id that has actionable MED findings, in ascending order', items: {
      type: 'object', additionalProperties: false, required: ['task','files','findings'],
      properties: {
        task: { type: 'string', description: 'the task id e.g. 14.2' },
        files: { type: 'array', items: { type: 'string' }, description: 'absolute source paths this task edits' },
        findings: { type: 'string', description: 'the concrete MED findings for this task (file:line + what to fix), concatenated' },
      } } },
    note: { type: 'string', description: 'anything skipped (already-resolved) + risk notes' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['group','pass','diff_real','build_clean','goldens_green','thorough_green','no_synthetic','regression','commit_hash','note'],
  properties: {
    group: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if: the diff genuinely fixes this group\'s MED findings + Release build clean (warnings-as-errors) + default ctest 39/39 green vs the real goldens + STEPPE_THOROUGH qpadm green (CpuBackend oracle bit-identical) + NO regression + NO synthetic data' },
    diff_real: { type: 'boolean' }, build_clean: { type: 'boolean' },
    goldens_green: { type: 'boolean', description: 'default ctest (GPU-vs-real-AADR-golden) all green' },
    thorough_green: { type: 'boolean', description: 'STEPPE_THOROUGH qpadm (CpuBackend oracle vs golden) green' },
    no_synthetic: { type: 'boolean' }, regression: { type: 'boolean' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed (which tasks/files) + the gate results; for FAIL exactly what broke' },
  },
}

const ledger = []
for (const G of GROUPS) {
  const title = 'G' + G.g + ' ' + G.name.split(' ')[0].toLowerCase()
  phase(title)

  // 1) PLAN (read-only): extract this group's actionable MED findings vs current HEAD
  const plan = await tryAgent([
    'You are planning the MED fixes for GROUP ' + G.g + ' (' + G.name + ') of the steppe big-refactor. READ-ONLY: read the "## Group ' + G.g + '" section of every ' + FIND + '/*.md, collect the [' + G.g + '.x][MED] findings, and group them by task id. EXCLUDE findings already resolved by Phase A (kQpMax/H2, block_sink/H1, opt_A-opt_B/H3) or that are pure cross-refs — verify against current HEAD by reading the cited code. Also opportunistically note [' + G.g + '.x][LOW] that fold in for free.', STD, '',
    'Return the structured plan: for each task with actionable MED findings, the task id, the files it edits (absolute paths), and the concrete findings (file:line + fix). total_med = actionable count. If ZERO actionable (all resolved), return tasks=[] and say so in note.',
  ].join('\n'), { schema: PLAN_SCHEMA, label: 'plan:g' + G.g, phase: title })

  if (!plan || !Array.isArray(plan.tasks) || plan.tasks.length === 0) {
    ledger.push({ group: 'G' + G.g, pass: true, commit_hash: '', note: 'no actionable MED (all resolved/none) — skipped: ' + (plan ? plan.note : 'planner died') })
    log('=== G' + G.g + ': no actionable MED — skip')
    continue
  }
  log('G' + G.g + ' (' + G.name + '): ' + plan.total_med + ' actionable MED across ' + plan.tasks.length + ' tasks')

  // 2) clean HEAD, then SEQUENTIAL per-task fixers (accumulate edits, no clean between)
  await tryAgent([
    'You are preparing the working tree for GROUP ' + G.g + ' fixes. Run EXACTLY: ' + CLEAN + ' — to ensure a clean tree at HEAD. Confirm `git status --short` is empty. Return "clean" or the dirty files.', STD,
  ].join('\n'), { label: 'clean:g' + G.g, phase: title })

  const taskReports = []
  for (const t of plan.tasks) {
    const tr = await tryAgent([
      'You are a senior CUDA/C++ engineer applying the MED fixes for TASK ' + t.task + ' (group ' + G.g + ' — ' + G.name + ') of steppe. Do NOT clean the tree (a prior task in this group may have edits) and do NOT commit (the group verdict commits). Do NOT build (the group verdict builds once at the end).', STD, '', DEVLOOP, '',
      'TASK ' + t.task + ' — fix these MED findings:\n' + t.findings, '', 'FILES: ' + (t.files || []).join(', '),
      '', 'Apply ONLY these findings (+ trivially-folding LOW of the same task). Behavior-preserving where possible. VERIFY any CUDA/cuBLAS API claim against the CUDA 13.x docs (cite). Update stale doc-comments the fix causes. Edit locally; do NOT build, do NOT commit, do NOT clean. Report every file+line changed and what changed (so the group verdict can audit). If a finding is already resolved at HEAD, skip it and say so.',
    ].join('\n'), { label: 'fix:g' + G.g + ':' + t.task, phase: title })
    taskReports.push(t.task + ': ' + (tr ? String(tr).slice(0, 400) : 'fixer died'))
  }

  // 3) group verdict: build + both gates + commit/revert
  const verdict = await tryAgent([
    'You are the INDEPENDENT VERDICT for GROUP ' + G.g + ' (' + G.name + ') MED fixes of steppe (you did NOT write them — be adversarial). The per-task fixers reported:\n<<<\n' + taskReports.join('\n---\n') + '\n>>>', STD, '',
    'THE GROUP PLAN:\n' + JSON.stringify(plan.tasks.map(t => ({ task: t.task, files: t.files })), null, 0),
    '', 'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — confirm the accumulated edits GENUINELY fix this group\'s MED findings (not sham/comment-only, not out-of-scope changes to other groups). (2) RE-RUN the gates yourself: ' + BUILD + ' ; then ' + THOROUGH + ' . (3) PASS only if ALL: diff real + in-scope; Release build clean (warnings-as-errors); default ctest 39/39 green vs the real-AADR goldens; STEPPE_THOROUGH qpadm green (CpuBackend oracle bit-identical to golden_fit0/fit1); NO regression; NO synthetic data. ',
    'ON PASS: cd ' + R + ' && git add ONLY the changed/new source+test+doc files for THIS group (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (group ' + G.g + ' MED fixes — what+why, the real-golden gate result, any CUDA-doc citation) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.',
    'ON FAIL: ' + CLEAN + ' (leave the repo green) and report exactly what broke + whether the verify can localize which task caused it (so a re-run can be finer-grained). Return the structured verdict.',
  ].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:g' + G.g, phase: title })

  if (verdict === null) { ledger.push({ group: 'G' + G.g, pass: false, note: 'verdict died — SKIPPED, tree may be dirty' }); log('--- G' + G.g + ' verdict died'); continue }
  ledger.push(verdict)
  if (verdict.pass) log('+++ G' + G.g + ' committed ' + verdict.commit_hash + ' — ' + verdict.note)
  else log('--- G' + G.g + ' FAILED (' + verdict.note + ') — reverted; continuing')
}

const passed = ledger.filter(x => x.pass && x.commit_hash).length
const skipped = ledger.filter(x => x.pass && !x.commit_hash).length
log('PHASE B done: ' + passed + ' groups committed, ' + skipped + ' skipped (no actionable MED), ' + (ledger.length - passed - skipped) + ' failed')
return { ledger }
