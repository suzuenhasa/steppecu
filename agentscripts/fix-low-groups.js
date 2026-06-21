export const meta = {
  name: 'fix-low-groups',
  description: 'PHASE C of the big-refactor fix: the LOW findings, GROUP-BATCHED + held to the approved NAMING-STYLE-STANDARD.md, same machine as Phase B (planner -> per-task fixers accumulate -> build-repair -> default+THOROUGH golden gate + adversarial verdict -> commit group / revert). SKIP the debatable buckets (user decision): the host/device pointer-wrapper tag-type [4.7 and any "introduce DevicePtr/HostPtr wrapper" finding] and blanket const __restrict__ [Group 11] — NOT touched. DEFER the group-7 device-side leftovers (qpadm_fit_kernels.cu, block_sink.cu kernel hoists). Fixers READ + APPLY docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md (esp. §3.2 protected parity vocabulary = RENAME FORBIDDEN; §5 non-goals). REAL-AADR goldens only; CUDA-doc verification; capable-path priority + 5090 fallback; core dumps cleared per build; commit between groups.',
  phases: [
    { title: 'G3 dead-code' }, { title: 'G4 type-numeric' }, { title: 'G5 magic-num' }, { title: 'G6 naming' },
    { title: 'G7 dup-host' }, { title: 'G8 comments' }, { title: 'G9 config' }, { title: 'G10 init' },
    { title: 'G12 launch' }, { title: 'G13 errors' }, { title: 'G14 mem-alloc' }, { title: 'G15 mem-xfer' },
    { title: 'G16 RAII-own' }, { title: 'G17 RAII-life' }, { title: 'G20 perf-mem' }, { title: 'G21 perf-occ' }, { title: 'G22 perf-compute' },
  ],
}

const R = '/home/suzunik/steppe'
const FIND = R + '/docs/cleanup/bigrefactor/findings'
const STDDOC = R + '/docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === DEFAULT CTEST (GPU-vs-golden, REAL AADR) === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
const THOROUGH = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) C++20 reimplementation of ADMIXTOOLS 2. Branch phase2-fit-engine (Phase A HIGH + Phase B MED already committed). This is PHASE C: the LOW findings, group-batched, held to the APPROVED naming/style standard. Per-file findings: ' + FIND + '/<unit>.md ("## Group N" sections). Standards: docs/architecture.md (§4 layering, §12 precision/PARITY, §13 testing).',
  'THE STANDARD (MANDATORY): READ ' + STDDOC + ' and apply it to every fix. CRITICAL guardrails from it: §3.2 PROTECTED PARITY VOCABULARY = RENAME FORBIDDEN (nl, nr, r, m, nb, dof, chisq, Q/Qinv/Qf, A/B, w, loop i/j/k/b/g/p, and the AT2/cuBLAS/cuSOLVER math names) — DO NOT rename these; §5 NON-GOALS (no mass-rename of public API, no restyling already-correct names, tight-loop counters are FINE, parity-load-bearing names stay); member convention (trailing _ = private class member, plain snake_case = public POD field — do NOT add/strip _). Constants -> kPascalCase inline constexpr in config.hpp; types PascalCase; functions snake_case.',
  'SKIP (user decision — do NOT fix these): the host/device POINTER-WRAPPER tag-type findings (any [4.7] / "introduce DevicePtr/HostPtr wrapper" — a risky real refactor, opt-in later); blanket const __restrict__ (Group 11 entirely is skipped). DEFER (do NOT touch): the group-7 device-side leftovers in src/device/cuda/qpadm_fit_kernels.cu and src/device/cuda/block_sink.cu (the large kernel hoists). If a LOW finding IS one of these, mark it skipped/deferred and move on.',
  'REAL DATA ONLY (memory real-data-only-all-results): the gate is the REAL-AADR AT2 goldens golden_fit0/golden_fit1_NRBIG/golden_rot. NO synthetic data, EVER. Default ctest = GPU-vs-golden; STEPPE_THOROUGH=1 = the CpuBackend oracle vs the same goldens (REQUIRED to catch CpuBackend regressions).',
  'RELEASE / NDEBUG PITFALL: the gate build is RELEASE (-DNDEBUG, -Werror). STEPPE_ASSERT/STEPPE_DEBUG_ONLY compile out under NDEBUG -> any helper whose params are only used inside an assert has UNUSED params on Release -> -Werror=unused-parameter. Mark such params [[maybe_unused]]. (A build-repair step also patches trivial -Werror, but do not introduce them.)',
  'DOC-VERIFY: verify any CUDA/cuBLAS/cuSOLVER/C++-stdlib API claim against the CUDA 13.x docs (ToolSearch select:WebSearch,WebFetch) and cite. HARDWARE: box5090 (2x RTX 5090, P2P DISABLED) = fallback/baseline; if a fix touches P2P/multi-GPU/pro-GPU/CUDA-13+ capability, capable-path (PRO6000/P2P) is the PRIORITY, 5090 the tagged degrade — never hardcode to the 5090.',
  'SCOPE DISCIPLINE: fix ONLY the [g.x][LOW] findings of the CURRENT group (that are not skipped/deferred). LOW are mostly behavior-neutral (naming, dead code, comments, magic numbers, init) — but the golden gate STILL runs every group to catch any accidental parity break. No speculative rewrites. Skip findings already resolved by Phase A/B.',
  'BOX = box5090. ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists, RELEASE only. NOTHING builds locally. Core dumps cleared before+after every build.',
].join('\n')

const DEVLOOP = 'DEV LOOP: the tree is at a CLEAN HEAD at group start. Task-fixers ACCUMULATE edits (do NOT clean between tasks of the same group). Edit locally; the build-repair + verdict do the rsync (' + RSYNC + ') + build (' + BUILD + ') + thorough (' + THOROUGH + '). Do NOT commit (the verdict commits the whole group). Do NOT use synthetic data. Apply the NAMING-STYLE-STANDARD.'

// Phase C LOW groups. Group 11 (const __restrict__) SKIPPED entirely (debatable). 18/19 have ~0 LOW.
const GROUPS = [
  { g: 3,  name: 'Dead / commented-out code', scope: 'all' },
  { g: 4,  name: 'Type & numeric (LOW, minus the 4.7 pointer-wrapper)', scope: 'all' },
  { g: 5,  name: 'Hardcoded values / magic numbers', scope: 'all' },
  { g: 6,  name: 'Naming', scope: 'all' },
  { g: 7,  name: 'Duplication (host/core only — device-side deferred)', scope: 'all' },
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
  type: 'object', additionalProperties: false, required: ['tasks','total_low','note'],
  properties: {
    total_low: { type: 'integer', description: 'count of ACTIONABLE LOW findings for this group at HEAD (excluding skipped wrapper/restrict, deferred device-side, already-resolved)' },
    tasks: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['task','files','findings'],
      properties: { task: { type: 'string' }, files: { type: 'array', items: { type: 'string' } }, findings: { type: 'string' } } } },
    note: { type: 'string', description: 'what was skipped (wrapper/restrict)/deferred (device-side)/already-resolved' },
  },
}
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['group','pass','diff_real','standard_followed','build_clean','goldens_green','thorough_green','no_synthetic','no_protected_renames','regression','commit_hash','note'],
  properties: {
    group: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if: the diff genuinely fixes this group LOW findings per the NAMING-STYLE-STANDARD + Release build clean + default ctest 39/39 green vs real goldens + STEPPE_THOROUGH qpadm green + NO regression + NO synthetic + NO rename of §3.2 protected parity vocabulary + skipped buckets untouched' },
    diff_real: { type: 'boolean' },
    standard_followed: { type: 'boolean', description: 'the changes conform to NAMING-STYLE-STANDARD.md (kPascalCase constants, member _ convention, etc.)' },
    build_clean: { type: 'boolean' }, goldens_green: { type: 'boolean' }, thorough_green: { type: 'boolean' },
    no_synthetic: { type: 'boolean' },
    no_protected_renames: { type: 'boolean', description: 'NO §3.2 protected parity vocabulary (nl/nr/r/m/nb/Q/A/B/dof/chisq/loop counters/AT2 names) was renamed; the host/device wrapper + const __restrict__ buckets were NOT touched' },
    regression: { type: 'boolean' }, commit_hash: { type: 'string' }, note: { type: 'string' },
  },
}

const ledger = []
for (const G of GROUPS) {
  const title = 'G' + G.g + ' ' + G.name.split(' ')[0].toLowerCase()
  phase(title)

  const plan = await tryAgent([
    'You are planning the LOW fixes for GROUP ' + G.g + ' (' + G.name + ') of the steppe big-refactor, held to the NAMING-STYLE-STANDARD. READ-ONLY: read ' + STDDOC + ' AND the "## Group ' + G.g + '" section of every ' + FIND + '/*.md; collect the [' + G.g + '.x][LOW] findings; group by task id. EXCLUDE: (a) the host/device POINTER-WRAPPER findings (4.7 / "introduce DevicePtr/HostPtr"); (b) anything in src/device/cuda/qpadm_fit_kernels.cu or src/device/cuda/block_sink.cu (deferred group-7 device-side); (c) findings already resolved by Phase A/B (verify against HEAD); (d) any rename that would touch §3.2 protected parity vocabulary.', STD, '',
    'Return the structured plan: per task with actionable LOW, the task id + files (absolute) + concrete findings (file:line + fix, aligned to the standard). total_low = actionable count. note what you skipped/deferred. If ZERO actionable, tasks=[].',
  ].join('\n'), { schema: PLAN_SCHEMA, label: 'plan:g' + G.g, phase: title })

  if (!plan || !Array.isArray(plan.tasks) || plan.tasks.length === 0) {
    ledger.push({ group: 'G' + G.g, pass: true, commit_hash: '', note: 'no actionable LOW — skipped: ' + (plan ? plan.note : 'planner died') })
    log('=== G' + G.g + ': no actionable LOW — skip'); continue
  }
  log('G' + G.g + ' (' + G.name + '): ' + plan.total_low + ' actionable LOW across ' + plan.tasks.length + ' tasks')

  await tryAgent(['Prepare the tree for GROUP ' + G.g + ' LOW fixes: run EXACTLY ' + CLEAN + ' and confirm `git status --short` empty. Return "clean" or the dirty files.', STD].join('\n'), { label: 'clean:g' + G.g, phase: title })

  const taskReports = []
  for (const t of plan.tasks) {
    const tr = await tryAgent([
      'You are a senior CUDA/C++ engineer applying the LOW fixes for TASK ' + t.task + ' (group ' + G.g + ' — ' + G.name + ') of steppe, HELD TO THE NAMING-STYLE-STANDARD. Do NOT clean (a prior task in this group may have edits), do NOT build, do NOT commit.', STD, '', DEVLOOP, '',
      'FIRST read ' + STDDOC + ' (apply it). TASK ' + t.task + ' — fix these LOW findings:\n' + t.findings, '', 'FILES: ' + (t.files || []).join(', '),
      '', 'Apply ONLY these LOW findings per the standard. RESPECT §3.2 (do NOT rename protected parity vocabulary) + §5 non-goals. Do NOT touch the skipped pointer-wrapper / const __restrict__ buckets or the deferred device-side kernels. Behavior-preserving. VERIFY any CUDA/cuBLAS API claim vs the CUDA 13.x docs (cite). Edit locally; do NOT build/commit/clean. Report every file:line changed + what changed + which standard rule it satisfies. Skip any finding already resolved at HEAD.',
    ].join('\n'), { label: 'fix:g' + G.g + ':' + t.task, phase: title })
    taskReports.push(t.task + ': ' + (tr ? String(tr).slice(0, 350) : 'fixer died'))
  }

  // build-repair (patch trivial -Werror before the verdict, no full revert over a one-liner)
  await tryAgent([
    'You are the BUILD-REPAIR step for GROUP ' + G.g + ' of steppe. The per-task fixers accumulated edits (do NOT clean/revert). Reach a CLEAN Release build, patching only TRIVIAL warnings.', STD, '',
    'DO: rsync (' + RSYNC + ') then build (' + BUILD + '). If it FAILS on a TRIVIAL -Werror (unused-parameter/unused-variable from a touched helper, esp. STEPPE_ASSERT-only params on NDEBUG) FIX MINIMALLY ([[maybe_unused]] / drop the dead param) and rebuild, LOOP up to 4x until clean. Do NOT change fix logic, do NOT revert, do NOT touch out-of-scope code. If it fails for a NON-trivial reason, STOP and report (do not paper over). Report final build status + every trivial patch applied.',
  ].join('\n'), { label: 'repair:g' + G.g, phase: title })

  const verdict = await tryAgent([
    'You are the INDEPENDENT VERDICT for GROUP ' + G.g + ' (' + G.name + ') LOW fixes of steppe (adversarial). The fixers reported:\n<<<\n' + taskReports.join('\n---\n') + '\n>>>', STD, '',
    'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — confirm the edits genuinely fix this group LOW findings PER THE STANDARD; confirm NO §3.2 protected parity vocabulary was renamed, NO public-API mass-rename, the member-underscore convention respected, and the SKIPPED buckets (pointer-wrapper, const __restrict__) + DEFERRED device-side kernels were NOT touched. (2) RE-RUN: ' + BUILD + ' ; then ' + THOROUGH + ' . (3) PASS only if ALL: diff real + standard-conformant; Release build clean; default ctest 39/39 green vs real goldens; STEPPE_THOROUGH qpadm green (oracle bit-identical); NO regression; NO synthetic; NO protected renames. ',
    'ON PASS: cd ' + R + ' && git add ONLY the changed source+test+doc files for THIS group (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (group ' + G.g + ' LOW per the naming standard — what+why, gate result) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.',
    'ON FAIL: ' + CLEAN + ' (leave repo green) and report exactly what broke + which task. Return the structured verdict.',
  ].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:g' + G.g, phase: title })

  if (verdict === null) { ledger.push({ group: 'G' + G.g, pass: false, note: 'verdict died' }); log('--- G' + G.g + ' verdict died'); continue }
  ledger.push(verdict)
  if (verdict.pass) log('+++ G' + G.g + ' committed ' + verdict.commit_hash + ' — ' + verdict.note)
  else log('--- G' + G.g + ' FAILED (' + verdict.note + ') — reverted; continuing')
}

const committed = ledger.filter(x => x.pass && x.commit_hash).length
const skipped = ledger.filter(x => x.pass && !x.commit_hash).length
log('PHASE C done: ' + committed + ' groups committed, ' + skipped + ' skipped (no actionable LOW), ' + (ledger.length - committed - skipped) + ' failed')
return { ledger }
