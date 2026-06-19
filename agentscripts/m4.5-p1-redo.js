export const meta = {
  name: 'm4.5-p1-redo',
  description: 'REDO P1 (the ~1.4s combine zero-fill lever) — the ONE genuine gap from the forensic audit: the original P1 fixer was a SHAM (empty commit + fabricated report; verdict caught it pass=false; never redone). Single item, fixer + independent verdict, RELEASE build on rtxbox, SHAM-HARDENED: the verdict git-diff-verifies a real non-empty diff AND requires the measured G2@768 bench to actually DROP (a sham leaves it unchanged). HALT-on-fail (a 2nd sham => stop, human takes over). Parity memcmp BIT-IDENTICAL gate.',
  phases: [ { title: 'P1 combine no-zero-fill' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh rtxbox'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ rtxbox:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -20 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -40'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ./build-rel/bin/test_f2_multigpu_parity 2>&1 | tail -45'"
const BENCH = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/bench_f2_multigpu /workspace/data/aadr 200 400 768 2>&1 | grep -vE \"P2P combine unavailable\"'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell reimpl of ADMIXTOOLS 2 f-statistics, branch m4.5-multigpu, HEAD has P0/P2/P3/P4 of the perf-fix-pass landed. The multi-GPU is still 0.70x at P=768 because P1 — the #1 measured lever — was NEVER landed: the original P1 fixer was a SHAM (it fabricated a build/parity/bench report and committed an EMPTY diff; the independent verdict correctly caught it). THIS redo actually lands it. Read docs/cleanup/m4.5/perf-discovery.md §3 (P1/W9) + docs/cleanup/missing-fixes.md §1 (the P1 row, exact sites).',
  'BOX = rtxbox (2x RTX PRO 6000 Blackwell, sm_120, CUDA 13, 96 GB, REAL P2P). ssh rtxbox; nvcc not on PATH -> ' + PATHENV + ' . Data /workspace/data/aadr/{raw, derived_acc, derived_full P=768}. RELEASE build only (perf is meaningless in debug). NOTHING builds locally.',
  'THE FIX (exact): the combine ZERO-FILLS the full [P^2*n_block] f2+vpair output via assign(total,0.0) then a wholesale overwrite — ~1440 ms/run of wasted zeroing. (a) Add a CUDA-FREE default-init allocator include/steppe/default_init_allocator.hpp (an Allocator<T> whose construct() default-initializes — i.e. no-ops for trivial T — so vector::resize allocates WITHOUT value-init/zeroing). (b) In include/steppe/fstats.hpp make F2BlockTensor.f2 and .vpair use it (e.g. using F2Storage = std::vector<double, steppe::default_init_allocator<double>>; both fields F2Storage). (c) At the DEVICE-TIER combine sites swap assign(total,0.0) -> resize(total): cuda_backend.cu:255-256 AND p2p_combine.cu:180-181 (the subsequent wholesale D2H/place overwrites ALL `total` elements, so skipping the zero is bit-identical). (d) DO NOT touch the HOST-STAGED tier f2_combine.cpp:64-65 — the -0.0/unowned-slab caveat (perf-discovery §3) makes it yes-if-careful only; leave it assign(). Fix any consumer type-mismatch from the F2Storage change (the build will surface them; there are no std::vector<double>& takers of f2/vpair).',
  'PARITY LAW: resize() over the default-init allocator allocates uninitialized; the combine then overwrites EVERY element, so the result is BIT-IDENTICAL to the old assign(0.0)+overwrite. test_f2_multigpu_parity (memcmp, EmuFp64{40} G==2 host-staged AND P2P == single-GPU, derived_acc + derived_full) MUST stay bit-identical — if ANY element is left uninitialized (not fully overwritten), parity FAILS and that is the guard. No combine-order / NCCL / precision change.',
  'ANTI-SHAM (the prior attempt faked this): you MUST produce a REAL non-empty git diff. After editing, SHOW `cd ' + R + ' && git --no-pager diff --stat` in your report. Build + ctest + parity + bench for REAL on rtxbox and paste the actual outputs. Do NOT fabricate numbers. The expected payoff is the G2 bench at P=768 dropping from ~3322 ms toward ~1900-2100 ms.',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean tree at HEAD (' + CLEAN + '); edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); parity (' + PARITY + '); bench (' + BENCH + '). Do NOT commit (the verdict commits).'

const fixPrompt =
  'You are a senior CUDA/C++ engineer. REDO P1 — actually land it this time. The previous P1 fixer was a SHAM (empty commit, fabricated report); the user is (rightly) watching for that.\n\n' + STD + '\n\n' + DEVLOOP + '\n\nReturn a thorough, HONEST report: (1) git --no-pager diff --stat (must be NON-EMPTY); (2) every file changed + what; (3) the FULL Release build result; (4) ctest summary; (5) the parity output (the EmuFp64 G==2 == single-GPU bit-identical lines); (6) the bench table (G1/G2/speedup at 200/400/768) — the REAL measured G2@768 (expected to drop ~1.4s from ~3322ms). If you cannot reach a clean Release build + bit-identical parity + a measurably-faster G2@768, report exactly what blocked it — do NOT fabricate success.'

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass', 'diff_is_real', 'parity_bit_identical', 'g2_768_ms', 'g2_768_dropped', 'commit_hash', 'note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: the git diff is REAL+non-empty (NOT a sham) implementing the default-init allocator + the device-tier resize swap (f2_combine.cpp untouched) + Release build clean + ctest green + parity memcmp BIT-IDENTICAL + the measured G2@768 actually dropped vs ~3322ms baseline' },
    diff_is_real: { type: 'boolean', description: 'true if git diff --stat is non-empty AND the default_init_allocator + the 2 device-tier resize swaps are genuinely present (you re-ran git diff yourself; not trusting the fixer report)' },
    parity_bit_identical: { type: 'boolean' },
    g2_768_ms: { type: 'number', description: 'the YOU-measured bench G2 wall-clock at P=768 (ms), re-run yourself' },
    g2_768_dropped: { type: 'boolean', description: 'true if g2_768_ms dropped measurably (>~500ms) vs the ~3322ms pre-P1 baseline — a sham (empty commit) leaves it unchanged, so this is the sham-proof gate' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the before/after G2@768 + speedup; for FAIL, exactly why (sham? parity? no drop?)' },
  },
}

const verdictPrompt = (fixReport) =>
  'You are the INDEPENDENT VERDICT for the P1 redo (be MAXIMALLY adversarial — the prior P1 fixer FABRICATED its entire report and committed an empty diff). The fixer reported:\n<<<\n' + fixReport + '\n>>>\n\n' + STD + '\n\nDO, yourself (do NOT trust the report): (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM it is NON-EMPTY and genuinely contains: the new include/steppe/default_init_allocator.hpp, F2BlockTensor.f2/.vpair switched to the default-init storage, and assign(total,0.0)->resize(total) at cuda_backend.cu (~:255-256) AND p2p_combine.cu (~:180-181), with f2_combine.cpp UNCHANGED. If the diff is empty or missing these, it is a SHAM => pass=false, diff_is_real=false. (2) RE-RUN the box yourself: ' + BUILD + ' ; ' + PARITY + ' ; ' + BENCH + ' . (3) PASS only if ALL: diff real; Release build clean (warnings-as-errors); ctest green (no regression; debug death-tests inactive in Release is OK); parity test_f2_multigpu_parity memcmp BIT-IDENTICAL (EmuFp64 G==2 host-staged AND P2P == single-GPU, both datasets); AND the YOU-measured G2@768 dropped measurably (>~500ms) below the ~3322ms baseline (the sham-proof gate). \n\nON PASS: cd ' + R + ' && git add ONLY include/steppe/default_init_allocator.hpp include/steppe/fstats.hpp src/device/cuda/cuda_backend.cu src/device/cuda/p2p_combine.cu (+ any genuine consumer fixups) — NEVER git add dot; commit with a ROADMAP §6 message (the measured before/after G2@768) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly why (sham / parity-break / no-drop).\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retrying once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

phase('P1 combine no-zero-fill')
log('=== P1: fixing (redo — sham-hardened) ===')
const fix = await tryAgent(fixPrompt, { label: 'fix:P1', phase: 'P1 combine no-zero-fill' })
let verdict = null
if (fix !== null) {
  verdict = await tryAgent(verdictPrompt(fix), { schema: VERDICT_SCHEMA, label: 'verdict:P1', phase: 'P1 combine no-zero-fill' })
}
if (verdict && verdict.pass) log('+++ P1 LANDED ' + verdict.commit_hash + ' — G2@768 ' + verdict.g2_768_ms + 'ms (dropped=' + verdict.g2_768_dropped + ', parity=' + verdict.parity_bit_identical + ')')
else log('--- P1 redo FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over directly')
return { fix, verdict }
