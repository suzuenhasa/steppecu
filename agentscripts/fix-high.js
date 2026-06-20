export const meta = {
  name: 'fix-high',
  description: 'PHASE A of the big-refactor fix: the 3 distinct HIGH findings, each as its OWN fix->build->test->adversarial-verdict->commit unit (strictly sequential; independent fixer + verdict per item; commit-green / revert on fail; skip-and-continue; retry-once on transient API error). H1 [13.3] block_sink HostRam swallows a failed cudaEventSynchronize then memcpys an undrained slot -> SILENT f2 corruption (make it fail-fast like DiskSink). H2 [5.3] kQpMax* small-path bound is a bare-literal TRIPLE-copy -> drift / device buffer-overflow (single CUDA-free source of truth, reference from all 3 sites). H3 [7.1] cpu_backend opt_A/opt_B 58-line parity-load-bearing duplication (extract a shared als_ridge_solve). REAL DATA ONLY — the gate is the real-AADR AT2 goldens (golden_fit0/golden_fit1/golden_rot); NO synthetic data anywhere. Fixers MUST verify CUDA/cuBLAS/cuSOLVER API claims against the official CUDA 13.x docs. Capable-path priority (PRO6000/P2P/CUDA13+) with the 5090 as the tagged fallback where relevant. Core dumps cleared before+after every build.',
  phases: [ { title: 'H1 [13.3] block_sink' }, { title: 'H2 [5.3] kQpMax' }, { title: 'H3 [7.1] opt_A/opt_B' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
// build + the FAST default ctest (GPU-vs-AT2-golden, real data). Core dumps cleared before AND after.
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === DEFAULT CTEST (GPU-vs-golden, REAL AADR) === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
// the THOROUGH qpadm gate (runs the CpuBackend oracle + asserts == golden) — REQUIRED for H3 (CpuBackend change is invisible to the default GPU-only ctest)
const THOROUGH = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -45; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) reimplementation of ADMIXTOOLS 2. Branch phase2-fit-engine. The big-refactor REVIEW is done (docs/cleanup/bigrefactor/findings/<unit>.md per file). This is PHASE A: the 3 distinct HIGH findings, each its own fix->test->verify->commit. Standards: docs/architecture.md (§2 DRY/RAII/fail-fast, §4 layering [io leaf; CUDA PRIVATE to steppe_device; core CUDA-free], §12 precision/PARITY, §13 testing).',
  'REAL DATA ONLY (memory real-data-only-all-results): the acceptance gate is the REAL-AADR AT2 goldens — golden_fit0 (9-pop), golden_fit1_NRBIG (nr=39), golden_rot (84-model rotation) under tests/reference/goldens/at2/. NO synthetic data anywhere, for ANY check. The default ctest validates the GPU path vs these goldens; STEPPE_THOROUGH=1 additionally runs the CpuBackend oracle vs the same goldens.',
  'DOC-VERIFY (memory refactor-process-rules): for ANY CUDA / cuBLAS / cuSOLVER / C++-stdlib API-behavior claim your fix relies on, VERIFY it against the official CUDA 13.x docs (load web tools via ToolSearch select:WebSearch,WebFetch if needed) and cite it in your report. Do not fix API semantics from memory.',
  'HARDWARE (memory refactor-process-rules + steppebox5090): box5090 = 2x RTX 5090 (consumer, P2P DISABLED) = the FALLBACK/baseline, always keep it green. If a fix touches P2P / multi-GPU / pro-GPU / CUDA-13+ capability, treat the CAPABLE path (RTX PRO 6000 / stock-driver P2P / newest feature) as the PRIORITY and the 5090 limit as the explicitly-tagged graceful-degrade — never hardcode to the 5090. (None of the 3 HIGH below touch P2P, but keep it in mind.)',
  'BOX = box5090. ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists, RELEASE only. NOTHING builds locally. Core dumps are cleared before+after every build (the BUILD cmd does it).',
].join('\n')

const DEVLOOP = 'DEV LOOP (nothing builds locally): the tree is at a CLEAN HEAD when you start (the orchestrator cleaned it). Edit locally; (1) rsync: ' + RSYNC + ' ; (2) build + the real-AADR golden ctest: ' + BUILD + ' . Build MUST be clean (warnings-as-errors) and ALL ctest green (no regression). Do NOT commit (the verdict agent commits). Do NOT use synthetic data.'

const ITEMS = [
  { id: 'H1', title: 'H1 [13.3] block_sink', files: 'src/device/cuda/block_sink.cu, src/device/cuda/block_sink.cuh',
    gate: BUILD, gate_name: 'default ctest (the streaming-spill tests exercise block_sink; the fix is error-path only so the happy path stays byte-identical + goldens green)',
    fix: '[13.3][HIGH] block_sink.cu:91-98 — HostRamSink::writer_loop checks cudaEventSynchronize(s.done) but on FAILURE only STEPPE_LOG_WARNs and then UNCONDITIONALLY memcpys the (possibly undrained/stale) pinned slot into host_.f2/host_.vpair -> SILENT f2 corruption (a §12 parity violation that surfaces as a wrong statistic, not a thrown error). DiskSink::writer_loop (lines ~246-249) treats the SAME failure as fatal (records writer_failed_, re-throws at finish()). FIX: make HostRam fail-fast IDENTICALLY to Disk — on a non-success event sync, record the writer-failed flag and re-throw at finish(); do NOT memcpy the undrained slot. This also makes the related [13.1][MED] sticky-error-drain moot (the finding says so). VERIFY cudaEventSynchronize return/error semantics against the CUDA 13 docs (what a non-cudaSuccess return means; sticky vs non-sticky). Prefer routing BOTH sinks through one shared fail-fast event-wait helper if clean (the [13.3][LOW] note), but the minimum is HostRam fail-fast.' },
  { id: 'H2', title: 'H2 [5.3] kQpMax', files: 'a NEW CUDA-free shared header (e.g. src/core/qpadm/qpadm_bounds.hpp OR include/steppe/config.hpp), src/core/qpadm/model_search.cpp, src/device/cuda/cuda_backend.cu, src/device/cuda/qpadm_fit_kernels.cu (+ .cuh)',
    gate: BUILD, gate_name: 'default ctest (qpadm_parity + qpadm_rotation exercise the small-path dispatch vs the goldens — catches any mis-routing; behavior-neutral so goldens bit-identical)',
    fix: '[5.3][HIGH] the small-path envelope nl<=5 && nr<=10 && r<=4 is a bare-literal TRIPLE copy: (a) model_search.cpp:73 (model_in_small_path host gate), (b) cuda_backend.cu:1490 (CudaBackend::model_fits_small_path), (c) the authoritative NAMED kQpMaxNl=5/kQpMaxNr=10/kQpMaxR=4 in qpadm_fit_kernels.cu:41-43 that SIZE the kernel per-thread local arrays. DRIFT is a correctness bug: widening the host gate without widening the kernel bounds admits oversized models -> DEVICE BUFFER OVERFLOW/UB. FIX: define kQpMaxNl/Nr/R (and any derived kQpMaxM/kQpMaxT) EXACTLY ONCE in a CUDA-FREE shared header (so both the host core gate and the .cu kernel TU can include it — keep core CUDA-free per §4; a small src/core/qpadm/qpadm_bounds.hpp is clean), and have ALL THREE sites reference the single source. Behavior-NEUTRAL: the values are unchanged (5/10/4), so the af6a8c2 + NRBIG + rotation goldens MUST stay bit-identical. This also resolves the [9.2][MED] cross-ref. Confirm the kernel TU still compiles with the constants moved (constexpr int, usable in array bounds).' },
  { id: 'H3', title: 'H3 [7.1] opt_A/opt_B', files: 'src/device/cpu/cpu_backend.cpp',
    gate: THOROUGH, gate_name: 'STEPPE_THOROUGH=1 ctest -R qpadm — REQUIRED: this changes CpuBackend code, which the DEFAULT (GPU-only) ctest does NOT exercise; the thorough run executes the CpuBackend oracle and asserts it == the real-AADR goldens, catching any bit-drift',
    fix: '[7.1][HIGH] cpu_backend.cpp:787-845 (opt_A) vs 850-904 (opt_B) are near-identical 58-line copies of the AT2 GLS-ridge solve (build xvec=c(t(xmat)); W=qinv·linop; coeffs=linopᵀ·W; rhs=xvecᵀ·W; ridge the diagonal tr+fudge*tr; core::solve; reshape row-major into the factor). They differ ONLY by (a) the Kronecker operator lambda (I⊗B vs A⊗I) + its index arithmetic, (b) t=nl*r vs r*nr, (c) the final reshape. PARITY-LOAD-BEARING: the CpuBackend is the native-FP64 ORACLE — its numbers MUST stay BIT-IDENTICAL or parity breaks. FIX: extract a shared `als_ridge_solve(linop_lambda, m, t, xvec, qinv, fudge) -> solved vector` (the W/coeffs/rhs/ridge/solve core), leaving only the operator lambda + reshape per caller. While there, fold the byte-identical xvec build [7.2][LOW] and the thrice-repeated ridge-on-diagonal idiom [7.4][LOW] (jackknife_cov @480-484, opt_A @832-834, opt_B @891-893) into helpers IF it does not change the arithmetic/op-order. The extraction MUST preserve the exact FP op order (same accumulation order, same ridge, same solve) so the oracle is bit-identical.' },
]

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item','pass','diff_real','build_clean','goldens_green','no_synthetic','regression','commit_hash','note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if: the diff genuinely fixes the finding (not a sham/comment-only) + Release build clean (warnings-as-errors) + the required real-AADR golden ctest green (the item gate) + NO regression + NO synthetic data used. Else false.' },
    diff_real: { type: 'boolean', description: 'you re-ran git diff: the change genuinely addresses the HIGH finding' },
    build_clean: { type: 'boolean' },
    goldens_green: { type: 'boolean', description: 'the real-AADR AT2 goldens pass (golden_fit0/fit1/rot via the item gate; for H3 the THOROUGH CpuBackend-oracle run)' },
    no_synthetic: { type: 'boolean', description: 'no synthetic data was introduced — the gate is the real goldens only' },
    regression: { type: 'boolean', description: 'true if any prior test regressed' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed + the gate result; for FAIL exactly what blocked it (and whether it needs human/design input)' },
  },
}

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  // clean HEAD so a prior item's revert/edits cannot contaminate this fix
  const fixer = [
    'You are a senior CUDA/C++ engineer applying ONE HIGH fix to steppe. Do NOT commit (the independent verdict agent commits). FIRST ensure a clean tree at HEAD: ' + CLEAN + ' .', STD, '', DEVLOOP, '',
    'FIX ' + it.id + ': ' + it.fix, '', 'PRIMARY FILES: ' + it.files, '', 'THE GATE (objective): ' + it.gate_name + '. Run it via: ' + it.gate,
    '', 'Apply the fix per the architecture standards; update any doc-comments the fix makes stale. VERIFY any CUDA/cuBLAS API claim against the CUDA 13.x docs (cite). Iterate until build clean + the gate green. Report: every file changed + what changed; the CUDA-doc citation if any; the FULL build result; the FULL gate ctest result (paste the summary lines + the golden PASS lines). If you cannot reach clean+green, do NOT pretend — report exactly what blocked it. REAL DATA ONLY.',
  ].join('\n')
  const fix = await tryAgent(fixer, { label: 'fix:' + it.id, phase: it.title })
  if (fix === null) { ledger.push({ item: it.id, pass: false, note: 'fixer terminal API error — SKIPPED' }); log('--- ' + it.id + ' fixer died — skip'); continue }

  const verdictPrompt = [
    'You are the INDEPENDENT VERDICT for ' + it.id + ' of steppe (you did NOT write the fix — be adversarial). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
    'THE FINDING: ' + it.fix, '', 'THE GATE: ' + it.gate_name,
    '', 'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — confirm the diff GENUINELY fixes ' + it.id + ' (not a sham, not comment-only). (2) RE-RUN the gate yourself: ' + it.gate + ' (and for H1/H2 you MAY also spot-run ' + BUILD + '). (3) PASS only if ALL: diff real; Release build clean (warnings-as-errors); the real-AADR golden gate GREEN; NO regression; NO synthetic data. ',
    'ON PASS: cd ' + R + ' && git add ONLY the specific changed/new source+test+doc files for THIS fix (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (the HIGH finding fixed + the real-golden gate result; cite the CUDA doc if relevant) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.',
    'ON FAIL: ' + CLEAN + ' (leave the repo green) and report the exact reason (and whether it needs human/design input). Return the structured verdict.',
  ].join('\n')
  const v = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:' + it.id, phase: it.title })
  if (v === null) { ledger.push({ item: it.id, pass: false, note: 'verdict terminal API error — SKIPPED' }); log('--- ' + it.id + ' verdict died — skip'); continue }
  ledger.push(v)
  if (v.pass) log('+++ ' + it.id + ' committed ' + v.commit_hash + ' — ' + v.note)
  else log('--- ' + it.id + ' FAILED (' + v.note + ') — reverted; continuing')
}

const passed = ledger.filter(x => x.pass).length
log('PHASE A done: ' + passed + '/' + ITEMS.length + ' HIGH committed')
return { ledger, passed }
