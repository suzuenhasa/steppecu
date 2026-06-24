export const meta = {
  name: 'host-cleanup-tail',
  description: 'FINISH the host-compute audit campaign — the remaining MEDIUM tail (docs/research/host-compute-audit.md M5, M6, M7). Three independent small on-device fixes; the two CRITICALs (decode seam) + the jackknife family (M1/M2) + the qpfstats reduce are already landed. M5 (DATES target repack): src/core/stats/dates.cpp ~296-313 re-packs the admixed TARGET pop packed genotypes onto the kept SNP axis as a HOST bit-repack loop -> move to a device gather (the kept-axis indices already exist from the decode seam). M6 (DATES exp-fit grid): the ~few-hundred/thousand-point exponential-decay fit (A*exp(-t*d)+c) grid/host loop -> a small device fit (reuse the qpGraph/bench fleet or a tiny batched least-squares; the curve is already device-resident from the cuFFT). M7 (qpAdm single-model SE): src/core/qpadm/nested_models.cpp sample_cov_diag (the S7 jackknife SE variance reduction = diag(cov(wmat))) runs on the HOST in the single-model/large-tail path, EVEN THOUGH the on-device kernel ALREADY EXISTS (launch_qpadm_se_from_wmat_batched / qpadm_se_from_wmat_kernel, used by the batched S8 path) — just WIRE the existing kernel into the single-model se_from_loo path (add/route a backend se_from_wmat virtual so the single-model path runs the SE reduction on-device too; the kernel is bit-identical to sample_cov_diag by its own comment). REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host reduction/repack hot loop); VERIFY (real code file:line); REUSE no parallel bolt-on (M7 reuses the EXISTING kernel; M5/M6 reuse the decode kept-axis + the fleet); native FP64 carve-out for the cancellation-sensitive reductions; SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing. GOLDEN-EXACT: the DATES date golden (M5/M6) AND the qpAdm goldens incl the large-model/nr>32 SE path (M7) MUST hold at their tiers + CpuBackend==CudaBackend + full STEPPE_THOROUGH ctest green. NOTE DATES is decode-bound (M5/M6 are bounded host loops, low wall impact — the value is closing the host-violation, not a big speedup); M7 is the cheapest real win (the kernel exists). L1-L4 (rank_Q SVD, ploidy detect, qpGraph edge/worst-z) are bounded per-model diagnostics, DEFERRED by design — out of scope. Discipline: diagnose (GATE — confirm the 3 fixes + golden-preserving, M7 reuses the existing kernel) -> implement (all 3) -> verify (DATES + qpAdm goldens + ctest) -> commit. FAIL-PROTOCOL: verify->commit on green; on BAD (any golden shifts) HALT + report + defer; NEVER silently fail/revert; on failure git stash push -u + HALT.',
  phases: [ { title: 'Diagnose M5+M6+M7 (GATE: golden-preserving on-device, M7 reuses the existing kernel)' }, { title: 'Implement M5 (dates repack) + M6 (dates fit) + M7 (wire SE kernel) + build' }, { title: 'Verify DATES + qpAdm goldens + host loops gone + commit-or-HALT' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -22; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ 4dbff58. The host-compute audit campaign is nearly done: the 2 CRITICAL decode-seam violations (Stage 1 9ad33d9 + Stage 2 5fc808b) + the jackknife family M1/M2 (864eadd) + the qpfstats reduce (1b36a0b, 40-pop 59s->8.1s) are landed. This closes the MEDIUM tail: M5, M6, M7.',
  'M5 (DATES target repack): src/core/stats/dates.cpp ~296-313 re-packs the admixed target pop packed genotypes onto the kept SNP axis with a HOST bit-repack loop. Fix: a device gather onto the kept axis (the decode seam already produces the kept-axis indices/compaction on-device).',
  'M6 (DATES exp-fit grid): the exponential-decay fit A*exp(-t*d)+c over the ~few-hundred binned-distance points runs on the host. Fix: a small device fit (reuse the qpGraph/bench fleet optimizer or a tiny batched least-squares; the cov(d) curve is already device-resident from the cuFFT engine). DATES is decode-bound, so this is about closing the host-violation, not a big speedup.',
  'M7 (qpAdm single-model SE — the cheapest real win): src/core/qpadm/nested_models.cpp sample_cov_diag (the S7 SE variance reduction diag(cov(wmat))) runs on the HOST in the single-model/large-tail path (nl>5 / nr>10 / r>4, the >kQpMax* models that route through run_impl->se_from_loo, NOT the batched S8 path). The on-device kernel ALREADY EXISTS: launch_qpadm_se_from_wmat_batched / qpadm_se_from_wmat_kernel (qpadm_fit_kernels.cu ~1621-1623, used by the batched path, bit-identical to sample_cov_diag by its own comment). Fix: add/route a backend se_from_wmat virtual so the single-model se_from_loo path runs the SE reduction on-device (D2H only the final nl-length se). REUSE the existing kernel — NO new kernel.',
  'STANDING REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host reduction/repack hot loop); VERIFY (file:line); REUSE no parallel bolt-on (M7 = the EXISTING kernel; M5/M6 = the decode kept-axis + the fleet); native FP64 carve-out for cancellation; SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing.',
  'GOLDEN-EXACT (NON-NEGOTIABLE): the DATES date golden (M5/M6) AND the qpAdm goldens — incl the LARGE-model/nr>32 SE path that exercises se_from_loo (M7) — MUST hold at their tiers + CpuBackend==CudaBackend + full STEPPE_THOROUGH ctest green. If any cannot be held on-device, HALT + defer that item (the others can still land).',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER git checkout -- . / git clean -fd. verify->commit ONLY on green (all goldens held + the host loops gone). On a BAD result (any golden shifts) HALT + report + DEFER — do NOT silently fail/revert. On any failure ' + STASH + ' "wip:host-cleanup-FAILED-<reason>" + HALT.',
  'SINGLE-GPU --device 0. RELEASE build-rel. REAL AADR; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. §4 layering. nothing builds locally. L1-L4 are DEFERRED (out of scope).',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Diagnose M5+M6+M7 (GATE: golden-preserving on-device, M7 reuses the existing kernel)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','m5_dates_repack','m6_dates_fit','m7_se_kernel','golden_plan','any_defer','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff at least M7 (+ as many of M5/M6 as hold) can move on-device preserving the goldens. false only if ALL three are blocked' },
    m5_dates_repack: { type: 'string', description: 'the dates target-repack host loop (file:line) + the on-device gather plan (reusing the decode kept-axis)' },
    m6_dates_fit: { type: 'string', description: 'the dates exp-fit host loop (file:line) + the on-device fit plan (reuse the fleet / batched LS)' },
    m7_se_kernel: { type: 'string', description: 'the host sample_cov_diag (file:line) + how the EXISTING launch_qpadm_se_from_wmat_batched is wired into se_from_loo (the new/routed virtual)' },
    golden_plan: { type: 'string', description: 'the DATES date golden (M5/M6) + the qpAdm goldens incl the large-model SE path (M7) to gate each' },
    any_defer: { type: 'string', description: 'if any of M5/M6/M7 must be deferred (cannot hold a golden on-device), which + why' },
    blocker: { type: 'string', description: 'if feasible=false (all blocked): the blocker' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are diagnosing the host-cleanup tail M5+M6+M7 (verify-before-implement; NO code changes). READ src/core/stats/dates.cpp (the target repack ~296-313 + the exp-fit), the dates curve device path (cuda_backend.cu dates_curve), src/core/qpadm/nested_models.cpp (sample_cov_diag + se_from_loo) + the EXISTING launch_qpadm_se_from_wmat_batched / qpadm_se_from_wmat_kernel (qpadm_fit_kernels.cu ~1621) + the batched path that already uses it + backend.hpp. Confirm each fix on-device + golden-preserving; M7 must REUSE the existing kernel (no new kernel). Identify the goldens that gate each (DATES date; qpAdm large-model SE path). Return the structured plan; mark any item that must be deferred. feasible=false only if ALL three are blocked.', STD,
].join('\n'), { schema: DESIGN_SCHEMA, label: 'diagnose:tail', phase: 'Diagnose M5+M6+M7 (GATE: golden-preserving on-device, M7 reuses the existing kernel)' })
if (design === null) { log('--- diagnose died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- host-cleanup-tail all-blocked (HALT + defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('host-cleanup feasible; defers: ' + String(design.any_defer).slice(0,140))

phase('Implement M5 (dates repack) + M6 (dates fit) + M7 (wire SE kernel) + build')
const fixer = await tryAgent([
  'You are implementing the host-cleanup tail per this plan (M5+M6+M7 on-device; M7 reuses the existing kernel; GOLDEN-EXACT; skip any item the plan deferred):\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT: M5 (dates target repack -> device gather on the kept axis), M6 (dates exp-fit -> device fit, reuse the fleet/batched-LS), M7 (route the EXISTING launch_qpadm_se_from_wmat_batched into the single-model se_from_loo path via a backend virtual — NO new kernel). Native FP64 carve-out for cancellation. Build + full STEPPE_THOROUGH ctest. SANITY: the DATES date golden + the qpAdm goldens (incl the large-model/nr>32 SE path) hold; CpuBackend==CudaBackend; the host repack/fit/sample_cov_diag loops no longer run in production. Report files, the 3 fixes (file:line), the M7 kernel reuse, the golden re-checks, the FULL ctest. Do NOT commit. BAD/NON-trivial (golden shifts) -> STOP + report (do NOT revert).',
].join('\n'), { label: 'implement:tail', phase: 'Implement M5 (dates repack) + M6 (dates fit) + M7 (wire SE kernel) + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for the host-cleanup tail. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build + green STEPPE_THOROUGH ctest (DATES + qpAdm goldens MUST hold), patching only trivial errors. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 5x. NON-trivial / goldens shift -> STOP + report. Report build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement M5 (dates repack) + M6 (dates fit) + M7 (wire SE kernel) + build' })

phase('Verify DATES + qpAdm goldens + host loops gone + commit-or-HALT')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','items_landed','items_deferred','dates_golden_held','qpadm_goldens_held','host_loops_gone','m7_reuses_existing','goldens_green','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true if the landed items (M7 at minimum, M5/M6 unless deferred) are on-device, the DATES + qpAdm goldens (incl the large-model SE path) hold, M7 reuses the existing kernel (no new kernel), CpuBackend==CudaBackend, ctest green, single-GPU' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (HALT + defer); ANY golden shift = bad' },
    items_landed: { type: 'string', description: 'which of M5/M6/M7 landed on-device' },
    items_deferred: { type: 'string', description: 'which (if any) were deferred + why' },
    dates_golden_held: { type: 'boolean' }, qpadm_goldens_held: { type: 'boolean' },
    host_loops_gone: { type: 'boolean', description: 'the landed items host loops no longer run in production' },
    m7_reuses_existing: { type: 'boolean', description: 'M7 routed the EXISTING se_from_wmat kernel (no new kernel)' },
    goldens_green: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'what landed + the goldens checked; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the host-cleanup tail (adversarial; hold the DATES + qpAdm goldens; confirm the host loops gone + M7 reuses the existing kernel). plan:\n<<<\n' + JSON.stringify(design) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — M5 (dates repack) + M6 (dates fit) are on-device (no host loop), M7 ROUTES the EXISTING launch_qpadm_se_from_wmat_batched into se_from_loo (NO new kernel), native FP64 carve-out, deferred items (if any) clearly not regressed. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest green; the DATES date golden + the qpAdm goldens (incl the large-model/nr>32 SE path) reproduce; CpuBackend==CudaBackend. (3) single-GPU, real AADR. PASS if the landed items are on-device + every golden HELD. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/cmake (NEVER git add dot; never aadr/ atlas_results/), commit (perf(host-cleanup): M5/M6 DATES repack+fit on-device + M7 single-model qpAdm SE via the existing batched kernel — the host-compute audit MEDIUM tail closed; DATES + qpAdm goldens held) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs/research/host-compute-audit.md (M5/M6/M7 done; the campaign closed except the deferred L1-L4) + RESUME/TODO. ',
  'ON BAD (any golden shifts): DO NOT git checkout/clean, DO NOT revert. ' + STASH + ' "wip:host-cleanup-FAILED" (capture the ref). Classify fail_severity=bad + the blocker for the user. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:tail', phase: 'Verify DATES + qpAdm goldens + host loops gone + commit-or-HALT' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ host-cleanup-tail ' + verdict.commit_hash + ' — landed: ' + verdict.items_landed + ' — ' + verdict.note)
else log('--- host-cleanup-tail FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — HALT+defer — ' + verdict.note)
return { design, verdict }
