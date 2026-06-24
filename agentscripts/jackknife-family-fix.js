export const meta = {
  name: 'jackknife-family-fix',
  description: 'FIX the remaining jackknife-family host violations from the audit (docs/research/host-compute-audit.md M1 + M2): the f4-RATIO and qpDstat block-jackknives run on the HOST — the same host-long-double-per-item pattern as the qpfstats jackknife already moved on-device (245b1aa). M1: src/core/qpadm/f4ratio.cpp ratio_jackknife (3 long-double passes over n_block PER ratio tuple; AND assemble_f4_quartets D2Hs x_blocks/x_loo [2N*nb each] to host just to feed it). M2: src/core/stats/dstat.cpp dstat_jackknife (host num/den block-jackknife per quadruple). BOTH are the SAME structure: a RATIO of two block-summed quantities (f4ratio: alpha=f4_num/f4_den; dstat: D=Σ(a-b)(c-d)/Σ(a+b-2ab)(c+d-2cd)) jackknifed over blocks (AT2 jack_mat_stats / the f4-ratio jackknife-of-the-ratio). THE FIX: ONE SHARED on-device batched ratio-block-jackknife ComputeBackend virtual — input per-(item,block) num + den (already device-resident: f4ratio from assemble_f4_quartets x_blocks/x_loo, dstat from dstat_block_reduce numsum/densum/cnt), output the ratio est + SE/z/p via the block-jackknife, batched over items + reducing over blocks (mirror the on-device jackknife_diag / the qpfstats on-device jackknife). Reused by BOTH f4ratio (num=f4_num,den=f4_den) and dstat (num,den) — NO parallel bolt-on, NO forked path. Eliminates the f4ratio x_blocks/x_loo D2H too (the jackknife consumes them resident). GOLDEN-EXACT: the f4-ratio golden AND the qpDstat goldens (Part-A f2-path + Part-B genotype) MUST still hold at their tiers + CpuBackend==CudaBackend + full STEPPE_THOROUGH ctest green. The host uses long double; the on-device must be NATIVE FP64 with the EXACT ascending-b operand order (the §12 cancellation carve-out — NOT emulated; double-double fallback only if the tier misses, like the qpfstats jackknife fix). REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host per-item/per-block jackknife loop); VERIFY (real code file:line + the AT2 jack_mat_stats math); REUSE no parallel bolt-on (one shared virtual for both); native FP64 carve-out; SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing. Discipline: diagnose+plan (GATE — the shared ratio-jackknife virtual + golden-preserving accumulation for both stats) -> implement (the kernel + rewire f4ratio + dstat, drop the x_blocks/x_loo D2H) -> verify (f4ratio + qpDstat goldens + the host jackknife loops gone + ctest) -> commit. FAIL-PROTOCOL: verify->commit on green; on BAD (any golden shifts / host loop remains) HALT + report + defer; NEVER silently fail/revert; on failure git stash push -u + HALT.',
  phases: [ { title: 'Diagnose M1+M2 + design the shared ratio-jackknife virtual (GATE: golden-preserving for both stats)' }, { title: 'Implement the shared on-device ratio-jackknife + rewire f4ratio + dstat + build' }, { title: 'Verify f4ratio + qpDstat goldens + host loops gone + commit-or-HALT' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -22; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ 1b36a0b. The host-compute audit (docs/research/host-compute-audit.md) flagged M1 (f4ratio ratio_jackknife) + M2 (dstat dstat_jackknife) as host-jackknife violations — the SAME pattern as the qpfstats jackknife already moved on-device (245b1aa) and the qpfstats reduce already fixed (1b36a0b, 40-pop 59s->8.1s).',
  'M1 — src/core/qpadm/f4ratio.cpp ratio_jackknife (~lines 79-155), driven per tuple (~232-240): 3 long-double passes over n_block per f4-ratio tuple producing alpha/se/z; assemble_f4_quartets has D2Hd x_blocks/x_loo [2N*nb] to host (cuda_backend.cu) just to feed it. M2 — src/core/stats/dstat.cpp dstat_jackknife (~70-157): host num/den block-jackknife per quadruple. BOTH are a RATIO of two block-summed quantities jackknifed over blocks (AT2 jack_mat_stats): f4ratio alpha=f4_num/f4_den; dstat D=Σ(a-b)(c-d)/Σ(a+b-2ab)(c+d-2cd). The ratio block-jackknife xtau form: Rb=num_loo_b/den_loo_b, tau_b=h*tot-(h-1)*Rb, est+var over blocks.',
  'THE FIX: ONE shared on-device batched ratio-block-jackknife ComputeBackend virtual (backend.hpp) — input per-(item,block) num + den (device-resident: f4ratio x_blocks/x_loo, dstat numsum/densum), block weights; output est + se (+ z + p) batched over items, reduce over blocks (mirror jackknife_diag / the on-device qpfstats jackknife shape). Reused by BOTH f4ratio + dstat — NO parallel bolt-on. Drop the f4ratio x_blocks/x_loo D2H (consume resident). CpuBackend keeps the existing long-double ratio_jackknife/dstat_jackknife as the parity oracle (unchanged).',
  'GOLDEN-EXACT (NON-NEGOTIABLE): the f4-ratio golden AND the qpDstat goldens (Part-A f2-path + Part-B genotype) MUST hold + CpuBackend==CudaBackend + full STEPPE_THOROUGH ctest green. Native FP64, EXACT ascending-b operand order matching the host long-double reference (the §12 cancellation carve-out; NOT emulated). Double-double fallback ONLY if the tier misses (as in the qpfstats jackknife fix). If a golden cannot be held on-device, HALT + defer.',
  'STANDING REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host per-item/per-block jackknife loop); VERIFY (file:line + the AT2 jack_mat_stats / the f4ratio.cpp header math); REUSE no parallel bolt-on (one shared virtual for f4ratio + dstat); native FP64 carve-out; SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing.',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER git checkout -- . / git clean -fd. verify->commit ONLY on green (both goldens held + the host jackknife loops gone from the production path). On a BAD result (any golden shifts, host loop remains) HALT + report + DEFER — do NOT silently fail/revert. On any failure ' + STASH + ' "wip:jackknife-family-FAILED-<reason>" + HALT.',
  'SINGLE-GPU --device 0. RELEASE build-rel. REAL AADR; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. §4 layering. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Diagnose M1+M2 + design the shared ratio-jackknife virtual (GATE: golden-preserving for both stats)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','m1_f4ratio','m2_dstat','shared_virtual','golden_accumulation','consumers_rewire','cuda13','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff one shared on-device ratio-block-jackknife virtual can replace both host jackknives preserving the f4ratio + qpDstat goldens. false -> HALT + defer' },
    m1_f4ratio: { type: 'string', description: 'the f4ratio host ratio_jackknife (file:line) + the x_blocks/x_loo D2H that feeds it' },
    m2_dstat: { type: 'string', description: 'the dstat host dstat_jackknife (file:line) + its inputs' },
    shared_virtual: { type: 'string', description: 'the shared on-device ratio-block-jackknife virtual design (the kernel: per-(item,block) num/den -> ratio xtau -> est+se, batched over items)' },
    golden_accumulation: { type: 'string', description: 'how native-FP64 reproduces the host long-double ratio jackknife for BOTH stats at their golden tiers (operand order / double-double fallback)' },
    consumers_rewire: { type: 'string', description: 'how f4ratio + dstat are rewired to the shared virtual; the x_blocks/x_loo D2H dropped' },
    cuda13: { type: 'string', description: 'any new kernel/CUB API verified vs CUDA 13.x' },
    blocker: { type: 'string', description: 'if NOT feasible: the blocker (HALT + defer)' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are diagnosing M1+M2 + designing the shared ratio-jackknife virtual (verify-before-implement; NO code changes). READ src/core/qpadm/f4ratio.cpp (ratio_jackknife + the per-tuple driver + the header AT2 jack_mat_stats math) + src/core/stats/dstat.cpp (dstat_jackknife) + how their inputs are produced (assemble_f4_quartets x_blocks/x_loo, dstat_block_reduce numsum/densum/cnt) + the on-device jackknife_diag / the qpfstats on-device jackknife (the template) + backend.hpp (where to add the virtual). Design ONE shared on-device batched ratio-block-jackknife virtual serving both, preserving the f4ratio + qpDstat goldens (native FP64, exact operand order). VERIFY any new kernel/CUB vs CUDA 13.x. Return the structured plan. If one shared virtual cannot preserve both goldens on-device, feasible=false + blocker (HALT + defer).', STD,
].join('\n'), { schema: DESIGN_SCHEMA, label: 'diagnose:jk', phase: 'Diagnose M1+M2 + design the shared ratio-jackknife virtual (GATE: golden-preserving for both stats)' })
if (design === null) { log('--- diagnose died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- jackknife-family STRUCTURAL (HALT + defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('jackknife-family feasible; shared virtual: ' + String(design.shared_virtual).slice(0,140))

phase('Implement the shared on-device ratio-jackknife + rewire f4ratio + dstat + build')
const fixer = await tryAgent([
  'You are implementing the shared on-device ratio-block-jackknife per this plan (GOLDEN-EXACT for f4ratio + qpDstat; one shared virtual, no bolt-on):\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT the shared on-device ratio-block-jackknife virtual (the kernel batched over items, reducing over blocks, native FP64 exact operand order) + rewire f4ratio (consume resident x_blocks/x_loo, drop the D2H) + dstat (consume resident numsum/densum/cnt) to it; CpuBackend keeps the long-double oracle. Build + full STEPPE_THOROUGH ctest. SANITY: the f4-ratio golden + qpDstat Part-A + Part-B genotype goldens hold at tier; CpuBackend==CudaBackend; the host ratio_jackknife/dstat_jackknife loops no longer run in the production (CudaBackend) path. Report files, the shared virtual (file:line), the rewire + the dropped D2H, the golden re-check, the FULL ctest. Do NOT commit. BAD/NON-trivial (any golden shifts) -> STOP + report (do NOT revert).',
].join('\n'), { label: 'implement:jk', phase: 'Implement the shared on-device ratio-jackknife + rewire f4ratio + dstat + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for the jackknife-family fix. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build + green STEPPE_THOROUGH ctest (f4ratio + qpDstat goldens MUST hold), patching only trivial errors. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 5x. NON-trivial / goldens shift -> STOP + report. Report build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement the shared on-device ratio-jackknife + rewire f4ratio + dstat + build' })

phase('Verify f4ratio + qpDstat goldens + host loops gone + commit-or-HALT')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','f4ratio_golden_held','qpdstat_goldens_held','host_jackknife_gone','shared_virtual','d2h_dropped','goldens_green','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the f4-ratio golden AND qpDstat goldens (Part-A + Part-B) hold at tier, the ratio block-jackknife runs on-device (host loops gone from production), one shared virtual (no bolt-on), CpuBackend==CudaBackend, ctest green, single-GPU' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (HALT + defer); ANY golden shift = bad' },
    f4ratio_golden_held: { type: 'boolean' }, qpdstat_goldens_held: { type: 'boolean' },
    host_jackknife_gone: { type: 'boolean', description: 'the host ratio_jackknife/dstat_jackknife no longer run in the CudaBackend/production path' },
    shared_virtual: { type: 'boolean', description: 'one shared backend virtual serves both f4ratio + dstat (not two forked impls)' },
    d2h_dropped: { type: 'boolean', description: 'the f4ratio x_blocks/x_loo D2H is eliminated (consumed resident)' },
    goldens_green: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'what moved on-device + the goldens checked; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the jackknife-family fix (adversarial; hold the f4ratio + qpDstat goldens AND confirm the host loops are gone). plan:\n<<<\n' + JSON.stringify(design) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — ONE shared on-device ratio-block-jackknife virtual serves both f4ratio + dstat (no forked path), native FP64 exact operand order, the host ratio_jackknife/dstat_jackknife now only on CpuBackend (the oracle), the f4ratio x_blocks/x_loo D2H dropped. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest green; the f4-ratio golden + qpDstat Part-A + Part-B genotype goldens reproduce at tier; CpuBackend==CudaBackend. (3) single-GPU, real AADR. PASS only if BOTH stats goldens HELD and the host jackknife loops are gone from production. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/cmake (NEVER git add dot; never aadr/ atlas_results/), commit (perf(jackknife): shared on-device ratio-block-jackknife virtual for f4-ratio + qpDstat — kill the host long-double per-item loops (M1+M2) + the f4ratio x_blocks/x_loo D2H; f4ratio + qpDstat goldens held) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs/research/host-compute-audit.md (mark M1+M2 done; note the campaign: jackknife/decode/reduce done, qpfstats 40-pop 59s->8.1s) + RESUME/TODO. ',
  'ON BAD (any golden shifts / host loop remains): DO NOT git checkout/clean, DO NOT revert. ' + STASH + ' "wip:jackknife-family-FAILED" (capture the ref). Classify fail_severity=bad + the blocker for the user. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:jk', phase: 'Verify f4ratio + qpDstat goldens + host loops gone + commit-or-HALT' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ jackknife-family ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- jackknife-family FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — HALT+defer — ' + verdict.note)
return { design, verdict }
