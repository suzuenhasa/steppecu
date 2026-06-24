export const meta = {
  name: 'optimizer-rematch',
  description: 'FAIR RE-MATCH of the two GPU optimizer shapes — the first spike (b97d79c) implemented IDEA 2 (population CMA-ES/DE) UNFAIRLY: its lambda population ran as a SERIAL in-thread loop (one thread per instance looping over lambda), NOT its actual parallel form, so it was stripped of the N*lambda parallel axis that is its whole point. RE-IMPLEMENT IDEA 2 AS IT SHOULD BE: block/warp-per-instance, the lambda candidates evaluated ACROSS THREADS in parallel, with a block-cooperative reduction for the CMA-ES/DE update (rank/recombine/update mean+sigma+C, or DE mutation+crossover+greedy-select) — so the parallel axis is N*lambda, not N. Keep IDEA 1 (the batched-sequential L-BFGS-B/projected-Newton fleet) as-is (it was implemented properly). ALSO fix the other caveat: the spike was nadmix=1 (degenerate — L-BFGS collapses to projected-Newton + the objective too small to exercise emulated-FP64); re-bench at nadmix=2 AND 3 (a less-degenerate real-f2 admixture-graph topology with 2 admixture events) so the population method robustness + the L-BFGS history BOTH matter and the matmul-heavy objective exercises the EmulatedFp64 GEMM seam. RE-RACE both, properly, at ALL N (1k/10k/100k/1M) AND nadmix in {1,2,3}, LOOKING FOR THE CROSSOVER (IDEA2 expected to win small-N / multimodal where its N*lambda fills the GPU that IDEA1 N-alone cannot; IDEA1 expected to win huge-N). BOTH must be GPU-BOUND (nvidia-smi proof for each); REAL-f2 objective (no synthetic), both converge to the host-pinned optimum. Reuse the existing tests/reference/bench_optimizers.cu (IDEA1 + the serial IDEA2 to REPLACE) + the real f2 fixture. EmulatedFp64 for the matmul evals, CUB/cuBLAS/cuSOLVER/cuRAND verified vs CUDA 13.x docs. SINGLE-GPU. Update docs/research/optimizer-comparison.md with the FAIR results (+ the crossover if any) + the revised qpGraph recommendation. Discipline: design (the proper block-cooperative IDEA2 + nadmix>=2 objective) -> implement -> re-race + GPU-bound proof for both + doc + commit. FAIL-PROTOCOL: NEVER git checkout/clean; on failure git stash push -u + HALT.',
  phases: [ { title: 'Design the PROPER block-cooperative IDEA2 + nadmix>=2 objective (GATE)' }, { title: 'Implement the parallel IDEA2 + nadmix>=2 + build' }, { title: 'Fair re-race (both, all N, nadmix 1/2/3) + crossover + doc + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -20 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -18; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ b97d79c. The optimizer spike (tests/reference/bench_optimizers.cu + docs/research/optimizer-comparison.md + qpgraph-optimizer-spike.md) compared IDEA1 (batched-sequential L-BFGS-B/projected-Newton fleet) vs IDEA2 (population CMA-ES/DE) on a real-f2 admixture-graph GLS objective. IDEA1 won — BUT IDEA2 was UNDER-IMPLEMENTED (serial in-thread lambda loop, not lambda-across-threads) so the race was unfair, and it was nadmix=1 (degenerate).',
  'THE FIX (this re-match): (1) re-implement IDEA2 PROPERLY — block-per-instance (or warp-per-instance), the lambda candidates evaluated ACROSS THREADS in parallel (the N*lambda parallel axis), a BLOCK-COOPERATIVE reduction for the update (CMA-ES: cooperative rank of the lambda by score + weighted recombine of the best mu into the new mean + sigma/C update; or DE: per-candidate mutation+crossover+greedy-select). cuRAND device API for the Gaussian sampling. NO serial per-thread lambda loop. (2) re-bench at nadmix=2 AND 3 (a real-f2 admixture-graph topology with 2 admixture events, more drift edges) so the objective is non-degenerate (L-BFGS history matters; the GLS GEMM is big enough to exercise EmulatedFp64) AND multimodal enough that the population method can show its robustness. Keep IDEA1 as-is (it was proper).',
  'THE FAIR RE-RACE: both optimizers, ALL N in {1k,10k,100k,1M}, nadmix in {1,2,3}, SAME objective+tol per cell. Metrics: wall-clock, GPU util% (nvidia-smi, the GPU-bound proof for EACH), objective evals, fraction-converged-to-the-host-pinned-optimum, peak VRAM. LOOK FOR THE CROSSOVER: IDEA2 (now properly N*lambda-parallel) is EXPECTED to win at SMALL N (where IDEA1 N-alone underfills the GPU but IDEA2 N*lambda saturates it) and on multimodal nadmix>=2 surfaces; IDEA1 expected to win at HUGE N. Report honestly where each wins — a crossover is a GOOD finding (it means use IDEA2-shape when N small / surface multimodal, IDEA1 when N huge).',
  'BOTH must be GPU-BOUND (the f-stat-sweep lesson; if either pegs 1 CPU core at GPU-0% that is a finding). REAL-f2 objective ONLY (the committed 9-pop fixture / real AADR), no synthetic (cuRAND is IDEA2 search-sampling, not data). Both must converge to the host-pinned checkable optimum. EmulatedFp64{40} for the matmul-heavy objective evals + native carve-out. Reuse gemmStridedBatched/potrfBatched/CUB/cuRAND; VERIFY every API vs the CUDA 13.x docs.',
  'SINGLE-GPU --device 0 (multi-GPU PARKED). RELEASE build-rel; do not regress STEPPE_THOROUGH ctest. Box ' + SSH + '; nvcc -> ' + PATHENV + '. This updates the spike comparison; the WINNER (with the crossover) informs the qpGraph optimizer.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout -- . / git clean -fd. On ANY failure ' + STASH + ' "wip:optimizer-rematch-FAILED-<reason>" + HALT. NON-trivial blocker -> STOP + report. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Design the PROPER block-cooperative IDEA2 + nadmix>=2 objective (GATE)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','idea2_parallel_design','nadmix_objective','fairness_controls','cuda13_apis','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff a proper block-cooperative lambda-across-threads IDEA2 + a nadmix>=2 real-f2 objective with a checkable optimum is buildable, GPU-bound, reusing the bench harness. false -> structural blocker' },
    idea2_parallel_design: { type: 'string', description: 'the PROPER IDEA2: block/warp-per-instance layout, lambda candidates across threads, the block-cooperative reduction for the CMA-ES/DE update, cuRAND device sampling — the N*lambda parallel axis (NOT a serial lambda loop)' },
    nadmix_objective: { type: 'string', description: 'the nadmix=2 (and 3) real-f2 admixture-graph topology (2 admixture events, more drift edges), its theta+bounds, the host-pinned checkable optimum, why it is non-degenerate + multimodal enough' },
    fairness_controls: { type: 'string', description: 'the controlled re-race: N x nadmix grid, same objective/tol per cell, the metrics, the crossover hypothesis' },
    cuda13_apis: { type: 'string', description: 'cuRAND device API (Philox normal) + cuBLAS/cuSOLVER/CUB batched APIs VERIFIED vs CUDA 13.x docs for the N*lambda batch' },
    blocker: { type: 'string', description: 'if NOT feasible: the structural blocker + options' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are designing the FAIR optimizer re-match (verify-before-implement; NO code changes). READ the existing tests/reference/bench_optimizers.cu (IDEA1 proper; the SERIAL IDEA2 to replace), docs/research/optimizer-comparison.md + qpgraph-optimizer-spike.md (the methodology + the two flagged caveats), the reusable seams (gemmStridedBatched/potrfBatched in cuda_backend.cu; dev_chisq_of_core; the f2 fixture; assemble_f3_triples). Design: (1) the PROPER block-cooperative lambda-across-threads IDEA2 (CMA-ES and/or DE; cuRAND device sampling; the N*lambda parallel axis + the cooperative update reduction); (2) the nadmix=2/3 real-f2 admixture-graph objective (non-degenerate, multimodal-enough, host-pinned optimum); (3) the fair N x nadmix re-race grid + the crossover hypothesis. VERIFY the cuRAND/cuBLAS/cuSOLVER/CUB APIs vs the CUDA 13.x docs. Return the structured design. If infeasible, feasible=false + blocker. Do NOT implement.', STD,
].join('\n'), { schema: DESIGN_SCHEMA, label: 'design:rematch', phase: 'Design the PROPER block-cooperative IDEA2 + nadmix>=2 objective (GATE)' })
if (design === null) { log('--- design died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- rematch STRUCTURAL (defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('rematch feasible; proper IDEA2: ' + String(design.idea2_parallel_design).slice(0,120))

phase('Implement the parallel IDEA2 + nadmix>=2 + build')
const fixer = await tryAgent([
  'You are implementing the FAIR re-match per this design (replace the serial IDEA2 with the proper block-cooperative lambda-across-threads form; add nadmix=2/3; keep IDEA1):\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build (' + BUILD + ').', '',
  'IMPLEMENT in tests/reference/bench_optimizers.cu: (1) the PROPER IDEA2 — block/warp-per-instance, lambda candidates across threads, block-cooperative CMA-ES/DE update, cuRAND device sampling (N*lambda parallel, NO serial lambda loop). (2) the nadmix=2 + nadmix=3 real-f2 admixture-graph objective (host-pinned optimum). Keep IDEA1. EmulatedFp64 for the matmul evals. Build (Release) + no ctest regression. SANITY: the proper IDEA2 converges to the host-pinned optimum at nadmix 1/2/3; at SMALL N a quick run shows IDEA2 now uses more GPU than the serial version did. Report the proper-IDEA2 impl (the cooperative layout), the build, the small-N sanity + your nvidia-smi observation for IDEA2 now. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:rematch', phase: 'Implement the parallel IDEA2 + nadmix>=2 + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for the optimizer re-match. Accumulated edits (do NOT clean/revert; git stash only if forced). Reach a CLEAN Release build + green ctest, patching only trivial -Werror / CMake / cuRAND-cuBLAS-cuSOLVER-CUB API misuse (VERIFY vs CUDA 13 docs; the block-cooperative reduction is the likely error site). DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 5x. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement the parallel IDEA2 + nadmix>=2 + build' })

phase('Fair re-race (both, all N, nadmix 1/2/3) + crossover + doc + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','idea2_now_parallel','both_converge','crossover','wallclock_table','idea2_gpu_util','revised_recommendation','no_synthetic','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if IDEA2 is now PROPERLY parallel (lambda-across-threads, GPU-util materially higher than the serial version esp. at small N), both converge to the host-pinned optimum across N x nadmix, the fair re-race is reported with the crossover (if any), no regression, single-GPU, no synthetic' },
    fail_severity: { type: 'string' },
    idea2_now_parallel: { type: 'boolean', description: 'IDEA2 lambda is now across-threads (block-cooperative), not a serial in-thread loop — verified in the diff + the higher GPU util' },
    both_converge: { type: 'boolean' },
    crossover: { type: 'string', description: 'is there a crossover? at what N / nadmix does IDEA2 win vs IDEA1, now that IDEA2 is fair? (the key finding)' },
    wallclock_table: { type: 'string', description: 'the fair table: per optimizer, per N, per nadmix — wall-clock + evals + conv-frac + GPU util + VRAM' },
    idea2_gpu_util: { type: 'string', description: 'IDEA2 GPU util range now (vs the serial version) — proof it is genuinely parallel/GPU-bound' },
    revised_recommendation: { type: 'string', description: 'the revised qpGraph optimizer recommendation given the FAIR comparison (which shape, when, the fallback)' },
    no_synthetic: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the fair result + the crossover + the revised recommendation; on FAIL the blocker' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the FAIR optimizer re-match (adversarial; the WHOLE point is fairness — confirm IDEA2 is now genuinely lambda-across-threads parallel, not still serial; if it is still serial or under-implemented, FAIL). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — IDEA2 is now block/warp-cooperative lambda-across-threads (a cooperative reduction, cuRAND device sampling), NOT a serial per-thread lambda loop; IDEA1 unchanged; nadmix=2/3 objective added. (2) ' + BUILD + ' — ctest green. (3) RUN THE FAIR RE-RACE on box5090: both optimizers, N in {1k,10k,100k,1M} x nadmix in {1,2,3}, sampling nvidia-smi for EACH; confirm IDEA2 GPU util is now materially higher (esp. small N) than the serial version; confirm both converge to the host-pinned optimum; capture wall-clock + evals + conv-frac + util + VRAM per cell. IDENTIFY THE CROSSOVER. (4) single-GPU, real f2, no synthetic. PASS only if IDEA2 is genuinely parallel + both benchmarked fairly + the crossover/winner reported. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed bench/doc files (NEVER git add dot; never aadr/ atlas_results/), UPDATE docs/research/optimizer-comparison.md with the FAIR results (the N x nadmix tables + the crossover + the revised qpGraph recommendation; clearly mark this supersedes the unfair first race), commit (spike(optimizer rematch): FAIR race — IDEA2 re-implemented lambda-across-threads (block-cooperative) + nadmix 2/3; <crossover/winner>; revised qpGraph recommendation) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:optimizer-rematch-FAILED" (capture the ref). Classify fail_severity (bad if IDEA2 is still serial/under-implemented). Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:rematch', phase: 'Fair re-race (both, all N, nadmix 1/2/3) + crossover + doc + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ fair rematch ' + verdict.commit_hash + ' — crossover: ' + String(verdict.crossover).slice(0,160))
else log('--- fair rematch FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { design, verdict }
