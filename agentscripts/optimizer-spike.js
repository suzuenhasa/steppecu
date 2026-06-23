export const meta = {
  name: 'optimizer-spike',
  description: 'PROTOTYPE + COMPARE two GPU optimizer shapes for qpGraph (de-risk the highest-risk new piece BEFORE building qpGraph): IDEA 1 = a BATCHED SEQUENTIAL FLEET (one bounded quasi-Newton / L-BFGS-B per instance, all instances run concurrently SIMT, the fleet is the parallel axis) vs IDEA 2 = a BATCHED POPULATION method (CMA-ES or differential-evolution per instance; population x N is the parallel axis). THE BATCH SCENARIO = the qpGraph fleet: N INDEPENDENT box-constrained NONLINEAR least-squares fits (the qpGraph objective shape), scaled N small->huge to find where each saturates the GPU + how each scales. BOTH must be GPU-BOUND (the f-stat-sweep lesson — if either pegs 1 CPU core at GPU-0%, that is THE finding; prove with nvidia-smi). REAL-DATA-GROUNDED objective (NO synthetic): a small admixture-graph-shaped nonlinear GLS residual ||f_obs - f_pred(theta)||^2_{Sigma^-1} over REAL-AADR f2-derived f-stats (a fixed small topology with 1-2 admixture events => f_pred nonlinear in the edge drifts>=0 + mixtures in [0,1]); a known/checkable optimum so convergence correctness is verifiable. The benchmark measures NUMERICAL-METHODS metrics (wall-clock, GPU util%, objective evals, fraction-converged-to-optimum) on this real-f2 objective at increasing N. EmulatedFp64 for the matmul-heavy objective evals; native-FP64 carve-out for cancellation; reuse the cuBLAS-batched / cuSOLVER / CUB seams (potrfBatched/gemmStridedBatched, the f2 cache, small_linalg as the host oracle). VERIFY CUB/cuBLAS/cuSOLVER APIs against the CUDA 13.x docs. SINGLE-GPU (multi-GPU PARKED). This is a SPIKE/benchmark (a bench TU + a comparison doc), NOT the qpGraph build — the WINNER becomes qpGraph optimizer later. Discipline: design (objective + both optimizers + methodology, GPU-bound + CUDA-13 verified) -> implement both + harness -> run the comparison + GPU-bound proof for BOTH + write the doc + commit. FAIL-PROTOCOL: NEVER git checkout/clean; on failure git stash push -u + HALT.',
  phases: [ { title: 'Design the objective + both optimizers + methodology (GATE)' }, { title: 'Implement both GPU optimizers + the batched objective + harness + build' }, { title: 'Run the comparison + GPU-bound proof for both + doc + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -20 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -20; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'
const F2_500 = '/workspace/data/f2_500'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 07df9af. The new-tools research (docs/research/qpgraph-gpu-design.md + new-tools-roadmap.md) flagged the qpGraph OPTIMIZER as the highest-risk piece + the most reusable new asset. This SPIKE compares the two candidate GPU optimizer shapes empirically BEFORE building qpGraph.',
  'IDEA 1 (batched sequential fleet): N independent fits, each its OWN sequential bounded quasi-Newton (L-BFGS-B or a projected-gradient variant), all N advancing in lockstep SIMT — the FLEET is the parallel axis; each thread/block carries one instance state (the m history vectors + x); the objective+gradient per tick are device-batched. The recurrence stays per-instance; parallelism = the N count + the data-parallel objective.',
  'IDEA 2 (batched population): each fit runs a POPULATION optimizer (CMA-ES or differential-evolution) with population lambda; per generation ALL lambda candidates are evaluated in parallel; for N fits that is N*lambda parallel evals/generation. Derivative-free; the population is the parallel axis (more parallelism, more evals/fit, simpler per-thread state).',
  'THE OBJECTIVE (REAL-DATA-GROUNDED, NO synthetic): a small admixture-graph-shaped NONLINEAR box-constrained GLS least-squares: minimize ||f_obs - f_pred(theta)||^2 weighted by Sigma^-1 (the block-jackknife covariance), where f_obs are REAL f-stats from the AADR f2 cache (' + F2_500 + ' or the 9-pop fixture) and f_pred(theta) is a FIXED small graph topology (a few nodes, 1-2 admixture events) -> nonlinear in theta (edge drifts >=0, mixtures in [0,1]). Pick a topology with a KNOWN/checkable optimum so convergence correctness is verifiable. The "huge batch" = N instances (multistart perturbations + a few real sub-topologies). Reuse the f2 cache + jackknife_cov + small_linalg (host oracle).',
  'BOTH OPTIMIZERS MUST BE GPU-BOUND (the f-stat-sweep lesson): everything on-device; if either pegs 1 CPU core at GPU-0%, that is THE comparison finding (report it, do not hide it). EmulatedFp64{40} for the matmul-heavy objective evals (the GLS residual / f_pred matmuls), native-FP64 carve-out for cancellation, reuse engage_*/emulation_honorable. Reuse the cuBLAS-batched (gemmStridedBatched) / cuSOLVER (potrfBatched/potrsBatched) / CUB seams already proven in cuda_backend.cu. VERIFY any CUB/cuBLAS/cuSOLVER API against the CUDA 13.x docs.',
  'METHODOLOGY: same objective, same N, same convergence tolerance. Metrics PER optimizer at increasing N (e.g. 1k / 10k / 100k / 1M fits): WALL-CLOCK to converge all N, GPU util% (sampled nvidia-smi — the GPU-bound proof for BOTH), objective evals, fraction of instances reaching the true optimum (convergence quality), peak VRAM. Report the crossover + which wins at the huge-batch end + WHY.',
  'SINGLE-GPU --device 0 (multi-GPU PARKED). This is a SPIKE: a bench TU (e.g. tests/reference/bench_optimizers.cu) + a comparison doc — NOT the qpGraph build; the WINNER becomes qpGraph optimizer later. Do not regress the existing suite. Box ' + SSH + '; nvcc -> ' + PATHENV + '. REAL AADR only, no synthetic.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout -- . / git clean -fd. On ANY failure ' + STASH + ' "wip:optimizer-spike-FAILED-<reason>" + HALT. NON-trivial/structural blocker -> STOP + report. Classify minor vs bad. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Design the objective + both optimizers + methodology (GATE)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','objective','idea1_design','idea2_design','gpu_bound_both','methodology','cuda13_apis','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff both optimizers can be built GPU-bound on a real-f2-grounded objective with a checkable optimum, reusing the existing seams. false -> structural blocker' },
    objective: { type: 'string', description: 'the real-f2-grounded small-graph nonlinear box-constrained GLS objective: the topology, theta params + bounds, f_obs from which real f2, the checkable optimum, how N instances are formed' },
    idea1_design: { type: 'string', description: 'the batched-sequential-fleet optimizer: the per-instance bounded quasi-Newton/L-BFGS-B (or projected-gradient), the SIMT-over-fleet execution, per-instance state, the device-batched objective+gradient' },
    idea2_design: { type: 'string', description: 'the batched-population optimizer: CMA-ES or DE, population lambda, the parallel per-generation eval, the update' },
    gpu_bound_both: { type: 'string', description: 'why EACH is GPU-bound (no host per-instance/per-iteration hot loop); the CPU-bound failure mode each had to avoid' },
    methodology: { type: 'string', description: 'the comparison: N scale points, the metrics (wall-clock, GPU util, evals, convergence fraction, VRAM), the fairness controls (same objective/tol)' },
    cuda13_apis: { type: 'string', description: 'the cuBLAS/cuSOLVER/CUB/cuRAND APIs used, VERIFIED against the CUDA 13.x docs (e.g. gemmStridedBatched, potrfBatched, curand for CMA-ES sampling)' },
    blocker: { type: 'string', description: 'if NOT feasible: the structural blocker + options' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are designing the optimizer SPIKE (verify-before-implement; NO code changes). READ docs/research/qpgraph-gpu-design.md + new-tools-roadmap.md (the proposed optimizer + the CPU-bound-prevention), the reusable seams (cuda_backend.cu gemmStridedBatched :3310 / potrfBatched :3342 / potrsBatched :3370; src/core/internal/small_linalg.hpp; jackknife_cov; the f2 cache; the sweep CUB usage as the batched-kernel template). Design: (1) a small REAL-f2-grounded admixture-graph-shaped nonlinear box-constrained GLS objective with a checkable optimum; (2) IDEA 1 (batched sequential fleet quasi-Newton/L-BFGS-B); (3) IDEA 2 (batched population CMA-ES/DE); (4) why each is GPU-BOUND; (5) the comparison methodology (N scale points + metrics). VERIFY the cuBLAS/cuSOLVER/CUB/cuRAND APIs against the CUDA 13.x docs. Return the structured design. If a fair GPU-bound comparison is infeasible, feasible=false + blocker. Do NOT implement.', STD,
].join('\n'), { schema: DESIGN_SCHEMA, label: 'design:optspike', phase: 'Design the objective + both optimizers + methodology (GATE)' })
if (design === null) { log('--- design died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- optimizer spike STRUCTURAL (defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('spike feasible; objective: ' + String(design.objective).slice(0,120))

phase('Implement both GPU optimizers + the batched objective + harness + build')
const fixer = await tryAgent([
  'You are implementing the optimizer spike per this design (BOTH optimizers GPU-bound; real-f2 objective; a bench TU):\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build (' + BUILD + ').', '',
  'IMPLEMENT: the real-f2-grounded batched objective (f_obs from real AADR f2, f_pred(theta) for the fixed small topology, the GLS residual + gradient, device-batched over N) + IDEA 1 (batched sequential fleet quasi-Newton/L-BFGS-B, on-device) + IDEA 2 (batched population CMA-ES/DE, on-device, curand sampling) + a bench TU (tests/reference/bench_optimizers.cu) that runs BOTH at increasing N + samples nvidia-smi. Reuse gemmStridedBatched/potrfBatched/CUB/small_linalg; EmulatedFp64 for the matmul evals. Build (Release) + do not regress STEPPE_THOROUGH ctest. SANITY: both optimizers CONVERGE to the checkable optimum on a small N; a medium-N run is GPU-bound (you SHOULD see high GPU util). Report files added, both optimizer impls, the build, the small-N convergence check, your nvidia-smi observation. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:optspike', phase: 'Implement both GPU optimizers + the batched objective + harness + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for the optimizer spike. Accumulated edits (do NOT clean/revert; git stash only if forced). Reach a CLEAN Release build + green ctest (no regression), patching only trivial -Werror / CMake / CUB-cuBLAS-cuSOLVER-cuRAND API misuse (VERIFY vs the CUDA 13 docs). DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 5x. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement both GPU optimizers + the batched objective + harness + build' })

phase('Run the comparison + GPU-bound proof for both + doc + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','both_converge','idea1_gpu_util','idea2_gpu_util','wallclock_table','winner_at_scale','both_gpu_bound','no_synthetic','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if BOTH optimizers converge to the checkable optimum on real-f2 objectives, BOTH were benchmarked at increasing N with nvidia-smi util captured, the comparison + a clear winner-at-huge-N are reported, no regression, single-GPU, no synthetic' },
    fail_severity: { type: 'string' },
    both_converge: { type: 'boolean' },
    idea1_gpu_util: { type: 'string', description: 'Idea 1 (batched fleet) measured GPU util range at large N + whether GPU-bound' },
    idea2_gpu_util: { type: 'string', description: 'Idea 2 (population) measured GPU util range at large N + whether GPU-bound' },
    wallclock_table: { type: 'string', description: 'the comparison table: per optimizer, per N, wall-clock + evals + convergence fraction + VRAM' },
    winner_at_scale: { type: 'string', description: 'which wins at the huge-batch end + WHY (wall-clock + convergence quality + GPU util); the recommendation for qpGraph optimizer' },
    both_gpu_bound: { type: 'boolean', description: 'were BOTH proven GPU-bound (or is the finding that one is CPU-bound?)' },
    no_synthetic: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the comparison result + the qpGraph optimizer recommendation; on FAIL the blocker' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the optimizer spike (adversarial; hold the GPU-BOUND bar — if either optimizer is CPU-bound that is a key finding to REPORT, not hide). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — both optimizers on-device, real-f2 objective, no synthetic, no regression. (2) ' + BUILD + ' — ctest green. (3) RUN THE COMPARISON on box5090: both optimizers at increasing N (1k..1M) on the real-f2 objective, sampling nvidia-smi for EACH; confirm both converge to the checkable optimum; capture wall-clock + GPU util + evals + convergence fraction + VRAM per optimizer per N. Report which is GPU-bound + which wins at the huge-batch end + why. (4) single-GPU, real AADR. PASS if both converge + are benchmarked with util captured + a clear comparison/winner is reported. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new bench/source/cmake/doc files (NEVER git add dot; never aadr/ atlas_results/), write docs/research/optimizer-comparison.md (the methodology + the wall-clock/util/convergence tables + the winner + the qpGraph recommendation), commit (spike(optimizer): batched-sequential-fleet (L-BFGS-B) vs batched-population (CMA-ES/DE) on the GPU — real-f2 graph-GLS objective, N up to <max>; <winner> wins at scale (util X% vs Y%); recommendation for qpGraph) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:optimizer-spike-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:optspike', phase: 'Run the comparison + GPU-bound proof for both + doc + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ optimizer spike ' + verdict.commit_hash + ' — winner@scale: ' + String(verdict.winner_at_scale).slice(0,160))
else log('--- optimizer spike FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { design, verdict }
