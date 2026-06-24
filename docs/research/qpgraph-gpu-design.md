# qpGraph — GPU-shape-first design

Status: research only, no implementation. Branch `phase2-fit-engine`.
Effort: HIGH (~45% reuse / 55% new).

qpGraph = admixture-graph fit. A graph topology maps to expected f-statistics via
path algebra; a NESTED optimization fits it: an OUTER nonlinear search over
admixture weights, an INNER linear edge-length GLS solve, scored under the f3
jackknife covariance. The outer dimension is TINY (`nadmix`, typically 1-6); the
inner is a small SPD solve (`nedge`, typically <40); the data coupling is the
design `ppwts_2d` (`npair x nedge`) and the dense `ppinv` (`npair x npair`).

## 0. The CPU-bound failure mode — designed out FIRST (the lesson)

THE SWEEP-DISASTER ANALOGUE IS ACUTE HERE and is the single biggest design risk.
The naive port of AT2 is the textbook CPU-bound failure: R's `qpgraph` calls
`optim(method='L-BFGS-B')` on the HOST, repeatedly invoking a HOST objective
`optimweightsfun` (fill_pwts + a host solve + a host quadratic form) — an
iterative host optimizer calling a host objective, x `numstart=10` restarts x
potentially thousands of candidate graphs. That pegs one CPU core at GPU-0%,
EXACTLY the rolled-back host-enumeration sweep the project had to rebuild.

FOUR specific traps + prevention:
1. **Host optimizer calling a device objective (the worst).** If L-BFGS-B runs on
   the host and only the objective is a kernel, every iteration is a
   host->device->host round-trip with a launch per eval — launch-latency-bound,
   GPU near-idle. PREVENTION: the ENTIRE optimizer runs ON-DEVICE (in-kernel
   projected-gradient / L-BFGS-B per `(graph,restart)` thread); the host launches
   ONCE per topology-batch and gets back only final scores+weights. The
   multistart x maxit x finite-diff-gradient loop never crosses the seam.
2. **Host graph enumeration for topology search** (the direct sweep-disaster
   repeat). PREVENTION: enumerate/mutate topologies on-device (sweep_unrank-style,
   `qpadm_fit_kernels.cu:697`) or upload a bounded batch of path-tables and keep
   top-K by score via CUB compaction — host never loops over graphs.
3. **Host-side small solves** (the `nedge x nedge` edge solve or the `npair^2`
   ppinv per fit). PREVENTION: `potrfBatched`/`potrsBatched` across the
   `(graph,restart)` batch; `ppinv` computed ONCE and kept resident.
4. **Re-deriving the f3 basis per graph.** `f3_est`/`ppinv` depend only on the
   pop-set, NOT the topology. PREVENTION: compute the basis ONCE, keep
   `f3_est`+`ppinv` resident, reuse across all graphs/restarts.

GPU-BOUND PROOF (mirroring the existing tests): assert a
`batched_dispatch_count`-style counter showing dispatches `<< graphs*restarts`
(one batched launch per topology-bucket, not per fit), and that the host
enumerates/solves nothing on the hot path.

## 1. The algebra (verified vs admixtools R/qpgraph.R + src/cpp_qpgraph.cpp)

THE BASIS (`qpgraph_precompute_f3`, R `qpgraph.R:476-525`). qpGraph fits NOT to f4
but to the `npair = choose(npop,2)` OUTGROUP-f3 vector `f3(base; i, j)`, built from
f2 as `f3_blocks = (f2[,1,] + f2[1,,] - f2)/2`, then `arr3d_to_pairmat(f3[-1,-1,])`.
This is steppe's f3 identity VERBATIM: `assemble_f3_triples_gather_kernel` computes
`0.5*(f2(C,A)+f2(C,B)-f2(A,B))` per block (`qpadm_fit_kernels.cu:645-664`) — set
C=base, `(A,B)` range over `cmb=combn(0:(npop-1),2)`. The covariance is the f3
jackknife `f3_var` with `diag += sum(diag(f3_var))*diag_f3` (`diag_f3=1e-5`),
inverted to `ppinv = solve(f3_var)` — exactly steppe's `jackknife_cov` producing
dense `Q[m x m]` + Qinv, m=npair (`backend.hpp:130`).

THE PATH ALGEBRA (`graph_to_pwts` `qpgraph.R:3`, `fill_pwts` `:106`,
`cpp_fill_pwts` `cpp_qpgraph.cpp:42`). A graph -> a `[nedge x (npop-1)]`
path-weight matrix `pwts`: `pwts[e, leaf]` = summed product of admixture weights
along every root->leaf path through edge `e`. `fill_pwts` is the ONLY nonlinear
step: per path, `pathweight = prod over admixedges of (w or 1-w)`
(`cpp_qpgraph.cpp:54-60`), scattered/summed onto `pwts`. Center on base:
`pwts = pwts[,-baseind] - pwts[,baseind]`. The f3 DESIGN MATRIX is
`ppwts_2d = t(pwts[,cmb1]*pwts[,cmb2])` — a `[npair x nedge]` matrix LINEAR in edge
lengths: `f3_fit = ppwts_2d %*% branch_lengths`.

THE NESTED OPTIMIZATION. INNER (LINEAR, `opt_edge_lengths` `qpgraph.R:148` /
`cpp_opt_edge_lengths` `:18`): given `w` (-> `ppwts_2d`), solve edge lengths by GLS
normal equations: `pppp = ppwts_2d' ppinv`; `cc = pppp ppwts_2d` (nedge x nedge);
`diag(cc) += fudge*mean(diag(cc))` (`fudge=diag=1e-4`); column-scale by
`sc=sqrt(diag(cc))`; `q1 = pppp f3_est / sc`; then UNCONSTRAINED
`solve(cc,q1)/sc` OR CONSTRAINED `qpsolve` (box: lengths >= 0). OUTER (NONLINEAR):
`score = res' ppinv res`, `res = f3_est - f3_fit` (`qpgraph_score` `:165`),
minimized over the `nadmix` admixture weights by the outer optimizer with
`numstart` random restarts.

OPTIMIZER PARITY DIVERGENCE — VERIFIED on box5090. The R port uses
`multistart(..., method='L-BFGS-B', lower=0, upper=1)`, `numstart=10`
(`qpgraph.R:380-383`). BUT `grep -niE 'lbfgs|nelder|simplex|amoeba|powell'`
over the DReichLab C `/workspace/AdmixTools_src/src/qpGraph.c` returns NO matches
— the C source uses its own custom scoring/minimization (the file does have
`score`/`chisq`/`halfscore`/`yscore` plumbing but no named library optimizer).
THIS IS A REAL OPEN DECISION: the R and C qpGraph may converge to the same global
optimum but via different optimizers; confirm which reference produced the
goldens before designing the fitter.

GPU REFORMULATION. The score+gradient evaluation (`fill_pwts -> ppwts_2d ->
cc=ppwts_2d' ppinv ppwts_2d -> solve -> f3_fit -> res' ppinv res`) is the ENTIRE
inner kernel: a sequence of small dense GEMM + one SPD solve + one quadratic
form — IDENTICAL in shape to steppe's `dev_chisq_of_core`
(`vec(E)' Qinv vec(E)`, `qpadm_fit_kernels.cu:370`) and `dev_solve` LU (`:126`).
The outer optimizer is reformulated as a BATCH: one thread/threadblock per
`(graph, restart)` evaluating the objective entirely on-device, with
forward-finite-difference gradients (nadmix+1 objective evals) feeding a
device-side projected-gradient / L-BFGS-B step — so the host NEVER calls back a
host objective.

> **OPTIMIZER DECISION (2026-06-23, settled by the spike `docs/research/optimizer-comparison.md`,
> commit `932d108`):** qpGraph uses the **IDEA-1 batched-sequential fleet** (the on-device
> projected-gradient / L-BFGS-B per `(graph,restart)` thread) as its **single, default optimizer**.
> A fair re-race against an IDEA-2 batched-population (CMA-ES, λ-across-warp-lanes) showed IDEA 1
> wins the production large-N envelope (~15× at 1M, ~9–20× fewer evals, 100% converged); IDEA 2's
> only edge is small-N *occupancy* (≤~1k), **not** robustness (the multimodal-robustness hypothesis
> was a measured null — the real surface is gradient-friendly). **A second optimizer codepath is
> not worth a small-N occupancy win, so IDEA 2 is PARKED** (the CMA bench code remains as the
> comparison record, not productized). The fleet is the standard.

## 2. GPU-first, GPU-bound pipeline (single-GPU --device 0)

The GPU-bound product here is the `(graph x restart x optimizer-iteration)`
batch — the SAME many-small-fits shape as the qpAdm S8 rotation and the f-stat
sweep, both already GPU-bound on box5090.

1. **BASIS (once per pop-set, resident).** From the resident `DeviceF2Blocks`,
   launch `assemble_f3_triples_gather` (`qpadm_fit_kernels.cu:1832`) with C=base
   fixed and the cmb pair list -> per-block f3 tensor `[npair x n_block]` in VRAM;
   then `jackknife_cov` (CudaBackend override) -> dense `f3_var` Q + inverted
   `ppinv` via cuSOLVER potrf/potri (the path S4 already uses). ZERO D2H:
   `f3_est` (npair) + `ppinv` (npair^2) stay resident as the shared fit
   target/weight for every graph and restart.
2. **OBJECTIVE BATCH (the GPU-bound core).** One CUDA thread (small graphs) or
   threadblock-with-cuBLAS (larger) per `(graph g, restart s)` evaluates
   `optimweightsfun` ENTIRELY on-device:
   - `fill_pwts`: a per-thread scatter over the path tables
     (`path_admixedge_table`/`path_edge_table` uploaded ONCE per topology to VRAM;
     tiny int arenas, same shape as the qpAdm per-model index arenas) ->
     `ppwts_2d` in a per-thread/per-block VRAM arena (runtime-sized stride, the
     proven `*_large` arena pattern, `als_large_kernel` `qpadm_fit_kernels.cu:1133`,
     `loo_large_batched_kernel` `:1223`).
   - `cc = ppwts_2d' ppinv ppwts_2d` via `cublasDgemmStridedBatched` across the
     `(g,s)` axis; ridge+scale elementwise; SPD solve via
     `potrfBatched`/`potrsBatched` (the EXACT batched primitive
     `cuda_backend.cu:3342`/`:3370`). For the constrained (nonneg) case, an
     on-device projected/active-set NNLS in the per-thread arena (small nedge).
   - `f3_fit = ppwts_2d %*% branch_lengths`; `score = res' ppinv res` via
     `dev_chisq_of_core` VERBATIM (one quadratic form against resident ppinv).
3. **OPTIMIZER ON-DEVICE.** Each `(g,s)` thread runs projected-gradient /
   L-BFGS-B over its `nadmix` weights in `[0,1]`, gradient by forward
   finite-differences (nadmix+1 on-device objective evals/step); the whole
   multistart x maxit loop runs in-kernel; ONLY the final per-`(g,s)`
   `{score, weights, edge_lengths}` (a few dozen doubles) crosses the host seam.
   `numstart` restarts = just more rows in the batch.
4. **MANY-GRAPH SWEEP (topology screening, deferred).** The `graph` axis is
   enumerated on-device exactly like the f-stat sweep's `sweep_unrank` +
   CUB-compaction keeping bounded top-K by score — host never enumerates graphs.

Host drives ONLY: upload path-tables per distinct topology, launch the batched
objective/optimizer, receive bounded survivors. cuSOLVER
(potrfBatched/potrsBatched), cuBLAS (DgemmStridedBatched, the f2 emulated-FP64
math-mode scope for the well-conditioned GEMMs), CUB (DeviceSelect for the
sweep). A future multi-GPU shard tiles `(graph,restart)` across devices exactly
as `plan_model_shards` tiles models — note only, NOT designed now (5090s have no
P2P; MULTI-GPU PARKED).

PRECISION: `EmulatedFp64{40}` default for the well-conditioned GEMMs
(`cc=ppwts'ppinv ppwts`); native FP64 for the SPD solve (the cancellation /
conditioning carve-out, same policy qpAdm uses), promotable via
`set_solve_precision`. NOTE `diag_f3` (1e-5) and `fudge` (1e-4) have a LARGE
effect on the absolute score (AT2 note) — must match exactly for parity.

## 3. Reuse inventory (all VERIFIED file:line)

- f3 BASIS ASSEMBLY (the exact qpGraph f3 identity): `assemble_f3_triples_gather_kernel`
  `qpadm_fit_kernels.cu:645-664`; launcher `launch_assemble_f3_triples_gather`
  `:1832`; CudaBackend override `cuda_backend.cu:2068` region; host run_f3 path
  `src/core/qpadm/f3.cpp`. Set C=base to get `f3(base; i,j)` for every cmb pair.
- f2 CACHE / device-resident input: `device::DeviceF2Blocks`
  (`f2_device()`/`vpair_device()`, zero-D2H), `device/device_f2_blocks.hpp`.
- JACKKNIFE COVARIANCE (the `ppinv = solve(f3_var)`): `jackknife_cov`
  `backend.hpp:791`, `JackknifeCov{Q, Qinv, m}` `backend.hpp:130` — dense m x m Q
  + SPD inverse (CudaBackend potrf/potri). m = npair.
- THE GLS QUADRATIC FORM (= `qpgraph_score res' ppinv res`): `dev_chisq_of_core`
  `qpadm_fit_kernels.cu:370` — reusable VERBATIM for the score.
- THE SMALL SPD/LINEAR SOLVE (= `opt_edge_lengths` inner solve): device
  `dev_lu_factor`/`dev_solve` `qpadm_fit_kernels.cu:97-126`; host `core::solve` /
  `inverse` / `jacobi_svd` `small_linalg.hpp:145/165/204` (CPU oracle).
- PER-THREAD RUNTIME-SIZED VRAM ARENA (for `ppwts_2d`/`cc`/scratch per
  `(graph,restart)`): the `*_large` arena pattern, `als_large_kernel`
  `qpadm_fit_kernels.cu:1133`, `loo_large_batched_kernel` `:1223`.
- BATCHED MANY-SMALL-FITS HARNESS (structural template for the graph x restart
  batch): `fit_models_batched` `backend.hpp:954` + `cuda_backend.cu:3075+`
  (potrfBatched/potrsBatched, DgemmStridedBatched, per-model index arenas);
  `model_search_core` `plan_model_shards`.
- ON-DEVICE ENUMERATION + CUB COMPACTION (many-topology screen): `sweep_unrank`
  `qpadm_fit_kernels.cu:697-744` + the bounded top-K reservoir
  (`cuda_backend.cu` f4_sweep `:1470`) + `SweepSurvivors` `backend.hpp:176`.
- NONLINEAR-FUNCTIONAL JACKKNIFE TEMPLATE (for any ratio/product of f-stats
  qpGraph reports): the f4ratio explicit-ratio-xtau pattern,
  `f4ratio.cpp:22-31` — NOTE its VERIFIED caveat that it does NOT route through
  `jackknife_cov` (different xtau decomposition); reuse the `est_to_loo`
  replicate seam, write the functional's xtau explicitly.
- PRECISION SEAM: `set_solve_precision`/`CusolverMathModeScope`
  (`backend.hpp:643`, `handles.hpp:535`); fit-engine.md §1.4.
- CLI/BINDING SCAFFOLD: `cmd_qpadm.cpp`/`cmd_rotate.cpp` chain;
  `PopResolver` (`pop_resolver.hpp`); `result_emit`; the nanobind module.

## 4. New machinery (steppe has NONE of this)

steppe's only optimizer is the qpAdm-specific bilinear ALS (`dev_opt_A_core`/
`dev_opt_B_core`, `qpadm_fit_kernels.cu`), NOT reusable for a graph.

1. **THE GRAPH DATA MODEL + PATH ALGEBRA** (`graph_to_pwts` / `fill_pwts`). Parse a
   DAG topology (leaves, root, drift edges, paired admixture edges); precompute
   the path tables (`path_edge_table`, `path_admixedge_table`); the device
   `fill_pwts` kernel that, given weights, scatters `prod-of-(w,1-w)` onto
   `pwts` and forms `ppwts_2d`. The ONE nonlinear, graph-structure-dependent
   piece. (Port from `cpp_fill_pwts` `cpp_qpgraph.cpp:42-72`.)
2. **A GENERAL BOUNDED NONLINEAR OPTIMIZER, ON-DEVICE.** In-kernel
   projected-gradient / spectral-projected-gradient (simpler, more deterministic)
   OR L-BFGS-B (matches the R, fiddlier in-kernel) over the `nadmix` weights in
   `[0,1]`, with forward-finite-difference gradients; whole multistart x maxit
   loop in-kernel. The reusable nonlinear optimizer steppe lacks (DATES needs the
   same engine for its decay fit — build it general). RECOMMEND prototyping as the
   CpuBackend oracle first, then port to the device kernel.
3. **AN ON-DEVICE CONSTRAINED EDGE-LENGTH SOLVER** (`opt_edge_lengths`).
   Unconstrained = the existing `dev_solve` on the ridged/scaled SPD `cc`.
   Constrained (the AT2 default, drift >= 0) = a small device NNLS /
   box-constrained active-set on `cc` (nedge small). AT2 delegates to R's
   `qpsolve`; steppe needs a device equivalent.
4. **THE BATCHED OBJECTIVE/SCORE PIPELINE** binding 1-3: `fill_pwts -> ppwts_2d ->
   cc=ppwts'ppinv ppwts (batched GEMM) -> ridge+scale -> solve -> f3_fit ->
   res' ppinv res`, batched over `(graph,restart)` with runtime-sized VRAM arenas.
5. **(OPTIONAL, later) TOPOLOGY SEARCH** (`find_graphs` analogue): on-device graph
   mutation/enumeration + CUB top-K-by-score — reuses the sweep shape but needs
   graph-mutation operators (add/remove/move admixture edge). Defer past the
   single-graph fit.
6. **PARITY/SCORE GLUE**: `f3basepop`/`baseind` centering, `diag_f3` (1e-5),
   `fudge` (1e-4), `lsqmode` (identity-ppinv) option, the score scaling, a
   `QpGraph` result POD (score, weights, edge lengths, lo/hi from the restart
   spread) + emitter + bindings.

## 5. Effort + build order

HIGH (confirmed by `desirable-features-survey.md` #16 and `standalone-fstats.md`).
~45% reuse / 55% new. The heavy GPU infra is DONE (f3 basis assembly, dense
jackknife covariance + cuSOLVER SPD inverse, the GLS quadratic-form scorer
`dev_chisq_of_core`, the device LU/SPD solve, the runtime-sized arena, the
batched many-small-fits harness, the enumerate+CUB-compact pattern, the precision
seam, the CLI/binding scaffold). The two things steppe has never had are (a) the
graph topology data model + path algebra and (b) a GENERAL bounded nonlinear
optimizer on-device + an on-device constrained edge-length QP — these are the
HIGH part.

BUILD ORDER: single-graph fit GPU-bound first (basis-once + batched objective +
in-kernel optimizer over restarts), golden-gate vs AT2 `qpgraph()` on real AADR;
topology SEARCH (`find_graphs`) deferred as a follow-on layered on the sweep
machinery.

## 6. Risks + parity

PARITY ORACLE: AT2 `qpgraph()` on real AADR (box5090 has the R + the C
`qpGraph.c`). Gate `{score, fitted admixture weights, edge lengths}`. Subtleties:
(1) the outer optimizer with `numstart` random restarts seeded by the reference's
RNG will NOT reproduce per-restart trajectories — gate the converged BEST
score/weights with a tolerance from the restart spread; flat/degenerate
likelihood surfaces make weights non-identifiable (AT2 reports lo/hi from restart
quantiles) — gate the score, treat weights as an interval there. (2) the
constrained QP (`qpsolve`, drift>=0) vs unconstrained toggles AT2's `constrained`
flag — pick the same default and verify the device active-set QP matches on
boundary cases.

HARDEST PART: the on-device general nonlinear optimizer (deterministic
single-stream fixed-op-order in-kernel optimizer with finite-difference gradients
per `(graph,restart)` thread) — robust across the flat likelihood surfaces
qpGraph hits. Projected-gradient is simpler/more-deterministic but slower;
L-BFGS-B matches the R but is fiddly in-kernel.

OPEN QUESTIONS / DECISIONS: (a) WHICH optimizer — and confirm whether the goldens
come from the R (L-BFGS-B) or the C (custom) qpGraph, since they diverge (VERIFIED
above). (b) `fill_pwts` per-path scatter as one-thread-per-`(graph,restart)` vs a
cooperative threadblock (depends on nedge/npath). (c) `numstart` restarts as
batch-rows vs in-kernel loop (memory vs occupancy). (d) `diag_f3` magnitude has a
LARGE effect on the absolute score — must match exactly. (e) `lsqmode` (identity
ppinv) as a cheaper screening mode for topology search.

SPECULATIVE / FLAGGED: in-kernel optimizer convergence at scale is untested (no
steppe precedent for a general optimizer); topology-search graph-mutation
operators are sketched, not designed. Multi-GPU PARKED.

## Sources

PUBLISHED: AT2 R `qpgraph.R` (gh uqrmaie1/admixtools, 909 lines):
`graph_to_pwts:3`, `fill_pwts:106`, `optimweightsfun:120`, `opt_edge_lengths:148`,
`qpgraph_score:165`, `qpgraph():231` with `multistart(...,method='L-BFGS-B',
lower=0,upper=1):382`, `numstart=10:380`, `qpgraph_precompute_f3:476`
(f3 basis from f2_blocks, ppinv=solve(f3_var), diag_f3=1e-5). AT2
`src/cpp_qpgraph.cpp`: `cpp_opt_edge_lengths:18`, `cpp_fill_pwts:42`. DReichLab C
`/workspace/AdmixTools_src/src/qpGraph.c` (VERIFIED on box5090: score/chisq/
halfscore plumbing, NO lbfgs/nelder/powell symbols — custom optimizer, the
R-vs-C divergence flag). `desirable-features-survey.md` #16; `standalone-fstats.md`
(qpGraph HIGH). STEPPE CODE (read at HEAD): `qpadm_fit_kernels.cu:645,370,97-126,
697-744,1133,1223,1832`; `backend.hpp:130,176,728,791,954,643`;
`small_linalg.hpp:145,165,204`; `f4ratio.cpp:22-31`; `cuda_backend.cu:3075,3342,
3370,1470`; `handles.hpp:535`; `device_f2_blocks.hpp`; fit-engine.md §1.4;
`model_search_core.hpp`; `cmd_qpadm.cpp`/`pop_resolver.hpp`/`result_emit`.
