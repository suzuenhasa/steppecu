# qpGraph optimizer spike — batched-sequential fleet vs batched-population

Status: SPIKE (de-risking, not the qpGraph build). Single-GPU (device 0). REAL AADR
data only. The WINNER becomes the qpGraph optimizer (ROADMAP §6;
`qpgraph-gpu-design.md` flagged the optimizer as the highest-risk / most-reusable new
asset).

Bench TU: `tests/reference/bench_optimizers.cu` (manual, NOT a ctest gate — like
`bench_rotation_1240k.cu`). Box: box5090 (2x RTX 5090 Blackwell sm_120, CUDA 13.0),
Release build.

## The objective (real-f2-grounded, no synthetic)

Source: the committed 9-pop AADR f2 fixture
`tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin` (P=9, nb=710 jackknife blocks)
— the SAME real tensor `test_qpadm_parity` / `test_qpwave_parity` use.

Topology (fixed, 1 admixture event): base/outgroup = Mbuti(3). Sources Turkey_N(2)
[Anatolian farmer] and Iran_GanjDareh_N(5) [Caucasus/steppe-related]. Admixed leaf
Czechia_EBA_CordedWare(1) = `w*Turkey_N + (1-w)*Iran` through one admixture node, plus
4 drift edges. `theta = w in [0,1]`; the >=0 drift edges are fit by the inner GLS.

Algebra (steppe's exact `assemble_f3_triples_gather` identity): fit the npair=6
outgroup-f3 vector `f3(C;i,j)=0.5*(f2(C,i)+f2(C,j)-f2(i,j))`, C=Mbuti. `f_obs` = the
block-size-weighted total of this f3 vector. `Sigma` = the f3 block-jackknife covariance
Q (diag_f3=1e-5), inverted to Qinv on the host oracle (`small_linalg::inverse`).
`f_pred(w)` = `fill_pwts(w) -> ppwts_2d [6 x 4]` (linear in edge lengths, NONLINEAR in w
via the product-of-(w,1-w) lineage overlaps); the inner edge lengths are the GLS-optimal
solve of `cc = ppwts' Qinv ppwts` (4x4 SPD, ridge fudge=1e-4 trace-scaled). Objective =
`res' Qinv res`, `res = f_obs - ppwts*bl`. A nonlinear box-constrained least-squares with
the inner linear GLS folded in.

Checkable optimum (host grid-pinned, and independently verified by a NumPy prototype):
`w* = 0.829051`, `score* = 1.971180`. A smooth UNIMODAL minimum (score 26.33 @ w=0 ->
1.97 @ w* -> 2.85 @ w=1). w*~0.83 means CordedWare ~= 83% Turkey_N-related / 17%
Iran-related in this 3-leaf graph — a well-constrained, identifiable optimum on the real
data.

The fleet = N multistart perturbations of the SAME objective (deterministic per-instance
initial w0). `f_obs` and `Qinv` are computed ONCE on the host and shared (basis-once-
resident); only the per-instance optimizer state differs. This is the S8 many-small-fits
envelope; N scales as pure batch rows.

## The two optimizer shapes (both fully on-device)

- IDEA 1 — batched sequential fleet. One CUDA thread per instance; each runs its OWN
  bounded projected-Newton / secant quasi-Newton on the 1-D w in [0,1] (the nadmix=1
  reduction of L-BFGS-B: a finite-difference gradient + 3-point curvature, projected
  trust-clamped step, backtracking). The FLEET (N) is the parallel axis. Per-step =
  ~4 objective evals.
- IDEA 2 — batched population. One thread per instance; each runs a CMA-ES-flavored
  1-D self-adaptive Gaussian population search (lambda=12, mu=6): per generation sample
  lambda candidates `w_k = clamp(m + sigma*z_k)`, `z_k ~ N(0,1)` via `curand`
  (Philox device API, seeded per instance), rank by score (in-register insertion sort),
  weighted-recombine the mu best into the new mean, adapt sigma. Derivative-free. The
  population (N*lambda) is the extra parallel axis.

Both: the WHOLE multistart x iter / generation loop lives in ONE kernel launch; the host
computes the basis once, uploads f_obs/Qinv once, and reads back only the per-instance
`{w, score}`. The inner GLS solve + the GLS quadratic form are native FP64 (the
cancellation-sensitive carve-out, mirroring `dev_chisq_of_core`).

## Results (single GPU, device 0; tol=1e-3 on |w-w*| AND |score-score*|; maxit=200)

| OPT | N | wall (ms) | obj evals | conv frac | dispatches | median \|w-w*\| |
|-----|---|-----------|-----------|-----------|------------|-----------------|
| IDEA1 | 256       | 1.02   | 7,386       | 100.00% | 1 | 2.0e-08 |
| IDEA2 | 256       | 15.05  | 165,184     | 100.00% | 1 | 2.3e-07 |
| IDEA1 | 1,000     | 1.10   | 28,405      | 100.00% | 1 | 2.0e-08 |
| IDEA2 | 1,000     | 15.96  | 646,888     | 100.00% | 1 | 2.1e-07 |
| IDEA1 | 10,000    | 1.14   | 287,316     | 100.00% | 1 | 2.0e-08 |
| IDEA2 | 10,000    | 16.31  | 6,467,560   | 100.00% | 1 | 2.2e-07 |
| IDEA1 | 100,000   | 3.15   | 2,870,817   | 100.00% | 1 | 2.0e-08 |
| IDEA2 | 100,000   | 46.84  | 64,549,732  | 100.00% | 1 | 2.2e-07 |
| IDEA1 | 1,000,000 | 23.94  | 28,696,234  | 100.00% | 1 | 2.0e-08 |
| IDEA2 | 1,000,000 | 363.06 | 645,335,068 | 100.00% | 1 | 2.2e-07 |

Both reach the host-pinned optimum at 100% of instances at every N (w matched to ~1e-8
for IDEA1, ~2e-7 for IDEA2; both well inside tol). The matched GPU-vs-host-oracle w*
(to 1e-8) also confirms the device objective is bit-faithful to the host `small_linalg`
oracle.

GPU-bound proof (the f-stat-sweep discipline): with `nvidia-smi -lms 20` during the 1M
run, the util histogram is `>=90%: 35 samples | 50-89%: 0 | 10-49%: 0 | <10%: 43`.
There is NO sample in the 10-89% band — the device is either pinned at >=90% (the
optimizer kernel) or idle <10% (the host basis-build + readback between launches). Peak
util 100%, peak VRAM delta 532 MiB (1M instances). Dispatches = 1 per optimizer per N
(the whole optimizer is one launch). BOTH optimizers are GPU-bound; NEITHER pegs a CPU
core during the kernel phase. (The transient 100% host-CPU is the single-threaded host
grid-pin + readback, NOT the optimizer.)

## Verdict

IDEA 1 (batched-sequential fleet) WINS decisively, and wins MORE as N grows:

- ~15x faster wall-clock at every N (23.9 ms vs 363 ms at 1M).
- ~22x fewer objective evals (gradient/curvature information converges in ~7 iters vs
  the population's ~54 generations x 12 candidates), and it early-exits on the projected-
  gradient-norm proxy.
- Tighter final w (1e-8 vs 2e-7) and lower VRAM (no per-instance lambda candidate batch).
- No crossover: IDEA1 dominates from N=256 to N=1M. The expected "population wins the
  huge-batch end via maximal N*lambda parallelism" did NOT materialize, because at the
  9-pop spike scale BOTH already saturate the GPU (the parallel axis is N >= occupancy
  for either), so the extra lambda parallelism buys IDEA2 nothing — it only spends ~22x
  more evals reaching the same optimum.

Why IDEA1 wins here: this objective is SMOOTH and UNIMODAL in w (the design's
"hardest-part" flat/degenerate-likelihood concern did NOT bite on this real, well-
identified topology), so the gradient is informative and finite-difference noise is not a
problem. IDEA2's derivative-free robustness is insurance against flat surfaces that this
objective does not exhibit.

## Recommendation for the qpGraph build

Adopt the IDEA 1 batched-sequential fleet (per-instance bounded quasi-Newton, fleet =
the parallel axis, the whole optimizer in-kernel) as the qpGraph optimizer SHAPE, with
two carry-forward caveats:

1. nadmix>1: this spike is nadmix=1 (w 1-D), where L-BFGS-B collapses to projected
   Newton. For real qpGraph (multiple admixture events) the thread must carry the full
   L-BFGS m-history (s_k,y_k pairs) + the projected line-search per the design; the
   eval-efficiency advantage of gradient information should grow, but re-bench at
   nadmix=2..4 before locking it in.
2. Flat/degenerate surfaces: keep IDEA2 (CMA-ES/DE) as a fallback for topologies where
   the gradient is uninformative (the design's hardest-part). A production qpGraph could
   run IDEA1 by default and escalate stalled instances to IDEA2 — the per-instance done/
   stall flag already distinguishes them.
3. Precision + ground-truth gates (deferred to the qpGraph build, not the spike): the
   matmul-heavy objective assembly should route through the EmulatedFp64{40} GEMM seam
   (`engage_f2_precision`) once `ppwts/cc` move to `cublasDgemmStridedBatched` at
   production graph sizes (the in-thread spike objective is too small to exercise it);
   the inner SPD solve keeps the native-FP64 carve-out (promotable via
   `cusolverDnSetMathMode` CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH, NEW in CUDA 13.3, at
   1M-solve scale). w* must additionally be cross-checked against AT2 `qpgraph()` on
   box5090 for the same topology before trusting convergence-quality at scale.

Run: `./bench_optimizers [fixture] [Ncsv] [maxit] [tol]` (defaults: the 9-pop fixture,
`1000,10000,100000,1000000`, maxit=200, tol=1e-3).
