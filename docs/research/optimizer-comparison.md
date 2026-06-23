# qpGraph optimizer comparison — batched-sequential fleet (L-BFGS-B) vs batched-population (CMA-ES/DE)

Status: SPIKE verdict (de-risking, NOT the qpGraph build). Single-GPU (device 0,
RTX 5090 Blackwell sm_120, CUDA 13.0). REAL AADR data only. The WINNER becomes the
qpGraph optimizer (ROADMAP §6; `qpgraph-gpu-design.md` flagged the optimizer as the
highest-risk / most-reusable new asset).

Bench TU: `tests/reference/bench_optimizers.cu` (manual, NOT a ctest gate — like
`bench_rotation_1240k.cu`). Companion design write-up: `qpgraph-optimizer-spike.md`.
This document is the independently-reproduced comparison: methodology, the
wall-clock / GPU-util / convergence tables, the winner, and the qpGraph recommendation.

## Methodology

Same objective, same N scale points, same convergence tolerance for both optimizers.
The host computes the basis ONCE (real-f2 -> f_obs/Qinv via block-jackknife on the
`small_linalg` oracle) and pins the checkable optimum on a dense host grid. Each
optimizer is then ONE batched kernel launch per N (the whole multistart x iteration /
generation loop lives in-kernel); only the per-instance `{w, score}` cross back. GPU
util is sampled with `nvidia-smi -lms 20` in a parallel process during the run.

Metrics per optimizer per N: wall-clock to converge all N (pure kernel time, bracketed
by `cudaDeviceSynchronize`), objective evals, fraction of instances reaching the
host-pinned optimum (convergence quality, tol on BOTH `|w-w*|` and `|score-score*|`),
dispatch count, peak VRAM, and the GPU-util histogram (the GPU-bound proof for BOTH).

### The objective (real-f2-grounded, no synthetic)

Source: the committed 9-pop AADR f2 fixture
`tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin` (P=9, nb=710 jackknife blocks)
— the SAME real tensor `test_qpadm_parity` / `test_qpwave_parity` use. Topology (fixed,
1 admixture event): base/outgroup = Mbuti(3); sources Turkey_N(2), Iran_GanjDareh_N(5);
admixed leaf CordedWare(1) = `w*Turkey_N + (1-w)*Iran`, plus 4 drift edges fit by the
inner GLS. `theta = w in [0,1]`. Objective = `res' Qinv res` with the inner GLS-optimal
edge lengths; `f_obs` = the block-size-weighted outgroup-f3 vector
`f3(C;i,j)=0.5*(f2(C,i)+f2(C,j)-f2(i,j))`; `Qinv` = inverse of the f3 block-jackknife
covariance. The inner SPD solve + the GLS quadratic form are native FP64 (the
cancellation-sensitive carve-out).

Checkable optimum (independently reproduced host grid-pin): **w\* = 0.829051,
score\* = 1.971180** — a smooth UNIMODAL interior minimum (score 26.33 @ w=0 -> 1.97
@ w\* -> 2.85 @ w=1).

### The two optimizer shapes (both fully on-device, one thread per instance)

- **IDEA 1 — batched-sequential fleet.** Per-instance bounded projected-Newton (the
  nadmix=1 reduction of L-BFGS-B: finite-difference gradient + 3-point curvature,
  projected trust-clamped step, backtracking). The FLEET (N) is the parallel axis;
  ~4 objective evals/step.
- **IDEA 2 — batched-population.** Per-instance CMA-ES-flavored 1-D self-adaptive
  Gaussian search (lambda=12, mu=6, `curand` Philox device sampling, in-register sort,
  weighted recombination). Derivative-free.

## Results — INDEPENDENTLY REPRODUCED (verdict run, device 0; tol=1e-3; maxit=200)

| OPT | N | wall (ms) | obj evals | conv frac | dispatches | median \|w-w*\| |
|-----|---|-----------|-----------|-----------|------------|-----------------|
| IDEA1 | 256       | 1.02   | 7,386       | 100.00% | 1 | 2.04e-08 |
| IDEA2 | 256       | 15.10  | 165,184     | 100.00% | 1 | 2.26e-07 |
| IDEA1 | 1,000     | 1.10   | 28,405      | 100.00% | 1 | 2.04e-08 |
| IDEA2 | 1,000     | 15.95  | 646,888     | 100.00% | 1 | 2.14e-07 |
| IDEA1 | 10,000    | 1.14   | 287,316     | 100.00% | 1 | 2.04e-08 |
| IDEA2 | 10,000    | 16.27  | 6,467,560   | 100.00% | 1 | 2.16e-07 |
| IDEA1 | 100,000   | 3.15   | 2,870,817   | 100.00% | 1 | 2.04e-08 |
| IDEA2 | 100,000   | 47.05  | 64,549,732  | 100.00% | 1 | 2.18e-07 |
| IDEA1 | 1,000,000 | 24.00  | 28,696,234  | 100.00% | 1 | 2.04e-08 |
| IDEA2 | 1,000,000 | 364.65 | 645,335,068 | 100.00% | 1 | 2.17e-07 |

Both reach the host-pinned optimum at **100% of instances at every N** (also verified at
the tighter tol=1e-5: 100% for both, IDEA2 then tightening to ~1.4e-8). The
GPU-optimizer w matched to the independent host grid-pin to ~1e-8 (IDEA1) / ~2e-7
(IDEA2) confirms the device objective is bit-faithful to the host oracle.

### GPU-bound proof (INDEPENDENT `nvidia-smi -lms 20` capture, device 0)

Util histogram over the full sweep: **>=90%: 32 samples | 50-89%: 0 | 10-49%: 0 |
<10%: 52** (84 samples total). Peak util 100%, peak VRAM delta 532 MiB (1M instances),
1 dispatch per optimizer per N. There is NO sample in the 10-89% mid-band — the device
is either pinned >=90% (the optimizer kernels) or idle <10% (the host basis-build +
readback between launches). **BOTH optimizers are GPU-bound; NEITHER is CPU-bound.**

The process pegs ~99% of one CPU core overall, but that is the SINGLE-THREADED host
grid-pin (a 62,001-point dense objective scan) + the per-N readback/convergence loops
over up-to-1M-element arrays — bench scaffolding, NOT the optimizer. The optimizer
kernel timings (bracketed by `cudaDeviceSynchronize`) are pure GPU time, and the GPU
sits at 100% during exactly those phases. (Independently confirmed: the <10% GPU
samples coincide with the host-CPU-busy phases.)

## Verdict — IDEA 1 (batched-sequential fleet) WINS, no crossover

- ~15x faster wall-clock at every N (24.0 ms vs 364.7 ms at 1M); the gap WIDENS at
  tighter tol (~18x at tol=1e-5).
- ~22x fewer objective evals (gradient/curvature converges in ~7 iters vs the
  population's ~54 generations x 12 candidates), with an early-exit on the
  projected-gradient-norm proxy.
- Tighter final w (1e-8 vs 2e-7) and lower VRAM.
- No crossover: IDEA1 dominates from N=256 to N=1M. The expected "population wins the
  huge-batch end via N*lambda parallelism" did NOT materialize, because at the 9-pop
  spike scale BOTH already saturate the GPU on the N axis alone, so IDEA2's extra evals
  buy nothing — they only cost ~22x more work to reach the same optimum.

Why IDEA1 wins here: the real, well-identified objective is SMOOTH and UNIMODAL in w
(the design's flat/degenerate-likelihood concern did NOT bite), so the gradient is
informative and finite-difference noise is not a problem.

### Caveats carried into the qpGraph build

1. **nadmix>1.** This spike is nadmix=1 (w 1-D), where L-BFGS-B collapses to projected
   Newton. Real qpGraph (multiple admixture events) needs the full L-BFGS m-history +
   projected line-search; re-bench at nadmix=2..4 before locking it in.
2. **IDEA2 mapping.** IDEA2 ran the lambda population as a SERIAL in-thread loop (one
   thread per instance), not lambda-across-threads. The eval-count metric is
   mapping-independent and at N>=occupancy a lambda-parallel mapping would not reduce
   total work, so the IDEA1 eval win is robust — but a thread-cooperative population
   mapping is the fair alternative to bench if IDEA2 is ever revisited.
3. **Precision seam (not exercised here).** The in-thread spike objective is too small
   (nedge<=4) to hit the EmulatedFp64{40} GEMM seam; at production graph sizes the
   `ppwts/cc` assembly should route through `cublasDgemmStridedBatched` +
   `engage_f2_precision`, and the inner SPD solve keeps the native-FP64 carve-out
   (promotable via `cusolverDnSetMathMode` at 1M-solve scale).
4. **Ground-truth cross-check.** w\* should additionally be cross-checked against AT2
   `qpgraph()` on the same topology before trusting convergence-quality at scale.

## Recommendation for the qpGraph build

Adopt the **IDEA 1 batched-sequential fleet** (per-instance bounded quasi-Newton, fleet
= the parallel axis, the whole optimizer in-kernel) as the qpGraph optimizer SHAPE.
Keep IDEA 2 (CMA-ES/DE) as the flat-surface fallback: run IDEA1 by default and escalate
stalled instances (the per-instance done/stall flag already distinguishes them) to
IDEA2. Wire the EmulatedFp64 GEMM seam + the AT2 `qpgraph()` w\* cross-check at the
production build, not the spike.

Run: `./build-rel/bin/bench_optimizers [fixture] [Ncsv] [maxit] [tol]` (defaults: the
9-pop fixture, `1000,10000,100000,1000000`, maxit=200, tol=1e-3).
