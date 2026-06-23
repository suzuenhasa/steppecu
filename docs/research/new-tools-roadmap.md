# New-tools roadmap — qpfstats / qpGraph / DATES

Status: research only. Branch `phase2-fit-engine`. Companion to
`code-inventory-for-new-tools.md` + the three per-tool GPU-design docs.

THE GOVERNING LESSON (every design here obeys it): we shipped a CPU-bound f-stat
sweep by accident (host enumeration/filter/IO at GPU-0%), rolled it back, and
rebuilt it GPU-bound. Each design below names its CPU-bound failure mode and
designs it out ON-DEVICE. The two in-tree defenses these stand on are (a) the
on-device unrank + CUB compaction + bounded radix-sort sweep
(`cuda_backend.cu:1470`) and (b) the model-BATCHED one-dispatch-per-bucket fit
with the `batched_dispatch_count` proof (`cuda_backend.cu:3075+`).

## Where these sit in the larger sequence

Per the build-sequence memory: (1) FINISH the fit-engine backend, (2) THEN
CLI+Python bindings for what exists, (3) THEN new standalone stats each WITH its
own CLI/bindings. These three tools are step (3). They should be scheduled AFTER
the fit-engine backend is finished and the existing tools' CLI/bindings are
solid (the scaffold they reuse).

## Per-tool summary

| Tool | Effort | Reuse/New | Reads f2 cache? | Big new machinery | CPU-bound risk designed out as |
|------|--------|-----------|-----------------|-------------------|-------------------------------|
| qpfstats | MED-HIGH | ~55/45 | YES | shared-factor batched solve + NaN-block downdate kernel | per-block solve -> ONE shared cuSOLVER factor + potrsBatched (no host block loop) |
| qpGraph | HIGH | ~45/55 | YES | graph/path-algebra data model + a GENERAL on-device nonlinear optimizer + constrained edge QP | in-kernel optimizer + basis-once-resident + potrfBatched (no host optimizer/host objective round-trip) |
| DATES | HIGH | ~40/60 | NO (genotype path) | cuFFT autocorrelation stage (+ new CUDA::cufft link) + LD weight/scatter/grid kernels + decay fitter | ALDER FFT reformulation: O(M^2) pairs -> O(G log G) on-device autocorrelation |

## Dependencies

```
   [existing: f2 cache, jackknife_cov, cuSOLVER potrf/potrs(+Batched),
    assemble_f3/f4, dstat_block_reduce, decode_af, sweep+CUB, precision seam,
    CLI/binding scaffold]
         |                          |                         |
         v                          v                         v
     qpfstats                    qpGraph                     DATES
   (assemble + a            (f3 basis-once +            (decode front-end +
    shared-factor            general nonlinear            cuFFT autocorr +
    batched solve)           optimizer + path             decay fitter)
                             algebra)
                                  ^                          ^
                                  |   shared, build-once     |
                                  +--- GENERAL ON-DEVICE -----+
                                       NONLINEAR OPTIMIZER
```

Hard dependencies:
- **qpfstats** depends on f4/f3 being done (it consumes the same numerator +
  jackknife seams). Lowest gap-blocked; depends on NOTHING new — only a new
  arrangement of existing cuSOLVER primitives + a downdate kernel.
- **qpGraph** depends on a NEW general bounded nonlinear optimizer (on-device)
  that does not exist (gap #1), plus a new graph/path-algebra data model (gap #4).
  Its per-edge linear algebra reuses existing seams.
- **DATES** depends on a NEW `CUDA::cufft` link + cuFFT autocorrelation stage
  (gap #2), a NEW LD/per-SNP-pair kernel family (gap #3), and a nonlinear decay
  fitter (gap #1, but tiny by reformulation). It does NOT touch the f2 cache.

SHARED OPPORTUNITY: the general bounded nonlinear optimizer needed by qpGraph
(over admixture weights) and by DATES (over the decay rate) is the SAME engine —
build it ONCE, general, validated as a CpuBackend oracle first, then ported to a
device kernel. qpGraph's need is the more demanding (flat likelihood surfaces,
multistart); DATES' need is trivial (4 params, 1-D-in-lambda). Building the
optimizer for qpGraph makes DATES' fitter nearly free.

## Recommended build order (after the fit-engine backend + existing CLI/bindings)

1. **qpfstats FIRST.** Lowest new-machinery; reuses the most; introduces NO new
   gap (no optimizer, no FFT, no graph). It exercises the shared-factor batched
   cuSOLVER solve + the NaN-block downdate kernel, both of which harden the
   solve/precision seam the others lean on. Its downdate kernel is the only hard
   new piece and is self-contained.
2. **qpGraph SECOND.** It forces the GENERAL on-device nonlinear optimizer into
   existence — the single most reusable new asset — plus the graph/path-algebra
   data model. Build the optimizer as a CpuBackend oracle first (against AT2),
   then port to the device kernel. Single-graph fit first, golden-gated; topology
   search deferred.
3. **DATES THIRD (or alongside qpGraph's optimizer).** It is the highest-new but
   its fitter is trivial once the qpGraph optimizer exists; its real new surface
   is the cuFFT autocorrelation + the LD kernels, which are orthogonal to the
   other two (no f2-cache coupling). It can proceed in parallel with qpGraph's
   path-algebra work once the optimizer lands.

WHY this order: qpfstats de-risks the solve/precision seam with no new gaps;
qpGraph creates the optimizer DATES reuses; DATES' independent cuFFT surface can
overlap. It is also the ascending-effort order (MED-HIGH -> HIGH -> HIGH), which
front-loads the cheapest win.

## Open decisions (need a user call before implementing)

1. **qpGraph — the optimizer choice AND the reference.** The R port uses
   `optim(method='L-BFGS-B')` with `numstart=10` restarts; the DReichLab C
   `qpGraph.c` has NO named library optimizer (custom routine — VERIFIED no
   lbfgs/nelder/powell symbols). DECIDE: (a) which reference produces the goldens
   (R vs C), and (b) whether the on-device optimizer is projected-gradient
   (simpler, more deterministic, slower) or in-kernel L-BFGS-B (matches the R,
   fiddly). RECOMMEND: prototype as a CpuBackend oracle against AT2 first to pin
   the converged-best score/weights, THEN choose the device kernel.

2. **qpfstats — `doscale` and the regularization path.** VERIFIED the C
   `qpfstats.c` defaults `doscale=YES` (lambdascale rescales f2/f2sig before the
   solve) and `lsqmode=NO`, while the R `qpfstats()` path does ridge only. DECIDE
   which the goldens use and match byte-for-byte (it materially changes `b`).
   Also: the all-NaN-block -> `b=0` policy (drop vs zero+warn).

3. **DATES — the per-SNP weight form AND the FFT-vs-binning detail.** (a) The
   DATES eLife methods describe BOTH a population allele-freq weight
   `delta=p_A-p_B` (computable from `decode_af` Q) and a per-sample likelihood
   form `K_i=(a_i-b_i)/L_i` (would need a per-sample kernel). DECIDE which the
   released DATES computes for the population statistic — this is the biggest
   algebra-parity uncertainty and needs the DATES source (NOT on box5090; clone
   `github.com/priyamoorjani/DATES`). (b) Confirm the d-bin -> FFT-lag mapping
   (0.05 cM grid vs 0.1 cM output bins — re-bin lags or use the grid directly)
   and whether to use cuFFT autocorrelation vs an on-device distance-binning
   reformulation. RECOMMEND cuFFT (the lib is present in CUDA 13.0, verified) with
   a host-FFTW CPU oracle for the within-codebase parity pin.

4. **Shared optimizer scope.** DECIDE whether to invest in one general bounded
   nonlinear optimizer (serving both qpGraph and DATES) vs two purpose-built
   fitters. RECOMMEND the shared general engine — qpGraph needs it regardless, and
   it makes DATES' fitter nearly free.

## Cross-cutting constraints (all three)

- PRECISION: `EmulatedFp64{40}` default for matmul-heavy work; native-FP64
  carve-out for cancellation-sensitive combines + small ill-conditioned solves;
  promotable via `set_solve_precision` / `CusolverMathModeScope` (commit 8d4f22f).
  cuSOLVER has NO FP64-emulated math mode in CUDA 13.0 — the seam degrades to
  native today.
- MULTI-GPU: PARKED. Design single-GPU `--device 0`. Note where a future shard
  goes (qpfstats: blocks; qpGraph: (graph,restart) like `plan_model_shards`;
  DATES: chromosomes) but do NOT design it.
- VALIDATION: REAL AADR only, no synthetic (including any throughput number). The
  AADR TGENO goldens are known-corrupt — regenerate via convertf>=8.0.0 TGENO->PA
  then the AT2 reference.
- API VERIFICATION: any CUB/cuBLAS/cuSOLVER/cuFFT call must be checked against the
  CUDA 13.x / CCCL 3.x docs before coding (the box has CUDA 13.0 + CCCL 3.x).

## Verification basis

Every steppe `file:line` in these docs was read at HEAD (`605ef6f`). The gaps
(no optimizer/FFT/LD/graph) were confirmed absent by grep over `src/`+`include/`.
box5090 confirmed to hold `qpfs.pdf`, `qpfstats.c`, `qpGraph.c`, and `libcufft.so`
(CUDA 13.0); NO DATES/ALDER source on the box. Two parity divergences were
discovered by reading the C source directly and are flagged as open decisions:
qpGraph's R-vs-C optimizer, and qpfstats' C-default `doscale=YES`.
