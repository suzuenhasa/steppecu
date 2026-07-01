# DATES — GPU-shape-first design

Status: research only, no implementation. Branch `phase2-fit-engine`.
Effort: HIGH (~40% reuse / 60% new). New standalone tool, genotype-path sibling
of qpDstat Part B.

DATES = admixture dating via the weighted ancestry-LD/covariance decay vs genetic
distance (the ALDER/DATES family). Admixture creates ancestry tracts that
recombination fragments each generation, so the weighted ancestry covariance
between SNP PAIRS decays exponentially with the genetic distance between them; the
decay rate gives the time since admixture.

## 0. The CPU-bound failure mode — designed out FIRST (the lesson)

THREE CPU-bound traps, each designed out:

TRAP 1 (the central one): the naive O(M^2) host double-loop over SNP pairs binned
by genetic distance. At ~1.2M autosomal SNPs that is ~10^12 pairs — a host loop
here pegs one core at GPU-0%, EXACTLY the f-stat-sweep disaster. PREVENTION:
REFORMULATE the algebra (the ALDER FFT). The pairwise double-sum is ALGEBRAICALLY
an autocorrelation of a genetic-map-binned grid array, so it NEVER materializes
pairs — it is O(G log G) cuFFT on-device. The 10^12-pair object is never formed in
any loop, host or device.

TRAP 2: host materialization of Q/V then a host weight/scatter loop. The current
`decode_af` copies Q/V/N to host (`cuda_backend.cu:1215-1219`, VERIFIED) and
`run_dstat` does a host autosome-subset loop (`dstat.cpp:245-255`). At M scale a
host scatter-to-grid loop is a second CPU-bound risk. PREVENTION: keep Q/V
device-resident (a new device-resident decode variant) and do weight+scatter as
on-device kernels; the host never touches per-SNP data.

TRAP 3: a host nonlinear optimizer iterating over the full per-SNP/per-pair data
each step (a fit-in-the-hot-loop trap). PREVENTION: the fit operates ONLY on the
already-reduced `[n_bin]` ~= 1000-point curve, NOT on SNPs/pairs. ~1000 points x
23 fits is genuinely tiny-host work (at fixed lambda, (A,c) are LINEAR — solve a
2x2 — so it reduces to a 1-D search over lambda + Gauss-Newton). This is the
LEGITIMATE small-result-on-host seam, not a CPU-bound loop over data items.
NEVER fold the per-SNP reduction into the optimizer's residual evaluation.

DESIGN-FOR-SCALE NOTE: the envelope is large target panels over full AADR
autosomes; the FFT cost is INDEPENDENT of M (depends only on grid length ~=
genome_cM/0.05), so scaling is flat in SNP count — the opposite of the O(M^2)
trap.

## 1. The algebra (grounded in the papers — NO source on box5090, see Risks)

THE STATISTIC (Loh 2013 ALDER; Chintalapati/Moorjani 2022 DATES). Model:
`A(d) ~= A*exp(-lambda*d) + c`, where `lambda = (t+1)` generations since admixture
(the +1 because mixing begins the generation AFTER admixture; DATES
`A(d) ~ e^-(t+1)d`), `c` an affine background term, fit over `d in [0.45 cM,
100 cM]` (DATES default; the 0.45 cM floor avoids background LD), bin 0.1 cM =
0.001 Morgans.

WEIGHTING: each SNP `x` carries weight `delta(x) = p_A(x) - p_B(x)` (allele-freq
difference between the two reference/source pops) — for the population-level
statistic this is computable DIRECTLY from the `decode_af` Q matrix (NO
per-haplotype ancestry calling). The weighted mean-centered per-SNP value is
`z(x) = (g(x) - g_bar) * delta(x)` for the admixed target g (Loh's weighted-LD
curve `a(d) = [sum_{|x-y|~d} z(x)z(y)] / [normalizer]`).

THE GPU REFORMULATION (the whole point — Loh 2013 "FFT algorithm", reused by
DATES "we implement the FFT for calculating ancestry covariance as described in
ALDER"): the binned curve `A(d)` is a CONVOLUTION/AUTOCORRELATION. ALDER's trick:
do NOT bin SNP-PAIRS by distance (the O(M^2) trap). Instead DISCRETIZE THE MAP
POSITIONS onto a ~0.05-cM grid PER CHROMOSOME: build `W[g] = sum_{SNPs in cell g}
z(SNP)`. Then `A(d=lag*delta) = autocorr(W) at lag = sum_g W[g]*W[g+lag] =
IFFT(|FFT(W)|^2)` — a real-to-complex forward FFT, magnitude-square,
complex-to-real inverse FFT. Also FFT a parallel count/weight-normalizer array for
the per-lag denominator. Cost: O(G log G) per chromosome (G ~ a few thousand),
replacing O(M^2). THIS is the algebra that makes it GPU-bound, not host-O(M^2)-bound.

## 2. GPU-first, GPU-bound pipeline (single-GPU --device 0)

Host drives only a tiny per-chromosome loop (<=22 iterations) + one final
4-parameter fit.

1. **DECODE FRONT-END (reuse, made device-resident).** `read_tile` + `decode_af`
   -> Q/V `[P x M]` for the target + 2 reference pops. CHANGE FROM `dstat.cpp`: do
   NOT copy Q/V to host (the D2H at `cuda_backend.cu:1215-1219`). Add a
   device-resident decode variant so Q/V stay in VRAM as the FFT front-end's
   input. `assign_blocks` gives the SNP->block/chromosome map.
2. **PER-SNP WEIGHT + CENTERED-VALUE KERNEL (new, on-device).** One thread per SNP
   computes `delta(x)=Q[A,x]-Q[B,x]`, the centered target value, and `z(x)`,
   masked by V (finiteness for target+A+B). Pure elementwise, native-FP64
   (cancellation carve-out — delta is a difference of like-magnitude freqs).
   Output: device arrays `z[M]`, `delta[M]`, + the per-SNP grid-cell index from
   genpos (the down-payment already retained, `dstat.cpp:240-254`).
3. **GRID SCATTER KERNEL (new, on-device).** Scatter-add `z` and the
   weight-normalizer into per-chromosome grid arrays `W_chr[g]` (atomicAdd, or a
   sort-by-grid-cell + CUB `DeviceSegmentedReduce` to avoid atomics). Pad each
   chrom's grid to an FFT-friendly length.
4. **cuFFT BATCHED AUTOCORRELATION (new, cuFFT).** `cufftPlanMany` D2Z over all 22
   chromosome grids in ONE batched plan; `|F|^2` elementwise kernel; Z2D inverse
   -> per-lag autocorrelation = `A(d)` numerator + denominator, all in VRAM. The
   device output is a tiny `[n_chrom x n_bin]` matrix. VERIFIED: `libcufft.so`
   (`libcufft.so.12.0.0.61`) ships with CUDA 13.0 on box5090, so D2Z/Z2D is
   available — but VERIFY the `cufftPlanMany` D2Z/Z2D signature + batched-plan
   semantics against the CUDA 13.x cuFFT docs before coding (the API is stable
   across CUDA 12->13 but confirm).
5. **SUM-OVER-CHROMOSOMES** + the `[n_bin]` curve crosses the host seam (~1000
   doubles). LEAVE-ONE-CHROMOSOME-OUT jackknife: form 22 LOO curves by subtracting
   one chrom's contribution from the total (device or trivially host —
   22x1000 doubles).
6. **THE FITTER (tiny-host, NOT a CPU-bound risk).** Nonlinear least squares
   `A(d)=A*exp(-lambda*d)+c` over ~1000 bins x 23 fits (full + 22 LOO). 4 unknowns;
   at fixed lambda, (A,c) are LINEAR (solve a 2x2 normal-equation) -> a 1-D search
   over lambda + Gauss-Newton — microseconds on one core. lambda -> t per fit;
   `mean(t)` and block-jackknife SE over the 22 LOO replicates. NRMSD < 0.7
   goodness gate (DATES default).

Result crossing the seam: `t`, SE, NRMSD — a handful of scalars.

PRECISION: the per-SNP weight/center kernel is native-FP64 (the cancellation
carve-out). The FFT magnitude-square is well-conditioned. The fit is on ~1000
points — native FP64, no promotion needed.

## 3. Reuse inventory (VERIFIED file:line)

- DECODE FRONT-END: `io::GenoReader` + `read_ind(Explicit{pop_union})` +
  `read_snp` + `reader.read_tile` + `ComputeBackend::decode_af` -> `DecodeResult`,
  `dstat.cpp:195-234`; `backend.hpp:615` (decode_af virtual), `:304-315`
  (DecodeResult). The genotype-stat on-ramp is literally run_dstat's front-end.
- GENOTYPE-STAT SEAM / S2 DIVERGENCE: the pattern of a genotype-reading sibling
  that REUSES decode + assign_blocks and diverges into its OWN device kernel,
  never touching the f2 cache — `dstat.cpp:1-31` (doc comment), `backend.hpp:749-781`
  (`dstat_block_reduce` virtual + base-throw idiom to copy for a new `dates_*`
  virtual).
- PER-SNP GENPOS (the down-payment, already paid): `dstat.cpp:240-254` retains
  `snptab.genpos_morgans` per kept SNP in lockstep (VERIFIED `genpos_kept`).
  `SnpTable.genpos_morgans` at `snp_reader.hpp:41`.
- `assign_blocks` (chrom/genpos -> block_id, per-chrom reset, Morgans) +
  `block_ranges`: `block_partition_rule.hpp:162`, `:179`; used `dstat.cpp:267`.
  Gives the per-chromosome segmentation for the leave-one-chrom jackknife and the
  per-chrom FFT batching.
- JACKKNIFE FAMILY: the num/den / ratio block-jackknife shape (`est_to_loo` +
  weighted-mean + xtau variance, long-double accumulation) — `dstat.cpp:70-157`
  (`dstat_jackknife`) and `f4ratio.cpp` (ratio_jackknife). DATES uses the
  leave-one-CHROMOSOME variant — same algebra, chromosome as the block.
- `small_linalg.hpp`: `solve(A,b)` LU-partial-pivot (`:145`) + `jacobi_svd`
  (`:204`) for the 2x2/4x4 normal-equation solve in the Gauss-Newton fitter.
- GPU-BOUND EXEMPLAR (the pattern to copy): `run_fstat_sweep_device` — on-device
  unrank+compute+filter, host drives only the chunk loop, CUB
  `DeviceSelect::Flagged`/`DeviceRadixSort`, free-VRAM chunk sizing —
  `cuda_backend.cu:1470`. CUB already wired (`cuda_backend.cu:60-64`).
- PRECISION SEAM: `EmulatedFp64{40}` default + native cancellation carve-out +
  `capabilities().emulated_fp64_honorable` — fit-engine.md (the precision policy).
  cuBLAS/cuSOLVER linked PRIVATE: `device/CMakeLists.txt:56`.
- CLI/binding scaffold: `cmd_qpdstat.cpp` (the standalone genotype-stat command
  shape) + the nanobind module pattern.

## 4. New machinery (the architecture doc excluded an LD engine by design)

1. A device-resident decode variant (or a flag on `decode_af`) so Q/V stay in VRAM
   instead of the D2H at `cuda_backend.cu:1215-1219`. Small but new.
2. WEIGHT + CENTERED-VALUE KERNEL: per-SNP `delta`, `z`, V-masked. New .cu,
   native-FP64.
3. GRID-SCATTER KERNEL: per-chromosome scatter-add of `z` + normalizer into the
   ~0.05-cM grid from per-SNP genpos. New .cu (atomicAdd or sort-by-cell + CUB
   `DeviceSegmentedReduce`). Padding to FFT lengths.
4. THE cuFFT AUTOCORRELATION STAGE: `cufftPlanMany` batched D2Z + `|F|^2`
   elementwise kernel + Z2D inverse + per-lag bin extraction. NEW LINK
   DEPENDENCY: `CUDA::cufft` (VERIFIED currently NOT linked —
   `device/CMakeLists.txt:56` links only cudart/cublas/cusolver; the lib exists in
   CUDA 13.0). This is the single new third-party surface.
5. THE NONLINEAR EXPONENTIAL FITTER: `A*exp(-lambda*d)+c` least squares with the
   (A,c)-linear / lambda-1-D partial-linearization + Gauss-Newton, reusing
   `small_linalg` solve; NRMSD goodness; `lambda -> (t+1)` inversion. steppe has
   NO general optimizer today (gap #1) — this is new but tiny (operates on the
   ~1000-point curve). Could share the qpGraph general-optimizer engine if built
   first.
6. THE LEAVE-ONE-CHROMOSOME jackknife driver (chromosome as block) wrapping the
   fit — new orchestration, reusing the jackknife algebra.
7. The `run_dates` entry + `DatesResult` + CLI command + binding.

## 5. Effort + ordering

HIGH. ~40% reuse / 60% new. REUSE: decode front-end, assign_blocks/block_ranges,
per-SNP genpos surfacing, the jackknife algebra, small_linalg solve, the
GPU-bound sweep pattern, CUB plumbing, precision seam, CLI/binding scaffold. NEW:
the cuFFT autocorrelation stage + the new `CUDA::cufft` link (the biggest new
surface, load-bearing), the weight/scatter/grid kernels, the device-resident
decode variant, the nonlinear fitter, the leave-one-chrom orchestration. The cuFFT
stage is the load-bearing new component; the fitter is small; the kernels are
moderate. Within the post-Phase-2 ordering (finish backend -> CLI/bindings -> new
standalone stats each WITH CLI/bindings), DATES is built after qpfstats/qpGraph
or alongside, with its own CLI+binding.

## 6. Risks + parity

HARDEST PART: the cuFFT normalization/centering parity — getting the weighted-LD
numerator AND denominator (the per-lag count/weight normalizer that ALSO must be
FFT'd) bit-comparable to DATES, plus the per-chromosome grid padding and the
d-bin->FFT-lag mapping (0.05 cM grid vs 0.1 cM output bins — verify whether DATES
re-bins lags or uses the grid directly).

OPEN QUESTIONS / TO VERIFY (admixtools R + DReichLab C are on box5090, but NO
DATES/ALDER source — `find /workspace -iname '*dates*' -o -iname '*alder*'`
returns EMPTY, VERIFIED this session; would need to clone
`github.com/priyamoorjani/DATES` or the DReichLab ALDER C):
- The EXACT per-SNP weight + centering DATES uses (population-level
  `delta=p_A-p_B` vs the per-sample likelihood `K_i=(a_i-b_i)/L_i` form quoted in
  the eLife methods — the text describes BOTH; the K-form would need a per-sample
  likelihood kernel, not just `decode_af` Q). FLAG: this is the single biggest
  algebra-parity uncertainty.
- Whether DATES weights by `Q=alpha*beta` (ancestry proportions) as an overall
  scale and whether it affects t (it scales A, not lambda — likely date-neutral,
  confirm).
- The exact affine-term + lovalfit (0.45 cM) handling and the `(t+1)` vs `t`
  reporting convention.

PARITY ORACLE: run the released DATES on a REAL AADR target with known references
and match `t +/- SE`; NO synthetic. A CPU oracle in steppe can do the same FFT
(FFTW/host) at small scale for a within-codebase parity pin before the cuFFT path.
MULTI-GPU: PARKED — a future shard would split chromosomes across GPUs (the FFT
batch is already per-chromosome-independent), but design single-GPU now.

## Sources

PAPERS: Loh et al. 2013 (ALDER), Genetics (PMC3606100) — the weighted-LD `a(d)`
statistic, the exponential fit `M*e^(-nd)+K` least squares, and THE FFT algorithm
(discretize map positions to ~0.05 cM, autocorrelation of the weighted grid array
via FFT, ~linear cost). Chintalapati/Patterson/Moorjani 2022 DATES, eLife 77625
(PMC9293011) — `A(d) ~ e^-(t+1)d`, `y=A*e^-lambda*d+c` affine fit, bin 0.1 cM, fit
window 0.45-100 cM, leave-one-chromosome jackknife, "FFT as described in ALDER",
NRMSD < 0.7. DATES code: `github.com/priyamoorjani/DATES` (NOT on box5090 — flag).
cuFFT 13.x: docs.nvidia.com/cuda/cufft (CUFFT_D2Z double-precision real-to-complex
batched, `cufftPlanMany`); `libcufft.so.12.0.0.61` VERIFIED present in CUDA 13.0
on box5090. STEPPE CODE (read at HEAD): `dstat.cpp:195-234,240-254,70-157`;
`backend.hpp:304-315,615,749-781`; `cuda_backend.cu:1215-1219` (decode D2H to fix),
`:1470` (GPU-bound exemplar), `:60-64` (CUB wired); `block_partition_rule.hpp:162,
179`; `f4ratio.cpp` (ratio_jackknife); `small_linalg.hpp:145,204`;
`snp_reader.hpp:41`; `device/CMakeLists.txt:56` (cufft NOT yet linked);
fit-engine.md (precision policy); `dates-genotype-stat-seam.md` (the on-ramp
decision).
