# `cpu_backend.cpp` reference

## 1. Purpose

`src/device/cpu/cpu_backend.cpp` is the CPU reference backend — the correctness
oracle for the entire compute pipeline. It implements the same `ComputeBackend`
interface as the GPU backend, but in plain host C++ with no CUDA headers, using
the obviously-correct scalar math. Every result the GPU produces is continuously
diffed against what this file produces.

Two facts follow from that role:

- **No GPU required.** Because it pulls in no CUDA toolkit headers, this backend
  lets the library be imported and exercised on a machine with no GPU, and it is
  the reference anchor the GPU is validated against. It still lives in the same
  `steppe::device` namespace as the GPU backend and compiles into the same
  library target — both are implementations of the one CUDA-free
  `ComputeBackend` interface — but this one depends on nothing CUDA-specific.

- **It ignores the precision knob.** Every method takes a `Precision` argument,
  and every method ignores it. The precision setting governs only which flavor of
  arithmetic the GPU's matrix multiplications use. This oracle always computes in
  native double precision, and for numerically sensitive accumulations it goes one
  step further and accumulates in `long double` (wider than double). It is the
  gold reference that every GPU precision mode is validated against, so it is
  never itself downgraded.

The math throughout reproduces ADMIXTOOLS 2 exactly (or, where floating-point
order matters, within a tight tolerance). Many methods were validated bit-for-bit
against saved ADMIXTOOLS 2 reference outputs.

## 2. Why this is an oracle, not a template the GPU copies

The GPU and CPU paths have deliberately different shapes. On the GPU, f2 is
computed as three batched matrix multiplies plus a couple of fused elementwise
kernels. Here, f2 is computed by walking the exact pairwise-complete scalar loop.
The two share no control structure — yet they must agree to tight tolerance.

They agree because the **per-element formula is shared, not copied**. Both paths
call the same small primitive functions (the heterozygosity correction, the
unbiased per-SNP summand, the final divide, and the genotype decode helpers). So
the correction denominator, the unbiased summand, and the divide-by-zero guard
physically cannot drift between the oracle and the GPU feeder — there is one copy
of each formula, consumed by both.

This is the guiding principle for the whole file: reproduce the ADMIXTOOLS 2 math
for parity, but keep the reference in its simplest, most obviously-correct scalar
form, and route both backends through the same shared primitives so neither can
diverge on the formula.

## 3. The cancellation-free f2 reference

f2 between two populations is, per SNP, the squared difference of their allele
frequencies minus two heterozygosity corrections, averaged over SNPs. How that
average is computed is what makes this the accuracy oracle.

### The cancellation-free property

For each population pair the reference forms the per-SNP difference of allele
frequencies and squares it **directly** — it never expands it into
`p_i² − 2·p_i·p_j + p_j²`, which is the form the matrix-multiply path builds. Each
per-SNP summand is therefore already a small number. Those small summands are then
summed across SNPs in `long double`.

This sidesteps the catastrophic cancellation that lives in the expanded form
(`Σp_i² + Σp_j² − 2·Σp_i·p_j`), where three large sums nearly cancel. The GPU keeps
that expanded step in native double precision for exactly this reason; the oracle
avoids the problem entirely by never expanding. The reference owes its accuracy to
the per-SNP difference plus wide accumulation, not to a different formula.

### Pairwise (cascade) summation

`pairwise_sum` adds up the per-SNP summands by recursively halving the range and
straight-summing any block of 128 or fewer elements. Halving recursively reduces
the growth of accumulated rounding error from proportional to the number of terms
down to proportional to its base-2 logarithm, so the cross-SNP accumulator error
stays far below the f2 magnitude. The 128-element base case is fixed to match the
validated reference exactly.

### Pairwise-complete counting and the diagonal

A SNP contributes to a pair only when it is valid (non-missing) in **both**
populations. The count of jointly-valid SNPs is `vpair`; the final f2 divides the
accumulated numerator by `vpair`, with a guard that returns 0 when `vpair` is 0.

The diagonal entry (a population against itself) is also computed: it works out to
minus twice the mean within-population heterozygosity correction, with `vpair`
equal to that population's valid-SNP count. Nothing downstream reads the diagonal
(f3 and f4 use only off-diagonal f2), but it is written consistently across all
paths so it can never spuriously differ between backends.

Outputs are column-major `[P × P]` and symmetric, so each off-diagonal result is
mirrored into both triangles.

### One shared body, two entry points

`f2_pair_over_range` is the single per-pair body — validity check, corrections,
unbiased summand, long-double pairwise accumulation, final divide — computed over a
given SNP range. Two public methods call it:

- `compute_f2` runs it over the whole SNP range `[0, M)` and produces one
  `[P × P]` f2 tensor plus its `vpair` companion.
- `compute_f2_blocks` runs it per genome block. Blocks are contiguous runs of SNPs
  that share a block id, so each block is a half-open SNP range; the partition is
  validated once by a shared range-finder. The output is `n_block` stacked
  column-major `[P × P]` slabs, entry `(i, j, b)` at `i + P·j + P·P·b`, plus the
  per-block SNP counts.

Sharing the one body means the whole-tensor oracle and the per-block oracle cannot
drift from each other on the formula or the accumulation order. Scratch is sized to
the SNP count (or the largest block), so the huge `[P × P × M]` intermediate is
never materialized: the reference runs in `O(P²·M)` time and `O(M)` extra space.

## 4. Genotype decode and per-sample ploidy

`decode_af` turns packed genotype bytes into allele frequencies. For each
population (a contiguous segment of gathered individuals) and each SNP, it sums the
reference-allele codes and counts the non-missing individuals over the segment,
then finalizes the allele frequency through the shared decode primitives — the same
unpack, accumulation, missing-handling, and divide the GPU kernel uses. Because the
accumulators are integers and there is a single double-precision divide at the end,
the allele frequency is exact; the count and validity outputs are integer-valued and
therefore exact. The output is column-major `[P × M]`.

Per-sample ploidy is resolved by a fixed precedence:

1. An explicit per-sample ploidy vector, if supplied, wins.
2. Otherwise, if the "detect ploidy on device" flag is set, a host detection
   prepass runs and its result is used.
3. Otherwise the uniform scalar ploidy is used (the legacy all-diploid path).

`detect_ploidy_host` is that prepass. For each individual it scans the first
several SNPs (a fixed window, or the whole record if shorter) looking for a
heterozygous call; the sample is marked diploid on the first het found, otherwise
pseudo-haploid. It is a literal port of the input-layer ploidy detector, but
implemented here over the shared core decode primitives so that the device layer
does not have to depend on the input-reader layer, while staying bit-identical to
that detector (same window, same het rule, same most-significant-bit-first byte
layout). `detect_sample_ploidy_device` simply delegates to it, and is the reference
the GPU's on-device detector is diffed against.

## 5. Dropping partially-covered blocks

`survivor_blocks` reproduces ADMIXTOOLS 2's behavior of dropping genome blocks that
contain missing data before the jackknife. It returns, in ascending order, the
original block ids that survive.

A block is treated as **missing only when it is partially covered** — at least one
population pair has `vpair == 0` in that block *and* at least one pair has data.
This distinction matters:

- A block where every pair is zero is not a missing block; it is the "no `vpair`
  information" sentinel. Some legacy and parity inputs zero-fill `vpair`, and the
  block-assignment step never emits an empty block, so a real block always has a
  positive diagonal `vpair`. Requiring the *mix* of missing-and-present makes the
  drop fire exactly on the ADMIXTOOLS 2 case and leaves the "no info" path keeping
  every block.
- When `vpair` is absent entirely (older fixtures that do not carry it, where a
  global-intersection guarantee means there are no missing blocks) or is present
  but the wrong size (a programming error), the function keeps every block. The
  fail-soft default is "no drop", byte-identical to the behavior before missing-block
  handling existed.

Dropped blocks are excluded from the jackknife entirely — they are **not** imputed
as zero. The assembly steps in section 6 compact their block arrays onto the
survivor axis and weight the jackknife by the survivor SNP counts.

## 6. Assembling f2 into f3 and f4 statistics

Three methods turn the per-block f2 tensor into the per-block statistic that the
jackknife then consumes. All three build the missing-block survivor set once (from
section 5), compact onto it, and compute the leave-one-out replicates plus the
point estimate once (section 7) so downstream steps share one array.

- `assemble_f4` builds the four-population statistic for a qpAdm-style model. With
  a left set of `[target, sources…]` and a right set of `[R0, rights…]`, each block
  entry is
  `X[i, j, b] = ( f2(Li, R0) + f2(L0, Rj) − f2(L0, R0) − f2(Li, Rj) ) / 2`.
  The `(i, j)` pair is flattened row-major as `k = j + nr·i` — the ADMIXTOOLS 2
  vectorization order, which the later weight-fit indexing relies on.

- `assemble_f4_quartets` builds f4 for a batch of `N` explicit quartets
  `(p1, p2, p3, p4)`:
  `X[k, b] = 0.5·( f2(p2, p3) + f2(p1, p4) − f2(p1, p3) − f2(p2, p4) )`.
  This is the same four-slab identity specialized to a one-source, one-right model;
  it sets the model dimensions so the batch axis is the `N` quartets. Zero new math
  compared with `assemble_f4`.

- `assemble_f3_triples` builds the three-population statistic for a batch of `N`
  triples `(C, A, B)`:
  `X[k, b] = 0.5·( f2(C, A) + f2(C, B) − f2(A, B) )`.
  This three-slab identity is the one genuinely new formula f3 adds; everything
  else (survivor drop, leave-one-out, totals) is reused.

## 7. The block jackknife

The jackknife turns the per-block statistic into a point estimate and a covariance,
reproducing ADMIXTOOLS 2's resampling code. It is split across one private helper
and two public methods.

### The leave-one-out pass (private)

`compute_loo_and_total` runs once per assembly, in a single pass over the blocks,
producing three things per statistic entry `k` (with `bl_b` the block's SNP count
and `n` the total across surviving blocks):

- `tot` — the weighted mean of the per-block values, weighted by block SNP count.
- `loo[k, b] = (tot − X[k,b]·rel_b) / (1 − rel_b)`, where `rel_b = bl_b / n`. This
  is the leave-one-block-out estimate.
- `tot_line` — the weighted mean of the leave-one-out values, weighted by
  `(1 − bl_b/n)`. This is the centering term the covariance uses.
- `est[k]` — the jackknife point estimate,
  `mean(tot_line − loo)·n_block + weighted_mean(loo, bl)`.

All accumulation is in `long double`. `tot_line` is cached on the backend instance
(one model is fit at a time, so it is rebuilt per assembly and consumed by the
covariance).

### The covariance and its diagonal

`jackknife_cov` builds the pseudo-values
`xtau[k, b] = ( est·h − loo·(h−1) − tot_line ) / sqrt(h−1)`, where `h = n / bl_b`,
then forms `Q = xtau · xtauᵀ / n_block` (a symmetric `m × m` matrix, `m = nl·nr`),
and finally inverts a ridged copy of `Q`. The ridge adds `fudge · trace(Q)` to the
diagonal before inverting. If the ridged matrix is not invertible, the status is
flagged and the inverse is sized and zero-filled anyway, so every downstream
consumer sees a well-formed `m × m` buffer instead of reading out of bounds. `Q`
itself is stored unfudged (the golden convention).

`jackknife_diag` computes only the diagonal of `Q` —
`var[k] = (1/n_block)·Σ_b xtau[k,b]²` — without ever forming the dense matrix or its
inverse. The f3 and f4 sweeps read only the diagonal, and forming the full matrix
for a huge batch would exhaust memory. The op order matches the dense path's
diagonal entries exactly, so the result is bit-equal to taking the diagonal of the
full covariance.

## 8. The ratio jackknife for f4-ratios and D-statistics

`ratio_block_jackknife` is one shared long-double reference behind two different
ratio statistics, selected by a mode flag. It folds two formerly-separate host
jackknives into a single oracle copy, computed over generic per-item, per-block
descriptors:

- **Mode 0 (f4-ratio).** Blocks survive when their denominator magnitude is at
  least a threshold. It forms weighted numerator and denominator totals, the overall
  ratio, and then the jackknife estimate and variance via the `tau`/`xtau`
  construction. Fewer than two surviving blocks yields NaN.
- **Mode 1 (D-statistic).** It forms per-block numerator and denominator estimates,
  leave-one-out num/den, a per-block ratio, and then the estimate and variance. The
  survivor mask here is "count greater than zero".

Both modes finish with `z = est / se` and, optionally, a p-value from the
complementary error function.

Two fused public doors compose this with the assembly:

- `f4ratio_blocks_jackknife` assembles the interleaved numerator/denominator quartet
  f4 (numerator in row `k`, denominator in row `N+k` of a `2N` axis) and runs mode 0.
- `dstat_blocks_jackknife` reduces over host genotype arrays via
  `dstat_block_reduce` (below) and runs mode 1, weighting by SNP count and computing
  p-values.

### The per-SNP D reduction

`dstat_block_reduce` is the per-block, per-quadruple segmented reduction that mode 1
feeds on. For each block's contiguous SNP columns, and each quadruple of populations,
it accumulates the normalized-D numerator `(a − b)·(c − d)` and denominator
`(a + b − 2ab)·(c + d − 2cd)` over the SNPs valid in all four populations, and counts
those SNPs. Inputs are column-major `[P × M]`; outputs are row-major `[N × n_block]`.
Accumulation is in `long double` because the numerator and denominator are
cancellation-sensitive. The qpfstats path in section 12 reuses this same reduction.

## 9. The qpAdm rank test and weight fit

The qpAdm fit engine reproduces ADMIXTOOLS 2's qpadm and resampling code in native
double precision and was validated bit-for-bit against a saved golden. Its
vectorization order is the row-major `k = j + nr·i` order throughout, so the matrix
reshapes line up with ADMIXTOOLS 2's indexing.

### Rank sweep

`rank_sweep` walks candidate ranks `r = 0 … rmax`, where `rmax = min(nl, nr) − 1`.
For each rank it computes:

- `chisq(r)` — the rank-`r` alternating-least-squares-refined residual quadratic
  form (via `als_weights`, including the trivial `r == 0` branch).
- `dof(r) = (nl − r)·(nr − r)`.
- `p(r)` — the upper chi-square tail.

It then builds the nested "rank-drop" table (one row per rank in descending order,
each row's nested difference compared against the next-lower rank, the last row
marked not-applicable) and reports `f4rank`, the smallest non-rejected rank (the
first rank, scanning upward, whose p-value exceeds `alpha`). It also reports
`rank_Q`, the numerical rank of the covariance from its singular values above a
relative tolerance, and which singular-value-decomposition path would be selected
downstream (the batched Jacobi routine when both dimensions are at most 32,
otherwise the QR routine).

A capability query, `provides_rank_sweep`, returns true so the orchestrator routes
the rank-drop work here. It deliberately does not provide a batched fit, so that
capability stays false and callers use the per-model oracle shape.

### Weights and the alternating least squares

`gls_weights` is the public entry: seed by singular value decomposition, refine by
alternating least squares, solve for constrained weights, normalize to sum 1.

The private `als_weights` is the full body. If the rank is 0 the weights are all 1
(the single-source trivial case). Otherwise it seeds `A` and `B` (`seed_AB`: `B` is
the top-`r` right singular vectors of the input matrix, `A` is the input times `B`
transposed), iterates `opt_A` then `opt_B` for the configured number of iterations,
and finally solves the constrained weight system — build the augmented matrix
`cbind(A, 1)`, form its cross-product, solve, and normalize — and computes the
chi-square via `chisq_of`.

`opt_A` and `opt_B` are the two halves of the alternating least squares: holding one
factor fixed, each solves for the other to minimize the generalized-least-squares
residual quadratic form. Each expresses its update as a Kronecker-structured linear
operator and hands it to a shared solver.

Supporting these are several helpers whose accumulation order is held bit-identical
for parity:

- `als_ridge_solve` solves the ridged normal equations
  `coeffs = Lᵀ·Qinv·L`, `rhs = xvecᵀ·Qinv·L`, `diag(coeffs) += fudge·trace`, then
  `solve`. The intermediate `Qinv·L` is formed once and reused for both `coeffs` and
  `rhs`. It is templated on the linear operator so `opt_A` and `opt_B` share one
  copy; the only per-caller difference is the operator's index arithmetic, which
  yields the same doubles in the same order.
- `chisq_of` computes `vec(E)ᵀ · Qinv · vec(E)` where `E` is the residual
  `xmat − A·B`, vectorized row-major, accumulated in `long double`.
- `ridge_diagonal` adds `fudge · trace` to a matrix diagonal in place — the single
  source of the fudge-ridge idiom shared by the covariance inverse and the ALS
  solver.
- `xmat_from_total` and `als_xvec` are the layout conversions between the row-major
  flatten and the column-major matrix.

## 10. Standard-error estimation

`se_from_wmat` is the leave-one-block-out standard error. It runs the per-block
re-fit loop to get an `n_block × nl` matrix of replicate weights, applies
ADMIXTOOLS 2's scale of `(n_block − 1) / sqrt(n_block)`, takes the sample-covariance
diagonal, and returns the element-wise square root — the `nl`-length standard error.
It needs at least two blocks.

`gls_weights_loo_batched` is that re-fit loop. For each block it builds a one-block
statistic whose total is the leave-one-out replicate slice for that block, reuses the
same covariance inverse unchanged (the parity pin), refits the weights via
`gls_weights`, and stores the result. It returns an `n_block × nl` row-major matrix.

`se_sample_cov_diag` is the sample-covariance diagonal matching R's `cov()`: it
treats columns as variables, divides by `nrows − 1`, and accumulates the
sum-of-squares in `long double`. The wide accumulator is the test oracle's extra
precision (to match `cov()` on the last digits), not a change to the reported
result.

## 11. DATES admixture-dating curve and fit

`dates_curve` computes the weighted-linkage-disequilibrium curve used to date
admixture. Its key idea is an algebraic identity: the pairwise double-sum over SNP
pairs at a given genetic-distance lag equals the **autocorrelation of the
genetic-map-binned grid**. Autocorrelating the grid is algebraically identical to
the result the GPU's fast-Fourier-transform path produces, so this direct
autocorrelation is both the within-codebase parity pin and an exact reproduction of
the DATES program. Crucially, the roughly 10^12-entry SNP-pair object is never
formed — the code autocorrelates the grid (length equal to the number of fine bins),
not the SNP pairs.

Per target sample it collects the valid SNPs, computes a per-SNP population-delta
weight and a dosage, runs a regression of dosage against the source-frequency delta,
forms the residual signal, and scatters onto the fine map grid three quantities per
cell: a count, the signal, and the signal squared. Per chromosome it then computes
the direct autocorrelation over lags up to a maximum of six product-moments (count
against count, signal against signal, and the cross terms). Finally it re-bins each
fine-grid lag into an output bin and accumulates the six correlation sufficient
statistics. Everything is native double precision; the per-SNP weight is the
cancellation-sensitive part.

Two companions complete DATES:

- `dates_repack` is a bit-exact host bit-shuffle of the target genotype records,
  delegating to the shared core routine. Because it is integer and bit-exact, the
  repacked record is identical on both backends.
- `dates_fit` fits an exponential decay to each curve via a native `long double`
  two-by-two normal-equation solve, returning the date in generations and an error
  estimate. The GPU device fit is held to a loose 2 percent date tolerance against
  this reference.

## 12. qpfstats joint f2 smoothing

`qpfstats_smooth` is the joint-f2 smoothing solve, reproducing ADMIXTOOLS 2's
qpfstats regression exactly. It builds one shared normal-equation matrix
`A_shared = xᵀx + ridge·I`, then for each block downdates it by the outer product of
the block's not-a-number rows and solves against `xᵀ·(y with not-a-number entries
zeroed)`. An all-not-a-number block yields a zero solution. The global solution is
the same shared solve over the global jackknife estimate. The layouts are
column-major, and the solver is a native-double LU factorization (the GPU path
shares one Cholesky factor instead, giving the same answer to floating-point
precision).

`qpfstats_blocks_smooth` is the fused reduce-then-jackknife-then-smooth-then-recenter
path. It composes existing oracles so the reference math is unchanged from the
pre-fusion host pipeline: the per-SNP D reduction from section 8, then a per-comb
long-double global jackknife estimate, then `qpfstats_smooth`, then a per-pair
recenter shift computed from a long-double per-pair jackknife estimate. The jackknife
pieces come from the shared jackknife primitives, so this fused path stays
single-sourced with the host driver it replaced.

## 13. The qpGraph fleet optimizer

`qpgraph_fit_fleet` is the reference for the admixture-graph edge-weight optimizer.
For a graph with `D` admixture proportions (the theta vector), it runs a multistart
projected-Newton optimizer:

- Each restart seeds theta deterministically from a splitmix hash, so the fleet is
  reproducible and the restarts are spread across basins.
- Each iteration sweeps the dimensions. Per dimension it forms a forward-difference
  gradient and a three-point diagonal curvature, takes a projected trust-clamped
  Newton step with a backtracking line search, and clamps back into the valid
  `[0, 1]` range.
- Each function evaluation calls the shared graph score: fill the path weights, run
  the constrained edge-length solve, and evaluate the generalized-least-squares
  quadratic form.

It returns the best-of-restarts score, the spread across restarts, the best theta,
per-dimension theta bounds, and the edge lengths and f3 fit at the best theta. A pure
tree (no admixture, `D == 0`) short-circuits to a single edge-length solve. The
optimizer constants — the splitmix multipliers, the finite-difference step, the trust
clamp, the backtracking factor — are single-sourced with the GPU fleet so this body
stays bit-identical to it for the parity diff.

`qpgraph_fit_fleet_batch` loops the per-topology fleet over a batch of topologies,
reusing the same resident observed-f-statistic and covariance basis for every
topology (they share a population set). A topology whose every restart degenerates
reports positive infinity — it loses the argmin — rather than failing the whole
batch, so the search still ranks the well-identified candidates.

## 14. How the backend is created

`make_cpu_backend` constructs a `CpuBackend` and returns it as a base-class
`ComputeBackend` pointer, so callers depend only on the CUDA-free interface. This is
the dependency-injection seam: the backend is injected into the shared resources
rather than referenced directly. The GPU factory mirrors this signature in the same
namespace, so the two backends are interchangeable at the call site.
