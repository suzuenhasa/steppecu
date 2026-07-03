# `cuda_backend_qpadm_fit.cu` reference

## 1. Purpose

`src/device/cuda/cuda_backend_qpadm_fit.cu` holds the out-of-line
implementations of the CUDA backend's qpAdm *fit* engine — the second half of a
qpAdm run, which takes the already-computed f-statistic building blocks and turns
them into admixture-weight estimates, standard errors, chi-square statistics, and
the rank/population "drop" tables. The class declaration itself lives in a header
(`cuda_backend.cuh`); this file is where the method bodies live.

The methods here fall into a few groups:

- **The block jackknife** — `jackknife_cov` and `jackknife_diag` build the
  covariance (or just the diagonal variance) of the f4 statistics by
  partitioning the genome into blocks and leaving each one out in turn.
- **The two fit paths** — a fast bit-parity "small" path for typical model sizes,
  and a "large" path that uses cuSOLVER's singular-value decomposition when a
  model is too big for the small kernels. The helpers `model_fits_small_path`,
  `gesvdj_applicable`, `large_svd_scratch_sizes`, the two `large_svd_V`
  overloads, the scratch-size helpers, and `large_fit_one` all serve this split.
- **The rank sweep** — `rank_sweep` fits a single model at every candidate rank
  and reports which ranks the data reject.
- **Weights and standard errors** — `gls_weights` solves one model; the
  leave-one-out trio `populate_loo_wmat_resident` / `gls_weights_loo_batched` /
  `se_from_wmat` produce the jackknife standard errors.
- **The batched multi-model fit** — `fit_models_batched`, `fit_one_bucket`,
  `fit_chunk`, and `assemble_result` fit a whole list of models at once, which is
  the production workload (thousands of models in a rotation).
- **A shared table builder** — `fill_rankdrop` builds the nested rank-drop table
  used by both the single-model sweep and the batched assemble.

A recurring goal across the file is **bit-for-bit reproducibility** and an exact
match to ADMIXTOOLS 2's numbers. Many comments in the source explain why a
particular rearrangement (pooling scratch, running blocks in parallel, tiling by
chunk) does *not* change any reported value. Those "moves no bits" guarantees are
load-bearing: they are what let the GPU path claim it matches the reference.

---

## 2. The precision policy: which stages run native, which run emulated

steppe runs its heavy matrix multiplies in an emulated form of double precision
that is much faster than the GPU's native double precision at essentially the same
accuracy. But not every stage of the fit is safe to run that way. This file draws
a firm line, and the same line is drawn everywhere a fit stage touches the GPU.

**Stages that honor the requested precision (default: emulated double precision).**
The well-conditioned covariance matrix multiply — the symmetric product that forms
the covariance matrix `Q` from the centered deviations — engages whatever
precision the caller passed, exactly like the f2 matrix multiplies do. If the
build or device cannot honor emulation, it automatically falls back to native
double precision.

**Stages pinned to native double precision, always, regardless of what was
requested:**

- **The centering step** (`xtau`, which forms `leave-one-out − estimate`). This is
  a subtraction of two nearly-equal numbers — a catastrophic-cancellation site.
  Emulation can faithfully form a *product*, but it cannot recover bits that a
  prior *subtraction* already annihilated. So the centering is held native, the
  same carve-out the f2 numerator uses.
- **The covariance inverse** (the Cholesky factor-and-invert). The matrix being
  inverted is ill-conditioned, so it is pinned to native double precision by the
  conditioning rule.

**The math-mode scope.** The matrix-multiply calls used here (`cublasDsyrk` and
`cublasDgemmStridedBatched`) are the legacy cuBLAS API — there is no per-call
argument to pick the compute precision, so precision is driven by the *handle's*
math mode. Because the handle is shared with the later native solves, every place
that engages the emulated (or native-fallback) mode does so inside a small scope
that captures the handle's current mode, applies the new one for just that one
call, and restores the captured mode at scope exit. That is what keeps the
emulated mode from leaking into the next native solve on the same handle — a
determinism requirement, not a nicety.

**The promotion seam.** There is live plumbing (`solve_precision_` →
`engage_solver_precision`) that *could* one day run the ill-conditioned inverse in
an emulated tensor-core mode. Today it always falls back to native, because the
installed cuSOLVER exposes no emulated-double-precision math mode. The scope is
still real (it performs an actual set/restore round-trip) and is kept local so it
never leaks a mode either. It exists so a future per-stage policy can promote that
inverse after validating it against the native oracle at production scale.

---

## 3. The block jackknife covariance — `jackknife_cov`

`jackknife_cov` produces the covariance matrix of the f4 statistics using a block
jackknife: the genome is split into blocks, each block is left out in turn, and the
spread of the leave-one-out estimates is what the covariance measures.

Inputs are the f4 blocks (`x`, which carries the per-block leave-one-out estimates
`x_loo` and the full estimate `x_total`), the per-block SNP counts, a `fudge`
ridge amount, and the precision policy. Let `m = nl·nr` be the number of f4
statistics and `nb` the number of blocks; an empty problem (`m ≤ 0` or `nb ≤ 0`)
returns immediately with an OK status.

The steps:

1. **Upload and center.** The leave-one-out estimates, the full estimate, a
   per-run baseline line, and the block sizes are copied to the device, and a
   kernel forms the centered, tau-weighted deviation matrix `xtau` (stored
   column-major, one column per block). This centering is native precision (see
   section 2).
2. **Form `Q`.** The covariance is `Q = xtau · xtauᵀ / nb`, a symmetric `m×m`
   matrix, computed with a single symmetric-rank-`k` update (`cublasDsyrk`,
   lower-triangle, no transpose). This is the one stage that honors the requested
   precision, bracketed by the math-mode scope. The lower triangle is then
   mirrored to a full matrix, and `Q` is copied back to the host.
3. **Add the fudge ridge.** The trace of `Q` is summed on the host (it is tiny),
   and a fudged copy `Qf = Q` is made on the device with `fudge · trace` added to
   each diagonal entry. `Q` itself is returned *unfudged*; only the copy used for
   the inverse gets the ridge.
4. **Invert `Qf`.** The inverse is computed by a Cholesky factorization
   (`potrf`) followed by an in-place inverse-from-factor (`potri`), both native
   double precision. If either reports a non-positive-definite or singular pivot
   (device info `> 0`), the result carries `NonSpdCovariance` status and returns.
   The lower triangle of the inverse is mirrored to full and copied back.

**One subtlety worth calling out — the pooled solver workspace.** The two cuSOLVER
routines used here need *different* amounts of scratch, and `potri` needs far more
than `potrf` (for a 10×10 matrix, `potrf` wants 20 doubles but `potri` wants
65536). The code queries *both* buffer sizes, sizes one shared pool to the larger
of the two, and then feeds each routine its own required size. Sizing the pool by
`potrf` alone would silently under-allocate for `potri`, which on some GPUs is a
quiet device-heap overrun and on others a hard illegal-write failure. Growing the
pool changes no math — it is more scratch, identical results.

The return value bundles `m`, the unfudged `Q`, its inverse `Qinv`, and a status.

---

## 4. The block jackknife diagonal — `jackknife_diag`

`jackknife_diag` is the cheap sibling of `jackknife_cov`: it computes only the
diagonal of the covariance — the per-statistic variance — and skips the full
matrix product and the Cholesky inverse entirely.

It runs the exact same prologue as `jackknife_cov` (upload the leave-one-out and
full estimates, form the centered `xtau`), then computes
`var[k] = (1/nb) · Σ_b xtau[k + m·b]²` with a single reduction kernel. That is
`O(m·nb)` work and only `O(m)` memory, versus the covariance path's `O(m²)`
matrix and inverse. It always runs native double precision (the precision argument
is ignored), because both the centering and the per-item sum of squares are on the
cancellation-sensitive side of the line.

---

## 5. Choosing the fit path: small vs large

Two tiny predicates decide how a model is fit.

`model_fits_small_path(nl, nr, r)` delegates to the shared core routine that owns
the "small-path envelope" — the single source of truth for whether a model's
dimensions fit the fast, bit-parity small kernels. Keeping this delegation (rather
than re-deriving the bound here) means the GPU backend and the rest of the code can
never disagree about which models are "small."

`gesvdj_applicable(nl, nr)` returns true when *both* dimensions are at most 32.
That threshold is the size limit of the batched Jacobi singular-value routine
(`gesvdj`); at or below it the faster Jacobi routine is used, and above it the
QR-based routine (`gesvd`). The value 32 is frozen for reproducibility and must not
change — the reference results assert that a matrix with a dimension of 39 took the
QR path and a small one took the Jacobi path.

The large path (used when `model_fits_small_path` is false) is built on cuSOLVER's
singular-value decomposition and is described in the next sections.

---

## 6. The large-path SVD: computing V

The large fit path needs the right singular vectors `V` of the model matrix — the
leading `r` columns of `V` seed the low-rank factorization. `large_svd_V` produces
exactly that, and it comes in two overloads.

**Orientation.** cuSOLVER's SVD wants a tall-or-square matrix (rows ≥ columns), so
the code first decides an orientation. If `nr ≥ nl` it works with the transpose
`Xᵀ` (which is `nr × nl`); the left singular vectors of `Xᵀ` *are* the right
singular vectors of `X`. If `nl > nr` it works with `X` itself (`nl × nr`); there
`V` is the transpose of the first `r` rows of the returned `Vᵀ`.

**The two branches.** In the `nr ≥ nl` branch, `X` is transposed, the SVD is run in
economy mode (Jacobi `gesvdj` when applicable, otherwise `gesvd` with `jobu='S'`,
`jobvt='N'`), and the leading `r` columns of the returned `U` are copied straight
into the output — a contiguous prefix, since the leading dimension already matches.
In the `nl > nr` branch, `X` is first copied into a *non-const* scratch buffer
(cuSOLVER overwrites its input, and the caller's `X` must survive for the seed and
alternating-least-squares kernels that follow), the SVD returns `Vᵀ`, and a small
transpose recovers `V` before copying its leading `r` columns out.

**The scratch-owning overload.** The second `large_svd_V` overload takes no scratch
pointers. It sizes the SVD scratch for this model's shape, grows the backend's
persistent member arenas to fit (they only ever grow, never shrink, and are shared
with the Cholesky path since the fit runs on one single stream), then calls the
first overload. Growing the arenas is bit-neutral: the buffer-size query does not
depend on the data, and the scratch is fully overwritten each call.

**A discarded but required output.** cuSOLVER's SVD routines require a non-null
device "info" out-argument (negative means a bad parameter; for the Jacobi routine
a positive value means it did not converge). On this large path that info is
intentionally *not* read back — the argument cannot be dropped, but it is not
checked, because this path is off the bit-parity guarantee anyway.

---

## 7. Scratch-size helpers

Three small helpers compute exactly how much device scratch the large path needs
for a given `(nl, nr, r)`, so the callers can allocate once and reuse:

- `large_dbl_scratch` returns the larger of two working-set sizes — the
  alternating-least-squares refinement's working set, and the weight-plus-chi-square
  solve's working set — since those two phases run one after the other and can share
  the same buffer.
- `large_int_scratch` returns the integer scratch, the larger of the factorization
  dimension and `nl`.
- `large_loo_dbl_refit` returns the per-refit double scratch used by the
  leave-one-out standard-error path: the model matrix, the two factor matrices `A`
  and `B`, plus one `large_dbl_scratch` working set.

These exist so every large-path allocation is sized from one place and the batched
LOO kernel can stride cleanly by the per-refit size.

---

## 8. Fitting one model on the large path — `large_fit_one`

`large_fit_one` is the single-model large-path fit, and it is the piece the rank
sweep and the single-model weight solver both reuse when a model is too big for the
small kernels.

For a positive rank it: computes the leading `r` right singular vectors `V`
(section 6), seeds the low-rank factors `A` and `B` from `V`, and runs the
alternating-least-squares refinement to convergence. Then, for any rank (including
0), it runs the constrained weight solve that produces the admixture weights and
the chi-square. Rank 0 is the degenerate "no free parameters" case: the SVD, seed,
and refinement are all skipped and only the weight/chi-square step runs.

---

## 9. The rank sweep — `rank_sweep`

`rank_sweep` fits a single model at *every* candidate rank `r` from 0 up to
`rmax = min(nl, nr) − 1`, and reports the chi-square, degrees of freedom, and tail
p-value at each rank. This is how qpAdm decides how many admixture sources the data
actually support. It always runs native double precision to match the reference
oracle exactly.

Key points:

- **Degenerate guard.** If `rmax < 0` (a zero-row or zero-column model), it returns
  `RankDeficient` with no candidate rank.
- **One path for the whole sweep.** The widest rank, `rmax`, decides whether the
  whole sweep runs the small or large path — the bit-parity envelope is monotone in
  `r`, so if the widest fit is small then all narrower fits are too.
- **Upload once, reread per rank.** The full estimate and the inverse covariance
  are uploaded once and the model matrix is built once; the per-rank `V`/`A`/`B`
  are recomputed inside the loop from that shared data.
- **One D2H, not a ladder.** There is a separate chi-square and status slot *per
  rank*, so all `rmax + 1` fits enqueue back-to-back on the stream and the host
  pays a single device-to-host copy after the loop. The earlier design did a copy
  and a synchronize after every rank, and each of those stalls blocked the next
  rank's work behind a tiny copy draining — this removes that ladder.
- **Post-processing.** After the single copy-back, each rank's degrees of freedom
  and tail p-value are computed on the host, and any degenerate fit flips the sweep
  to `RankDeficient`.
- **The rank-drop table** is filled by the shared `fill_rankdrop` (section 14).
- **`f4rank`** is reported as the smallest rank the data do *not* reject (the first
  rank, scanning upward, whose p-value exceeds `alpha`); if every rank is rejected
  it falls back to `rmax`.
- **`rank_Q`** is the numerical rank of the *unfudged* covariance matrix, computed
  on the device with a one-sided Jacobi sweep that counts the singular values above
  a scaled machine-epsilon threshold. It is a bit-identical mirror of the CPU
  reference (the epsilon is passed in from the host so the constant matches
  exactly). It is a diagnostic only — an observed property of the covariance, off
  the rank-test math path — not something the rank decision depends on.

---

## 10. GLS weights for one model — `gls_weights`

`gls_weights` fits a single model at a fixed rank `r` and returns the generalized-
least-squares admixture weights `w` (length `nl`), the low-rank factor matrices `A`
(`nl × r`) and `B` (`r × nr`), and the chi-square. It picks the small or large path
by `model_fits_small_path` and, on the small path with positive rank, runs the
seed → alternating-least-squares → weight/chi-square sequence; the large path calls
`large_fit_one`.

The weights, factors, chi-square, and status are copied back to the host. If the
device reports the rank-deficient status code, the result carries `RankDeficient`;
otherwise `Ok`. (A note in the source records that the kernel emit sites still
write a bare numeric status code rather than the named symbol — a deferred
device-side cleanup — but the host already decodes it through the named symbol.)

---

## 11. Standard errors from leave-one-out refits

Standard errors come from re-fitting the model with each genome block left out and
measuring how much the weights move. Three methods cooperate here, sharing one
resident producer.

**`populate_loo_wmat_resident` — the shared producer.** This fills a device-resident
weight matrix, one row of `nl` weights per leave-one-block-out refit. It does *not*
copy anything back or synchronize; it just enqueues the work and leaves the matrix
resident so a caller can either copy it back or reduce it in place. On the small
path this is a single batched kernel. On the large path it runs in two stages:

- **Stage A** produces, for each block, a cuSOLVER SVD seed into per-block strided
  arenas. Crucially there is *no* per-block synchronize and *no* per-block buffer
  free. The SVD scratch is allocated once above the loop and shared across all
  blocks — the per-block SVDs are enqueued serially on the one stream, so they
  never overlap and can safely reuse one workspace. Allocating (and freeing) scratch
  per block would pay a device-wide synchronize on every block's free, which is
  exactly the serialization this parallel rewrite exists to remove. The SVD stays
  cuSOLVER-based per block precisely so the seed is bit-identical to the older
  serial code — an on-device Jacobi seed would shift the refinement's fixed point
  in the low bits and the standard errors would no longer match.
- **Stage B** is a single many-threaded launch — one thread per (model, block) —
  that runs the refinement and weight solve for every block concurrently from its
  own scratch slice.

This large-path parallelization is what turned a formerly serial per-block loop
(hundreds of seconds on the biggest models) into one concurrent launch, while
keeping the weight matrix bit-identical to the serial version.

**`gls_weights_loo_batched`** calls the producer and copies the weight matrix back
to the host. It is kept as a reusable primitive even though the fast standard-error
path no longer needs the host copy.

**`se_from_wmat`** calls the producer and reduces the resident weight matrix on the
*device* with the same standard-error kernel the batched path uses (treating it as
a single model), copying back only the `nl`-length standard-error vector — the
sum-of-squares reduction never leaves the GPU. It then reapplies the ADMIXTOOLS 2
scale factor `(nb − 1) / √nb` that the unscaled weight matrix lacks (an exact,
linear rescale). This requires at least two blocks; with fewer it returns zeros.

---

## 12. The batched multi-model fit

This is the production workload: fit a whole list of models in one call, reading
the resident f2 blocks that are already on the GPU. Four methods form the pipeline.

### `fit_models_batched` — bucketing and the survivor set

The entry point first decides a **precision tag** for the whole batch — emulated
double precision if it was requested *and* the device can honor it, otherwise
native — and records that tag in every result so a consumer knows what produced the
number.

It then **buckets the models by shape** `(nl, nr, r)`. Every model in a bucket has
identical dimensions, so a bucket can be packed into dense strided arenas. The
common rotation workload (all `k`-subsets of one source pool against one right set)
collapses to just a few buckets — one per left-set size. The rank per model is the
caller's fixed rank when given, otherwise `nl − 1`.

It computes the **survivor block set once**, since which blocks are usable is a
property of the resident f2 shared by every model. A block that is missing for any
pair is dropped, exactly as the single-model path does (matching ADMIXTOOLS 2's
"remove missing" read behavior). When nothing is missing — the usual case — the
survivor set is the identity and the survivor-index pointer stays null, which makes
the gather take its bit-identical no-drop branch. The survivor block sizes and the
resident survivor map are threaded down into every bucket and chunk.

### `fit_one_bucket` — VRAM budgeting and chunking

For one shape bucket, `fit_one_bucket` computes how many models fit in GPU memory at
once (the chunk size `B`) and then tiles the bucket into chunks of that size.

The per-model byte cost is summed from the strided arenas each model needs — the
gathered f4 data, the leave-one-out and centered arrays (three of size `m·nb`), the
four `m²` covariance matrices, and the small fit outputs. When the jackknife policy
is on, the *second-pass* standard-error arenas (a compacted leave-one-out array, an
inverse covariance, a weight matrix, and a standard-error vector) coexist with the
first-pass arenas in the same scope, so their cost is folded into the per-model
figure as well. That second-pass term is gated to zero when the jackknife is off,
so the no-jackknife chunk size is exactly what it always was. (An earlier budget
counted only the first pass, which over-committed and ran out of memory once the
second pass allocated on top.)

The available memory uses a 4 GiB fallback when the free-memory probe returns
nothing and reserves a fixed 512 MiB of headroom; the chunk size is the remaining
budget divided by the per-model cost, clamped to at least one model and at most the
whole bucket. Chunk width moves no bits — the standard error is per-model and
chunk-independent — so pools that were already fit at one chunk width stay
byte-identical when the width changes.

### `fit_chunk` — the per-chunk pipeline

`fit_chunk` is the heart of the batched fit and runs one chunk of `B` models
through the full pipeline. It bumps a dispatch counter for observability, builds the
per-model target/left/right index arenas on the host and uploads them, then:

1. **Gather and center.** One batched gather reads the resident f2 (survivor-
   compacted) into the strided f4 arena, then the leave-one-out, total, and centered
   arrays are formed — all batched over the `B` models.
2. **Covariance.** `Q = xtau · xtauᵀ / nb` for every model via one strided-batched
   matrix multiply. This is the stage that engages the requested precision through
   the handle math mode, scoped so it does not leak into the native inverse that
   follows. The fudge ridge is added per model to produce `Qf`.
3. **The batched inverse — column by column.** Each model's `Qf` is Cholesky-
   factored in place with `potrfBatched`. The batched triangular solve
   (`potrsBatched`) supports only a single right-hand side, so the `m`-column
   inverse is built one column at a time: the inverse starts as a per-model
   identity, and for each column `c` the code solves `Qf · x = e_c` in place on
   column `c` across *all* `B` models at once. That is `m` batched solves total (and
   `m` is at most about 50). A per-model factorization that reports non-positive-
   definite is recorded as `NonSpdCovariance` and the batch continues. This inverse
   is native double precision. To make it efficient, all the per-column pointer
   arrays are built once and uploaded in a single host-to-device copy, so the loop
   only issues the batched solves.

   Two device-memory correctness notes live here. The solve's "info" argument is a
   *device* scalar (unlike the many host scalars the same routine takes) — passing a
   host address is undefined behavior that faults on some GPUs — so it is backed by a
   real device buffer, and since per-column positive-definiteness is already gated by
   the factorization's info array, that scalar is written and never read. And the
   stream must be drained after the batched solves: an undrained solve lane makes the
   *next* asynchronous copy fail with an invalid-value error (a measured behavior).
4. **The fit kernel.** One thread per model runs the rank sweep, weight solve,
   chi-square, population-drop, and feasibility, producing the weights, standard
   errors (as a placeholder here), chi-square, status, and the rank/population
   chi-square arrays.

### The two-pass standard-error policy

After the cheap point estimate is complete for all `B` models, the chunk decides
which models are worth the expensive leave-one-out standard error and computes it
only for those.

The cheap fields are copied to the host, and the **survivor filter runs on the host
from bytes that are already there** (no extra copy). A model is *eligible* only if
it produced a valid fit (both the factorization and the rank test succeeded) — a
domain-failed model has no weights, so no standard error, in any mode. Among the
eligible models the jackknife policy decides: `None` computes none, `All` computes
all, and `FeasibleOnly` keeps the models whose reported weights are all in `[0, 1]`
(the same feasibility test the result assembler uses), optionally further requiring
the p-value to clear a threshold.

The second pass then runs the leave-one-out standard errors over just the
survivors:

- **The all-survive fast path.** When every eligible model survives and every model
  in the chunk is eligible, the standard-error block runs verbatim over the whole
  chunk. This is provably the pre-policy code path, so the `All` mode is
  byte-for-byte identical to what the code did before the policy existed — that is
  the parity pin.
- **The compacted path.** Otherwise the survivors' leave-one-out and inverse-
  covariance slices are gathered into dense arenas with one gather kernel (a pure,
  parity-neutral data move), the same standard-error kernels run over the compacted
  set, and the results are scattered back into the full-chunk positions. Because the
  gather preserves ascending order, each survivor's compacted slice is bit-identical
  to its full-arena slice.

The jackknife rescale `(nb − 1) / √nb` is applied inside the kernel here. Finally,
each model is assembled into its result, written positionally into the slot for the
model's original position in the input span.

---

## 13. Assembling a result — `assemble_result`

`assemble_result` turns one model's host-side fit outputs into a fully populated
result record. It mirrors the single-model host path exactly, so the batched
rotation and a one-off single-model fit produce the same record.

It records the model index (echoed so the orchestrator can re-sort a
pre-sized-slot output deterministically), the precision tag, the estimated rank,
and the degrees of freedom. A non-positive-definite covariance returns
`NonSpdCovariance` immediately; a non-zero fit status returns `RankDeficient`.

For a valid fit it fills the weights, then applies the **standard-error sentinel
policy**: the standard errors and z-scores are filled *only* when this model was in
the survivor set that actually got a standard error computed. A non-survivor is left
with *empty* standard-error and z vectors — that emptiness is the "not computed"
sentinel, deliberately never a fake zero or NaN. A survivor's standard error and
z-score are bit-identical to what the `All` mode would produce. The z-score is
weight over standard error, or zero when the standard error is not positive.

It then fills the tail p-value, the per-rank chi-square / degrees-of-freedom /
p-value sweep (and the `f4rank` — the smallest non-rejected rank), the nested
rank-drop table via the shared helper (section 14), and the **population-drop
table**: a full row over all sources followed by one row per single-source drop,
mirroring the reference implementation's population-drop routine exactly, including
its degrees-of-freedom formula and its feasibility rule (all non-NaN reported
weights in `[0, 1]`, with at least one present). Only the full row carries
per-source weights; the drop rows carry chi-square, degrees of freedom, and p-value
and record feasibility as false, matching the reference, which reads feasibility
only from the full row.

Finally, the status is `ChisqUndefined` when the degrees of freedom are not
positive (the tail p-value is then undefined even though the fit itself succeeded),
otherwise `Ok`. This makes the batched rotation surface the same domain outcome the
single-model path does instead of quietly returning a NaN p-value with an `Ok`
status.

---

## 14. The shared rank-drop table — `fill_rankdrop`

`fill_rankdrop` builds the nested rank-drop table that both the single-model rank
sweep (section 9) and the batched result assembler (section 13) need. It lives in
this file's private namespace and is single-homed precisely so those two callers
cannot drift apart — the table is a mirror of the reference implementation, and one
shared builder guarantees the single-model sweep and the batched assemble stay
byte-identical.

The table has one row per rank, written in *descending* rank order (row 0 is rank
`rmax`, the last row is rank 0). For each row it copies that rank's degrees of
freedom, chi-square, and p-value from the per-rank source arrays, then computes the
*nested* difference against the next-lower rank: the degrees-of-freedom difference,
the chi-square difference, and the p-value of that difference. The final row (rank
0) has no lower rank to difference against, so its difference fields are set to the
"not available" sentinels — a specific integer minimum for the degrees-of-freedom
difference and a quiet NaN for the chi-square difference and its p-value. The
callers differ only in which destination fields and which source arrays they pass;
the loop order and every arithmetic step are shared, which is what keeps the result
byte-identical to the two former inline copies.
