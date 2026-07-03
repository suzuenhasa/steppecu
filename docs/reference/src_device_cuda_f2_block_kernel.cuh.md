# `f2_block_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/f2_block_kernel.cuh` declares the small set of launch-wrapper
functions that compute the **f2 statistic** — a pairwise measure of genetic
distance between populations — on the GPU. It is a header of *declarations only*:
the actual GPU kernels, and the `<<<...>>>` launch syntax that starts them, live
in the matching `.cu` source file. The host-side orchestration code calls the
plain `void launch_*` / `run_*` functions declared here and never sees a kernel
body or a launch bracket.

This split exists so that the general backend code can stay free of raw CUDA
launch syntax while still driving the GPU. The header does, however, name CUDA
library types (such as the matrix-multiply library handle `cublasHandle_t`), so
it is **private to the GPU layer** of steppe. It is the internal seam between the
backend and the kernel translation unit, not part of the public,
CUDA-free interface that outside callers see.

Everything here works on one **tile** of SNPs at a time — a chunk of the genome
processed in one pass — and covers the whole f2 pipeline for that tile: turn the
decoded genotype data into matrix-multiply inputs, run the matrix multiplies, and
assemble the final f2 values.

---

## 2. Shapes and symbols used throughout

A few conventions recur in every function signature.

| Symbol | Meaning |
|---|---|
| `P` | The number of populations. It is a small-to-moderate `int`. |
| `M` | The number of SNPs in the current tile. It is a `long`, because SNP counts run into the millions and can overflow a 32-bit `int`. |
| `d` prefix | Every pointer named `dSomething` points to memory that lives on the GPU (device memory), not on the host. |
| column-major | All matrices are stored column-major (the layout the matrix-multiply library expects): consecutive elements in memory walk down a column, not across a row. |
| `lda` / leading dimension | The row stride of a column-major matrix. When two matrices are stacked on top of each other into one buffer, the leading dimension is the combined height (see the `[2P × M]` and `[2P × P]` buffers below). |

The matrices are sized in terms of these: a `[P × M]` matrix has one row per
population and one column per SNP; a `[P × P]` matrix has one entry per ordered
pair of populations.

---

## 3. The three-matrix-multiply reformulation

The f2 value between two populations is, in essence, an average over SNPs of the
squared difference in their allele frequencies, minus a small per-population
correction for sampling noise. Computing that naively for every pair of
populations would build a huge three-dimensional intermediate of size
`SNP × population × population` and then collapse it — expensive in both memory
and time.

steppe avoids that intermediate entirely by rewriting the whole computation as a
handful of **matrix multiplies** (GEMMs — general matrix–matrix multiplications)
that produce results for *all* population pairs at once. The pipeline for one
SNP tile has three stages, one per group of functions in this header:

1. **Feeder** (`launch_f2_feeder`) — a single fused pass over the decoded tile
   that produces the three input matrices the multiplies need.
2. **Three GEMMs** (`run_f2_gemms`) — the matrix multiplies themselves, which
   contract away the SNP axis and leave per-population-pair sums.
3. **Assemble** (`launch_assemble_f2`) — combine those sums into the final f2
   value for each pair.

The key win is that the enormous per-SNP intermediate is never materialized. The
SNP axis only ever appears as the *contraction* (the summed-over dimension) of a
matrix multiply, so the memory cost scales with the number of populations, not
with populations squared times SNPs.

---

## 4. The feeder pre-pass — `launch_f2_feeder`

```
void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream);
```

The feeder runs one fused elementwise sweep over a decoded SNP tile and produces
the three matrices the matrix multiplies consume. Doing it in one sweep — rather
than several passes that each re-read the tile — is a deliberate performance
choice.

### The inputs (the "Q/V/N contract")

Three `[P × M]` matrices describe the decoded tile:

| Input | Name | Meaning |
|---|---|---|
| `dQ_raw` | Q, raw allele frequency | The reference-allele frequency of each population at each SNP. |
| `dV_raw` | V, validity | Whether that population/SNP entry is usable (non-missing). |
| `dN_raw` | N, count | The number of non-missing haploid samples behind that frequency. |

### The outputs

| Output | Shape | Contents |
|---|---|---|
| `dQ_masked` | `[P × M]` | Q with the invalid entries zeroed out (Q multiplied element-wise by V). Zero-filling the invalid entries means they contribute nothing when this matrix is later multiplied. |
| `dV_out` | `[P × M]` | The validity mask as numbers: `1.0` where the entry is valid, `0.0` where it is missing. This is what lets a later matrix multiply *count* how many SNPs a pair of populations share. |
| `dS` | `[2P × M]` | Two `[P × M]` matrices stacked on top of each other into one buffer of height `2P` (so its leading dimension is `2P`). The top half is `Qsq` — the masked frequency squared. The bottom half is `Hc` — the per-SNP heterozygosity correction, `q(1-q) / max(N-1, floor) · V`. |

The `max(N-1, floor)` in the heterozygosity term guards against dividing by zero
when a population has only a single non-missing sample (where `N-1` would be `0`);
the floor is a shared constant equal to `1`.

### Why one shared formula

The heterozygosity correction is computed through the same shared primitive that
the CPU reference implementation uses. Because both the GPU feeder and the CPU
oracle call into the identical formula, the two can never quietly disagree about
what the correction is — which matters, because the CPU path is the trusted
reference the GPU path is validated against. All of the feeder's arithmetic is in
native double precision.

### Precondition

The feeder requires `P >= 1` and `M >= 1`. A zero or negative extent would ask
the driver to launch a zero-sized grid of threads, which it rejects as an invalid
configuration. The caller handles the degenerate empty case with an early return
*before* reaching this wrapper, so the feeder is never called with a
non-positive extent.

---

## 5. Choosing the precision for the matrix multiplies

Three functions decide, together, which flavor of arithmetic the f2 matrix
multiplies run in. steppe's default is an **emulated double precision** built from
fixed-size slices, which is much faster than the GPU's native double precision
while staying about as accurate. But that emulated mode is only safe to use under
specific conditions, and these three functions make sure the decision is made
once and consistently.

### `emulation_honorable` — the single gate

```
[[nodiscard]] bool emulation_honorable(const Precision& precision) noexcept;
```

This is the one predicate that decides whether an emulated-double-precision
request can actually be *honored*. It returns `true` only when both:

- the requested precision is emulated double precision, **and**
- the GPU code was built with the fixed-slice tuning capability compiled in.

Why the second condition matters: without that build capability, the matrix
multiply library would still engage emulation, but under its *dynamic*
mantissa-control default. That dynamic mode was measured to overshoot to roughly
60 mantissa bits and collapse back to the speed of native double precision — all
cost, no benefit — so steppe rejects it. When emulation is not honorable, the
policy instead downgrades to native double precision.

Both of the functions below route their decision through this one predicate, so
the two halves of the precision choice — the math mode and the compute type — can
never disagree. It is exposed (rather than kept private) so that callers and tests
can observe the honorability state directly, without needing the build-capability
macro to be visible in their own code.

### `engage_f2_precision` — apply the policy to the handle

```
void engage_f2_precision(cublasHandle_t handle, const Precision& precision);
```

This applies the precision decision to the matrix-multiply library handle before
the multiplies run. For an *honorable* emulated request it issues the
load-bearing sequence that turns on emulated-double-precision math with a **fixed**
mantissa bit count (taken from `precision.mantissa_bits`). For a native
double-precision request — and for an emulated request the build cannot honor — it
sets strict native math instead and emits a one-time, capability-tagged warning
noting the downgrade. Either way, the path never silently runs the rejected
dynamic-default emulation.

It is exposed as its own function (rather than buried inside the multiply routine)
so that the batched/grouped variant of the f2 path can engage the exact same
policy *once* before its loop over groups, instead of re-deriving it. That keeps
the precision engagement defined in a single place.

### `f2_compute_type` — map to the library's compute type

```
[[nodiscard]] cublasComputeType_t f2_compute_type(const Precision& precision);
```

This maps the requested precision to the matrix-multiply library's compute-type
enum, routing through `emulation_honorable` so the compute type and the math mode
are always derived from the same decision:

- an honorable emulated request → the emulated fixed-point double-precision
  compute type;
- native double precision, TF32, **and** an emulated request the build cannot
  honor → native double-precision compute type.

The same mapping is used by both the single-tile and the batched f2 paths.

---

## 6. The three matrix multiplies — `run_f2_gemms`

```
void run_f2_gemms(cublasHandle_t handle, const Precision& precision,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR);
```

This runs the three matrix multiplies at the heart of the f2 computation. They
take the feeder's three output matrices and contract away the SNP axis (`M`
becomes the summed-over dimension), leaving per-population-pair sums. All three
are column-major.

| Output | Shape | Computed as | What it holds |
|---|---|---|---|
| `dG` | `[P × P]` | Q · Qᵀ | For each pair of populations, the sum over SNPs of the product of their masked frequencies. |
| `dVpair` | `[P × P]` | V · Vᵀ | For each pair, the number of SNPs valid in *both* — the shared-SNP count. This is **retained** for later use as the block-jackknife weight, not just an intermediate. |
| `dR` | `[2P × P]` | S · Vᵀ | Two stacked `[P × P]` results (leading dimension `2P`): the top half is the per-population sums of squared frequencies, the bottom half the summed heterozygosity corrections. |

### `precision` governs only these multiplies

The `precision` argument controls the arithmetic of *these matrix multiplies and
nothing else*. An honorable emulated request runs them in fixed-slice emulated
double precision; native double precision, TF32, and an unhonorable emulated
request run them in native double precision (the last emitting a one-time
capability-tagged note). The delicate, cancellation-prone assembly step that
follows is held in native double precision regardless — see section 7.

### Why this routine takes no stream

`run_f2_gemms` deliberately takes **no** stream argument and never calls the
library's set-stream function. The handle it receives must already have both its
stream *and* its emulated-double-precision determinism workspace bound onto it
once, up front. Re-binding the stream here would reset the workspace back to the
library's default pool, which would defeat the fixed-workspace guarantee that
makes the emulated results reproducible run to run. So the stream/workspace
binding is a one-time setup done by the caller on the handle, and this routine
leaves it untouched.

### Precondition

Like the feeder, this requires `P >= 1` and `M >= 1`. Here `M` becomes the
contraction length of the multiplies; a length of zero would be a degenerate
no-op that leaves the outputs unpopulated. The caller guards the empty case with
an early return before this point.

---

## 7. Assembling the final f2 values — `launch_assemble_f2`

```
void launch_assemble_f2(const double* dG, const double* dVpair, const double* dR,
                        double* dF2, int P, cudaStream_t stream);
```

This is the final step: it combines the three matrix-multiply outputs into the f2
value for every pair of populations and writes the result into `dF2` (`[P × P]`).
It uses the same shared numerator-assembly and finalize primitives that the CPU
reference uses, so the GPU and CPU cannot diverge on the formula.

### Held in native double precision on purpose

This step is where large, nearly-equal sums are subtracted from one another — the
classic setup for **catastrophic cancellation**, where the leading digits cancel
and only the low-order (least accurate) digits survive. To keep those surviving
digits trustworthy, the assembly is done in **native double precision in every
precision mode**, even when the matrix multiplies before it ran in the faster
emulated mode. This is the deliberate carve-out: the bulk matrix multiplies get
the fast emulated arithmetic, while the one numerically delicate step stays on the
gold-standard native path.

`dVpair` is passed in only as a read input — it is the shared-SNP count produced
by the earlier multiply and is carried through unchanged, so it survives for its
later role as the jackknife weight.
