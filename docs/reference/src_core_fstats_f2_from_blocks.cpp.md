# `f2_from_blocks.cpp` reference

## 1. Purpose

`src/core/fstats/f2_from_blocks.cpp` is the host-side orchestration for the "f2"
computation — the step that turns per-population, per-SNP genotype summaries into a
bias-corrected f2 matrix (and, in the batched form, a per-block f2 tensor used for
jackknife error bars).

This file does not compute anything itself. It is a thin *composition root*: it owns
the policy decisions (which precision to use, how SNPs are grouped into blocks) and
it enforces the input contract, then hands the actual math off to an injected
`ComputeBackend`. The same orchestration code runs unchanged against two backends:

- **`CpuBackend`** — the plain-scalar reference implementation, used as the oracle
  that everything else is validated against.
- **`CudaBackend`** — the GPU implementation, which recasts the f2 computation as a
  small number of large matrix multiplies.

Because the backend is passed in as a parameter rather than chosen inside this file,
the whole pipeline can be exercised and tested without a GPU present. This file
itself never issues a GPU call, never allocates device memory, and never includes a
CUDA header. It reaches the GPU only through the CUDA-free `ComputeBackend`
interface. That property — driving the GPU work purely through an injected,
CUDA-free seam — is what keeps the orchestration pure host C++ and independently
testable.

The clean split of responsibilities is the point:

- **This file owns:** the input contract (what a valid set of inputs looks like),
  the block policy, and the precision policy.
- **The backend owns:** *how* the f2 is actually computed.

---

## 2. The two entry points

Both public functions live here. They share the same shape: validate the inputs,
then dispatch to the backend.

### `compute_f2_block` — one SNP block

```
F2Result compute_f2_block(ComputeBackend& backend,
                          const MatView& Q, const MatView& V, const MatView& N,
                          const Precision& precision);
```

Drives a single block of SNPs. The three input views `Q`, `V`, and `N` are the
genotype contract: each is a `P × M` matrix (P populations, M SNPs) and the three
must describe the same P and M. The result (`F2Result`) carries two things:

- the bias-corrected f2 matrix, `P × P`, and
- `Vpair`, the retained count of pairwise-valid SNPs, which is later used as a
  weight in the block jackknife.

### `compute_f2_blocks` — the full per-block tensor

```
F2BlockTensor compute_f2_blocks(ComputeBackend& backend,
                                const MatView& Q, const MatView& V, const MatView& N,
                                const BlockPartition& partition,
                                const Precision& precision);
```

Drives the whole genome at once. It takes the full per-SNP `Q`/`V`/`N` plus a
`BlockPartition` — a rule that says which jackknife block each SNP belongs to — and
produces a `P × P × n_block` tensor: one f2 matrix per block. The backend batches
the work over the block axis. The `partition` comes from the single shared
block-assignment rule (`assign_blocks`), so the same block ids are used everywhere.

### `precision`

The `precision` argument selects the arithmetic mode for the matrix-multiply-heavy
GPU work. It only affects the GPU path; the CPU oracle ignores it and always runs in
its own reference arithmetic.

### The seam is deliberately narrow

Notice that `compute_f2_blocks` does not pass the `BlockPartition` struct across to
the backend. It passes the raw pointer and the block count:

```
backend.compute_f2_blocks(Q, V, N, partition.block_id.data(),
                          partition.n_block, precision);
```

Handing over a `(const int*, int)` pair instead of the struct keeps the
`ComputeBackend` interface free of any standard-library container type, which is
what lets that interface stay CUDA-free and ABI-simple. The cost of that narrow seam
is that it *erases* information the struct carried — most importantly the length of
the block-id array and its ordering guarantees. Section 3 explains why that erasure
is the reason the input contract is checked in this file.

---

## 3. Why the input contract is checked here

The backend cannot fully police its own inputs, and this file is the one place that
can. Two facts combine to make this the right home for validation:

1. **This is the only point that sees everything together.** When `compute_f2_blocks`
   runs, the full `Q`/`V`/`N` views and the complete `BlockPartition` (including the
   owning vector, so its `.size()` is knowable) are all in scope at once.

2. **The seam throws information away.** Once the partition crosses into the backend
   as a bare `(const int*, int)` pair, the length of the block-id array is gone. The
   backend builds a view over the raw pointer it was handed and trusts that pointer,
   so it *cannot* detect a short array, a null pointer paired with a nonzero block
   count, or an array whose length disagrees with the SNP count.

The backend does re-check what it still can see — for example, that every block id
is in range and that ids are non-decreasing. But it does that only *after* the data
has already crossed the seam, and it structurally cannot check the length or the
null-pointer case. If a malformed partition slipped through, the failure would show
up as a silent out-of-bounds read or write deep inside the backend, far from the
real cause.

Checking here — while the owning vector is still in scope and the true length is
known — keeps a fault attributable to the orchestration's own input, and turns a
memory-corruption bug into a clean, located abort. The same reasoning applies to the
`Q`/`V`/`N` views: this file is where their mutual agreement can be verified before
any backend trusts them.

---

## 4. The Q/V/N precondition (`assert_qvn_consistent`)

`assert_qvn_consistent` (defined in `core/internal/qvn_assert.hpp`) enforces the
shared precondition that the backend documents but assumes rather than checks. Both
entry points call it first. It asserts three things:

- `Q`, `V`, and `N` agree on `P` (the population count).
- `Q`, `V`, and `N` agree on `M` (the SNP count).
- `P` and `M` are both non-negative.

The non-negative check matters because a negative extent is the signature of an
uninitialized or garbage view. If a negative `P` or `M` reached the backend it would
be cast to an enormous unsigned size and drive a huge, wrong allocation. The view
type does no bounds checking on element access, so catching a mismatch here prevents
a read past the end of a too-short view.

This check now lives in the shared `assert_qvn_consistent` helper
(`core/internal/qvn_assert.hpp`) rather than as a copy local to this file, so it is
reused across the f2 files (for example `f2_blocks_multigpu.cpp`) as well as by both
entry points here — the single-block and batched paths cannot drift apart on what
"valid inputs" means.

---

## 5. The block-partition contract (`validate_partition`)

`validate_partition` enforces everything the block partition must satisfy before it
crosses the seam. It is called only by `compute_f2_blocks`, and it is given `M`
(taken from `Q.M`) as the SNP count the backend will trust. The partition must
describe exactly those columns. The checks, in order:

1. **Length equals `M`.** `partition.block_id.size()` must be exactly `M` (treating a
   negative `M` as zero). The block-id array runs parallel to the SNP columns, one
   entry per SNP. This check also covers the null-pointer case for free: an empty
   vector paired with a nonzero block count fails the size check before anyone
   dereferences its data, which is important because an empty vector's data pointer
   may legitimately be null.

2. **Positive block count when there are SNPs.** If `M > 0` then `n_block` must be
   greater than zero.

3. **No more blocks than SNPs.** `n_block` may not exceed `M`, because every block
   holds at least one SNP.

4. **Dense and non-decreasing ids.** Every id must lie in `[0, n_block)` and the
   sequence must never decrease. This is the invariant the block-assignment rule
   guarantees: each block's SNPs form one contiguous run, so the ids only ever hold
   steady or step up as you scan the columns in order.

Check 4 is delegated to a helper, `block_ids_dense_nondecreasing`, which does a
single forward scan over all `M` ids: it rejects any id that is out of range or that
is smaller than the previous id. It is an O(M) pass.

The backend's own machinery re-validates the range and ordering, but again, only
after the seam has erased the length and only from the pointer it was handed.
Catching all four conditions here keeps the fault attributed to this file's input.

---

## 6. The debug-only fail-fast mechanism

All of the checks in sections 4 and 5 run through `STEPPE_ASSERT`. That facility is
**compiled out entirely under a release (`NDEBUG`) build**. The intent is:

- In a debug build (or under a debugger or a memory checker), a violated contract
  aborts immediately with a file and line number and a human-readable message,
  right at the point the bad input entered the pipeline.
- In a release build, the checks vanish, so the hot path pays nothing at all. In
  release, the caller is responsible for supplying valid inputs.

Compiling the checks out cleanly, without tripping the build's
warnings-as-errors policy, takes two small deliberate tricks that are worth
understanding because they look odd at first glance:

- **`[[maybe_unused]]` on the parameters.** When the asserts disappear under
  `NDEBUG`, the validation functions' parameters become unreferenced. Marking them
  `[[maybe_unused]]` tells the compiler that is expected, so an "unused parameter"
  warning does not fail the release build.

- **The dense/non-decreasing scan helper stays defined in every build.** Even though
  `block_ids_dense_nondecreasing` is only ever *called* from inside a
  `STEPPE_ASSERT`, its definition is unconditional and marked `[[maybe_unused]]`. In
  release, the assert still references the helper's name in an unevaluated way (a
  `sizeof`-style reference used to mark assert-only locals as "used"), so the name
  must still resolve — but because that reference never actually evaluates the call,
  the optimizer drops the function body and the O(M) scan never runs in release.

The net effect: full contract enforcement with clear diagnostics in debug, zero
runtime cost and a warning-clean compile in release.

---

## 7. CUDA-free layering

The file is deliberately kept pure host C++ with no GPU dependency. It includes only:

- the CUDA-free `ComputeBackend` seam (and the `F2Result` it returns),
- the shared host-only `Q`/`V`/`N` view type,
- the CUDA-free index-cast helpers (`core/internal/index_cast.hpp`, for `idx` /
  `nonneg_count`),
- the CUDA-free Q/V/N consistency helper (`core/internal/qvn_assert.hpp`, for
  `assert_qvn_consistent`),
- the host-only block-partition rule and the public `F2BlockTensor` result type,
- the debug-assert facility, and
- the public configuration header (for the `Precision` type).

No CUDA header is pulled in anywhere. This is what lets the file compile into the
pure-host `core` library without dragging the GPU toolkit along, and it is the same
discipline that makes the orchestration testable against the CPU backend with no GPU
present.
