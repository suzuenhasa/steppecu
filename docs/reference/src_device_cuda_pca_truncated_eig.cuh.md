# `pca_truncated_eig.cuh` reference

## 1. Purpose

`src/device/cuda/pca_truncated_eig.cuh` declares one function —
`pca_truncated_topk` — the single seam between the CUDA PCA backend and the
randomized top-K eigensolver that finds the leading principal components of the
`steppe pca` GRM. It is a small, declaration-only header: a struct
(`PcaTruncatedEig`) carrying the result, and the one host entry point. The whole
algorithm — the sketch, the QR, the subspace iterations, the tiny projected
eigensolve, and the lift — lives in the paired `pca_truncated_eig.cu`; this
header is what the backend `#include`s to reach it. For the step-by-step pipeline
and the precision carve-out, read that `.cu` and its reference doc; this doc
covers the *contract* the header promises.

The header exists because the naive path OOMs. The old PCA solve was a
full-spectrum `cusolverDnDsyevd(CUSOLVER_EIG_MODE_VECTOR, N)` on the `N x N` GRM,
and its workspace (roughly `2*N^2*8` bytes, about 8.5 GB at `N = 23,089`) is the
one marginal allocation that pushes the full AADR cohort over the edge for a
top-K = 10 ask. The randomized Rayleigh-Ritz solve declared here allocates only
`O(N*(K+p))` extra workspace — about 190 MB at `N = 23k` with the default L of
256 — so the eigensolve stops being the memory wall.

Because it names `cublasHandle_t`, `cusolverDnHandle_t`, and `cudaStream_t`, the
header pulls in cuBLAS/cuSOLVER and stays private to the `steppe_device` target.
It was factored out of `cuda_backend_pca.cu` on purpose: with the truncation math
behind one clean function, it can be unit-tested on a planted spectrum without
standing up a full backend.

---

## 2. What the function computes

`pca_truncated_topk` is a truncated top-K eigensolve of a device-resident,
column-major `N x N` SPD matrix `dC` (the GRM). It is the Halko-2011 randomized
subspace-iteration method (Alg. 4.4 + 5.3): draw a deterministic Gaussian sketch
`Omega` (`N x L`, `L = min(K + oversample, N)`), form `Y = C*Omega`, orthonormalize
to `Q`, run a few subspace-iteration passes (`Y = C*Q`, re-orthonormalize) to
sharpen the closely-spaced trailing PCs, project down to the tiny `B = Q^T C Q`
(`L x L`), eigendecompose `B`, and lift the eigenvectors back with `V = Q * B_evecs`.

The `L` Ritz values are the top-`L` eigenvalue estimates of `C`, returned
**ascending**, and the top `K` well-separated ones are the principal components.
It hands back the full `L`-wide subspace, not just `K` columns, so the caller can
pick and inspect the oversampled tail.

---

## 3. The call contract

```cpp
PcaTruncatedEig pca_truncated_topk(cublasHandle_t blas, cusolverDnHandle_t solver,
                                   cudaStream_t stream, const double* dC, int N, int K,
                                   int oversample, int subspace_iters,
                                   const Precision& precision, double* d_evecL,
                                   double* d_evalL, int* out_L);
```

- **`dC`** — device, column-major `N x N`, SPD, leading dim `N`. It is
  **read-only**: the solve never overwrites it, on purpose, so `trace(dC)` stays
  valid for the caller's `var_explained` denominator after the call.
- **`N`, `K`** — matrix size and the number of requested top eigenpairs
  (`K <= N`).
- **`oversample`, `subspace_iters`** — the randomized-SVD tuning knobs (`p` and
  `q`). They set `L = min(K + oversample, N)` and how many refinement passes run.
- **`precision`** — the sketch-GEMM precision policy (see section 5).
- **`d_evecL`** — device out, column-major `N x L`: the `L` lifted Ritz vectors,
  ordered by **ascending** Ritz value, so column `L-1` is the top PC. The caller
  sizes it `N*L`.
- **`d_evalL`** — device out, length `L`: the ascending Ritz values. Caller sizes
  it `L`.
- **`out_L`** — host out, the actual `L` used (equal to `result.L`).

The returned `PcaTruncatedEig` carries `L` (the actual subspace width) and a
`Status`. The `blas` handle must already be bound to `stream` by the caller; the
`solver` handle's stream is set inside the function.

---

## 4. Ascending order and the N×L output shape

Two output conventions are easy to trip on, so name them plainly. First, the
eigenpairs come back **ascending** by Ritz value — the top PC is the *last*
column (`L-1`), not the first — matching how the rest of steppe's eigen paths
report. Second, both outputs are the full width `L = min(K + oversample, N)`, not
`K`. `L` is at least `K` (oversampling only adds columns) but is capped at `N`, so
for a small GRM where `K + oversample` would exceed `N` the caller gets the whole
spectrum. Always size `d_evecL` at `N*L` and `d_evalL` at `L`, and read `out_L`
back rather than assuming `K + oversample`.

---

## 5. Precision: emulated where it's matmul, native where it cancels

The precision split mirrors steppe's SYRK/eigen carve-out, and the `precision`
argument only steers the matmul-heavy part. The big `O(N^2*L)` sketch GEMMs
(`C*Omega`, `C*Q`, the final lift) run **emulated-FP64** (Ozaki) — that is the
matmul-heavy house default, and it is what `precision` controls. The small,
cancellation-sensitive pieces — forming `B`, the QR orthonormalization, and the
`L x L` `Dsyevd` — run **native FP64** as the carve-out. The key point the
truncation buys precision-wise: that native-FP64 eigensolve now runs on the tiny
`L = 256` matrix by default, not on the 23k GRM, so it is cheap and well
conditioned. If you write a precision note for this path, say emulated-FP64 for
the sketch GEMMs and native-FP64 for the projected `B`-eigen, and say *why*
(matmul vs. cancellation).

---

## 6. Contracts and invariants at a glance

- **`dC` is read-only.** The solve does not overwrite the GRM; `trace(dC)` stays
  valid for the variance-explained denominator afterward.
- **Ascending Ritz order.** Column `L-1` of `d_evecL` (and entry `L-1` of
  `d_evalL`) is the top PC.
- **Full `L`-wide output.** `L = min(K + oversample, N)`; size outputs at `N*L`
  and `L`, and trust `out_L`/`result.L` rather than recomputing it.
- **`K <= N`.** The requested count cannot exceed the matrix size.
- **Handle/stream binding.** `blas` must be bound to `stream` by the caller; the
  function binds `solver` to `stream` itself.
- **Failure signal.** Returns `Status::NonSpdCovariance` if any cuSOLVER
  factorization reports a nonzero `info` (a non-SPD or degenerate `dC`).
- **Emulated sketch, native projected eigen** — the matmul path honors
  `precision`; the QR / `B` / `Dsyevd` carve-out is native FP64 (section 5).
- **Header stays private.** It names cuBLAS/cuSOLVER/CUDA stream types, so it is
  internal to `steppe_device`; the implementation is in `pca_truncated_eig.cu`.
