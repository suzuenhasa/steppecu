# `pca_truncated_eig.cu` reference

## 1. Purpose

This file is the eigensolver behind `steppe pca`. You hand it the N×N genetic
relationship matrix (the GRM — a device-resident, column-major, symmetric
positive-definite covariance over N samples) and a number `K`, and it hands
back the top `K` eigenpairs: the leading principal components and their
eigenvalues. Those are the axes a PCA plot is drawn on.

The catch it exists to solve is memory. The obvious way to get eigenvectors is
to ask cuSOLVER for the *whole* spectrum with `cusolverDnDsyevd` in vector mode.
That works fine until N gets large, and then its workspace — roughly `2·N²·8`
bytes, about 8.5 GB at the full AADR cohort of N ≈ 23,089 — becomes the single
allocation that pushes the run out of memory, all to compute a top-10 ask. This
file replaces that full solve with a *randomized* one that only ever touches a
thin N×L slab (L a few hundred), so the eigensolve stops being the memory wall.
Extra workspace drops from O(N²) (~8.5 GB) to O(N·L) (~190 MB at N=23k, L=256).

The single entry point is `pca_truncated_topk`. Everything else in the file is
two small device helpers it leans on.

---

## 2. The randomized Rayleigh-Ritz pipeline

The method is Halko 2011 (Algorithms 4.4 + 5.3) — randomized subspace iteration.
The idea in one breath: instead of diagonalizing the giant N×N matrix `C`, find a
thin orthonormal basis `Q` (N×L) that captures the subspace `C`'s top eigenvectors
live in, then diagonalize the tiny L×L matrix `C` looks like *inside* that basis.
Every step below stays device-resident:

1. **Sketch.** Draw a Gaussian random matrix `Omega` (N×L) and form `Y = C·Omega`.
   Multiplying by `C` pulls `Omega` toward `C`'s dominant eigenvectors — a random
   probe that lights up the top of the spectrum.
2. **Orthonormalize.** `Q = orth(Y)` via thin QR. `Q` (N×L) is now an orthonormal
   basis for the sampled subspace.
3. **Subspace iteration.** Repeat `subspace_iters` times: `Q = orth(C·Q)`. Each
   pass applies `C` again and re-orthonormalizes, sharpening the basis so the
   closely-spaced PCs at the tail of the top-L separate cleanly. This is done as
   *apply-then-reorthonormalize* on purpose — it never forms `C·C·Omega` as a raw
   product, which would let the trailing PCs drown in rounding.
4. **Project (Rayleigh-Ritz).** Form the small matrix `B = Qᵀ(C·Q)` (L×L). `B` is
   what `C` looks like restricted to the `Q` subspace.
5. **Small eigen.** Diagonalize `B` with `Dsyevd` — its eigenvalues are the *Ritz
   values* (the top-L eigenvalue estimates of `C`, ascending) and its eigenvectors
   `Yb` are the coordinates of `C`'s eigenvectors in the `Q` basis.
6. **Lift.** `V = Q·Yb` (N×L) turns those coordinates back into full-length
   eigenvectors of `C`.

The result is `L = min(K + oversample, N)` Ritz pairs, ascending by eigenvalue, so
column `L-1` of the output is the top PC. The caller keeps the top `K` of them; the
extra `oversample` columns are slack that makes the top `K` more accurate (the
tail of a truncated basis is always the least trustworthy part).

---

## 3. Two device helpers

Two file-local functions carry the heavy lifting, and they exist as separate
functions partly so the precision policy (section 4) can differ between them.

- **`sketch_CQ`** — the workhorse GEMM `C_out = dC · Q_in` (N×N times N×L). This is
  the matmul-heavy sketch path, and it is called `2 + subspace_iters` times: once
  for the initial `Y = C·Omega`, once per subspace-iteration pass, and once more to
  build `C·Q` before the projection. It runs **emulated-FP64** (see section 4).
- **`orthonormalize`** — a thin Householder QR that orthonormalizes an N×L matrix
  *in place*, using `cusolverDnDgeqrf` (the factorization) followed by
  `cusolverDnDorgqr` (materializing `Q`). It sizes its own workspace from both
  buffer-size queries, takes the max, and floors it at 1 so an empty query never
  asks for a zero-byte allocation. It reads back cuSOLVER's `info` and **returns
  `false` on any nonzero value** — that is how a non-SPD or numerically broken `C`
  surfaces as a clean failure rather than garbage vectors. It runs native-FP64.

---

## 4. Precision: emulated sketches, native carve-outs

This file follows steppe's split precision policy exactly, and the split falls
along "is it a big matmul or a small reduction/factorization."

- **Emulated-FP64 (the default) for the sketch GEMMs.** Every `sketch_CQ` call and
  the final lift `V = Q·Yb` set `CUBLAS_PEDANTIC_MATH` and call
  `engage_f2_precision`, so they run on the Ozaki emulated-FP64 path. These are the
  O(N²·L) matmuls — the matmul-heavy work the emulated default is built for.
- **Native-FP64 for the carve-outs.** Three places drop to real `double`:
  - the **QR** inside `orthonormalize` (`CusolverMathModeScope`, not honorable);
  - the **projection** `B = Qᵀ(C·Q)`, which sets `CUBLAS_PEDANTIC_MATH` but does
    *not* call `engage_f2_precision`, so it stays native. This one is the
    cancellation-sensitive inner-product reduction (O(N·L²)) whose values *are* the
    Ritz values, so it gets the native carve-out rather than emulation;
  - the **`Dsyevd`** on `B`.

  Crucially the native eigen/QR carve-out now runs on the *tiny* L×L (or N×L) work,
  not the 23k×23k GRM — that is the whole point. The expensive part is emulated; the
  precision-critical-but-small part is native.

If you are writing or auditing a precision note for this file: emulated-FP64 for
the sketch/lift GEMMs, native-FP64 for QR, the `QᵀCQ` projection, and the small
`Dsyevd`. Never call the whole thing "native" or "emulated" — it is deliberately
both.

---

## 5. Contracts and invariants

- **`dC` is read-only.** The GRM is never overwritten. That is a real contract, not
  an accident: the caller still needs `trace(dC)` intact as the denominator for
  `var_explained`, so this routine must not clobber it.
- **Outputs.** `d_evecL` is column-major N×L, the lifted Ritz vectors ascending by
  Ritz value (column `L-1` is the top PC). `d_evalL` is length L, the ascending Ritz
  values. `out_L` (if non-null) receives the actual `L`. The caller sizes all three;
  this routine allocates only its own scratch.
- **`L = min(K + oversample, N)`.** `oversample` and `subspace_iters` are clamped up
  to 0 if passed negative, so bad tuning values degrade gracefully rather than
  underflowing a size.
- **Stream binding.** `blas` must already be bound to `stream` by the caller;
  `solver`'s stream is not re-set here (the backend owns that).
- **Deterministic sketch.** `Omega` is filled from a fixed seed
  (`kPcaOmegaSeed`), so two runs on the same GRM produce the same PCs bit-for-bit.
  That reproducibility is what makes the truncated solve golden-parity friendly —
  the randomness is only "random" in the linear-algebra sense, not run-to-run.
- **Ping-pong, no aliasing.** Subspace iteration writes `C·Q` into a separate
  `dScratch`, orthonormalizes that, then `std::swap`s the buffers so `dQ` always
  names the current basis. `sketch_CQ`'s input and output are always distinct
  buffers — cuBLAS GEMM does not allow the output to alias an input.

---

## 6. The `B` upper triangle is intentionally ignored

Step 5's `Dsyevd` is told `CUBLAS_FILL_MODE_LOWER`, so it reads **only the lower
triangle** of `B`. That matters because `B = Qᵀ(C·Q)` is formed by a GEMM, not a
symmetric product, and the emulated rounding upstream can leave `B` very slightly
asymmetric. Because the eigensolver never looks at the upper triangle, that
asymmetry is simply harmless — there is no need to symmetrize `B` first, and the
file deliberately skips that step.

---

## 7. Failure modes and edge cases

- **Invalid config.** `N <= 0`, `K <= 0`, or `K > N` returns immediately with
  `Status::InvalidConfig` and does no work.
- **Non-SPD / broken covariance.** Any nonzero cuSOLVER `info` — from either QR pass
  or the final `Dsyevd` — is reported as `Status::NonSpdCovariance`. This is the
  catch-all for a GRM that isn't actually positive-definite, or a factorization that
  numerically fell over: the routine returns the partial `out` with that status
  rather than emitting nonsense eigenvectors.
- **Success** returns `Status::Ok` with `out.L == L`.
- Each cuSOLVER `info` check is a small D2H copy plus a `cudaStreamSynchronize` —
  these are the only host syncs in the routine, and they exist precisely so a bad
  factorization is caught before its output is used downstream.
