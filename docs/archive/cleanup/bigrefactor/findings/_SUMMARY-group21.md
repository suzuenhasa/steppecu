# Group 21 — Performance: occupancy & registers — ROLLUP

Scope: kernel-level review of the CUDA backend (`src/device/cuda/*`). Tasks:
21.1 warp divergence (perf), 21.2 excessive shared memory cratering occupancy,
21.3 register spills / monolithic-kernel split candidates, 21.4 missing register
hints (`#pragma unroll`, `__launch_bounds__`, `__forceinline__`) — applied
DELIBERATELY, not reflexively.

FP64/§12 context applied: native + Ozaki emulated FP64 and the single
deterministic statistic stream are intentional and were NOT flagged. The
data-dependent singular-pivot branch in `dev_lu_factor`/`dev_solve` is an
intentional §12-determinism choice (cannot be replaced with a predicated
divergence-free path) and was correctly NOT flagged as divergence.

## 1. Coverage

- Units in scope: 13 (all carry a `## Group 21` section).
- Clean (no findings): 10
  - block_sink, check, cuda_backend, device_buffer, device_f2_blocks,
    device_partial, f2_block_kernel, f2_blocks_out, p2p_combine, pinned_buffer
  - Of these, 9 are host-side orchestration / RAII / transport / I/O TUs with
    ZERO `__global__` kernels — all four tasks are vacuous (N/A) there.
  - f2_block_kernel is the lone real-kernel unit that came back fully clean.
- With findings: 3
  - decode_af_kernel (1 LOW), f2_blocks_kernel (1 LOW), qpadm_fit_kernels (1 MED + 2 LOW)

## 2. Counts by task + severity

| Task                        | HIGH | MED | LOW | Total |
|-----------------------------|------|-----|-----|-------|
| 21.1 warp divergence        |  0   |  0  |  0  |   0   |
| 21.2 excessive shared mem   |  0   |  0  |  0  |   0   |
| 21.3 register spills / split|  0   |  1  |  0  |   1   |
| 21.4 missing register hints |  0   |  0  |  3  |   3   |
| **Total**                   | **0**| **1**| **3**| **4** |

Totals: 13 units, 4 findings — 0 HIGH / 1 MED / 3 LOW.

21.1 and 21.2 are clean across the entire backend: no kernel has a heavy
divergent warp body (the only data-dependent branches are tiny/predicated or
the unavoidable §12 singular-pivot), and no kernel uses occupancy-cratering
shared memory (the sole `__shared__` anywhere is an 8-byte scalar `s_tr` in
`add_fudge_diag_models_kernel`; every `<<<>>>` launches with 0 dynamic smem).

## 3. Top findings (HIGH first)

No HIGH findings.

MED (1):
- [21.3][MED] src/device/cuda/qpadm_fit_kernels.cu:1091-1191
  (`qpadm_fit_models_kernel`) — the monolithic per-model fit: ONE thread runs
  full-rank fit + a rank sweep r=0..rmax (each a full seed→ALS→weight→chisq via
  `dev_als_weights`) + an nl+1× popdrop lambda. Per-thread LOCAL frame dominated
  by the reduced-Qinv `qr[50*50]` (~20 KB, :1151) plus stacked large arrays from
  inlined `dev_seed_ab`/`dev_opt_A`/`dev_opt_B`; the launch block is only 64
  (:1358) precisely because per-thread local/register pressure is so high. At
  the S8 envelope (n_models in the millions) this is the dominant occupancy
  limiter. NOT a correctness bug (op order + bit-parity intact, frame is
  documented-launchable). Suggested: split full-rank fit / rank sweep / popdrop
  into separate model-batched (and (model,r)-grid) kernels with smaller frames;
  re-verify §12 bit-parity after any split.

LOW (3):
- [21.4][LOW] src/device/cuda/qpadm_fit_kernels.cu:1091, :1198, :832, :922 —
  none of the high-register per-thread-fanout kernels (`qpadm_fit_models_kernel`,
  `qpadm_loo_models_kernel`, `loo_large_batched_kernel`, `loo_batched_kernel`)
  carries `__launch_bounds__`; launched at hand-picked block=64/128 with no
  register-budget hint. Suggested: add `__launch_bounds__(N)` matching each
  launcher's block; measure with `-Xptxas -v`/ncu before/after.
- [21.4][LOW] src/device/cuda/qpadm_fit_kernels.cu:66-110, :118-176,
  :228-246/279-297, :327-331 — tight single-thread LA inner loops (dev_solve
  back/forward subst with n<=5, dev_opt_A/B coeffs/rhs dot products) have no
  `#pragma unroll` though their SMALL-bucket trip counts are tiny and bounded.
  Suggested: add `#pragma unroll` ONLY to the innermost tiny-bound loops; leave
  the nb-length (up to 757) jackknife reductions rolled.
- [21.4][LOW] src/device/cuda/decode_af_kernel.cu:56 (`decode_af_kernel`) —
  fixed 256-thread launch (32×8) carries no `__launch_bounds__(256)`; adding it
  pins the occupancy contract against a future codegen regression. LOW because
  the kernel is bandwidth-bound with a tiny register footprint (no spill risk).
  Suggested: `__launch_bounds__(kDecodeBlockX * kDecodeBlockY)`.
- [21.4][LOW] src/device/cuda/f2_blocks_kernel.cu:69, :129 — `gather_group_kernel`
  and `assemble_blocks_group_kernel` have a compile-time-known 256-thread block
  (16×16) but no `__launch_bounds__`. Both are short straight-line low-register
  map/scatter, so occupancy is likely not register-limited. Suggested: add
  `__launch_bounds__(kCdivBlock*kCdivBlock)` only if a measurement shows wasted
  register headroom; otherwise leave as-is.

(The 21.4 LOW count above lists 4 bullets but is 3 findings + the f2_blocks one
= 4 distinct LOWs across files: decode_af 1, f2_blocks 1, qpadm_fit 2. The
qpadm_fit `__launch_bounds__` and `#pragma unroll` bullets are its 2 LOWs.)

## 4. Cross-cutting patterns

1. The entire occupancy/register concern set collapses onto ONE file:
   `qpadm_fit_kernels.cu`. It is the only TU with a genuine MED — the
   per-model fit monolith whose large per-thread LOCAL frame (the
   ALS/seed/opt nest, esp. the 50×50 reduced-Qinv) forces block=64 and becomes
   the occupancy wall exactly at the S8 (millions-of-models) scale the project
   says to design for. Every other kernel is a light, straight-line map/scatter
   or a bandwidth-bound front-end.

2. Pervasive missing `__launch_bounds__` on fixed-block-size kernels (21.4):
   decode_af (256), f2_blocks pair (256), and the four qpadm fanout kernels
   (64/128) all launch at compile-time-known block sizes with no register-budget
   hint, leaving ptxas free to over-allocate registers and silently regress
   occupancy. This is the single recurring hygiene theme — a deliberate,
   measure-then-pin guard, not a reflexive add. Complements Group 12 [12.4]
   (block size not occupancy-derived) from the register-hint side.

3. Deliberate hints already correct (NOT flagged): `dev_opt_A`/`dev_opt_B` are
   intentionally `__device__ __noinline__` to stop their large frames stacking
   in the ALS loop; `loo_large_batched_kernel` deliberately moves big arrays
   OUT of local into a VRAM scratch arena (the launch-OOM fix) so it is NOT a
   spill candidate; inner-loop `#pragma unroll` is correctly ABSENT on the
   runtime/large-trip-count reductions (decode segment loop, nb jackknife).

4. 21.1 and 21.2 are structurally clean by design: lane-uniform branching
   (adjacent lanes are same-shape bucketed models / same column), the §12
   single-stream determinism, and the zero-shared-memory map/scatter +
   per-thread-local-or-VRAM LA workspace pattern mean neither divergence nor
   shared-memory pressure limits occupancy anywhere in the backend.
