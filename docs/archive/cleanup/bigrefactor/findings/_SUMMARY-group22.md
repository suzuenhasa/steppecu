# GROUP 22 — Performance: compute & launch (rollup)

Tasks: 22.1 atomics where a reduction/scan is cheaper · 22.2 integer div/mod in loops · 22.3 loop-invariant work / repeated index recompute · 22.4 launch overhead (fuse / CUDA-graph, profiler-gated).

## 1. Coverage

13 units in scope (scope = kernel). All 13 reviewed and carry a `## Group 22` section.

| State | Count | Units |
|---|---|---|
| With findings | 4 | cuda_backend, f2_block_kernel, f2_blocks_out, qpadm_fit_kernels |
| Clean (no Group 22 issues) | 9 | block_sink, check, decode_af_kernel, device_buffer, device_f2_blocks, device_partial, f2_blocks_kernel, p2p_combine, pinned_buffer |

The 9 clean units fall into two buckets: (a) host-side / declaration-only TUs with no kernels, launches, atomics, or GPU loops (block_sink, check header, device_buffer, device_f2_blocks, device_partial, p2p_combine, pinned_buffer — and the latter two ARE the amortization fix for their class), and (b) actual kernels that are already optimal for Group 22 (decode_af_kernel: one launch, invariants hoisted, div/mod is compile-time power-of-two; f2_blocks_kernel: IS the launch-overhead FIX — strided-batched GEMMs collapsing ~2271 tiny launches into one-per-bucket, collision-free per-thread writes, no per-thread loops).

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|---|---|---|---|---|
| 22.1 atomics vs reduction | 0 | 0 | 0 | 0 |
| 22.2 div/mod in loops | 0 | 0 | 1 | 1 |
| 22.3 loop-invariant / repeated recompute | 0 | 1 | 2 | 3 |
| 22.4 launch overhead / fuse-or-graph | 0 | 4 | 0 | 4 |
| **Total** | **0** | **6** | **3** | **9** |

22.1 produced ZERO findings across all 13 units and this is structural, not an oversight: every device-side reduction in steppe (f4 LOO/total, jackknife sums, SE-from-wmat, the trace, GLS dot-products) is a deliberate SINGLE-THREAD sequential loop in fixed index order — the §12 deterministic-order parity requirement (G=1 == G=2 bit-identical) — so there is no multi-thread accumulation into a shared cell for an atomic to be papering over. The big Σ-over-SNP contractions are delegated to cuBLAS GEMMs, not hand-rolled atomics. The "atomic where a reduction would be cheaper" anti-pattern cannot arise by design.

## 3. Findings (no HIGH; MED first, then LOW)

### MED (6)

- [22.4][MED] src/device/cuda/cuda_backend.cu:2270-2281 — S8 batched-inverse column loop runs `m`(=nl·nr) iterations; each issues a per-column H2D of B RHS pointers + ONE `cusolverDnDpotrsBatched` on the single statistic stream — many-small-repeated-launches on the hot per-chunk rotation path. The B-pointer arena is fully computable up front. Cross-ref [15.1] (the H2D half). Suggested: pre-upload one `[m×B]` pointer arena before the loop, then fire the m solves back-to-back or capture the column loop into a CUDA graph (profiler-confirmed).
- [22.4][MED] src/device/cuda/cuda_backend.cu:1780-1810 — S5 rank-sweep loop forces a full `cudaStreamSynchronize` (:1804) every iteration to read back a 1-int status + 1-double chisq, turning the sweep into a launch→sync→launch ladder with no overlap between ranks. Same per-iteration host round-trip the LOO Stage-A rewrite explicitly removed (:1976-1979). rmax small (≤~9 small-path) so absolute cost modest. Suggested: accumulate chisq[r]/status[r] into `[rmax+1]` device arrays, do ONE D2H + one sync after the loop.
- [22.4][MED] src/device/cuda/f2_block_kernel.cu:307-384 — a single f2 block = FIVE serialized launches (feeder + 3 cublasGemmEx + assemble). Fine for the single-block M0 path, but at the S8 envelope (thousands of models × n_block≈757) the single-block path would be a launch storm dominated by per-launch dispatch over tiny GEMMs. Suggested: ensure the batched sibling (f2_blocks_kernel) is the scale path (it already is); consider CUDA-graph capture of the feeder→3-GEMM→assemble chain ONLY after nsys/ncu confirms launch overhead dominates.
- [22.4][MED] src/device/cuda/f2_blocks_out.cu:78,88-96 — `read_block_to_host` (FIT per-block tile reader, n_block up to ~757) re-issues `cudaGetDevice`+`cudaSetDevice`+RAII-restore around 2 synchronous D2H `cudaMemcpy` on EVERY block; the device-bind is loop-invariant per the resident buffer's home device yet paid block-by-block. Same seam as [15.1]. Suggested: a multi-block / whole-tensor read that hoists the device-bind out of the per-block path and issues one batched D2H, or let the GPU fit read `f2_device()`/`vpair_device()` in place (no per-block D2H).
- [22.3][MED] src/device/cuda/qpadm_fit_kernels.cu:222-227,240,244 (dev_opt_A B2) and 273-278,291,295 (dev_opt_B A2f); same in *_large siblings 601-605,615,619 and 646-650,660,664 — Kronecker index lambdas recompute their quotient/remainder split (`i=a/r; p=a%r`, `i=k/nr; j=k%nr`) on EVERY call in the innermost GEMM-build loop while the OUTER-index split is loop-invariant — so `a/r`,`a%r` run t·m times per `a` instead of once. Runtime, non-power-of-two divisors (r, nr) = true integer div/mod, the most expensive GPU integer op, in the innermost loop of the per-model fit at S8 scale. Same math/order (no §12 impact). Suggested: hoist the outer-index split out of the inner loop (precompute `i,p` once per `a`, index `B[p+r*j]` directly); the genuinely-per-element inner `k`-decode stays.
- [22.4][MED] src/device/cuda/qpadm_fit_kernels.cu:1461-1537 — the legacy/large per-model fit path chains 4-5 back-to-back `<<<1,1,0,stream>>>` single-thread launches per model (xmat→seed→ALS→weights, each + `STEPPE_CUDA_CHECK_KERNEL()`), each paying full per-launch latency to run ONE thread. The S8 model-batched path already fuses this for the SMALL bucket; the >32/large tail still pays per-launch overhead per model. Suggested (profiler-gated): fuse the per-model seed→ALS→weights single-thread kernels into one `<<<1,1>>>` (or block-per-model), or CUDA-graph the fixed per-model launch sequence and replay per model; verify bit-parity.

### LOW (3)

- [22.3][LOW] src/device/cuda/cuda_backend.cu:2270-2272 — inner `for j` recomputes the loop-invariant per-model base `dQinv.data()+j*Mm` on every column `c` (only `+c*M` varies) — m·B multiplies where B precomputed bases would do. Minor host arithmetic, dominated by the H2D/launch cost above. Suggested: hoist B per-model bases into a vector once before the `for c` loop (mirrors the existing `h_Aptr` precompute at :2252).
- [22.2][LOW] src/device/cuda/qpadm_fit_kernels.cu:397-400,471-472,704-705,962-967,992-993,1037-1038,1078-1079,1206-1207,1234-1235,1270-1271 — every grid-stride element-wise kernel decodes its flat index with runtime integer div/mod (`model=gid/per; idx=gid%per`, `k=idx%m`, `i=k/nr; j=k%nr`, etc.) INSIDE the grid-stride loop body. Divisors (m, nr, per, nb, nl) are runtime non-power-of-two so no shift/mask substitution; at the typical envelope the grid covers the work in one pass and the div/mod is hidden under global-load latency, so the cost is real but second-order. Suggested: optional — only if a kernel ever runs many stride iterations per thread, maintain the running (model,idx) by addition; leave unless a profiler flags it.
- [22.3][LOW] src/device/cuda/f2_block_kernel.cu:120,124,128 — `Q_raw[idx]` fetched at three sites in the feeder (and `V_raw[idx]` read then re-expressed) rather than hoisted into one register. No loop here (one thread = one element) and under `const double* __restrict__` the compiler near-certainly CSEs each into a single LDG — hygiene, not a measured re-read (also under Groups 7.2/20.3). Suggested: hoist `const double qraw = Q_raw[idx];` once, feed both the mask and `het_correction`.

## 4. Cross-cutting patterns

1. **22.4 dominates (4 of 6 MED) and the signature is identical: a per-iteration host↔device round-trip / per-element launch on a hot loop that should be batched or arena-uploaded once.** Three of the four are the SAME structural wart at different sites — cuda_backend rank-sweep (:1780-1810) and column-inverse (:2270-2281), and the qpadm_fit_kernels single-thread tail (:1461-1537) — all "the small/rotation path is already fused, but THIS path (rank sweep / large+>32 tail / single-block) still pays per-iteration launch+sync." The codebase has already proven the fix direction internally: the LOO Stage-A rewrite removed "the dominant 701× host round-trip" and the f2 strided-batched GEMM collapsed ~2271 launches into one-per-bucket. These findings are the remaining un-batched siblings.

2. **The fix for 22.4 here is consistently "batch / upload-once / resident-consume," NOT CUDA-graph as a first resort.** Every 22.4 finding is profiler-gated per the task rule, and several units correctly argue a graph is the WRONG tool (data-dependent block indexing + the §12 deterministic-sync requirement would have to be preserved in the capture). The cheaper, parity-neutral win in each case is removing the interleaved per-iteration H2D/sync (pre-upload a pointer arena, accumulate results into a `[rmax]` device array, one batched D2H).

3. **22.4 findings overlap the Group 15 transport seam.** cuda_backend:2270 ([15.1]) and f2_blocks_out:78 ([15.1]/[15.3]) are the launch-COUNT angle of transfers already filed under Group 15; not double-counted — same root, the device-resident fit reading f2 from VRAM (cf. MEMORY: "real fix is the Phase-2 fit engine reading f2_blocks from VRAM, not more D2H tuning").

4. **22.2/22.3 div/mod findings cluster in qpadm_fit_kernels Kronecker index decode.** The one MED (B2/A2f outer-split recomputed innermost) is a genuine hoist; the LOW (grid-stride index decode) is bandwidth-hidden. All are parity-neutral (same op order), runtime non-power-of-two divisors so no shift/mask trick applies.

5. **No HIGH and no 22.1 anywhere — the §12/FP64 design is doing its job.** Single-thread deterministic reductions (intentional, parity-load-bearing) mean there is no atomic-vs-reduction surface; no integer-overflow-at-scale or UB surfaced under the Group 22 lens (those, if present, belong to the index-widening groups). Every Group 22 finding is a latent perf-at-scale concern on a path the codebase already knows is the non-fused sibling — none changes a result.

## Headline numbers

- Units in scope: 13 (4 with findings, 9 clean)
- Total findings: 9 (0 HIGH, 6 MED, 3 LOW)
- HIGH: 0
- By task: 22.1 = 0 · 22.2 = 1 · 22.3 = 3 · 22.4 = 4
