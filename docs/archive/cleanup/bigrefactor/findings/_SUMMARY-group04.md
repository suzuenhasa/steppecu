# Group 4 — Type & Numeric — Roll-up Summary

Scope: the steppe big-refactor unit review, Group 4 (type & numeric: float-vs-double,
index width, allocation sizing, unsigned countdown, signed/unsigned compares, int
overflow before widening, host/device pointer typing). steppe is FP64-by-design, so
task 4.1 (float narrowing) is N/A by policy unless a genuinely wrong narrowing in a
parity-critical path is found — none was.

## 1. Coverage

| Metric | Count |
|---|---|
| Units reviewed | 61 |
| Clean (no Group-4 issues) | 50 |
| With at least one finding | 11 |
| Total findings | 12 |

## 2. Findings by task and severity

| Task | HIGH | MED | LOW | Total |
|---|---|---|---|---|
| 4.1 float-vs-double | 0 | 0 | 0 | 0 |
| 4.2 index width (`int` global index > 2^31) | 0 | 0 | 2 | 2 |
| 4.3 allocation sizing (`* sizeof(T)`) | 0 | 0 | 0 | 0 |
| 4.4 unsigned countdown | 0 | 0 | 0 | 0 |
| 4.5 signed/unsigned loop-bound compare | 0 | 0 | 0 | 0 |
| 4.6 int overflow before widening | 0 | 0 | 1 | 1 |
| 4.7 host/device pointer typing | 0 | 0 | 9 | 9 |
| **Total** | **0** | **0** | **12** | **12** |

Severity totals: **0 HIGH, 0 MED, 12 LOW.**

The scale-critical concerns (4.2 / 4.6 — the P*P*n_block f2 tensor up to ~10^10 elements,
and the M*P genotype matrix) were the ones most likely to surface real overflow bugs.
Every TU that does the big-index arithmetic was independently verified to widen each
operand to `size_t` / `long` / `uint64_t` (or `long long` for cuBLAS strides) BEFORE the
multiply. No `int`-domain product overflow before widening was found on any hot path.

## 3. Top findings (all LOW — no HIGH/MED exist)

There are no HIGH or MED findings. The two substantive LOW findings touch the index-width
tasks; the remaining nine are a single recurring host/device pointer-typing hygiene note.
Ordered by importance:

1. [4.2][LOW] `src/device/cuda/qpadm_fit_kernels.cu` (global indices at :394,:468,:701,
   :842,:957-959,:989,:1031-1032,:1074-1075,:1203,:1231,:1264-1266 + launch wrappers) —
   every global index into the f2 tensor and the model arenas is widened to 64-bit `long`
   before the product (verified, no overflow). Hygiene only: the index type is `long`, not
   `int64_t`/`size_t` — correct on Linux LP64 (the only supported target) but 32-bit on
   LLP64/Windows. Suggested: standardize on `std::int64_t`/`size_t` to make the 64-bit
   intent platform-independent.
2. [4.2][LOW] `src/device/device_partial.hpp:39,40` — `int P`, `int n_block_local` shape
   fields are the width *source* for consumers that compute `P*P*n_block_local` (>2^31 at
   P~2500, n_block~757). No arithmetic here (delegates to DeviceBuffer), so no overflow in
   this TU; the obligation to widen lands on the consuming TUs (cuda_backend.cu /
   p2p_combine.cu), which were verified to widen correctly. Suggested: leave as-is.
3. [4.6][LOW] `src/device/cuda/qpadm_fit_kernels.cu:1153-1157,:1115,:1150,:933,:861,:1214` —
   thread-local index arithmetic (`ia = surv[a/nr]*nr + (a%nr)`, `qinv[ia + m*ib]`, etc.)
   stays in `int`, bounded by m = nl*nr <= kQpMaxM = 50 in the SMALL bucket (host enforces
   nl<=5, nr<=10, r<=4), so products max ~50*49 — far below 2^31. The block-offset factor
   `(long)m*b` IS widened. Correctly not widened; no overflow. Suggested: none.
4. [4.7][LOW] `src/core/fstats/f2_blocks_multigpu_core.cpp:246` —
   `compute_multigpu_partials_into` takes raw `double* dst_f2, double* dst_vpair,
   int* block_sizes_dst` (genuinely host/pinned D2H destinations); bare pointers can't
   prevent a future caller passing a device pointer.
5. [4.7][LOW] `src/device/backend.hpp:485-494` — `compute_f2_blocks_into` raw
   `double*`/`int*` PINNED-HOST destinations; the CUDA-free seam shape (device residency
   crosses via opaque handles). `slab_off = (size_t)P*P*b0` at :467 correctly widened.
6. [4.7][LOW] `src/device/cuda/cuda_backend.cu:1504,:1628,:2165,:2465` — large-path/batched
   helpers (`large_svd_V`, `large_fit_one`, `fit_chunk`, `assemble_result`) pass raw
   `double*`/`int*` for both host and device pointers; every call site verified correct
   (the `d`-prefix naming is the only guard).
7. [4.7][LOW] `src/device/cuda/block_sink.cuh:59-60` / `block_sink.cu:107-108,:268-269` —
   `spill_block` takes device `const double*` while the sink holds host pinned `double*` of
   the identical raw type; the `cudaMemcpyDeviceToHost` kind arg is the only guard.
8. [4.7][LOW] `src/device/cuda/p2p_combine.cu:138-141,:188-191,:264-267` — raw `double*` for
   both host (`out.f2.data()`) and device (`dResult_f2.data()`/peer `src_f2`); the
   scale-critical index path was separately verified CLEAN (all widened to size_t).
9. [4.7][LOW] `src/device/cuda/f2_block_kernel.cuh:38-106` / `.cu:102-108,...` — kernel
   params and launch wrappers use raw `const double*`/`double*`, project-wide device seam
   pattern.
10. [4.7][LOW] `src/device/cuda/device_buffer.cuh:95-96` — `data()` hands out a raw device
    `T*`; a caller could pass it to host-side `memcpy`/deref without a compile error
    (accepted: raw device ptr is required for cuBLAS/kernel args).

Also (same 4.7 pattern): `src/core/internal/views.hpp:54` (`MatView::data` raw
`const double*` seam) and `src/device/cuda/pinned_buffer.cuh:102-103,:158,:260`
(`PinnedBuffer<T>::data()` / registry raw `const void*`, host-only by design).

## 4. Cross-cutting patterns

- **One recurring pattern dominates (9 of 12 findings): 4.7 host/device pointer typing.**
  Across the device seam (`backend.hpp`, `cuda_backend.cu`, `block_sink`, `p2p_combine`,
  `f2_block_kernel`, `device_buffer`, `pinned_buffer`, `views.hpp`,
  `f2_blocks_multigpu_core`) raw `double*`/`int*`/`void*` carry both host and device
  storage with no type-level space distinction; the only guards are the `cudaMemcpy*`
  direction enum, the `d`-prefix naming convention, and call-site discipline. Every call
  site was verified correct — this is latent hygiene, not a live bug. If the project ever
  wants compile-time protection, the single highest-leverage fix is one thin
  `DevicePtr<T>`/`HostPtr<T>` tag-wrapper adopted at the backend seam; it would retire all
  nine notes at once. Not blocking.

- **The scale-critical index-width risk (4.2/4.6) was the headline thing to find, and the
  code is disciplined about it.** Every TU touching the ~10^10-element f2 tensor or the M*P
  genotype matrix widens each operand to `size_t`/`long`/`uint64_t` (and `long long` for
  cuBLAS strides) before the multiply, and the one whole-matrix narrowing
  (`k = (int)M` in `run_f2_gemms`, cuda_backend.cu) is fail-fast guarded by
  `if (M > INT_MAX) throw` before any use. The only residual notes are stylistic
  (`long` vs `int64_t`/`size_t` as the chosen 64-bit type — fine on the Linux-only target).

- **4.1 / 4.3 / 4.4 / 4.5 are clean across all 61 units.** FP64 literals and `double` math
  are intentional and parity-load-bearing (no wrong narrowing found); allocations carry
  `* sizeof(T)` or use element-count RAII (`DeviceBuffer<T>`); no unsigned countdown loop;
  no signed/unsigned loop-bound mismatch.

## 5. Headline

**61 units reviewed, 50 clean / 11 with findings, 12 findings total — 0 HIGH, 0 MED, 12 LOW.**
No correctness or overflow bugs. The at-scale int-overflow risk (4.2/4.6), which was the
real hazard to hunt, is handled correctly everywhere; the residual is a single cosmetic
host/device pointer-typing hygiene theme (9 notes) plus two index-type style notes and one
bounded thread-local-int note.
