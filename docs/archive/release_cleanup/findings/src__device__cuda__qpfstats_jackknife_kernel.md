# src__device__cuda__qpfstats_jackknife_kernel
Files: /home/suzunik/steppe/src/device/cuda/qpfstats_jackknife_kernel.cu, /home/suzunik/steppe/src/device/cuda/qpfstats_jackknife_kernel.cuh
Subsystem: device-cuda

## Findings

### G7
- [G7.src__device__cuda__qpfstats_jackknife_kernel][LOW] qpfstats_jackknife_kernel.cu:53-95 — `cnt[base + b]` is read from global in four separate passes (lines 54, 62, 69, 82) and `numer[base + b]` written then re-read in two later passes (lines 68, 81). Each pass walks the same contiguous row independently. Suggested: this is collapsible into fewer passes (e.g. accumulate `sum_n_all` while materializing numer), but the multi-pass split is structurally required by the AT2 transliteration (tot needs sum_n_all first, loo needs tot first) — leave only if a single-pass formulation cannot preserve the §12 ascending-b accumulation order; otherwise low value.

### G20
- [G20.src__device__cuda__qpfstats_jackknife_kernel][LOW] qpfstats_jackknife_kernel.cu:54,62,69,82 — uncoalesced global access: numsum/cnt/numer are ROW-MAJOR `[npopcomb × n_block]` and each thread `c` walks a contiguous row, so adjacent threads (adjacent `c`) read addresses strided by `n_block` (~711). Acknowledged-intentional per the design comment (lines 20-21: "a thread reads a contiguous run"); flagged for completeness. Suggested: a COL-MAJOR numsum/cnt layout (cell `(c,b)` at `b*npopcomb+c`) would coalesce the per-b reads across threads, but would change the resident producer layout — defer unless a profiler shows this kernel is memory-bound (it is not on the hot path per the file rationale).
- [G20.src__device__cuda__qpfstats_jackknife_kernel][LOW] qpfstats_jackknife_kernel.cu:116-148 — in the recenter kernel each `b` cell is re-read from global via the `arr` lambda in three passes (lines 123, 129, 139) and `block_sizes[blk]` is read in all three passes (123, 131, 140). `block_sizes` is the same for every thread/pair, so it is a candidate for `__shared__`/register caching or `__ldg`/read-only path. Suggested: optionally cache `block_sizes` into shared memory once per block; low value at n_block~711 and off the hot path.

## Summary
No HIGH or MED issues. The grid-stride loops, `long` index widening (lines 47-50, 113-117 cast `blockIdx/gridDim/blockDim` products and `npopcomb`/`npairs` strides to `long` before multiply, avoiding the P~2500/M~584k overflow trap), the `core::kMaxGridX` grid-dim cap (lines 162, 175), `const __restrict__` on every read-only kernel pointer, the `STEPPE_CUDA_CHECK_KERNEL()` launch checks (lines 165, 178), the native-FP64 precision carve-out, the early-exit guards (`npopcomb<=0`/`npairs<=0`/`n_block<=0`), and the absence of any `__syncthreads()` (per-thread register-only reductions, no shared state) are all correct. Groups G2-G6,G8-G19,G21,G22 checked: no issues.
