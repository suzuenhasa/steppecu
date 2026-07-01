# Review findings — src__device__cuda__device_buffer

Files: /home/suzunik/steppe/src/device/cuda/device_buffer.cuh

## Group 4 — Type & numeric

- [4.7][LOW] device_buffer.cuh:95-96 — `data()` / `const data()` hand out a raw `T*` that is a *device* pointer with no host-vs-device space distinction, so a caller can pass it to a host-side `memcpy`/deref without a compile error. Suggested: acceptable per steppe design (raw device ptr for cuBLAS/kernel args); a thin `DevicePtr<T>` wrapper would make the wrong space unpassable, but this is hygiene, not a bug.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

- [6.1][LOW] device_buffer.cuh:78,81 — the move-source parameter is named `o` (single-letter) in both the move-ctor and move-assign signatures; `other` is the conventional and more readable name for a move/copy source, and the file's own header prose already refers to it as `other` (line 12: "`buf = std::move(other)`"). Suggested: rename `o` to `other` for the move ctor/assign params. Note `n`, `e`, `T` are conventional/tightly-scoped (element count, short-lived `cudaError_t`, template param) and fine.
- [6.3][LOW] device_buffer.cuh:12 vs 78,81 — minor inconsistency between doc and code: the header comment spells the move source `other` while the parameter is coded as `o`. Suggested: align them (covered by the 6.1 rename to `other`).

## Group 7 — Duplication

- [7.1][LOW] device_buffer.cuh:78-79,84-85 — the move-source steal `ptr_ = std::exchange(o.ptr_, nullptr); size_ = std::exchange(o.size_, 0);` is repeated verbatim in the move-ctor (init list) and the move-assign body. Suggested: leave as-is — this is the canonical two-member move idiom (one form is a mem-initializer list, the other a post-`reset()` assignment), so a helper would not fold both call shapes cleanly and reduces clarity over the idiom; not worth extracting.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.

## Group 11 — Qualifiers & const-correctness

No Group 11 issues found.

## Group 12 — Launch config & indexing

No Group 12 issues found.

## Group 13 — Error handling

No Group 13 issues found.

## Group 14 — Memory: allocation & lifetime

- [14.2][MED] device_buffer.cuh:74,117 — the sole owning device-buffer type allocates/frees with plain `cudaMalloc`/`cudaFree`, both of which implicitly synchronize across ALL streams; there is no `cudaMallocAsync`/`cudaFreeAsync` (pooled, stream-ordered) variant. As the single allocation owner this is fine for the once-allocated resident tensors (the P*P*n_block f2 tensor allocated once at setup), but at the S8 batched-solve envelope (thousands of models, millions of small buffers) any short-lived/per-solve DeviceBuffer would serialize the whole device on every construct/destruct. Suggested: latent/scale concern only — consider an opt-in stream-ordered (cudaMallocAsync + pool) construction path keyed off a stream for hot per-solve scratch; the plain path is correct for resident one-shot buffers. Do not change the resident-buffer path.
- [14.1] device_buffer.cuh:74 vs 117 — alloc (`cudaMalloc`) is correctly paired with `cudaFree` in `reset()`; no `free`/`delete[]` on a cudaMalloc'd ptr, no cudaFree on host memory. No issue.
- [14.5] device_buffer.cuh:64-76 — error-path leaks checked: the overflow `throw` (line 70) precedes any malloc; if `cudaMalloc` (line 74) fails, `STEPPE_CUDA_CHECK` throws, `ptr_` stays `nullptr` (cudaMalloc does not write the out-ptr on failure) and the throwing ctor suppresses the destructor — nothing was allocated, so nothing leaks and no double-free. No issue.
- [14.3]/[14.4] — no `cudaMallocAsync`/`cudaFreeAsync` used, so no async/sync free mismatch; `cudaFree` is fully synchronizing so the current free cannot race in-flight work on the buffer (no cross-stream use-after-free in the present implementation). No issue. (Forward note: if 14.2's async path is ever adopted, `reset()` would need stream/lifetime tracking before the async free, since `data()` hands out untracked raw pointers — not a present bug.)

## Group 15 — Memory: transfers

No Group 15 issues found. (This TU is a pure device-memory owner: it calls only `cudaMalloc` (line 74) / `cudaFree` (line 117), performs NO H<->D transfers, holds no host memory, and contains no `cudaMemcpy`. 15.1 no transfer in a loop to hoist; 15.2 no `cudaMemcpyKind` direction enum present to mismatch; 15.3 pageable-vs-pinned host staging is the sibling `pinned_buffer.cuh`'s concern, not this device-only owner.)

## Group 16 — RAII: ownership & wrapper hygiene

No Group 16 issues found. (This TU is the textbook reference for the rules this group checks. 16.1: the only resource it owns is device memory and it is fully wrapped — `cudaMalloc` (line 74) is paired with `cudaFree` (line 117) in `reset()`; no streams/events/graphs/library handles/pinned memory live here, so none can leak unwrapped. 16.2: copy ctor and copy-assign are `= delete` (lines 90-91), and both moves null the source — move-ctor `std::exchange(o.ptr_, nullptr)` / `std::exchange(o.size_, 0)` (lines 78-79), move-assign does the same after `reset()` with a self-assignment guard `if (this != &o)` (lines 81-88) — so a moved-from buffer holds a null handle and its later destructor free is a no-op (no double-free). 16.3: rule of five is complete and consistent — freeing dtor (line 93 -> `reset()`), deleted copy pair (90-91), defined move pair (78-88); the destructor is `noexcept` via `reset() noexcept` (line 111) and routes a nonzero teardown status to a warning rather than throwing (lines 117-121). 16.4: single clear ownership — `data()`/`const data()` (lines 95-96) hand out a raw non-owning `T*` view never freed by the consumer, and the type is move-only so it cannot be silently copied/passed-by-value as an owning copy. 16.5: the hand-rolled wrapper is justified, not a reinvention — this is one of the three architecture.md §2 allowlisted cudaMalloc/cudaFree TUs, and a `thrust::device_vector` would value-initialize/memset every element, unwanted overhead for the large resident P*P*n_block f2 tensor, while a `unique_ptr` + CUDA deleter would not carry the element `size_`/`bytes()` the §11.2 VRAM budget needs.)

## Group 17 — RAII: lifetime & deleter pitfalls (CUDA-specific)

- [17.5][MED] device_buffer.cuh:64,74,117,127-128 — multi-GPU device-correct free is NOT guaranteed by this owner. `DeviceBuffer` stores only `ptr_`/`size_` (lines 127-128); it does NOT record the device ordinal that was current at `cudaMalloc` (line 74). `reset()` then issues `cudaFree(ptr_)` (line 117) with NO `cudaSetDevice` to the alloc device. steppe is explicitly multi-GPU (cuda_backend.cu:2599 `guard_device()` = `cudaSetDevice(device_id_)`; one backend per physical ordinal), and these buffers escape the device-guarded method that allocated them: `DevicePartial::Impl` and `DeviceF2Blocks::Impl` (device_partial_impl.cuh:16-18, device_f2_blocks_impl.cuh:14-17) own `DeviceBuffer<double> f2/vpair` documented "resident on device_id" (e.g. device 1), and their destructors are `= default` (device_partial.cu:11, device_f2_blocks.cu:21). A defaulted dtor destroys the members under WHATEVER device is current at the caller's teardown point (typically the entry device 0), not the alloc device. `cudaFree` of another device's pointer returns an error, which reset() swallows to a debug warning (lines 118-121) — so the VRAM LEAKS on the wrong-device path. (The codebase already knows this matters: the alloc/upload paths wrap a `struct G { int d; ~G(){cudaSetDevice(d);} }` restore guard around `cudaSetDevice(device_id)` before constructing the buffers, device_f2_blocks.cu:51-52,79-80 — but the symmetric set-on-free is absent.) Suggested: record the alloc-device ordinal in the ctor (`cudaGetDevice(&dev_)`) and, in `reset()`, set+restore that device around `cudaFree` (the canonical CUDA multi-GPU deleter), OR make the escaping owners' destructors (DevicePartial/DeviceF2Blocks) re-select `device_id` before the members destruct. Latent for the single-GPU default (device 0 always current) — fires only on the M4.5 multi-GPU path.
- [17.1][LOW] device_buffer.cuh:117-121 — teardown-order at process/context exit: `reset()` calls `cudaFree` unconditionally when `ptr_ != nullptr`, but at static-destruction/atexit the CUDA context may already be torn down, in which case `cudaFree` returns `cudaErrorCudartUnloading` (or `cudaErrorContextIsDestroyed`) and emits a spurious `STEPPE_LOG_WARN` for a non-real leak. The destructor correctly never throws (reset() is `noexcept`, status is logged not rethrown — 17.1's core rule is met), so this is hygiene only. Suggested: optionally treat `cudaErrorCudartUnloading`/`cudaErrorContextIsDestroyed` as benign at teardown (skip the warning), since the OS reclaims the context's memory regardless. steppe's DeviceBuffers are normally backend-owned (not file-scope statics) so this rarely triggers in practice.
- 17.2 deleter/allocator match — clean. `cudaMalloc` (line 74) is paired with `cudaFree` (line 117); no `cudaMallocHost`/`cudaMallocArray`/`cudaMallocAsync` family is used here, so no `cudaFreeHost`/`cudaFreeArray`/`cudaFreeAsync` mismatch is possible.
- 17.3 unique_ptr<T[]> on device memory — N/A. This is a hand-rolled owner with an explicit `cudaFree`; there is no `std::unique_ptr<T[]>` / `delete[]` over a `cudaMalloc`'d pointer (the exact UB this task hunts for), and the type holds the element-count `size_` a `unique_ptr` could not.
- 17.4 RAII-vs-async lifetime — clean in the present implementation. `reset()` frees with synchronous `cudaFree` (line 117), which implicitly synchronizes the device, so a scope-exit free cannot race in-flight stream work that still reads `data()` — no use-after-free. (Forward note, not a present bug: if a `cudaMallocAsync`/`cudaFreeAsync` stream-ordered path is ever added per the Group 14.2 note, reset() would need to either stream-order the free or sync first, because `data()` hands out untracked raw pointers with no stream-lifetime tie.)

## Group 18 — Correctness traps (wrong numbers, not crashes)

No Group 18 issues found. (This TU is a host-side header-only RAII device-memory owner — it contains NO CUDA kernels: there are no `__global__`/`__device__` functions, no shared memory, no `__syncthreads()`/`__syncwarp()`, no warp intrinsics (`__shfl`/`__ballot`/`__any`/`__all`), no thread indexing, and no reductions. Every Group 18 task is in-kernel: 18.1 divergent `__syncthreads()`, 18.2 missing barrier on shared-mem RAW/WAR, 18.3 warp-synchronous assumptions, 18.4 non-`_sync` warp intrinsics, 18.5 missing `if(idx<n)` bounds guard, 18.6 cross-thread read without barrier/atomic, 18.7 order-dependent float reduction — none of these constructs exist anywhere in `device_buffer.cuh` (lines 1-133). The only device-API calls are `cudaMalloc` (line 74) and `cudaFree` (line 117), neither of which is a kernel launch or warp/block operation. Nothing in scope for this group.)

## Group 19 — Performance: debug leftovers

No Group 19 issues found. (This TU is a host-side header-only RAII device-memory owner with NO CUDA kernels — all three Group 19 tasks are in-kernel/launch concerns. 19.1 stray `cudaDeviceSynchronize()`: none present; the only CUDA-runtime calls are `cudaMalloc` (line 74) and `cudaFree` (line 117), both legitimate allocation-family calls in the architecture.md §2 allowlisted TU, not debug syncs. 19.2 leftover `printf`/`#if 0`: there is no `printf` (line 119's `STEPPE_LOG_WARN` is the documented §7 teardown-warning sink for a nonzero `cudaFree` status — intentional, not a debug leftover), and the only preprocessor conditionals are the include guard `#ifndef`/`#define`/`#endif` (lines 25-26, 133), with no `#if 0` dead block. 19.3 redundant `__syncthreads()`: none — there are no `__global__`/`__device__` kernels, no shared memory, and no block/warp barriers anywhere in lines 1-133.)

## Group 20 — Performance: memory access

No Group 20 issues found. (This TU is a host-side header-only RAII device-memory owner with NO CUDA kernels, so all three Group 20 tasks — which are in-kernel global/shared memory access patterns — are out of scope. 20.1 uncoalesced global access: there are no `__global__`/`__device__` functions, no thread indexing, and no global-memory loads/stores from device code — the only device-API calls are the allocation-family `cudaMalloc` (line 74) and `cudaFree` (line 117), which allocate/free the buffer but perform no element access. 20.2 shared-memory bank conflicts: no `__shared__` declarations and no shared-memory access exist anywhere in lines 1-133. 20.3 re-reading the same global value instead of caching: the only repeated member reads are host-side scalar field accesses (`ptr_`, `size_`) in the move/reset/accessor methods — these are ordinary host registers/stack, not device global memory, and the move idiom's `std::exchange` reads each member exactly once. Nothing in scope for this group.)

## Group 21 — Performance: occupancy & registers

No Group 21 issues found. (This TU is a host-side header-only RAII device-memory owner with NO CUDA kernels, so all four Group 21 tasks — which are in-kernel occupancy/register concerns — are out of scope. There are no `__global__`/`__device__` functions anywhere in lines 1-133; the only device-API calls are the allocation-family `cudaMalloc` (line 74) and `cudaFree` (line 117), which carry no kernel launch, no thread/warp execution, no shared memory, and no register footprint. 21.1 warp divergence: no kernel, no warp execution, no per-thread branching — the only branches (`if (n)` line 65, the overflow guard line 66, `if (this != &o)` line 82, `if (ptr_)` line 112, `if (e != cudaSuccess)` line 118) are ordinary host control flow, not divergent device branches. 21.2 excessive shared memory: no `__shared__` declarations and no `cudaFuncSetAttribute`/dynamic-shared launch config exist. 21.3 register spills / monolithic kernels: there is no kernel body to spill — the type's methods (ctor/dtor/moves/accessors) are tiny host functions, not a monolithic `__global__`. 21.4 missing register hints: `#pragma unroll`, `__launch_bounds__`, and `__forceinline__` are kernel/device-code directives that would be meaningless on this host-side header — their absence is correct, not a missing hint. Nothing in scope for this group.)

## Group 22 — Performance: compute & launch

No Group 22 issues found. (This TU is a host-side header-only RAII device-memory owner with NO CUDA kernels and NO kernel launches, so all four Group 22 tasks are out of scope. 22.1 atomics vs reduction/scan: there are no atomics (`atomicAdd`/`atomicCAS`/...) and no reductions/scans anywhere in lines 1-133 — the type owns a buffer, it does not compute over it. 22.2 integer div/mod in loops: there are NO loops in the TU; the only division is the host-side overflow guard `SIZE_MAX / sizeof(T)` (line 66) and the only multiplies are `n * sizeof(T)` (lines 66, 74) and `size_ * sizeof(T)` (line 108) — `sizeof(T)` is a compile-time constant the compiler folds to a constant multiply/shift, and each runs at most once per ctor/`bytes()` call on the CPU, not in any GPU loop. 22.3 loop-invariant work / repeated index recompute: no loops to hoist out of; `n * sizeof(T)` appears at lines 66 and 74 but on mutually exclusive paths — the overflow `throw` (line 70) returns before line 74 — so it is evaluated at most once on any execution path, not redundantly per iteration. 22.4 launch overhead / fuse-or-graph: there are no kernel launches to fuse or capture; the only CUDA-runtime calls are the allocation-family `cudaMalloc` (line 74, once per non-empty ctor) and `cudaFree` (line 117, once per non-null teardown), which are not launches. The S8-scale per-solve allocation-churn concern (many small short-lived buffers serializing the device) is the stream-ordered-pool question already captured in Group 14.2 [14.2][MED], an allocation-policy issue, not a launch/compute-fusion issue, so it is not duplicated here.)
