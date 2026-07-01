# Review findings — src__device__shard_plan

Files: /home/suzunik/steppe/src/device/shard_plan.cpp, /home/suzunik/steppe/src/device/shard_plan.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verified clean against the 4.1-4.7 checklist (with the FP64 + SCALE context):
- 4.1 float/double: no float/double arithmetic in this unit at all (pure integer index/SNP-count planning). N/A.
- 4.2 index width: SNP columns are carried as `long` end-to-end — DeviceShard::s0/s1 are `long` (shard_plan.hpp:41-42), sourced from ranges[].begin/.end which are `long` (block_partition_rule.hpp:152-153). DeviceShard::b0/b1 are `int` (shard_plan.hpp:39-40) but hold BLOCK ids, count O(1e3), so the static_cast<int>(...) at shard_plan.cpp:89-90,102-104 cannot overflow.
- 4.3 allocation sizing: no cudaMalloc/new; the only allocation is `std::vector<DeviceShard> plan(G)` (shard_plan.cpp:37) — element-count ctor, correct.
- 4.4 unsigned countdown: the two loops (shard_plan.cpp:53, 78) both count UP with std::size_t. No `for(unsigned i=n-1; i>=0; --i)` pattern.
- 4.5 signed/unsigned compares: loop bounds are size_t vs size_t throughout — `b < n_block`, `g + 1 < G`, `b + 1 < n_block` (shard_plan.cpp:53,78,84,86). No mixed-sign compare.
- 4.6 int overflow before widening: total_snps and device_snps are `long`; cdiv(total_snps, G_signed) deliberately routes through the cdiv(long,long) overload (shard_plan.cpp:63-65, launch_config.hpp:89-91), so (n + b - 1) is computed in long. total ≈ M (~6e5), no overflow. No int intermediate sits in a long-index expression.
- 4.7 host/device pointer typing: CUDA-free by contract, no raw pointers. N/A.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Verified clean against the 2.1-2.4 checklist. This unit is host-pure and CUDA-FREE by contract (shard_plan.cpp:5-6 includes only <cstddef>/<stdexcept>/<vector> + two core headers; shard_plan.hpp:22-26 includes only <cstddef>/<span>/<vector> + core/block_partition_rule.hpp). No <cuda_runtime.h>, no .cu code, no device entry points.
- 2.1 Dropped archs (Maxwell/Pascal/Volta, min sm_75): no CMakeLists, no nvcc -arch / -gencode / sm_* / compute_* flags or CUDA_ARCHITECTURES lists in either file. N/A.
- 2.2 Texture/surface REFERENCES removed in CUDA 12: no texture<...>, surface<...>, cudaBindTexture*, or tex1Dfetch in the unit. N/A.
- 2.3 Non-_sync warp intrinsics: no __shfl*/__ballot/__any/__all/__activemask (no device code at all). N/A.
- 2.4 cudaThreadSynchronize -> cudaDeviceSynchronize: no CUDA runtime calls whatsoever (the only synchronization-adjacent concept is the deterministic single-pass greedy loop, pure host code). N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Verified clean against the 3.1-3.4 checklist:
- 3.1 Commented-out code blocks: all comments in both files are explanatory documentation/rationale prose (header docs shard_plan.cpp:1-9 / shard_plan.hpp:1-18, inline rationale e.g. cpp:24-35,46-66,67-73,99-111). NONE is commented-out source kept "just in case". N/A.
- 3.2 Unreachable code: the two early returns (cpp:42-44 n_block==0, plus the G==0 throw cpp:27-31) guard degenerate cases; the non-degenerate path falls through normally. No code follows any return/break/throw unreachably; no #if 0. The final `return plan` (cpp:112) is the function tail. N/A.
- 3.3 Unused symbols: every include is used — cpp: <cstddef> (std::size_t), <stdexcept> (std::runtime_error cpp:28), <vector> (std::vector cpp:37), block_partition_rule.hpp (core::BlockRange in signature/ranges[b]), launch_config.hpp (core::cdiv cpp:65); hpp: <cstddef> (std::size_t), <span> (std::span:99), <vector> (std::vector:98), block_partition_rule.hpp (BlockRange:99). Both params (ranges, G) used. All locals read: n_block, plan, total_snps, G_signed, target_per_device, g, b0, device_snps, b1, more_devices_left, reached_target. No unused symbol.
- 3.4 Computed but unread: every assignment is subsequently read. device_snps reset to 0 (cpp:95) is read in the next iteration (cpp:79,85); all DeviceShard fields (b0/b1/s0/s1) are consumed by the orchestrator/parity test. No dead store.
-->

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Verified clean against the 5.1-5.5 checklist (with the FP64 + SCALE + §12 context):
- 5.1 Unnamed literals -> named constants: the only numeric literals are `0` (the G==0 fail-fast guard cpp:27; the long/size_t accumulator inits total_snps=0 cpp:52, device_snps=0/95, g=0 cpp:74, b0=0 cpp:75; the struct-default empty shard {0,0,0,0} comments cpp:37,108,111) and `1` (the neighbor/boundary index arithmetic g+1 cpp:84, b+1 cpp:86,87, b1-1 cpp:92, n_block-1 cpp:106). These are structural off-by-one neighbor access and half-open-range boundary math, NOT tunable magic constants — naming them would obscure, not clarify. The one computed quantity, target_per_device, is EXPLICITLY DERIVED (core::cdiv(total_snps, G_signed) cpp:64-65), never a hardcoded threshold (the header doc and inline rationale call this out, hpp:67-69, cpp:46-66). N/A.
- 5.2 Hardcoded sizes/bounds that should be params/derived: G is the function parameter (cpp:23), n_block is derived from ranges.size() (cpp:36), total_snps is summed from the inputs (cpp:52-55), target_per_device is derived via cdiv (cpp:64-65). No hardcoded size or loop bound — all bounds are size_t expressions over G / n_block (cpp:53,78,84,86). N/A.
- 5.3 Duplicated constants (DRIFT): no device code, so no block-dim-in-launch-AND-shared-mem-array pattern. Ceiling division is NOT re-implemented inline — it routes through the single home core::cdiv (cpp:65; launch_config.hpp), so the (total + G - 1)/G constant lives in exactly one place. The empty-shard {0,0,0,0} is materialized by the struct member defaults (shard_plan.hpp:39-42 value-init via plan(G) cpp:37), not duplicated as repeated literals. No drift-prone duplicate constant. N/A.
- 5.4 Hardcoded paths/IDs/device ids: no filesystem paths, no stream/event IDs, no hardcoded CUDA device ordinals (CUDA-FREE by contract). `g` is a derived loop counter over [0,G), not a baked-in device id. N/A.
- 5.5 Ambiguous 32 (warp size vs other): no literal 32 (and no 64/128/256/1024 block-dim constants) anywhere — this is host-pure planning with no warp/thread notion. N/A.
-->

## Group 6 — Naming

No Group 6 issues found.

<!--
Verified clean against the 6.1-6.4 checklist (with the §12/FP64/scale context):
- 6.1 Cryptic names: the short names are domain-standard SPMG/jackknife notation, all documented at point of use, not opaque tmp/data2/arr/flag. `G` = device count (param, doc'd shard_plan.hpp:92,98-100; matches architecture §11.4 SPMG "G devices" notation); `g` = current device index (cpp:74 `// current device index`); `b`/`b0`/`b1` = block iterator / range-begin / range-end, tight-loop counters explicitly allowed and doc'd (cpp:74-75); `s0`/`s1` = SNP column bounds, doc'd as full sentences (shard_plan.hpp:41-42). The two boolean predicates are spelled out (`more_devices_left` cpp:84, `reached_target` cpp:85). No single-letter name outside a loop-counter/domain-notation role.
- 6.2 Misleading names: every name matches its contents. `total_snps`/`device_snps` ARE SNP counts (long accumulators cpp:52,76); `target_per_device` IS the derived per-device target (cpp:64); `n_block` IS the block count (ranges.size() cpp:36); `plan` IS the vector<DeviceShard> result. No count-named-as-index or list-named-as-map. `DeviceShard::empty()` (shard_plan.hpp:46) truthfully reports b0>=b1. N/A.
- 6.3 Inconsistent conventions in one file: all locals are snake_case (total_snps, device_snps, target_per_device, n_block, more_devices_left, reached_target, device_snps) — consistent. `G` and `G_signed` are uppercase, but `G` is the documented parameter (the SPMG device-count symbol) and `G_signed` is its same-symbol signed copy for the cdiv(long,long) overload (cpp:63-65); keeping the symbol is intentional, not a convention drift. Struct fields b0/b1/s0/s1 are uniform. No nElements-vs-num_elements-vs-n mix.
- 6.4 Nonstandard abbreviations: `cdiv` = ceiling division, the project-wide single-home name (cpp:17 comment, launch_config.hpp), not local jargon; `snps` (SNP, the domain primitive), `b#`/`s#` (block / SNP column), `n_block` — all standard, all expanded in surrounding doc. No nonstandard/invented abbreviation. N/A.
-->

## Group 7 — Duplication

- [7.1][LOW] shard_plan.cpp:88-92 and 102-106 — the two `DeviceShard{static_cast<int>(b0), static_cast<int>(b1_or_nblock), ranges[b0].begin, ranges[b1-1].end}` constructions are the same 4-field shape (two int-casts of the block bounds + the begin/end column lookup off `ranges`), differing only in the upper block bound (`b+1` at the mid-loop close vs `n_block` at the final close). A tiny local `make_shard(b0, b1)` lambda would fold the cast+lookup boilerplate into one place and remove the risk of the two copies drifting (e.g. one getting the `b1-1` end-index fixed without the other). Suggested: extract `auto make_shard = [&](std::size_t lo, std::size_t hi){ return DeviceShard{int(lo), int(hi), ranges[lo].begin, ranges[hi-1].end}; };` and call it at both sites.

<!--
Other Group 7 tasks verified clean (FP64/§12/scale context applied):
- 7.2 Repeated loop-invariant expressions: `static_cast<long>(G)` is already computed ONCE into `G_signed` (cpp:63) and reused (cpp:65); nothing loop-invariant is recomputed inside the per-block loop (cpp:78-97) — `target_per_device` is hoisted above the loop (cpp:64), and `ranges[b].size()` is the per-iteration value (genuinely b-dependent, not invariant). No repeated invariant expr to hoist.
- 7.3 Repeated sizeof/casts: no sizeof anywhere. The repeated cast is `static_cast<int>` on the block ids — already covered as part of the 7.1 construction duplication (it disappears if make_shard is extracted); on its own it is a trivial, type-correct narrowing of an O(1e3) block id, not worth a standalone template.
- 7.4 Collapsible boilerplate: the only macro/helper-foldable block is the dual shard construction (7.1). The G==0 throw (cpp:27-31) and the two early returns are single-use fail-fast / degenerate guards, not a repeated pattern. No other boilerplate folds.
The 7.1 finding is LOW: both copies are within ~15 lines in one short function, the duplication is small (4 fields), and the existing inline comments document the b1-1 end-index intent at each site — drift risk is low but nonzero.
-->

## Group 8 — Comments

No Group 8 issues found.

<!--
Verified clean against the 8.1-8.4 checklist (FP64/§12/scale context applied):
- 8.1 Restating code: all inline comments explain RATIONALE/INVARIANTS, not the mechanical operation. The greedy close-on-cross proof (cpp:58-62), the half-open-range close semantics (cpp:82-83), the block-aligned invariant (cpp:67-73), and the role-annotations on the loop locals (`// current device index` cpp:74, etc.) document WHY/what-role, not `++g; // increment g`. No restating comment.
- 8.2 Stale comments: every behavior claim matches the current code. The `block_sizes`-was-removed prose (cpp:32-34, hpp:65-68, hpp:89-91) is accurate — there is no block_sizes param; the per-block count is `ranges[b].size()` (cpp:54,79). The empty-shard `{0,0,0,0}` comments (cpp:37,108,111) match the 4-field struct defaults (hpp:39-42) and the value-init `plan(G)` (cpp:37). The cdiv single-home note (cpp:60-62, hpp:67) matches the cdiv(long,long) call (cpp:65). The downstream-contract references (hpp:37 cuda_backend.cu degenerate guard; hpp:56-57 combine fixed-order) describe consumer behavior, not this unit's code, and are consistent with the design prose. No stale comment.
- 8.3 Missing rationale: non-obvious choices ARE justified — why ceiling division / why the last device absorbs the remainder (cpp:58-62), why `G_signed` exists (the cdiv(long,long) overload to widen before the +G-1, cpp:60-63), why balance-by-SNP-not-block (hpp:59-63), and the §12 bit-identical motivation for block-aligned sharding (hpp:5-11). No bare constant or workaround left unexplained.
- 8.4 Orphan TODO/FIXME/HACK: none present — no TODO/FIXME/HACK/XXX/NOTE marker in either file. The "cleanup B6 / X1 / shard_plan D-1" tags (cpp:34, hpp:67-68,91) are CLOSED-cleanup citations with full inline context (they explain the removed block_sizes), not dangling unowned action items.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Verified clean against the 9.1-9.3 checklist (FP64/§12/scale context applied):
- 9.1 Should-be-const/constexpr left mutable: every value that does not change after init IS already const — `n_block` (cpp:36), `G_signed` (cpp:63), `target_per_device` (cpp:64), and the per-iteration `b1` (cpp:87), `more_devices_left` (cpp:84), `reached_target` (cpp:85). The four genuinely mutable locals all mutate across the greedy pass: `total_snps` accumulates (cpp:52-55), and `g`/`b0`/`device_snps` advance/reset as devices close (cpp:74-76, 79, 94-95). Nothing is a compile-time constant that should be constexpr (G is a runtime parameter; target_per_device is derived from runtime inputs). No mutable-that-should-be-const.
- 9.2 Tangled config (tunable knobs buried in logic): there are NO tunable knobs in this unit. The only "configuration" is `G` (the device count, a function parameter resolved from Resources::device_count(), shard_plan.cpp:23 / hpp:92) and the balance target `target_per_device`, which is DERIVED at cpp:64-65 (core::cdiv(total_snps, G_signed)) rather than a buried magic threshold — the header doc and inline rationale call this out explicitly (hpp:67-69, cpp:46-66). The greedy close criterion is structural (reached SNP target AND devices remain), not a tunable parameter. Nothing to surface to a config struct / file top.
- 9.3 Positional booleans foo(true,false,true): no call in either file passes positional boolean literals. The two booleans (more_devices_left cpp:84, reached_target cpp:85) are NAMED local predicate variables combined in the if (cpp:86) — already the recommended self-documenting pattern, the opposite of the anti-pattern. The DeviceShard{...} aggregate initializations (cpp:88-92, 102-106) are positional but their fields are int/long block/column bounds (b0,b1,s0,s1), not boolean flags, and each field is documented at the struct (hpp:39-42); positional aggregate init of a documented 4-field POD is idiomatic, not an opaque boolean-flag call. N/A.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Verified clean against the 10.1-10.2 checklist (FP64/§12/scale context applied):
- 10.1 Late/distant declaration OR uninitialized-then-assigned: EVERY local in this unit is declared with an initializer AT its declaration, and each is declared immediately before (or at) its first use — no late/distant decls, no uninit-then-assign. `n_block` (cpp:36, init = ranges.size(), used cpp:42 onward), `plan` (cpp:37, value-init, returned cpp:112), `total_snps` (cpp:52, =0 directly above its accumulation loop cpp:53-55), `G_signed` (cpp:63) and `target_per_device` (cpp:64, =cdiv(...)) used immediately at cpp:65/85, the greedy-loop trio `g`/`b0`/`device_snps` (cpp:74-76, all =0 directly above the loop they drive at cpp:78), and the per-iteration consts `b1`/`more_devices_left`/`reached_target` (cpp:84-87, all init at decl inside the loop body). No `T x;` followed by a later `x = ...`. N/A.
- 10.2 Zero-init assumption that does not hold: the ONE place this unit relies on implicit value-init is `std::vector<DeviceShard> plan(G)` (cpp:37) — the count-ctor value-initializes each DeviceShard, and the trailing-empty-device guarantee (cpp:108-111: devices g+1..G-1 keep their EMPTY {0,0,0,0} shard for the n_block < G case) depends on that. This assumption HOLDS and is robust: DeviceShard carries default member initializers b0=0,b1=0,s0=0,s1=0 (hpp:39-42), so value-init produces {0,0,0,0} regardless of POD/aggregate value-init subtleties, and empty() == (b0>=b1) is then true (hpp:46). All other locals (total_snps=0, device_snps=0, g=0, b0=0 — cpp:52,74-76) are EXPLICITLY zero-initialized, not relying on implicit zero. No missing-init-relying-on-zero. N/A.
-->

## Group 13 — Error handling

No Group 13 issues found.

<!--
Verified clean against the 13.1-13.4 checklist (host-pure, CUDA-FREE-by-contract unit; §12/FP64/scale context applied):
- 13.1 Unchecked cuda* API return: there are NO cuda* runtime/driver/cuBLAS/cuSOLVER calls in either file. shard_plan.cpp includes only <cstddef>/<stdexcept>/<vector> + core/block_partition_rule.hpp + core/internal/launch_config.hpp (cpp:12-17); shard_plan.hpp includes only <cstddef>/<span>/<vector> + core/block_partition_rule.hpp (hpp:22-26). No <cuda_runtime.h>, no API to wrap in a CHECK. N/A.
- 13.2 Unchecked launches: no kernel launch (<<<...>>>) anywhere — this is host-pure greedy planning (the single per-block loop cpp:78-97). No cudaGetLastError()/sync pair is owed. N/A.
- 13.3 Inconsistent checking: there is no CUDA error checking because there is no CUDA. The host-side error handling that exists is appropriate and consistent — the sole precondition, G == 0, fail-fasts with a descriptive std::runtime_error (cpp:27-31, architecture §2); the n_block == 0 case is a documented clean degenerate early-return (cpp:42-44), not an error. No mixed guarded/unguarded API pattern. N/A.
- 13.4 Error-swallowing homegrown macro: no macros at all in this unit (no CHECK/CUDA_CHECK/STEPPE_* macro defined or used). Nothing can hide or downgrade an error to a no-op. The throw is a hard fail, not a swallowed return. N/A.
Note: the planner does not re-validate that each ranges[b].size() is >= 0 or that the partition is contiguous — but that is the documented block_ranges postcondition / input-contract (a Group 11/16 concern), not error handling of an API call, so it is out of Group 13 scope.
-->

## Group 14 — Memory: allocation & lifetime

No Group 14 issues found.

<!--
Verified clean against the 14.1-14.5 checklist (host-pure, CUDA-FREE-by-contract unit; §12/FP64/scale context applied):
- 14.1 Alloc/free mismatch: there is NO manual memory management in either file — no cudaMalloc, no new/new[], no malloc, no cudaFree/free/delete. The ONLY allocation is `std::vector<DeviceShard> plan(G)` (shard_plan.cpp:37), a fully RAII-managed std::vector whose storage is owned/freed by the container destructor (and moved out on the `return plan;` at cpp:112). No allocator is ever paired with the wrong deallocator. N/A.
- 14.2 Stream-ordered alloc on hot paths: no cudaMalloc/cudaFree anywhere (CUDA-FREE by contract — includes only <cstddef>/<stdexcept>/<vector> + core headers, cpp:12-17, hpp:22-26). plan_block_shards is a pure host planning function reached ONCE per SPMG precompute to compute the block->device mapping (architecture.md §11.4); it is not a per-block hot loop and touches no GPU stream. No cudaMallocAsync/pool concern applies. N/A.
- 14.3 Async/sync free pairing: no async-vs-plain allocation pair exists — there is no CUDA allocation of either flavor. N/A.
- 14.4 Free before async work completes (use-after-free across streams): the only buffer (`plan`) is host memory consumed synchronously by the caller after the function returns by value; it is never handed to a stream/kernel, so no async-completion ordering can be violated. The returned vector outlives the call via move/RVO. N/A.
- 14.5 Missing frees on error paths: the single error path is the G == 0 fail-fast throw (cpp:27-31), which occurs BEFORE the `plan` vector is constructed (cpp:37), so no allocation is in flight to leak. Even if any allocation were live, std::vector's destructor would release it during stack unwinding (no raw owning pointer to leak). No manual cleanup is owed on any path. N/A.
-->

## Group 15 — Memory: transfers

No Group 15 issues found.

<!--
Verified clean against the 15.1-15.3 checklist (host-pure, CUDA-FREE-by-contract unit; §12/FP64/scale context applied):
- 15.1 cudaMemcpy (H<->D) in a loop to hoist/batch/keep-resident: there are NO cudaMemcpy/cudaMemcpyAsync/cudaMemcpyPeer/cudaMemset calls anywhere. shard_plan.cpp includes only <cstddef>/<stdexcept>/<vector> + core/block_partition_rule.hpp + core/internal/launch_config.hpp (cpp:12-17); shard_plan.hpp includes only <cstddef>/<span>/<vector> + core/block_partition_rule.hpp (hpp:22-26). No <cuda_runtime.h> at all. The only loops are the host-pure SNP-sum accumulation (cpp:53-55) and the greedy single-pass block assignment (cpp:78-97) — pure integer arithmetic over the in-RAM `ranges` span, no transfer of any kind. This is the planning step that DECIDES the block->device mapping the transfers later use, not a step that performs transfers. N/A.
- 15.2 Direction enum vs actual transfer: there is no transfer, hence no cudaMemcpyKind argument (no cudaMemcpyHostToDevice / DeviceToHost / DeviceToDevice) to mismatch. N/A.
- 15.3 Pageable host memory for frequent transfers (pinned ~2x bandwidth): the unit allocates one ordinary pageable `std::vector<DeviceShard> plan(G)` (cpp:37) of G (device-count, O(1)) tiny POD shards, returned by value to the host caller. It is host metadata consumed on the host (the orchestrator reads plan[g] to build sub-views; the parity test reads it to shard identically) — it is NEVER the source/dest of an H<->D copy, so pinning is irrelevant. The frequent/large H<->D traffic in steppe (the f2_blocks tensor) lives in cuda_backend.cu / combine, not here; this is the mapping computed once per precompute (architecture.md §11.4). No pinned-memory opportunity exists in this unit. N/A.
-->

## Group 16 — RAII: ownership & wrapper hygiene

No Group 16 issues found.

<!--
Verified clean against the 16.1-16.5 checklist (host-pure, CUDA-FREE-by-contract unit; §12/FP64/scale context applied):
- 16.1 Wrap EVERY resource: this unit owns NO wrappable resource. CUDA-FREE by contract — shard_plan.cpp includes only <cstddef>/<stdexcept>/<vector> + core/block_partition_rule.hpp + core/internal/launch_config.hpp (cpp:12-17); shard_plan.hpp includes only <cstddef>/<span>/<vector> + core/block_partition_rule.hpp (hpp:22-26). There is no <cuda_runtime.h>, hence no stream/event/graph/graph-exec, no texture/surface object, no memory pool, no CUDA array, no pinned host allocation, and no cuBLAS/cuSOLVER handle (*Create/*Destroy) anywhere to wrap or leak. The sole owned resource is `std::vector<DeviceShard> plan(G)` (cpp:37), already RAII-managed by the std::vector destructor. N/A.
- 16.2 Move-only + null-on-move: there is no hand-written freeing wrapper, so no copy/move semantics to author. DeviceShard (hpp:38-47) is a trivial POD aggregate (two `int`, two `long`, all with default member initializers, plus a `[[nodiscard]] bool empty() const noexcept`) — it owns no handle, so its implicit copy/move are correct and there is no moved-from-must-null obligation. std::vector<DeviceShard> already has correct move (resets moved-from to empty) and copy semantics. No double-free / un-nulled-moved-from risk. N/A.
- 16.3 Rule of five for a freeing destructor: nothing in the unit declares a destructor that frees a resource. DeviceShard is implicitly-destructible POD (no user dtor); plan_block_shards is a free function. No class needs the rule of five. N/A.
- 16.4 Single clear ownership / non-owning raw pointers: ownership is unambiguous — `plan` is the sole owner of the shard storage and is moved out on `return plan;` (cpp:112). The input `std::span<const core::BlockRange> ranges` (cpp:22) is a NON-OWNING view (correct: a span by definition does not own; the function only reads ranges[b].size()/.begin/.end and never frees it), and DeviceShard carries plain int/long bounds (b0/b1/s0/s1), not owning pointers. No owning wrapper is passed by value (the by-value `std::vector` return is the move-out result, not a copy of an owning handle). No raw kernel/device pointer is freed here (there are none). N/A.
- 16.5 Don't reinvent the wrapper: the unit uses std::vector directly (cpp:37) — the standard container — rather than a hand-rolled owning buffer, and DeviceShard is a deliberately minimal value type (a documented 4-field POD), not a re-implemented smart pointer. No thrust::device_vector / unique_ptr+CUDA-deleter is even applicable (no device memory). Nothing is reinvented. N/A.
-->

## Group 17 — RAII: lifetime & deleter pitfalls (CUDA-specific)

No Group 17 issues found.

<!--
Verified clean against the 17.1-17.5 checklist (host-pure, CUDA-FREE-by-contract unit; §12/FP64/scale context applied). Every Group 17 task is CUDA-specific and has NO applicable surface here:
- 17.1 Non-throwing destructor (failed cudaFree/cudaStreamDestroy can't throw during unwinding; teardown-order guard): the unit declares NO destructor at all. DeviceShard (shard_plan.hpp:38-47) is a trivial POD aggregate (two int, two long with default member initializers, plus a noexcept empty() accessor) with the implicit destructor; plan_block_shards is a free function (cpp:21-113). No dtor performs any cuda* teardown, so there is nothing that could throw mid-unwind or touch a torn-down context. N/A.
- 17.2 Deleter matches allocator: there is NO CUDA allocation in either file — no cudaMalloc/cudaMallocHost/cudaHostAlloc/cudaMallocAsync/cudaMallocArray, and correspondingly no cudaFree*/free pairing to mismatch. shard_plan.cpp includes only <cstddef>/<stdexcept>/<vector> + core/block_partition_rule.hpp + core/internal/launch_config.hpp (cpp:12-17); shard_plan.hpp includes only <cstddef>/<span>/<vector> + core/block_partition_rule.hpp (hpp:22-26). No <cuda_runtime.h>. The sole allocation is `std::vector<DeviceShard> plan(G)` (cpp:37), allocated AND freed by std::vector's own allocator/destructor — allocator and deallocator are paired by the container by construction. N/A.
- 17.3 unique_ptr<T[]> on cudaMalloc (default delete[] is UB; needs custom deleter): no unique_ptr / shared_ptr / smart pointer of any kind, and no cudaMalloc buffer to wrap. The only owning container is std::vector (cpp:37). No array-form smart pointer over device memory exists. N/A.
- 17.4 RAII vs async lifetime (scope-exit free != async work finished -> use-after-free; sync first or tie to stream): the only buffer (`plan`, cpp:37) is HOST memory, never handed to a CUDA stream or kernel — this unit reaches no GPU (CUDA-FREE by contract, hpp:13-18). It is returned by value (move/RVO, cpp:112) and consumed synchronously on the host by the orchestrator/parity test. There is no async work, no stream, and thus no scope-exit-vs-async-completion ordering to violate. N/A.
- 17.5 Multi-GPU device-correct free (deleter must cudaSetDevice to the alloc device, record+restore): no per-device CUDA allocation and no custom deleter exist, so there is no free that must select the owning device. Although the unit's PURPOSE is multi-GPU shard PLANNING, it only computes the block->device index mapping (the DeviceShard {b0,b1,s0,s1} bounds, cpp:88-106) as plain integers — it never allocates device memory or calls cudaSetDevice (the actual per-device allocation/free lives in cuda_backend.cu / resources, not here). No device-correct-free obligation applies. N/A.
-->

