# Review findings — src__core__fstats__f2_from_blocks

Files: /home/suzunik/steppe/src/core/fstats/f2_from_blocks.cpp, /home/suzunik/steppe/src/core/fstats/f2_from_blocks.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

This unit is a thin host-side orchestration/dispatch layer: it validates the Q/V/N + BlockPartition contracts and dispatches through the ComputeBackend seam. It performs no FP math, no device allocation, and computes no global indices/offsets into the f2 tensor or genotype matrix (that arithmetic lives in the backend, not here).
- 4.1: No floating-point math; only dispatch + integer comparisons. N/A.
- 4.2: The only loop (f2_from_blocks.cpp:80-87) uses `long s` against `long M` and subscripts via `static_cast<std::size_t>(s)` (line 83). M (<=~584131) fits in `long`; no 32-bit `int` index into a >2^31 array is built here. Clean.
- 4.3: No cudaMalloc/new/DeviceBuffer in this unit. N/A.
- 4.4: Only loop (line 82) is `for (long s = 0; s < M; ++s)` — signed, ascending; no unsigned countdown. Clean.
- 4.5: All loop/bound comparisons are same-signedness — `id >= n_block` (int/int, line 84), `block_id.size() == static_cast<std::size_t>(...)` (line 108), `static_cast<long>(n_block) <= M` (lines 116/118), `s < M` (long/long, line 82). Clean.
- 4.6: No multiplicative index arithmetic (no `i*P+j` style); line 83 is a single subscript. No int overflow before widening. Clean.
- 4.7: `partition.block_id.data()` -> `const int*` across the seam (line 150) is the defined CUDA-free backend contract for host memory, not a host/device-pointer confusion within this unit. Clean.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

This unit is host-pure C++20 (layer=core, is_cuda=false): it includes only the CUDA-free seam (device/backend.hpp), the shared views, the block rule, the public config/fstats headers, and `<cstddef>` (f2_from_blocks.cpp:37-46; f2_from_blocks.hpp:11-15) — no `<cuda_runtime.h>`/`<cuda.h>`. A grep over both files for `sm_*`/`compute_*` arch flags, `texture`/`surface`/`cudaBindTexture*`, non-`_sync` warp intrinsics (`__shfl`/`__ballot`/`__any`/`__all`/`__activemask`), `cudaThreadSynchronize`/`cudaDeviceSynchronize`, kernel-launch `<<<>>>`, and `__global__`/`__device__` returned no matches.
- 2.1 (dropped Maxwell/Pascal/Volta archs): no architecture flags or CMake arch lists in this unit; it is not a translation unit compiled by nvcc. N/A.
- 2.2 (removed texture/surface references): no `texture<...>`/`surface<...>` or `cudaBindTexture*` — no device memory access of any kind here. N/A.
- 2.3 (non-`_sync` warp intrinsics): no warp intrinsics; the only loop (f2_from_blocks.cpp:82-87) is plain host iteration. N/A.
- 2.4 (`cudaThreadSynchronize` -> `cudaDeviceSynchronize`): no CUDA runtime sync calls; `core` reaches the GPU only via the ComputeBackend seam and issues no device call directly. N/A.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

This unit is a thin host-side orchestration/dispatch layer with no dead or commented-out code.
- 3.1: All comments in both files (f2_from_blocks.cpp:1-36, 52-61, 72-78, 91-102, 127-131, 139-147; f2_from_blocks.hpp:1-7, 19-31, 36-51) are documentation/rationale (layering, B11 fail-fast contract, IWYU notes) — no commented-out statements or expressions kept "just in case". Clean.
- 3.2: No `#if 0` and no code after `return`/`break`. The `#ifndef NDEBUG`/`#endif` block (f2_from_blocks.cpp:79-89) is intentional conditional compilation, not unreachable dead code: `block_ids_dense_nondecreasing` is referenced from the debug STEPPE_ASSERT at line 118 (same `#ifndef NDEBUG` regime). Clean.
- 3.3: All includes are used — `<cstddef>` (cpp:46) for `std::size_t` (lines 83, 108); `device/backend.hpp` (`ComputeBackend`/`F2Result`); `views.hpp` (`MatView`); `host_device.hpp` (`STEPPE_ASSERT`); `block_partition_rule.hpp` (`BlockPartition`); `fstats.hpp` (`F2BlockTensor`); `config.hpp` (`Precision`). Helpers `validate_qvn` (cpp:62), `validate_partition` (cpp:103), `block_ids_dense_nondecreasing` (cpp:80) are all called (lines 132, 148, 149, 118). The `[[maybe_unused]]` params (cpp:62-63, 103-104) are intentionally so for the NDEBUG build where STEPPE_ASSERT collapses to a no-op — not unused symbols to remove. Clean.
- 3.4: No computed-but-unread values. `prev` (cpp:81) is written and read across loop iterations (lines 84-85); the `[[nodiscard]]` result of `block_ids_dense_nondecreasing` is consumed by the STEPPE_ASSERT at line 118. Clean.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

This unit is a thin host-side orchestration/dispatch + contract-validation layer: it has no tunables, no kernel launch config, no device allocation, and no FP math. The only numeric literals are structural sentinels/comparison anchors, not magic numbers.
- 5.1 (unnamed literals): No `0.001f`/`1024`/`0.5`-style tunables. The literals present are `-1` (f2_from_blocks.cpp:81, `int prev = -1`) — a "no id seen yet" sentinel that is self-documenting in context (every valid id >= 0 exceeds it, used immediately at line 84) — and `0` in comparisons (lines 84, 108, 114, 116, 118), which are structural zero-bounds, not named-constant candidates. Clean.
- 5.2 (hardcoded sizes/bounds): All bounds are derived from inputs/params — `M` is `Q.M`, the partition is checked against `n_block` and `M` (lines 84, 108, 114-118); no hardcoded sizes, capacities, or limits. The P up to ~2500 / M up to ~584131 / n_block up to ~757 scale envelope is carried in the live `long M`/`int n_block` values, not baked in. Clean.
- 5.3 (duplicated constants subject to drift): No block dim duplicated between a launch and a shared-mem array size (no kernel here). The repeated `M <= 0 ||` short-circuit guard and the `M < 0 ? 0 : M` clamp (lines 108, 114, 116, 118) are guard expressions over the same parameter, not a duplicated literal that could drift. Clean.
- 5.4 (hardcoded paths/IDs/device ids): None — no file paths, no `cudaSetDevice`/device ordinals, no stream/handle ids; `core` issues no device call. Clean.
- 5.5 (ambiguous 32 — warp size vs other): No `32` (or other warp-size-coincident) literal anywhere; this is host-pure with no warps. N/A.

## Group 6 — Naming

No Group 6 issues found.

This unit is a thin host-side orchestration/dispatch + contract-validation layer; every name is either a documented domain term or a tight-loop counter.
- 6.1 (cryptic names): The single-letter names are all domain-standard or tight-loop counters — `Q`/`V`/`N` (the documented Q/V/N contract from views.hpp, f2_from_blocks.cpp:62-69, 125-126, 136-138), `P`/`M` (population/SNP count, the canonical dimension names, lines 64-69), `s` (SNP-index loop counter in the tight scan, line 82), `id` (the block id read inside that loop, line 83), `prev` (last id seen, self-documenting, lines 81/84-85). No `tmp`/`data2`/`arr`/`flag`. Clean.
- 6.2 (misleading names): `n_block` is a genuine count (used as the upper bound `id >= partition.n_block`, line 84; `n_block <= M`, lines 116-118), `block_id` is a genuine id array (subscripted at line 83), `M`/`P` are genuine dimensions. No count-that-is-an-index or list-that-is-a-map. Clean.
- 6.3 (inconsistent conventions in one file): Consistent throughout — functions/locals are `snake_case` (`compute_f2_block`, `compute_f2_blocks`, `validate_qvn`, `validate_partition`, `block_ids_dense_nondecreasing`), struct fields `snake_case` (`block_id`, `n_block`), and the domain matrices/dimensions are uppercase (`Q`/`V`/`N`/`P`/`M`) per the established f-statistics convention. No `nElements` vs `num_elements` vs `n` style mixing. Clean.
- 6.4 (nonstandard abbreviations): All abbreviations are domain-standard f-statistics/qpAdm terms — `qvn` (Q/V/N) in `validate_qvn` (line 62), `f2` (lines 125/136), `Vpair` (referenced in doc comments) — and the seam types `MatView`/`BlockPartition`/`F2Result`/`F2BlockTensor` are spelled out. No opaque or invented abbreviations. Clean.

## Group 7 — Duplication

This unit is a thin host-side orchestration/dispatch + contract-validation layer. Most apparent repetition is genuine shared-helper reuse (the two entry points both call `validate_qvn`, which is the explicitly-documented DRY home for the Q/V/N precondition — see f2_from_blocks.cpp:60-61 "One home for both the M0 and M4 entry points so they cannot diverge (§8 DRY)"), not copy-paste to extract.
- 7.1 (copy-pasted blocks differing by a constant): The two entry points (`compute_f2_block` cpp:125-134, `compute_f2_blocks` cpp:136-152) share a validate-then-dispatch shape, but they validate different contracts (one Q/V/N, the other Q/V/N + partition) and dispatch to different backend methods (`compute_f2` vs `compute_f2_blocks`) with different argument lists — not a constant-differing block. The three `STEPPE_ASSERT` calls at cpp:114/116/118 are not copy-paste: each carries a genuinely distinct predicate (`n_block > 0`, `n_block <= M`, dense/non-decreasing) and message. Clean.
- 7.2 (repeated loop-invariant expressions): The only loop (`block_ids_dense_nondecreasing`, cpp:82-87) re-reads `partition.n_block` each iteration (cpp:84), but it is a trivial member load, the loop is debug-only (`#ifndef NDEBUG`, never compiled in Release), and hoisting it would not change the release hot path. Not worth a finding. Clean.
- 7.3 (repeated sizeof/casts): No repeated `sizeof`. The casts are each used once on distinct operands — `static_cast<std::size_t>(s)` (cpp:83), `static_cast<std::size_t>(M < 0 ? 0 : M)` (cpp:108), `static_cast<long>(partition.n_block)` (cpp:116); no hoist/template candidate. Clean.
- 7.4 (collapsible boilerplate): [LOW] f2_from_blocks.cpp:114,116,118 — the `M <= 0 ||` short-circuit guard prefix is repeated across the three partition-sanity `STEPPE_ASSERT`s (so each "skip when there are no SNPs" check restates the same guard). Cosmetic only and debug-only; the predicates/messages differ, so this is a micro-readability note, not a correctness or scale concern. Suggested (optional): wrap the three M-positive asserts in a single `if (M > 0) { ... }` block so the guard appears once, OR leave as-is (the per-assert guard keeps each line independently readable). Low priority — borderline keep-as-is.

## Group 8 — Comments

This unit is comment-dense (file-header rationale + per-helper B11 fail-fast rationale + IWYU include notes). The comments are overwhelmingly genuine "why" rationale, not mechanical restatement, and carry the non-obvious justifications (the `-1` no-id sentinel, `[[maybe_unused]]` under NDEBUG, the `#ifndef NDEBUG` debug-only O(M) scan, the raw-pointer length-erasing seam, the `M < 0 ? 0 : M` clamp). One finding: a hard line-number citation has drifted.
- [8.2][LOW] f2_from_blocks.cpp:53 — stale cross-reference: the `validate_qvn` comment cites `(backend.hpp:144 "Q, V, N share the same P and M")`, but that precondition text now lives at backend.hpp:375 (verified: line 144 is unrelated RankSweep doc; the quoted string is at line 375). Hard line-number citations into another file drift on every edit. Suggested: drop the brittle `:144` and cite the symbol/section instead (e.g. "backend.hpp `compute_f2` preconditions") so it cannot go stale.
- 8.1 (restating code): clean — no `i++; // increment i`-style restatement. The inline include comments (cpp:39-44, hpp:11-15) name the symbols each header supplies, which is intentional IWYU rationale, not redundant restatement of behavior.
- 8.3 (missing rationale): clean — every non-obvious choice is justified in place: the `int prev = -1` sentinel (cpp:81), `[[maybe_unused]]` for the NDEBUG no-op path (cpp:56-61, 101-102), the debug-only `#ifndef NDEBUG` scan and why it duplicates the backend's `block_ranges` check (cpp:72-78, 91-102), passing the raw `block_id.data()` to keep the seam CUDA-free / ABI-free (cpp:144-147), and the `M < 0 ? 0 : M` clamp before the size cast (cpp:108).
- 8.4 (orphan TODO/FIXME/HACK): clean — grep for TODO/FIXME/HACK/XXX/WIP/TBD/"temporary"/"for now"/"placeholder" over both files returned no matches.

## Group 9 — Constants & configuration

No Group 9 issues found.

This unit is a thin host-side orchestration/dispatch + contract-validation layer. It defines no tunable knobs (block size, thresholds, precision values): `precision` is a passed-in `const Precision&` parameter whose policy is owned by the caller (f2_from_blocks.cpp:126,138; hpp doc comments explicitly state "the orchestration owns the policy"). No function in either file takes a boolean parameter, so no positional-bool call sites exist.
- 9.1 (should-be-const/constexpr left mutable): The only mutable local is `int prev = -1` (f2_from_blocks.cpp:81) — a genuine loop accumulator written each iteration (line 86), correctly NOT const. `const int id` (line 83) is already const; `long s` is the loop counter. All function parameters are `const&`/value with no mutation needed. No should-be-const left mutable. Clean.
- 9.2 (tangled config / buried tunables): No tunable knobs at all — no block dim, no precision mantissa, no batching threshold defined or hardcoded in logic here. The block policy (BlockPartition) and precision policy both arrive as parameters; this layer only validates contracts and dispatches. Nothing to surface to a file-top/config struct. Clean.
- 9.3 (positional booleans): No function in this unit has a `bool` parameter. `validate_qvn(Q,V,N)`, `validate_partition(partition,M)`, `backend.compute_f2(Q,V,N,precision)` (line 133), and `backend.compute_f2_blocks(Q,V,N,block_id.data(),n_block,precision)` (lines 150-151) pass no boolean literals; `block_ids_dense_nondecreasing` returns bool but takes none. No `foo(true,false)` ambiguity. Clean.

## Group 10 — Initialization

No Group 10 issues found.

This unit is a thin host-side orchestration/dispatch + contract-validation layer. The two entry points (`compute_f2_block` cpp:125-134, `compute_f2_blocks` cpp:136-152) declare no locals at all — they validate and dispatch over `const&`/value parameters. The only locals live in the debug-only `block_ids_dense_nondecreasing` scan (cpp:80-88), and both are declared at point of first use with an explicit initializer.
- 10.1 (late/distant decl, or uninitialized-then-assigned): All declarations are at first use with an initializer. `int prev = -1` (cpp:81) is declared immediately before the loop that reads/writes it (cpp:82-86); `const int id = partition.block_id[...]` (cpp:83) and `long s` (cpp:82) are declared and initialized in the same statement as their first use. No uninitialized-then-later-assigned local, no declaration hoisted far from use. Clean.
- 10.2 (zero-init assumptions that do not hold): No reliance on implicit/zero initialization. `prev` is EXPLICITLY initialized to the `-1` "no id seen yet" sentinel (cpp:81), deliberately NOT 0 — a zero-init here would be a real bug since block id 0 is valid and would wrongly satisfy `id < prev` on the first iteration; the explicit `-1` (documented cpp:81) is exactly the guard against that. No default-constructed-then-relied-upon objects, no static/global zero-init dependence, no partially-initialized aggregates. Clean.

