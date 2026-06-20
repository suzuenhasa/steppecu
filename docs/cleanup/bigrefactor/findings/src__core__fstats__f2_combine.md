# Review findings — src__core__fstats__f2_combine

Files: /home/suzunik/steppe/src/core/fstats/f2_combine.cpp, /home/suzunik/steppe/src/core/fstats/f2_combine.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verified against the SCALE context (P up to ~2500, n_block up to ~757, total up to ~10^10 > 2^31):
- 4.1: FP64 by design; the only literals are 0.0 (+0.0 init, cpp:64-65) — intentional, no wrong narrowing.
- 4.2/4.6: every global index/offset is widened to std::size_t BEFORE the multiply.
  slab = (size_t)P * (size_t)P (cpp:59-60); total = slab * (size_t)n_block_full (cpp:63, slab already size_t);
  part_slabs = slab * (size_t)part.n_block (cpp:100-101); destination offsets slab*b0 (cpp:103-104) and b0 (cpp:108-110)
  all carry a size_t operand, so the arithmetic promotes to size_t — no int*int product can overflow before widening.
- 4.3: no cudaMalloc/new; std::vector::assign(count,value) (cpp:64-66) takes element counts — sizeof(T) N/A.
- 4.4: only ascending loop `for (std::size_t g=0; g<partials.size(); ++g)` (cpp:96) — terminates.
- 4.5: g and partials.size() are both size_t (cpp:96); part.n_block<=0 is int-vs-int (cpp:98) — no signed/unsigned mismatch.
- 4.7: host-pure unit; all pointers are host double*/int* from std::vector::data() — no device pointers.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Host-pure, CUDA-FREE unit (is_cuda=false, layer=core; hpp:13-14 "CUDA-FREE, host-pure, in steppe::core").
- 2.1 (dropped archs sm_50/60/70): no .cu, no CMake/nvcc arch flags in this unit; both files are C++ compiled into steppe_core without the device toolkit (hpp:14-16). N/A.
- 2.2 (texture/surface references removed in CUDA 12): no texture<>/surface<>/cudaBindTexture* — includes are only <algorithm>,<cstddef>,<span> + project headers (cpp:16-22). N/A.
- 2.3 (non-_sync warp intrinsics): no __shfl*/__ballot/__any/__all or any warp intrinsic — host code only. N/A.
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): no cuda* runtime calls at all (combine is std::copy_n over host vectors, cpp:103-110). N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
- 3.1: No commented-out CODE kept "just in case". All comments are narrative design/rationale
  (B5/B7/N2/P2). References to removed code describe deletions, not retained dead code:
  "the old `< 0 ? 0 :` ternary was dead" (cpp:62) and "the prior scalar `+=`" (cpp:92) document
  code that was REMOVED, not commented out and kept.
- 3.2: No `#if 0`, no code after return/break. Sole `return out;` at cpp:113 (function end);
  `continue;` at cpp:98 is reachable (empty-shard skip).
- 3.3: All includes used — cpp: <algorithm>→std::copy_n (103-110), <cstddef>→std::size_t,
  <span>→std::span, steppe/fstats.hpp→F2BlockTensor, device/shard_plan.hpp→DeviceShard,
  f2_partials_validate.hpp→validate_f2_partials (36); hpp: <span>/fstats.hpp/shard_plan.hpp all
  used in the decl (76-79). All locals read: slab (63,101,103,104), total (64,65), part (98,100,
  103,104,108), b0 (103,104,110), part_slabs (103,104). All 4 params (partials/shards/P/n_block_full) used.
- 3.4: No computed-but-unread values; every assignment (out.P, out.n_block, out.f2/vpair/block_sizes,
  slab, total, b0, part_slabs) feeds a later read or the returned `out`.
-->

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Host-pure, CUDA-FREE combine; no kernels, no launch dims, no device ids.
- 5.1: The only literals are 0.0 (cpp:64-65, +0.0 zero-init of f2/vpair) and 0 (cpp:66, block_sizes
  zero-init). These are the semantic identity/zero value passed to std::vector::assign(count, value),
  not tunable magic numbers — naming would not clarify. Loop start g=0 (cpp:96) and the empty-shard
  guard part.n_block <= 0 (cpp:98) are structural, not unnamed tunables.
- 5.2: All sizes derive from params/inputs — slab = (size_t)P*(size_t)P (cpp:59-60),
  total = slab*(size_t)n_block_full (cpp:63), part_slabs = slab*(size_t)part.n_block (cpp:100-101),
  offsets slab*b0 (cpp:103-104) and b0 (cpp:110). No hardcoded dim/bound that should be a param.
- 5.3: No duplicated constant that could drift. slab is computed once (cpp:59) and reused for f2 and
  vpair; the +0.0 init appears at cpp:64 and 65 but is the same trivial identity for the two tensors
  (no launch-dim / shared-mem-array pairing exists here — host-pure). No drift-risk pair.
- 5.4: No hardcoded paths, file ids, or device ids. Device order is the loop index g over
  partials.size() (cpp:96), driven by the caller's g=0..G-1 order — not a baked-in id.
- 5.5: No '32' (or any warp-size literal) anywhere — host code, no warp/block concept.
-->

## Group 6 — Naming

- [6.2][LOW] src/core/fstats/f2_combine.cpp:100-101 — `part_slabs` is named as a count of slabs but holds the element (double) count of the partial's f2/vpair run (`slab * part.n_block`, i.e. P·P·n_block doubles); the actual slab count is `part.n_block`. It is used correctly as the `std::copy_n` element count (cpp:103-104), and the inline comment "f2/vpair run length" clarifies intent, so this is cosmetic only. Suggested: rename to `part_elems` / `part_doubles` (or `part_run`) to match the value's unit.

<!--
Verified against the §12/FP64 context (g=0..G-1 fixed-device-order naming is the documented parity convention, NOT cryptic):
- 6.1: Names are short but each is a documented domain term or a tight loop counter. `slab` (cpp:59) = the
  P×P per-block element count, used as a noun throughout the comments; `total` (cpp:63) = full element count;
  `part` (cpp:97) = the current partial F2BlockTensor (clear in context); `b0` (cpp:99) mirrors the
  DeviceShard::b0 field name; `g` (cpp:96) is the §12 device index in the FIXED g=0..G-1 order (heavily
  documented, a loop counter); `P`/`n_block_full` are the public-API param names. No `tmp`/`data2`/`arr`/`flag`.
- 6.3: One consistent convention per file — snake_case lowercase with the `n_`/`block` family
  (n_block, n_block_full, block_sizes, b0/b1) and the public `P` capital (leading-dim domain convention).
  No nElements-vs-num_elements-vs-n mixing.
- 6.4: Abbreviations are standard/consistent — `part` (partial), `b0`/`b1` (the DeviceShard field names,
  reused verbatim), `lb` (local block, comment-only at cpp:81). None are obscure project-coined abbrevs.
-->

## Group 7 — Duplication

- [7.1][LOW] src/core/fstats/f2_combine.cpp:103-104 — the f2 and vpair placements are a copy-pasted pair of `std::copy_n` differing only by the member name (`part.f2`/`out.f2` vs `part.vpair`/`out.vpair`); both share the same `part_slabs` run length and the same `slab * b0` destination offset. A small local lambda `place(src, dst)` (or a loop over the two `{src,dst}` pairs) would fold the duplicated copy. Cosmetic only — both copies are correct and the block_sizes copy at 108-110 legitimately differs in type/count/offset so it would not fold into the same helper. Suggested: hoist a `auto place = [&](const double* s, double* d){ std::copy_n(s, part_slabs, d); };` and call it for f2 and vpair.
- [7.2][LOW] src/core/fstats/f2_combine.cpp:103-104 — the destination base `slab * b0` is recomputed in both the f2 and vpair `std::copy_n` calls (same operands, same iteration). Trivial recompute (one multiply), but it is a repeated loop-body expression. Suggested: compute `const std::size_t out_base = slab * b0;` once per iteration and reuse for both `out.f2.data() + out_base` and `out.vpair.data() + out_base`.

## Group 8 — Comments

- [8.2][MED] src/core/fstats/f2_combine.hpp:6-11 — the header's top narrative describes the OLD summing semantics the code no longer has: "It PLACES + SUMS the G per-device COMPACT partials" (hpp:6) and "Summing in a configuration-INDEPENDENT fixed order is exactly what makes the result BIT-IDENTICAL" (hpp:8-9). The current implementation is a pure PLACEMENT over DISJOINT shards (one `std::copy_n` per device, NO read-add-store), as the cpp narrative (cpp:11-12 "a PLACEMENT, not an accumulation") and the hpp's own doc-comment (hpp:35-39, 45-52) correctly state — this is the post-B7/N2 contract. The "+ SUMS"/"Summing" language is stale residue of the pre-B7 `+=`-onto-+0.0 version. Suggested: replace "PLACES + SUMS" with "PLACES" and "Summing in a … fixed order" with "Placing in a … fixed order" (the bit-identity argument is the disjoint-placement one, not a summation-order one).
- [8.2][LOW] src/core/fstats/f2_combine.hpp:18-20 — same stale wording in the P2P-sibling cross-reference: "both sum the same fixed order onto a zero-initialized full tensor" (hpp:19-20) describes summation, not the placement the disjoint shards actually do (cf. the doc-comment at hpp:39-43 "performs the SAME fixed-order placement on-device (onto a cudaMemset(0) accumulator)"). Suggested: reword to "both place the same fixed-order shards onto a zero-initialized full tensor", consistent with hpp:39-43.

<!--
- 8.1 (restating code): none. All comments are design/parity rationale (fixed g-order §12, IEEE-754 −0.0 preservation, disjoint placement vs +=), not mechanical restatements. Inline notes add semantics, e.g. cpp:98 "empty shard owns nothing (b0 == b1)" and cpp:101 "f2/vpair run length" explain WHY, not WHAT.
- 8.2 (stale): the two findings above (hpp:6-11, hpp:18-20). The cpp file itself is consistent (placement-only throughout). The cpp:62 and cpp:92 references to removed code ("the old `< 0 ? 0 :` ternary was dead", "the prior scalar `+=`") DOCUMENT past states deliberately and are accurate, not stale.
- 8.3 (missing rationale): none missing. The non-obvious choices all carry rationale: the +0.0 init + copy_n vs += and the −0.0 sign-flip hazard (cpp:46-55, 86-95; hpp:45-52), the no-clamp on n_block_full (cpp:61-62, cleanup C2), the fixed g-order parity law (cpp:68-95; hpp:6-20), the shared validator with the device tier (cpp:30-37, B5). Constants 0.0/0 are the semantic identity values (already covered Group 5).
- 8.4 (orphan TODO/FIXME/HACK): none — grep for TODO|FIXME|HACK|XXX over both files returns NONE.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Host-pure, CUDA-FREE combine; no tunable knobs (no thresholds/block-dims/buffer-sizes; precision policy never reaches the combine, cpp:93-95).
- 9.1 (should-be-const left mutable): every local that can be const already is — slab (cpp:59), total (cpp:63),
  b0 (cpp:99), part_slabs (cpp:100) are all `const std::size_t`; part (cpp:97) is `const F2BlockTensor&`.
  `out` (cpp:56) is intentionally mutable (built field-by-field then returned). Loop counter g (cpp:96) is
  necessarily mutable. The `std::span<const ...>` params (cpp:27-28) are already read-only views. No literal
  warrants constexpr — the only literals are the 0.0/0 identity passed to std::vector::assign (cpp:64-66).
- 9.2 (tangled config): no config knobs exist to tangle or surface. Every size derives from the API params
  (P, n_block_full) and the input partials/shards (slab, total, part_slabs, b0); no magic threshold or
  tunable buried in the loop body (cpp:96-111).
- 9.3 (positional booleans): no boolean arguments at any call site — validate_f2_partials takes
  (name, partials, shards, P, n_block_full) (cpp:36-37) and the three std::copy_n calls take (src, count, dst)
  (cpp:103-110). No foo(true,false,...) smell.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
- 10.1 (late/distant decl, or uninit-then-assigned): every local is declared AT first use WITH an
  initializer — slab (cpp:59-60), total (cpp:63), g (cpp:96, init 0), part (cpp:97), b0 (cpp:99),
  part_slabs (cpp:100-101). The out tensor is declared at cpp:56 and its fields are populated in the
  immediately following lines (P/n_block cpp:57-58, f2/vpair/block_sizes cpp:64-66) — adjacent, no gap.
  No `T x;` declared early then assigned far below; no uninitialized read.
- 10.2 (zero-init assumptions that don't hold): all "zero" state is set EXPLICITLY, not relied upon.
  out.f2/out.vpair are std::vector::assign(total, 0.0) (cpp:64-65), out.block_sizes assign(n_block_full, 0)
  (cpp:66) — explicit +0.0 / 0 fill, not an implicit value-init assumption. out.P/out.n_block are explicitly
  assigned (cpp:57-58). The disjoint shards then PLACE every owned slab (cpp:103-110); the +0.0 init survives
  only for a genuinely-unowned tail that validate_f2_partials proves cannot exist (covered == n_block_full,
  cpp:44-46). No field is left to rely on a default-member-init / global zero-init that does not hold — the
  vector::assign overwrites the full allocation deterministically before any read.
-->


