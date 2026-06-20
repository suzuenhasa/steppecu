# Review findings — src__device__vram_budget

Files: src/device/vram_budget.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (why clean), for the record:
- 4.1: only FP64 used is the intentional utilization-fraction multiply at line 107
  (`kMaxVramUtilizationFraction * static_cast<double>(net)`); no wrong narrowing.
- 4.2/4.6: at scale (P≤2500, n_block≤757) `2·P²·n_block` ≈ 9.5e9 overflows 32-bit
  int, but lines 64-65 cast P/n_block to std::size_t BEFORE multiplying (line 67),
  and per_block_chunk_bytes (lines 81-84) and the min/clamp (lines 150-155) are all
  done in std::size_t before the final narrowing. The `2u`/`4u` literals promote to
  size_t in the mixed expression. The final `static_cast<int>` at line 156 is
  provably ≤ kMaxGridZ (65535), so it cannot overflow.
- 4.3: no allocation here (pure budget arithmetic); `* sizeof(double)` present at
  lines 67 and 84 where byte counts are produced.
- 4.4/4.5: no loops.
- 4.7: no pointers — all params are scalar std::size_t / int.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (why clean), for the record:
- This header is host-pure / CUDA-FREE by design (lines 13-18, "No CUDA header here";
  only includes <algorithm>, <cstddef>, launch_config.hpp, steppe/config.hpp).
- 2.1: no arch/sm_xx build flags or CMake arch lists here (build-system concern, not
  this header). kMaxGridZ (65 535, line 152) is a capability-independent grid-z limit,
  valid on every CUDA-13 arch (min sm_75); not a dropped-arch reference.
- 2.2: no texture<...> / surface<...> / cudaBindTexture* — no texture/surface API at all.
- 2.3: no warp intrinsics (no __shfl/__ballot/__any/__all, _sync or otherwise).
- 2.4: no cudaThreadSynchronize / cudaDeviceSynchronize — no CUDA runtime call exists
  in this TU.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (why clean), for the record:
- 3.1: no commented-out code blocks kept "just in case"; all comments (lines 1-28,
  the /// doc-comments, the inline rationale at 66, 74-75, 81-82, 92-97, 113-122,
  125-132, 148-149, 153-154) are explanatory prose, not disabled code.
- 3.2: no #if 0, no #ifdef-disabled regions; the only return-before-code is the
  early `return 0` at line 142 (nb_total <= 0 guard) and line 63 (P/n_block <= 0) —
  both are guard clauses on a separate path, the code after them is reachable on the
  other branch. No code after an unconditional return/break.
- 3.3: every include is used — <algorithm> (std::min line 151, std::max line 155),
  <cstddef> (std::size_t throughout), launch_config.hpp (core::kMaxGridZ line 152),
  steppe/config.hpp (kMaxVramUtilizationFraction lines 45/107, kCublasWorkspaceBytes
  line 105). All 4 functions (resident_tensor_bytes, per_block_chunk_bytes,
  chunk_budget_bytes, max_blocks_per_chunk) are inline header API consumed by the
  backend and host tests (header purpose, lines 8-12), not dead symbols. No params
  go unread: every parameter feeds the return value in each function.
- 3.4: no computed-but-unread locals — p, nb (62-67); p, sp, slab (81-84); reserved,
  net (105-107); budget, per_block, fit, capped, clamped (143-156) each flow into the
  returned expression. The static_assert (45-47) is consumed at compile time.
-->

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/device/vram_budget.hpp:67 — the `2u` resident-tensor pair count (f2 AND vpair) is an unnamed literal; it encodes the "two co-resident tensors" fact that is the whole point of the file (B26 2× under-count fix). Documented inline (line 66) so meaning is clear, but a named `kResidentTensorCount` would make the budget contract self-describing. Suggested: optionally name it `kResidentTensorCount = 2`; LOW since it is documented and local.
- [5.1][LOW] src/device/vram_budget.hpp:84 — the two `4u` per-block stack counts (`4·P·s_pad` inputs + `4·P²` outputs) are unnamed literals. They are explained as "structural input/output stack counts, not tunable" (lines 74-75) and derive from Qg+Vg+Sg(2) = 4 and Gg+Vpairg+Rg(2) = 4, so they are structural, not tunable. Suggested: optionally `kChunkInputStacks`/`kChunkOutputStacks = 4`; LOW (documented, single expression).

<!--
Notes (why otherwise clean), for the record:
- 5.1: the only other literals are sentinel/identity values, not magic numbers:
  the `0`/`0u` guard returns (lines 63, 142, 81-82, 106) and the `1u` single-block
  floor (line 155, explained lines 153-154). These are structural sentinels, not
  tunables to name.
- 5.2: no hardcoded sizes/bounds that should be params — every bound is either a
  function param (P, n_block, s_pad, nb_total, free_vram) or a named centralized
  constant (kMaxVramUtilizationFraction line 107, kCublasWorkspaceBytes line 105,
  core::kMaxGridZ line 152). The grid-z cap is correctly the single-home named
  constant, not an inline 65535.
- 5.3: no DRIFT-prone duplication. The two `4u` on line 84 are conceptually distinct
  (input vs output stack counts) but live in ONE expression — there is no
  independently-edited second site (e.g. no launch-dim-vs-shared-mem pair), so no
  drift-correctness hazard. The `2u`/`4u`/sizeof(double) appear once each per role.
- 5.4: no hardcoded paths, IDs, or device ids — this is pure integer budget math;
  free_vram is passed in (e.g. from cudaMemGetInfo at the call site), no device
  selection here.
- 5.5: no ambiguous `32` (no warp-size constant, no thread-block dim — this header
  has no kernel launch). The only large literal, 65535, is the named kMaxGridZ
  (grid-z hardware limit), unambiguous.
- sizeof(double) is used (lines 67, 84) rather than a magic `8`, so the byte width
  is type-derived, not a hardcoded 8 — good practice, not a finding.
-->

## Group 6 — Naming

- [6.2][LOW] src/device/vram_budget.hpp:147 — `fit` is an int-count local (blocks that fit per chunk) named like a verb/predicate; it reads as a boolean/action rather than a quantity ("largest number of blocks ... given the budget", doc line 110). Minor at this scope (single function, defined+documented at line 147). Suggested: optionally rename to `fit_blocks` / `blocks_that_fit` so the noun-count meaning is self-evident.

<!--
Notes (why otherwise clean), for the record:
- 6.1: no cryptic/opaque names. The short locals p/nb (lines 64-65), sp/slab (lines
  81-83) are lowercase echoes of the documented params P/n_block/s_pad used within
  ≤6-line constexpr bodies — clear in tight scope, the §6.1 loop-counter exemption
  spirit. No tmp/data2/arr/flag anywhere. `slab` (P²) is named, not a bare `s`.
- 6.3: conventions are consistent for the codebase. Uppercase `P` (populations) is the
  project-wide domain convention (matches fstats.hpp / architecture.md usage); every
  other param/local is snake_case (n_block, s_pad, nb_total, free_vram, per_block,
  resident_tensor_bytes, max_blocks_per_chunk). No nElements-vs-num_elements-vs-n style
  drift in one file. Constants use the kFoo convention (kMaxGridZ, kCublasWorkspaceBytes,
  kMaxVramUtilizationFraction) consistently.
- 6.4: no nonstandard abbreviations in the API surface. `nb`/`sp` are local
  abbreviations of n_block/s_pad but are documented one line above their use and never
  escape the function body. Qg/Vg/Sg/Gg/Vpairg/Rg (lines 71-74) appear ONLY in
  explanatory comments referencing the kernel's own variable names (cross-reference,
  not a declared identifier here), so no rename obligation in this header.
-->

## Group 7 — Duplication

- [7.4][LOW] src/device/vram_budget.hpp:63-65, 81-82 — the "non-negative `int` param → `std::size_t`" conversion is done TWO different ways across the two byte-count fns: `resident_tensor_bytes` uses an early `return 0` guard (line 63) then a bare cast (64-65), while `per_block_chunk_bytes` uses an inline negative-clamp ternary (`P < 0 ? 0 : P`, lines 81-82). Same intent (clamp negatives to 0, widen to size_t), divergent boilerplate — a `static_cast`-with-clamp helper (e.g. `nonneg(int)`) would fold both and make the two fns consistent. LOW: both are correct and the divergence is cosmetic; the callers already guard P,n_block,s_pad > 0. Suggested: optional `inline constexpr std::size_t nonneg(int) noexcept` helper used by both casts.

<!--
Notes (why otherwise clean), for the record:
- 7.1: no copy-pasted block differing only by a constant. resident_tensor_bytes
  (62-68) and per_block_chunk_bytes (80-85) share a family resemblance (cast P, form
  a product, `* sizeof(double)`) but the bodies differ structurally — `2·p²·nb` vs
  `4·p·sp + 4·p²`, and different negative-input handling (see 7.4). Not a
  constant-only delta; extracting a single template/fn would obscure two distinct
  budget terms. The 3 functions chunk_budget_bytes/max_blocks_per_chunk are each
  unique single-purpose composers, not clones.
- 7.2: no repeated loop-invariant expression to hoist — there are NO loops in this
  header (it is pure scalar budget arithmetic). `p * p` appears in both
  resident_tensor_bytes (line 67) and per_block_chunk_bytes (line 83 as `slab`) but
  they are SEPARATE function-local `p` values in separate TUs of arithmetic — not the
  same value recomputed in one scope, so nothing to compute-once. Within each fn no
  subexpression is evaluated twice (e.g. line 83 already names `slab = p*p` and reuses
  it on line 84).
- 7.3: `sizeof(double)` appears at lines 67 and 84, and `static_cast<std::size_t>`
  several times (64,65,81,82,107,151,152) — but these are unavoidable per-expression
  type operations in distinct functions, not a hoistable repeated cast of the SAME
  value. sizeof(double) is intentionally used over a magic `8` (Group 5 note). The
  std::min braced cast list (151-152) casts two distinct operands once each. No
  repeated cast of one operand that could be bound to a single local.
-->

## Group 8 — Comments

No Group 8 issues found.

<!--
Notes (why clean), for the record:
- 8.1 (restating code): every comment is rationale/why, not a what-restatement. The
  inline comments at 66 ("f2 AND vpair: two equal [P×P×n_block] FP64 tensors, the B26
  2× term"), 74-75 (the structural 4-stack breakdown), 81-82, 92-97, 148-149, 153-154
  all explain INTENT, not the literal operation. No "i++ // increment i" pattern.
- 8.2 (stale comments): all comments match the current code. resident_tensor_bytes
  doc + line-66 comment ("2·P²·n_block, both f2 and vpair") match the code `2u*p*p*nb`
  (line 67). per_block doc (71-75: Qg+Vg+Sg=4, Gg+Vpairg+Rg=4) matches `4u*p*sp+4u*slab`
  (line 84). max_blocks_per_chunk doc (113-122: min(quotient,nb_total,kMaxGridZ) floored
  at 1) matches lines 147-156. The line-142 "empty bucket" comment matches `nb_total<=0
  return 0`. The grid-z narrative (125-132) matches the kMaxGridZ clamp (line 152). No
  comment describes behavior the code no longer has.
- 8.3 (missing rationale): the non-obvious constants/choices all CARRY rationale — the
  2u pair count (66), the 4u stacks "not tunable" (74-75), the single-block floor
  (153-154), the saturate-to-0 fail-fast (96-97), the (0,1] static_assert (40-44), the
  grid-z cap on the high-VRAM tier (125-132), and the X-5/F9 size_t-before-narrow wrap
  fix (116-122, 148-149). Provenance tickets (B26/X-5/X-13/B5/B6, architecture.md
  §11.1/§11.2) anchor each deviation. Nothing under-explained.
- 8.4 (orphan TODO/FIXME/HACK): grep for TODO|FIXME|HACK|XXX|WIP found NONE. The
  F14/F15/B26/X-5/X-7 tokens are cited cleanup/review identifiers WITH context
  (architecture.md sections), not orphan action markers.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Notes (why clean), for the record:
- 9.1 (should-be-const/constexpr left mutable): nothing mutable that should be
  const. resident_tensor_bytes (62), per_block_chunk_bytes (80) and
  max_blocks_per_chunk's helpers are `inline constexpr` (62, 80) / `inline`
  (chunk_budget_bytes 103, max_blocks_per_chunk 140 — these multiply by the
  runtime-evaluated double kMaxVramUtilizationFraction so are inline, not
  constexpr-forced, which is correct). EVERY local is already declared `const`:
  p/nb (64-65), p/sp/slab (81-83), reserved/net (105-106), budget/per_block/fit/
  capped/clamped (143-155). The (0,1] domain pin is a compile-time static_assert
  (45-47), not a runtime mutable check. All four fns are [[nodiscard]] + noexcept.
- 9.2 (tangled config / buried tunables): there are NO tunable knobs buried in
  this logic — every policy number is pulled from its named central home:
  kMaxVramUtilizationFraction and kCublasWorkspaceBytes from steppe/config.hpp
  (used at 45/107 and 105), core::kMaxGridZ from core/internal/launch_config.hpp
  (line 152, the single-home grid-z constant = 65535u). This header's whole reason
  for existing (lines 8-12) is to be the ONE shared home for the budget math so the
  build()-time check and the in-stream chunk sizing cannot drift — config is
  surfaced, not tangled. The only in-file literals (2u line 67, 4u line 84) are
  STRUCTURAL counts (resident-tensor-pair count; input/output stack counts "not
  tunable", lines 74-75), already captured as Group 5.1 LOW; they are not config
  knobs that belong in a config struct.
- 9.3 (positional booleans): NONE. No function in this header takes a bool
  parameter — all params are scalar std::size_t (free_vram) / int (P, n_block,
  s_pad, nb_total). No foo(true,false,...) call sites exist here (there are no
  calls out to bool-flagged APIs; chunk_budget_bytes and the helpers take only
  numeric quantities). No flag-vs-enum opportunity.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Notes (why clean), for the record:
- 10.1 (late/distant or uninit-then-assigned): every local is declared `const`
  AT its point of first use WITH an initializer — p/nb (lines 64-65), p/sp/slab
  (81-83), reserved/net (105-106), budget/per_block/fit/capped/clamped (143-155).
  None are declared uninitialized and assigned later, and none are declared far
  from first use (each function body is ≤7 lines and every local feeds the very
  next expression). No raw-declare-then-fill pattern anywhere.
- 10.2 (zero-init assumptions that don't hold): NONE. There are no aggregates,
  arrays, structs, std::array, or static/global storage in this header — only
  explicitly-initialized scalar std::size_t/int locals. No `= {}`, no memset, no
  default-construct-then-rely. The literal `0`/`0u` returns (lines 63, 81-82, 106,
  142) and the `1u` floor (line 155) are EXPLICIT initializers/sentinels chosen as
  values, not reliance on implicit zero-init. The function-local arithmetic never
  reads a value before it is set.
-->

