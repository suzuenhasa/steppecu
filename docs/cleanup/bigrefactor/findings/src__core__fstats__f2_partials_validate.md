# Review findings — src__core__fstats__f2_partials_validate

Files: /home/suzunik/steppe/src/core/fstats/f2_partials_validate.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verification notes (not findings):
- 4.1 (float/double): no floating math in this header — pure integer/index extent
  checks plus std::to_string for error strings. N/A.
- 4.2 / 4.6 (index width / overflow before widening): the one place that could
  overflow is the P*P*n_block extent (P up to ~2500, n_block up to ~757 ⇒ ~4.7e9
  > 2^31). It is computed correctly in size_t: `slab = (size_t)P * (size_t)P`
  (lines 92-93) then `want_slabs = slab * (size_t)part.n_block` (lines 122-123),
  matching F2BlockTensor::size()'s own size_t widening (fstats.hpp:74-76). No int
  intermediate. `span_blocks = sh.b1 - sh.b0` (lines 102, 198) is a block COUNT
  (≤ ~757), safe in int; `covered` is `long` and compared as `(long)n_block_full`
  (lines 98, 140-142, 194, 227-229) — no truncation.
- 4.3 (allocation sizing): no cudaMalloc/new/DeviceBuffer here — validation-only,
  CUDA-free header. N/A.
- 4.4 (unsigned countdown): both loops count UP `for (size_t g=0; g<size(); ++g)`
  (lines 99, 195) — no decrementing-unsigned wrap. N/A.
- 4.5 (signed/unsigned compare): loop counter `g` is size_t compared against
  `partials.size()` (size_t) — same signedness (lines 99, 195). Other comparisons
  (`part.n_block != span_blocks`, `part.P != P`) are int-vs-int. Clean.
- 4.7 (host/device pointer typing): no raw pointers; operates on std::span of
  CUDA-free value types (F2BlockTensor / DevicePartial / DeviceShard). N/A.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Verification notes (not findings):
- 2.1 (dropped archs): no CMake / arch lists / sm_* build flags in this header
  (CUDA-free, header-only INLINE per lines 15-22). N/A.
- 2.2 (texture/surface references): no texture<...>, surface<...>, or
  cudaBindTexture* — header is CUDA-free by contract (includes only <cstddef>,
  <span>, <stdexcept>, <string> + CUDA-free steppe headers, lines 31-38). N/A.
- 2.3 (non-_sync warp intrinsics): no __shfl/__ballot/__any/__all/warp intrinsics;
  pure host integer/index validation code. N/A.
- 2.4 (cudaThreadSynchronize): no CUDA runtime API calls of any kind. N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Verification notes (not findings):
- 3.1 (commented-out blocks "just in case"): no commented-out code anywhere — the
  comments (lines 1-27, 42-73, 150-177, inline 91-97, 102, 111-112, 119-125, 194,
  198) are all doc/contract prose explaining the §8 single-source guard, not stashed
  code. Clean.
- 3.2 (unreachable code): no `#if 0`, no code after return/break. Each `throw`
  (lines 82, 87, 105, 114, 129, 143, 186, 191, 201, 208, 214, 221, 230) is on a
  reachable conditional branch; the loops fall through to the final tiling check.
  No early return before a tail. Clean.
- 3.3 (unused symbols/includes/params): all 4 std includes are used — <cstddef>
  (std::size_t, lines 92, 124, 220), <span> (std::span params, lines 76-77,
  180-181), <stdexcept> (std::runtime_error throws), <string> (std::string/
  std::to_string, line 79 onward). All 3 steppe includes name a type actually used:
  steppe/fstats.hpp → F2BlockTensor (line 76, 100), device/shard_plan.hpp →
  DeviceShard (lines 77, 101, 181, 197), device/device_partial.hpp → DevicePartial
  (lines 180, 196). Every param of both functions (who/partials/shards/P/
  n_block_full) is read. Clean.
- 3.4 (computed but unread): `slab` (lines 92-93) is read at lines 122-123;
  `covered` accumulated and read at 142 / 229; `span_blocks` read at 104/140 and
  200/227; `prefix`, `want_slabs`, `want_counts` all consumed in throw strings. No
  dead assignment. Clean.
-->

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Verification notes (not findings):
- 5.1 (unnamed literals -> named constants): the only numeric literals are `0` —
  the negative-bound guard `P < 0 || n_block_full < 0` (lines 86, 190), the
  non-empty-partial gate `part.n_block > 0` / `part.n_block_local > 0` (lines 113,
  121, 213, 219), and the accumulator init `covered = 0` (lines 98, 194). All are
  genuine identity/boundary comparisons (negativity, emptiness, zero-init) whose
  meaning is self-evident, not tunable magic numbers. No 0.001f / 1024 / 0.5-style
  literals. N/A.
- 5.2 (hardcoded sizes/bounds that should be params/derived): every size is either
  a parameter (`P`, `n_block_full`, lines 78, 182) or DERIVED from the inputs —
  `slab = P*P` (92-93), `want_slabs = slab*n_block` (122-123), `want_counts =
  n_block` (124-125), `span_blocks = sh.b1 - sh.b0` (102, 198), and counts read
  from `partials.size()` / `.size()`. No hardcoded extent or bound. Clean.
- 5.3 (duplicated constants / DRIFT correctness bug): no shared numeric constant is
  copied between sites that could drift. The two overloads duplicate *logic* (the
  partials/shards count check, the `P<0||n_block_full<0` check, the tiling
  accumulator, the `span_blocks` derivation, the prefix string) — but that is the
  intentional twin-overload design (host F2BlockTensor vs device DevicePartial, doc
  lines 150-157), not a magic-number copy, and there is no kernel here so no
  launch-dim-vs-shared-array drift class exists. No DRIFT bug. Clean.
- 5.4 (hardcoded paths/IDs/device ids): none — CUDA-free header (doc lines 15-22),
  no filesystem paths, no device IDs, no stream/handle ids. N/A.
- 5.5 (ambiguous 32, warp size vs other): no literal 32 (or any warp-size constant)
  anywhere in the file — pure host integer/extent validation. N/A.
-->

## Group 6 — Naming

No Group 6 issues found.

<!--
Verification notes (not findings):
- 6.1 (cryptic names): the short identifiers are all intentional, scoped, and
  documented. `g` (lines 99, 195) is the canonical shard/partial index g=0..G-1
  used by the referenced types' own docs (DevicePartial, DeviceShard) and the loop
  counter is a tight-loop counter — within scope. `part`/`sh` (lines 100-101,
  196-197) are obvious local aliases for the partial and shard. `slab` (92),
  `covered` (98, 194), `span_blocks` (102, 198), `want_slabs`/`want_counts`
  (122-125), `prefix` (79, 183) are all descriptive. `who` (param, lines 75, 179)
  is non-obvious in isolation but fully documented (@param who: the calling
  combine's qualified name prefixed onto every error, lines 65-67, 171) — purpose
  is unambiguous. No tmp/data2/arr/flag. Clean.
- 6.2 (misleading names): `covered` is a running block COUNT and is compared as a
  count (`covered != (long)n_block_full`, lines 142, 229) — accurately named.
  `span_blocks` is a block count (b1-b0), not a byte span. `n_block_full` is the
  total block count, `n_block`/`n_block_local` the per-partial counts — each name
  matches its quantity (count axis, not index). `b0`/`b1` are the half-open block
  range bounds, consistent with their use `[b0, b1)`. No count-that-is-an-index or
  list-that-is-a-map mismatch. Clean.
- 6.3 (inconsistent conventions in one file): consistent snake_case throughout —
  `n_block`/`n_block_local`/`n_block_full`/`span_blocks`/`block_sizes`/`want_slabs`/
  `want_counts`. No mixed nElements/num_elements/n styles; the `n_*` prefix family
  is uniform. The two overloads (validate_f2_partials / validate_resident_partials)
  use the same prefix/covered/span_blocks idioms. Clean.
- 6.4 (nonstandard abbreviations): the abbreviations used are project-standard and
  carried from the referenced public/device types — `vpair` (pairwise-valid count,
  fstats.hpp:54-60), `P` (population count), `f2` (the f2 statistic), `b0`/`b1`
  (block range), `g`/`G` (shard index/count), `sh` (shard). All are domain vocab
  matching the cited headers, not invented contractions. Clean.
-->

## Group 7 — Duplication

- [7.1][LOW] f2_partials_validate.hpp:74-148 vs 178-235 — the two validator overloads (`validate_f2_partials` over `F2BlockTensor` and `validate_resident_partials` over `DevicePartial`) share an identical scaffold copy-pasted byte-for-byte: the prefix build (79 / 183), the `partials.size() != shards.size()` check and its full error string (81-85 / 185-189), the `P < 0 || n_block_full < 0` check (86-88 / 190-192), the `long covered = 0` accumulator + `span_blocks = sh.b1 - sh.b0` + `covered += span_blocks` tiling loop scaffold (98-102/140 / 194-198/227), and the final tiling-mismatch check with its full error string (142-147 / 229-234). Only the per-g middle body differs (host checks f2/vpair/block_sizes storage extent; resident checks b0-offset + block_sizes count). The header's own doc (lines 11-13) frames eliminating exactly this kind of lock-step duplication as the §8 single-source goal. Note: Group 5.3 already classifies this as intentional twin-overload design; flagging at LOW only since the *shared scaffold* (not the type-specific body) is mechanically foldable. Suggested: hoist the prefix + count-check + negative-bound check + tiling accumulator/final-check into a small shared helper taking `(who, P, n_block_full, std::span<const DeviceShard>)` and a per-g callback (or a `validate_shard_count_and_tiling` pre/post pair), leaving each overload only its type-specific per-g body.
- [7.2][LOW] f2_partials_validate.hpp:79,183 — `prefix = std::string(who) + ": "` is materialized once per call but only ever consumed inside throw branches that are taken at most once before the function exits; this is fine (off the bandwidth-critical path per doc §11.4, and the alloc is trivial). No loop-invariant recomputation inside the per-g loops. No action needed beyond the 7.1 fold. Suggested: none (subsumed by 7.1).

## Group 8 — Comments

- [8.2][MED] f2_partials_validate.hpp:23-27 — STALE comment: the header-level note claims "The P2P tier additionally checks `device_ids.size() == G` (it threads a third parallel span the host tier has no notion of); that one extra check stays at the P2P call site". But the device-resident tier no longer threads a separate `device_ids` span — the current `combine_f2_partials_resident` signature is `(partials, shards, P, n_block_full, root_device_id)` (p2p_combine.cu:61-64) with the device id carried PER-HANDLE in `DevicePartial.device_id` (device_partial.hpp:42), and the sibling header p2p_combine.hpp:64-66 explicitly states "the CUDA-free caller no longer threads a parallel `device_ids` span". Neither `validate_resident_partials` nor the P2P call site performs any `device_ids.size() == G` check (the only two `device_ids` mentions left in the whole P2P path are this comment and the p2p_combine.hpp prose). The note describes an "extra check" that no longer exists and could send a maintainer hunting for a phantom guard. Suggested: rewrite to reflect that the device tier folds the per-handle `device_id`/`b0` into the `DevicePartial` and that `validate_resident_partials` instead cross-checks `partials[g].b0 == shards[g].b0` (line 207), rather than referencing a removed parallel span.
- [8.1][—] f2_partials_validate.hpp:98,102,194,198 — Verified NOT restating: the inline comments (`// running count of blocks the shards account for`, `// shard's block count (>= 0)`) add the accumulator's purpose and the `>= 0` invariant rather than echoing the code. No finding.
- [8.3][—] f2_partials_validate.hpp:91-97,111-112,119-125,166-168 — Verified rationale PRESENT: the size_t widening (matches F2BlockTensor::size()), the empty-partial-exempt-from-P rationale, the short-partial OOB guard (cleanup B5/C1), and why the resident f2/vpair extent is "not re-checkable from this CUDA-free header" are all explained. No missing-rationale finding.
- [8.4][—] No orphan TODO/FIXME/HACK/XXX markers in the file (grep clean). No finding.

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Verification notes (not findings):
- 9.1 (should-be-const left mutable): every local that can be const already is.
  `prefix` is `const std::string` (lines 79, 183); `slab` is `const std::size_t`
  (lines 92-93); inside both loops `part`/`sh` are `const&` (100-101, 196-197),
  `span_blocks` is `const int` (102, 198), and `want_slabs`/`want_counts` are
  `const std::size_t` (122-125). The only mutable local is `covered` (98, 194),
  which is a genuine running accumulator (`covered += span_blocks`, 140/227) and
  MUST stay mutable. Both functions are `inline` (74, 178) — correct for a
  header-only guard — and are intentionally NOT constexpr (they throw + build
  std::strings). No should-be-const-but-mutable case. Clean.
- 9.2 (tangled config / buried tunable knobs): this is a pure fail-fast precondition
  validator with NO tunable knobs — no thresholds, buffer sizes, block dims, stream
  counts, or policy constants. The only literals are `0` identity/boundary checks
  (negativity 86/190, emptiness 113/121/213/219, accumulator init 98/194), already
  verified by Group 5 as non-tunable. Nothing belongs in a surfaced config struct.
  N/A.
- 9.3 (positional booleans): neither validate_f2_partials (74-78) nor
  validate_resident_partials (178-182) takes any bool parameter, and there are no
  function CALLS with positional bool args inside the file (only std::to_string /
  std::runtime_error / std::string ctor). No foo(true,false)-style ambiguity. N/A.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Verification notes (not findings):
- 10.1 (late/distant decl, uninitialized-then-assigned): every local is declared
  AND initialized at its point of first use. `prefix` (79/183) is built from `who`
  at the top, just before the first check that consumes it. `slab` (92-93) is
  declared-initialized once immediately before the loop that reads it (122-123).
  `covered` (98/194) is declared `long covered = 0` directly above the loop that
  accumulates into it. The per-g loop locals `part`/`sh`/`span_blocks` (100-102 /
  196-198) and the conditional locals `want_slabs`/`want_counts` (122-125) are all
  declared-initialized at use inside their scope. No `T x; ... x = ...;` split-decl
  pattern, no variable declared far from first use. Clean.
- 10.2 (zero-init assumptions that don't hold): the sole accumulator `covered`
  carries an EXPLICIT `= 0` initializer (98/194) — it does not rely on implicit
  zero-init — and is then summed via `covered += span_blocks` (140/227). No static
  local, no global, no class member, no aggregate `= {}`/default-construction, and
  no malloc/cudaMalloc/new buffer anywhere in this CUDA-free header, so there is no
  surface that could silently depend on (non-guaranteed) zero-fill. Clean.
-->

