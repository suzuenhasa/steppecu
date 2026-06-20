# Review findings — src__device__tier_select

Files: /home/suzunik/steppe/src/device/tier_select.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verification notes (not findings):
- 4.2/4.6 (index/overflow at scale): All sizing arithmetic widens to std::size_t
  BEFORE multiplying. resident_working_set_bytes (line 58) does
  static_cast<std::size_t>(P) * static_cast<std::size_t>(M) first; streamed_working_set_bytes
  (lines 85-91) widens P/max_tile/max_nb/max_s_pad to size_t before every product; the
  reused resident_tensor_bytes (vram_budget.hpp:64-67) likewise widens P/n_block first.
  No int global index/offset, no i*P+j in int. Clean at P~2500, M~584131, n_block~757.
- 4.1 (float/double): only float math is the budget fractions (lines 125, 128),
  kResidentTierVramFraction/kHostTierRamFraction * static_cast<double>(free_vram/host_ram).
  FP64 is intentional and the result is only a parity-neutral budget threshold. Not a finding.
- 4.3 (alloc sizing): no cudaMalloc/new/DeviceBuffer here; byte computations correctly
  include sizeof(double) (lines 59, 92). N/A.
- 4.4/4.5 (unsigned countdown / signed-unsigned compares): no loops; guards (P<=0, M<=0,
  max_tile<0) are signed-vs-literal. Clean.
- 4.7 (host/device pointer typing): only pointer is const char* env_value (host env string,
  line 138); no device pointer. N/A.
- Note: 7u/4u/2u/3u unsigned-int literals multiplied against std::size_t operands promote
  the literal to size_t (LP64), so no 32-bit truncation. Correct as written.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Verification notes (not findings): host-pure, CUDA-FREE header (only #include <cstddef>;
no CUDA header per lines 12, 22). 2.1 no CMake/arch flags or sm_* lists here. 2.2 no
texture<>/surface/cudaBindTexture* — pure std::size_t budget arithmetic. 2.3 no warp
intrinsics (no __shfl/__ballot/__any/__all/__activemask). 2.4 no cudaThreadSynchronize /
no CUDA runtime calls at all. All four Group 2 tasks N/A.
-->

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/tier_select.hpp:82 — `streamed_working_set_bytes` is an unreferenced inline helper: a codebase-wide grep finds zero call sites (only this definition). Its own doc comment (lines 73-75) admits it is "NOT used by select_output_tier" and is merely "exposed for the bench's high-P feasibility narrative and any future select" — but no bench or other caller invokes it. A documented-intent dead helper kept "just in case". Suggested: drop it (and its `max_tile`/`max_nb`/`max_s_pad` params) until a real caller needs it, or wire it into the bench/select that the comment promises.

<!--
Verification notes (not findings):
- 3.1 (commented-out blocks): none. All non-code lines are explanatory doc comments
  describing live policy (lines 1-18, 29-32, 42-55, 62-76, 95-117, 133-136); no
  commented-out statements kept "just in case".
- 3.2 (unreachable / #if 0 / code after return): no #if 0; the early `return`s in
  resident_working_set_bytes (57), streamed_working_set_bytes (84), select_output_tier
  (121), are guard returns with reachable code after only on the other branch — all
  reachable. No dead branches.
- 3.3 (unused includes/symbols): includes both used — vram_budget.hpp (24) for
  resident_tensor_bytes (lines 15,122) + DeviceF2Blocks footprint reuse; config.hpp (25)
  for kCublasWorkspaceBytes (59,92), kResidentTierVramFraction (125), kHostTierRamFraction
  (128), DeviceConfig::ForceTier (138). <cstddef> for std::size_t. All other declared
  symbols ARE referenced elsewhere: OutputTier (f2_blocks_out.hpp:63, stream_f2_blocks.hpp:36,
  host_ram.cpp, cuda_backend.cu:405/411, f2_blocks_multigpu.cpp), resident_working_set_bytes
  (used by select_output_tier line 123), free_host_ram_bytes (host_ram.cpp:17,53; tests;
  f2_blocks_multigpu.cpp:352), select_output_tier (host_ram.cpp:53; tests; bench),
  resolve_output_tier (f2_blocks_multigpu.cpp:353). Only streamed_working_set_bytes is
  unreferenced — see the [3.3] finding above.
- 3.4 (computed but unread): no local that is assigned and never read; every local in
  select_output_tier (result_bytes, resident_need, vram_budget, host_budget) feeds a
  return/compare. The intentionally-unused `M` param of streamed_working_set_bytes is
  marked /*M*/ (line 83) with a doc note (line 78) — deliberate call-site symmetry, not a
  finding.
-->

## Group 5 — Hardcoded values / magic numbers

- [5.1][MED] src/device/tier_select.hpp:59 — the resident feeder footprint coefficient `7u` (in `7u * pm * sizeof(double)`) is an unnamed literal whose meaning lives only in prose ("3·P·M raw + 4·P·M feeder out = 7·P·M", lines 44-48). The same `3` and `4` decomposition is re-expressed literally in the streamed helper (line 89: `3u * p * t + 4u * p * t`), so the 7 here and the 3+4 there must stay in lockstep with the actual feeder allocation in cuda_backend.cu — a silent drift between this policy literal and the real malloc set undersizes the Resident budget and risks a tier mis-select / OOM. Suggested: name the feeder layout coefficients once (e.g. `kFeederRawBufsPerPop = 3`, `kFeederOutBufsPerPop = 4`) in config/vram_budget and use them in both helpers so the 7 is derived (3+4), not a separate magic number.
- [5.3][MED] src/device/tier_select.hpp:89-91 — the streamed working-set coefficients are duplicated, undocumented-at-site magic numbers that must mirror the actual kernel/ring allocation: `3u`/`4u` (raw inputs / tile feeder outputs), `4u`/`4u` (gather + GEMM scratch, `4·P·s_pad + 4·P²`), and `2u` for the device ring (line 91 comment says "×2 = kStreamDeviceChunks"). The `2` is the literal value of `kStreamDeviceChunks` (the doc on line 69 names that constant) yet is hardcoded as a bare `2u` here, so if the ring depth ever changes this estimate drifts from the real ring without a compile error — a DRIFT correctness hazard for the streamed-feasibility math. Suggested: replace the bare `2u` with `kStreamDeviceChunks` and pull the gather/GEMM `4·P·s_pad + 4·P²` term from the single `per_block_chunk_bytes` source it is supposed to mirror, rather than re-spelling the coefficients.
- [5.1][LOW] src/device/tier_select.hpp:90 — the gather/GEMM scratch term re-hardcodes `4u * p * sp + 4u * p * p` with two independent `4` literals; per the line-90 comment this is meant to equal `per_block_chunk_bytes` defined elsewhere, but the value is copied rather than referenced. Suggested: factor a named `per_block_chunk_elems(P, s_pad)` (or reuse the existing per_block_chunk_bytes) so the two `4`s are not maintained in two places.

Verification notes (not findings):
- 5.1/5.2: the three actual policy thresholds are NOT magic — kResidentTierVramFraction (0.70),
  kHostTierRamFraction (0.60) and kCublasWorkspaceBytes (64 MiB) are all named constexpr in
  include/steppe/config.hpp (lines 88, 119, 125) and consumed by name here (lines 25, 59, 92,
  125, 128). Correctly externalized. The 0.70/0.60 fractions even carry static_asserts in
  config.hpp. No bound is hardcoded that should be a param: P/M/n_block/free_vram/free_host_ram
  are all passed in (lines 118-120, 138-140).
- 5.4: no hardcoded path/ID/device id IN THIS HEADER. The "gpus[0]" device-0 reference is only
  in a doc comment (line 116) describing the caller's argument source, not code here; the env key
  STEPPE_FORCE_TIER and the strings "resident"/"host"/"disk" (lines 6-7, 133-135) are documented
  policy tokens resolved in resolve_output_tier's .cpp, not literals in this file.
- 5.5: no ambiguous `32`/warp-size literal anywhere in this host-pure header (no kernel launch,
  no shared mem). N/A.

## Group 6 — Naming

- [6.1][LOW] src/device/tier_select.hpp:86 — local `t` is a cryptic single-letter name for `max_tile` (the chunk column-union tile width), used in dense sizing arithmetic at line 89 (`3u * p * t + 4u * p * t`) where its meaning is not recoverable at the use site without scrolling back. Unlike the file's other widening locals (`p`, `nb`, `sp`, `pm`) which mirror established codebase vocabulary, `t` has no such precedent and its meaning is overloaded across the codebase: the sibling kernel widens this exact quantity to the clearer `max_tile_z` (cuda_backend.cu:893), while `t` elsewhere names an unrelated byte size (cuda_backend.cu:1598,1608). Suggested: rename to `tile` (or match the sibling `max_tile_z`) for a self-documenting use site.

<!--
Verification notes (not findings):
- 6.1 (cryptic names): the remaining short locals are NOT findings — `pm` (line 58, P*M) is an
  established codebase idiom (verbatim in cuda_backend.cu:201/514/768/1058 and cpu_backend.cpp:321);
  `p` (85), `nb` (87), `sp` (88) match the sibling vram_budget.hpp vocabulary (it uses `nb` for the
  n_block widening at vram_budget.hpp:65 and `sp` for the s_pad widening at vram_budget.hpp:82), and
  each is introduced on its own clearly-commented line. `feeder`/`slabs`/`ring` (89-91) are
  self-documenting. The single-letter math symbols `P`, `M` are the codebase-wide domain convention
  (vram_budget.hpp, config.hpp), not opaque. No `tmp`/`data2`/`arr`/`flag`.
- 6.2 (misleading names): each name matches its content — `result_bytes` (122) holds bytes,
  `resident_need`/`vram_budget`/`host_budget` (123-128) hold byte budgets, `OutputTier` enumerates
  tiers, `select_output_tier`/`resolve_output_tier` select/resolve a tier, `free_host_ram_bytes`
  returns bytes. `n_block` is a count and is used as a count (not an index). No count-as-index or
  list-as-map misnomer.
- 6.3 (inconsistent conventions in one file): the "number of blocks" appears as the public param
  `n_block`, the param `max_nb`, and the local `nb` — but this is the SAME convention as the sibling
  vram_budget.hpp (n_block param / nb local), and `max_nb` is just the `max_` prefix on `nb`; not an
  in-file inconsistency. Functions are snake_case (resident_working_set_bytes, select_output_tier),
  types are PascalCase (OutputTier), constants are kCamelCase (kResidentTierVramFraction) — each
  category internally consistent. P/M uppercase are the deliberate math-symbol convention. No
  nElements-vs-num_elements-vs-n mix.
- 6.4 (nonstandard abbreviations): `pm`, `nb`, `sp`, `s_pad`, `nb`/`max_nb`, `vram`, `n_block` are
  all standard project abbreviations shared with vram_budget.hpp/config.hpp. Only `t` (line 86) is a
  one-off abbreviation without precedent — covered by the [6.1] finding above.
-->

## Group 7 — Duplication

- [7.1][LOW] src/device/tier_select.hpp:124-128 — the budget-threshold block `static_cast<std::size_t>(kXxxFraction * static_cast<double>(free_yyy))` is copy-pasted twice in `select_output_tier`, differing only by the fraction constant (`kResidentTierVramFraction` vs `kHostTierRamFraction`) and the free-RAM source (`free_vram` vs `free_host_ram`) — a copy-pasted block differing by a constant (task 7.1). The identical shape also recurs at vram_budget.hpp:107 (`chunk_budget_bytes`), so this "floor(fraction·free) in size_t" idiom is repeated 3× across the device layer. Suggested: extract a tiny `inline std::size_t budget_bytes(double fraction, std::size_t free) noexcept { return static_cast<std::size_t>(fraction * static_cast<double>(free)); }` (in vram_budget.hpp/config) and call it from all three sites.
- [7.2][LOW] src/device/tier_select.hpp:89 vs :59 — the per-pop feeder term `3·P·X + 4·P·X` is expressed two ways for the SAME 3-raw + 4-feeder-output layout: the streamed helper spells it out as `3u * p * t + 4u * p * t` (line 89) while the resident helper collapses it to the magic `7u * pm` (line 59). The two encodings of one decomposition must stay in lockstep (see also [5.1] at line 59). Suggested: derive both from a single named layout (`kFeederRawBufsPerPop=3` + `kFeederOutBufsPerPop=4`, so the resident `7` becomes `(3+4)` and the streamed pair reuses the same names) — collapsing the duplicated 3/4 (and the derived 7) to one source.
- [7.4][LOW] src/device/tier_select.hpp:86-88 — the negative-clamp-then-widen idiom `static_cast<std::size_t>(x < 0 ? 0 : x)` is hand-repeated three times for `t`/`nb`/`sp`, collapsible boilerplate (task 7.4). Suggested: a one-line `inline std::size_t nn(int v) noexcept { return v < 0 ? 0u : static_cast<std::size_t>(v); }` folds the three lines.

<!--
Verification notes (not findings):
- 7.2 (repeated loop-invariant expr): no loops in this header, so no loop-hoist target.
  `resident_tensor_bytes(P, n_block)` is called once (line 122) and reused via the
  `result_bytes` local in both subsequent comparisons (123, 129) — already computed once,
  not re-evaluated. `resident_working_set_bytes(P, M)` is called once (line 123). No CSE miss.
- 7.3 (repeated sizeof/casts): `sizeof(double)` appears at lines 59 and 92 but in two
  DIFFERENT functions (one term each), not a repeat within a scope — folding it would not
  reduce duplication and the size_t widening of P/M/max_* is the necessary one-cast-per-
  operand overflow guard (Group 4 verified), not a redundant repeat. Not flagged.
- 7.4: the per-helper `if (P <= 0 ...) return 0/Resident` guards (lines 57, 84, 121) are
  similar in shape but guard three different return semantics (0 bytes vs 0 bytes vs
  OutputTier::Resident) across three functions — a deliberate per-API degenerate-input
  contract, not collapsible boilerplate. Not flagged.
-->

## Group 8 — Comments

- [8.2][MED] src/device/tier_select.hpp:47 — stale cross-reference: the resident_working_set_bytes doc says "The chunk slabs reuse the freed-raw VRAM (cuda_backend.cu :544-560) so they fit under this envelope", but cuda_backend.cu:544-560 is the pinned-DMA / page-walk amortization comment, NOT the freed-raw-VRAM reuse. The actual reuse text is at cuda_backend.cu:567 ("the freed raw VRAM is reused by the...") with the slab pre-sizing at :577. A reader following the cited range lands on unrelated prose. Suggested: repoint the citation to cuda_backend.cu:567 (and :577 slab pre-size), or drop the line numbers and cite the symbol.

<!--
Verification notes (not findings):
- 8.1 (restating code): none. Every inline comment names the PHYSICAL quantity its term
  represents rather than echoing the syntax — line 89 `// raw + tile feeder`, line 90
  `// gather/GEMM scratch`, line 91 `// §5 device ring (×2)`, line 121 `// degenerate empty
  -> existing no-op`, line 122 `// 2·P²·n_block·8` (the closed form of resident_tensor_bytes).
  Each adds meaning beyond the code. No `i++ // increment i`.
- 8.2 (other cross-refs verified ACCURATE, so NOT flagged): vram_budget.hpp:14-18 (cited at
  line 9 for the CUDA-free-in-src/device rationale) matches; test_f2_multigpu_parity.cu:223-230
  (cited at line 47 for the 7·P·M mirror) matches — line 225 is estimate_peak_vram_bytes, line
  227 holds `7u * pm * sizeof(double)`; cuda_backend.cu:893 (cited in the Group 6 note for
  max_tile_z) matches. The kStreamDeviceChunks named-constant referenced in prose (lines 69, 91)
  is a real `constexpr int = 2` at cuda_backend.cu:934 — the prose is current; the hardcoded `2u`
  drift is a Group 5 concern, not a stale comment. Only the :544-560 range above is stale.
- 8.3 (missing rationale): none. Every non-obvious choice carries an explicit WHY — the
  CUDA-free placement (lines 8-12), the 7·P·M / 3+4 feeder decomposition (lines 42-55), the
  M-independent streamed envelope and why it escapes the P≲768 feeder wall (lines 62-76), the
  sysinfo "available" proxy and the deliberate exclusion of reclaimable page cache (lines 95-99),
  the "fail -> Disk is the safe direction" return-0 contract (line 99), and the P<=0/n_block<=0
  -> Resident degenerate rule (lines 110-111, 121). The 0.70/0.60 fractions are named constexpr
  with their own static_asserts in config.hpp. (The bare-literal DRIFT hazard of 7u/3u/4u/2u is
  a Group 5 magic-number concern, not absent rationale — the rationale prose IS present.)
- 8.4 (orphan TODO/FIXME/HACK): none — grep for TODO/FIXME/HACK/XXX over the file returns
  nothing. The "any future select" / "for the bench's high-P feasibility narrative" forward-
  looking notes (lines 53, 74-75) are scoped rationale for streamed_working_set_bytes, not
  ownerless action markers (the dead-helper concern is the Group 3 [3.3] finding).
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Verification notes (not findings):
- 9.1 (should-be-const/constexpr left mutable): nothing mutable that should be const.
  All four free functions (resident_working_set_bytes 56, streamed_working_set_bytes 82,
  select_output_tier 118, resolve_output_tier 137) take params by value and never mutate
  them; the locals in select_output_tier (result_bytes 122, resident_need 123, vram_budget
  124, host_budget 127) are ALREADY `const`. OutputTier is an `enum class` (33). No
  file-scope mutable/static; no `#define` that should be constexpr — the policy thresholds
  are already constexpr in config.hpp (kResidentTierVramFraction 119, kHostTierRamFraction
  125, kCublasWorkspaceBytes 88). All inline functions are noexcept. Clean.
- 9.2 (tangled config / buried knobs): the three tunable knobs are surfaced, not buried —
  kResidentTierVramFraction (0.70), kHostTierRamFraction (0.60), kCublasWorkspaceBytes
  (64 MiB) are named constexpr at the top of include/steppe/config.hpp (lines 88/119/125,
  the latter two with static_asserts at 127-130) and consumed by name here (lines 25,59,
  92,125,128). The STEPPE_FORCE_TIER override is resolved in the .cpp, not buried in this
  policy logic. (The bare coefficient literals 7u/3u/4u/2u are a magic-number/drift hazard
  already captured under Group 5 [5.1]/[5.3]/[5.1-line90] — that is a constant-naming
  concern, not a config-surfacing one; not re-filed here.)
- 9.3 (positional booleans): none. The tier choice is an `enum class OutputTier` (33-40)
  and the override is an `enum class DeviceConfig::ForceTier { Auto, Resident, HostRam,
  Disk }` (config.hpp:348) passed to resolve_output_tier (138) — exactly the named-enum
  pattern 9.3 prescribes. No function here takes a bool; no foo(true,false,...) call site.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Verification notes (not findings):
- 10.1 (late/distant decl, uninitialized-then-assigned): every local in this header is
  declared `const` and initialized AT the point of declaration, immediately before its use.
  resident_working_set_bytes: `pm` (line 58) initialized at decl, used next line (59).
  streamed_working_set_bytes: `p` (85), `t` (86), `nb` (87), `sp` (88), `feeder` (89),
  `slabs` (90), `ring` (91) — each const, each initialized at decl, none declared far from
  first use. select_output_tier: `result_bytes` (122), `resident_need` (123), `vram_budget`
  (124-125), `host_budget` (127-128) — all const, all initialized at decl and consumed in the
  very next branch. No declare-then-assign-later pattern anywhere; no two-phase init.
- 10.2 (zero-init assumptions): nothing relies on implicit zero-init. No file-scope/static
  storage, no aggregate/array/struct constructed un-initialized, no member fields (the unit is
  four free functions + an enum + two extern decls). The degenerate guards explicitly RETURN
  literal values (return 0 at 57/84, return OutputTier::Resident at 121) rather than leaning on
  a default-zeroed accumulator. free_host_ram_bytes (100) and resolve_output_tier (137) are
  declarations only — their bodies live in host_ram.cpp / the .cpp (out of this unit's scope).
-->

