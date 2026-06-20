# Review findings — src__core__internal__launch_config

Files: /home/suzunik/steppe/src/core/internal/launch_config.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Reviewed against all 4.1–4.7 tasks with the FP64 + SCALE context:
- 4.1 (float/double): N/A — the file is pure integer arithmetic + named integer
  constants; no FP literals or math.
- 4.2 / 4.6 (index width / overflow before widening): The header explicitly
  handles the SNP/M-scale axis. A `cdiv(long,long)` overload (lines 89–91) exists
  for the M axis, and the int `cdiv` (82–84) / `grid_for` (114–127) are documented
  as P-axis (≤2500) only. Verified at the call sites: f2_block_kernel.cu:320 and
  decode_af_kernel.cu:106 cast M to `long` before `cdiv`, so `n + b - 1` is a long
  add (no int overflow); `grid_for` is only ever passed P or s_pad (P-scale,
  fits int). The (n + b - 1) add in the int cdiv does not overflow for any
  documented/observed input (P, s_pad, kCdivBlock all small). Contract is correct
  and honored.
- 4.3 (allocation sizing): N/A — no cudaMalloc / DeviceBuffer / new in this file.
- 4.4 (unsigned countdown): N/A — no loops.
- 4.5 (signed/unsigned compares): The two asserts (122, 144) guard the value
  (`extent >= 0`; `n_in_group >= 1`) BEFORE the `static_cast<unsigned>` compare
  against the unsigned cap — correct, no signed/unsigned mismatch.
- 4.7 (host/device pointer typing): N/A — no pointers; all values are scalars
  returned by value.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Reviewed against all 2.1–2.4 tasks. The file is a pure STEPPE_HD constexpr
integer-math + named-constant header (CUDA-free-compilable per its own banner,
lines 36–40); it issues no CUDA runtime calls, declares no kernels, and pulls in
only host_device.hpp + config.hpp.
- 2.1 (dropped Maxwell/Pascal/Volta archs, sm_50/60/70 build flags, CMake arch
  lists): N/A — no build flags / CMake / arch lists in a .hpp. The grid-limit
  constants kMaxGridX/Y/Z (lines 65/68/73) and the §7 banner (lines 49–60) are
  documented as verified for CUDA 13.x / Blackwell sm_120 — current, not for a
  dropped arch.
- 2.2 (texture/surface references, texture<...>/cudaBindTexture*): N/A — no
  texture or surface usage anywhere in the file.
- 2.3 (non-_sync warp intrinsics, e.g. __shfl/__ballot/__any/__all): N/A — no
  warp intrinsics; the file is grid/block extent math only.
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): N/A — no CUDA runtime
  API calls of any kind.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Reviewed against all 3.1–3.4 tasks. The file is a pure STEPPE_HD constexpr
integer-math + named-constant header; every symbol it declares is exported and
consumed elsewhere (verified by grep across src/ tests/ include/).
- 3.1 (commented-out blocks kept "just in case"): N/A — every comment block
  (banner lines 1–40, §-divider headers, the [[maybe_unused]] rationale at
  116–120) is live documentation/rationale, not stashed code. No `// foo();`
  style dead statements anywhere.
- 3.2 (unreachable code; #if 0; code after return/break): N/A — no #if 0, no
  preprocessor dead branches; the four functions are single-return constexpr/
  inline bodies with no code after their `return`. The only #if is the standard
  include guard (41–42, 171).
- 3.3 (unused symbols — vars/params/includes/helpers):
  * Both includes are used: host_device.hpp provides STEPPE_HD/STEPPE_ASSERT
    (lines 82, 89, 114, 122, 143, 144), config.hpp provides kCdivBlock (the
    default arg at line 114).
  * All exported symbols have real call sites: cdiv(int/long) — shard_plan.cpp:65,
    decode_af_kernel.cu:106, f2_block_kernel.cu:320; grid_for — f2_block_kernel.cu
    :321/380/381, f2_blocks_kernel.cu:197/198, decode_af_kernel.cu:107; grid_z_extent
    — f2_blocks_kernel.cu:199; kMaxGridZ — vram_budget.hpp:152, cuda_backend.cu:828;
    kDecodeBlockX/Y — decode_af_kernel.cu:45/46/105/106. kMaxGridX/kMaxGridY are
    exercised by tests/unit/test_launch_config.cpp (static_asserts 85–87, runtime
    81–139) and kMaxGridY is the load-bearing default of grid_for's max_grid param
    (line 115) — not dead.
  * The `max_grid` param of grid_for (line 115) is read only inside the assert
    (line 122), which compiles out under NDEBUG; this is correctly annotated
    `[[maybe_unused]]` (line 115) with an explicit rationale — intentional, not
    an unused-param defect.
- 3.4 (computed but unread): N/A — `extent` (line 121) is both asserted (122) and
  returned (126); no other locals. All four functions return their computed value.
-->

## Group 5 — Hardcoded values / magic numbers

- [5.3][LOW] src/core/internal/launch_config.hpp:73 — `kMaxGridZ = 65535u` repeats the literal value of `kMaxGridY` (line 68) rather than deriving from it. The y and z axes are genuinely distinct hardware grid axes that today share the 65 535 cap, so the duplication is defensible, but encoding the literal twice means a future per-axis hardware change must edit two sites in lock-step. Suggested: optionally `kMaxGridZ = kMaxGridY` (or a shared `kMaxGridYZ`) so the shared-cap fact is single-sourced; LOW because both are correctly named and the values are the verified §7 hardware spec.

<!--
Reviewed against all 5.1–5.5 tasks with the §12/FP64/scale context. This is a
pure named-constant + constexpr-integer-math header; nearly every literal is
already bound to a named constant with documented rationale.
- 5.1 (unnamed literals -> named constants): N/A — every literal is named:
  2147483647u -> kMaxGridX (line 65, comment "2^31 − 1"), 65535u -> kMaxGridY
  /kMaxGridZ (68/73), 32 -> kDecodeBlockX (162), 8 -> kDecodeBlockY (167). The
  `1` in cdiv's `(n + b - 1)` (83/90) is the structural ceiling-division offset,
  not a magic tunable. grid_for/cdiv take `block`/`max_grid` as PARAMS with named
  defaults (kCdivBlock, kMaxGridY), not inlined literals.
- 5.2 (hardcoded sizes/bounds that should be params/derived): kMaxGridX/Y/Z
  (65/68/73) are HARDWARE grid-axis limits (architecture.md §7; CUDA C++ Prog.
  Guide), correctly fixed compile-time constants, NOT tunables; making them
  params would be wrong. kCdivBlock is centrally defined in config.hpp (=16) and
  consumed as the named default (114). No bound that should have been a param is
  inlined.
- 5.3 (duplicated constants / DRIFT): The square-f2 block dim uses kCdivBlock in
  BOTH the block (f2_block_kernel.cu:319/379, f2_blocks_kernel.cu:196/279) and
  grid_for's default (this file 114) — single-sourced from config.hpp:51, no
  drift. The decode kernel uses kDecodeBlockX/Y in BOTH its block dim
  (decode_af_kernel.cu:105) and grid calc (106) — no launch-vs-shared-mem drift.
  Only the kMaxGridY/kMaxGridZ same-value duplication (above) is a (LOW) note.
- 5.4 (hardcoded paths/IDs/device ids): N/A — no filesystem paths, no string IDs,
  no cudaSetDevice / device ordinals; the header is grid/block extent math only.
- 5.5 (ambiguous 32 — warp size vs other): The only 32 is kDecodeBlockX (line
  162), explicitly NAMED and documented as "one full warp" with the
  warp-coalescing rationale (158–161) and contrasted against the half-warp
  kCdivBlock=16; the "32×8"/"32*8" mentions (33/166) refer to that same named
  constant. Not an ambiguous bare 32.
-->

## Group 6 — Naming

- [6.1][LOW] src/core/internal/launch_config.hpp:82,89 — the `cdiv` block-divisor parameter is the single letter `b` (both the int overload line 82 and the long overload line 89), outside a tight loop-counter context. It is documented in the doc-comment ("`b` must be > 0 (a block dimension)") and in a 2-arg ceiling-division it reads acceptably, but `b` is less self-describing than the sibling `block` parameter used in `grid_for` (line 114) for the very same concept, so the two block-dimension params are named inconsistently across the file. Suggested: rename `cdiv`'s `b` to `block` (or `denom`) to match `grid_for`'s `block` and self-document the divisor; LOW because it is doc-commented and conventional for cdiv.

<!--
Reviewed against all 6.1–6.4 tasks with the §12/FP64/scale context.
- 6.1 (cryptic names): The only single-letter identifiers are `cdiv`'s `n`/`b`
  (82/89) — both documented; `n` (count/dimension) is fine, `b` is the LOW note
  above. `extent` (121), `block`/`max_grid` (114), `n_in_group` (143) are clear.
  No `tmp`/`data2`/`arr`/`flag`-style opaque names anywhere.
- 6.2 (misleading names): None. `grid_for` returns a tile/extent count and is so
  documented + `[[nodiscard]] int`; `grid_z_extent` returns the z extent;
  `kMaxGridX/Y/Z` are genuine maxima; `cdiv` is genuinely ceiling division. No
  count-named-index or list-named-map mismatches.
- 6.3 (inconsistent conventions in one file): The file consistently uses
  k-prefixed PascalCase for constants (kMaxGridX/Y/Z 65/68/73, kCdivBlock via
  config, kDecodeBlockX/Y 162/167) and snake_case for functions + params (cdiv,
  grid_for, grid_z_extent, max_grid, n_in_group, extent). That is a coherent,
  single convention (k-constants + snake functions), not nElements/num_elements/n
  drift. The only cross-site inconsistency is the block-divisor naming `b` vs
  `block` flagged under 6.1.
- 6.4 (nonstandard abbreviations): `cdiv` is documented in the banner (lines 7–9,
  79–81) as ceiling division and is an established project term ("architecture.md
  §2 one cdiv()"). `dim`/`grid`/`block`/`pad` (s_pad mentions) are standard CUDA
  vocabulary. No invented/opaque abbreviations.
-->

## Group 7 — Duplication

- [7.3][LOW] src/core/internal/launch_config.hpp:144,148 — `grid_z_extent` computes `static_cast<unsigned>(n_in_group)` twice: once in the assert's upper-bound compare (line 144) and again in the return (line 148). Trivial duplication of one cast. Suggested: compute `const unsigned z = static_cast<unsigned>(n_in_group);` once, assert `n_in_group >= 1 && z <= kMaxGridZ`, then `return z;` — LOW, folds-out and has no correctness impact.

<!--
Reviewed against all 7.1–7.4 tasks with the §12/FP64/scale context. This is a
tiny pure-header (4 fns + 5 named constants), so duplication surface is minimal.
- 7.1 (copy-pasted blocks differing by a constant): Two candidates, both
  intentional / not worth folding:
  * The int (82–84) and long (89–91) `cdiv` overloads share the body
    `return (n + b - 1) / b;` differing only by the integer TYPE. This is a
    deliberately-documented overload pair, NOT copy-paste-by-constant: the header
    contract (lines 11–13, 86–88) REQUIRES the long overload for the SNP/M axis
    (M > 2^31) and the int one for the P axis. A single template would erase the
    type-routing the design depends on and is consciously avoided. NOT flagged.
  * The cap-check asserts in `grid_for` (122–125) and `grid_z_extent` (144–147)
    are structurally similar (`static_cast<unsigned>(x) <= <cap>` + a §7/X-7/B6
    message) but differ in lower bound (extent>=0 vs n_in_group>=1), cap
    (parameter `max_grid` vs the fixed `kMaxGridZ`), return type (int vs unsigned),
    and message. The header EXPLICITLY documents why `grid_z_extent` is a separate
    single-home and not routed through `grid_for` (lines 26–31, 133–142: the z
    axis is set DIRECTLY so a `grid_for` clamp would miss it). Folding the two
    into one helper would re-introduce exactly the coupling the design split apart;
    only ~2 assert lines overlap. NOT flagged.
- 7.2 (repeated/loop-invariant expressions): N/A — no loops; `cdiv(n, block)` is
  evaluated once into `extent` (121) then reused; no expression is recomputed.
- 7.3 (repeated sizeof/casts): the `static_cast<unsigned>(n_in_group)` double-eval
  in `grid_z_extent` (144 + 148) is the one (LOW) note above. The casts at 122
  (`extent`) and 144 (`n_in_group`) are on distinct operands — not duplication.
  No `sizeof` anywhere (no allocation/byte math in this header).
- 7.4 (collapsible boilerplate a macro/helper would fold): N/A — the only
  repeated shape is the cap-check assert, addressed under 7.1; a macro/helper
  there is net-negative per the documented single-home split. The §-divider
  comment banners are documentation, not foldable code.
-->

## Group 8 — Comments

No Group 8 issues found.

<!--
Reviewed against all 8.1–8.4 tasks with the §12/FP64/scale context. This header
is comment-heavy by design (the spec-cited single-home for launch math), but the
comments are rationale/contract documentation, not noise.
- 8.1 (restating code): None. No `i++; // increment i` style. Every comment adds
  information the code does not state: the §4/§8 single-home mandate (banner
  1–40), the hardware cap provenance (49–60, "returns
  cudaErrorInvalidConfiguration"), the `[[maybe_unused]]` justification (116–120),
  the warp-coalescing reason for 32 (158–161). None merely echo the expression.
- 8.2 (stale comments): None — every doc-claim matches the live code:
  * cdiv int (82–84) + long (89–91) overloads exist exactly as the banner (11–13)
    and 86–88 describe; bodies are the documented `(n + b - 1) / b`.
  * grid_for (114–127) defaults `block=kCdivBlock` / `max_grid=kMaxGridY` and
    asserts the extent fits max_grid — matches the doc (97–113) and the
    `[[maybe_unused]]` note (116–120, "consumed ONLY by the STEPPE_ASSERT…compiles
    out under NDEBUG"), which is accurate (the param is read only at line 122).
  * grid_z_extent (143–149) asserts `1 ≤ n ≤ kMaxGridZ` and returns the unsigned
    extent exactly as 26–31 / 133–142 state.
  * kMaxGridX/Y/Z (65/68/73) carry the values their comments claim (2^31−1, 65535,
    65535). The "moved here verbatim out of f2_estimator.hpp … cleanup X-4" banner
    (7–9) is historical provenance, not a behavior claim, so it cannot go stale.
    The call-site references in the banner/notes (e.g. f2 [P × P] / [P × s_pad],
    M4 strided-batched gridDim.z=n_in_group) are corroborated by the Group 3/5
    grep audit above — current, not stale.
- 8.3 (missing rationale): None — rationale coverage is unusually complete. The
  non-obvious constants ALL carry a cited reason: 65535 caps -> §7 + CUDA C++ Prog.
  Guide, verified CUDA 13.x/sm_120 (49–60, 67–73); 32 -> "one full warp …
  coalescing", with the explicit reason it does NOT reuse kCdivBlock=16 (158–161);
  8 -> "32 × 8 = 256 threads, occupancy-friendly" (165–167); the `[[maybe_unused]]`
  workaround -> the -Werror-under-both-build-types rationale (116–120). The
  intentional deviation (square f2 vs non-square decode block) is documented at
  107–113. No bare constant or workaround lacks a why.
- 8.4 (orphan TODO/FIXME/HACK): None — grep for TODO/FIXME/HACK/XXX/WIP/TBD over
  the file returns no matches.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Reviewed against all 9.1–9.3 tasks with the §12/FP64/scale context. This header
is, by design, the spec-cited single-home for launch-config constants, so it is
the SOLUTION to Group 9's anti-patterns, not an instance of them.
- 9.1 (should-be-const/constexpr left mutable): None. Every named value is
  `inline constexpr` (kMaxGridX 65, kMaxGridY 68, kMaxGridZ 73, kDecodeBlockX 162,
  kDecodeBlockY 167); the only local, `extent` (121), is already `const`;
  grid_z_extent's `static_cast<unsigned>(n_in_group)` is computed inline (no
  mutable local at all). All four functions are `constexpr`/`inline` and pure.
  Nothing that should be const is left mutable.
- 9.2 (tangled config — tunable knobs buried in logic): None. This file is the
  surfaced config home itself: the hardware grid caps and the decode block dims
  are named constants at the file top with cited rationale (architecture.md §7,
  ROADMAP §4), and the square-f2 block edge kCdivBlock is single-sourced from
  config.hpp and consumed as a named default (line 114) rather than re-picked in
  kernel logic. No knob is buried inside a function body.
- 9.3 (positional booleans foo(true,false,true)): N/A — no function in the file
  takes a bool parameter. cdiv(int,int) / cdiv(long,long) (82/89), grid_for(int,
  int,unsigned) (114), grid_z_extent(int) (143) are all integer-typed; there is
  no positional-bool flag signature, so no opaque foo(true,false) call shape can
  arise at the call sites.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Reviewed against both 10.1–10.2 tasks with the §12/FP64/scale context. This is a
pure STEPPE_HD constexpr integer-math + named-constant header (4 functions, 5
named constants); no mutable state, no aggregates, no static/global storage.
- 10.1 (late/distant decl; uninitialized-then-assigned): None.
  * cdiv int (82–84) and long (89–91) are single-expression returns — no locals.
  * grid_for (114–127): the only local, `extent` (line 121), is declared AT its
    first use and initialized in the same statement (`const int extent =
    cdiv(n, block);`), then read immediately in the assert (122) and return (126).
    No declare-now/assign-later split, no distant declaration.
  * grid_z_extent (143–149): no local at all; the value is computed inline
    (`static_cast<unsigned>(n_in_group)`) in the assert (144) and return (148).
  No "declared at top, assigned far below" or "declared uninitialized then
  conditionally assigned" pattern exists anywhere in the file.
- 10.2 (zero-init assumptions that do not hold): None. Every named constant has
  an EXPLICIT initializer — kMaxGridX = 2147483647u (65), kMaxGridY = 65535u (68),
  kMaxGridZ = 65535u (73), kDecodeBlockX = 32 (162), kDecodeBlockY = 8 (167); all
  are `inline constexpr`, so they are compile-time-evaluated, never relying on
  implicit/zero init. The single function local (`extent`, 121) is explicitly
  initialized from cdiv. There are no arrays, structs, `= {}` aggregate inits,
  member fields, or static/thread_local objects whose value-initialization could
  be silently assumed — so no missing-init-relying-on-zero hazard.
-->


