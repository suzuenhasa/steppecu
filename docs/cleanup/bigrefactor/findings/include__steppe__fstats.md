# Review findings — include__steppe__fstats

Files: /home/suzunik/steppe/include/steppe/fstats.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (not findings):
- 4.1: `f2`/`vpair` are std::vector<double> (lines 52, 60) — FP64-by-design, correct. N/A.
- 4.2/4.6: The flat index `i + P·j + P·P·b` (lines 36-38, 48, 56) is documentation only; no
  int index arithmetic is computed in this header. The sole computed quantity, size() (lines
  74-77), casts P, P, n_block to std::size_t BEFORE the multiply, so P·P·n_block (~10^10) does
  not overflow int. This is the correct widening pattern; the overflow-prone flat-index math
  lives in consumers (src/device, src/core), not here.
- 4.3: No cudaMalloc/new/raw allocation — std::vector handles `* sizeof(T)`. N/A.
- 4.4/4.5: No loops in this header. N/A.
- 4.7: No raw pointers; intentionally CUDA-free host storage (std::vector). N/A.
- P, n_block as int (lines 68, 71) and block_sizes as std::vector<int> (line 65) fit comfortably
  at scale (P<=2500, n_block<=757); not a finding.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (not findings):
- This is the installed, intentionally CUDA-FREE public header (lines 12-15, "No CUDA header
  here"). It includes only <cstddef> and <vector> (lines 27-28) and defines a plain host-only
  POD struct (F2BlockTensor, lines 47-78). No device code, no CUDA runtime/driver API.
- 2.1 Dropped archs: no build flags, no CMake, no SASS/PTX arch lists in this header. N/A.
- 2.2 Texture/surface references: no texture<...>/surface<...>/cudaBindTexture*; no CUDA at all. N/A.
- 2.3 Non-_sync warp intrinsics: no warp intrinsics / no device code. N/A.
- 2.4 cudaThreadSynchronize: no CUDA runtime sync calls. N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (not findings):
- 3.1: All comments in this header (lines 1-23 file header, 32-46 struct doc, 48-77 member docs)
  are architecture/doc comments, NOT commented-out code "kept just in case". None are dead.
- 3.2: No #if 0, no code after return/break, no unreachable code. The single function size()
  (lines 74-77) is one unconditional return. N/A.
- 3.3: Both includes are used — <cstddef> (line 27) by std::size_t in size() (lines 74-76);
  <vector> (line 28) by std::vector<double>/<int> (lines 52, 60, 65). No unused vars/params/
  helpers; all struct members (f2, vpair, block_sizes, P, n_block) are the M4 public-API
  deliverable. vpair is explicitly RETAINED (lines 54-58) as the S4 jackknife weight, so it is
  not dead even though the diagonal of f2 is documented as never consumed downstream.
- 3.4: Nothing is computed-but-unread in this header; it is a POD struct + one convenience
  accessor. The "computed but unread" diagonal note (lines 44-46) concerns downstream consumers,
  not storage here. N/A.
-->

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Notes (not findings):
- 5.1: The only literals are the default member initializers `P = 0` (line 68) and `n_block = 0`
  (line 71). These are empty/sentinel defaults, not unnamed magic numbers. No 0.001f / 1024 / 0.5
  / threshold-style literals anywhere.
- 5.2: No hardcoded sizes/bounds. P and n_block ARE the parameters; storage lengths (`P·P·n_block`
  for f2/vpair, `n_block` for block_sizes) are derived from them, and size() (lines 74-77) computes
  that derivation rather than hardcoding it.
- 5.3: No executable constants to duplicate. The flat-index layout `i + P·j + P·P·b` (lines 36-38,
  48, 56) is documentation prose only — it is not a literal stamped into code in two places, so
  there is no drift hazard here (the actual indexing math lives in consumers).
- 5.4: No hardcoded paths, IDs, or device ids — intentionally CUDA-free host header.
- 5.5: No `32` literal and no warp-size constant; no ambiguity to disambiguate.
-->

## Group 6 — Naming

No Group 6 issues found.

<!--
Notes (not findings):
- 6.1 Cryptic names: member names `f2`, `vpair`, `block_sizes`, `n_block`, `P` (lines 52, 60, 65,
  68, 71) are all canonical f-statistics / ADMIXTOOLS-2 domain vocabulary — `f2` is the
  f2-statistic, `P` is the standard symbol for population count (architecture.md §11.1/§11.2
  spell the tensor `P × P × n_block`), `n_block` the jackknife block count, `vpair` the
  pairwise-valid SNP count. The single-letter `i`, `j`, `b` (lines 36-38, 48, 54, 56) appear only
  in DOC PROSE as index symbols, not as code identifiers. Not cryptic.
- 6.2 Misleading names: `block_sizes` is genuinely per-block SNP counts (lines 62-65), `vpair`
  genuinely a count (lines 54-60), `n_block` genuinely the block-axis length (line 71), `size()`
  genuinely returns the flat element count (lines 73-77). No count-that-is-an-index / list-that-
  is-a-map mismatch.
- 6.3 Inconsistent conventions: snake_case members (`block_sizes`, `n_block`) are mutually
  consistent. `P` (capital) is the deliberate canonical math symbol for population count, not a
  convention slip; this exactly matches src/device/backend.hpp (`P`, `n_block`, `block_sizes`,
  member `vpair` lowercase / `Vpair` in prose). The lowercase member `vpair` vs capitalized
  `Vpair` in comments (lines 7, 32 prose vs line 60 member) mirrors backend.hpp:66-73 identically
  — consistent across the seam, not drift.
- 6.4 Nonstandard abbreviations: `vpair` (valid-pair count), `n_block`, `P`, `f2` are all standard
  in this domain and in the backend seam (backend.hpp). No nonstandard abbreviations.
-->

## Group 7 — Duplication

No Group 7 issues found.

<!--
Notes (not findings):
- 7.1 Copy-pasted blocks: no duplicated CODE. `f2` (line 52) and `vpair` (line 60) are two data
  members with parallel doc comments, but a member declaration is not a copy-pasted code block and
  is the canonical way to carry the two co-resident tensors (file header lines 18-23). Nothing to
  extract into a fn/template.
- 7.2 Repeated loop-invariant expressions: the only computed expression is size() (lines 74-77).
  The subexpression `static_cast<std::size_t>(P)` does appear twice (lines 75-76) within that one
  return, but it is a single compiler-folded line with exactly one call site and is the correct
  widen-before-multiply idiom (P·P·n_block ~10^10 must not overflow int). Introducing a temp/helper
  for a one-line, single-call-site expression would add boilerplate, not remove it. Not worth a
  finding.
- 7.3 Repeated sizeof/casts: three static_cast<std::size_t>(...) in the size() expression
  (lines 75-76); no sizeof anywhere (std::vector owns its sizing). These casts are all in ONE
  expression at ONE site — there is no second site to drift from, so nothing to hoist/template.
- 7.4 Collapsible boilerplate: nothing a macro/helper would fold. This is a single POD struct
  (lines 47-78) plus one trivial accessor (lines 74-77). No repeated guard/declaration patterns.
-->

## Group 8 — Comments

No Group 8 issues found.

<!--
Notes (not findings):
- 8.1 Restating code: every comment (file header 1-23, struct doc 32-46, member docs 48-77) is
  non-obvious design/architecture rationale (why include/ not src/core, FP64-always storage, the
  flat-index layout, the retained-vpair reasoning, the diagonal convention), NOT trivial restating
  like `i++; // increment i`. No mechanical-narration comments.
- 8.2 Stale comments: all member docs match the actual declarations (f2 line 52, vpair line 60,
  block_sizes line 65, P line 68, n_block line 71, size() lines 74-77 — all present and accurate).
  Verified the two cross-reference factual claims against current source: (a) line 23 "device budget
  helper reserves for both" matches src/device/vram_budget.hpp:66-67 (`2u * p * p * nb * sizeof(double)`,
  the B26 2× term); (b) lines 42-46 diagonal convention "NOT a forced 0 / -2·mean within-pop het
  correction (cleanup X-2/B4)" matches src/device/backend.hpp:54-62 verbatim. Neither is stale.
- 8.3 Missing rationale: the non-obvious choices are all justified — why include/ not src/core
  (lines 10-15), FP64 in every precision mode (lines 17-23), vpair RETAINED as the S4 jackknife
  weight + COMPOSE-not-double-normalize caveat (lines 54-58), the diagonal convention (lines 42-46).
  Sentinel defaults P=0/n_block=0 (lines 68,71) are self-evident. Rationale present where needed.
- 8.4 Orphan TODO/FIXME/HACK: none in the file (grep -E 'TODO|FIXME|HACK|XXX' returns no match).
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Notes (not findings):
- 9.1 Should-be-const/constexpr left mutable: The five members (f2 line 52, vpair line 60,
  block_sizes line 65, P line 68, n_block line 71) are INTENTIONALLY mutable — F2BlockTensor is
  the M4 data-carrier deliverable, populated by the S0-S2 precompute engine and consumed/contracted
  by the S3-S8 fit engine across the include/ seam (file header lines 10-15). Making them
  const/constexpr would make the struct unfillable. The sole method size() (lines 74-77) IS
  correctly const noexcept. No mutable-that-should-be-const candidate.
- 9.2 Tangled config: No tunable knobs exist in this header. There are no thresholds, buffer sizes,
  block counts, or precision flags buried in logic — P and n_block (lines 68, 71) are runtime DATA
  dimensions of the deliverable, not configuration. The Precision knob is explicitly an operation
  mode that lives elsewhere and is documented as NOT a storage type here (lines 17-23). Nothing to
  surface at a file top / config struct.
- 9.3 Positional booleans: No function takes a bool. The only callable is the zero-argument accessor
  size() (lines 74-77); there are no foo(true,false,true)-style call sites in this header. N/A.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Notes (not findings):
- 10.1 Late/distant declaration or uninitialized-then-assigned: This is a POD struct (F2BlockTensor,
  lines 47-78), not a function body — there are no local variables, no scopes, and no
  declare-then-assign-later patterns to misorder. The only function, size() (lines 74-77), is a
  single unconditional return with no locals. N/A.
- 10.2 Zero-init assumptions that do not hold: Both scalar members carry EXPLICIT default member
  initializers — P = 0 (line 68) and n_block = 0 (line 71) — so they do NOT rely on implicit/
  value-init holding. A default-constructed or aggregate-initialized F2BlockTensor therefore has
  P and n_block deterministically 0 (no garbage-dimension hazard), and size() (lines 74-77) yields
  a well-defined 0 on such an instance. The container members f2/vpair (lines 52, 60) and
  block_sizes (line 65) are std::vector, which default-constructs to a defined empty state
  independent of any zero-init assumption. No missing-init relying on unguaranteed zero-init.
-->

