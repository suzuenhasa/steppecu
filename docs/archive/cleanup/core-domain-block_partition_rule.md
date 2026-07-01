# Review — `core/domain/block_partition_rule` (`block_partition_rule.hpp` + `.cpp`)

> **SUPERSEDED IN PART (AT2-walk reconciliation, `docs/research/block-partition-at2.md`).**
> This review was written against the ORIGINAL fixed-grid floor-bin `assign_blocks`
> (`block_of(genpos)=floor(genpos/blgsize)`, cut on every grid-line crossing). That
> loop has since been replaced by the AT2 `setblocks` SNP-ANCHORED cumulative walk
> (a new block opens on a chromosome change OR when the distance from the block's
> FIRST SNP reaches blgsize, `>=` inclusive; the anchor re-sets to the opening SNP).
> Consequences for this doc: (1) the `assign_blocks` loop no longer calls `block_of`
> or casts a floored quotient to `int`, so the float→int-UB analysis below applies
> only to the surviving `block_of` PRIMITIVE, not to the partition loop; the loop's
> width guard now prevents a silent over-partition / whole-chrom-merge, not UB.
> (2) The standing internal-consistency counts are now the WALK counts: full v66
> (chr 1–24) = **748** (was 757), chr 1–23 = **747** (was 756), autosome chr 1–22 =
> **711** (was 719); the AT2 cache parity target on the Haak polymorphic union is
> **709**. The structural/layering findings (length seam, `block_ranges` contract,
> single-home) are unchanged and still apply.

Adversarial second pass. Unit under review: the single-source-of-truth SNP→block
assignment rule (M3). Three entry points: the inline per-SNP `block_of`, the inline
`block_size_cm_to_morgans` (the one cM↔Morgan site), and the out-of-line
whole-ordering `assign_blocks` producing `BlockPartition{block_id[], n_block}`.

Files read in full this pass: `src/core/domain/block_partition_rule.hpp` (132 lines),
`src/core/domain/block_partition_rule.cpp` (60 lines). Context re-read in full to
re-verify every cross-layer claim: `include/steppe/config.hpp`,
`src/core/internal/views.hpp`, `tests/unit/test_block_partition.cpp`,
`src/device/backend.hpp` (the `compute_f2_blocks` seam),
`src/core/fstats/f2_from_blocks.{hpp,cpp}` (the orchestration that erases
`block_id`'s length to a raw pointer), `src/device/cuda/cuda_backend.cu:132-190` and
`src/device/cpu/cpu_backend.cpp:207-226` (the two block-layout scans that index
`block_offsets[b]`/`block_sizes[b]` with `b = block_id[s]`),
`src/device/cuda/f2_blocks_kernel.cu:67-110` (the **device-side** `block_offsets[id]`/
`block_sizes[id]` consumer), `src/io/snp_reader.{hpp,cpp}` (where `genpos` enters
unvalidated), and the include/symbol graph (grep). Standards judged against
architecture.md §2, §4, §5/S2, §8 (single-home table — `block_of` listed by name,
line 528), §9, §11.4, §12 ("Seed control" names this file, line 741), §13; ROADMAP
§3 (M3 done, line 69), §4 (line 99 names `block size 0.05 Morgans → RunConfig::block_size_cm`),
§5, §6 (line 131 — compute-sanitizer + warnings-as-errors gate); TODO.md capability-tier
section (workflow `wxz1fiiln`).

**External facts verified against primary sources this pass (not asserted from memory):**
- **Float→int out-of-range conversion (incl. NaN/±Inf) is undefined behavior.** C++
  `[conv.fpint]`/7.3.11: "The conversion truncates … The behavior is undefined if the
  truncated value cannot be represented in the destination type." Confirmed via
  [eel.is/c++draft/conv.fpint](https://eel.is/c++draft/conv.fpint) (fetched this pass).
- **`std::floor` preserves specials:** `floor(NaN)=NaN`, `floor(±∞)=±∞`, `floor(−0.0)=−0.0`
  (C++ `<cmath>` delegating to C Annex F / IEC 60559). Confirmed via
  [cppreference floor](https://en.cppreference.com/w/cpp/numeric/math/floor) and P0533R9.
- **`std::floor` is `constexpr` only since C++23** (P0533R9, "constexpr for `<cmath>`").
  This project is C++20 (`CMakeLists.txt:26 CMAKE_CXX_STANDARD 20`), so `block_of`
  *cannot* be `constexpr` — confirms 7.7 is a non-issue, not an oversight.
- **`std::span::size()` returns `size_type` (`std::size_t`)** (`[span.obs]`), so the
  `size_t → long` cast at cpp:23 is the one width transition to scrutinize (2.2).
- **`std::istream operator>> double` routes through `num_get`→`strtod`,** which parses
  `inf`/`nan` tokens to non-finite doubles (only out-of-magnitude tokens like `1e400`
  set `failbit`). Confirmed via [cppreference num_get](https://en.cppreference.com/w/cpp/locale/num_get.html).
  This is the live path by which a non-finite `genpos` can reach `block_of` (1.2b).

## Role & layering

This is the one **shared domain rule** architecture.md elevates to a first-class DRY
concern: §2 (line 55) names "one `block_assignment()` rule" as "the single most important
DRY consequence in this codebase"; §4 (line 192) carves out `block_partition_rule.hpp` as
*the* explicit exception both `io` consumers and `device` kernels read; §8's single-home
table (line 528) lists `int block_of(genetic_pos, blgsize)` by name. **Verified this pass:**
the header includes only `<cmath>`, `<span>`, `<vector>`, and the public `steppe/config.hpp`
(for `kCentimorgansPerMorgan`); the `.cpp` adds only `<cstddef>`. No CUDA header, no `io`
header, no upward dependency. The header **is** `#include`d into `.cu` translation units
(`tests/reference/test_f2_blocks_equivalence.cu:55`, `test_filter_oracle.cu:48`) and compiled
by nvcc as host code, cleanly — so the host-pure claim is exercised, not aspirational. Grep
confirms the single-source invariant: exactly one `assign_blocks`, one `block_of`, one
`block_size_cm_to_morgans`, and **no bare `* 0.01` / `/ 100` anywhere in `src`/`include`**
outside this unit's own doc comment. Layering and single-sourcing: **10/10**. Deductions
below are correctness/robustness and a small set of contract/portability gaps, almost all
in the inline math primitives, not in the (clean) `assign_blocks` loop.

**Two framing corrections to the first pass, both re-verified by grep this pass:**
1. **`RunConfig::build()` / `ConfigBuilder` do not exist in the tree.** A grep of
   `src`+`include` finds no `class ConfigBuilder`, no `struct/class RunConfig`, no
   `block_size_cm()` accessor — `RunConfig` appears *only* in this unit's three doc-comment
   mentions and one `error.hpp` comment. So the only enforceable validation today lives in
   *this* unit: the in-unit guard is the **primary** defense, not the fallback. Keeps 1.1
   HIGH and actionable now (not "wait for the builder").
2. **`core/internal/log.hpp` does not exist either.** The first pass's 1.4 fix proposed a
   `STEPPE_LOG_WARN` include "acceptable, host-pure §8 plumbing." Verified: there is **no**
   `log.hpp` under `src/core` and no `STEPPE_LOG_*` symbol in `src/core` today (it is named
   in architecture §870 as planned, not built). So the **implementable-now** half of 1.4 is
   the debug `assert`; the logged-warning half is deferred until that facility lands. This
   downgrades the actionability of the log portion (noted inline).

## Score: 8.5/10 — solid, idiomatic, correctly-layered domain code held off a 9.5 by latent-UB edges in the math primitives (`block_size ≤ 0`, NaN/huge `genpos`, the `int` narrowing of `floor`) that the prose contracts *describe* but the code does not *enforce* (§2 fail-fast), plus one genuinely load-bearing length/contiguity seam (`block_id.size()` vs `MatView::M`, and the dense/non-decreasing shape) that is erased to a raw `const int*` and unchecked at the boundary the device dereferences.

The whole-ordering pass is genuinely good: a single deterministic file-order loop, no
allocation beyond the result, correct dense-renumber + per-chromosome-reset semantics,
defensive length handling, and a real unit test pinning every documented property (chrom
reset, all-zero chromosome, empty input — and, post-reconciliation, the SNP-anchored cut +
remainder-roll-forward) plus a real-AADR internal-consistency check (748 blocks under the
AT2 walk; was 757). It is *not* a trivial header — it carries domain
semantics three layers depend on bit-for-bit, and a device kernel dereferences its derived
arrays (`f2_blocks_kernel.cu:88,96` index `block_sizes[id]`/`block_offsets[id]` *on-device*),
so the bar is the full 9.5–10 and the review is correspondingly detailed. What keeps it off
a 9.5: the two math primitives have UB edges that §2 fail-fast requires the code to *enforce*,
not merely document; and the cross-layer shape contract (parallel-array length, non-decreasing/
dense ids) lives only in prose at the exact seam where it is erased to a raw `const int*` and
an independently-passed `MatView::M`.

---

## Findings

### (1) Correctness & bugs

**1.1 — `block_size_morgans <= 0` (or NaN) produces UB or silent-wrong-answer, not the documented "must be > 0" failure (HIGH). CONFIRMED; fix re-targeted to this unit.**
`block_of` (hpp:56-58) computes `std::floor(genpos_morgans / block_size_morgans)` then
`static_cast<int>(...)`. The doc comment (hpp:49, hpp:121) says "Must be > 0," but nothing
enforces it. With `block_size_morgans == 0.0`, IEEE-754 division gives `+Inf` (positive
genpos), `-Inf` (negative), or `NaN` (0/0); `std::floor` propagates these unchanged (verified
above); then `static_cast<int>(±Inf)` / `static_cast<int>(NaN)` is **undefined behavior**
([conv.fpint], verified above). A *negative* `block_size_morgans` is subtler and worse: it
silently *inverts the bin order* (floor of a negative quotient), so `assign_blocks` still
produces a dense, non-decreasing-*looking* partition — **wrong block assignment with no
error**, defeating the point of a single-source rule that must match AT2. `block_size_cm_to_morgans`
is `constexpr` but validates nothing, and the test even asserts `block_size_cm_to_morgans(0.0)
== 0.0` (test:55) — that zero flows straight into `block_of` as a divisor with no downstream
check.
*Why it matters:* §2 mandates fail-fast — "surface immediately … not as silent corruption";
ROADMAP §6 DoD (line 131) requires compute-sanitizer clean, and `SteppeOptions.cmake:42-46`
makes `ubsan` a selectable sanitizer set that would flag the float→int cast. The "honest
caller" prose defense is exactly the unchecked cross-layer precondition §2 forbids.
*Fix (PRIMARY — the builder does not exist; verified by grep this pass):* in `assign_blocks`,
before the loop, `if (!(block_size_morgans > 0.0)) return out;` — the `!(x > 0)` form rejects
`0.0`, negatives, **and NaN** in one test. Document `block_of`'s precondition as "caller
guarantees `block_size_morgans > 0` (enforced by `assign_blocks`; and by `ConfigBuilder::build()`
if/when it lands)." Add the matching test pinning the post-fix empty-return. Severity **high**,
effort **S**, before-M4.5? **yes** (M4.5 shards inputs across GPUs; a wrong/UB rule corrupts
every device partial).

**1.2 — NaN / huge / non-finite `genpos_morgans` is UB at the `int` cast, and the cast is the genuine narrowing risk (MED). CONFIRMED and sharpened.**
Even with a valid `block_size_morgans`, `block_of` casts the floored quotient to `int`. Two
distinct triggers, neither caught by 1.1's `bs > 0` guard:
  (a) **Out-of-`int`-range finite genpos.** `block_of` returns the *local* bin
  `floor(genpos/bs)`, unbounded in the supplied `genpos`. The hpp:84-88 comment justifies
  `int` because "block counts are O(1e3)" — true for the *occupied global* count in
  `assign_blocks`, but **not** for `block_of`'s raw local-bin return. At `bs = 0.05` any
  `genpos > ~1.07e8` Morgans overflows `INT_MAX`, making the cast UB ([conv.fpint]).
  (b) **NaN / ±Inf genpos.** Verified at the source: `snp_reader.cpp:62,65,75` reads genpos
  via `ls >> genpos` into a `double` (default `0.0` only on a *parse failure*, not on a parsed
  `inf`/`nan` token). Confirmed via the standard `num_get`→`strtod` path: an `inf` or `nan`
  token in the .snp genpos column parses *successfully* to a non-finite double and reaches
  `block_of` directly; `static_cast<int>(NaN/±Inf)` is UB.
*Why it matters:* §2 fail-fast on degenerate input; the unit's own `int`-is-safe comment proves
it only for `assign_blocks`'s *output*. Edge-case category, ROADMAP DoD / UBSan.
*Fix:* the realistic mitigation for (b) is upstream — `io` (and the future `build()`)
validating finite, plausibly-sized Morgan ranges. Within this unit: **widen the local bin to
`long`** — `block_of` could return `long`, and `prev_local_bin` become `long`; the local bin is
only ever compared for equality and stored, never narrowed-for-storage, so widening removes the
narrowing risk for finite genpos *entirely* (only the *global* id need stay `int`). Document the
precondition "`genpos_morgans` finite; `|genpos/bs|` within the local-bin integer range." (Note:
widening to `long` closes the finite-overflow path on LP64 but does **not** close the NaN/±Inf
path — that is still upstream's job; 1.1's `!(bs>0)` form is a model for a finiteness guard if
one is ever added here.) Severity **med**, effort **S**, before-M4.5? no.

**1.3 — `n_block`/`block_id` `int` narrowing is bounded in practice but asserted-by-comment, not by code (LOW). CONFIRMED.**
`global` is `long` (cpp:35), `static_cast<int>`-ed at cpp:51 (each id) and cpp:56 (`n_block`).
The dense count is bounded by `m` (≤ ~584k today; O(1e3) occupied), so `int` cannot overflow on
real data. The cast is unchecked, but `BlockPartition::block_id` being `std::vector<int>` is a
deliberate tree-wide contract (`backend.hpp:167` takes `const int* block_id` / `int n_block`;
both backends carry `n_block` as `int`) — a conscious tradeoff, not a bug.
*Fix:* none required; optionally `assert(global < INT_MAX)` in debug, or reword the hpp:84-85
comment from the unprovable "block counts are O(1e3)" to the provable "`n_block ≤ M ≤` genome
SNP count, so it fits `int`." Severity **low**, effort **S**, before-M4.5? no.

**1.4 — Mismatched-length spans are silently truncated, and the result's length is then erased to a raw pointer + decoupled `M` at the seam the device reads (MED — the most consequential contract gap). CONFIRMED, sharpened, and the in-practice likelihood re-assessed.**
cpp:23-24 takes `m = min(chrom.size(), genpos.size())`; the comment calls a mismatch "a
programming error upstream" but chooses to "be defensive … rather than reading out of bounds."
OOB-safety of the loop itself: good. But two compounding silent failures remain:
  (a) **Silent SNP loss inside `assign_blocks`.** A caller passing N chroms and N−1 genpos
  gets a partition over N−1 SNPs with no diagnostic — the last SNP vanishes from the jackknife.
  §2 fail-fast targets exactly this. *Re-assessed likelihood:* in the current callers this is
  unlikely — `io::read_snp` pushes `chrom` and `genpos_morgans` together (`snp_reader.cpp:74-75`),
  and `SnpTable` documents all vectors share `count` (`snp_reader.hpp:37`), so the two spans are
  producer-guaranteed equal-length. The risk is a *future* caller (filter survivors, merge) that
  builds the two arrays independently. Still a silent-loss smell worth an assert.
  (b) **The length is erased and decoupled from `M` — the live seam.** Verified at
  `f2_from_blocks.cpp:50-51`: `compute_f2_blocks` passes `partition.block_id.data()` (a raw
  `const int*`, length gone) while the SNP count travels *separately* as `Q.M` in a `MatView`.
  The consumers scan `block_id[s]` for `s ∈ [0, M)` (`cuda_backend.cu:138-145`,
  `cpu_backend.cpp:215-224`) trusting `Q.M`, **not** `block_id.size()`. `partition.block_id.size()`
  derives from the .snp/SNP-count path; `Q.M` derives from the decoded tile (`tile.n_snp`,
  `backend.hpp:71`/`cuda_backend.cu:120`). **Nothing ties the two.** If `block_id.size() < Q.M`
  (any skew between the partition's SNP count and the decode's), the consumer reads
  `block_id[s]` past the vector end — a host OOB read — and the resulting garbage `id` then
  indexes `block_offsets[id]`/`block_sizes[id]` in the host scan **and `block_offsets[id]`/
  `block_sizes[id]` on the device** (`f2_blocks_kernel.cu:88,96`), turning a host contract slip
  into a device OOB.
*Why it matters:* §2 fail-fast + §4 the `core`→`device` data-struct ABI; ROADMAP §6
compute-sanitizer catches the device OOB only if the skew is hit at test time.
*Fix:* (i) in `assign_blocks`, `assert(chrom.size() == genpos_morgans.size())` in debug, keep the
min-extent guard in release — the `assert` is implementable today; the `STEPPE_LOG_WARN`-on-skew
the first pass proposed is **deferred** (no `core/internal/log.hpp` exists yet — verified). (ii)
The high-value fix lives in the orchestration: assert `partition.block_id.size() ==
static_cast<size_t>(Q.M)` in `f2_from_blocks.cpp` before the `compute_f2_blocks` call (that fix
is in `f2_from_blocks.cpp`, not here, but the contract originates here and the header should state
it). Severity **med**, effort **S**, before-M4.5? yes.

**1.5 — The load-bearing `(s == 0)` first-block trigger is untested in isolation (LOW, testability). CONFIRMED; first-pass's claimed test-path OOB CORRECTED.**
cpp:33-34 init `prev_chrom = 0`, `prev_local_bin = 0`, `global = -1`; the `(s == 0)` term in
`open_new` (cpp:46) guarantees the first SNP always opens block 0 *regardless* of the sentinels,
so the sentinel values are inert. No existing test isolates this: every case's first SNP also
differs from the sentinel in `chrom` or `bin` (e.g. `test_all_zero_chrom_one_block` uses chrom 7
≠ 0; `test_single_chrom_one_bin` uses chrom 1 ≠ 0), so the `(c != prev_chrom)` term *also* fires
on SNP 0 and masks whether `(s == 0)` is necessary. A future refactor dropping `(s == 0)` on the
assumption "the sentinel mismatch covers it" would break a dataset whose first SNP is chrom 0 /
bin 0.
*Correction to the first pass, re-verified at test:61-71:* the first pass asserted such a
regression would cause "OOB in the `seen` vector of `dense_and_nondecreasing`." That is **wrong**:
`dense_and_nondecreasing` (test:67) checks `if (id < 0 || id >= bp.n_block) return false` *before*
indexing `seen[id]`, so an all-`-1` `block_id` returns `false` (a clean test failure), not an OOB.
The *consumer* OOB claim is real (an `id == -1` would index `block_offsets[-1]`), but the test
would catch the regression first. Net: a valid testability gap; its severity is missing-test, not
a live OOB.
*Fix:* add a case whose first SNP is chrom 0 / bin 0 so `(s == 0)` is the *only* opener; or rename
`prev_chrom`/`prev_local_bin` to obviously-invalid sentinels with a comment that `(s == 0)` is the
true trigger. Severity **low**, effort **S**, before-M4.5? no.

**1.6 — `block_of`'s `@return` doc says "zero-based block index (>= 0 for non-negative position)" but it returns negative bins (LOW, doc-vs-behavior). CONFIRMED.**
hpp:50-51 documents the return as "zero-based block index (>= 0 for non-negative position)." The
function deliberately returns `-1` (and lower) for negative genpos — a documented, *tested*
feature (`test_negative_position_own_block`, real chr17). The "zero-based" phrasing is technically
guarded by "for non-negative position," but a reader skimming the `@return` line gets "zero-based
block index" and may write a downstream `block_of(...) >= 0` assumption. The header's narrative
(hpp:107-108) and the `.cpp` comment (cpp:44-45) get this right; only the terse `@return`
undersells it.
*Fix:* one-line `@return` clarification: "block index = `floor(genpos/bs)`; negative for negative
positions (a distinct bin, by design — see `assign_blocks`)." Severity **low**, effort **S**,
before-M4.5? no.

### (2) Edge cases & failure modes

**2.1 — Zero/negative/NaN `block_size_morgans`:** covered in 1.1 (HIGH). CONFIRMED. The empty-input
path (cpp:25-27) is correct and tested. The `m <= 0` guard on `long m` makes a size-0 span return
the empty `out`; `<= 0` (vs `== 0`) is harmless — `m` is a `min` of two non-negative `size_t`-derived
values cast to `long`.

**2.2 — Very large M (`size_t → long` cast at cpp:23):** `std::span::size()` returns `size_type` =
`std::size_t` (verified). On LP64 Linux (this project's platform) `long` is 64-bit, so the cast is
value-preserving for any realistic M; a span with `size() > LONG_MAX` (~9.2e18, impossible) would
make `m` negative and hit the `m <= 0` early-return — accidentally safe, not deliberately. `s`/`m`
are `long`, matching `MatView::M` (views.hpp:60) and the consumers' `long M` (cuda_backend.cu:120,
cpu_backend.cpp:197). Correct width choice; no action.

**2.3 — Negative genetic positions (real AADR chr17):** CONFIRMED strength. `floor(-0.000258/0.05) =
-1`; negatives form their own local bin; `test_negative_position_own_block` pins it on the real chr17
layout. Code and test match real-data behavior (ROADMAP M3).

**2.4 — Denormal / tiny-positive `block_size_morgans`:** a tiny but positive `bs` (e.g. 1e-300,
`kAbsFloor`-scale) makes `genpos / bs` overflow to `±Inf` → `floor` = `±Inf` → `static_cast<int>`
UB — same family as 1.1 but **not caught by the `bs > 0` guard**. Worth folding into 1.1's
precondition wording ("a *normal*, plausibly-sized positive Morgan width"); a `block_of` cast guard,
if ever added, should also reject non-finite quotients. Severity low; folded into 1.1/1.2.

**2.5 — `+0.0` vs `-0.0` genpos:** `floor(-0.0) = -0.0` (verified), `-0.0 / bs = -0.0`,
`static_cast<int>(-0.0) = 0` — a `-0.0` genpos lands in bin 0 identically to `+0.0`. Correct;
checked, no action.

**2.6 — Equality of bins vs chrom code:** `open_new` compares `local != prev_local_bin` and
`c != prev_chrom` as **integer** equality — no float comparison of positions in the *assignment*
(the only float op is inside `block_of`'s floor). No tolerance/ULP issue in the partition logic.
CONFIRMED strength.

**2.7 — Interleaved / unsorted chromosomes produce more blocks, never aliasing, never OOB (NEW, edge-case strength). CONFIRMED.**
Re-checked the loop against a deliberately interleaved chrom stream (e.g. `1,2,1`): each chromosome
*change* opens a fresh global block (`c != prev_chrom`), so an out-of-order chromosome cannot merge
into a previous chromosome's block — it simply yields more (smaller) blocks. `global` is a monotone
counter, so `block_id` stays non-decreasing and dense regardless of input order, and each global id
occupies exactly one contiguous run (the property the consumer scan at cuda_backend.cu:142 /
cpu_backend.cpp:219 relies on). The rule is correctly *order-as-given*, not order-requiring — and it
cannot silently corrupt on unsorted input, only produce a (correct-but-fragmented) partition. No
action; this is the intended, robust behavior (sortedness is `io`'s concern, §5).

### (3) Numerical / precision vs §12

**3.1 — The division `genpos / block_size_morgans` is the only float op; order-free, deterministic (OK by design). CONFIRMED.**
§12 "Seed control" (line 741) names this file: "SNP→block assignment is the deterministic pure
function of genetic position from `core/domain/block_partition_rule.hpp`." A single FP division +
`floor` per SNP — no accumulation, no reduction, no cancellation — so the Ozaki-vs-native-FP64
discussion (§12) is **N/A** here (no matmul). The block id is a pure function of `(chrom[s],
genpos[s], bs)` and file order, identical on host and device by construction (host-only code
consumed as data by both). Determinism: clean.

**3.2 — Floating-point bin-boundary ties are deterministic but undocumented and untested (LOW). CONFIRMED; first pass's example adversarially corrected.**
A genpos exactly on a bin boundary depends on whether `genpos/bs` rounds just-below or exactly to
the integer in FP64. `5.0/100.0 == 0.05` exactly (the test pins this), but `0.05` is *not* exactly
representable in binary FP64 (≈0.05000000000000000277…). The first pass claimed `0.10 / 0.05` "may
yield 1.9999… → bin 1 instead of bin 2." Re-checked adversarially: with the actual stored values,
`0.10` rounds to ≈0.1000000000000000055… and `0.05` to ≈0.0500000000000000028…; the IEEE-754
round-to-nearest quotient of those is `2.0` exactly, so `block_of(0.10, 0.05) == 2`, not 1. The
*general point* stands (FP floor at a boundary is not the exact-arithmetic answer for some genpos),
but the specific `0.10/0.05` example is not a counterexample.
*Why it matters:* §12 parity is exact-to-AT2; a boundary SNP AT2 floors differently is a 1-SNP
block-membership mismatch. This unit cannot fix it alone (it must *match* AT2); the behavior should
be a named test pinning the current FP floor plus an M7 golden line item.
*Fix:* add a `block_of` test on a genpos whose exact-arithmetic quotient is integral (to pin the FP
floor result), and flag "verify the floor-tie convention against the AT2 `blgsize` golden" in the M7
parity checklist. Severity **low**, effort **S**, before-M4.5? no.

### (4) CUDA idioms / RAII / stream & async / launch config / occupancy vs §7

**N/A — host-pure, CUDA-free by design (architecture.md §4). CONFIRMED.** No allocation beyond
`out.block_id` (one `resize`, cpp:29 — RAII via the std library, no manual `new`/`delete`), no
stream, no kernel, no handle. The one adjacent observation: the device *does* dereference this
unit's *derived* output — `cuda_backend.cu:136-190` builds `block_offsets`/`block_sizes` from
`block_id` and copies them to the device (correctly RAII: `DeviceBuffer<long>`/`DeviceBuffer<int>`
+ `cudaMemcpyAsync` + `STEPPE_CUDA_CHECK`), and `f2_blocks_kernel.cu:88,96` reads
`block_sizes[id]`/`block_offsets[id]` *on-device*. That consumer relies on this unit's
non-decreasing/dense guarantee; see 9.1 for the seam.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**5.1 — No surviving magic numbers (STRENGTH). VERIFIED.** ROADMAP §4 (line 99) demands "no literal
may survive except true mathematical constants." `block_size_cm_to_morgans` divides by the named
`kCentimorgansPerMorgan` (hpp:72); the default lives in `kDefaultBlockSizeCm` (config.hpp:94) and is
converted at this single site; `assign_blocks` contains only the structural literals `-1` (the
pre-increment sentinel, cpp:35), `+1` (`max+1 == n_block`, cpp:56), and `0` (the `s == 0` first-SNP
test). All true structural constants. Grep confirms **no bare `* 0.01` / `/ 100`** anywhere in
`src`/`include` outside this unit's own doc comment.

**5.2 — `kCentimorgansPerMorgan` lives in `config.hpp`, included for that one symbol (STRENGTH).**
hpp:33 includes `steppe/config.hpp` with a trailing comment naming the symbol. Right dependency
direction (`core` → public config) and IWYU-clean. No action.

### (6) Decomposition / single-responsibility / function size vs §2

**6.1 — Excellent decomposition (STRENGTH). CONFIRMED.** Three functions, one responsibility each:
`block_of` (per-SNP local bin), `block_size_cm_to_morgans` (unit conversion), `assign_blocks` (the
whole-ordering pass). `assign_blocks` is ~25 substantive lines, a single loop, no nesting beyond it.
The two cheap primitives are header-inline (so `block_of` hot-path-inlines and the conversion is
`constexpr`); the stateful loop is the `.cpp` — exactly the rationale in the file headers, matching
§8's "`block_of` … host-pure" home. No over-decomposition, no god-function. The `src/core/CMakeLists.txt:40`
comment confirms the inline-primitive / out-of-line-loop split is a deliberate build decision.

### (7) Readability, naming, const-correctness, `[[nodiscard]]`/`noexcept`, comment density

**7.1 — Header narrative slightly oversells "same arithmetic on host and device"; no device-side caller exists, but the header IS compiled by nvcc as host code (LOW — downgraded from the first pass's MED, reasoning corrected and re-verified).**
architecture.md §8 (line 528) reads "`block_of` … host-pure, consumed by `io` *and* device kernels
(§5)" and §2 says the rule is depended on "bit-for-bit." **Verified, with two corrections:** (a) the
header *is* `#include`d into two `.cu` TUs and compiled by nvcc — and works, because nvcc routes
plain (non-`__device__`) host code to the host compiler, so the host-side `assign_blocks`/`block_of`
calls in those TUs compile fine; (b) no `__global__`/`__device__` code calls `block_of` anywhere
(grep this pass: the only device consumption is of the *result arrays* `block_id`/`block_offsets`/
`block_sizes`), which is the better design (compute the partition once on the host, the device
consumes data). So §8's "consumed by device kernels" is satisfied by data flow, and there is no live
compile/UB problem — only a doc-clarity gap: the header comment "Same arithmetic on host and device"
(hpp:42-43) reads as if a device-callable primitive exists, when it does not.
*Fix:* a comment clarifying "device kernels consume the *result* (`block_id[]`/derived offsets), not
this function; no device-side caller exists — if one is ever added, qualify with a `__host__ __device__`
macro and keep it CUDA-free." Do **not** add a `STEPPE_HD` macro speculatively (no consumer needs it;
§2 "optimize/abstract only what need proves"). Severity **low**, effort **S**, before-M4.5? no.

**7.2 — `const`-correctness inside `assign_blocks` (OK). CONFIRMED.** `c`, `local`, `open_new`, `m`
are all `const` (cpp:38,39,46,23); `prev_chrom`/`prev_local_bin`/`global` are necessarily
loop-carried mutable. Clean.

**7.3 — `assign_blocks`'s throwing behavior is undocumented (LOW). CONFIRMED.** `assign_blocks`
allocates (`out.block_id.resize`, cpp:29) so it can throw `std::bad_alloc` and is correctly *not*
`noexcept`; both inline primitives *are* `noexcept` (correct, pure arithmetic). The header `@return`
doc omits the throw.
*Fix:* one line: "Throws `std::bad_alloc` if the result vector cannot be allocated; otherwise
non-throwing." Severity **low**, effort **S**, before-M4.5? no.

**7.4 — `[[nodiscard]]` coverage complete (STRENGTH).** All three functions are `[[nodiscard]]`
(hpp:56,71,125); discarding any return is always a bug. Good.

**7.5 — Comment density high, matches `core` style (STRENGTH, one nit). CONFIRMED.** File-level and
per-function docs cite the exact arch/ROADMAP sections, matching `views.hpp`/`config.hpp`. The `.cpp`
loop comment (cpp:41-45) explains the three open-new triggers and their consequences. Nit: the
hpp:84-85 "block counts are O(1e3)" comment is an assertion the code doesn't enforce (1.3) — reword
to the provable `n_block ≤ M`.

**7.6 — Unit-explicit naming (STRENGTH).** `genpos_morgans` / `block_size_morgans` / the `_cm` suffix
make the unit explicit at every call site — directly serving the "do not conflate cM and Morgans"
warning (hpp:18). The right way to prevent unit bugs.

**7.7 — `block_of` correctly NOT `constexpr` under C++20 (NEW, informational, no action). VERIFIED.**
`block_size_cm_to_morgans` is `constexpr`; `block_of` is `inline` but not `constexpr` because
`std::floor` is `constexpr` **only since C++23** (P0533R9, verified this pass), and this project pins
C++20 (`CMakeLists.txt:26`). So the asymmetry is correct, not an oversight. Worth a one-word header
note ("`block_of` is runtime-only: `std::floor` is not `constexpr` until C++23") so a future reader
does not "fix" it under C++20. No action beyond the optional note.

### (8) Performance

**8.1 — Single linear pass, one allocation — optimal (STRENGTH). CONFIRMED.** `assign_blocks` is
`O(M)`, one `resize` (cpp:29), no per-SNP allocation, no re-scan. The strictly sequential dependency
(each SNP's block depends on the previous SNP's bin/chrom) precludes parallelization, and at M ≈ 584k
this is sub-millisecond host work that runs *once* per dataset — negligible vs the GPU f2 pass.
Correct to leave scalar.

**8.2 — `block_of` header-inline, will inline into the loop (OK). CONFIRMED.** `[[nodiscard]] inline …
noexcept`, single division+floor; the compiler inlines it (no call overhead, no LTO dependence). Good
that it is in the header.

**8.3 — `static_cast<std::size_t>(s)` repeated three times per iteration (cpp:38,39,51) — cosmetic
(LOW). CONFIRMED.** Free at runtime on LP64 but visual noise. Hoist `const auto su =
static_cast<std::size_t>(s);` once per iteration. (Indexing the spans with `s` directly would *not*
compile cleanly — `span::operator[]` takes `size_type`, so a `long` index triggers a sign-conversion
warning under `-Wconversion`; the cast is the correct silencer. So keep casting, just hoist it.) Pure
readability. Severity **low**, effort **S**, before-M4.5? no.

### (9) Layering / API / ABI vs §4

**9.1 — The non-decreasing + dense + 0-based contract is a load-bearing ABI seam, documented in prose but not machine-checked at the boundary the device dereferences (MED). CONFIRMED and sharpened.**
`cuda_backend.cu:133` and `cpu_backend.cpp:209` both *assume* `block_id` is non-decreasing so "each
block's SNPs are the half-open range [offset, offset+size)" — a single forward scan with no
verification. `assign_blocks` *does* guarantee this (`global` only ever `++`, never decreasing; 2.7
re-verified that this holds even on interleaved chromosomes) and the test's `dense_and_nondecreasing`
checks it. But the contract lives only in prose (hpp:79-81, the consumers' comments). The device
stakes: a non-dense/out-of-range `id` would index `block_offsets[id]`/`block_sizes[id]` in the host
scan **and `block_sizes[id]`/`block_offsets[id]` on the device** (`f2_blocks_kernel.cu:88,96`) — a
logically-wrong-but-possibly-in-bounds layout that compute-sanitizer would not flag.
*Why it matters:* §4 — `core` hands `device` a plain data struct; the *shape* contract (non-decreasing,
dense, 0-based) is the ABI, and an unchecked ABI at a device-dereferenced seam is fragile.
*Fix:* a cheap debug postcondition in `assign_blocks` (`#ifndef NDEBUG`: assert `block_id`
non-decreasing and `block_id.back() == n_block - 1`) **and** an explicit "Postconditions:" block in
the header so the consumers' assumption traces to a guarantee. Severity **med**, effort **S**,
before-M4.5? yes (M4.5 multiplies consumers; pin the seam first).

**9.2 — Value type, no raw pointers, span inputs — clean API (STRENGTH, with one caveat). CONFIRMED.**
`assign_blocks` takes `std::span<const T>` (read-only views, no ownership) and returns an owning
`BlockPartition` by value (NRVO/move) — the modern, ABI-safe shape §7 prefers. The caveat: the
value-by-value cleanliness is partly *undone downstream* by `f2_from_blocks.cpp:50` passing
`partition.block_id.data()` as a raw `const int*` to the `ComputeBackend` seam — itself a deliberate,
documented choice (`backend.hpp:148-149`: keep the backend interface free of std-container ABI). That
is fine *if* the length contract is enforced; see 1.4. No failure crosses the boundary as
`std::expected`/`Error` here because the only failures are precondition violations (1.1), which belong
at the call boundary, not as a per-call return.

### (10) Testability vs §13

**10.1 — Strong `assign_blocks` coverage; the failure modes (the riskiest part) are untested (MED). CONFIRMED.**
`test_block_partition.cpp` pins every documented happy-path property (one-bin, gap-absorption,
two-chrom reset, all-zero chrom, negative bin, empty input), uses a `dense_and_nondecreasing`
structural checker, and adds a real-AADR consistency check (748 blocks under the AT2 walk; was 757)
gated on an argv path — the §13 + ROADMAP §6 internal-consistency gate. Missing, each tied to a finding:
  - no test that invalid `block_size_morgans` (`<= 0` / NaN) early-returns (1.1) — and the test
    *asserts* `block_size_cm_to_morgans(0.0) == 0.0` (test:55) without then checking what
    `block_of`/`assign_blocks` do with that zero, leaving the UB path untested;
  - no `block_of` int-narrowing / non-finite-genpos test (1.2);
  - no `(s == 0)`-isolation test (1.5);
  - no mismatched-length test (1.4) — "use the shorter extent" is unverified;
  - no exact-bin-boundary FP-floor test (3.2).
*Why it matters:* §13/DoD requires the new rule pass "at the tight tier"; the happy path is covered,
the failure modes (1.1/1.4, the riskiest) are not.
*Fix:* add the five cases; the invalid-`bs`/mismatched-length cases should pin the *post-fix* behavior.
Severity **med**, effort **S**, before-M4.5? partly (the invalid-`bs` and mismatched-length tests land
with the 1.1/1.4 fixes).

**10.2 — Pure host functions, trivially unit-testable with no GPU (STRENGTH). CONFIRMED.** The whole
unit compiles and tests on the CPU with no CUDA — §2 testability satisfied — and the dual gtest /
self-checking-`main` harness gates in CI either way. Exemplary.

### (11) Capability tiers (PRO-6000-capable vs budget-5090) — TODO.md CAPABILITY-TIER

**11.1 — Capability-tier-neutral, correctly so (mostly N/A, one orchestrator-level note). CONFIRMED.**
The rule is host-only, runs once, sub-millisecond, touches neither GDS, P2P, NCCL, nor `ncu` — no
capable-vs-budget fork, and it should not gain one. That is the right call: the partition must be
bit-identical regardless of tier (§11.4/§12: "bit-identical across G and to the single-GPU reference"),
and this unit is precisely what guarantees it — it produces the same `block_id[]` on any tier because
it never touches the device. The one adjacent note: at M4.5 the partition is computed once on the host
and the *same* `block_id`/`block_offsets`/`block_sizes` must reach every device so each device's partial
uses an identical block layout (§11.4 host-side fixed-order combine relies on `n_block` and the ordering
being identical across devices). There is no per-device recomputation today (good). The M4.5
orchestrator (not this unit) should add an explicit, **logged** assertion that all devices share one
`BlockPartition` (a one-line `STEPPE_LOG_INFO`, e.g. `[partition] n_block=748 shared across G=2 devices`)
so the parity precondition is observable on both tiers — the "runtime-detected + explicitly-logged
degrade" discipline TODO.md (`wxz1fiiln`) asks for. The hook belongs in M4.5 (and depends on the
not-yet-built `log.hpp`), not in `assign_blocks`. Severity **low**, effort **S**, before-M4.5? the log
hook lands *with* M4.5.

---

## Considered & rejected

- **"`block_of` should pre-divide by a reciprocal to avoid a per-SNP division."** Rejected: FP
  reciprocal-multiply is not bit-identical to division, so it would break parity with AT2's division;
  the per-SNP division is correct and must stay. Not a bug. (Re-verified.)

- **"`m <= 0` should be `m == 0`."** Rejected: `m` is `min` of two non-negative `size_t`-derived values
  cast to `long`; it cannot be negative on realistic input, and `<= 0` is a harmless defensive superset.
  Cosmetic at most.

- **"`n_block` should be `long` to match `global`."** Rejected after checking consumers: `block_id` is
  deliberately `std::vector<int>` (hpp:84-85), `backend.hpp:167` takes `int n_block`, both backends carry
  `int` throughout; the count is provably `≤ M ≤` genome SNP count. Widening would desync the tree-wide
  `int`-count contract for no benefit. The gap is only that the bound is *asserted by comment*, not by
  code (1.3, low).

- **"`static_cast<int>(global)` per iteration (cpp:51) is redundant work."** Rejected: `global` is `long`,
  `block_id` is `int`; the narrowing is necessary and free. Storing `global` as `int` would drop the cast
  but lose the "counter can't overflow mid-loop" belt-and-suspenders. Defensible as-is.

- **"`assign_blocks` should validate that `chrom` is sorted/grouped."** Rejected (re-verified at 2.7): the
  rule is deliberately *order-as-given* (file order); per-chromosome reset is relative to the *previous*
  SNP, not a global sort. Interleaved chromosomes simply produce more (correct, fragmented) blocks — and
  the loop cannot OOB or alias on unsorted input. Documented, intended behavior; sortedness is `io`/upstream's
  concern. Adding a check would impose a policy this layer correctly does not own.

- **"Use `std::ranges`/`std::views::zip` over the two spans."** Rejected: C++20 has no `views::zip`
  (C++23), the loop carries state (`prev_*`, `global`) that does not map to a range adaptor, and the
  explicit index loop is the clearest expression of a stateful single pass.

- **"`block_size_cm_to_morgans` should live in `config.hpp` next to the constants."** Rejected:
  config.hpp:91-93 deliberately says the conversion "lives in exactly one place, next to
  `block_partition_rule.hpp`." Co-locating the conversion *function* with the rule that consumes it (the
  constant `kCentimorgansPerMorgan` stays in config.hpp) is the intended split. Correct.

- **"Make `block_of` `constexpr`."** Rejected for C++20: `std::floor` is `constexpr` only since C++23
  (P0533R9, verified this pass), and `CMAKE_CXX_STANDARD 20` pins C++20, so it cannot be. Noted as 7.7
  (informational) so a future reader does not attempt it.

- **First pass's 3.2 example "`0.10 / 0.05` → bin 1."** Rejected as stated: re-checked the actual FP64
  values; the round-to-nearest quotient of the stored `0.10` and `0.05` is exactly `2.0`, so
  `block_of(0.10, 0.05) == 2`. The *general* boundary-tie concern is kept (3.2); the specific off-by-one
  example was wrong and is removed.

- **First pass's 1.5 "OOB in the test's `seen` vector."** Rejected as stated: `dense_and_nondecreasing`
  guards `id < 0 || id >= n_block` *before* indexing `seen` (test:67), so an all-`-1` regression yields a
  clean test failure, not an OOB. The testability gap (no `(s==0)`-isolation case) is kept; the OOB
  consequence is corrected.

- **First pass's 1.4 `STEPPE_LOG_WARN` fix "adds a `core/internal/log.hpp` include — acceptable."**
  Partially rejected: verified `core/internal/log.hpp` does **not** exist and no `STEPPE_LOG_*` symbol is
  present under `src/core` today (named in architecture §870 as planned, not built). So the log-on-skew
  half of 1.4 is **deferred**, not immediately implementable; the debug `assert` half is implementable now.
  Re-targeted in 1.4 accordingly.

- **"Couple `chrom`/`genpos` into one `std::span<const SnpRow>` to kill the length-mismatch class
  entirely."** Considered, rejected as the *fix* (kept as a thought): it would remove 1.4's mismatch class
  structurally, but the producers (`snp_reader` emits parallel `chrom`/`genpos_morgans` vectors;
  `views.hpp` is column-major parallel) are all SoA, and restructuring them into AoS for one callee is a
  larger change than the assert fix and fights the SoA layout the GPU feeder wants. The assert fix (1.4) is
  the right-sized remedy.

- **"`block_of` should clamp/saturate the cast (e.g. `lround` + clamp) instead of `static_cast<int>`."**
  Rejected as the fix: silently clamping a non-finite or huge `genpos` to `INT_MAX`/`INT_MIN` would *hide*
  the upstream data error and still produce a wrong (but now non-UB) partition — the opposite of §2
  fail-fast. The correct posture is reject-at-the-boundary (upstream finiteness validation) + widen the
  local bin to `long` (1.2), not saturate.

## What it takes to reach 10/10

1. **(1.1, HIGH) Enforce `block_size_morgans > 0` in `assign_blocks` NOW** (the builder does not exist —
   verified by grep — so this is the primary guard): `if (!(block_size_morgans > 0.0)) return out;`
   (rejects 0/negative/NaN). Document the precondition as enforced here (and at `ConfigBuilder::build()`
   once it lands). Add the matching test. Removes the only HIGH UB path.
2. **(1.4 / 9.1, MED) Pin the two cross-layer seams.** (a) `assert` equal `chrom`/`genpos` lengths in
   debug, keep the min-extent guard in release (the log-on-skew is deferred until `log.hpp` exists);
   recommend the orchestration assert `block_id.size() == static_cast<size_t>(Q.M)` in
   `f2_from_blocks.cpp` (the *live* seam — `block_id.size()` is decoupled from `Q.M`). (b) Add a debug
   postcondition (non-decreasing + `back() == n_block-1`) and an explicit "Postconditions:" header block,
   so the device's `block_offsets[id]`/`block_sizes[id]` dereference traces to a guarantee. Add tests.
3. **(1.2 / 2.4, MED) Close the `block_of` int-cast edge.** Widen the *local* bin to `long` (it never
   needs to be `int`; `block_of → long`, `prev_local_bin → long`), or document+guard the finite/in-range
   precondition; the `snp_reader` genpos parse is the upstream culprit (`inf`/`nan` tokens parse to
   non-finite via `num_get`→`strtod`, verified) and should clamp or reject non-finite Morgan positions.
4. **(10.1, MED) Backfill the failure-mode tests:** invalid `bs`, mismatched length, `(s==0)`-isolation,
   int-narrowing, exact-bin-boundary FP floor.
5. **(3.2, LOW) Pin the FP floor-tie behavior** in a `block_of` test and flag "verify the floor-tie
   convention against the AT2 `blgsize` golden" in the M7 parity checklist.
6. **(7.1, LOW) Clarify the device-consumption comment** (device reads the *result arrays*, not a
   device-callable `block_of`; no `STEPPE_HD` macro until a consumer needs one).
7. **(11.1, LOW) Add the M4.5 "one shared partition across G devices" logged assertion** in the multi-GPU
   orchestrator (not this unit; depends on the not-yet-built `log.hpp`) so the parity precondition this
   unit underwrites is observable on both capability tiers.
8. **(1.3, 1.6, 7.3, 7.5, 7.7, 8.3, LOW polish)** Reword the `int n_block` comment to the provable bound;
   fix the `block_of` `@return` "zero-based" wording; document `assign_blocks`'s `bad_alloc`; note
   `block_of` is runtime-only under C++20; hoist the per-iteration `size_t` cast.

Items 1, 2, and 3 are the substantive ones; 1 and 2 should land **before M4.5** multi-GPU sharding (a
wrong/UB rule or an unchecked length seam there corrupts every device partial). With 1.1 enforced
in-unit, the length and contiguity seams made explicit at the device-dereferenced boundary, and the
failure-mode tests in, this is a clean 9.5–10/10 single-source domain rule.

## Good patterns to keep

- **The single-source discipline is real, not aspirational.** Grep confirms exactly one cM↔Morgan site,
  one `assign_blocks`, one `block_of`; every consumer (`io`, both backends, all reference tests, the device
  kernel) reads them and none re-derives — the §2/§8 invariant this unit exists to uphold actually holds
  across the tree.
- **Host-pure, CUDA-free, yet compiles cleanly under nvcc** (it is `#include`d into two `.cu` TUs as host
  code) — the one documented §4 layering exception, implemented exactly as specified, so it parity-anchors
  the multi-GPU path for free.
- **Unit-explicit naming** (`genpos_morgans`, `block_size_morgans`, `_cm` suffix) makes the cM-vs-Morgan
  distinction impossible to get wrong at a call site.
- **Order-as-given, allocation-light, single-pass `assign_blocks`** — `O(M)`, one `resize`, no intermediate
  materialization, runs once; robust even on interleaved/unsorted chromosomes (more blocks, never aliasing
  or OOB); the right shape for a precompute-once host stage.
- **Correct, tested handling of the genuinely-tricky real-data edges** (negative chr17 positions anchor the
  first block; all-zero chr24 → one block; SNP-anchored cuts with the remainder rolling forward;
  per-chromosome reset) with a `dense_and_nondecreasing` structural checker and a real-AADR consistency
  gate (748 blocks under the AT2 walk; chr 1-23 → 747, chr 1-22 → 711; AT2 cache parity target 709 per
  `docs/research/block-partition-at2.md`) — exactly the ROADMAP §6 "property identities + internal-consistency
  on real data, never synthetic" gate.
- **Clean modern API**: `std::span<const T>` inputs, value-returned owning result, `[[nodiscard]]`
  everywhere, `noexcept` on the pure primitives and (correctly) not on the allocating pass; `constexpr` on
  the conversion (and correctly *not* on `block_of`, which cannot be under C++20).
- **High, section-cited comment density** matching the surrounding `core` style; the `.cpp` loop comment
  explains the three open-new triggers and their downstream consequences precisely.
