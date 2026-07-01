# Review — `io::read_ind` (`src/io/ind_reader.hpp` + `src/io/ind_reader.cpp`)

Reviewer pass: exhaustive senior CUDA/C++ review against `docs/architecture.md` (§2, §4, §5, §7, §12, §13), `docs/ROADMAP.md` (§4–§6), `docs/TODO.md` (capability-tier section). Read in full: both unit files; the directly-related `io` leaf siblings (`snp_reader.{hpp,cpp}`, `geno_reader.{hpp,cpp}`, `eigenstrat_format.hpp`); the consumer `geno_reader.cpp::read_tile`; the two callers (`tests/reference/test_decode_equivalence.cu`, `tests/reference/test_filter_oracle.cu`); and the **canonical oracle** `aadr/build_tgeno_matrix.py` that this reader must reproduce bit-for-bit.

## Role & layering

`read_ind` is a pure-host C++20 function in the `io` leaf. It parses an EIGENSTRAT `.ind` file (`<id> <sex> <pop>` per row), groups individual-record (row) indices by population label, applies one of three selection modes (Explicit / AutoTopK / MinN), and returns `IndPartition` — the selected populations in the final Q/V/N row order (sorted ascending by label). The output is consumed downstream by `GenoReader::read_tile`, which gathers the packed bytes for exactly those rows into population-contiguous tile order. This is the seam that binds a population to the genotype records it owns.

Layering is **correct and clean**: the header includes only `<cstddef> <string> <vector>`; the TU adds only standard-library headers; there is no CUDA, no `core`/`device` dependency, no upward edge. It surfaces I/O / empty-selection failures as `std::runtime_error` (the leaf's documented contract, architecture §4). `[[nodiscard]]` is present on the public entry point. This matches the `io`-leaf contract and the sibling readers exactly.

The unit is small (132 + 92 lines) and largely well-written, with comment density matching the surrounding `io` files. The dominant axis of risk is **not** CUDA — it is **oracle parity** (architecture §12, ROADMAP §6: "decode reproduces the numpy oracle bit-for-bit"). The single hard requirement is that the selected set and its ordering match `build_tgeno_matrix.py` exactly, because the GPU decoder's Q/V/N rows are validated against matrices that oracle produced. I traced the C++ against the Python line by line; most of it matches, but there are real divergences and several latent correctness/robustness gaps below.

## Score: 7.5/10 — solid, correctly-layered, oracle-aware; held back by one true semantic bug (`n_individuals_total`), two latent parity divergences (non-ASCII sort, default mode), and missing unit-test coverage of the selection logic itself.

---

## Findings

### (1) Correctness & bugs

**1.1 — `n_individuals_total` does not mean what its docs/name claim; it conflates "rows seen" with "individual axis", and diverges from the oracle's `n_records`.** [HIGH] [S] [before-M4.5: yes]
- Location: `.cpp` lines 51, 57–59, 68, 72; `.hpp` lines 69–72 (the field doc).
- The header says `n_individuals_total` is "the .ind row count (the genotype individual axis), independent of selection" and "total .ind rows == TGENO records". The implementation sets `part.n_individuals_total = row;` (line 72) where `row` was incremented (a) only on **parseable** lines and (b) **including** lines past the `n_records_present` cap (line 58 `++row` inside the skip branch). So:
  - It is **not** the total `.ind` row count: blank/short lines (line 56 `continue`, no `++row`) are excluded, so a `.ind` with trailing blank lines or comment rows yields `n_individuals_total < file_row_count`.
  - It is **not** "== TGENO records present" either: when `n_records_present` caps the file, `row` keeps incrementing past the cap (line 58), so `n_individuals_total` becomes `min(parseable_rows, file_total)` which can **exceed** `n_records_present`.
- The oracle never computes an analogue of "total rows seen including post-cap"; it caps with `pops_all[:n_records]` and the only count that flows downstream is the per-pop membership. So this field's value is an artifact, not a quantity any consumer needs.
- Why it matters: the field is **never read** by any current consumer (`read_tile` uses only `part.groups`; grep confirms no other reader). A struct field that is (a) misdocumented, (b) ambiguous between three different counts, and (c) unused is a latent trap — the first caller that trusts the doc ("== TGENO records") will be silently wrong on a capped or comment-bearing file. Architecture §2 "fail-fast / no silent corruption."
- Fix: decide the contract and enforce it. Either (i) remove the field until a consumer needs it; or (ii) define it precisely as "number of parseable `.ind` rows **within the cap** that were grouped" — increment a separate counter only inside the grouped branch, and do **not** `++row` past the cap for the total. Update the doc to match. If the intended meaning is "the individual axis the partition indexes into", that is `min(parseable_rows, n_records_present)`, which is what `read_tile` would need to validate row bounds — make that the value and say so.

**1.2 — The cap counter `row` does triple duty (cap key / stored row index / running total) with no asserted invariant — correct today, fragile to any future edit.** [MED] [S] [before-M4.5: yes]
- Location: `.cpp` lines 57–68, 72.
- Today this is correct: because `++row` happens on every parseable line, the stored row index equals the index into the parseable-row stream, which is exactly Python's `enumerate(pops_all[:n_records])` index (the oracle drops blank/short lines *before* the `[:n_records]` slice, and so does C++ via the `continue` at line 56 without incrementing). I traced both and the **grouped** indices match. But the triple-duty of `row` is an implicit invariant with no assertion; any future edit (e.g. skipping a malformed line after the cap check, or moving a `++row`) silently breaks parity.
- Why it matters: the parity contract (architecture §12, ROADMAP §6 operative M1–M4 gate) rests on this index equality, enforced only by the precise placement of two `++row` statements.
- Fix: separate concerns — one monotonic `parse_index` used as cap key + row index, and (if 1.1 keeps the field) a distinct grouped-count. Add a comment asserting "row index == parseable-row ordinal == oracle `enumerate(pops_all[:n_records])` index". Consider asserting every stored index in a group is `< n_records_present`.

**1.3 — `RawGroup::first_seen` is set to `groups.size()` (its own slot index), making the field redundant.** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` line 64: `groups.push_back(RawGroup{pop, groups.size(), {row}});`.
- The value is correct (monotonic in first-appearance order; matches the oracle's "first encountered" tie-break — verified below). But because we only push, never reorder `groups`, `first_seen` is **always equal to the element's own index in `groups`** — a denormalized copy that invites a future bug if `groups` is ever reordered. It also re-derives `groups.size()` on two adjacent lines (line 63 `emplace`, line 64 push), coupling them to read identically (architecture §2 DRY).
- Fix: compute the slot once: `const std::size_t slot = groups.size(); index_of.emplace(pop, slot); groups.push_back(RawGroup{pop, slot, {row}});`. Or drop `first_seen` entirely (see 1.4).

**1.4 — AutoTopK tie-break is doubly-specified: `stable_sort` + an explicit `first_seen` comparator.** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` lines 94–104.
- `by_count` is built by iterating `groups` (first-appearance order), then `std::stable_sort` is applied with a comparator comparing count **and**, on equal count, `first_seen`. Because the input is already in first-appearance order and the sort is *stable*, the `first_seen` tie-break is redundant — a stable sort keyed on count alone already preserves first-appearance order. Conversely, if the comparator fully orders ties, plain `std::sort` would suffice. Belt and braces.
- Why it matters: minor; correct. But it muddies the single source of the tie-break rule — a reader cannot tell whether stability or the comparator is load-bearing (architecture §2).
- Fix: pick one. Cleanest: `std::stable_sort` keyed on count descending only (input order = first-appearance = the tie-break), drop `first_seen`, and comment the reliance on input order.

**1.5 — Explicit mode silently ignores requested-but-absent labels; the oracle *warns*.** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` lines 82–89; oracle lines 80–85 (`missing = [...]; print("[warn] ... requested pops absent ...")`).
- The C++ Explicit path keeps the present subset (parity-correct for the *result*) but emits nothing when requested labels are absent. For a leaf with no logging facade this is defensible, but it is an observability gap vs the oracle and the project's "explicit, logged" ethos. Low severity (no numeric effect): a typo'd label list silently yields a smaller set.
- Fix: when `io` gains a logging seam (or via an optional diagnostics out-param), report the absent labels; document the current silent behavior in the header until then (see 11.1).

**1.6 — Duplicate labels in `sel.labels` (Explicit) are silently de-duplicated — correct but undocumented.** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` line 84 (`std::unordered_set want(...)`).
- Building a set from `sel.labels` de-dups, and iterating `groups` adds each present label at most once — so the result is dedup-correct regardless of caller duplicates. No bug. Add a one-line header note that duplicates in `labels` are ignored.

### (2) Edge cases & failure modes

**2.1 — `n_records_present == 0` is not special-cased and yields a misleading error.** [MED] [S] [before-M4.5: yes]
- Location: `.cpp` lines 57–60, 74–76.
- With `n_records_present == 0`, every parseable row hits `row >= n_records_present` and is skipped; `groups` stays empty; the function throws `"no individuals parsed from <path>"` (line 75) — blaming the `.ind` file when the real cause is the zero cap. (`GenoReader` throws first at its own line 59 today, so this is reachable only via a direct/test caller, but it is still a wrong diagnostic.)
- Why it matters: fail-fast must give the correct cause (architecture §2). `SIZE_MAX` ("use every row") is fine; `0` is the problematic case.
- Fix: validate `n_records_present` up front — if `0`, throw a message naming the cap, not the file content.

**2.2 — `sel.k == 0` in AutoTopK selects zero populations, and the *default-constructed* `PopSelection` is therefore always invalid AND does not match the oracle's default selection path.** [MED] [S] [before-M4.5: yes]
- Location: `.cpp` lines 103–104, 115–116; `.hpp` lines 45, 48 (`mode = AutoTopK`, `k = 0`).
- `std::min(sel.k, by_count.size())` with `k==0` gives `0` → `selected` empty → the empty-selection throw fires. So a default `PopSelection{}` always throws. More importantly, the oracle's default with both flags unset (`--auto-top 0`, `--min-n 1`) **falls through to the MinN branch** (oracle lines 86–89: `auto_top > 0` is false → `else: sel = [p for ... c >= min_n]`). The C++ default `mode = AutoTopK` therefore does **not** mirror the oracle's default selection path (MinN/1).
- Why it matters: parity of *defaults*. Anything constructing `PopSelection{}` and relying on "default behavior" gets a throw in C++ vs "all pops with ≥1 individual" in the oracle. Low blast radius today (callers set mode+k explicitly), but a latent divergence and a confusing default.
- Fix: either default `mode = MinN` with `min_n = 1` (matching the oracle fall-through), or document the default as intentionally-invalid; and validate `k > 0` for AutoTopK with a clear message rather than reaching the generic empty-selection throw.

**2.3 — `sel.min_n == 0` selects every group; default `min_n == 1` matches the oracle.** [LOW] [—] [before-M4.5: no]
- Location: `.cpp` lines 107–112; `.hpp` line 51.
- With `min_n == 0`, every non-empty group passes (all groups have ≥1 row). Harmless. Default `1` and the predicate `rows.size() >= sel.min_n` are identical to the oracle's `--min-n` default `1` and `c >= args.min_n`. Parity-correct. Noting the boundary.

**2.4 — Empty-string / short-line handling is correct and matches the oracle.** [LOW] [—] [before-M4.5: no]
- Location: `.cpp` line 56 vs oracle lines 41–43 (`len(t) >= 3`).
- `ls >> id >> sex >> pop` skips leading whitespace and reads non-whitespace runs, so it succeeds only on ≥3 whitespace-delimited tokens and `pop` is never the empty string. Lines with <3 tokens are skipped without incrementing `row` — exactly the oracle's `len(t) >= 3` filter applied before the cap slice. No bug. A row with >3 tokens takes the 3rd as the pop (oracle `t[2]`) — parity-correct.

**2.5 — Integer width: all counts/indices are `std::size_t`; `static_cast<std::ptrdiff_t>(k)` cannot overflow.** [LOW] [—] [before-M4.5: no]
- Location: `.cpp` line 104.
- `k = min(sel.k, by_count.size())` ≤ distinct-pop count (thousands), so the cast to `ptrdiff_t` is safe; `row` counts individuals (v66 ≈ 27k). No `size_t` overflow concern. (The architecture's "`size_t` mandatory above P≈32k" note is about the *matrix* index, not relevant here.)

**2.6 — A readable-but-non-`.ind` file parses without a sanity check — matches the oracle (which also has none).** [LOW] [M] [before-M4.5: no]
- Location: `.cpp` lines 40–69.
- `std::ifstream` opens in text mode; a wrong file either fails extraction every line (→ a reasonable "no individuals parsed" error) or pathologically extracts noise tokens. The oracle has no header check either, so parity holds. The `.geno` is opened/validated separately by the caller. Mention only.

### (3) Numerical / precision vs §12

**No floating-point arithmetic exists in this unit** — it parses labels and integer row indices. There is nothing to evaluate against Ozaki / native-FP64 / determinism §12. But §12's *ordering-determinism* spirit is this unit's whole job, and it has a real subtlety:

**3.1 — The final ascending-label sort uses `std::string::operator<` (unsigned-byte order), which is NOT identical to Python's `sorted()` (code-point order) for non-ASCII labels; the header claims it is.** [MED] [S] [before-M4.5: yes for the doc; data is ASCII today]
- Location: `.cpp` lines 119–123; header/comment claim "`std::string < is byte/lexicographic order, matching Python's sorted() on str`".
- Verified against the C++ standard: `basic_string` comparison delegates to `char_traits<char>::compare`, whose `eq`/`lt` are "defined identically to the built-in operators == and < for type **unsigned char**" — ordering is by **unsigned byte value**. Python `str` sort orders by **Unicode code point**. For pure-ASCII labels (≤ 0x7F) the two coincide exactly. For labels with non-ASCII UTF-8 bytes (accented population names), C++ compares raw bytes while Python compares decoded code points; these can disagree.
- Why it matters: this is the population (row) order of Q/V/N. A mismatch silently permutes rows relative to the oracle's matrices → the bit-for-bit decode-equivalence test (architecture §13, ROADMAP §6) fails — or worse, passes on ASCII fixtures and breaks on a non-ASCII real dataset. The comment overstates the guarantee.
- Fix: scope the comment to "matches Python `sorted()` for ASCII labels; for non-ASCII UTF-8 the orderings can differ (C++ = byte order, Python = code-point order)." If non-ASCII labels are ever in scope, choose a canonical order (most robust: compare UTF-8 bytes on *both* sides — an oracle change to `sorted(..., key=lambda p: p.encode())`). For now, an asserted ASCII precondition makes the divergence fail-fast rather than silently mis-order. This is the single most parity-relevant subtlety in the unit.

**3.2 — Whitespace tokenization differs subtly from Python `str.split()` (Unicode whitespace); ASCII agrees.** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` lines 54–56 vs oracle line 41.
- `std::istringstream >> std::string` splits on `std::isspace` under the "C" locale (space, `\t\n\v\f\r`), coalescing runs and stripping ends — Python `str.split()` does likewise for ASCII but also treats Unicode whitespace (e.g. NBSP) as a separator. Real EIGENSTRAT `.ind` files are ASCII space/tab-delimited, so parity holds in practice. Add a comment if you want belt-and-braces.

### (4) CUDA idioms / RAII / streams vs §7

**4.1 — N/A: pure-host leaf, no CUDA.** There is no stream, device allocation, kernel launch, or library workspace to evaluate against §7's RAII-owning-wrappers, `STEPPE_CUDA_CHECK`, launch-config, async-pool, or stream/event idioms. The correct §7-relevant fact is the **absence** of CUDA, which is exactly right for an `io` leaf (architecture §4). File-handle RAII is delegated to `std::ifstream` (closes on scope exit) — idiomatic; no raw resource escapes scope.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**5.1 — No offending magic numbers.** The only literals are `0` (counters/offsets) and the `{row}` initializer. The `.ind` column layout (`id`, `sex`, `pop`) is encoded structurally via three named `>>` targets — *better* than the oracle's `t[2]` index. The `SIZE_MAX` "all rows" sentinel is documented. The mode defaults (`k = 0`, `min_n = 1`) live in the config struct itself (ROADMAP §4's correct home for config defaults), not in a compute path — acceptable. (`k = 0` as an invalid sentinel is a usability issue, not a magic-number issue — see 2.2.)

### (6) Decomposition / single-responsibility / function size vs §2

**6.1 — `read_ind` is one ~90-line function doing parse + group + 3-way select + sort + materialize; the selection policy should be an extractable pure function.** [MED] [M] [before-M4.5: no]
- Location: `.cpp` lines 37–130.
- Readable and comment-delimited, but it bundles four responsibilities: parse+group (40–76), selection dispatch (78–113), final ordering (119–123), materialization (125–129). The selection switch is three independent policies inline; testing one in isolation requires writing a file. Architecture §2 (separation of concerns, testability) favors extracting the selection.
- Why it matters: testability (§13) — see 10.1. Parsing and selection policy are independently valuable to test; coupling them behind a file path forces every selection test to write a fixture.
- Fix: extract `select_groups(const std::vector<RawGroup>&, const PopSelection&) -> std::vector<const RawGroup*>` (anonymous-namespace, pure, no I/O); keep `read_ind` as parse → `select_groups` → sort → materialize. Each mode then gets a direct unit test. Highest-leverage structural change for the unit.

**6.2 — `RawGroup` is internal-only and correctly placed in the anonymous namespace.** [LOW] [—]
- The parse-time representation (with `first_seen`) is hidden from the public ABI; only `PopGroup`/`IndPartition` are exposed. Correct public/internal split. No action (modulo 1.3 on `first_seen`).

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comments

**7.1 — Two header docs over-state guarantees the implementation does not make.** [MED] [S] [before-M4.5: yes]
- `n_individuals_total == "total .ind rows == TGENO records"` is false (1.1); "`std::string < ... matching Python's sorted()`" is ASCII-only (3.1).
- Why it matters: comment density is otherwise excellent and the file is documentation-heavy, so readers *will* trust these comments — making the inaccuracies higher-impact than in a sparse file (architecture §2, ROADMAP cross-cutting standards).
- Fix: correct both per 1.1 and 3.1.

**7.2 — Naming is good; `row` is overloaded.** [LOW] [S]
- `index_of`, `groups`, `by_count`, `selected`, `want` are clear. `row` plays cap-key / index / total — rename per 1.2 to e.g. `parse_index`. No other naming issues.

**7.3 — `noexcept` correctly omitted; `[[nodiscard]]` present; const-correctness good.** [LOW] [—]
- `read_ind` legitimately throws, so non-`noexcept` is correct (a `noexcept` would `std::terminate` on a normal error path). `[[nodiscard]]` is on the declaration (`.hpp` line 86). Selection comparators take `const RawGroup*`; `selected` holds `const RawGroup*`. No missing `const`.

**7.4 — `PopGroup{g->label, g->rows}` copies; could move (negligible at this scale).** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` line 127. `selected` holds `const RawGroup*` into a soon-discarded `groups`, so the rows could be moved; but the `const` would have to be dropped. See 8.2.

### (8) Performance

Once-per-run host parse of a ~27k-line text file; metadata-only (architecture §5 S−2 is explicitly "no genotype read"); never on the GPU hot path. Findings are craft, not wall-clock.

**8.1 — Per-line `std::istringstream` construction allocates each iteration.** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` line 54 (inside the `while`). For 27k lines this is sub-millisecond. A `string_view` manual split would avoid it for no measurable gain. Note only.

**8.2 — `rows` could be moved out of `RawGroup` instead of copied.** [LOW] [S] [before-M4.5: no]
- Location: `.cpp` line 127. Negligible at this scale; do it only if you restructure for 6.1 (which can make `groups` mutably selectable).

### (9) Layering / API / ABI vs §4

**9.1 — Layering is exemplary; no upward dependency, no CUDA, plain-data output.** [—]
- Header includes only `<cstddef> <string> <vector>`; output is plain structs consumed by `geno_reader` (another `io` leaf) and the `app`/test layer. Matches architecture §4 ("`io` ... produces plain data structs ... depends on nothing in core/device"). No `block_partition_rule` dependency is needed here (that is `snp_reader`'s concern). Correct.

**9.2 — Minimal, value-semantic API; no global state.** [LOW] [—]
- One free function returning a value type; immutable inputs by const-ref; the `PopSelection` value object is the "typed immutable config" §9 pattern (a mutable aggregate the caller treats as one-shot config). Satisfies §2 "no global mutable state."

**9.3 — `IndPartition` is the natural seam to carry the effective individual-axis length.** [LOW] [S] [before-M4.5: no]
- `read_tile` re-derives `n_ind` by summing `g.rows.size()` (geno_reader.cpp line 92) and ignores `n_individuals_total`. If 1.1 redefines the field to the effective individual-axis count, `read_tile` could assert its row indices in range against it. Tie to the 1.1 fix.

### (10) Testability vs §13

**10.1 — There is NO direct unit test of `read_ind` / `PopSelection`; the two non-AutoTopK modes have zero coverage.** [HIGH] [M] [before-M4.5: yes]
- Evidence: grep shows `read_ind`/`PopSelection`/`IndPartition` referenced only in `tests/reference/test_decode_equivalence.cu` and `tests/reference/test_filter_oracle.cu`, both exercising **only `Mode::AutoTopK`** as a *step* of an end-to-end decode test on the real AADR file — neither asserts the partition's contents, and Explicit / MinN / the tie-break / the cap / empty selection / the label sort are **untested**. There is no `tests/unit/test_ind_reader.cpp` analogous to the existing `test_block_partition.cpp` / `test_filters.cpp`.
- Why it matters: architecture §13 and ROADMAP §5/§6 make the reference/oracle gate the operative acceptance criterion, and this selection logic *is* the oracle-parity-critical code (it decides which rows become which Q/V/N rows). The exact things a parity bug hides in — the AutoTopK tie-break, the `n_records_present` cap (1.1), the non-ASCII sort (3.1), the default-mode divergence (2.2) — are entirely unexercised.
- Fix: add `tests/unit/test_ind_reader.cpp` (host-only, no GPU — the unit is pure host, ideal per §13 "exercisable GPU-free"): synthetic `tmpfile`/in-memory `.ind` fixtures covering (a) all three modes; (b) AutoTopK tie-break (equal counts → assert first-appearance order); (c) the cap (rows past `n_records_present` excluded from groups; assert `n_individuals_total` per the 1.1 contract); (d) empty-selection throw for each mode; (e) blank/short-line skipping; (f) ascending-label order; (g) one fixture cross-checked against `build_tgeno_matrix.py`'s selection on identical content to lock parity. Make 6.1's extracted `select_groups` directly testable so most cases need no file.

**10.2 — Otherwise trivially testable (pure host, deterministic, no globals/RNG/time).** [LOW] [—]
- The only reason coverage is absent is that no test was written, not that the design impedes it (modulo 6.1). Good design, missing tests.

### (11) Capability tiers (PRO-6000-capable vs budget-5090) vs TODO

**11.1 — This unit is capability-tier-neutral; its contribution is the run's *capability-tagged selection provenance*, not a degrade path.** [LOW] [S] [before-M4.5: no]
- `read_ind` touches neither GDS, P2P, profiling, nor VRAM — it is metadata-only host I/O, identical on the PRO 6000 and the 5090, so there is **nothing to degrade** here (correctly; state this explicitly in the run record so it is not mistaken for an un-handled lever). Per TODO's capability-tier section ("every run records which path it took") the relevant gap is **observability**: the selection result (mode, K/min_n, P selected, individuals used, absent-requested labels per 1.5) is exactly the provenance a reproducible run-record should capture (architecture §12 "recorded in golden metadata"; the oracle already prints `[sel] P=... pops, M=... individuals used=...` to stderr). `read_ind` returns the partition but logs nothing.
- Fix (when the `io` logging seam exists): emit one structured INFO line `{mode, requested K-or-minN, P_selected, n_individuals_used, n_absent_requested}` + the chosen-population list, auditable against the oracle's `meta.json` (`pops`, `n_indiv_per_pop`). Parity-neutral; no tier branch needed.

---

## Considered & rejected

- **"`std::sort` at line 122 is non-stable → could non-deterministically reorder equal-key groups."** Rejected: the sort key is the population *label*, and `index_of` guarantees exactly one `RawGroup` per distinct label, so all keys are **distinct** — stability is irrelevant and the order is fully determined. (Verified `std::sort` carries no stability guarantee; it simply does not matter here.)
- **"`row` increments past `n_records_present` (line 58), so post-cap row indices are wrong / parity breaks."** Rejected as a *selection* bug: rows past the cap are never grouped (the `continue` at line 59 fires before grouping), so no out-of-range index enters `groups`. Traced against the oracle's `enumerate(pops_all[:n_records])` — the **grouped** indices match exactly. The `++row` past the cap only corrupts the unused `n_individuals_total` (finding 1.1), not the partition.
- **"AutoTopK `first_seen` tie-break does not match Python `Counter.most_common`."** Rejected: verified the Python docs — `most_common` breaks equal counts by *first-encountered* order, and the oracle's `Counter` is built from a `defaultdict` populated in row order (insertion-ordered dict), so first-encountered == first `.ind` appearance. C++ `first_seen` = slot index = first-appearance order reproduces this. Parity holds (the only nit is redundancy — 1.3/1.4).
- **"Explicit-mode result order might differ from the oracle."** Rejected: in *both* implementations the final step is `sorted(sel)` / ascending-label `std::sort`, so the pre-sort order of the Explicit branch is irrelevant to the result. Matches.
- **"`std::stoi` / numeric-token concerns."** Rejected as out of scope — that logic lives in `snp_reader.cpp::chrom_code`, not this unit.
- **"Per-line `istringstream` allocation is a performance problem."** Rejected as material: once-per-run metadata parse (architecture §5 S−2), sub-millisecond, never on the GPU hot path. Logged as 8.1 for completeness.
- **"Missing `noexcept` on `read_ind`."** Rejected: the function legitimately throws on open failure / empty parse / empty selection, so `noexcept` would be wrong and would `std::terminate` on a normal error path.
- **"`unordered_set<string> want` ordering affects results."** Rejected: used only for membership (`want.count`), never iterated for ordering, so its unordered nature cannot affect the deterministic output.
- **"Empty population label could group as `""`."** Rejected: `ls >> id >> sex >> pop` cannot yield an empty `pop` on a successful extraction (it reads a non-whitespace run); a <3-token line fails extraction and is skipped. No path to an empty label.

---

## What it takes to reach 10/10

In priority (parity/correctness first, then testability, then polish):

1. **Fix `n_individuals_total` (1.1):** give it a single, precise, *used* meaning (effective individual-axis count within the cap) or remove it; correct the header doc. — HIGH, S.
2. **Add `tests/unit/test_ind_reader.cpp` (10.1):** cover all three modes, the AutoTopK tie-break, the cap, empty-selection throws, blank-line skipping, ascending-label order, and a Python-oracle cross-check fixture. — HIGH, M.
3. **Correct the over-strong header claims (3.1, 7.1):** scope "matches Python `sorted()`" to ASCII; document the non-ASCII byte-vs-code-point divergence (or add a fail-fast ASCII precondition). — MED, S.
4. **Validate inputs up front (2.1, 2.2):** explicit, well-messaged errors for `n_records_present == 0` and `AutoTopK` with `k == 0`; reconsider the default `mode` to match the oracle's flag-default fall-through (MinN/1) or document the default as intentionally-invalid. — MED, S.
5. **Extract `select_groups(...)` as a pure, file-free function (6.1):** unlocks isolated selection tests and clarifies single-responsibility. — MED, M.
6. **De-duplicate the tie-break specification (1.3, 1.4):** compute the slot index once; rely on stable-sort-over-first-appearance OR the comparator, not both; drop `first_seen` if stability suffices. — LOW, S.
7. **Disambiguate the `row` variable (1.2, 7.2):** separate cap key / row index / grouped-count; document the "row index == oracle enumerate index" invariant. — LOW/MED, S.
8. **Selection provenance hook (1.5, 11.1):** when the `io` logging seam lands, emit the capability-tagged selection line mirroring the oracle's stderr. — LOW, S.

None of these block M4 numerics (metadata-only host code), but #1–#4 are parity-relevant and should land before the AT2-golden gate (M7), since they protect the exact selection the goldens assume.

## Good patterns to keep

- **Exemplary layering:** pure host C++20, standard-library-only includes, plain-data output, zero CUDA / `core` / `device` coupling — a textbook `io` leaf (architecture §4).
- **Oracle-driven design with the parity contract written into the comments:** the header names `build_tgeno_matrix.py`, the three selection modes, the `most_common` tie-break, and `sel = sorted(sel)`. Exactly the "tie every behavior to the golden" discipline ROADMAP §6 wants (just make the comments *precise* — 3.1/1.1).
- **Structural column handling over index literals:** `ls >> id >> sex >> pop` encodes the `.ind` layout via named targets, not `t[2]`-style magic indices — cleaner than the oracle, zero magic numbers (ROADMAP §4).
- **Public/internal type split:** `RawGroup` hidden in the anonymous namespace; only `PopGroup`/`IndPartition` cross the ABI. Correct ownership and a minimal public surface.
- **Fail-fast with file context:** every error path throws `std::runtime_error` naming the path and cause, surfaced as exceptions per the leaf contract (architecture §4). (Add the two missing input-validation cases, 2.1/2.2.)
- **Value-semantic, no global state:** immutable `const&` inputs, returned-by-value output, no singletons/statics — satisfies architecture §2 directly; thread-safe and testable.
- **`[[nodiscard]]` on the entry point** and correct const-correctness on the selection comparators.
