# Review — `io::snp_reader` (`src/io/snp_reader.{hpp,cpp}`)

Reviewer: senior CUDA/C++ reviewer. Scope: the `.snp` parser only (`read_snp`, `chrom_code`, `SnpTable`), plus the directly-related context it implements against: `core/domain/block_partition_rule.{hpp,cpp}` (the consumer of `genpos_morgans`/`chrom`), `io/filter/filter_decision.hpp` + `io/filter/snp_filter.*` (consumer of `ref`/`alt`/`id`), the sibling `io` readers (`ind_reader`, `geno_reader`, `eigenstrat_format`) for convention parity, the AADR oracle (`aadr/build_tgeno_matrix.py`, `experiments/aadr/02_build_matrix.py`), and the equivalence tests that call `read_snp`.

Standard-library / language claims below are cited to primary sources where load-bearing.

## Role & layering

`io::read_snp` is the EIGENSTRAT `.snp` parser. It returns a `SnpTable` of parallel arrays (`id`, `chrom`, `genpos_morgans`, `ref`, `alt`, `count`) in file order, capped to the first `max_snps` records. It is a host-pure `io`-leaf TU (architecture.md §4): no CUDA, no `core`/`device` dependency, surfaces I/O failure as `std::runtime_error`. Its outputs feed two consumers, both correctly *downstream* and never re-implemented here:

- `chrom` + `genpos_morgans` → `core::assign_blocks` / `block_of` (the shared SNP→block rule, architecture.md §2/§8). The header is explicit and correct that the block rule is **not** re-derived in `io` — it surfaces `genpos` in Morgans unchanged and lets `core` own the partition. This is exactly the §2/§8 "block rule lives in `core`, never `io`" invariant, and it is honoured. Good.
- `id`/`ref`/`alt` → `io/filter` (M2 membership key + drop-not-flip allele-class predicates).

The layering is clean and the documentation is unusually good — the header comments correctly cite §4/§5 and explain *why* polarity comes from column 5 and why the block rule is not computed here. The file is small (84 lines of `.cpp`), so the absolute count of findings is moderate, but the unit sits on the **parity-critical SNP→block path** and contains a genuine correctness gap (non-finite / overflowing numeric tokens propagate into a `static_cast<int>` UB downstream), several real edge-case and robustness defects, and notable testability and consistency gaps versus its sibling readers. This is **not** a "near-perfect, nothing to say" header; the review length is justified by substantive findings, not padding.

## Score: 6.5/10 — solid, well-documented leaf with one real latent-UB correctness gap, several edge-case holes, and a testability/consistency gap below the 9.5–10 bar

A competent, idiomatic parser with excellent comments and correct layering, but it is not yet at the senior bar: it admits malformed/non-finite/overflowing numeric fields that turn into undefined behavior in the shared block rule, it diverges silently from the oracle's skip semantics and from its sibling readers' validation discipline, it carries no dedicated unit test, and several decisions (multi-char allele truncation, `next_other` underflow, fixed X/Y/MT mapping not sourced from a single-home table) are undocumented hazards. None is catastrophic in the current M1 happy path on clean AADR data, but each is exactly the class of latent defect a 9.5/10 leaf eliminates.

## Findings

### (1) Correctness & bugs

**F1 [HIGH, S, before-M4.5] — Non-finite / overflowing `genpos` is accepted silently and becomes `static_cast<int>(NaN/Inf)` UB in the shared block rule.**
Location: `read_snp`, `snp_reader.cpp:62,65,75`.
`genpos` is parsed via `ls >> genpos` into a `double`. Since C++11 the `num_get` facet that backs `operator>>` for floating point follows `strtold` semantics, so it accepts `inf`, `nan`, and even hex-float literals like `0x1p3` as "successful" extractions, and on magnitude overflow it stores `±HUGE_VAL` (infinity) *and sets `failbit`* — but here the overflow case is masked: a token like `1e400` in column 3 still succeeds if the remaining fields parse, or it falls into the `ls2 >> id >> chrom_tok >> genpos` fallback which also accepts `inf`/`nan`. (`strtod`/`num_get` floating semantics: [LWG 2381](https://cplusplus.github.io/LWG/issue2381), [strtod overflow → HUGE_VAL](https://en.cppreference.com/w/c/string/byte/strtof).) A `nan`/`inf` `genpos_morgans` then flows unguarded into `core::block_of` → `std::floor(NaN / bs)` = NaN → `static_cast<int>(NaN)`, which is **undefined behavior** (conversion of a NaN/out-of-range float to `int`), corrupting the dense block ids the jackknife depends on bit-for-bit (verified: `block_partition_rule.cpp:39` calls `block_of` then `block_partition_rule.cpp:51` does `static_cast<int>(global)` keyed off the poisoned `local`). This is the parity-critical path (architecture.md §12 "the block id is a pure, launch-order-independent function of (chrom, genpos)") — silent corruption here is exactly the §2 fail-fast violation the spec forbids ("not as silent corruption … three stages later").
Why it matters: real `.snp` files can carry sentinel/garbage genetic positions (e.g. `-9`, blank, or text); a single bad row UB-poisons the whole partition. The fail-fast doctrine (architecture.md §2 "fail-fast", §10 `STEPPE_ERR_IO_FORMAT` "malformed/unsupported `.snp`") says malformed input must surface immediately with context, not corrupt downstream.
Fix: after extraction, reject `!std::isfinite(genpos)` with a `std::runtime_error("io::read_snp: non-finite genetic position at SNP <s> (\"" + line + "\")")`. (Negative finite positions are legitimate and intentionally handled by the block rule — see `block_partition_rule.cpp:45` and ROADMAP §3 M3 "negative chr17 positions … handled" — so reject only non-finite, not negative.)

**F2 [MED, S, before-M4.5] — `std::stoi` in `chrom_code` can throw `std::out_of_range`/`std::invalid_argument`, escaping as an opaque exception rather than a contextual `io::read_snp` error.**
Location: `chrom_code`, `snp_reader.cpp:34-36`.
A chromosome token that is *all digits* but exceeds `INT_MAX` (e.g. `99999999999`) makes `std::stoi` throw `std::out_of_range`. That exception propagates out of `read_snp` *uncaught and uncontextualized* — the caller gets a bare `std::out_of_range: stoi` with no file/line/SNP context, violating the §10 contract that `io` malformed-input failures carry context (`io::read_snp: …`). It also means a single absurd chromosome label crashes a 100k-SNP parse with a message that does not even name the file. `std::stoi` throws `std::out_of_range` on overflow and `std::invalid_argument` on no conversion ([cppreference std::stoi](https://en.cppreference.com/cpp/string/basic_string/stol)).
Why it matters: §2 fail-fast + §10 "caller fixes input" require a *diagnosable* `STEPPE_ERR_IO_FORMAT`-shaped failure, not a bare std-lib throw. Chromosome codes are O(1e2) at most; an all-digit overflow is malformed data.
Fix: use `std::from_chars` (no exceptions, no locale, no allocation) and on failure either throw a contextual `io::read_snp` error or route the token through the `other_codes` sentinel path. `std::from_chars` is also faster and avoids the locale dependency `std::stoi` carries.

**F3 [LOW, S, no] — `chrom_code` numeric detection rejects a leading `+`/`-`, silently sentinel-mapping a legitimately signed code.**
Location: `chrom_code`, `snp_reader.cpp:30-36`.
The `numeric` check requires *every* character to be `isdigit`, so `-1`, `+1` are treated as non-numeric and fall through to the negative-sentinel branch. EIGENSTRAT chromosome codes are conventionally unsigned small integers, so this is unlikely on AADR, but it is a silent semantic surprise: a chromosome literally labeled `-1` would be assigned a *distinct* sentinel rather than the integer `-1`, and since the sentinel allocator itself hands out `-1, -2, …`, a real `-1` chromosome and the first sentinel would conceptually overlap (avoided today only because the all-digit test fails first, never reaching `stoi("-1")`). Only adjacent-equality matters to the block rule (`block_partition_rule.hpp:115` "only equality between adjacent SNPs matters"), so this is low-severity, but it is an undocumented corner.
Fix: either document explicitly that chromosome tokens are unsigned-only by EIGENSTRAT convention, or parse with `std::from_chars` allowing a sign and keep the sentinel space strictly disjoint (e.g. start sentinels at `INT_MIN/2`).

**F4 [LOW, S, no] — `next_other` underflow is unguarded.**
Location: `chrom_code`, `snp_reader.cpp:42`.
`const int code = next_other--;` allocates a fresh sentinel per distinct non-numeric label, decrementing `next_other` from `-1` downward. With a pathological file of >~2.1 billion *distinct* non-numeric chromosome labels this underflows past `INT_MIN` (signed overflow, UB). This is purely theoretical for `.snp` (chromosome cardinality is tiny), so it is low severity, but a 9.5/10 leaf either bounds it or documents the assumption. The more realistic value of a bound: a huge distinct-label count is a strong signal of a mis-parsed file and should be surfaced, not silently absorbed.
Fix: assert/throw if distinct non-numeric labels exceed a small documented bound (a mis-parsed file is the real cause), or document the assumption that non-numeric chromosome cardinality is small.

### (2) Edge cases & failure modes

**F5 [MED, S, before-M4.5] — Short/blank-line skip semantics are silently lenient and can desynchronize the SNP axis from the `.geno`, breaking the "same prefix the oracle decodes" contract.**
Location: `read_snp`, `snp_reader.cpp:59,65-72`.
The header (`snp_reader.hpp:48-50`) promises: "the first `max_snps` records in file order … the SAME prefix the oracle decodes." But the skip rule here is *more lenient* than any axis-preserving policy. The reader: (a) on a 6-field failure, retries with a 3-field read and `continue`s (skips the row) only if even `id chrom genpos` fail; (b) a blank or `<3`-field line is skipped **without advancing `count`**. Crucially, the canonical oracle `aadr/build_tgeno_matrix.py` does **not read `.snp` at all** (it derives Q/V/N from `.geno`+`.ind`), and `experiments/aadr/02_build_matrix.py`'s SNP cap counts *records by `.geno` chunk index*, not by `.snp` line filtering. So the `read_snp` "prefix" is defined by *its own* skip policy, which can desync the SNP axis from the `.geno`/oracle axis: if a `.snp` line is skipped (blank/short) but the corresponding `.geno` record is not, every subsequent SNP's metadata is shifted by one row relative to its genotype — a silent off-by-N. The equivalence test only catches this incidentally, by asserting `snptab.count == M` *after the fact* (`test_f2_blocks_equivalence.cu:209`), not that no interior row was skipped.
Why it matters: the SNP→block partition and the ref/alt polarity are positional; a skipped `.snp` line with no corresponding skipped `.geno` row corrupts the alignment without any error. The header's "SAME prefix" claim is therefore not actually enforced.
Fix: decide and document one policy. The safest for parity: do **not** silently skip interior lines — a malformed/short interior `.snp` line is a format error (`STEPPE_ERR_IO_FORMAT`), because the `.snp` row index *is* the SNP index. Only a trailing blank line (common) should be tolerated, and even then it must not consume a SNP slot. At minimum, count skipped lines and log a WARN with the line number.

**F6 [MED, S, before-M4.5] — `max_snps == 0` silently yields an empty table; no validation, no log.**
Location: `read_snp`, `snp_reader.cpp:59`.
`while (t.count < max_snps && …)` with `max_snps == 0` reads nothing and returns `count == 0`. The header documents `SIZE_MAX` for "every SNP" but says nothing about `0`. An empty `SnpTable` then flows into `assign_blocks`, which returns `n_block == 0` (`block_partition_rule.cpp:25-26`) — a degenerate jackknife with zero blocks, which downstream f2/jackknife code is unlikely to expect. This is a fail-fast gap: `max_snps == 0` is almost certainly a caller bug (an uninitialized cap), and §2 says invalid input surfaces immediately.
Why it matters: silent zero-SNP runs waste a full pipeline pass and produce meaningless output; the equivalence test's `count == M` check would catch `M == 0` only if `M` were also 0.
Fix: either treat `max_snps == 0` as a programming error (throw with context) or document it explicitly as "read nothing." Given the file's fail-fast posture, throwing is more consistent.

**F7 [MED, S, before-M4.5] — File opens but yields zero valid records → returns an empty table instead of erroring, unlike the sibling `ind_reader`.**
Location: `read_snp`, `snp_reader.cpp:49-80`.
`ind_reader` throws `io::read_ind: no individuals parsed` when a file opened but produced no groups (`ind_reader.cpp:74-76`), and `geno_reader` throws on `no complete records on disk` (`geno_reader.cpp:59-61`). `read_snp` has **no equivalent**: an existent-but-empty `.snp`, or one whose every line is blank/short, returns `SnpTable{count=0}` with no error. This is an inconsistency with the unit's own siblings and a fail-fast violation for a clearly-malformed input. The header even promises to "Throw … on a missing/unreadable file" but is silent on the empty-but-present case.
Fix: after the loop, if `max_snps > 0 && t.count == 0`, throw `io::read_snp: no SNP records parsed from <path>`, mirroring `ind_reader`.

**F8 [LOW, S, no] — Multi-character alleles (indels) are truncated to their first char with no signal.**
Location: `read_snp`, `snp_reader.cpp:76-77`.
`t.ref.push_back(ref_tok.empty() ? 'N' : ref_tok[0]);` keeps only the first character. EIGENSTRAT `.snp` is single-char by spec, so on conforming AADR this is fine, but a multiallelic/indel record (`ref_tok == "AT"`) is silently truncated to `'A'`, which then mis-classifies in the filter's allele-class predicates (`filter_decision.hpp:120-159`: a truncated `'A'` looks like a clean base instead of being dropped as multiallelic). The drop-not-flip contract (architecture.md §1) wants indels/multiallelics *dropped*, and a truncation can defeat that. Low severity because the HO panel is biallelic single-char, but it is a silent data-corruption corner.
Fix: if `ref_tok.size() > 1 || alt_tok.size() > 1`, store a sentinel (`'\0'`) that the filter reliably drops, or document that multi-char alleles are out of scope and truncated. `'\0'` is cleanest: `filter_decision::normalize_allele` already maps non-ACGT to `'\0'` and the multiallelic predicate drops it.

**F9 [LOW, S, no] — `physpos` is read into a `std::string` purely to consume the column, then discarded — a wasted allocation per SNP and a confusing "unused variable" pattern.**
Location: `read_snp`, `snp_reader.cpp:63,65,69`.
`std::string physpos;` is extracted only to advance past column 4. Over 100k–584k SNPs this is 100k+ short-string allocations (most fit SSO, so cheap, but non-zero) for a value that is never stored. It also obscures intent. Minor.
Fix: extract into a throwaway with a comment, or skip the field with a `std::from_chars`-based field walker (see F14). Since no downstream needs physical position today, discarding it is correct — only *how* it is discarded is the nit.

**F10 [LOW, S, no] — CRLF tolerance is undocumented; trailing data beyond column 6 is silently ignored.**
Location: `read_snp`, `snp_reader.cpp:60,65,76-77`.
`std::getline` keeps a trailing `\r` on a CRLF file. `operator>>` treats `\r` as whitespace, so the last token strips it in the normal path (benign), but this is an implicit assumption worth one line. Likewise a 7th+ field is ignored without comment. Low severity; flagged for completeness.
Fix: document CRLF tolerance, or strip a trailing `\r` from `line` once after `getline`.

### (3) Numerical / precision vs §12

**F11 [LOW, S, no — doc] — The correctly-rounded decimal→`double` parity assumption for `genpos` is load-bearing but undocumented.**
Location: `read_snp`, `snp_reader.cpp:62,75`; header `snp_reader.hpp:41`.
The genetic position feeds `block_of` = `floor(genpos / block_size_morgans)` (`block_partition_rule.hpp:56`). The floor is robust to last-ULP rounding *except* at exact bin boundaries: a position at `0.05` (the default block width) may round to just-below or just-above the boundary, flipping a SNP between blocks. AT2 and the oracle must parse the identical decimal text to the identical `double` for parity. C++ `num_get` uses correctly-rounded `strtold`→`double` ([LWG 2381](https://cplusplus.github.io/LWG/issue2381)); Python `float()` and R `as.numeric` are likewise correctly-rounded IEEE-754, so parity holds — but this is a *load-bearing* §12 determinism assumption that is undocumented here. No numerical *bug*; a future swap to a faster non-correctly-rounded float parser would silently break block-boundary parity.
Fix: add one header line: "genpos is parsed with correctly-rounded decimal→double (`num_get`/`strtold`); block boundaries rely on this matching the oracle/AT2 float parse exactly (§12)."

There is otherwise no Ozaki/native-FP64 surface here — this is a host text parser, no accumulation, no cancellation. The §12 emulated-vs-native discussion is genuinely N/A to the body of this unit.

### (4) CUDA idioms / RAII / streams vs §7

N/A — this is a pure host `io`-leaf TU with no CUDA, no device allocation, no streams, no launches (architecture.md §4 "io is an isolated leaf"; correctly so). One RAII point worth a nod: the `std::ifstream in(path)` is RAII-closed on every return path including the throw, which is correct. No findings.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**F12 [MED, S, before-M4.5] — The X/Y/MT→23/24/90 chromosome map and the sentinel base (`-1`) are hardcoded literals in `chrom_code`, not promoted to a single-home `io` format-constants table.**
Location: `chrom_code`, `snp_reader.cpp:37-42`.
ROADMAP §4 is unambiguous: "No literal may survive … except true mathematical constants." `23`, `24`, `90`, and the sentinel seed `-1` are **EIGENSTRAT format conventions**, not mathematical constants — exactly the class of literal the inventory says belongs in "`io` format constants" (the same table lists "TGENO header 48, ceil(nsnp/4)" → promoted to `eigenstrat_format.hpp`). The sibling `eigenstrat_format.hpp` is the established single home for EIGENSTRAT literals (`kGenoHeaderBytes`, `kCodesPerByte`, `kMissingCode`, `code_in_byte`), yet the `.snp` chromosome-label conventions are spelled inline here. This is a real DRY/§4 violation: a second file that needs the same X/Y/MT mapping (a future merge/harmonize stage, or autosome-only filtering at `io/filter`) would re-spell `23/24/90`.
Why it matters: the autosome-only filter (architecture.md §5 S0′, ROADMAP M2; the 757-vs-719 block counts in ROADMAP §3 M3 hinge on chr 1–22 vs 1–24) must know that 23=X, 24=Y to drop sex chromosomes — and that mapping lives *only* here, so the filter either re-derives it (the named smell) or hardcodes it again.
Fix: promote to `eigenstrat_format.hpp` as named constants (`kChromCodeX = 23`, `kChromCodeY = 24`, `kChromCodeMT = 90`, `kChromSentinelBase = -1`) and a small free `chrom_code(...)` (or constexpr lookup), so the `.snp` reader and the autosome filter share one source. This also makes the mapping unit-testable in isolation (F20).

**F13 [LOW, S, no] — The default-allele `'N'` literal appears four times un-named.**
Location: `read_snp`, `snp_reader.cpp:70,71,76,77`.
`'N'` (the EIGENSTRAT/IUPAC "missing/unknown base") is repeated as a bare char literal and pairs with the filter's `normalize_allele`, which already treats `'N'` specially (`filter_decision.hpp:123-125`). A named `kUnknownAllele = 'N'` in `eigenstrat_format.hpp` makes the cross-unit contract explicit. Low severity.
Fix: name it once in the format header.

### (6) Decomposition / single-responsibility / function size vs §2

`read_snp` is ~30 lines, single-purpose, readable; `chrom_code` is ~18 lines, single-purpose. Decomposition is fine. One mild observation:

**F14 [LOW, S, no] — The parse loop's missing-allele fallback re-tokenizes the line, duplicating the field-list prefix.**
Location: `read_snp`, `snp_reader.cpp:60-78`.
The two `istringstream` constructions (`ls`, then `ls2` re-reading the same `line`) duplicate the `id >> chrom_tok >> genpos` prefix — a small DRY wrinkle within the function. It works, but a single tokenization into an array of fields (or a `std::from_chars` field-walker) would express the "≥3 fields required, alleles optional" rule once. Folds naturally into the F2/F9 `from_chars` cleanup.
Fix: tokenize once; branch on token count.

### (7) Readability, naming, const-correctness, `[[nodiscard]]`/`noexcept`, comments

- `read_snp` is correctly `[[nodiscard]]` (header `snp_reader.hpp:59`). Good.
- **F15a [LOW, S, no]** — `chrom_code` (anonymous-namespace helper) is **not** `[[nodiscard]]`; per the codebase's own discipline (`eigenstrat_format.hpp` marks every pure accessor `[[nodiscard]]`) it should be. It is correctly *not* `noexcept` today because `std::stoi` can throw (F2); after the F2 `from_chars` switch it could become `noexcept`.
- Naming is good and consistent with siblings (`chrom_tok`, `ref_tok`, `other_codes`, `next_other`).
- Const-correctness: `const int code` (line 42) is good; `genpos`/`id`/etc. are necessarily mutable for `>>`. Fine.
- **F15b [LOW, S, no]** — comment density: the `.cpp` body is *lighter* than its siblings (`ind_reader.cpp`, `geno_reader.cpp` annotate nearly every block; `read_snp`'s loop has two short comments). The mandate is "comment density matching surrounding code" — the lenient skip rule (F5), the truncation (F8), and the `physpos`-discard (F9) each deserve a one-line rationale they currently lack.

### (8) Performance

This is a one-time `.snp` parse (~100k–584k lines), dwarfed by the genotype stream, so performance is low-priority. Two minor notes:

**F16 [LOW, S, no] — Per-line `std::istringstream` construction + per-SNP `std::string` allocations.**
Location: `read_snp`, `snp_reader.cpp:60-64`.
Constructing an `istringstream` per line and extracting four `std::string`s allocates repeatedly. For 584k SNPs this is measurable wall-clock (tens of ms), still negligible vs the genotype pass, but a `std::from_chars`/`std::string_view` field-walker over the `getline` buffer would eliminate nearly all of it and also fix F2 (no-throw integer parse) and F9 (no `physpos` allocation) in one move. Recommended *because* it kills two correctness findings, not for the speed.

**F17 [LOW, S, no] — `SnpTable` vectors are not `reserve`d.**
Location: `read_snp`, `snp_reader.cpp:55,73-77`.
When `max_snps != SIZE_MAX` the final size is bounded by `max_snps`; the five vectors grow by reallocation. The caller almost always knows the count (the equivalence tests pass `M`).
Fix: `reserve(max_snps)` when `max_snps` is a sane finite bound (guard against `reserve(SIZE_MAX)`).

### (9) Layering / API / ABI vs §4

Clean. `io` leaf, no upward dependency, no CUDA, exceptions surfaced to the `app`/test caller (matches `ind_reader`/`geno_reader`). One API observation:

**F18 [LOW, M, no] — Free-function shape diverges from `geno_reader`'s RAII class, and `SnpTable::count` is a redundant invariant hazard.**
Location: header `snp_reader.hpp:38-59`.
(a) `geno_reader` is a move-only RAII class owning the open stream for tiled reads (`geno_reader.hpp:35`), anticipating the M5 tile loop; `read_snp`/`read_ind` are one-shot free functions. Defensible — `.snp`/`.ind` are small and read whole — but worth a one-line note that the `.snp` reader is deliberately *not* tiled (the M5 loop tiles `.geno`; `.snp` metadata is read once and kept resident, consistent with architecture.md §11.1).
(b) `SnpTable::count` is redundant with `id.size()` (all five vectors must be equal-length). A redundant size field is a latent invariant-violation hazard: a future edit that pushes to `id` but forgets `++t.count` desynchronizes them, and nothing asserts `count == id.size()`. `ind_reader`'s `IndPartition` avoids this (`n_individuals_total` is genuinely *not* `groups.size()`).
Fix: either drop `count` and expose a `size()` accessor returning `id.size()`, or add a debug assert `t.count == t.id.size()` (and equal-length across all five) before returning — which also reinforces the F5 alignment story.

### (10) Testability vs §13

**F19 [MED, M, before-M4.5] — There is no dedicated unit test for `read_snp` or `chrom_code`; coverage is incidental, via the f2/decode/filter equivalence binaries only.**
Location: tests (`tests/reference/test_f2_blocks_equivalence.cu:204`, `test_decode_equivalence.cu:183`, `test_filter_oracle.cu:211`).
architecture.md §13 / ROADMAP §6 require each unit's logic to be unit-tested host-side. `block_partition_rule` has `tests/unit/test_block_partition.cpp`; the filters have `tests/unit/test_filters.cpp`. **`snp_reader` has no `tests/unit/test_snp_reader.cpp`.** The only exercise is reading the real AADR `.snp` inside GPU equivalence tests, which (a) require the 4 GB data + a GPU, (b) only check `count == M` and that block assignment doesn't crash, and (c) never exercise the corners this review flags: non-numeric chromosomes (X/Y/MT and sentinels), missing-allele fallback, blank/short lines, multi-char alleles, `max_snps == 0`, empty file, non-finite genpos. Every finding F1–F10 is invisible to the current tests.
Why it matters: the unit's most defect-prone logic (`chrom_code` mapping, skip semantics, the fallback) is entirely untested in isolation, on a parity-critical path.
Fix: add `tests/unit/test_snp_reader.cpp` (GoogleTest, host-only, tmpfile `.snp` fixtures) covering: numeric pass-through; X/x/Y/y/MT/mt/M → 23/24/90; two distinct unknown labels → two distinct stable sentinels; a 3-field record → `ref/alt == 'N'`; a blank/short interior line (pin the decided F5 policy); `max_snps` truncation; `max_snps == 0` and `SIZE_MAX`; a non-finite genpos (pin the F1 throw); an all-digit overflow chromosome (pin the F2 behavior). This is the single highest-leverage change to move the unit toward the 9.5 bar.

**F20 [LOW, S, no] — `chrom_code` is in an anonymous namespace, so it cannot be unit-tested directly without including the `.cpp`.**
Location: `snp_reader.cpp:22-45`.
Couples to F12: if the X/Y/MT mapping moves to `eigenstrat_format.hpp` as a free function, it becomes directly testable from `test_snp_reader.cpp`. As-is, the chromosome-encoding logic is reachable only through `read_snp` end-to-end.
Fix: promote `chrom_code` (or its core mapping) to the format header per F12; test it directly.

### (11) Capability tiers (PRO-6000 vs budget-5090)

**F21 [LOW, S, no — doc] — Correct that this unit has no GDS/P2P/ncu surface; but the non-tiled "read whole" assumption should be stated so an M5/M7 author does not try to tile `.snp`.**
Location: whole unit; cross-referenced to TODO.md §"Keeping it GPU-dominant" + the `wxz1fiiln` capability-tier table.
The capability tiers (TODO.md: capable PRO-6000 = GDS/cuFile NVMe→VRAM; budget-5090 = POSIX-pread into pinned double-buffer) concern the **`.geno`** ingest, not `.snp` metadata — correctly: `.snp` is small per-SNP metadata read once and kept resident (architecture.md §11.1), so it does **not** sit on the GDS/pinned lever, and it should **not** grow a capability probe. So strictly this is N/A for a *fallback-path change*: there is no GDS/P2P/ncu surface in a text metadata parser, and the capability-tag surface (TODO.md §137) belongs to the `.geno`/compute path. The one architecturally-relevant note: the unit's free-function "read the whole file now" shape bakes in a non-tiled assumption. At whole-genome scale (584k SNPs × five vectors ≈ a few MB) this is fine and intended to stay resident, but the header should say so explicitly.
Fix (doc-only): one header line — ".snp metadata is read whole and kept resident (a few MB at whole-genome scale); it is deliberately NOT tiled — the M5 streaming tiler tiles `.geno`, not this. No GDS/pinned/capability surface here (TODO capability tiers concern `.geno` ingest)."

## Considered & rejected

- **`std::isdigit` undefined behavior on signed `char`.** `chrom_code` already casts: `std::isdigit(static_cast<unsigned char>(c))` (`snp_reader.cpp:32`). This is the correct, documented idiom ([cppreference isdigit](https://en.cppreference.com/cpp/string/byte/isdigit); the cast avoids UB on negative `char`). **Rejected** — the code is right; the cast is present and correct. (The filter's `normalize_allele`/`complement` also cast correctly, so the codebase is disciplined here.)
- **`operator>>` mishandling tab/space-separated columns.** EIGENSTRAT `.snp` is whitespace-separated and the oracle uses `line.split()` (any-whitespace). `operator>>` skips any whitespace run identically. **Rejected** — matches the oracle's `split()` and AT2 conventions.
- **Negative genetic positions being a bug.** `block_partition_rule.cpp:45` and ROADMAP §3 M3 ("negative chr17 positions … handled") deliberately give negative positions their own block via `floor → -1`. `read_snp` correctly passes them through unchanged. **Rejected** — intended; only *non-finite* (F1) is the problem, not *negative*.
- **`SnpTable` SoA layout being cache-unfriendly / poor design.** This is an `io`-leaf "plain data struct" output (architecture.md §4 "produces plain data structs"); SoA is exactly right for the column-wise consumers (`chrom`+`genpos` to the block rule, `id` to the filter membership set, `ref`/`alt` to the allele-class predicates) — each consumer touches one column. **Rejected** — SoA is correct and matches the column-major Q/V/N philosophy.
- **`std::map` (RB-tree) for `other_codes` being slow vs `std::unordered_map`.** `ind_reader` uses `std::map` for `index_of` too; non-numeric-label cardinality is O(1e1), so tree-vs-hash is irrelevant and `std::map`'s deterministic ordering is harmless. **Rejected** — non-issue at this cardinality; consistent with the sibling.
- **`physpos` should be retained in `SnpTable`.** No current consumer needs physical position (the block rule speaks genetic position in Morgans; filters use ref/alt/id). Retaining it would be speculative. **Rejected** — YAGNI; correctly discarded (only F9's *mechanism* is the nit).
- **`read_snp` should return `std::expected` instead of throwing.** architecture.md §10 explicitly says the `io` leaf surfaces I/O failures as **exceptions** to its `app`/test caller and does not use `core`/`device` error types (`std::expected<T,Error>` is internal-only, §8/§16). The sibling readers throw. **Rejected** — throwing is the documented `io`-leaf contract.
- **The equivalence test's `count == M` assert being a sufficient safety net for F5.** It checks only the *final* total, not that interior lines weren't skipped-then-compensated; and it runs only on a GPU with the 4 GB data present. **Rejected as sufficient** — it does not pin the alignment invariant (hence F5/F19).

## What it takes to reach 10/10

1. **F1 (HIGH):** reject non-finite `genpos` with a contextual error — close the `static_cast<int>(NaN/Inf)` UB on the parity-critical block path. *(S, before-M4.5)*
2. **F2 (MED):** replace `std::stoi` with `std::from_chars`; surface overflow/garbage chromosome tokens as a contextual `io::read_snp` error, not a bare `std::out_of_range`. *(S)*
3. **F5 (MED):** decide and document one skip policy; treat interior malformed `.snp` lines as `STEPPE_ERR_IO_FORMAT` (the row index *is* the SNP index), never desync the SNP axis from `.geno`, tolerate only trailing blank lines. *(S)*
4. **F6 + F7 (MED):** fail-fast on `max_snps == 0` and on an opened-but-empty/all-skipped file, mirroring `ind_reader`/`geno_reader`. *(S)*
5. **F12 (MED):** promote `23/24/90` and the sentinel base to named constants + a shared `chrom_code` in `eigenstrat_format.hpp` (single-home EIGENSTRAT conventions; shared with the autosome-only filter). *(S)*
6. **F19 (MED):** add `tests/unit/test_snp_reader.cpp` covering chromosome mapping, the missing-allele fallback, skip semantics, `max_snps` edges, non-finite genpos, and overflow chromosomes — host-only, no GPU/data. *(M)*
7. **F8 (LOW):** handle multi-char alleles deterministically (store `'\0'` so the filter drops them) instead of silent truncation. *(S)*
8. **F11 + F21 (LOW, doc):** add the two header notes — correctly-rounded-float parity assumption (§12), and ".snp is read whole/resident, not tiled; no capability surface." *(S)*
9. **F18 (LOW):** drop or assert the redundant `SnpTable::count == id.size()` invariant; assert equal lengths across the five vectors. *(S–M)*
10. **Polish:** `[[nodiscard]]` on `chrom_code` (F15a), comment-density parity with siblings (F15b), `reserve` (F17), and the `from_chars` field-walker (F16/F14) that subsumes F2/F9.

## Good patterns to keep

- **Layering is exemplary.** The header and TU correctly cite §4/§5, keep the unit a pure host leaf, and — critically — surface `genpos` in Morgans *unchanged* while explicitly refusing to compute block ids (`snp_reader.hpp:8-12`). This is the §2/§8 "block rule lives in `core`, never re-derived in `io`" invariant honoured precisely, with the reasoning written down. The single best thing about the unit.
- **Documentation quality.** The header explains *why* column 5 (ref allele) fixes Q's polarity and ties it to the `views.hpp` Q/V/N contract and the oracle — domain reasoning, not just mechanics. Comment quality on the public surface is well above average.
- **`std::isdigit(static_cast<unsigned char>(c))`** — the correct, UB-free idiom, applied consistently with the filter helpers.
- **Exception-based failure with file context on open** (`io::read_snp: cannot open .snp file: <path>`) matches the `io`-leaf contract (§4/§10) and the sibling readers.
- **SoA `SnpTable`** is the right output shape for the column-wise downstream consumers and matches the codebase's column-major Q/V/N philosophy.
- **The X/Y/MT mapping rationale** ("only adjacent-equality matters to the block rule") is correct and well-explained — the *values* should move to a shared header (F12), but the *reasoning* is sound and worth preserving in the comment.
- **`[[nodiscard]]` on the public `read_snp`** and the clear `SIZE_MAX`-means-all convention in the header.

---
Sources consulted: [LWG 2381 — parsing floating point / `num_get`→`strtold`](https://cplusplus.github.io/LWG/issue2381); [`strtod` overflow → HUGE_VAL + ERANGE](https://en.cppreference.com/w/c/string/byte/strtof); [`std::stoi` throws `std::out_of_range`/`std::invalid_argument`](https://en.cppreference.com/cpp/string/basic_string/stol); [`std::isdigit` UB on negative `char`, cast to `unsigned char`](https://en.cppreference.com/cpp/string/byte/isdigit).
