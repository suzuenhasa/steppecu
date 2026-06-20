# Review findings — src__io__snp_reader

Files: /home/suzunik/steppe/src/io/snp_reader.cpp, /home/suzunik/steppe/src/io/snp_reader.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/io/snp_reader.cpp:160,173-175 — The EIGENSTRAT .snp column layout is encoded as bare literals: `fields.size() < 3` (min `<id> <chrom> <genpos>`), `fields.size() >= 6` (full record), and the column indices `fields[4]`/`fields[5]` (ref/alt). The same `3` and `6` are re-stated inside the error-message text at lines 162-163. Meaning is documented in comments (151-159, 171-172) and the header (61-63), but the values are unnamed. Suggested: name as e.g. `kMinSnpFields = 3` / `kFullSnpFields = 6` (with `kRefAlleleCol = 4` / `kAltAlleleCol = 5`), single-homed near the EIGENSTRAT format constants in eigenstrat_format.hpp, so the count gate and the index access cannot drift.
- [5.1][LOW] src/io/snp_reader.cpp:174-175 — The default allele `'N'` (EIGENSTRAT missing/unknown base) is a bare char literal duplicated on both lines. eigenstrat_format.hpp already single-homes the geno-side `kMissingCode`; the allele-side missing char has no named constant. Suggested: introduce a named `kMissingAllele = 'N'` (alongside the other EIGENSTRAT format constants) and reference it on both lines.
- [5.1][LOW] src/io/snp_reader.cpp:134 — `int next_other = -1;` is the negative-sentinel seed for unknown chromosome labels; the `--1` invariant (codes are distinct negatives, never colliding with the numeric/X/Y/MT codes) is implicit. Suggested: a named seed constant (e.g. `kFirstOtherChromCode = -1`) documenting that other-chrom codes start at -1 and decrement.

## Group 6 — Naming

- [6.1][LOW] src/io/snp_reader.cpp:132 — `SnpTable t;` is the accumulator returned by `read_snp`; the single-letter `t` is referenced repeatedly across the whole 50-line function body (lines 132, 137, 177-182, 184), not as a tight loop counter, so it reads opaquely (e.g. `t.count`, `t.id.push_back`). Suggested: rename to `table` (or `snps`) for the function-scope result object.

## Group 7 — Duplication

- [7.1][LOW] src/io/snp_reader.cpp:59-61,113-115 — The `std::from_chars` setup triple is copy-pasted in `chrom_code` and `parse_genpos`, differing only by the value type (int vs double): both compute `const char* begin = tok.data();`, `const char* end = tok.data() + tok.size();`, then `const auto [ptr, ec] = std::from_chars(begin, end, value);`. Suggested: a small templated helper `template<class T> std::errc parse_full(const std::string& tok, T& out)` returning the errc and the "whole token consumed" check, so both call sites fold to one line.
- [7.2][LOW] src/io/snp_reader.cpp:174-175 — The ref/alt extraction is the same expression repeated, differing only by the column index (4 vs 5): `has_alleles && !fields[4].empty() ? fields[4][0] : 'N'` and `…fields[5]…`. The index `fields[N]` is also evaluated twice within each line (`.empty()` then `[0]`). Suggested: a tiny local lambda `auto allele = [&](std::size_t col){ return has_alleles && !fields[col].empty() ? fields[col][0] : 'N'; };` so `ref = allele(4); alt = allele(5);` (pairs naturally with the kRefAlleleCol/kAltAlleleCol naming suggested in Group 5).

## Group 8 — Comments

- [8.2][LOW] src/io/snp_reader.hpp:56 — The header docstring restates the single-homed chromosome-code literals in prose: `(kChromCodeX=23, kChromCodeY=24, kChromCodeMt=90)`. These values live in eigenstrat_format.hpp (verified: kChromCodeX=23, kChromCodeY=24, kChromCodeMt=90) and currently match, but the doc duplicates the magic numbers — if those constants are ever retuned, this comment silently goes stale (a latent stale-comment risk). Suggested: name the constants in prose without the literal values (e.g. "the EIGENSOFT codes kChromCodeX/Y/Mt single-homed in eigenstrat_format.hpp"), letting the header file remain the one source of truth.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.
