# Review findings — src__io__eigenstrat_format

Files: /home/suzunik/steppe/src/io/eigenstrat_format.cpp, /home/suzunik/steppe/src/io/eigenstrat_format.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

- [5.3][LOW] src/io/eigenstrat_format.cpp:49,52,84,86 — the literal `2` (number of decimal counts to parse from the header: n_ind, n_snp) is repeated as the `ints[2]` array dimension (49), the `got < 2` parse-loop bound (52), the index into `ints` (84), and the `got < 2` underflow guard (86). All four must move together; drift between the array size and the loop bound is an out-of-bounds write. Suggested: name it once, e.g. `constexpr std::size_t kHeaderCounts = 2;`, and size the array and bound the loop from it.
- [5.1][LOW] src/io/eigenstrat_format.hpp:106 — the low-2-bits mask `0x3u` in `code_in_byte` is a bare literal that re-encodes the 2-bit packing width already named by `kBitsPerCode`. It is the structural value `(1u << kBitsPerCode) - 1`, so a change to `kBitsPerCode` would silently desync the mask. Suggested: derive it (`(1u << kBitsPerCode) - 1u`) or add a `kCodeMask` constant beside `kBitsPerCode`.
- [5.4][LOW] src/io/eigenstrat_format.cpp:40,42 — the magic strings `"TGENO"` / `"GENO"` are inline string literals at the parse site rather than named format constants in the hpp (which otherwise centralizes every other packed-format literal: kGenoHeaderBytes, kCodesPerByte, etc.). Not a correctness risk (single parse site), but inconsistent with the file's stated "single home for format literals" policy. Suggested: promote to `inline constexpr std::string_view kMagicTgeno/kMagicGeno` in eigenstrat_format.hpp.

## Group 6 — Naming

- [6.1][LOW] src/io/eigenstrat_format.cpp:49,84,91-92 — the parsed-counts buffer is named `ints` (a generic "two integers" name) when it specifically holds the two header counts (n_ind, n_snp); the meaning only becomes clear at the `h.n_ind = ints[0]; h.n_snp = ints[1];` assignment 40 lines below the declaration, and `ints[got++]` reads as a positional unknown. Suggested: rename to `counts` (or `header_counts`), matching the `n_ind, n_snp` it stores.

## Group 7 — Duplication

- [7.3][LOW] src/io/eigenstrat_format.hpp:96-97 — `static_cast<std::size_t>(kCodesPerByte)` is spelled twice in `packed_bytes` (once in the `+ ... - 1` numerator, once as the divisor). The widened denominator is loop-invariant relative to itself; the duplicate cast is pure hygiene. Suggested: hoist `constexpr std::size_t cpb = kCodesPerByte;` once (or make `kCodesPerByte` a `std::size_t` constant) and use it in both spots.
- [7.4][LOW] src/io/eigenstrat_format.cpp:81-82,87-88 — the `h.format = GenoFormat::Unknown; return h;` malformed-header exit is duplicated verbatim in the overflow guard (81-82) and the `got < 2` guard (87-88). Two early returns (37,45) instead just `return h;` on the still-default-Unknown `h`, so the explicit-set form is the odd one out. Suggested: fold the explicit set+return into a tiny local helper/lambda (e.g. `auto fail = [&]{ h.format = GenoFormat::Unknown; return h; };`) or rely on the default-Unknown `h` and `return h;` at both sites.

## Group 8 — Comments

- [8.2][MED] src/io/eigenstrat_format.cpp:34-35 — stale/inaccurate comment: "A trailing-substring check keeps it robust to surrounding whitespace; we match the leading token exactly." The code (36-39) skips leading whitespace, isolates the leading token via `find_first_of(" \t")`, and does an exact `==` compare (40,42). There is no "trailing-substring check" anywhere — the comment describes a matching strategy the code does not use. Suggested: rewrite to describe the actual leading-token isolation + exact compare (drop the nonexistent substring check).
- [8.2][MED] src/io/eigenstrat_format.cpp:86 — comment contradicts the code: `// could not read both counts → leave format set but counts 0` claims the format is left set, but line 87 immediately does `h.format = GenoFormat::Unknown;`. The format is NOT left set; the file is routed to Unknown like the other malformed-header exits. Suggested: correct to "→ malformed header, route to Unknown" (matching the 80-82 overflow exit).

## Group 9 — Constants & configuration

No Group 9 issues found.

(9.1: all format constants in the hpp are `inline constexpr` and `packed_bytes`/`code_in_byte` are `constexpr`; the cpp locals that mutate — `text`, `pos`, `i`, `v`, `got`, `ints`, `any`, `overflow`, and the built-up `GenoHeader h` — genuinely require mutation. 9.2: every tunable/format knob is surfaced at the hpp top as a named constant (`kGenoHeaderBytes`, `kCodesPerByte`, `kBitsPerCode`, `kMissingCode`, `kChromCodeX/Y/Mt`) rather than buried in decode logic — this TU is the model the rest of the io layer is told to follow; the one in-logic literal is the count-loop bound `2` at cpp:49/52/84/86, already recorded under 5.3 as a magic-number and structural rather than a config knob. 9.3: `parse_geno_header` takes a single `const std::array&` and there are no positional-boolean parameters or `foo(true,false)` call sites anywhere in the unit.)

## Group 10 — Initialization

No Group 10 issues found.

(10.1: every local in `parse_geno_header` is declared at its first use and initialized at the point of declaration — `text` (cpp:26, reserved then filled immediately), `pos` (cpp:36), `magic_end`/`magic` (cpp:38-39), `ints[2]={0,0}` (cpp:49), `got=0` (cpp:50), `i=magic_end` (cpp:51), and the per-iteration `v=0`/`any=false`/`overflow=false` (cpp:55-57) and `d`/`kMax` (cpp:70-71). No uninitialized-then-assigned locals; nothing declared far from first use. 10.2: `GenoHeader h;` (cpp:22) relies on the struct's in-class member initializers (hpp:124-129: `format=Unknown`, `n_ind/n_snp/n_records/bytes_per_record=0`, `header_bytes=kGenoHeaderBytes`), which are real default-member-initializers — not an unguaranteed zero-init — so the default-Unknown early returns (cpp:37,45) and the `got<2` exit (cpp:86-88) return a fully-initialized header; `ints` is explicitly `{0,0}`, not implicitly zeroed.)
