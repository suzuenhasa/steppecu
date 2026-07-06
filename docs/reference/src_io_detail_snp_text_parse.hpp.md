# `snp_text_parse.hpp` reference

## 1. Purpose

`src/io/detail/snp_text_parse.hpp` holds the handful of small host-side functions
that turn the text columns of a SNP marker file into the numbers and codes steppe
stores in memory. A SNP file is the sidecar list that names every genetic marker in
a dataset — one line per marker, with columns for the marker's ID, which chromosome
it sits on, its genetic and physical positions, and the two alleles. Two different
file formats carry exactly this information in whitespace-separated text: EIGENSTRAT
`.snp` files and PLINK `.bim` files. Their columns are in a slightly different order,
but the *parsing* of each individual field — "is this token a valid number?", "what
integer code does chromosome `X` map to?" — is identical.

The reason these parsers live in one header is that steppe reads both formats, and
the two readers must never disagree about what a field means:

- the **`.snp` reader** (`snp_reader.cpp`, whose entry point is `read_snp`), and
- the **`.bim` reader** (`plink_reader.cpp`, whose entry point is `read_bim`).

Before this header existed the field-parsing logic was copy-pasted, byte for byte,
into both of those source files. Copy-paste is exactly the kind of thing that quietly
drifts: someone fixes a rounding or chromosome-code edge case in one copy and forgets
the other, and now the same marker file loads two different ways depending on which
format you came in through. Pulling the logic into one shared header makes that
impossible — there is a single source of truth, and both readers call into it.

Everything here is pure host C++20 with no GPU or CUDA involvement. These are string
and number helpers that run once, on the CPU, while a file is being read off disk;
the genotype data itself is decoded on the GPU elsewhere. Because they are plain
functions over `std::string`, they can be unit-tested directly with no device.

One design detail threads through the whole file: the **reader name** (`"read_snp"`
or `"read_bim"`) is passed in by the caller rather than baked in. That way, when a
malformed field makes a parser throw, the error message names the format the user
actually handed steppe — a `.bim` problem says `io::read_bim:` and a `.snp` problem
says `io::read_snp:` — even though the code raising it is shared. One implementation,
but per-format error strings.

---

## 2. The shared numeric-parse contract (`parse_full`)

```
template <class T>
parse_full(const std::string& tok, T& out) -> bool
```

This is the one place that decides what "this token is a valid number" means for the
whole file, so every field parser agrees on the rule. It hands the token to the
standard library's `std::from_chars` and then applies a strict acceptance test:

```
return ec == std::errc{} && ptr == end;
```

Two conditions both have to hold. `ec == std::errc{}` means the conversion itself
reported no error. `ptr == end` means the conversion consumed the **entire** token,
right up to its last character. That second half is the important, easy-to-miss part:
`std::from_chars` will happily read `12` out of the front of `12abc` and stop, calling
it a success. Requiring `ptr == end` rejects that — a trailing-junk token like `12abc`
or `3.5x` is treated as malformed, not silently truncated to `12` or `3.5`. Only a
token that is a number and *nothing but* a number is accepted.

It is a template over the numeric type `T`, so the same contract serves both integer
fields (chromosome codes, in section 5) and floating-point fields (the two position
columns, in sections 3 and 4). `out` is written only on success; the boolean return
tells the caller whether to trust it. Nothing here throws — `parse_full` just reports
yes-or-no, and each caller decides how strict to be about a "no" (compare sections 3
and 4, which make opposite choices).

---

## 3. Parsing the genetic position — strict (`parse_genpos`)

```
parse_genpos(const std::string& tok, std::size_t line_no,
             const std::string& reader_name) -> double
```

Reads the **genetic position** column — the marker's location measured in Morgans, a
unit of recombination distance rather than raw base pairs. This one is strict: if the
token is not a clean finite number, the file is considered malformed and the read
aborts.

```
if (!parse_full(tok, value) || !std::isfinite(value)) throw ...
```

It leans on `parse_full` (section 2) for the "is it a number, all of it?" check, and
then adds `std::isfinite` on top to also reject `inf` and `nan` — tokens that
`from_chars` will parse as legitimate `double`s but which are meaningless as a genetic
coordinate. On failure it throws a `std::runtime_error` whose message splices in the
`reader_name`, the offending token, and the 1-based `line_no`, e.g.

```
io::read_bim: malformed genetic position "abc" at line 42
```

so the user gets told exactly which line of which format to go look at. The genetic
position is treated strictly because it is a real coordinate steppe keeps and can use
downstream; a garbage value there is a genuine problem with the input, not something
to paper over.

---

## 4. Parsing the physical position — lenient (`parse_physpos`)

```
parse_physpos(const std::string& tok) -> double
```

Reads the **physical position** column — the marker's location in base pairs along
the chromosome. This is the deliberate opposite of section 3: instead of throwing on a
bad token, it quietly degrades to zero.

```
if (!parse_full(tok, value) || !std::isfinite(value)) return 0.0;
```

Same underlying test — a fully-consumed, finite number — but a failed test returns
`0.0` rather than raising. That is why it needs neither a `line_no` nor a
`reader_name`: it never produces an error message, so it has nothing to name. The
lenience is intentional. Real-world `.snp` and `.bim` files in the wild sometimes
carry a placeholder or junk physical position, and steppe does not want a cosmetic
oddity in a column it treats as informational to sink the entire load. So a broken
physical position becomes `0`, the record is still accepted, and the read carries on.
The contrast between this function and `parse_genpos` is the whole point: two fields,
two policies, one shared numeric core.

---

## 5. Mapping a chromosome name to an integer code (`chrom_code`)

```
chrom_code(const std::string& tok,
           std::map<std::string, int>& other_codes,
           int& next_other) -> int
```

Turns the chromosome column — which is text, and can hold `1`, `22`, `X`, `MT`, or an
arbitrary contig name — into the single integer code steppe stores per marker. The
constants it maps the special names to (`kChromCodeX = 23`, `kChromCodeY = 24`,
`kChromCodeMt = 90`) live in `io/eigenstrat_format.hpp`, which is why this header
includes it. The logic runs in order:

1. **All-digit tokens** are parsed straight through as their integer value. `"1"`
   becomes `1`, `"22"` becomes `22`. It first checks every character is a digit, then
   uses `parse_full` (section 2) to do the actual conversion, so an autosome number
   maps to itself.
2. **The named sex and mitochondrial chromosomes** get their canonical codes, matching
   the parity conventions[^at2]: `X`/`x` → 23, `Y`/`y` → 24, and `MT`/`mt`/`M` → 90.
3. **Anything else** — an unrecognized contig or scaffold name — is assigned a fresh
   **negative** code on a first-seen basis. The `other_codes` map remembers which name
   got which code, and `next_other` is the running counter that hands out the next one.

That last case is why `other_codes` and `next_other` are passed **in by reference**
rather than being local. The caller (the reader) creates them once at the top of a
file — starting `next_other` at `kFirstOtherChromCode = -1` — and threads the same
pair through every line. So within one file, the first unknown contig gets `-1`, the
second gets `-2`, and so on (`next_other--`), and a name that reappears later gets the
*same* code it got the first time, via the map lookup. The state has to persist across
lines, and living in the caller is what makes that stable and per-file. Numeric and
named chromosomes take the positive space, novel contigs take the negatives, and the
two can never collide.

---

## 6. Splitting a record into tokens (`split_ws`)

```
split_ws(const std::string& line) -> std::vector<std::string>
```

The first step both readers take on every line: chop one raw text record into its
whitespace-separated fields. It leans on the standard library's stream extraction
(`istringstream` with `>>`), which already does the tedious-to-get-right work —
skipping any run of leading, trailing, or interior spaces or tabs, and never emitting
empty tokens between them. The result is a clean `std::vector<std::string>` of just the
fields, which the caller then indexes by column.

Keeping this one function shared means both formats tokenize identically. A `.snp` and
a `.bim` line that are separated by the same whitespace will always split into the same
fields, so any difference between the two formats comes purely from which column means
what — decided by the caller — and never from a subtle difference in how the line was
carved up.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
