# `snp_reader.cpp` reference

## 1. Purpose

`src/io/snp_reader.cpp` implements `read_snp`, the parser for an EIGENSTRAT
`.snp` file. That file holds one whitespace-separated record per genetic marker
(SNP), and each record looks like:

```
<snp-id>  <chromosome>  <genetic-pos-Morgans>  <physical-pos>  <ref>  <alt>
```

for example `rs3094315  1  0.020130  752566  G  A`.

The reader walks the file in order, reads at most the first `max_snps` records,
and returns a `SnpTable` of parallel arrays — one entry per SNP, in file order.
It extracts five things per marker: the SNP id, an integer chromosome code, the
genetic position in Morgans exactly as written, the physical position in base
pairs, and the single-character reference and alternate alleles.

Two design ideas run through the whole file:

1. **The row index is the SNP index.** The position of a record in the `.snp`
   file *is* that SNP's index everywhere else in the pipeline (it lines up
   positionally with the genotype file). So the parser can never silently skip a
   record — dropping one row would shift every later SNP's metadata relative to
   its genotype. Anything it cannot make sense of is either a hard error with a
   line number, or (for genuinely optional fields) a well-defined default.

2. **This reader stays in its lane.** It surfaces the genetic position unchanged
   and does *not* compute jackknife block boundaries itself — that job belongs to
   the shared block-partition rule, which reads Morgans directly. Re-deriving the
   block rule here would be a duplication smell.

The file is pure host C++20. It uses no CUDA and depends on nothing in the core
or device layers. The only project header it includes is the format-constants
header, which is where the chromosome codes, column indices, and field-count
thresholds it uses are defined once.

---

## 2. The shared numeric-parse contract (`parse_full`)

Every numeric field in the file is parsed through one small template helper,
`parse_full`. It takes a token string and an output value, and returns true only
when **both** of these hold:

- the standard-library parse (`std::from_chars`) succeeded, and
- the *entire* token was consumed — there is no leftover trailing text.

The "whole token consumed" half is what makes garbage like `0.5x` a failure
rather than a silent success that reads `0.5` and ignores the `x`. The same
helper is overloaded for integers and for doubles, so the chromosome parser and
both position parsers all share exactly one definition of "cleanly parse this
whole token."

### Why `std::from_chars`, not `std::stoi`

This choice is load-bearing, not stylistic. `read_snp` promises to report every
malformed input by throwing `std::runtime_error` with context (the offending
token and its line number). `std::from_chars`:

- is locale-free (it will not interpret `,` or `.` differently on a
  differently-configured machine),
- allocates nothing, and
- **throws nothing** — it reports failure through a returned error code, not an
  exception.

`std::stoi` breaks that promise. On an all-digit token larger than the maximum
`int` (for example `"99999999999"`) it throws `std::out_of_range`. That is a
`logic_error`, not a `runtime_error`, so it would escape `read_snp` uncaught and
without the line-number context the reader guarantees. With `from_chars`, an
overflowing all-digit token instead reports an out-of-range error code, which the
chromosome path can catch and handle gracefully (see section 5).

---

## 3. Genetic position: strict, finite-checked

The genetic position (column 3, in Morgans) is parsed strictly by `parse_genpos`.
It must parse cleanly, consume the whole token, **and** be a finite number.
Anything else throws `std::runtime_error` naming the bad token and the line.

Two rejections come for free from the shared `parse_full` contract: trailing
garbage like `0.5x`, and magnitude overflow like `1e400`.

The finite check is deliberate and explicit — it is *not* left to the parser.
The C++ character-parsing routines are allowed to accept the C-style spellings
`"inf"` and `"nan"`, and the library used to build steppe does accept them: they
parse successfully with the whole token consumed. A non-finite genetic position
must never be allowed downstream, because the block-partition rule computes a
block index with an expression of the form `static_cast<int>(std::floor(genpos /
block_size))`. Feeding a NaN or infinity into that cast is undefined behavior
that would silently corrupt the block partition — and the block partition drives
the uncertainty (jackknife) estimates, so a corrupted partition is a
correctness-critical failure, not a cosmetic one. Rejecting `!std::isfinite`
right after the parse closes that door, and also catches implementations that map
overflow to `±inf` rather than to an out-of-range error.

The reason correctly-rounded decimal-to-double parsing matters here (and is why
`from_chars` is the right tool): the parsed value has to match the reference
implementation bit-for-bit[^at2] so that block boundaries land in exactly the same
places. The library's floating-point `from_chars` backend is correctly rounded,
which gives that exact match.

---

## 4. Physical position: lenient, degrades to zero

The physical position (column 4, in base pairs) is parsed by `parse_physpos`, and
it is intentionally the **opposite** of the strict genetic-position parser: a bad
or non-finite token quietly becomes `0.0` instead of throwing.

The asymmetry is by design. The physical position is optional metadata. It is
only used as a fallback axis for defining jackknife blocks *when the genetic map
is entirely zero* (which is common for modern data derived from VCF or PLINK).
When a real genetic map is present, the physical position is never read at all.
Because it is optional, parsing it must not introduce a new way for a `.snp` file
that reads fine today to suddenly fail. So a garbage or non-finite column-4 token
degrades to `0.0` rather than aborting the whole read.

A `0.0` physical position is harmless: it simply cannot anchor a base-pair block,
and the base-pair fallback only ever fires on a non-degenerate physical axis in
the first place. The value is parsed into a `double`, which represents any real
base-pair coordinate exactly (they are far below the 2^53 integer limit of a
double), and it rides the same double-based path the genetic position already
uses.

---

## 5. Chromosome codes

`chrom_code` turns a chromosome label (the text in column 2) into an integer. The
only thing the downstream block rule cares about is whether two *adjacent* SNPs
sit on the same chromosome — it compares codes for equality and nothing else —
so the code just has to be **stable and distinct per label**, not meaningful.

The mapping works in this order:

| Label | Result |
|---|---|
| An all-digit token that fits in `int` (e.g. `1`, `22`) | Its integer value. |
| `X` / `x` | `kChromCodeX` (`23`) — the standard EIGENSTRAT code. |
| `Y` / `y` | `kChromCodeY` (`24`). |
| `MT` / `mt` / `M` | `kChromCodeMt` (`90`). |
| Any other label, or an all-digit token too large for `int` | A fresh negative sentinel, one per distinct label. |

The X / Y / mitochondrial codes are single-homed in the format-constants header
rather than typed here, because the "autosomes only" filter elsewhere is defined
as "chromosomes 1 through 22" and therefore drops exactly these three codes —
keeping the numbers in one place keeps the two definitions from drifting apart.

The negative-sentinel path handles two cases with one mechanism. Any unrecognized
non-numeric label gets its own distinct negative number, remembered in a small map
so the same label always maps to the same code. And crucially, an all-digit token
that overflows `int` also lands here: the numeric parse reports an out-of-range
error code (it never throws — see section 2), so instead of aborting the read, a
pathological over-long numeric chromosome falls through to a stable, distinct
sentinel. Since only adjacent-equality matters, a sentinel for such a code is
completely correct. The sentinels count down starting from `kFirstOtherChromCode`
(`-1`), so they can never collide with a real numeric chromosome or with the
X/Y/MT codes.

---

## 6. Reading a record: token count decides the columns

`read_snp` opens the file (throwing if it cannot), then loops line by line. For
each non-blank line it splits the line into whitespace-separated tokens with
`split_ws`, and then the **number of tokens decides how to interpret the record**.
This token-count decision is what keeps a malformed line from silently
mis-aligning the SNP axis.

`split_ws` uses stream extraction, which skips any run of whitespace between
tokens. This matches how the reference implementation[^at2] splits a line (on
any-whitespace), so the two agree on where the field boundaries are.

The field-count rules, using the named thresholds from the format-constants
header:

| Token count | Interpretation |
|---|---|
| 0 (blank line) | Handled specially — see section 7. |
| Fewer than `kMinSnpFields` (`3`) | Malformed record — throw with the line number. |
| At least 3, fewer than 6 | Valid minimal record: id, chromosome, genetic position. Physical position defaults to `0.0`; both alleles default to `'N'`. |
| At least `kFullSnpFields` (`6`) | Full record: also read physical position and the explicit reference/alternate alleles. |

The fields are read at fixed column indices, all named constants: physical
position at `kPhysposCol` (`3`), reference allele at `kRefAlleleCol` (`4`),
alternate allele at `kAltAlleleCol` (`5`).

- **Physical position** is read only when the record has more than three tokens;
  otherwise it defaults to `0.0`. It is parsed leniently (section 4).
- **Alleles** are read only from a full six-column record. Otherwise they default
  to `kMissingAllele` (`'N'`), the EIGENSTRAT "missing / unknown base" marker.
  Each allele is the *first character* of its column's token. An empty token also
  falls back to `'N'`. A single shared helper does this first-character extraction
  for both alleles so the column token is looked at exactly once per allele.

A record shorter than three tokens is a hard error, not a skipped row. Skipping
it would drop the row and shift every later SNP's metadata out of alignment with
its genotype — so the reader fails fast with the line number instead.

The reference allele matters beyond just being stored: the genotype file counts
copies of the reference allele, so the reference allele is what fixes the polarity
of the allele-frequency values used later in the pipeline. It is recorded here so
that polarity is consistent across all populations with no per-population
re-orientation.

The loop stops as soon as it has collected `max_snps` records. Passing the
maximum possible size value reads every SNP in the file. This cap reads a prefix
of the file — the same leading records the reference oracle[^at2] decodes when a
cap is in effect.

---

## 7. Blank lines and the SNP-axis invariant

Because a record's row position is its SNP index, a blank line in the middle of
the file is dangerous: if it were silently skipped, every record after it would be
off by one relative to the genotype data. So blank (or whitespace-only) lines are
handled with a deliberate rule:

- A blank line **at end of file** is tolerated — this is just the common trailing
  newline. When the reader hits a blank line, it peeks ahead; if the next thing is
  end-of-file, it stops cleanly.
- A blank line **followed by more content** is a format error. It throws
  `std::runtime_error` with the line number and an explanation that interior blank
  lines would desync the SNP axis from the genotype file.

The line counter is 1-based and counts *every* physical line, including the blank
ones, so the number in any diagnostic points at the real line in the file.

---

## 8. Errors it raises

`read_snp` reports every problem the same way — by throwing `std::runtime_error`
with a message that includes enough context to find the problem. It throws on:

- a missing or unreadable file (the message includes the path);
- a malformed record with fewer than three whitespace-separated fields (the
  message includes the line number and how many fields were found);
- a non-finite or otherwise unparseable genetic position (the message includes the
  offending token and line number);
- an interior blank line (the message includes the line number).

It does **not** throw for a bad physical position (which degrades to `0.0`), for
an unrecognized or overflowing chromosome label (which maps to a stable sentinel),
or for a single trailing blank line at end of file (which is tolerated). Every
thrown message that refers to a location uses the 1-based line number so the
source of a bad file is easy to locate.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
