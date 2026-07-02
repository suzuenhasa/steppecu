# `snp_reader.hpp` reference

## 1. Purpose

`src/io/snp_reader.hpp` declares the reader for the EIGENSTRAT `.snp` file. That
file lists one record per genetic variant (one "SNP" — single-nucleotide
polymorphism), and this reader turns each record into a few pieces of metadata the
rest of steppe needs: the SNP's id, its chromosome, its position along the genetic
map, its physical position, and which two alleles it has.

The header is deliberately small and self-contained. It is pure host-side C++ (no
CUDA, no GPU code) and depends on nothing outside the C++ standard library, so any
part of the codebase can include it cheaply. It declares one data type
(`SnpTable`) and one function (`read_snp`), both in the `steppe::io` namespace.

An important design line: this reader **surfaces** the raw values from the file and
stops there. It does not compute jackknife block ids, does not re-orient alleles,
and does not apply any filtering. Anything that interprets these values (the block
rule, the include/exclude filters, the genotype math) lives elsewhere and reads
from the arrays this file produces. Keeping that interpretation out of the reader is
intentional — re-deriving, for example, the block-partitioning rule here would
duplicate logic that must stay in exactly one place.

---

## 2. The `.snp` file format

A `.snp` file has one whitespace-separated record per SNP. A full record has six
columns:

```
<snp-id>  <chromosome>  <genetic-pos-Morgans>  <physical-pos>  <ref>  <alt>
```

For example: `rs3094315  1  0.020130  752566  G  A`.

Two columns carry meaning that is easy to get wrong, so they are worth spelling out:

### Column 3 — genetic position, in Morgans

Column 3 is the position of the SNP along the genetic linkage map, expressed in
**Morgans** (a unit of recombination distance). The reader stores this value exactly
as read, with no unit conversion. It is consumed directly by the shared
block-partitioning rule, which also speaks in Morgans. Because the reader hands the
value through unchanged, block boundaries computed downstream line up exactly with
the reference tooling.

### Column 5 — the reference allele, which fixes allele-frequency polarity

Column 5 is the **reference** allele and column 6 is the **alternate** allele. This
matters beyond bookkeeping: the companion `.geno` file's 2-bit genotype codes count
copies of the *reference* allele. So the reference allele is what fixes the polarity
of every allele frequency steppe computes — the frequency `Q` is defined as the
reference-allele frequency. That definition is shared across all populations with no
per-population re-polarization, which is why the reference allele from this column is
the single anchor for that convention.

---

## 3. `SnpTable` — the per-SNP metadata

`SnpTable` holds the parsed metadata as a set of **parallel arrays in file order**.
That is, SNP number `s` is element `s` of every vector, and all the vectors have the
same length. The `count` field records that shared length (the number of SNPs read).

| Field | Type | Meaning |
|---|---|---|
| `id` | `vector<string>` | The SNP id from column 1 (for example an rs-number). This is the key that the include/exclude filters and any pruned-SNP membership list match against. |
| `chrom` | `vector<int>` | The chromosome code per SNP, in file order. Only whether two adjacent SNPs share the same code matters to the block rule — see the parsing notes below. |
| `genpos_morgans` | `vector<double>` | The genetic position from column 3, in Morgans, exactly as read. Fed to the shared block-partitioning rule. |
| `physpos` | `vector<double>` | The physical position from column 4, in base pairs, exactly as read (0 when the column is absent). This is the fallback axis the block rule uses only when the whole genetic map is zero. |
| `ref` | `vector<char>` | The single-character reference allele (column 5) — the one that fixes allele-frequency polarity. |
| `alt` | `vector<char>` | The single-character alternate allele (column 6). |
| `count` | `size_t` | The number of SNPs read; equals the length of every array above. Defaults to 0. |

---

## 4. `read_snp` — parsing the file

```cpp
[[nodiscard]] SnpTable read_snp(const std::string& path, std::size_t max_snps);
```

`read_snp` opens the `.snp` at `path`, reads records in file order, and returns a
filled `SnpTable`. It is marked `[[nodiscard]]` because the returned table is the
entire point of calling it.

### The `max_snps` cap

`max_snps` limits how many records are read: the function reads the first `max_snps`
records in file order and stops. Passing `SIZE_MAX` reads every SNP. Reading a
prefix this way is used, for example, to read the first 100,000 SNPs for a smaller
run — and because it is always the *first* N records in file order, it reads exactly
the same prefix that the reference decoder does, so the two stay aligned.

### Chromosome codes

Chromosome codes are turned into `int`s carefully so that parsing never throws:

- A numeric code is parsed as an integer using a locale-free character-to-number
  conversion that reports failure by return value rather than by throwing.
- The common non-numeric labels `X`, `Y`, and `MT` are mapped to the fixed
  EIGENSTRAT numeric codes for those chromosomes (the code constants live in one
  place, the EIGENSTRAT-format header).
- Any *other* non-numeric label — and any all-digit code too large to fit in an
  `int` (an overflow) — is mapped to a stable negative sentinel, one distinct
  sentinel value per distinct label. It is never left to throw.

This is safe because the block rule only ever asks whether two adjacent SNPs are on
the *same* chromosome. It never needs the code's true numeric identity, so a
sentinel for an unusual label is perfectly correct: it compares equal to itself and
unequal to everything else.

### Record shape (classification by token count)

Each record is classified by how many whitespace-separated tokens it has:

- A record with **fewer than 3** tokens is malformed (a valid record needs at least
  `<id> <chrom> <genpos>`).
- A record with a **full 6** tokens carries explicit reference and alternate alleles
  (columns 5 and 6).
- Anything in between (3 to 5 tokens) is accepted, but the reference and alternate
  alleles default to `'N'` because they were not supplied.

### Genetic position and the finite guard

The genetic position (column 3) is parsed with the same locale-free,
correctly-rounded decimal-to-`double` conversion used for chromosomes, consuming the
whole token. Using a correctly-rounded parser is what makes downstream block
boundaries match the reference tooling exactly.

There is one extra guard that is easy to overlook. That parser will happily accept
the text `"inf"` and `"nan"` as valid floating-point values. If such a value slipped
through, it would later reach a cast to `int` inside the block-assignment step, which
is undefined behavior for a non-finite `double`. So after parsing, the genetic
position is explicitly checked with an is-finite test, and a non-finite value is
rejected as a malformed record rather than trusted to be caught by the parser.

---

## 5. Error handling and fail-fast behavior

`read_snp` throws `std::runtime_error` rather than silently returning partial or
skewed data. It throws on:

- a **missing or unreadable file**;
- a **malformed record** — fewer than 3 tokens, or a non-finite / unparseable
  genetic position;
- an **interior blank line**.

The interior-blank-line rule deserves its reasoning. The row index in the `.snp`
file *is* the SNP index — row `s` corresponds to element `s` of the genotype data in
the companion `.geno` file. If the reader silently skipped a blank row, every SNP
after it would shift by one and the SNP axis would no longer line up with the
genotypes. Rather than risk that silent desync, an interior blank line is a hard
error.

Every diagnostic message includes the **1-based line number** where the problem
occurred, so a bad record is easy to locate. As a practical convenience, a single
trailing blank line at end-of-file is tolerated (many editors add one), and does not
count as an interior blank line.
