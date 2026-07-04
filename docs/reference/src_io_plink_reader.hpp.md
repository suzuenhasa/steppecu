# `plink_reader.hpp` reference

## 1. Purpose

`src/io/plink_reader.hpp` declares the two host-side parsers for the metadata
files of a PLINK genotype dataset: `read_bim` (the SNP table) and `read_fam` (the
individual/population table). A PLINK dataset is a triple of files — `.bed` (the
packed 2-bit genotypes), `.bim` (one row per SNP), and `.fam` (one row per
individual). This header covers only the two text sidecar files; the packed `.bed`
genotypes are decoded elsewhere.

These are the PLINK counterparts of the EIGENSTRAT readers `read_snp` and
`read_ind`. They exist as separate functions because **neither PLINK file shares
the EIGENSTRAT column layout**, so the EIGENSTRAT readers cannot be pointed at them:

- The `.bim` has the same six pieces of information as an EIGENSTRAT `.snp`, but
  with the chromosome and SNP-id columns swapped and the two alleles written as
  PLINK's "A1"/"A2" rather than an explicit reference/alternate pair.
- The `.fam` has a completely different six-column layout (family id, individual
  id, parents, sex, phenotype) and encodes the population label in an unexpected
  place — the phenotype column.

Both parsers deliberately return the **exact same struct types** their EIGENSTRAT
twins return (`read_bim` yields an `io::SnpTable`, `read_fam` yields an
`io::IndPartition`), so everything downstream — SNP filtering, jackknife block
partitioning, genotype decoding — consumes a PLINK dataset byte-for-byte
identically to an EIGENSTRAT or PACKEDANCESTRYMAP one. The only work these two
functions do is translate PLINK's layout and conventions into those shared structs.

The header is a leaf of the I/O layer: pure host C++20, no CUDA, and it depends on
nothing in the core or device layers. It pulls in only `ind_reader.hpp` (for the
`IndPartition` and `PopSelection` types that `read_fam` reuses) and `snp_reader.hpp`
(for the `SnpTable` type that `read_bim` produces). Every failure is surfaced as a
`std::runtime_error`.

---

## 2. `read_bim` — the SNP-table parser

```cpp
[[nodiscard]] SnpTable read_bim(const std::string& path, std::size_t max_snps);
```

Parses the `.bim` file at `path`, reading the first `max_snps` records in file
order. Pass `SIZE_MAX` to read every SNP. It returns the same `io::SnpTable` that
`read_snp` produces.

Each `.bim` line is one whitespace-separated record per SNP with six fields:

```
<chromosome>  <snp-id>  <genetic-pos-Morgans>  <physical-pos>  <A1>  <A2>
```

The parser fills the `SnpTable` fields from these columns as follows:

| `SnpTable` field | Comes from | Notes |
|---|---|---|
| `id` | column 2 (the SNP id) | The identifier used to match SNPs across files. |
| `chrom` | column 1 (the chromosome) | Parsed with the same chromosome-code convention `read_snp` uses: a numeric chromosome passes straight through; `X`/`Y`/`MT` map to their standard EIGENSOFT numeric codes; anything unrecognized becomes a stable negative sentinel. |
| `genpos_morgans` | column 3, as read (in Morgans) | Fed directly into the shared jackknife block rule. |
| `ref` | column 5 (PLINK's "A1") | The canonical reference allele — see section 3. |
| `alt` | column 6 (PLINK's "A2") | The alternate allele. |

Note the column swap relative to EIGENSTRAT: in a `.bim` the chromosome is column 1
and the SNP id is column 2, which is the reverse of the EIGENSTRAT `.snp` ordering.

### Error handling

`read_bim` throws `std::runtime_error` on any of:

- a missing or unreadable file;
- a malformed record — fewer than six whitespace-separated fields, or a
  non-finite/garbage genetic position;
- an interior blank line.

The interior-blank-line rule matters because the `.bim` row index **is** the SNP
index that lines up with the `.bed` genotype rows. A silently skipped record would
shift every following SNP out of alignment with its genotypes, so a blank line in
the middle of the file is treated as corruption rather than ignored. A single
trailing blank line at end-of-file is tolerated. Every diagnostic message carries
the 1-based line number of the offending row. This mirrors `read_snp`'s contract
exactly.

---

## 3. Canonical allele polarity (A1 is the reference)

This is the single load-bearing decision in the PLINK reader, so it is worth
stating on its own. steppe reports allele frequencies as the frequency of a
"reference" allele, and every SNP must have a well-defined choice of which of its
two alleles is that reference. For a `.bim`, **the reference is defined to be A1
(column 5) and the alternate to be A2 (column 6)** — the parser sets `ref := A1`
with no per-SNP flipping.

The reason is that the packed `.bed` 2-bit genotype code counts **copies of A1**.
By defining A1 as the reference, the genotype-decoding lookup table produces the
canonical reference-copy count directly, without any per-SNP polarity correction.
A standalone PLINK dataset is fully self-describing through its own `.bim`: there is
no external reference file to reconcile against, so making A1 the reference is
correct by construction rather than a guess.

This choice is pinned by a bit-exact test. When the conversion tool `convertf`[^eigensoft]
writes a PACKEDANCESTRYMAP/EIGENSTRAT dataset out as PLINK, it stores the
EIGENSTRAT reference allele as PLINK's A1. So for every SNP, the PLINK reference
(`:= A1`) equals the reference the EIGENSTRAT/PACKEDANCESTRYMAP reader would pick,
and reading the same underlying data through either path yields identical genotype
codes. Because PLINK is the highest-risk of the plug-in formats for a polarity
mistake, this equality is guarded by a bit-for-bit comparison against the
PACKEDANCESTRYMAP reading of the same data.

---

## 4. `read_fam` — the individual/population parser

```cpp
[[nodiscard]] IndPartition read_fam(const std::string& path,
                                    const PopSelection& sel,
                                    std::size_t n_records_present);
```

Parses the `.fam` file at `path`, applies the requested population selection, and
returns the same `io::IndPartition` that `read_ind` produces: the selected
populations grouped into rows, sorted alphabetically (ascending) by population
label.

Each `.fam` line is one whitespace-separated record per individual with six fields:

```
<FID>  <IID>  <pat>  <mat>  <sex>  <pheno>
```

The population label lives in **column 6 (the phenotype column)** — see section 5
for why, and for how "ignored" individuals are handled. `read_fam` groups the
individual-record indices by that column-6 label and then applies the caller's
`PopSelection` using the same selection logic as `read_ind`. Because the selection
rules and the final ordering are identical, the partition built from a PLINK dataset
matches the partition built from the EIGENSTRAT or PACKEDANCESTRYMAP version of the
same subset — and the parity partition for this `.fam`[^at2].

### The `n_records_present` cap

The `n_records_present` argument caps the individual axis to the number of records
actually present in the `.bed` file. Any `.fam` row at an index at or beyond
`n_records_present` is ignored. Pass the `.bed` individual count to read a partial
file, or `SIZE_MAX` to use every `.fam` row. This is the same partial-file cap
`read_ind` accepts.

### Error handling

`read_fam` throws `std::runtime_error` on a missing or unreadable file, or when the
selection resolves to an empty set. This mirrors `read_ind`'s contract: the I/O leaf
surfaces failures as exceptions rather than returning an empty result.

---

## 5. The population-label convention (column 6)

PLINK's `.fam` has no dedicated population/group column, so the population label is
carried in **column 6, the phenotype field**. `read_fam` interprets that column
exactly the way the parity PLINK reader does[^at2]:

| Column-6 value | Meaning |
|---|---|
| `1` | The individual's group label is `Control`. |
| `2` | The individual's group label is `Case`. |
| `9`, `-9`, or `0` | The individual is **ignored** — excluded from every group and never selectable. |
| any other string | That string **is** the population label. |

In practice, running the `convertf` tool in its PLINK ("PACKEDPED") mode with
`outputgroup: YES` writes the real EIGENSTRAT population label straight into this
column, which is why a converted dataset carries meaningful group names here.

Two other columns are deliberately **not** used for grouping. The family id
(column 1) is only a numeric counter that `convertf` invents, not a population, so
it is ignored. Grouping is driven solely by the column-6 label.

### The `.bed` alignment invariant

The `.bed` individual axis is 1:1 with the `.fam` lines — this is a universal PLINK
invariant. An "ignored" individual (one whose column 6 is `9`, `-9`, or `0`) still
occupies its `.bed` row: it is simply never placed into any group. Keeping that row
reserved rather than compacting it out is what keeps every remaining individual
lined up with its correct `.bed` genotype record.

### Selection semantics reused from `read_ind`

`read_fam` reproduces `read_ind`'s selection behavior exactly, so a caller's
`PopSelection` means the same thing regardless of input format. The three modes are:

- **Explicit** — take a named set of populations.
- **AutoTopK** — take the K largest populations, with a first-appearance tie-break
  when two populations are the same size.
- **MinN** — take every population with at least a minimum number of individuals.

Whichever mode is used, the resulting set is finally sorted ascending by population
label before it is returned, matching `read_ind` exactly.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
[^eigensoft]: **EIGENSOFT / convertf** — the EIGENSTRAT and ANCESTRYMAP genotype formats and the `convertf` converter. Patterson N, Price AL, Reich D. *Population structure and eigenanalysis.* PLoS Genetics 2006;2(12):e190.
