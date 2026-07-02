# `genotype_source.hpp` reference

## 1. Purpose

`src/io/genotype_source.hpp` is the single front door for turning a user's
`--prefix` into the actual genotype files on disk, and for reading the two
metadata sidecar files that come with them.

steppe accepts one command-line argument to name a dataset: a prefix `P`.
On disk, that prefix expands into a *triple* of three sibling files — the
genotype matrix plus two metadata files describing the SNPs (columns) and the
individuals (rows). The catch is that there are two different naming schemes:

- The EIGENSTRAT family of formats (called TGENO, GENO, and EIGENSTRAT) uses the
  extensions `P.geno`, `P.snp`, and `P.ind`.
- The PLINK format uses `P.bed`, `P.bim`, and `P.fam` — *different extensions*,
  and the two metadata files also use *different parsers* internally
  (`read_bim` instead of `read_snp`, `read_fam` instead of `read_ind`).

Five different tools in steppe need to open a genotype dataset (the f2
extraction path, the D-statistic path, the qpfstats path, the DATES path, and
the command-line tool). Without this header, each of them would have to
re-spell the `.geno/.snp/.ind`-versus-`.bed/.bim/.fam` fork on its own. This
header centralizes both decisions — *which extensions* to use and *which
parser* to run — in one place, so the five callers all agree and none of them
re-implements the fork.

This is a host-only header: plain standard C++ with no GPU code and no
dependency on the compute layer. It uses only three sibling readers from the
same input-output layer.

---

## 2. GenotypeTriple

`GenotypeTriple` is the small struct that holds the three resolved file paths
for one prefix, plus a flag saying which naming scheme was chosen.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `geno` | `string` | — | The genotype-matrix path: `P.geno` for the EIGENSTRAT family, or `P.bed` for PLINK. This is the file that the genotype reader opens to decode the actual data. |
| `snp` | `string` | — | The per-SNP metadata path: `P.snp` for the EIGENSTRAT family, or `P.bim` for PLINK. |
| `ind` | `string` | — | The per-individual metadata path: `P.ind` for the EIGENSTRAT family, or `P.fam` for PLINK. |
| `is_plink` | `bool` | `false` | True only when the `.bed/.bim/.fam` extensions were chosen. This records the *extension* decision, not the authoritative on-disk format (see below). |

The `geno` path is handed to the genotype reader, which is what actually decodes
the data and pins down the true format. The `snp` and `ind` paths are handed to
the two reader functions in sections 4 and 5, which pick their parser based on
that pinned format.

---

## 3. resolve_genotype_triple

```
GenotypeTriple resolve_genotype_triple(const std::string& prefix);
```

This function expands a prefix into its triple by probing the filesystem. The
rule is deliberately simple and has one important subtlety.

### The probe rule

- If `prefix.bed` exists **and** `prefix.geno` does **not** exist, the triple is
  PLINK: `.bed / .bim / .fam`, with `is_plink` set true.
- Otherwise the triple is the EIGENSTRAT family: `.geno / .snp / .ind`. This is
  the default, and it covers TGENO, GENO, and EIGENSTRAT, which all share those
  same three extensions.

This is often called the "`.geno`-present-wins" rule: whenever a `.geno` file
is present, the resolver always chooses the EIGENSTRAT-family extensions,
regardless of whether a `.bed` also happens to exist. PLINK is reached only in
the specific case where there is a `.bed` but no `.geno`.

The reason for that tie-breaker is backward compatibility. Every dataset that
already worked before PLINK support was added had a `.geno` file, so the
"`.geno`-present-wins" rule guarantees all of those prefixes keep resolving
*exactly* as they did before — there is no behavior change for the established
formats. PLINK is purely additive.

### Extension resolution only — never decoding

This function chooses *extensions*, nothing more. It does **not** open the
genotype file, read any magic bytes, or decide what the real on-disk format is.
The authoritative format is determined later, when the genotype reader's
constructor opens the resolved `geno` path and inspects its contents (magic
number, byte layout, or the `.bed` magic). In other words, `is_plink` reflects
only the filename decision made here; the true format (TGENO versus GENO versus
EIGENSTRAT versus PLINK) is settled downstream and then passed *back* into the
two functions below, so this front door never has to re-probe the magic.

---

## 4. read_snp_table

```
SnpTable read_snp_table(GenoFormat format, const std::string& path,
                        std::size_t max_snps);
```

Reads the per-SNP metadata table, choosing the parser from the already-detected
on-disk format:

- PLINK format → `read_bim(path, max_snps)`.
- Any other format → `read_snp(path, max_snps)`.

Both parsers return the *same* `SnpTable` type, carrying the same columns: the
SNP id, chromosome, genetic position in Morgans, and the reference and
alternate alleles. Because the output type is identical no matter which parser
ran, everything the caller does afterward — the SNP filtering, the block
partitioning for the jackknife, and the decode — is byte-for-byte the same
regardless of format. The `format` argument is the format the genotype reader
already detected, so this function trusts it rather than sniffing the file
again. `max_snps` caps how many rows are read.

---

## 5. read_ind_partition

```
IndPartition read_ind_partition(GenoFormat format, const std::string& path,
                                const PopSelection& sel,
                                std::size_t n_records_present);
```

Reads the individual/population metadata (the `.ind` or `.fam` file), again
dispatching on the detected format:

- PLINK format → `read_fam(path, sel, n_records_present)`.
- Any other format → `read_ind(path, sel, n_records_present)`.

Both parsers return the *same* `IndPartition` type. That partition contains the
selected populations arranged in a fixed row order — the left, right, and target
population groups — and within that, the labels are sorted in ascending order.
Because both parsers produce the identical partition layout, the downstream math
is unaffected by which naming scheme the dataset used.

The `sel` argument is the caller's population selection (which populations to
keep). `n_records_present` is the number of individual records the genotype
side expects, used to keep the metadata and the genotype matrix consistent. As
with the SNP reader, `format` is the format already pinned by the genotype
reader, so this function does not re-detect it.
