# `genotype_tile.hpp` reference

## 1. Purpose

`src/io/genotype_tile.hpp` defines a single small struct, `GenotypeTile`. It is
the hand-off object that the genotype reader (the `io` layer) produces and the
decode backend (CPU or GPU) consumes. Think of it as a labelled box of raw
genotype bytes plus just enough description to make sense of them.

The reader's job stops at reading and tiling. It does **not** decode: it does not
turn the packed bytes into allele frequencies. It reads the on-disk genotype
records, selects and gathers the individuals it was asked for, packs their bytes
into this struct, and hands the struct across to the backend, which does the
actual decoding into the numbers downstream stages need. This struct is the seam
between those two responsibilities.

Two design choices follow from that role:

- **Plain data only.** Every field is a standard-library container or a plain
  integer. There are no GPU types and no methods that compute anything (aside
  from one trivial size accessor). This lets the struct cross the boundary
  between the host-only reader and the compute backend without dragging any GPU
  dependency into the reader.
- **Population partition included.** Alongside the bytes, the struct carries the
  boundaries that split the individuals into their populations, so the decoder
  can reduce over exactly the individuals of one population without having to be
  told the grouping separately.

---

## 2. The packing layout

The bytes in `packed` follow the real AADR on-disk genotype layout, which is
**individual-major**: all of one individual's SNPs sit together, then the next
individual's, and so on. steppe keeps that same shape in the tile.

Concretely:

- **Individuals are gathered into population-contiguous order.** The reader picks
  out only the individuals belonging to the requested populations and copies them
  so that population 0's individuals come first, then population 1's, and so on —
  in the same population row order used throughout the rest of the pipeline. This
  gathered order is what every index in the struct refers to.
- **Each individual gets a fixed-length record.** Individual `g` (numbered from 0
  over the gathered set) occupies the byte range
  `packed[g * bytes_per_record .. (g + 1) * bytes_per_record)`. Every individual's
  record is the same length, so any individual can be located by simple
  multiplication.
- **Four SNPs per byte, top bits first.** Within one individual's record, SNP `s`
  is the 2-bit code found in byte `s / 4`, at position `s % 4`. The positions are
  packed most-significant-bits-first: position 0 is the top two bits of the byte,
  position 1 the next two, and so on down to position 3 in the bottom two bits.
  (This is the same bit addressing the EIGENSTRAT packed format uses.)
- **A SNP prefix, not the whole record.** On-disk each individual's record may
  cover more SNPs than this tile does. The tile holds only the first
  `bytes_per_record` bytes of each record — the prefix that covers SNPs
  `0 .. n_snp - 1`. That is what makes this a *tile*: it is a window over the SNP
  axis, and a large dataset is processed as several such tiles.

The population boundaries (`pop_offsets`, described below) then say which stretch
of gathered individuals belongs to each population, so a decoder can sum or
average over precisely one population's individuals.

---

## 3. Fields of `GenotypeTile`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `packed` | `vector<uint8_t>` | empty | The packed genotype bytes for all gathered individuals, laid out in population-contiguous, individual-major order as described above. Its length is `n_individuals * bytes_per_record`. |
| `bytes_per_record` | `size_t` | `0` | The stride, in bytes, of one individual's record inside `packed`. Equals `ceil(n_snp / 4)` for this tile's SNP count (four SNPs per byte). This is derived from the format's constants, never a hardcoded number. |
| `n_snp` | `size_t` | `0` | How many SNPs this tile covers — the length of the SNP axis for this tile. Valid SNP indices within each record run `0 .. n_snp - 1`. |
| `n_individuals` | `size_t` | `0` | The total number of gathered individuals across all selected populations — the sum of the population segment sizes. Equals `packed.size() / bytes_per_record`. |
| `pop_offsets` | `vector<size_t>` | empty | The population segment boundaries over the individual axis. Population `p` owns the gathered individuals in the half-open range `[pop_offsets[p], pop_offsets[p + 1])`. See section 4 for its exact invariants. |
| `pop_labels` | `vector<string>` | empty | The population names, in the same population row order as the segments and parallel to them. Its length is the population count `P`. `pop_labels[p]` names the population whose segment is `[pop_offsets[p], pop_offsets[p + 1])`. |
| `sample_ploidy` | `vector<int>` | empty | Optional per-individual ploidy, parallel to the gathered individual axis. See section 5. |

There is one member function, `n_pop()`, covered in section 6.

---

## 4. Invariants and size relationships

These relationships always hold for a well-formed tile, and downstream code
relies on them rather than re-checking the raw sizes:

- `packed.size() == n_individuals * bytes_per_record`.
- `bytes_per_record == ceil(n_snp / 4)` — four 2-bit SNP codes per byte, rounded
  up so the last, partly-filled byte is still counted.
- `n_individuals == packed.size() / bytes_per_record`, and it also equals the sum
  of all the population segment sizes.

The population partition has its own strict shape:

- `pop_offsets` has length `P + 1`, where `P` is the number of populations.
- `pop_offsets[0] == 0` (the first segment starts at the beginning).
- `pop_offsets[P] == n_individuals` (the last segment ends at the last gathered
  individual, so the segments exactly tile the whole individual axis with no gaps
  and no overlap).
- The offsets are non-decreasing; the size of population `p`'s segment is
  `pop_offsets[p + 1] - pop_offsets[p]`.

Finally, the two population-parallel arrays agree on the count:

- `P == pop_labels.size() == pop_offsets.size() - 1`.

`pop_labels[p]` and the segment `[pop_offsets[p], pop_offsets[p + 1])` describe
the same population `p`, so the labels and the boundaries can be read together
index-by-index.

---

## 5. Per-sample ploidy

`sample_ploidy` records the ploidy of each gathered individual: `2` for a diploid
sample, `1` for a pseudo-haploid sample. When present, it is parallel to the
gathered individual axis, so `sample_ploidy[g]` is the ploidy of gathered
individual `g`, and its length equals `n_individuals`.

**How it is filled.** The value is auto-detected per sample[^at2]: if a sample
shows any heterozygous call among the leading SNPs it is treated as diploid,
otherwise it is treated as pseudo-haploid.
A separate detection step is what populates this array.

**Empty means "not detected."** An empty `sample_ploidy` is a valid, meaningful
state: it signals that no per-sample detection was done, and the caller should
fall back to treating every sample as one uniform ploidy — the older path that
assumed all samples were diploid. So the field has two modes:

- empty → use the uniform fallback ploidy for every sample; or
- non-empty → it has exactly `n_individuals` entries, one real ploidy per sample.

There is no partially-filled state: if the array is non-empty, it is complete.

---

## 6. The `n_pop()` helper

`n_pop()` returns the number of populations `P` in the tile. It is a one-line
accessor that simply returns `pop_labels.size()`. `P` is the leading dimension of
the decoded output the backend produces from this tile (the per-population
result), so this accessor gives that dimension a name instead of making callers
reach for `pop_labels.size()` directly. It does no work and cannot fail.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
