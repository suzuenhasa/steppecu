# `snp_major_tile.hpp` reference

## 1. Purpose

`src/io/snp_major_tile.hpp` defines `SnpMajorTile`, a plain data-holding struct
that carries one chunk of a genotype file whose records are stored *one SNP at a
time* — the layout used by the PACKEDANCESTRYMAP / GENO formats (and later
EIGENSTRAT and PLINK). A "SNP-major" file stores, for each SNP in turn, the
genotype codes of every individual packed together; the individuals are
interleaved inside each SNP's bytes.

This struct is the counterpart of `GenotypeTile`, which is the *individual-major*
tile — the canonical layout that the genotype-decoding step consumes directly.
The two hold the same data organized along opposite axes. Because a SNP-major
source cannot be reshaped into the individual-major layout without a transpose,
and the transpose has to run on the GPU, `SnpMajorTile` exists to carry the raw
SNP-major bytes plus a selection list from the file-reading layer up to the point
where the GPU transpose can run.

`SnpMajorTile` is a pure host-side struct. It lives in the plain input/output
layer, uses only the C++ standard library, and contains no CUDA and no dependency
on the GPU or core code. Its fields line up one-for-one with the GPU-side view
that the transpose kernel reads, but keeping this a dependency-free struct is what
lets the file-reading layer stay free of GPU code.

---

## 2. Why this is a separate struct, not a `GenotypeTile`

A SNP-major source cannot be packed into an individual-major `GenotypeTile`
without first transposing the data, and that transpose is a GPU operation. The
file-reading layer is deliberately kept free of any GPU calls — it is not allowed
to launch a kernel. So the reader stops one step short: it produces this raw
SNP-major view together with the list of which individuals to keep and in what
order, and hands that upward.

The layer that wires the file-reading side to the GPU side is the only place
allowed to bridge the two. That layer takes a `SnpMajorTile` and runs the
transpose-to-canonical pass, which both reshapes the bytes into individual-major
order and applies the individual selection, producing a `GenotypeTile`. This
split is why the selection and reordering are described here but performed later:
the SNP-major source interleaves *all* individuals inside every SNP's bytes, so
the gather that picks out and reorders the wanted individuals can only happen at
transpose time.

---

## 3. How the SNP-major source bytes are packed

The raw bytes live in the `snp_major` field. They hold `n_snp` records laid end
to end, each record exactly `src_bytes_per_record` bytes wide. Record number `s`
is SNP number `s`.

Within a record, genotype codes are packed four per byte, two bits each, filled
from the most significant bits downward (the same bit order used elsewhere for
these codes, just with the individual axis interleaved inside each SNP's byte).
To find the code for **source individual `i` of SNP `s`**:

- byte offset: `s * src_bytes_per_record + i / 4`
- position within that byte: `i % 4`, counting from the most significant bits

### The record stride can be wider than the data

`src_bytes_per_record` is not simply `ceil(n_ind / 4)`. It is the GENO format's
record stride, defined as `max(kGenoHeaderBytes, ceil(n_ind / 4))` — that is, the
larger of the packed-data width and the file's header size. The GENO format
floors every record to at least the header length, so a dataset with very few
individuals still has records padded out to that minimum width. When there are
few individuals, `src_bytes_per_record` is therefore *larger* than the packed
data actually needs, and each record ends with padding bytes.

### Why the padding is never misread as extra individuals

Those padding bytes hold no real individuals, and a naive reader could decode them
as phantom samples. This struct avoids that by construction: the transpose only
ever reads the bytes for the *selected* rows, and every selected source row index
is less than the source's true individual count. Because each read uses `i / 4`
for a valid `i`, the byte offset always lands inside the meaningful portion of the
record and never reaches the trailing padding. No phantom individuals can be
decoded.

---

## 4. Fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `snp_major` | `vector<uint8_t>` | empty | The raw SNP-major source bytes for the SNP range this tile covers, at the full record stride. Record `s` is SNP `s`. Its length is `n_snp * src_bytes_per_record`. |
| `src_bytes_per_record` | `size_t` | `0` | The source record stride — `max(kGenoHeaderBytes, ceil(n_ind / 4))`. Can be wider than the packed data (see section 3). Only offsets computed from selected rows are ever read from a record. |
| `n_snp` | `size_t` | `0` | The number of SNPs this tile covers. This is both the count of source records and the number of columns in the output tile. |
| `sel_rows` | `vector<size_t>` | empty | For each output individual `g`, the source individual row it comes from: output column `g` gathers from source row `sel_rows[g]`. This is the selected set of individuals, flattened population by population in the order described below. Its length is `n_individuals`. The transpose applies this selection and reordering as it writes each output row. |
| `n_individuals` | `size_t` | `0` | The number of gathered output individuals — the number of individuals actually kept, and the output record count. |
| `pop_offsets` | `vector<size_t>` | empty | Where each population's block of individuals begins and ends along the output individual axis. Population `p` owns the output individuals in the half-open range `[pop_offsets[p], pop_offsets[p+1])`. Its size is `P + 1`. |
| `pop_labels` | `vector<string>` | empty | The population labels, one per population, in the same order as the segments. Carried so that the individual-major tile built later keeps its population names. Its size is `P`. |

### `n_pop()`

A small helper method that returns the number of populations, `P`. It is simply
`pop_labels.size()`. It is marked `[[nodiscard]]` and `noexcept`.

---

## 5. Ordering and invariants

### Population order

The selected individuals are laid out grouped by population, one population's
individuals fully before the next. Populations appear in ascending order of their
labels, and within each population its individuals are contiguous. Both
`sel_rows` (the source-row lookup) and `pop_offsets` (the segment boundaries)
follow this same output ordering, and `pop_labels` is parallel to it.

### Cross-field invariants

- `snp_major.size() == n_snp * src_bytes_per_record`.
- `sel_rows.size() == n_individuals`.
- `pop_offsets.size() == P + 1`, with `pop_offsets[0] == 0` and
  `pop_offsets[P] == n_individuals`. The offsets are non-decreasing, and each
  population's segment `[pop_offsets[p], pop_offsets[p+1])` is that population's
  slice of the output individuals.
- `n_pop() == pop_labels.size() == pop_offsets.size() - 1`.
- Every entry of `sel_rows` is a valid source individual row (strictly less than
  the source's individual count), which is what keeps the record padding from ever
  being decoded (see section 3).
