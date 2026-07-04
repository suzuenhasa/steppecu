# `views.hpp` reference

## 1. Purpose

`src/core/internal/views.hpp` defines the one small type — `MatView` — that both
the CPU reference code and the GPU feeder use to agree, exactly, on how the three
f2 input arrays are laid out in memory. It is the seam between the part of steppe
that reads and decodes genotype data and the part that computes f2 statistics from
that data.

The header is deliberately minimal and host-only:

- **No ownership.** A `MatView` is a *view*: it holds a borrowed pointer plus the
  row and column counts. It never allocates and never frees the storage it points
  at. Keeping the memory alive is the caller's job.
- **No CUDA.** The file uses nothing but a plain `double*` and two integers, so it
  can be included by pure host code and by GPU code alike without dragging in the
  GPU toolchain. A richer, span-and-mdspan-based view type exists for the device
  layer; this is the minimal host-side anchor that the input contract needs.

The value of having one shared struct is that the indexing rule lives in exactly
one place. If the CPU path and the GPU path each wrote their own `data[i + P*s]`
by hand, the two could silently disagree; here they both call the same
`element(i, s)`.

Note that which SNPs belong to which genome block is decided elsewhere (by the
shared block-partition rule, which is centimorgan-based). A `MatView` only
describes the layout of one block's worth of data; it says nothing about how
blocks are chosen.

---

## 2. The Q/V/N contract

steppe's f2 computation consumes three arrays per SNP block, always with the same
shape: `P` rows by `M` columns, where `P` is the number of populations and `M` is
the number of SNPs in the block. The three arrays are named Q, V, and N, and their
meanings are fixed regardless of how the genotypes were decoded, whether the
samples are diploid or pseudo-haploid, or which arithmetic precision is used.

This is the same byte layout written by the input-building tooling
(`build_tgeno_matrix.py`, which emits `Q.f64` / `V.f64` / `N.f64`) and read back
by the load path, so the in-memory contract and the on-disk fixtures are
byte-compatible.

| Array | Meaning |
|---|---|
| **Q** | The frequency of the fixed reference allele, a value in `[0, 1]`. The reference allele is the one named in the `.snp` file, and a raw genotype value of 0/1/2 counts how many copies of that reference allele a sample carries. Entries that are invalid (missing) are **filled with zero**. |
| **V** | A validity mask: `1.0` when the population has a non-missing genotype at that SNP, `0.0` when it does not. |
| **N** | The **non-missing haploid count** — the number of alleles observed, not the number of individuals. It is 2 for each non-missing diploid, or 1 for each non-missing pseudo-haploid. |

### Why Q is zero-filled at invalid entries

Filling invalid Q entries with zero is not just a convenience — it is what makes
the masked matrix multiply come out correct. Because the f2 math multiplies Q
values together, a zero at an invalid entry forces the squared term (`Q² = 0`) and
the cross term to vanish exactly, so an invalid entry contributes nothing rather
than polluting the result.

### Why N counts alleles, not individuals

N being a haploid (allele) count rather than a per-individual count matches the
bias-correction convention used for parity[^at2]. This distinction matters for
ancient-DNA data, where samples are frequently pseudo-haploid: such a sample
contributes 1 to N, not 2. Honoring this is mandatory. It changes *how N is
computed* upstream, but it does not change this contract — N is always "alleles
observed."

N feeds exactly one place: the per-SNP heterozygosity correction, a term of the
form `q(1 - q) / max(N - 1, 1)`. The `max(N - 1, 1)` floor is what keeps that
division safe when a population has only a single non-missing allele.

### The producer's invariant

The code that fills these arrays guarantees, and every consumer is allowed to
assume, that validity and count always agree:

> **`V != 0`  if and only if  `N > 0`.**

In other words, a SNP is marked valid for a population exactly when that
population has at least one observed allele there. There is never a "valid but
zero count" or "invalid but positive count" entry.

---

## 3. Column-major layout and indexing

All three arrays — and any `MatView` — are stored **column-major**, the same
convention the GPU matrix-multiply library expects. In a column-major `P × M`
matrix:

- The **leading dimension is `P`** (the number of rows / populations).
- One full column (all `P` populations for a single SNP) sits contiguously in
  memory.
- The element for population `i` and SNP `s` lives at the flat offset
  **`i + P·s`**.

So walking down a column (fixing a SNP, varying the population) steps through
adjacent memory, while stepping across columns (fixing a population, varying the
SNP) jumps by `P` each time.

The column count `M` is stored as a `long`, not an `int`, specifically so that a
view over a large SNP block cannot overflow when the offset `i + P·s` is computed.
Because `s` is already a `long`, the product `P·s` is evaluated in `long`
arithmetic, and only the row index `i` needs to be widened.

---

## 4. The `MatView` struct

`MatView` is the non-owning, column-major `[P × M]` view over `double` data. The
same struct is reused for Q, V, and N, and more generally for any column-major
matrix of doubles whose storage it does not own.

### Fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `data` | `const double*` | `nullptr` | Borrowed pointer to the column-major storage (leading dimension `P`). The view never frees it; keeping it alive is the caller's responsibility. |
| `P` | `int` | `0` | Number of rows, i.e. the number of populations. This is also the leading dimension used when computing offsets. |
| `M` | `long` | `0` | Number of columns, i.e. the number of SNPs in this block. Held as `long` to avoid overflow on large blocks (see the layout section above). |

### The `element` accessor

`element(int i, long s)` returns the value at population `i`, SNP `s` — that is,
`data[i + P·s]`. It is the single indexing rule shared by the CPU reference and
the GPU feeder, so neither side hand-writes the offset.

It performs **no bounds checking**, by design: it sits on a hot path, and callers
are required to stay within `0 ≤ i < P` and `0 ≤ s < M`. It is marked so that it
can never throw and its result must not be ignored.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
