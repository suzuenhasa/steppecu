# `genotype_front_end.hpp` reference

## 1. Purpose

`src/core/stats/genotype_front_end.hpp` declares the single shared genotype
decode front-end. Four tools read genotypes directly off disk rather than from a
precomputed f2 cache — the f2 extractor, the genotype-path D-statistic engine,
qpfstats, and DATES. This header is the one place all four turn a genotype triple
on disk into memory: either the `.geno`/`.snp`/`.ind` set or the PLINK
`.bed`/`.bim`/`.fam` set.

The output of the front-end is three things bundled together:

1. **The canonical tile** — the genotypes laid out individual-major (one row per
   individual) in the standard in-memory form the compute paths expect.
2. **The parsed SNP table** — the per-marker metadata read from the `.snp` or
   `.bim` file.
3. **The individual partition** — the selected populations arranged in the row
   order the callers rely on.

The helper does exactly this much and then stops. It reads the inputs, produces
those bundled results, and hands them back. It does not go on to compute
frequencies, apply per-tool filters, or run any GPU math — each caller does its
own decode after this point.

The value of concentrating this in one file is that a change to the read-time
front-end (a future ploidy tweak, a filter tweak, or a change to how the spanned
SNP count is chosen) is made in one edit point instead of being applied to four
separate tools in lockstep. Keeping four copies in sync by hand is exactly the
kind of drift that silently breaks parity between tools, and this shared helper
closes that risk.

---

## 2. The shared boundary and where it stops

The four tools share their work only up to a fixed boundary: the five values
`{tile, snptab, part, fmt, M0}` (the tile, the SNP table, the partition, the
on-disk format, and the spanned SNP count). Past that point each caller diverges
into its own decode:

- The D-statistic engine, qpfstats, and DATES all force diploid genotypes and
  keep autosomes.
- The f2 extractor instead does per-sample ploidy and applies its own extraction
  filter.

Because the callers diverge right after the boundary, the front-end deliberately
stops at the handoff and does **not** absorb the decode itself. It reads the raw
inputs into the shared canonical form and returns; it does not try to be a
one-size-fits-all decoder that also does the per-tool work. That divergence is
the reason the shared part ends where it does.

---

## 3. The `GenotypeFrontEnd` result struct

`GenotypeFrontEnd` is the bundle the helper returns — the five boundary values
described above, in one struct. Each of the four callers consumes it and then
does its own per-tool decode.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `tile` | `io::GenotypeTile` | — | The canonical individual-major tile: the decoded genotypes with one row per individual, produced by the underlying canonical-tile reader. |
| `snptab` | `io::SnpTable` | — | The parsed `.snp`/`.bim` marker metadata. Read with no row cap, so it holds every marker in the file. |
| `part` | `io::IndPartition` | — | The selected populations, laid out in the query/reference/outgroup row order the callers expect. |
| `fmt` | `io::GenoFormat` | `Unknown` | The on-disk genotype format, pinned by the reader when it opens the file (for example, native `.geno` versus PLINK `.bed`). |
| `M0` | `std::size_t` | `0` | The spanned SNP count — the number of SNPs actually read into the tile. It is the smaller of the marker count the genotype file's header claims and the number of rows in the parsed SNP table (see section 4). |

---

## 4. `read_genotype_front_end` — the primary entry point

The main overload takes the three input paths, a population selection, and a
compute backend, and returns a filled `GenotypeFrontEnd`:

```
GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                         const std::string& snp,
                                         const std::string& ind,
                                         const io::PopSelection& sel,
                                         ComputeBackend& backend);
```

It performs these steps in order:

1. Open the genotype reader on the `geno` path. This is what pins the on-disk
   format stored in `fmt`.
2. Read the individual partition from the `ind` file for the requested selection
   `sel`.
3. Read the full SNP table from the `snp` file, with no row cap, so `snptab`
   holds every marker.
4. Compute the spanned SNP count `M0` as the smaller of two numbers: the marker
   count in the genotype file's header, and the number of rows in the parsed SNP
   table. Taking the minimum guards against a genotype file and a SNP file that
   disagree on how many markers there are — only the SNPs both agree on are read.
5. Read the canonical individual-major tile covering SNPs `[0, M0)`.

### The `backend` argument

The `backend` is consulted **only** on the non-native transpose path — that is,
when the on-disk format needs a genotype transpose to reach the canonical
individual-major layout, the backend is forwarded to the underlying canonical-tile
reader to perform it. The native `.geno` path does not need it.

The helper does **not** own the backend. The caller creates the backend, passes
it in by reference, and then reuses that same backend for its own decode after the
front-end returns. The front-end only borrows it.

### Failure behavior

Any reader or transpose failure surfaces as a `std::runtime_error`. This is the
same exception contract the underlying canonical-tile reader and the individual
I/O readers already carry, so callers handle front-end failures exactly the way
they already handle failures from those lower-level readers — there is no new
error-handling convention to learn.

---

## 5. The `pop_labels` convenience overload

The second overload takes a list of population labels instead of a fully built
selection:

```
GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                         const std::string& snp,
                                         const std::string& ind,
                                         std::span<const std::string> pop_labels,
                                         ComputeBackend& backend);
```

It builds an explicit population selection from `pop_labels` — keeping the subset
of labels actually present in the data, sorted in ascending order by label — and
then delegates to the primary overload. It exists so the D-statistic, qpfstats,
and DATES callers do not each repeat the same short selection-building snippet;
that shared boilerplate lives here once.

---

## 6. Layering — no CUDA in this header

This header sits at the wiring point where the CUDA-free I/O layer meets the
CUDA-free compute-backend seam. It is not a CUDA translation unit. No CUDA header
is included here, and the compute backend is only forward-declared (rather than
fully included) to keep the header lightweight.

That is possible because the transpose is never done directly in this file; it is
reached indirectly through the underlying canonical-tile reader, which calls a
backend method that itself is declared on the CUDA-free side of the seam. The GPU
work happens behind that boundary, not in this header.

The helper is compiled into the core library (which links the I/O and device
libraries) and is reused two ways: by the core library's own genotype-path tools,
and by the standalone f2 extractor's run path (which links the core library).
Sharing one compiled helper across both keeps the read-time front-end identical
everywhere it is used.
