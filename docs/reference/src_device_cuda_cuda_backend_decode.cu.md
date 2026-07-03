# `cuda_backend_decode.cu` reference

## 1. Purpose

This file holds the out-of-line bodies of the `CudaBackend` methods that turn raw
packed genotype bytes into the per-population, per-SNP quantities the rest of the
pipeline consumes. It is the genotype-decode front-end of the GPU backend.

Six methods live here:

- `decode_af_resident` — the shared workhorse. Uploads a tile of packed genotypes,
  unpacks the 2-bit codes, and runs a segmented reduction that produces three
  device-resident arrays (`Q`, `V`, `N`). Everything else in the file builds on it.
- `decode_af` — the host-facing decode: calls `decode_af_resident`, then copies the
  three arrays back to host memory.
- `detect_sample_ploidy_device` — a standalone on-device pass that decides, per
  individual, whether that sample is diploid or should be treated as haploid.
- `transpose_to_canonical` — reshapes a SNP-major tile (all individuals for one SNP,
  then the next SNP) into the canonical individual-major layout (all SNPs for one
  individual, then the next individual).
- `decode_af_compact_autosome` and `decode_af_compact_filter` — the two
  device-resident "compact decode" paths. Each decodes, builds a per-SNP keep/drop
  mask, and then physically removes the dropped columns on the GPU, leaving a smaller
  resident result that never round-trips through host memory.

This is a CUDA translation unit. It is compiled into the private device library and
inherits the same code-generation settings as the rest of that library; nothing here
is part of the public, CUDA-free surface. The method bodies were moved verbatim out
of a larger backend file — the split changed nothing about the math, the precision,
or the order of operations.

The one behavior worth internalizing before reading the code: the big result arrays
stay on the GPU. The compact paths deliberately do **not** copy `Q`/`V`/`N` back to
the host; they hand back a device handle so the next stage can keep working in GPU
memory. Only `decode_af` (the explicit host/reference path) pays for a full copy
back.

---

## 2. What decode produces: `Q`, `V`, and `N`

Every decode path produces the same three arrays, each shaped as `P × M` — one row
per population (`P`) and one column per SNP (`M`), stored row-major:

- **`Q`** — the per-population, per-SNP allele frequency (the fraction of counted
  alleles that are the tracked allele).
- **`V`** — the paired variance / heterozygosity-correction term that the downstream
  f2 math needs alongside the frequency. It is decoded in lockstep with `Q`.
- **`N`** — the count of non-missing observations that went into that frequency (the
  denominator, in effect). It records how much data actually backed each cell.

These come from two GPU stages run back to back: an unpack stage that expands the
packed 2-bit codes, and a segmented reduction that sums each population's samples
into that population's row. "Segmented" means the reduction is bucketed by
population using a set of offsets (`pop_offsets`) that mark where each population's
individuals begin and end in the sample axis.

Not every path keeps all three. The autosome compact path decodes `N` but throws it
away (its consumers only use `Q` and `V`). The coverage-filter path keeps all three,
because its filter decision is defined in terms of coverage and it must carry `N`
forward.

---

## 3. The shared decode front-end (`decode_af_resident`)

`decode_af_resident` is the single place packed genotypes become resident `Q`/`V`/`N`.
It takes a tile description, the population count `P`, the SNP count `M`, three
already-allocated device buffers to fill, and a SNP offset `s_lo`. It uploads the
packed bytes, optionally resolves per-sample ploidy, and launches the unpack +
reduction. It never copies anything back — the results stay in the caller's device
buffers.

### SNP-tile slicing and the byte-alignment invariant

A tile can describe a horizontal slice of each individual's full genotype row rather
than the whole row. The slice covers SNPs `[s_lo, s_lo + M)`. This is how the caller
keeps GPU memory bounded: the resident set is proportional to `P × M` for the tile's
`M`, never `P × M_total` for the whole genome. When a caller wants the whole row it
passes `s_lo = 0` and `M` equal to the full width, and the slice degenerates to the
entire record with bit-identical results.

Because genotypes are packed 4 codes to a byte, the slice offset must land on a byte
boundary. The code asserts `s_lo % 4 == 0` (`kCodesPerByte` is 4). With that
guarantee, SNP index `local_s` within the tile maps cleanly to byte `local_s / 4`,
bit position `local_s % 4` — exactly the same code positions the full-width decode
would read. That is what lets the unpack, keep-mask, scan, and gather kernels stay
completely unchanged between the sliced and unsliced cases: they always address a
tile-local record stride and never need to know they are looking at a slice. The
compacted per-record stride is `ceil(M / 4)` bytes.

### The strided host-to-device upload

The packed bytes on disk use the **full** on-disk row stride (`bytes_per_record`),
but the GPU only needs the `ceil(M / 4)` bytes of the slice. The upload is therefore
a strided 2-D copy: for each individual (row `g`) it starts reading at
`packed + g * bytes_per_record + s_lo/4` and copies just `ceil(M / 4)` bytes into a
compact, tightly-packed device buffer. The source pitch is the full disk stride; the
destination pitch is the compact tile stride. This copies only the slice, not the
padding beyond it, so no wasted bandwidth and no wasted device memory.

All copies and kernel launches in this method go on the backend's single stream, so
they run in order and the caller's later work on that same stream is guaranteed to
see the finished `Q`/`V`/`N`. The temporary upload buffers free at scope exit even
though the copies are asynchronous, because the stream ordering keeps them alive
until the decode kernel has consumed them.

---

## 4. Per-sample ploidy: the three cases

Some samples are genuinely diploid; others (typically low-coverage ancient samples)
are "pseudo-haploid" and must be treated as a single allele call. The decode has to
know which is which per individual, matching the reference tool's pseudo-haploid
adjustment. `decode_af_resident` resolves this with a strict three-way precedence:

1. **Explicit per-sample vector.** If the tile carries a non-null `sample_ploidy`
   array, it is uploaded and used directly. This wins over everything else.
2. **On-device detection.** If there is no explicit vector but the tile requests
   on-device detection, the ploidy is derived on the GPU from the packed bytes that
   were just uploaded — one thread per individual scanning that individual's codes.
   This produces bit-identical results to the host-side detector, and because it runs
   on the same stream and writes a device buffer that immediately feeds the decode,
   it costs no copy back to the host. The detection kernel is enqueued **after** the
   upload, so it correctly reads finished data.
3. **Uniform fallback.** If neither is present, the decode kernel uses the tile's
   single scalar ploidy value for every sample (the plain all-diploid path). No
   per-sample buffer is allocated or copied; the kernel receives a null device
   pointer. A zero-length device buffer conveniently yields a null data pointer, so
   the same variable carries the "no per-sample ploidy" case without a special
   branch.

---

## 5. The host-oracle decode path (`decode_af`)

`decode_af` is the path that hands results back to host memory. It allocates the
three `P × M` device buffers, calls `decode_af_resident` to fill them, then copies
all three back to host vectors and synchronizes before returning a `DecodeResult`.

This full copy-back is what distinguishes it from the compact paths. It exists for
the host and reference-oracle side of the seam, where the caller genuinely needs the
numbers in CPU memory. The device-resident compact paths (section 8) deliberately
skip this copy — that dropped round-trip is the point of them.

Edge cases are handled up front: a tile with no populations or no SNPs returns a
zero-filled result immediately, before any GPU work. The `DecodeResult` reports back
the `P` and `M` it was built with.

---

## 6. On-device ploidy detection (`detect_sample_ploidy_device`)

`detect_sample_ploidy_device` is the standalone version of the ploidy pass — the same
work that case 2 of section 4 folds into decode, but exposed as its own method for
callers that just want the per-sample ploidy vector.

It uploads the full packed records, launches the detection kernel (one thread per
individual, reading that individual's codes across all SNPs), and copies the
resulting per-sample integer vector back to the host. The result vector is
pre-filled with the pseudo-haploid value, so a tile with zero individuals returns an
empty/default result without touching the GPU.

---

## 7. SNP-major to canonical transpose (`transpose_to_canonical`)

Some input formats arrive SNP-major: the file stores every individual's call for SNP
0, then every individual's call for SNP 1, and so on. The rest of the pipeline wants
the canonical individual-major layout: all of individual 0's SNPs, then all of
individual 1's SNPs. `transpose_to_canonical` performs that reshape on the GPU.

Key points:

- The output record stride is `ceil(n_snp / 4)` bytes — the same packing radix the
  rest of the codebase uses, taken from the shared `kCodesPerByte` constant so the
  two definitions cannot drift apart.
- The caller supplies a per-output-column source-row selection (`sel_rows`): output
  column `i` is filled from source row `sel_rows[i]`. This lets the transpose also
  reorder or subset individuals, not just flip the axes.
- The method uploads the SNP-major source and the selection, maps the CUDA-free
  encoding enum onto the device-private one (currently a one-to-one identity
  mapping), launches the transpose kernel, and copies the packed result back.
- An empty tile (no individuals or no SNPs) short-circuits to an empty result with
  the correct metadata filled in, before any device work.

---

## 8. The device-resident compact-decode regimes

`decode_af_compact_autosome` and `decode_af_compact_filter` both decode a tile, decide
per SNP whether to keep it, and then physically drop the unwanted SNP columns on the
GPU — leaving a smaller `P × M_kept` result that stays resident in device memory. The
returned handle owns the device buffers and escapes to the caller; the large arrays
are never copied to the host. Both follow the same six-step shape:

1. **Decode** the tile into resident `Q`/`V`/`N` via `decode_af_resident`. No copy
   back.
2. **Build a per-SNP keep mask** (`flags`, one byte per SNP: keep or drop). This is
   the only step where the two regimes differ (see below).
3. **Build the compacted column index** — an exclusive prefix sum of the flags, so a
   kept SNP learns its new column position. (See section 9.)
4. **Compact the small per-SNP side arrays** (chromosome, genetic position, and
   optionally physical position) down to just the kept SNPs, preserving file order.
   (See section 9.)
5. **Allocate the resident compacted result** and gather the kept columns of the big
   arrays into it, keyed by the scan index so the kept columns land in file order.
6. **Copy back only the small kept side arrays** (a few values per kept SNP) so the
   host-side, CUDA-free block-assignment step can run. The big `Q`/`V`/`N` stay on the
   GPU.

The kept count `M_kept` is read back as a single integer between steps 4 and 5, since
it is needed to size the resident result buffers. Conceptually it equals the last
exclusive-scan entry plus the last flag; in practice it is read from the count that
the compaction step reports. If nothing survives the mask, the method returns early
with an empty result.

### Regime A — `decode_af_compact_autosome`

The keep mask is a pure, integer-exact autosome test: keep SNP `i` if its chromosome
number falls within `[chrom_min, chrom_max]`. It compacts and keeps only `Q` and `V`;
`N` is decoded but discarded, because this regime's consumers never read it.

### Regime B — `decode_af_compact_filter`

The keep mask is a richer quality-control decision computed from the decoded data and
allele metadata (reference/alternate allele characters, chromosome), plus a separate
population-coverage missingness threshold (`maxmiss`). Two details matter:

- The per-sample missingness predicate inside the shared filter is forced off here
  (its threshold is set to 1.0, meaning "never drops on that axis"), because in this
  regime missingness is judged on the population-coverage axis via the separate
  `maxmiss` argument instead.
- That population-coverage decision needs a denominator: the total number of
  individuals across all populations. It is summed from the caller-supplied
  per-population counts when their length matches `P`; if that array is short or
  empty, the code falls back to deriving the counts from the tile's population
  offsets so the reduction can never read out of bounds. This mirrors the host-side
  filter's denominator logic.

Regime B keeps all three arrays — `Q`, `V`, **and** `N` — with three lockstep gathers
driven by the same flags and the same scan index, so `N` stays column-aligned with
`Q` and `V` and in file order. This regime also accepts the SNP-slice offset `s_lo`
and forwards it into `decode_af_resident`, so it can be driven column-tile by
column-tile to keep peak GPU memory bounded.

### The optional physical-position axis

Both regimes accept an optional physical-position (base-pair) array. It is a fallback
axis used later when a dataset ships without a genetic linkage map. It is uploaded
and compacted in exact lockstep with the other side arrays **only** when the caller
supplies a full-length span; an empty span leaves the kept physical-position output
empty and the fallback off. Because it is the same value type as genetic position, it
reuses that array's compaction sizing with no extra setup.

---

## 9. The CUB compaction protocol and its temp-storage guard

Steps 3 and 4 of both compact paths lean on two device primitives from the CUB
library. Understanding two idioms explains most of that code.

### The exclusive-prefix-sum column index

The compacted column position of each kept SNP is an exclusive prefix sum of the
keep-flags: kept SNP `i` moves to column `sum(flags[0 .. i-1])`. This is computed with
one exclusive-scan call over the flag bytes into a 64-bit index buffer, and the gather
in step 5 uses that index to place each kept column. Because the scan is monotone,
the kept columns keep their original relative order — file order is preserved through
the whole compaction.

### The two-call temp-storage idiom

Both CUB primitives need scratch device memory whose size they compute themselves. The
required pattern is to call the primitive **twice**: first with a null scratch pointer,
which only writes back the number of scratch bytes needed; then allocate exactly that
many bytes and call again to do the real work. The code guards against a zero-byte
request by allocating at least one byte, so the scratch buffer is always valid.

### The int-vs-double max-temp-size guard (a real bug this prevents)

The compaction primitive that filters the side arrays (`DeviceSelect::Flagged`) sizes
its scratch based on the **value type** being compacted — a chromosome array is 32-bit
integers, but genetic and physical positions are 64-bit doubles, and the double case
needs a larger scratch buffer. The code therefore queries the scratch size for **both**
the integer and the double case and allocates the **maximum** of the two, then reuses
that one buffer for all the flagged compactions.

This is not defensive over-engineering: reusing an integer-sized scratch buffer for a
double compaction silently under-sizes it and corrupts the output — the observed
symptom was an all-zero genetic-position array. Sizing to the larger (double) query is
what fixes it. The physical-position array is also a double, so the double query
already covers it and it needs no separate size query.
