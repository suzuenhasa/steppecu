# `readv2.cpp` reference

## 1. Purpose

`src/core/readv2/readv2.cpp` is the driver behind READv2: pseudo-haploid
windowed-mismatch kinship. Give it a genotype triple and a set of samples, and for
every unordered pair of samples it answers "how related are these two?" — it reports
the mean per-window allele-mismatch proportion, normalizes that against an all-pairs
background, classifies a relatedness degree (identical / first / second / unrelated),
and attaches a `z` confidence. One row per surviving pair, in the frozen schema:

```
sampleA  sampleB  n_windows  n_overlap_sites  P0_mean  P0_norm  degree  z
```

The single public entry point is `run_readv2`. It owns the whole pipeline: read the
SNP axis, pack it into a resident device bit-matrix, run the all-pairs mismatch sweep
on the GPU, form the background, and stream each pair's row to a caller-supplied sink.

This translation unit is **host-pure and CUDA-free**. It reaches the GPU only through
the `ComputeBackend` seam (`device::primary_backend`), exactly the way
`fstat_sweep.cpp` and `run_dates` do — it names no CUDA type and includes no GPU
header. The actual `__popc` mismatch kernels live behind that seam in
`src/device/cuda/`. That isolation keeps the driver readable as plain orchestration:
everything you see here is "what to compute and in what order", never "how the kernel
does it".

---

## 2. What comes in, what goes out

`run_readv2` takes the three genotype-file paths (`geno`, `snp`, `ind`), an
`io::IndPartition` that has already been resolved from the `.ind` by the caller (each
group is a singleton — one sample per "population"), a `Readv2Options` bag of knobs, a
`Readv2RowSink` callback, and the shared `device::Resources`.

The `ind` string argument is deliberately unused (`(void)ind;`) — the partition the
caller passes in already carries the resolved sample set, so the driver never re-reads
the `.ind` for identity. The path is kept in the signature for symmetry with the other
genotype-path drivers.

The knobs (`Readv2Options`):

| Knob | Default | Meaning |
|---|---|---|
| `window_snps` | 1000 | The non-overlapping SNP-count window (the READv2 default). |
| `norm` | `Median` | Background = median (default) or mean of the surviving pairs' `P0_mean`. |
| `min_overlap` | 0.0 | Drop a pair with fewer than this fraction of `m0` comparable sites. |
| `autosomes_only` | true | Restrict the SNP axis to chromosomes 1–22 (the READv2 convention). |
| `tiled` | false | Use the shared-memory tiled mismatch kernel instead of the plain one. |

The rows themselves **do not** come back in the return value. They are handed to
`sink` one at a time, in ascending pair-rank order. The returned `Readv2Result` is
just a run summary: how many individuals and pairs, how many rows were actually
emitted, the computed `background`, a `Status`, and the precision tag (always
`Fp64` — see section 8). The reason rows stream rather than accumulate is scale:
C(N,2) grows quadratically, and for a few thousand samples the full row set would be
millions of rows. Streaming means the formatted rows never all materialize at once.

---

## 3. The pipeline, end to end

`run_readv2` runs a fixed sequence:

1. **Size and sanity-check.** Count samples `N` and pairs `n_pairs = C(N,2)`. If
   `N < 2` or `window_snps <= 0`, there is nothing to compare — it returns
   `Status::InvalidConfig` immediately.
2. **Open the genotype axis.** Open the `.geno` via `io::GenoReader`, read the
   `.snp` table, and take `m0 = min(header.n_snp, snptab.count)` as the usable SNP
   count. An empty axis is `InvalidConfig`.
3. **Apply the autosome restriction** (section 4), which may trim `m0` down to a
   contiguous leading run of autosomal SNPs.
4. **Allocate the resident bit-matrix** (section 5) on the device, sized for `N`
   samples, the window geometry, and `m0` SNPs.
5. **Stream the SNP axis into it**, packing each chunk on the device, and gate on
   ploidy along the way (section 6).
6. **Run the all-pairs mismatch sweep** on the GPU, which returns four per-pair
   scalars (section 7).
7. **Pass 1 — form `P0_mean`, gate, and collect the background input** (section 7).
8. **Form the background** — median or mean of the survivors (section 7).
9. **Pass 2 — turn each survivor into a schema row and stream it** (section 7).

The two-pass shape at the end is deliberate: the background is a statistic *over* the
pairs, so every pair's `P0_mean` has to exist before any single pair can be
normalized. Pass 1 computes and filters; pass 2 normalizes and emits.

---

## 4. The autosome restriction (and why it fails fast)

READv2 conventionally runs on autosomes only, and this is on by default. The wrinkle
is a hard constraint from the reader below: the canonical-tile read only supports a
**byte-aligned SNP prefix beginning at index 0**. So the SNP axis can only be trimmed
to a *contiguous leading run* — you can lop off a trailing tail, but you cannot punch
holes in the middle.

For the standard 1240K layout (chr1–22, then chr23/X, chr24/Y) that is exactly right:
walk forward from index 0 while the chromosome is in `[kAutosomeChromMin,
kAutosomeChromMax]`, stop at the first non-autosome, and set `m0` to that prefix
length. The trailing sex/other chromosomes fall off cleanly.

But a panel where an autosome appears *after* a non-autosome (an interspersed or
unsorted layout) cannot be handled by a contiguous-prefix trim — silently keeping only
the prefix would drop real autosomal data. So the code scans the whole tail past the
prefix and, if it finds any autosome there, **throws `std::invalid_argument`** naming
the offending index and telling the user to sort the panel, pre-filter to autosomes,
or pass `--no-auto-only`. A fail-fast reject, never silent data loss. A general
interspersed keep-mask is out of scope until the pack kernel and window tiling learn a
per-SNP mask (scope T3).

Two guard cases: if the `.snp` chromosome column is shorter than the genotype SNP
axis it can't be applied at all (throw), and if the resulting prefix is empty it's
`InvalidConfig`.

---

## 5. The resident device bit-matrix

`be.readv2_alloc_bitmatrix(N, window_snps, m0)` allocates a `[sample × SNP-window]`
bit-matrix that lives in VRAM across the whole sweep. It is zeroed at allocation so
that any padding bits (the last window is rounded up to a whole number of 64-bit
words) stay valid = 0 and never register as a mismatch. The host never touches the
packed bits — it only streams packed chunks in and reads the reduced scalars out.

The single-allele-bit layout is what makes READv2 v1 pseudo-haploid-only: there is one
bit per sample per SNP, so there is simply no encoding for a heterozygote. That is why
the ploidy gate in section 6 exists.

---

## 6. Reading the SNP axis as one whole-genome tile

The SNP-streaming loop is written as a loop but, today, runs exactly once. Here's why.

The canonical-tile readers only support a byte-aligned SNP prefix starting at 0: both
TGENO (`GenoReader::read_tile`) and the SNP-major gather hard-reject any nonzero
`snp_begin` — the nonzero-begin tile loop ("M5") isn't implemented yet. So chunking
the SNP axis would issue a second, rejected read on any dataset bigger than one chunk.
The code therefore sets `chunk_snps = m0` and reads the whole SNP axis in one tile.

That is both correct and memory-safe: for the fixed ~1.24M-SNP 1240K panel the whole
SNP axis per sample is small and bounded, and the device bit-matrix already holds all
`m0` SNPs resident regardless of how the read is chunked. The loop stays a loop on
purpose — the moment the reader grows a nonzero-`snp_begin` gather (scope T3), this
becomes true SNP-tile streaming with no restructuring, and each chunk is already a
whole number of windows.

### The ploidy gate

On the **first** streamed chunk (which covers the 1000-SNP detection window),
`io::detect_sample_ploidy` runs over the tile. Any sample that isn't
`kPloidyPseudoHaploid` — i.e. any diploid/het sample — is an immediate fail-fast:
the code collects up to eight offender labels and throws `std::invalid_argument`
explaining that READv2 v1 requires pseudo-haploid hardcalls because the
single-allele-bit layout has no encoding for a het. `std::invalid_argument` is the
project's marker for a fail-fast *input* reject (the command maps it to
`kExitInvalidConfig`), distinct from a device runtime error. The gate fires *before*
the all-pairs enumeration, so a bad input is caught cheaply rather than after a full
sweep.

Each chunk is then packed into the bit-matrix with `be.readv2_pack_chunk`.

---

## 7. The sweep, the gate, the background, the rows

### The sweep

`be.readv2_mismatch(bits, n_pairs, opts.tiled)` runs the all-pairs windowed-mismatch
reduction on the GPU and returns a `device::Readv2Pairs`: four host-resident vectors,
one entry per unordered pair `r`, indexed in flat rank order `r = C(j,2)+i` for the
pair `i<j`:

| Field | Meaning |
|---|---|
| `sum_p0[r]` | Sum over windows of the per-window mismatch proportion P0. |
| `sum_p0_sq[r]` | Sum over windows of P0², the raw material for the window jackknife SE. |
| `n_win_used[r]` | Number of windows that had any comparable sites for this pair. |
| `tot_comp[r]` | Total comparable (non-missing-on-both) sites — the overlap count. |

### Pass 1 — form `P0_mean` and gate

For each pair it forms `P0_mean = sum_p0 / n_win_used` and applies three exclusion
tests. A pair is **kept** only if all pass:

- `n_win_used > 0` **and** `tot_comp > 0` — a pair with zero windows or zero overlap
  is excluded from *both* the background and the output. This is the guard against a
  0/0 NaN silently poisoning the median.
- `tot_comp >= min_overlap * m0` — the minimum-overlap gate. Too little shared data,
  and the pair is dropped.
- `P0_mean` is finite.

Surviving pairs get `keep[r] = 1`, their `P0_mean` recorded, and the value pushed into
the `finite` vector that the background reduces over.

If nothing survives, that's a **clean empty run**, not an error: `background` is set to
NaN and `Status::Ok` is returned. There is simply nothing to normalize against.

### The background

The background is the median (default) or mean of the survivors' `P0_mean`. Median is
computed by sorting `finite` and taking the middle element (or the average of the two
middle elements for an even count). This is the all-pairs normalizer that turns a raw
mismatch rate into a relatedness-comparable `P0_norm`.

One degenerate guard: if `background` isn't strictly positive — which would require
more than half the pairs to match perfectly, impossible on real AADR data — it returns
`InvalidConfig` rather than emitting an `inf` from the division below.

### Pass 2 — emit

For each kept pair it un-ranks the flat index `r` back to `(i, j)` via
`unrank_pair_host` (section 9), computes `P0_norm = P0_mean / background`, and fills a
`Readv2PairRow`: the pair indices, `n_windows`, `n_overlap`, `P0_mean`, `P0_norm`, the
classified `degree` (`rc::degree_from_p0norm`), and the `z` confidence
(`rc::readv2_z`, which returns NaN when `n_windows < 2` because the SE is undefined).
Each row is handed to `sink` immediately. The tally of emitted rows lands in
`res.n_emitted`, and the run finishes `Status::Ok`.

The degree classification and the `z` statistic themselves live in
`readv2_classify.cpp` — this driver only calls them. See that file's reference for the
cut points and the Z-upper convention.

---

## 8. Precision: integer on device, one FP64 ratio on host

READv2's precision story is simple and deliberately locked. The mismatch counting is
**integer `__popc`** on the device — exact by construction, no floating-point in the
hot loop. The only floating-point step of consequence is the single normalization
ratio `P0_mean / background`, done in **native FP64 on the host**.

It is emphatically **not** emulated-FP64. The rest of steppe defaults to emulated-FP64
for its matmul-heavy fit paths, but READv2 has no GEMM and no long cancellation chain —
a popcount ratio does not need the Ozaki treatment. So `res.precision_tag` is fixed to
`Precision::Kind::Fp64` and the run advertises exactly that. This is a scope-locked
decision, not a tunable.

---

## 9. Two small host helpers

- **`pair_count(n)`** — C(N,2) with an overflow-safe guard: it returns 0 for `n < 2`
  rather than computing a negative product.
- **`unrank_pair_host(r, i, j)`** — the host mirror of the device's
  `readv2_unrank_pair`. It inverts the flat pairing `r = C(j,2) + i` (with `i < j`)
  back to `(i, j)`. It seeds `j` from the closed-form quadratic
  `j ≈ (1 + √(1+8r)) / 2`, then nudges it up or down by one to correct for
  floating-point rounding at the boundary. Keeping this exactly in step with the
  device unrank is what guarantees the emitted `(i, j)` labels match the pair the GPU
  actually reduced.

---

## 10. Contracts and edge cases at a glance

- **Returns, doesn't throw, for configuration problems** it can foresee: `N < 2`,
  non-positive `window_snps`, empty SNP axis, empty autosome prefix, non-positive
  background all come back as `Status::InvalidConfig`.
- **Throws `std::invalid_argument`** for fail-fast *input* rejects: an
  autosome-restriction layout it cannot handle (section 4) and a diploid/het sample
  (section 6). The command layer maps these to the invalid-config exit code.
- **An empty result is success.** No pair clearing the gates yields `Status::Ok`, a
  NaN background, and zero emitted rows — a legitimate clean run.
- **Rows stream in ascending pair-rank order**, never all resident, always through
  `sink`.
- **The SNP loop runs once today** but is structured to become true tile streaming the
  moment the reader supports a nonzero SNP begin.
- **Pseudo-haploid only** in v1 — the single-allele-bit layout has no het encoding,
  and the ploidy gate enforces it before any enumeration.
