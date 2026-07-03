# `decode_af.hpp` reference

## 1. Purpose

`src/core/internal/decode_af.hpp` is the single home of the genotype-decode and
allele-frequency math. Genotypes arrive packed two bits per sample; this file
turns those two-bit codes into a per-population, per-SNP reference-allele
frequency. It holds the constants that describe the packing, the small helpers
that unpack one code and test it, the two accumulation steps that fold codes into
running totals, and the two finalize steps that turn those totals into a frequency.

Everything in the file is written as tiny `inline` functions and `constexpr`
constants marked so they compile for **both** the CPU and the GPU (the
`STEPPE_HD` marker on the functions). That is the whole point of the file: the
CPU reference path and the GPU kernel both call these exact same functions, so
the two can never drift apart on any decode detail. The results have been checked
bit-for-bit against an independent reference (the `build_tgeno_matrix.py` oracle)
with a maximum difference of zero.

Because the file lives in the `core` layer, it deliberately does **not** include
anything from the input/output (`io`) layer. Several constants here are copies of
values that also live in the `io` reader. They are kept identical by construction
and a cross-layer test pins them equal, so the core decoder stays self-contained
without depending on the reader.

The term "AT2" throughout means ADMIXTOOLS 2 — the established tool steppe
reproduces. Any constant or formula marked as matching AT2 is a parity value: you
may rename it, but you must not change what it computes, or results stop matching
ADMIXTOOLS 2.

---

## 2. The two-bit genotype packing

Genotypes are stored two bits per sample, four samples to a byte. The four
constants here describe that packing and are the single source every decoder
reads from, so no kernel or reference loop ever types a bare `4`, `2`, `3`, or
`0x3` of its own.

| Constant | Value | What it's for |
|---|---|---|
| `kMissingGenotypeCode` | `3` | The two-bit code that means "no call" (missing genotype). Codes `0`, `1`, `2` are counts of reference-allele copies; code `3` is the missing sentinel. A missing code is left out of both the allele count and the sample count. |
| `kCodesPerByte` | `4` | How many two-bit codes are packed into one byte. Used to split a sample index into a byte offset (`index / 4`) and a position within the byte (`index % 4`). |
| `kBitsPerCode` | `2` | How many bits each code occupies. Times `kCodesPerByte` this is 8, one byte. |
| `kCodeMask` | `0x3` (`3`) | The bit mask that keeps just the low two bits of a code. It is *derived* from `kBitsPerCode` as `(1 << 2) - 1`, not typed as a bare `0x3`, so the mask and the shift used to extract a code both trace back to the one named radix. |
| `kHeterozygousGenotypeCode` | `1` | The two-bit code for a heterozygous call (exactly one reference-allele copy). This is the code the pseudo-haploid auto-detection looks for — see section 3. |

The missing code, the codes-per-byte, and the bits-per-code each also exist in the
`io` reader under matching names. They are the same values by construction and a
cross-layer equivalence test enforces it, so the two definitions cannot silently
diverge.

---

## 3. Ploidy and pseudo-haploid auto-detection

A sample's **ploidy** is how many haploid genomes it contributes: modern samples
are diploid (2), while many ancient-DNA samples are treated as pseudo-haploid (1).
ADMIXTOOLS 2 can auto-detect this per sample, and steppe reproduces that. The
constants here support that path.

| Constant | Value | What it's for |
|---|---|---|
| `kPloidyDetectSnps` | `1000` | The detection window. To classify a sample, its **first** 1000 SNPs are scanned for any heterozygous call. A sample with at least one het in that window is diploid; one with none is pseudo-haploid (a truly haploid genome can never be heterozygous). This matches ADMIXTOOLS 2's default detection count. If a record has fewer than 1000 SNPs, the whole record is scanned. |
| `kPloidyPseudoHaploid` | `1` | Pseudo-haploid ploidy (ancient DNA): each non-missing sample contributes 1 to the haploid count. |
| `kPloidyDiploid` | `2` | Diploid ploidy (modern, or any sample that carried a het in the detection window): each non-missing sample contributes 2. |
| `kPloidyDivisorBase` | `3.0` | The base of ADMIXTOOLS 2's haploidization divisor `value / (3.0 - ploidy)`. This is a frozen ADMIXTOOLS 2 literal — you may rename it, but the value must stay `3.0`. It **must** stay a floating-point `3.0`, not an integer: see section 5 for why the float is load-bearing. |

Only `1` and `2` are ever meaningful ploidy values; ADMIXTOOLS 2 only emits those
two. Ploidy is a per-sample piece of metadata, never inferred from the genotypes
during finalize — the detection-window scan above is the only place a genotype
influences it.

---

## 4. Extracting and testing a single code

Two helpers turn a packed byte into a usable code and say whether that code
carries data.

### `genotype_code(packed_byte, k)`

Returns the two-bit code for the sample at position `k` (0-based) within a packed
byte. The bits are laid out most-significant-first: position 0 is bits 7-6,
position 1 is bits 5-4, position 2 is bits 3-2, position 3 is bits 1-0. In other
words it computes `(byte >> (6 - 2·(k mod 4))) & 3`, with the shift and the mask
built from the named radix constants rather than bare numbers. This is the exact
same bit order the host reader uses, pinned by the cross-layer test, so the GPU
decoder, the CPU decoder, and the reader all unpack a byte identically.

### `genotype_valid(code)`

Returns true when a code is a real (non-missing) genotype — that is, when it is
not `kMissingGenotypeCode` (`3`). A missing genotype is excluded from both the
allele count and the sample count, so this is the test every accumulation step
gates on.

---

## 5. The two accumulation folds

Decoding walks over every sample at a SNP and folds each one's code into running
totals. There are two folds: a simple one for a fixed, uniform ploidy, and a
per-sample one that reproduces ADMIXTOOLS 2's auto-ploidy accumulation. Both are
single-homed here so the GPU kernel's reduction and the CPU reference's scalar
loop can never disagree on the arithmetic.

### `accumulate_genotype(code, ac, an)` — uniform ploidy

The simple fold. For a non-missing code it adds the raw code value (the number of
reference-allele copies) to `ac`, the running sum of reference-allele copies, and
increments `an`, the running count of non-missing samples. A missing code touches
neither. Ploidy is not involved here; it is applied later, once, in the finalize
step (section 7). The accumulators are 64-bit integers, which is far more headroom
than needed — the largest possible allele count is ploidy times the number of
samples, well under the range where a double could no longer represent the total
exactly.

### `accumulate_genotype_ploidy(code, ploidy, ac, n)` — per-sample ploidy

The auto-ploidy fold, byte-faithful to ADMIXTOOLS 2. For each non-missing sample
whose ploidy is a legal `1` or `2` it does:

- `ac` (a **double**) `+= code / (3.0 - ploidy)` — the ADMIXTOOLS-2-weighted
  reference-allele count.
- `n` (an integer) `+= ploidy` — the running haploid count summed per sample.

A missing code is excluded from both. A ploidy outside `{1, 2}` — which would only
happen if the per-sample metadata were set wrong — is also excluded from both,
rather than dividing by a zero-or-negative `3.0 - ploidy` or inventing a count.
This fail-soft behavior mirrors the finalize guard in section 7.

**Why the `3.0` must be a float.** A sample can be *classified* pseudo-haploid
(no het in its first 1000 SNPs) yet still carry a het call — code `1` — at some
SNP outside that detection window. For such a SNP the sample's ploidy is `1`, so
`3.0 - 1 == 2.0`, and `1 / 2.0 == 0.5`: ADMIXTOOLS 2 adds half a reference allele.
If `3.0` were an integer, `1 / 2` would truncate to `0` and silently drop that
half-allele, diverging from ADMIXTOOLS 2 by exactly 0.5 (this was a measured
frequency error of `0.004`, i.e. `1/250`, before the float fix). Because halves
are exact in floating point, the sum stays exact and the final frequency is still
a single exact divide.

For a diploid sample (ploidy `2`) the divisor is `3.0 - 2 == 1.0`, so `code / 1.0`
is just `code`. That means an all-diploid stretch of data produces exactly the
same totals as the simpler `accumulate_genotype` would with `n = 2·an` — the two
paths are bit-identical on modern data, and the per-sample path only ever differs
when a real pseudo-haploid het shows up.

---

## 6. `AfResult`

The decoded result for one (population, SNP). It is a plain data struct with no
methods, so it crosses the boundary between GPU and CPU code unchanged.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `q` | `double` | `0.0` | The reference-allele frequency, in the range `[0, 1]`. Set to `0.0` when there is no data. |
| `n` | `double` | `0.0` | The non-missing haploid count — ploidy times the number of non-missing samples. |
| `v` | `double` | `0.0` | A validity flag: `1.0` when there was usable data (a positive count and a positive ploidy), otherwise `0.0`. |

The in-class defaults of `0/0/0` are exactly the "no data / masked out" result, so
the finalize functions only ever have to write the fields on the has-data path.

---

## 7. Finalizing a frequency

Once the totals are accumulated, one finalize call turns them into an `AfResult`.
There are two, one per fold, and both do the frequency divide exactly once in
double precision — that single, final divide is what keeps the frequency exact
when the counts were accumulated as integers.

### `finalize_af(ac, an, ploidy)` — from uniform-ploidy totals

Takes the integer totals from `accumulate_genotype` plus the uniform `ploidy`, and
computes:

- `n = ploidy · an` (the haploid count),
- `q = ac / n` (the frequency, `0` when there were no samples),
- `v = 1.0` when there is data.

**The fail-soft ploidy guard.** The has-data test is `an > 0 && ploidy > 0` —
ploidy is folded *into* the validity check, not just the count. This matters
because if `ploidy` were `0` (uninitialized or mis-set metadata) with `an > 0`,
then `n` would be `0` and `q = ac / 0`. On this hardware that divide is a
well-defined IEEE-754 infinity or NaN, and without the guard the result would
still report `v == 1` — a non-finite frequency slipping through the "we have data"
branch unmasked, poisoning every later computation that touches that column. With
the guard, a non-positive ploidy degrades cleanly to the masked-out `{0, 0, 0}`
result instead of silently corrupting. The upstream `io` filter rejects an illegal
ploidy outright by throwing; this device-side primitive has no throw path on the
GPU, so it masks out instead. The two agree on the contract — a bad ploidy never
produces a valid-looking result — they just enforce it differently.

### `finalize_af_counts(ac, n)` — from per-sample-ploidy totals

The counterpart for the per-sample fold. Here the ploidy has *already* been folded
into `ac` and `n` by `accumulate_genotype_ploidy`, so this just does the single
divide: `q = ac / n` (with `n` the summed haploid count), `v = 1.0` when `n > 0`,
and the masked-out `{0, 0, 0}` otherwise. On all-diploid data this returns exactly
what `finalize_af(ac, an, 2)` would, so the modern-data path is bit-identical
whichever fold produced the totals.
