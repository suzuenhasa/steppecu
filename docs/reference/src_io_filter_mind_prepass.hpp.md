# `mind_prepass.hpp` reference

## 1. Purpose

`src/io/filter/mind_prepass.hpp` declares the streaming pre-pass behind the
per-sample missing-data filter, known as `--mind`. This filter drops any sample
(individual) whose fraction of missing genotype calls, measured across *all* SNPs,
is too high.

The reason this filter needs its own pass is that it cannot be decided from a single
tile of data. The other quality-control filters — minor-allele frequency and the
per-SNP missing filter — look at one SNP at a time and can be applied while the data
streams past. Whether a *sample* has too much missing data, by contrast, depends on
every SNP that sample was ever seen at. So this pass streams over all SNPs once, up
front, counting how many non-missing calls each sample has, and produces the set of
sample indices that survive.

The pass reads genotypes through the exact same byte-and-2-bit decoding path used by
the main decode stage, so its idea of "missing" is guaranteed to be identical to the
rest of the pipeline. It contains no GPU code; it is plain host C++.

The header declares two data structures — one for the input, one for the result —
and a single function that runs the pass.

---

## 2. When the pre-pass runs, and when it is skipped

The pass only does work when the filter is actually turned on. The filter is
controlled by `cfg.mind_max_missing`, the maximum per-sample missing fraction a
sample may have and still be kept.

- If `mind_max_missing` is `1.0` (or anything at or above `1.0`), the filter is
  off. Since `1.0` is the largest possible missing fraction, no sample can ever
  exceed it, so nothing can be dropped. In this case the function is a no-op with
  respect to dropping: it still fills in and reports every sample's missing fraction
  so a caller can observe those numbers, but the kept set is simply every sample
  index from `0` to `n_individuals - 1`.
- If `mind_max_missing` is below `1.0`, the filter is active and the pass performs
  its full per-sample accumulation and resolves the kept set.

Callers that want to avoid the streaming cost entirely can check the threshold
themselves and skip calling the pass; the value `1.0` is the single agreed-upon
"filter is off" sentinel shared with the configuration.

---

## 3. `MindPrepassInput` — the packed-genotype view

`MindPrepassInput` is a lightweight, non-owning view describing the packed genotype
records to stream over. The caller owns the underlying memory; this struct just
points at it. It contains no GPU code.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `packed` | `const std::uint8_t*` | `nullptr` | Pointer to the packed genotype bytes, laid out as one record per individual. |
| `bytes_per_record` | `std::size_t` | `0` | The stride, in bytes, from the start of one individual's record to the next. |
| `n_snp` | `std::size_t` | `0` | How many SNPs each record spans — the meaningful prefix length within a record. This is the denominator for the missing fraction. |
| `n_individuals` | `std::size_t` | `0` | The number of individual records, i.e. the number of samples. |

### How a single genotype is located

The layout is individual-major: all of one individual's SNP calls sit together in
one contiguous record, and records for different individuals follow one another. It
is the same packing the main genotype tile uses.

Within a record, each genotype is a 2-bit code, so four SNPs pack into one byte. For
individual `g` and SNP `s`, the call lives in the byte at
`g * bytes_per_record + s / 4`, and within that byte the four codes are read
most-significant-bits first (so SNP `s` occupies bit position determined by
`s % 4`, counting from the top of the byte). A specific reserved code marks a call
as missing. This bit order and missing code are shared with the decode front-end by
construction, which is why the two never disagree about which calls are missing.

---

## 4. `MindSummary` — the per-sample result

`MindSummary` is the output of the pass. Every vector is parallel to the sample
(individual) axis, so entry `g` in `nonmissing` and `missing_frac` both describe
sample `g`. It is exposed in full (not just the kept set) so that a reference test
can recompute the same numbers independently and compare them exactly.

| Field | Type | Meaning |
|---|---|---|
| `nonmissing` | `std::vector<std::size_t>` | For each sample, the count of SNPs that were *not* missing. Length equals `n_individuals`. |
| `missing_frac` | `std::vector<double>` | For each sample, its missing fraction, computed as `1 - nonmissing / n_snp`. Length equals `n_individuals`. |
| `kept` | `std::vector<std::size_t>` | The indices of the samples that pass the filter, in ascending order. This is the set the rest of the pipeline uses. |

---

## 5. `run_mind_prepass` — behavior and edge cases

```cpp
[[nodiscard]] MindSummary run_mind_prepass(const MindPrepassInput& in,
                                           const FilterConfig& cfg);
```

The function streams over every SNP of every individual, accumulates each sample's
non-missing count, converts that to a missing fraction, and decides which samples
pass against `cfg.mind_max_missing` using the same shared per-sample predicate the
rest of the code uses. It returns the fully populated `MindSummary`.

Two behaviors are worth calling out explicitly.

### The filter-off case

When `cfg.mind_max_missing` is at or above `1.0`, the pass drops nothing. It still
reports each sample's `nonmissing` count and `missing_frac`, but `kept` is every
sample index `0 .. n_individuals - 1`. See section 2.

### The no-data / zero-SNP case

If there are no SNPs to look at (`n_snp == 0`, or no packed data), a sample's
missing fraction is genuinely undefined — there is no SNP on which to base a "too
much missing data" judgment. In that situation the pass takes a **keep-all**
fail-safe: every sample is reported with `missing_frac = 0` and every sample is
kept, *even when the filter is active*. The no-data outcome is never "drop
everything."

This is deliberately the opposite of the per-SNP missing filter's convention. That
filter, facing an empty denominator, treats the fraction as `1.0` and drops. The two
diverge on purpose because they ask different questions. The per-SNP filter asks
"does this SNP have data across samples?", while `--mind` asks "does this sample
have data across SNPs?" — and with zero SNPs the second question simply has no
answer, so the safe response is to keep the sample rather than discard it on no
evidence.

---

## 6. Invariants and layering

### The result is sample-global

The kept set is an index set over the whole sample axis: a sample is kept for *all*
SNPs or dropped for *all* SNPs. There is no per-SNP or per-population variation in
membership — a sample is never kept for some SNPs and dropped for others. Everything
downstream can rely on the sample axis being consistent across the entire dataset.

### Where this header sits in the layering

This is a leaf header on the input/output (`io`) side of the codebase. It is plain
host C++20 with no CUDA and no dependency on the core computation or GPU-device
layers. Its only real dependency is the configuration header, from which it takes
`FilterConfig` (specifically the `mind_max_missing` threshold).

For the low-level bit extraction it uses the `io`-side byte helpers rather than the
core computation layer's decode helpers. Reaching into the core layer would be an
upward dependency, which the layering forbids. Both sides extract the same 2-bit
codes in the same fixed bit order, so using the `io`-side helpers is not just a
layering nicety — it produces byte-for-byte the same "missing" decisions as the main
decode path.
