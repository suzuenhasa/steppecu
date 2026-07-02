# `snp_filter.hpp` reference

## 1. Purpose

`src/io/filter/snp_filter.hpp` turns one decoded block ("tile") of genotype data
into a single yes/no keep decision for each SNP in that tile. Given the decoded
per-population allele frequencies and counts, the per-population sample sizes, and
the SNP metadata (allele letters and chromosome), it derives three summary numbers
per SNP, applies every active quality filter, and produces one boolean per SNP: keep
or drop.

Two rules shape everything here:

- **One decision per SNP, never per (population, SNP).** A SNP is kept or dropped for
  all populations at once. The pooled-frequency and missing-fraction numbers each
  reduce *across* the kept populations into a single scalar for the SNP, so a drop
  zeroes a whole SNP column, never a single cell. Downstream matrix math depends on
  this: it assumes the validity mask is a clean 0/1 mask over whole columns.
- **Drop, never flip.** A SNP that fails a filter is dropped. No allele letter or
  frequency is ever altered. In particular, strand-ambiguous (palindromic) SNPs are
  dropped, never strand-corrected.

The file is a leaf of the `io` layer: plain host C++20, no CUDA, and no dependency on
the core or device libraries. It deliberately takes plain `double` pointers rather
than any internal matrix-view type, so it stays lightweight and free of GPU headers.
It does consume the exact same column-major arrays the decode backend produces — it
just does not reach into the core/device code to get them.

The actual per-SNP predicates (what counts as strand-ambiguous, multiallelic, a
transversion, an autosome, and where each numeric threshold applies) live once in a
shared predicate header, `filter_decision.hpp`, and the across-population reduction
plus the combined keep/drop rule live once in `snp_summary_reduce.hpp`. This file
wires those shared pieces to the decoded tile; it does not re-implement any rule.

---

## 2. `DecodedTileSummaryInput`

The decoded contents of one tile, as plain arrays plus the sizing metadata. This is
exactly what the decode backend returns (its `q` / `v` / `n` outputs), together with
the per-population sizes the sample-partition plan supplies.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `q` | `const double*` | `nullptr` | The reference-allele frequency for each (population, SNP), stored column-major as a `[P × M]` array — the element for population `i`, SNP `s` lives at index `i + P·s`. |
| `v` | `const double*` | `nullptr` | The validity mask for each (population, SNP): `1` where the cell has data, `0` where it is missing. Same `[P × M]` column-major layout. Present for completeness but not read by this file (see below). |
| `n` | `const double*` | `nullptr` | The non-missing haploid count for each (population, SNP): ploidy times the number of individuals with data. Same `[P × M]` column-major layout. |
| `P` | `int` | `0` | The number of (kept) populations — the row count of the arrays above. |
| `M` | `long` | `0` | The number of SNPs in the tile — the column count. |
| `pop_individuals` | `vector<size_t>` | empty | The per-population count of *kept* individuals, length `P`. This is the denominator of the per-SNP missing fraction: the total individuals (missing plus non-missing) in each population, summed across populations. The caller supplies it from the sample partition intersected with the kept-sample set. |
| `ploidy` | `int` | `2` | Samples' ploidy: `2` for diploid, `1` for pseudo-haploid. Used to convert the haploid count `n` back into a non-missing *individual* count (`n / ploidy`) for the missing fraction. Must be `1` or `2`; any other value is rejected (see Section 4). Default `2` is the common ancient-DNA case. |

### Why `v` is present but unused

By the decode contract, a cell is missing exactly when its count `n` is `0`, which is
also exactly when its validity `v` is `0`. The reductions in this file only ever
multiply by or sum `n`, so a missing cell already contributes zero without consulting
`v`. The mask is therefore redundant for these particular sums and is intentionally
left unread.

### `pop_individuals` vs. `n`

These are easy to confuse. `n` is a *haploid, non-missing* count that varies per SNP
(it drops as data goes missing). `pop_individuals` is a *fixed per-population* head
count of all individuals assigned to that population, missing or not — it does not
vary by SNP. For a fully-observed diploid SNP, `n` for a population equals
`2 × pop_individuals` for that population, but at a partially-missing SNP `n` is
smaller. The missing fraction is built from both: how many individuals actually had
data (`n / ploidy`) over how many there were in total (`pop_individuals`).

---

## 3. `PerSnpSummary`

The three-plus-one derived numbers per SNP that the filters actually read. Every value
is a reduction *across* the kept populations — a single scalar describing the whole
SNP, not a per-population quantity.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pooled_ref_af` | `double` | `0.0` | The pooled reference-allele frequency: the sum over populations of (reference count) divided by the sum over populations of (allele count). `0` when the SNP has no data at all. |
| `pooled_minor_af` | `double` | `0.0` | The pooled *folded* minor-allele frequency: `min(pooled_ref_af, 1 - pooled_ref_af)`. This is the value the MAF filter compares against. |
| `missing_frac` | `double` | `1.0` | The per-SNP missing fraction along the sample axis: `1 - (non-missing individuals) / (total individuals)`, both summed across populations. `1.0` (fully missing) when the total is `0`. |
| `pooled_allele_count` | `double` | `0.0` | The pooled allele count: the sum over populations of `n`. A value of `0` means the SNP is entirely missing. |

The "pooled folded" MAF is a specific, deliberate convention: the reference and allele
counts are pooled across all kept samples *first*, and the minor allele is chosen
*after* pooling. It is not a per-population frequency, and not an average of
per-population frequencies. This matches how the reference genotype-summary pipeline
pools its per-population sums.

The missing fraction here is measured over *individuals* (the PLINK `--geno`
convention). Note that ADMIXTOOLS 2's similar option measures its fraction over
*populations* instead, which is a different denominator.

---

## 4. `derive_per_snp_summary`

Computes a `PerSnpSummary` for every SNP in the tile and returns them as a length-`M`
vector, parallel to the SNP axis. This is the single place the pooled MAF and missing
fraction are derived, so the mask builder and an oracle test both read the same
numbers.

For each SNP `s`, summing population `pop` over `0 … P-1`:

- pooled reference count = sum of `Q(pop, s) × N(pop, s)`
- pooled allele count = sum of `N(pop, s)`
- `pooled_ref_af` = pooled reference count / pooled allele count (`0` if the denominator is `0`)
- `pooled_minor_af` = `min(pooled_ref_af, 1 - pooled_ref_af)`
- non-missing individuals = sum of `N(pop, s) / ploidy`
- total individuals = sum of `pop_individuals[pop]`
- `missing_frac` = `1 - non-missing individuals / total individuals` (`1` if the total is `0`)

A missing cell needs no special-casing: its `Q` is `0` and its `N` is `0`, so it
contributes zero to both the reference count and the allele count.

### Preconditions (fail-fast)

When `P > 0` and `M > 0`, the function throws `std::invalid_argument` — up front,
before touching the data — if any of these is violated. Each guard exists because the
alternative is a silent wrong answer or a crash three frames deep, not a clean error.

- **`ploidy` must be `1` or `2`.** A misset ploidy fabricates a wrong missing fraction
  (it is the divisor turning haploid counts into individual counts).
- **`pop_individuals.size()` must equal `P`.** An earlier version guarded only the
  denominator loop and still ran the numerator loop over all `P` populations, so a
  short `pop_individuals` understated the missing-fraction denominator and could produce
  a *negative* missing fraction, which then spuriously *passed* the geno filter — a
  missing-heavy SNP wrongly kept. That is a silent wrong answer, so it is now a hard
  error.
- **`q` and `n` must be non-null.** The reduction dereferences both once per cell, so a
  null pointer would otherwise segfault deep in the loop instead of surfacing a
  diagnosable error.

---

## 5. `snp_keep_decision`

The single keep/drop decision for one SNP, given its derived summary, its allele pair,
its chromosome, the filter configuration, and one precomputed membership bit. It is a
pure `inline`, `noexcept` function. It exists as a named entry point so that this host
mask builder and a future GPU keep-mask kernel run the *identical* decision over the
*identical* predicates and cannot drift apart at a boundary. It simply copies the
summary into the small plain struct the shared decision body takes and calls that body.

The `membership_ok` argument is the resolved include/exclude/prune-in result for this
SNP's id: `true` when the id passes the membership rule, or when membership imposes no
constraint at all. The caller precomputes it so the decision itself stays pure and does
no lookups.

### The keep order

A SNP is kept only if it passes *every* active filter, checked in this fixed order.
The order matters because the first failing check short-circuits the rest.

1. **Unconditional class drop — multiallelic.** If the ref/alt pair is not a clean pair
   of two distinct A/C/G/T bases (either allele is non-ACGT, or ref equals alt), the
   SNP is dropped no matter what any flag says.
2. **Strand-ambiguous drop (flag-gated).** Only when the strand mode is `Drop` (the
   default): a palindromic pair (A/T or C/G, which reads the same on either strand) is
   dropped. Under the `Keep` or `Flip` strand modes these palindromes are retained. The
   multiallelic drop above stays unconditional even under `Keep`/`Flip`, so those modes
   still require a clean biallelic ACGT pair.
3. **MAF.** Keep only if the pooled folded MAF is at least `maf_min` (inclusive).
4. **Geno.** Keep only if the missing fraction is at most `geno_max_missing`
   (inclusive).
5. **Drop-monomorphic (flag-gated).** When set, drop a SNP whose pooled folded MAF is
   exactly `0` (no variation). The test reads the *unfolded* pooled reference frequency
   and folds it once inside.
6. **Transversions-only (flag-gated).** When set, drop transitions (A↔G, C↔T), keeping
   only transversions.
7. **Autosomes-only (flag-gated).** When set, keep only chromosomes 1–22.
8. **Membership.** Finally, keep only if the precomputed membership bit is `true`.

Because the multiallelic check short-circuits first, the strand gate being placed after
it means that at the default `Drop` mode the combined result is byte-for-byte what a
single "multiallelic OR ambiguous" test would have produced — so adding the `Keep`/`Flip`
modes did not perturb any default-path result.

---

## 6. `build_snp_keep_mask`

The main entry point. It derives the per-SNP summaries (Section 4), then for each SNP
looks up the allele pair and chromosome from the SNP table, resolves the membership bit
from the resolved include/exclude/prune-in set, and calls `snp_keep_decision`. It
returns a length-`M` `std::vector<bool>`: `true` to keep each SNP, `false` to drop it.

- Thresholds come from `cfg` (the `FilterConfig`).
- The allele-pair letters and chromosome come from `snps` (a `SnpTable`), whose columns
  are parallel to the tile's SNP axis.
- Membership comes from `mem` (a `SnpMembership`, the resolved include/exclude/prune-in
  keep-set and drop-set). When that membership imposes no constraint, the per-SNP lookup
  is skipped entirely.

### Behavior at the default configuration

With a default `FilterConfig` — `maf_min = 0`, `geno_max_missing = 1`, every flag off,
empty membership — every numeric and flag filter is a no-op: MAF at least `0` is always
true, missing fraction at most `1` is always true. The *only* filter that can still fire
is the unconditional multiallelic/strand-ambiguous class drop. So at the default this
keeps every SNP whose allele pair is a clean biallelic ACGT non-palindrome.

For the real ancient-DNA HO panel, that means it keeps *everything*: that panel has no
A/T or C/G palindromes and no non-ACGT alleles, so the class drop removes nothing. The
oracle test confirms it drops zero SNPs there, which is the "no-op when default" property
the parity path relies on.

### Preconditions (fail-fast)

When `M > 0`, the function throws `std::invalid_argument` if the SNP table is too short
for what is being asked of it:

- `snps.ref.size()` and `snps.alt.size()` must each be at least `M` — the allele pair
  drives the unconditional class drop for every SNP.
- `snps.chrom.size()` must be at least `M` when `autosomes_only` is active.
- `snps.id.size()` must be at least `M` when membership actually constrains anything.

An earlier version silently substituted `'N'` / `0` / an empty id for a too-short table,
which made the multiallelic test fire on the fabricated `'N'`/`'N'` pair and quietly
dropped the tail of the tile (or altered membership). A mismatched SNP-metadata and
decode length is a wiring or data bug, so it must abort with context rather than return a
plausible-looking wrong mask. The function also propagates the Section 4 preconditions
(bad ploidy, mismatched `pop_individuals`, null `q`/`n`) via `derive_per_snp_summary`.
