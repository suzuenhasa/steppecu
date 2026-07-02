# `filter_decision.hpp` reference

## 1. Purpose

`src/io/filter/filter_decision.hpp` is the single source of truth for every
keep-or-drop decision steppe makes when it reads genotypes. It holds a small set
of pure functions — one per rule — that classify a SNP or a sample and return a
plain `true`/`false` meaning "keep." Three different callers use these same
functions: the in-tile filter that runs while streaming data, the separate
pre-pass that decides per-sample drops, and the tests. Because the rule lives in
exactly one place, those three paths cannot disagree on a boundary case or on how
an allele pair is classified.

Two properties are worth stating up front:

- **Every threshold comes from a `FilterConfig` field.** There are no bare magic
  numbers in the decision logic — a filter cutoff is always passed in, never typed
  into the predicate.
- **No function here ever alters a genotype, an allele, or a frequency.** These
  are classifiers. They inspect a value and return a decision. Changing data
  (for example, flipping a strand) is never done here.

The header is a leaf of the input/output layer. It is pure host C++20, uses no
CUDA, and depends on nothing in the core or device layers. It includes only the
CUDA-free public config header (for the `FilterConfig` thresholds and the
autosome chromosome bounds) and the standard library. The same functions are
marked so they can also compile for the GPU, but the file itself reaches into no
GPU code.

---

## 2. Filter scope: the non-negotiable invariants

Four rules define what these filters are allowed to do. They are structural, not
tuning knobs.

- **Drop by default, flag-gated.** Multiallelic SNPs are *always* dropped.
  Strand-ambiguous SNPs (an A/T or C/G pair) are dropped by default — the
  merge-safety choice — but can be retained by choosing a different strand mode,
  which is the path that reproduces ADMIXTOOLS 2. No filter ever changes a
  genotype value; even when strand-ambiguous SNPs are kept, steppe does not try to
  infer the correct strand. (The frequency-based reorientation that a "flip" mode
  would do is an accepted-but-not-yet-implemented option.) The actual gating
  happens once, elsewhere; the functions in this file are just the pure
  classifiers.
- **No linkage-disequilibrium computation.** An externally supplied pruned-SNP
  list is *read*, never computed. steppe does not calculate LD itself.
- **No on-disk rewrite.** The filters produce a *plan* — a keep-mask over SNPs
  and a set of kept samples — that is consumed per tile as data streams by. No
  filtered copy of the data is written out.
- **SNP-global or sample-global, never per-(population, SNP).** A SNP is kept or
  dropped for *all* populations at once, and a sample is kept or dropped for *all*
  SNPs. A per-(population, SNP) drop is forbidden because the downstream f2
  computation relies on a clean 0/1 mask: a drop must zero a whole SNP column or a
  whole sample, never a single cell. The functions here return one boolean *per
  SNP* (or *per sample*), so this invariant is enforced by their shape — there is
  no way to express a per-cell drop.

---

## 3. The pooled folded MAF convention

"MAF" (minor allele frequency) is defined here in exactly one place, because the
definition is a genuine choice and the rest of the codebase must agree on it.

The MAF used is the **pooled folded** minor allele frequency:

```
pooled_ref_af = (sum over kept pops of ref_count) / (sum over kept pops of allele_count)
MAF           = min(pooled_ref_af, 1 - pooled_ref_af)
```

Two things make this specific:

- **Pooled, not per-population.** The reference-allele counts and the total allele
  counts are summed across all kept samples (in the kept populations) *first*, and
  the frequency is taken of those sums. This is not a per-population frequency, and
  it is not the average of the per-population frequencies. In terms of the decoded
  per-population genotype `Q` and allele-count `N` values, the pooled reference
  count is `sum of Q·N` and the pooled allele count is `sum of N`. This matches how
  the reference pipeline pools its counts.
- **Folded.** "Folded" means the smaller of the reference-allele frequency and one
  minus it — so MAF always lands in `[0, 0.5]`. The allele that happens to be
  labelled "reference" does not matter; the minor allele is whichever is rarer.

ADMIXTOOLS 2's minimum/maximum-MAF documentation does not say whether it pools or
averages. The pooled reading is the one that matches steppe's reference pipeline
and is the documented best interpretation.

### `folded_maf`

`folded_maf(pooled_ref_af)` returns `min(q, 1 - q)`. It is the one place the
folding is implemented, shared by the MAF filter and the monomorphic test so the
fold lives once.

---

## 4. Threshold predicates and their pinned boundaries

Three functions apply a numeric cutoff. Each returns the keep decision, and each
has a deliberately-chosen boundary side (whether a value *exactly equal* to the
threshold is kept). The boundary side is pinned so that the default configuration
is a guaranteed no-op.

| Function | Keeps a SNP/sample iff | Boundary |
|---|---|---|
| `snp_passes_maf(pooled_minor_af, maf_min)` | pooled folded MAF `>= maf_min` | inclusive `>=` |
| `snp_passes_geno(per_snp_missing_frac, geno_max_missing)` | per-SNP missing fraction `<= geno_max_missing` | inclusive `<=` |
| `sample_passes_mind(per_sample_missing_frac, mind_max_missing)` | per-sample missing fraction `<= mind_max_missing` | inclusive `<=` |

The choice of `>=` for MAF and `<=` for the two missing-data caps is what makes
the default `FilterConfig` (`maf_min = 0`, `geno_max_missing = 1`,
`mind_max_missing = 1`) keep everything: a MAF is always `>= 0`, and a fraction is
always `<= 1`. The standard path is therefore untouched unless a filter is
explicitly requested. A SNP whose value sits exactly on the threshold is kept in
all three cases.

Notes on each:

- **MAF filter (`snp_passes_maf`).** The first argument is *already folded* —
  pass `folded_maf(pooled_ref_af)` for it. Keeps a SNP whose folded MAF is at least
  `maf_min`.
- **Geno filter (`snp_passes_geno`).** The missing fraction here is measured over
  the *sample/individual axis* — the fraction of kept individuals that have missing
  data at that SNP. This follows the PLINK `--geno` convention. (ADMIXTOOLS 2's
  similar filter uses a different denominator, over populations rather than
  individuals.)
- **Mind filter (`sample_passes_mind`).** Drops a *sample* whose missing fraction
  across all SNPs exceeds the cap. This one cannot be decided from a single tile of
  data, so it is evaluated in a separate streaming pre-pass over all SNPs.

---

## 5. The monomorphic test

`is_monomorphic(pooled_ref_af)` reports whether a SNP has no variation at all
across the kept samples — its pooled reference-allele frequency is exactly `0.0`
or exactly `1.0`, equivalently its pooled folded MAF is exactly `0`. It backs the
"drop monomorphic SNPs" flag. It is effectively the strict-positive boundary of
the MAF filter, kept as its own named function.

Two contract points matter:

- **Pass the unfolded pooled reference frequency, not the folded minor
  frequency.** The parameter is the pooled reference-allele frequency in `[0, 1]`,
  and this function folds it once internally. Passing an already-folded value would
  fold it a second time. That double-fold happens to be harmless only because
  folding a value already in `[0, 0.5]` leaves it unchanged — but relying on that
  is exactly the kind of "two call sites assuming different conventions" bug the
  single-source design exists to prevent, so the parameter name and the body are
  kept honest: give it the unfolded reference frequency.
- **The exact `== 0.0` comparison is intentional and safe.** A truly monomorphic
  site has each per-population `Q` value exactly `0.0` or exactly `1.0`, so each
  `Q·N` product is exact and the pooled reference frequency comes out exactly `0.0`
  or `1.0`. A SNP with even a single copy of the minor allele is therefore *not*
  monomorphic. This exactness would break if the upstream pooling were ever changed
  to average per-population frequencies (which rounds), so that change must not be
  made.

---

## 6. Allele-pair class predicates

These functions classify a SNP by its reference/alternate allele letters (the
`ref`/`alt` characters from the `.snp` file). They inspect the declared pair and
return a class; they never change an allele.

### Normalizing an allele letter

`normalize_allele(a)` maps an allele character to an uppercase `A`, `C`, `G`, or
`T`, and returns the null character `'\0'` for anything else (including `N`, `0`,
`-`, `.`, and indel or X codes). It uppercases by clearing bit `0x20`, which is
device-safe (it needs no locale-aware library call) and is bit-for-bit identical
to a standard uppercase conversion for the A/C/G/T letters. Having this in one
place lets every class predicate treat lower- and upper-case identically and
reject non-ACGT input cleanly.

### The Watson–Crick complement

`complement(a)` returns the complementary base — `A`↔`T`, `C`↔`G` — or `'\0'` if
the input is not a clean base. It is used only by the strand-ambiguity test; this
is the one place strand complementarity appears at all, and steppe *drops*
ambiguous pairs rather than flipping by them.

### Strand-ambiguous (palindromic) pairs

`is_strand_ambiguous(a, b)` is true when the ref/alt pair is its own complement,
so the two DNA strands read the same and cannot be told apart without extra strand
information. Those pairs are `{A, T}` and `{C, G}` (the four ordered forms AT, TA,
CG, GC). Such a SNP is dropped by default, never strand-flipped. Any non-ACGT
input is not ambiguous (the multiallelic test drops it instead). Equivalently, the
pair is ambiguous when one allele equals the complement of the other.

**A deliberate convention choice worth understanding.** This uses the standard
genetics definition of strand-ambiguous / palindromic: the self-complementary
pairs A/T and C/G. That is *not* the same thing as the transition/transversion
split. An example list phrased as "GT/AC dropped, GA/CT kept" describes the
transition-versus-transversion distinction — GT and AC are transversions, GA and
CT are transitions, and none of those four is self-complementary. steppe
implements the genetics-correct palindrome test here and exposes the
transition/transversion split through its own separate functions (below). On the
real AADR Human Origins panel there are no A/T or C/G pairs left at all — they were
removed upstream — so this predicate correctly drops zero SNPs there; the large
number of sites one might expect it to drop are actually the transversions,
handled by the transversions-only flag.

### Multiallelic / not-a-clean-SNP

`is_multiallelic(a, b)` is true when the pair is *not* two distinct A/C/G/T bases
— that is, either allele is non-ACGT (an `N`, `0`, `-`, or indel code), or the two
alleles are equal. Such a site is dropped. Because this reader only sees the
declared ref/alt pair, "multiallelic" here means "not a declarable clean biallelic
SNP"; genuine sites with more than two alleles arrive either as a non-ACGT code or
already split upstream.

### Transition and transversion

Both classify a substitution by the two DNA ring classes: purines are A and G,
pyrimidines are C and T.

- `is_transition(a, b)` — true for a within-ring change (A↔G or C↔T).
- `is_transversion(a, b)` — true for an across-ring change (A/G ↔ C/T). Used by
  the transversions-only flag, which keeps a SNP only if it is a transversion.

Both first check `is_multiallelic`: a pair that fails the clean-biallelic-ACGT
rule is neither a transition nor a transversion. The clean-pair rule lives once in
`is_multiallelic`, and over clean pairs the two classes are exact complements of
each other (`is_transversion` is simply "not a transition"), so the two functions
can never disagree on a clean pair.

---

## 7. The autosome (chromosome) predicate

`is_autosome(chrom)` is true when the chromosome code is in the range 1 through
22. The bounds come from named constants in the config header rather than a bare
`22` in this file. This is the ADMIXTOOLS-2-parity autosome set: extract_f2's
default keeps only chromosomes 1–22, so the sex chromosomes (X as 23, Y as 24) and
mitochondrial or other codes are not autosomes. It backs the autosomes-only flag.
