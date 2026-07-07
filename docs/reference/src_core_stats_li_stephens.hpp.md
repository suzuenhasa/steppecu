# `li_stephens.hpp` reference

## 1. Purpose

`src/core/stats/li_stephens.hpp` is the small, host-only math kit that feeds the
Li-Stephens haplotype-copying engine behind `steppe paint`. It holds the
CUDA-free numerics that turn a genotype triple — its cM genetic map plus the size
of the donor panel — into the three inputs the forward-backward actually runs on:

- **rho** — the per-SNP recombination ("switch") probability,
- **pi** — the copying prior over the donor panel,
- **mu / theta** — the per-site emission ("miscopy") rate,

plus a tiny decoder that maps a packed 2-bit genotype code to its haploid `{0,1}`
allele. Everything here is pure host C++20 with no device dependency, which is the
whole point: these are the parts of the model you can reason about, unit-test, and
pin down on the CPU, so both the CpuBackend reference oracle and the streaming GPU
paint pipeline build on the same trusted inputs rather than each re-deriving them.

The forward-backward *itself* does not live here — it is a `ComputeBackend`
virtual (`ls_forward_backward`, and the `ls_paint_coancestry` / `ls_localanc`
faces on top of it), with the exact scalar reference in `CpuBackend` and the
batched kernel in `src/device/cuda/li_stephens_fb_kernel.cu`. This header carries
only the shared, testable ingredients those recurrences consume.

---

## 2. Where this sits in the engine

`steppe paint` is a Li-Stephens copying model: each recipient haplotype is
explained as a mosaic copied from a panel of `K` donor haplotypes, and the
forward-backward computes the copying posterior `gamma_l(k)` — the probability
that donor `k` is the copied template at SNP `l`. The two published faces reduce
that posterior differently (coancestry chunkcounts/chunklengths in Phase 2, a
per-position ancestry marginal in Phase 3), but both start from the same three
model quantities, and this header is where those quantities come from.

Read it as the *setup* stage. Nothing here launches a kernel or touches a GPU; it
converts map + panel metadata into the numeric vectors the recurrence then scans.

---

## 3. The allele decode — `haploid_allele_from_code`

The engine requires **pre-phased haploid input** — steppe builds no phaser, so a
paint run is only ever fed haplotypes that some upstream tool already phased. On a
correctly phased haploid column each genotype is homozygous: the packed 2-bit code
is `0` (hom-ref) or `2` (hom-alt), never `1` (heterozygous).

`haploid_allele_from_code` honors exactly that: code `0` -> allele `0`, code `2`
-> allele `1`, and everything else -> `kLsMissingAllele` (`0xFF`). "Everything
else" folds two cases into one — the genuine missing code, and the *illegal*
het code `1`. The het code should have been caught upstream (the validator in
`li_stephens_validate.*` rejects a diploid triple before paint ever runs), but if
one slips through, mapping it to missing is the defensive choice: a copied
template is compared against a "missing" allele rather than silently treated as
ref or alt. The emission step reads `kLsMissingAllele` as "no information at this
site" (uniform emission), so a stray het degrades gracefully instead of biasing
the mosaic.

---

## 4. The emission rate — `watterson_emission_rate`

This is the Li & Stephens (2003) per-site miscopy rate over a panel of `K`
donors. Copying is never perfect: even the true template can differ from the
recipient at a site because of mutation since their common ancestor, and `theta`
is the model's budget for that. The classic Watterson estimate is

```
theta_hat = 1 / H_{K-1}          H_{K-1} = 1 + 1/2 + ... + 1/(K-1)
```

and the emission rate the recurrence uses is

```
mu = (1/2) * theta_hat / (theta_hat + K)
```

so at an emitting site a match carries weight `1 - mu` and a mismatch weight `mu`.
`H_{K-1}` is summed in the plainest possible loop (`i` from `1` to `K-1`), FP64
throughout. This is the `--theta auto` default; a user-supplied `--theta`
overrides it downstream. A panel too small to define the harmonic number
(`K < 2`) returns `0.0` — there is no meaningful miscopy rate for a one-donor
panel, and the caller is expected to have a real panel by the time it matters.

---

## 5. The recombination probability — `ls_recomb_prob` and `build_recomb_probs`

### The per-gap switch

`ls_recomb_prob(delta_morgans, Ne, K)` is the probability the copied template
switches somewhere across a single SNP gap. With an expected `4*Ne*delta`
pairwise recombinations over a gap of `delta` Morgans, thinned across `K` donors,

```
rho = 1 - exp(-4 * Ne * delta / K)
```

clamped into `[0, 1]`. The guard at the top returns `0.0` for any degenerate
input — `K < 1`, a non-positive `Ne`, or a non-positive gap (including a NaN,
since the comparison is written `!(x > 0.0)`) — so a zero-length or malformed gap
contributes no switch mass rather than a NaN or a negative probability. Bigger
gaps, larger `Ne`, and smaller panels all push `rho` toward 1 (more likely to have
recombined); a tight, well-linked gap keeps it near 0.

### The full vector

`build_recomb_probs(chrom, genpos_morgans, Ne, K)` lays down the whole per-SNP
`rho[]` in SNP order, and it encodes two conventions the recurrence depends on:

- **`rho[0] = 1.0`.** The first column has no predecessor, so it is fully
  unlinked: `alpha_0 = pi * e_0`, the copying prior times the first emission, with
  no switch term. The vector is initialized to all-ones precisely so column 0 falls
  out for free.
- **A chromosome change resets `rho` to 1.0.** Whenever `chrom[l] != chrom[l-1]`
  the gap is treated as fully unlinked — you cannot recombine *across*
  chromosomes, so the copying restarts from the prior at each new chromosome. This
  is the same inter-chromosome convention DATES uses. `chrom` and
  `genpos_morgans` are the `.snp` columns in SNP order; `chrom` is optional (an
  empty or wrong-length `chrom` is read as a single-chromosome panel, `have_chrom
  = false`), while `genpos_morgans` sets the length `M`.

Within a chromosome the gap is `g_l - g_{l-1}`, floored at `0` before it reaches
`ls_recomb_prob` so a non-monotone map entry can never produce a negative
distance.

---

## 6. The genetic-length weight — `build_genetic_weights`

The coancestry face reports not just *how many* chunks each donor contributed
(chunkcounts) but *how much genetic length* (chunklengths), and to integrate the
copying marginal into an expected copied length you weight each SNP by the slice of
the chromosome it stands for. `build_genetic_weights` builds that per-SNP weight
`w_l` in Morgans with a midpoint / trapezoidal rule:

```
w_l = 1/2 * (gap_left + gap_right)
gap_left  = g_l   - g_{l-1}      (0 at a chromosome start or the first SNP)
gap_right = g_{l+1} - g_l        (0 at a chromosome end   or the last SNP)
```

Each one-sided gap is attributed **only within a chromosome** — the same unlinked
convention as `build_recomb_probs`: at a chromosome boundary the gap that would
cross it is dropped to `0`, so an interior SNP owns half the distance to each
neighbor while a chromosome-edge SNP owns only its interior half. Both gaps are
clamped to `>= 0` against a non-monotone map. `chrom` may be empty (single
chromosome); `genpos_morgans` sets `M`. Pure host, unit-testable, and it never
looks at genotypes at all — it is geometry over the map.

---

## 7. The copying prior — `build_uniform_pi`

`build_uniform_pi(K, self)` builds `pi`, the prior probability of copying each of
the `K` donors before any data is seen. The default is flat — `1/K` on every
donor. The one twist is **leave-one-out self-painting**: when a recipient
haplotype is itself a member of the panel (painting a panel against itself), you
must not let it copy *itself*, or the mosaic trivially reads "100% me." Passing a
valid `self` index (`0 <= self < K`, with `K > 1`) zeroes that donor's prior and
shares the mass `1/(K-1)` across the remaining `K-1` donors. `self < 0` means "no
leave-one-out," the plain uniform case.

Either way the returned vector sums to 1 (`K = 0` returns empty; the `K = 1`
leave-one-out edge falls back to plain uniform rather than dividing by zero). This
matters because the whole forward-backward is a probability recurrence — a prior
that didn't normalize would quietly bias every posterior downstream.

---

## 8. Contracts and invariants

- **Pre-phased haploid input, always.** These helpers assume phased haploid
  columns; the diploid het code `1` is treated as an error folded to missing
  (section 3), never as a real observation. The engine has no phaser and does no
  phasing.
- **Native FP64, not the emulated default.** Every value here is a plain `double`
  computed in native FP64. This is deliberate and worth stating clearly: steppe's
  default emulated-FP64 (Ozaki) path is a *matmul-only* accelerator, and there is
  **no GEMM** anywhere in the Li-Stephens engine — the scan and its reductions are
  cancellation-sensitive sequential recurrences, so they run in native FP64 top to
  bottom. Nothing in the paint path is emulated.
- **`pi` sums to 1.** `build_uniform_pi` is the sole producer of the copying
  prior and always returns a normalized distribution (or empty for `K = 0`).
- **`rho[0] = 1.0` and unlinked chromosome boundaries.** The first column and
  every chromosome change are fully unlinked; the recurrence restarts from the
  prior there rather than carrying a switch term across a gap that has no genetic
  meaning.
- **Degenerate-input guards return neutral values, never NaN.** `K < 2` ->
  `theta = 0`; a non-positive/NaN gap, `Ne`, or `K` -> `rho = 0`; a non-monotone
  map gap is floored to `0` before use. The helpers absorb bad metadata into
  harmless zeros rather than propagating a NaN into the scan.

---

## 9. Edge cases worth calling out

- **Empty or mismatched `chrom`.** Treated as a single-chromosome panel
  (`have_chrom = false`); the length is driven entirely by `genpos_morgans`. This
  is the common case for a single-chromosome test panel and needs no special
  caller handling.
- **A single-SNP panel (`M = 1`).** `build_recomb_probs` returns just `{1.0}`
  (column 0 is always unlinked), and `build_genetic_weights` returns `{0.0}` (no
  neighbor on either side). Both are correct and self-consistent.
- **A one-donor panel (`K = 1`).** `theta` is `0` and any `self` index collapses
  to plain uniform (`pi = {1.0}`); leave-one-out is meaningless with a single
  donor and is not attempted.
- **A stray heterozygous or missing code in the recipient or donors.** Decoded to
  `kLsMissingAllele`; the emission step reads it as no-information at that site
  rather than a hard ref/alt call, so a single dirty site softens the mosaic
  locally instead of corrupting it.
- **A non-monotone genetic map.** Any backward or zero gap is floored to `0`
  before it reaches `ls_recomb_prob` or the trapezoidal weight, so a scrambled
  `.snp` map can degrade the answer but cannot produce a negative probability or a
  NaN.
