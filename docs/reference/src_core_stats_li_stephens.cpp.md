# `li_stephens.cpp` reference

## 1. Purpose

`src/core/stats/li_stephens.cpp` is the small, CUDA-free host layer that turns a
genotype triple's map columns and panel size into the numbers the Li-Stephens
forward-backward actually runs on. It is the "model" half of `steppe paint`: it
computes the per-SNP recombination (switch) probabilities `rho`, the copying prior
`pi` over the donor panel, the Watterson miscopy (mutation) rate `theta`, the
per-SNP genetic-length weights the paint face integrates against, and the one-line
map from a packed 2-bit genotype code to a haploid `{0,1}` allele.

What it is *not* is the engine. The forward-backward recurrence itself — the alpha
scan, the checkpoint/recompute, the gamma posterior, and the coancestry and
local-ancestry reductions — is a `ComputeBackend` virtual (`ls_forward_backward`),
implemented by the CpuBackend reference oracle and the batched CUDA kernel. Those
live in `src/device/`. This file is pure host C++20 with no kernel, no device
buffer, and no CUDA header, which is exactly why every function here is trivially
unit-testable on the host and gets exercised as its own small gate before the
GPU path is ever touched.

Think of it as the shared front porch: both the reference oracle and the streaming
GPU paint pipeline call these same helpers to build their inputs, so the two paths
are numerically fed from one place and can't drift on how `rho` or `pi` is defined.

---

## 2. The Li-Stephens model, in one paragraph

Li-Stephens paints a *recipient* haplotype as a mosaic copied from a *panel* of `K`
donor haplotypes. As you walk the SNPs left to right, at each gap the copying can
either stay on the current donor or "switch" to another (a recombination), and at
each site the copied donor's allele can differ from the recipient's (a miscopy, the
"mutation"). Three model quantities drive that hidden Markov chain: the switch
probability `rho` at each gap, the miscopy rate `theta` (emission), and the prior
`pi` over which donor you land on after a switch. This file computes all three, plus
the genetic-length weights used later to convert the copying posterior into an
expected *copied length* per donor. The engine consumes them; it doesn't recompute
them.

Two hard input contracts run through everything here, both enforced upstream by the
validator (`li_stephens_validate.cpp`) and assumed by this file:

- **Pre-phased haploid input.** steppe ships no phaser. Every column is one
  haplotype, homozygous by construction — a diploid heterozygous triple is a bug the
  validator rejects, and `haploid_allele_from_code` treats the het code defensively
  (section 7).
- **Native FP64 throughout.** These are scalar model numbers — harmonic sums,
  `exp`, midpoint gaps — with no matrix multiply anywhere in the paint engine. The
  project's emulated-FP64 (Ozaki) default is a *matmul-only* accuracy trick; there
  is no GEMM in Li-Stephens, so both this file and the FB scan/reductions it feeds
  use plain native `double`. Don't reach for emulated FP64 here — there's nothing
  for it to accelerate.

---

## 3. `watterson_emission_rate` — the `--theta auto` miscopy rate

The emission rate is how much probability the model gives to a copied allele *not*
matching, i.e. a mutation on the copied haplotype. The default follows Li & Stephens
(2003): with `K` donors,

```
theta_hat = 1 / H_{K-1}          (H_{K-1} = 1 + 1/2 + ... + 1/(K-1), the (K-1)th harmonic number)
mu        = (1/2) * theta_hat / (theta_hat + K)
```

The harmonic sum is accumulated in a straight `double` loop, `theta_hat` is its
reciprocal, and `mu` is the half-scaled ratio above. The `1/2` is the diploid-style
symmetric-mutation split standard to the ChromoPainter/Li-Stephens family: half the
mutation mass to each direction of allele mismatch.

Edge case: `K < 2` returns `0.0`. With fewer than two donors there is no panel to
copy from and the harmonic sum `H_{K-1}` would be empty (or a divide-by-zero), so
the rate is defined to be zero rather than left undefined.

This is only the *default*. `steppe paint --theta auto` calls this; an explicit
`--theta <value>` overrides it entirely and this function is never consulted.

---

## 4. `ls_recomb_prob` — the switch probability across one gap

For a single SNP gap of `delta_morgans` genetic distance, with effective population
size `Ne` and `K` donors, the probability of at least one copying switch is the
standard exponential:

```
expected = 4 * Ne * delta_morgans / K       (expected pairwise recombinations across the gap)
rho      = 1 - exp(-expected)               (clamped to [0, 1])
```

The `4*Ne*Δg` is the coalescent expectation of recombination events over the gap,
divided by `K` because the switch is shared across the panel. `1 - exp(-·)` turns an
event rate into a per-gap probability, and the result is clamped to `[0,1]` so
floating-point noise near the ends can never hand the HMM a probability outside its
valid range.

Guard: the function returns `0.0` (no switch) if `K < 1`, or if `Ne` is not strictly
positive, or if `delta_morgans` is not strictly positive. The `!(x > 0.0)` form is
deliberate — it also catches `NaN`, so a malformed map value degrades to "no switch
here" instead of poisoning the scan. A zero or negative gap (two SNPs at the same
genetic position, or a rounding artifact) means no genetic distance and therefore no
recombination opportunity, hence `rho = 0`.

---

## 5. `build_recomb_probs` — the whole `rho` vector, boundaries included

`ls_recomb_prob` is one gap; `build_recomb_probs` walks the ordered SNP list and
produces the length-`M` `rho` vector the scan indexes by column. Two conventions
matter:

- **`rho[0] = 1.0`.** The first column has no predecessor, so it is treated as
  fully unlinked: the forward pass starts from `alpha_0 = pi * e_0` (prior times
  emission) with a guaranteed switch. The vector is initialized to `1.0` everywhere,
  so column 0 keeps that value with no special-casing.
- **Chromosome boundaries reset to `rho = 1.0`.** When `chrom[l] != chrom[l-1]` the
  two SNPs are on different chromosomes and are unlinked; `rho` is forced back to
  `1.0` (a certain switch), matching the inter-chromosome convention used elsewhere
  in steppe. Only *within* a chromosome does the real `ls_recomb_prob` of the cM gap
  apply.

The `chrom` vector is optional in the loose sense: if `chrom.size() != M` (e.g. a
single-chromosome panel passed an empty `chrom`), `have_chrom` is false and every
gap is treated as same-chromosome, so a plain one-chromosome map works with no
`chrom` column at all. The genetic gap is clamped non-negative
(`delta > 0.0 ? delta : 0.0`) before it reaches `ls_recomb_prob`, so a
non-monotone map value can never produce a negative distance.

Contract: `genpos_morgans` must be length `M` (it defines `M`), `chrom` is either
empty or the same length, `Ne > 0`, `K >= 1`.

---

## 6. `build_genetic_weights` — the per-SNP length weight for chunklengths

The paint face reports not just *how many* chunks each donor contributed
(chunkcounts) but *how much genetic length* it copied (chunklengths, in Morgans). To
get an expected copied length you integrate the copying posterior against a per-SNP
genetic weight, and this function builds that weight vector `w`.

It uses a midpoint / trapezoidal rule: each SNP owns half of the gap on its left and
half of the gap on its right,

```
w[l] = 1/2 * (gap_left + gap_right)
gap_left  = g_l   - g_{l-1}    (0 at the left edge or across a chromosome change)
gap_right = g_{l+1} - g_l      (0 at the right edge or across a chromosome change)
```

The same unlinked convention as `rho` applies: a chromosome change on either side
zeroes that one-sided gap, so a SNP never claims length across a chromosome
boundary. Both gaps are clamped `>= 0`. End SNPs and single-SNP-per-chromosome cases
fall out naturally — the missing side is just `0`, so the first and last SNP of each
chromosome get a one-sided half-weight instead of a bogus edge value.

Same length contract as section 5: `genpos_morgans` is length `M`; `chrom` may be
empty for a single-chromosome panel.

---

## 7. `build_uniform_pi` — the copying prior, with leave-one-out

`pi` is the prior over which of the `K` donors you copy after a switch. The default
is uniform, `1/K` on every donor. The one wrinkle is *panel-vs-self* painting: when
a recipient is itself a member of the donor panel, letting it copy from its own
haplotype is degenerate (it would just copy itself perfectly), so that donor is
excluded with a leave-one-out prior.

- `self < 0` (or `self >= K`, or `K == 1`): plain uniform, `pi[k] = 1/K`.
- `0 <= self < K` with `K > 1`: the self donor gets `pi[self] = 0` and the other
  `K-1` donors share it evenly, `1/(K-1)` each.

Either way the vector sums to 1 (the leave-one-out mass is redistributed, not
dropped). `K <= 0` returns an empty vector — there is no panel to prior over.

---

## 8. `haploid_allele_from_code` — packed code to haploid allele

The forward-backward's emission compares the recipient's allele at a site to each
donor's allele, over the alphabet `{0, 1, missing}`. This one-liner maps steppe's
packed 2-bit genotype code to that alphabet:

| Code | Meaning | Allele |
|---|---|---|
| 0 | homozygous reference | `0` |
| 2 | homozygous alternate | `1` |
| 3 | missing | `kLsMissingAllele` (`0xFF`) |
| 1 | heterozygous | `kLsMissingAllele` (defensive) |

The load-bearing detail is code `1`. In a *phased haploid* column there is no
heterozygote — every genotype is homozygous by construction, so a `1` should never
arrive here. The validator rejects any diploid triple before the engine runs, so
this mapping is a defensive backstop: if a het code somehow reaches emission it
becomes `missing` (contributes no evidence) rather than being silently misread as an
allele. `kLsMissingAllele` is `0xFF`, a sentinel distinct from both real alleles;
the emission step treats a missing recipient or donor allele as uninformative.

---

## 9. Invariants and edge cases at a glance

- **Phased haploid only.** One haplotype per column, homozygous; the het code is a
  defended-against impossibility (section 8), not a supported input.
- **Native FP64, no emulation.** Scalar model math with no matmul; the Ozaki
  emulated-FP64 default is matmul-only and does not apply to the paint engine
  (section 2). The FB scan and its reductions that consume these numbers are native
  FP64 for the same reason.
- **Probabilities stay in range.** `rho` is clamped `[0,1]`; `pi` sums to 1 (with or
  without leave-one-out); the emission rate is a valid `[0, 1/2)`-scale mutation
  probability.
- **Boundaries are unlinked.** The first column and every chromosome change force
  `rho = 1.0` and zero the one-sided genetic weight — the same convention shared by
  `build_recomb_probs` and `build_genetic_weights`, so the switch probability and the
  length weight agree on where a chromosome ends.
- **Degenerate panels are defined, not crashing.** `K < 2` -> emission rate `0`;
  `K < 1` or bad `Ne`/gap -> switch prob `0`; `K <= 0` -> empty `pi`. Every small-panel
  or malformed-map case has a defined, benign fallback instead of a divide-by-zero.
- **`chrom` is optional.** An empty or wrong-length `chrom` is read as "one
  chromosome", so a single-chromosome panel needs no chromosome column.
- **`NaN`-safe guards.** The `!(x > 0.0)` comparisons in `ls_recomb_prob` reject
  `NaN` map values as "no switch", so a corrupt genetic position can't propagate a
  `NaN` into the forward scan.

---

## 10. Where these feed the engine

`src/app/cmd_paint.cpp` wires it together: it decodes a tile of the panel into
haploid alleles with `haploid_allele_from_code`, picks `theta` via
`watterson_emission_rate` (unless `--theta` is given), builds `rho` with
`build_recomb_probs` and the length weights with `build_genetic_weights`, and builds
a per-recipient `pi` with `build_uniform_pi` (passing the recipient's own panel
index as `self` for leave-one-out). Those four vectors plus the allele buffer are the
complete input to `ls_forward_backward`, which does the actual alpha scan, the
checkpoint/recompute so the `O(K*M)` alpha table is never fully resident, the gamma
copying posterior, and then either the coancestry reduction (chunkcounts +
chunklengths, the latter using the section-6 weights) or the per-SNP local-ancestry
posterior. This file is the deterministic, testable front end to all of that.
