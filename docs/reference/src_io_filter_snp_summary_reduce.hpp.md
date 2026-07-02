# `snp_summary_reduce.hpp` reference

## 1. Purpose

`src/io/filter/snp_summary_reduce.hpp` is the single, shared piece of code that
turns the per-population allele counts for one SNP into the four pooled numbers the
keep/drop filters need, and then makes the keep-or-drop decision from them.

It exists so that two very different callers can compute exactly the same answer:

1. The **host** keep-mask builder (in `snp_filter.cpp`), which walks SNPs on the CPU
   and decides which ones to keep.
2. The **GPU** keep-mask kernel (in `decode_compact_kernel.cu`), which does the same
   thing on the device, one SNP per thread.

Both paths call the same three functions defined here, so they cannot drift apart on
a boundary case or an allele-pair classification. Every function is marked so it
compiles for both the CPU and the GPU (that is what the `STEPPE_HD` macro does), and
the two builds are required to produce **bit-for-bit identical** results — not merely
close ones. This is the first place in the filter path that does non-integer
(floating-point) math shared between the two builds, so the rules that guarantee they
match live here, in one place.

The reduction happens per SNP, across all the kept populations. For one SNP `s` it
computes:

- **pooled reference-allele count** — the sum over populations of `Q · N`, where `Q`
  is a population's reference-allele frequency at that SNP and `N` is its allele
  count.
- **pooled allele count** — the sum over populations of `N`.
- **non-missing individuals** — the sum over populations of `N / ploidy`.

From those it derives the pooled reference frequency, the folded minor-allele
frequency, and the missing-data fraction, and packages them into a small struct.

---

## 2. The bit-exact requirement

The whole point of putting this code in one header is that the CPU mask and the GPU
mask must agree exactly, down to the last bit, for any dataset. Getting floating-point
sums to match across two different compilers and two different processors takes
deliberate care, and this section explains the two things that make it work.

### The sum runs in a fixed sequential order

The pooled counts are summed one population at a time, from population `0` up to
population `P-1`, in that order. On the GPU each SNP is handled by a single thread, so
a thread's loop runs in exactly the same order as the CPU's loop — there is no
parallel reduction across populations that could reorder the additions and change the
rounding. Same numbers, same order, same result.

### The one fragile operation: `Q · N` folded into the running sum

The only step where the two builds could disagree is the multiply-then-add that folds
each population's `Q · N` into the running total. The danger is a hardware feature
called a fused multiply-add (FMA, or on NVIDIA GPUs, FFMA): the expression
`accumulator + q*n` can either be computed as two separately rounded steps (multiply,
then add) or fused into a single step with only one rounding. Those two ways can
produce different bits.

The two builds default to opposite behavior:

- The GPU compiler defaults to **fusing** (`--fmad=true`), giving one rounding.
- The host compiler, in strict standard-C++ mode with fused contraction turned off
  (`-ffp-contract=off`), gives **two** roundings.

Left alone, the two paths would round differently and the masks would not match. The
fix is to force both paths to do the same two separately-rounded steps. That is what
the `pooled_ref_fma` helper in the next section is for. On the GPU it uses explicit
round-to-nearest multiply and add instructions that never fuse regardless of the
compiler flag; on the host it uses the plain two-step expression, which the standard
guarantees will not be fused across statements. This is what keeps the two builds
bit-identical, and it must not be changed casually — it is the pin that makes the GPU
mask match the reference exactly.

### Why the monomorphic (no-variation) check is immune to all of this

One filter drops SNPs that have no variation at all — where the pooled reference
frequency is exactly `0.0` (or exactly `1.0`). This exact-zero test is safe no matter
how the multiply-add rounds, and here is why: at a truly no-variation site every
population's `Q` is exactly `0.0` or exactly `1.0`, so each `Q · N` is a whole number
represented exactly in floating point, and summing whole numbers introduces no
rounding error on either path. The result is the same bit whether or not the multiply
fused. So the `== 0.0` comparison is genuinely reliable — but only as long as the
upstream pooling stays as the `Q · N` sum. Switching the upstream code to, say, an
average of per-population frequencies would introduce rounding and break this
exactness.

---

## 3. `PooledSnpSummary`

`PooledSnpSummary` is the small plain-data struct that carries the four pooled numbers
for one SNP. It is a plain-old-data type on purpose (no `std::vector`, no heap, nothing
that can't live in a GPU register or in device memory), so the same struct works
inside a GPU kernel as on the host. It mirrors a richer host-only summary type but is
stripped down to just what a device kernel can hold.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pooled_ref_af` | `double` | `0.0` | The pooled reference-allele frequency: the pooled reference count divided by the pooled allele count. `0` when the SNP has no data at all. This is the **unfolded** frequency (it can be anywhere in `[0, 1]`). |
| `pooled_minor_af` | `double` | `0.0` | The folded minor-allele frequency: `min(pooled_ref_af, 1 - pooled_ref_af)`. This is the value the minor-allele-frequency filter compares against. |
| `missing_frac` | `double` | `1.0` | The missing-data fraction over the individuals axis: `1 - (non-missing individuals / total individuals)`. Defaults to `1.0` (all missing) so a SNP with no data reads as fully missing. |
| `pooled_allele_count` | `double` | `0.0` | The pooled allele count, the sum of `N` across populations. A value of `0` means the SNP is missing in every kept population. |

---

## 4. `pooled_ref_fma` — the multiply-add that both builds agree on

`pooled_ref_fma(acc, q, n)` computes `acc + q*n` for one population's cell, but does it
in a way that produces the same bits on the CPU and the GPU. It is the concrete
mechanism behind the bit-exact requirement described above.

- On the **GPU** it calls the explicit round-to-nearest multiply and add
  intrinsics (`__dadd_rn(acc, __dmul_rn(q, n))`). These never fuse into a single
  FFMA, no matter what the compiler's fuse flag is set to, so the multiply and the add
  each round on their own.
- On the **host** it is the plain two-line expression: multiply `q * n` into a
  temporary, then add. In strict standard-C++ mode with contraction off — which is how
  this project is built — the standard forbids fusing a multiply and an add written as
  separate statements, so each rounds on its own too.

The result is that both builds do a multiply-round followed by an add-round, and land
on identical bits. Every population's contribution to the pooled reference count goes
through this helper.

---

## 5. `derive_pooled_summary_one` — the per-SNP reduction

`derive_pooled_summary_one(q, n, P, s, ploidy_d, total_indiv_d)` is the shared
reduction: it takes the per-population frequency and count arrays and produces the
filled-in `PooledSnpSummary` for a single SNP. Both the host mask builder and the GPU
kernel call this exact function.

### How the inputs are laid out

`q` and `n` are the reference-frequency and allele-count arrays, stored column-major
for a `P × M` grid (P populations, M SNPs). The cell for population `p` and SNP `s`
lives at index `p + P·s`. The function first computes the base offset `P·s`, then walks
`p = 0 .. P-1` to reach each population's cell for this SNP.

The other inputs:

- `P` — the number of kept populations.
- `s` — which SNP to reduce (a column index).
- `ploidy_d` — the ploidy as a `double` (used to convert allele counts into
  individual counts).
- `total_indiv_d` — the total number of individuals across all kept populations. This
  is the same for every SNP, so it is passed in rather than recomputed. It is the
  denominator of the missing-data fraction.

### What it computes

Walking the populations in order `0 .. P-1`, it accumulates three running sums:

- the pooled reference count, via `pooled_ref_fma` (the bit-exact multiply-add),
- the pooled allele count, a plain running sum of `N`,
- the non-missing individual count, a running sum of `N / ploidy`.

A missing cell needs no special handling: where a cell is missing, `Q` is zero and `N`
is zero, so that population contributes `0` to both the reference count and the allele
count — no branch, exactly as the host loop does it.

After the loop it derives the three output frequencies:

- `pooled_ref_af` = pooled reference count / pooled allele count, guarded so that a SNP
  with zero allele count yields `0` instead of dividing by zero.
- `pooled_minor_af` = the folded value `min(pooled_ref_af, 1 - pooled_ref_af)`. The
  folding is written out inline here rather than calling the shared folding helper,
  deliberately, so the device path does not have to pull in an extra header — the body
  is identical to that helper.
- `missing_frac` = `1 - non-missing / total`, guarded so that zero total individuals
  yields `1.0` (fully missing) instead of dividing by zero.

The arithmetic here matches the host mask builder step for step: same accumulation,
same guards, same order. That correspondence is what lets the GPU result equal the
reference result exactly.

---

## 6. `keep_decision_pooled` — the keep-or-drop decision

`keep_decision_pooled(sm, ref, alt, chrom, cfg, membership_ok)` is the single function
that decides whether one SNP is kept. It takes the pooled summary from the previous
section plus the SNP's reference and alternate allele characters, its chromosome, the
filter configuration, and a precomputed membership bit (whether the SNP survives the
include/exclude ID lists). It returns `true` to keep the SNP. Both the host mask
builder and the GPU kernel call this, so the two cannot diverge.

The checks run in a fixed order, and each is written to match the host path bit-for-bit:

1. **Multiallelic drop (always on).** If the ref/alt pair is not a clean pair of two
   distinct A/C/G/T bases, the SNP is dropped. This is unconditional — it holds even
   under the strand modes that otherwise keep more SNPs, because those modes still
   require a clean biallelic SNP.
2. **Strand-ambiguous drop (only under the Drop strand mode).** If the configuration's
   strand mode is `Drop` and the ref/alt pair is a self-complementary palindrome (an
   A/T or C/G pair), the SNP is dropped. Under the Keep and Flip modes this step is
   skipped, so palindromes are retained.
3. **Minor-allele-frequency filter.** Keep only if the folded pooled minor-allele
   frequency is at least `maf_min`.
4. **Missing-data filter.** Keep only if the missing fraction is at most
   `geno_max_missing`.
5. **Flag-gated content filters.** If `drop_monomorphic` is set, drop no-variation SNPs
   (using the exact-zero test on the **unfolded** pooled reference frequency). If
   `transversions_only` is set, drop transitions. If `autosomes_only` is set, drop SNPs
   that are not on chromosomes 1–22.
6. **Membership.** Finally, the SNP is kept only if the precomputed membership bit is
   also true.

### Why the strand gate is still bit-exact

Steps 1 and 2 together reproduce the original single boolean expression exactly. The
original code dropped a SNP if it was multiallelic **or** strand-ambiguous, and because
the `or` checked multiallelic first, the two separate statements here produce the same
outcome under the `Drop` mode as that combined test did — for any dataset. Splitting
the strand check into its own flag-gated statement does not change the default-drop
decision; it only lets the Keep and Flip modes skip the palindrome half while leaving
the multiallelic drop untouched.

---

## 7. Layering and dependencies

This header sits at the leaf of the input/output layer: it deliberately depends on
almost nothing.

- It includes the small header-only macro that makes a function compile for both the
  host and the device (`STEPPE_HD`). That macro brings in no CUDA and adds no link
  dependency — it is the same single-source macro other shared host/device primitives
  in the codebase use, reused here rather than being written out a third time.
- It includes the shared filter predicates (the pure allele-pair and threshold
  classifiers) and the public configuration struct.
- It pulls in nothing from the core compute library or the device library.

Keeping the dependencies this small is what lets the same file be compiled into the GPU
kernel and into the host mask builder without either one dragging in the other's
machinery. That shared compilation is the mechanism that guarantees the CPU mask and
the GPU mask are the same code, and therefore the same result.
