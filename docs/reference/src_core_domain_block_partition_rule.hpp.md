# `block_partition_rule.hpp` reference

## 1. Purpose

`src/core/domain/block_partition_rule.hpp` is the single source of truth for one
question: **which jackknife block does a given SNP belong to?** steppe estimates the
uncertainty of its statistics with a block jackknife, which needs the genome carved
into contiguous blocks of SNPs. Every part of the system that needs a block
assignment — the input/format-reading front end and the GPU (and CPU) compute
kernels — gets it from the functions here, so they all agree bit-for-bit.

Two properties make this file the *one* home for the rule:

1. It is **host-pure and free of any CUDA code**. It uses only the C++ standard
   library, so both the host-side readers and the device-side layer can include it
   without either dragging in the other's dependencies.
2. Because there is exactly one copy of the rule, the block id a SNP gets is
   guaranteed identical across the single-dataset and merged-dataset paths and
   across the CPU and GPU backends. Re-deriving the block rule anywhere else (for
   example, recomputing it inside the reader) is an explicitly-forbidden mistake:
   the numbers would be free to drift apart.

The file provides a small per-SNP primitive (`block_of`), the one unit-conversion
site (`block_size_cm_to_morgans`), the whole-genome assignment pass
(`assign_blocks` returning a `BlockPartition`), and the inverse that turns that
assignment back into per-block SNP ranges (`block_ranges` returning `BlockRange`s).

---

## 2. Units: centimorgans and Morgans

There are two units of genetic distance in play, and keeping them straight is a
frozen rule of this file.

- The **configuration surface speaks centimorgans**. A user sets the block size as
  `RunConfig::block_size_cm`, default `5.0` (five centimorgans).
- The **internal block math is done in Morgans**, to match the reference
  convention[^at2]. The `blgsize` block-size parameter defaults to `0.05` Morgans,
  which is the same distance as five centimorgans.

`block_size_cm_to_morgans(double cm)` is the **only** place the conversion is
applied. It divides by the named constant `kCentimorgansPerMorgan` (100.0) — there
is deliberately no bare `* 0.01` or `/ 100` anywhere else in the code. Calling it
on the default gives exactly `block_size_cm_to_morgans(5.0) == 0.05`.

Everything downstream (`block_of`, `assign_blocks`) speaks Morgans only. Callers are
expected to run the config value through this converter first and never to mix the
two units.

---

## 3. `block_of` — the per-SNP block primitive

`block_of(double genpos_morgans, double block_size_morgans)` is the simplest form
of the rule: given one SNP's genetic position (in Morgans) and the block width (in
Morgans), it returns `floor(genpos_morgans / block_size_morgans)` as an `int` — the
zero-based index of the fixed grid cell that position falls into.

This is the deterministic, pure function that fixes block membership: the same
arithmetic runs on the host and on the device, so no SNP's block is ever computed
two different ways. It is still a valid, shareable primitive, but note that the
whole-genome pass (`assign_blocks`) **no longer calls it** — the parity rule
does not bin SNPs onto a fixed grid[^at2] (see section 4 for why). `block_of` remains for
callers that genuinely want the plain floor-to-grid mapping.

**Precondition:** `block_size_morgans` must be greater than zero, because it is the
divisor. `block_of` itself does *not* check this — the guard lives in
`assign_blocks`, which rejects a zero, negative, or NaN width before ever computing
a block. Calling `block_of` directly with a non-positive or NaN width is undefined
behavior, because the division produces an infinity or NaN and casting that to `int`
is undefined.

Parameters:

- `genpos_morgans` — the SNP's genetic position in Morgans, expected to be
  non-negative within a chromosome. Callers supply per-chromosome positions and
  handle the chromosome boundaries themselves.
- `block_size_morgans` — the block width in Morgans (`blgsize`),
  which must be positive.

Returns the zero-based block index (non-negative for a non-negative position).

---

## 4. `assign_blocks` — assigning every SNP to a block

`assign_blocks` is the real workhorse: in a single deterministic pass over all the
SNPs **in file order**, it assigns each one a global block id. It is the host-pure
rule that both the reader front end and the compute kernels consume, and it lives
here and nowhere else.

### The rule: a SNP-anchored cumulative walk

The rule reproduces the reference block-building convention[^at2] (its `setblocks()`
routine). It is **not** a floor-onto-a-fixed-grid rule. Instead it is a *cumulative
walk anchored at real SNP positions*:

- Carry the genetic position of the **first SNP of the current block** — call it the
  anchor.
- A **new block opens** when **either** the chromosome changes **or** the distance
  from the anchor to the current SNP reaches the block width. The comparison is
  inclusive (`>=`), matching the original C code.
- When a new block opens, the anchor **re-sets** to the SNP that opened it, so any
  leftover sub-block-width distance rolls forward into the next block rather than
  being discarded.

Because blocks are anchored at actual SNP positions and re-anchored on every cut —
never at fixed multiples of the block width — the result differs from a plain
floor-to-grid binning. That re-anchoring is the whole difference.

### Invariants this produces

Each of these is a property that is relied upon downstream and pinned by tests:

- **Blocks never straddle chromosomes.** A chromosome boundary always forces a fresh
  block, because "chromosome changed" is one of the two cut conditions.
- **A block spans at least the block width** — and may be *wider*, because the SNP
  that trips the threshold can overshoot it. The one exception is the trailing
  remnant at the end of a chromosome (or a very short chromosome), which is kept as
  it is. This matches the reference's small chromosome-end blocks.
- **The block count is the walk's count, not an occupied-grid count.** A wide stretch
  of genome with few SNPs is a *single* block, not one block per empty grid cell that
  a fixed grid would have produced.

### Inputs, sizes, and output

The inputs are parallel arrays, all of the same length (one entry per SNP, in file
order). A whole-genome dataset can have on the order of 584,000 SNPs, so the loop
index is a `long`; the per-SNP block ids stay in an `int` vector because the block
*counts* are small (on the order of a few thousand even for a whole genome). No
memory is allocated beyond the result.

Parameters:

- `chrom` — per-SNP chromosome code. Any integer scheme works; only whether two
  adjacent SNPs share a value matters.
- `genpos_morgans` — per-SNP genetic position in Morgans, same length as `chrom`.
- `block_size_morgans` — the block width in Morgans (produce it with
  `block_size_cm_to_morgans`). Must be positive; see the fail-fast rule below.
- `physpos` — per-SNP physical position in base pairs. Used **only** for the
  fallback in section 5; ignored entirely when there is a real genetic map. Empty
  (the default) disables the fallback.
- `bp_window` — the base-pair fallback window (default `kBpFallbackWindow`, two
  million). Consulted only in the fallback.

Returns a `BlockPartition` (section 6).

### Fail-fast on a bad width

If `block_size_morgans` is illegal — zero, negative, or NaN — `assign_blocks` does
**not** proceed. It returns an empty partition (`n_block == 0`) rather than dividing
by it and producing the float-to-int undefined behavior or silently inverted bins
that would otherwise result. This is where the divisor is validated once for the
whole system.

---

## 5. The base-pair fallback for datasets with no genetic map

Some datasets ship without a genetic linkage map: the genetic-position column of the
`.snp`/`.bim` file is **all zeros**. This is common for modern data derived from VCF
or PLINK. If the walk of section 4 ran on all-zero positions, the distance from the
anchor would always be zero, the only cuts would be chromosome boundaries, and the
whole genome would collapse to **one block per chromosome**. That breaks the block
jackknife: a single-chromosome subset would have just one block, which yields an
undefined standard error and a non-invertible covariance.

The reference handles exactly this case[^at2]: it detects the missing map, prints a notice,
and partitions blocks by a **hardcoded two-megabase window of physical position**
instead. That two-million-base-pair window is fixed and independent of the genetic
block-size setting.

`assign_blocks` reproduces this precisely. When `genpos_morgans` is all zeros **and**
a usable physical-position axis (`physpos`, at least as long as the SNP count and not
all zero) is supplied, it runs the identical SNP-anchored walk over `physpos` with
the window `bp_window` (default two million), and it warns on standard error the way
the reference does. Walking raw base-pair values with a two-million window keeps the
arithmetic in exact integers (base-pair counts are well within the range of exactly
representable doubles), so the partition is robust, and it reproduces the reference
block count on the same data.

The fallback fires **only** on an all-zero map. A dataset that has a real genetic map
takes the section-4 genetic-map walk unchanged and is bit-identical to what it would
have produced before the fallback existed.

---

## 6. `BlockPartition` — the result of `assign_blocks`

`BlockPartition` is the return type of `assign_blocks`. It has two fields:

- `block_id` — a `std::vector<int>` parallel to the input SNP arrays (file order).
  `block_id[s]` is the global jackknife block of SNP `s`. It stays `int` because
  block counts are small even on a whole genome.
- `n_block` — the number of distinct blocks, equal to `max(block_id) + 1`, or `0`
  when there are no SNPs.

The block ids satisfy a **dense, non-decreasing** contract: every value in the range
`0 .. n_block-1` is used at least once, and the sequence never decreases as you move
through the SNPs in file order. That is exactly what makes each block's SNPs form one
contiguous run, which section 7 depends on.

---

## 7. `BlockRange` and `block_ranges` — the inverse

The compute backends don't want a per-SNP id; they want, for each block, the span of
SNP columns it covers. `block_ranges` is the single-source inverse of `assign_blocks`
that provides exactly that, and it validates the partition contract once on the way.

### `BlockRange`

`BlockRange` describes one block as a half-open column range `[begin, end)` in the
per-SNP arrays (file order):

- `begin` — the block's first SNP column, inclusive. On the GPU path this is the
  block's offset into the SNP arrays.
- `end` — one past the block's last SNP column, exclusive.
- `size()` — the block's SNP count, `end - begin`. This is the base of the
  jackknife weighting denominator, and also the block's stored size in the f2 tensor.

Both bounds are `long` to match the per-SNP column index used elsewhere.

### `block_ranges`

`block_ranges(block_id, M, n_block)` turns the dense, non-decreasing `block_id`
array into a vector of `n_block` `BlockRange`s, one per block id. Both backends — the
CUDA path (which copies the offsets and sizes to the device for the kernel to read)
and the CPU reference — derive their per-block layout from this one function, instead
of each re-scanning the id array on its own. Doing the scan once, here, is also what
lets the validation below live in one place.

**Contract (checked fail-fast).** `block_ranges` enforces the postcondition that
every consumer relies on:

- `block_id` must have at least `M` entries.
- Every `block_id[s]` for `s` in `[0, M)` must satisfy `0 <= id < n_block`.
- The sequence must be non-decreasing, so each block's SNPs form one contiguous run.

A violation is a programming or data error, so it throws `std::runtime_error` with a
descriptive message rather than reading or writing out of bounds. This matters
because a malformed `block_id` — one too short, or holding an id below zero or at or
above `n_block` — would otherwise have been a silent out-of-bounds write on the host
range vector and a silent out-of-bounds device read of the block-offset array.
`assign_blocks` always satisfies the contract, but a hand-built or recomputed
partition might not, so the check is here for all of them, in every build
configuration (not only debug).

The result is dense in `[0, n_block)`. A block with no SNPs would get an empty range
(`begin == end`); `assign_blocks` never produces one, but the validation does not
forbid it — it only rejects the genuinely unsafe ids above.

Parameters and result:

- `block_id` — the per-SNP ids in file order (length at least `M`), i.e. the
  `BlockPartition::block_id` from `assign_blocks`.
- `M` — the number of SNP columns to scan. It travels separately from the array's
  length, which is why the length check above also pins `block_id.size() >= M`.
- `n_block` — the number of distinct blocks; the length of the returned vector.
- Returns `n_block` ranges indexed by block id (`out[b]` is block `b`'s range).
  Empty input (`M <= 0` or `n_block <= 0`) returns an empty vector.

**Why this one is defined inline in the header.** Unlike `assign_blocks` (which is
compiled out of line into the core library), `block_ranges` is compiled directly into
the backends, which live in the device library. The device library cannot link the
core library — that would create a dependency cycle — so both layers can only reach
this host-pure code through the header itself. Since it is called once per f2-block
computation and is a single linear scan over the SNPs, the inlining cost is
negligible next to the matrix multiplies it feeds.

---

## 8. The index-cast helper (`idx`)

`idx(long i)` is a one-line helper that widens a signed per-SNP column index to the
unsigned `std::size_t` that a `std::vector` or `std::span` subscript expects. It
exists to write that widening cast once instead of repeating the
`some_vector[static_cast<std::size_t>(...)]` boilerplate in the per-SNP loops of both
`assign_blocks` and `block_ranges`. Callers always pass a non-negative index (SNP
columns run from zero up to the count), so the cast is value-preserving and never
changes a result.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
