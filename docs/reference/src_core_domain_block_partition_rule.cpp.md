# `block_partition_rule.cpp` reference

## 1. Purpose

`src/core/domain/block_partition_rule.cpp` implements the one function that
decides which jackknife block each SNP belongs to: `assign_blocks`. A jackknife
block is a contiguous stretch of the genome that the uncertainty estimator leaves
out one-at-a-time to compute standard errors, so getting this partition exactly
right — and identical everywhere it is used — matters for every reported error
bar.

The types, the per-SNP helpers (`block_of`, `block_size_cm_to_morgans`), and the
full contract for `assign_blocks` are declared and documented in the companion
header. This file holds only the part that could not be a one-line inline
function: the walk that sweeps across all the SNPs in order while carrying state,
plus the guards and the branch that picks between two ways of drawing block
boundaries.

The file is deliberately free of any GPU code. It uses only the C++ standard
library, so it can be shared, byte-for-byte, by both the file-reading front end
and the GPU compute path. That single-source rule is the whole point: the block a
SNP lands in must never be computed two different ways.

The block boundaries produced here are engineered to match ADMIXTOOLS 2 exactly.
Several specific choices below (an inclusive comparison, a hardcoded 2-megabase
fallback window, a verbatim warning message) exist to reproduce ADMIXTOOLS 2's
behavior bit-for-bit, and are noted as such.

---

## 2. The block walk — the core algorithm

The heart of the file is a private helper, `block_walk`. It makes a single pass
over the SNPs in file order and assigns each one a block number.

### The rule it follows

Think of it as walking along the genome one SNP at a time, carrying a memory of
where the *current* block started — its "anchor," which is the position of the
first SNP in that block. A new block is opened whenever either of these happens:

1. **The chromosome changes.** A block never straddles two chromosomes.
2. **The distance from the anchor reaches the block width.** Once the current SNP
   is at least `window` away from the anchor, the current block is closed and a
   new one begins.

When a new block opens, the anchor immediately **re-sets to the position of the
SNP that opened it.** This re-anchoring is the single most important detail. It
means blocks are pinned to real SNP positions, never to fixed grid lines at
multiples of the block width. Any leftover distance past the threshold rolls
forward into the next block instead of being discarded.

### Why this is not a simple "divide position by width" binning

A naive rule would compute `floor(position / width)` and call that the block
number. That draws boundaries on a fixed grid — at width, 2×width, 3×width, and so
on — regardless of where the SNPs actually are. The walk here is different in two
visible ways:

- **A block is at least one width wide, and often wider.** The SNP that trips the
  threshold may sit well past it (if SNPs are sparse), and the block still ends
  there. Blocks can overshoot the nominal width; only the last, short block at the
  end of a chromosome is allowed to be narrower.
- **A sparse stretch is one block, not many empty ones.** A long gap with few SNPs
  becomes a single block, whereas a fixed grid would carve that same gap into many
  boundaries. So the total block count reflects where the SNPs really are, not how
  many grid cells the genome spans.

These are the observable consequences that the parity tests pin down.

### The inclusive comparison

The distance test uses "greater than or equal": a SNP that sits *exactly* one
width from the anchor closes the block. This inclusive form is a direct port of
the equivalent comparison in ADMIXTOOLS 2 (both its C and its R implementations),
and is kept inclusive on purpose so the boundaries line up with the reference.

### One walk, two regimes

`block_walk` is written to be generic over the position axis and the window size.
It does not know or care whether it is measuring distance in Morgans (a genetic
map) or in base pairs (a physical fallback). The exact same loop, with byte-for-
byte identical structure, serves both cases; only the arrays and the window value
passed in differ. This keeps the two block-drawing modes from ever drifting apart,
because there is only one implementation of the walk.

### The sentinels that open the first block

Before the loop starts, three carried values are set to deliberately impossible
"sentinel" values so that the very first real SNP is guaranteed to open block 0
through the ordinary rule, with no special-casing inside the loop:

| Carried value | Starting sentinel | Why |
|---|---|---|
| anchor position | `-1e20` | So absurdly far below any real position that the first SNP's distance from it always exceeds the window, forcing a cut. |
| previous chromosome | `-1` | No real chromosome code is `-1` (steppe uses 1 through 24), so the first SNP always looks like a chromosome change. |
| block counter | `-1` | The first cut bumps it to `0`, so the first block is numbered 0. |

Because both the chromosome check and the distance check are satisfied for the
first SNP, the counter advances to 0 and the anchor snaps to that SNP's real
position. Every SNP after that is handled by the same two checks with no
exceptions.

### What comes out

The result is a `BlockPartition`: one block number per SNP (parallel to the input,
in file order) plus the total number of distinct blocks. The block numbers are
**dense** — every value from 0 up to one-less-than-the-count is used — and
**non-decreasing** down the list, so each block's SNPs form one contiguous run.
The count is simply the final value of the block counter plus one.

---

## 3. Detecting a missing genetic map

A second private helper, `all_zero`, answers one question: is every genetic
position in the input exactly zero? That is the signal that a dataset shipped
without a genetic linkage map.

- **The comparison is exact — no tolerance.** The file readers write a literal
  `0.0` for a position column of `"0.000000"`, so an exact equality check against
  `0.0` is the correct and intended test; there is no rounding to worry about.
- **It stops at the first non-zero value.** A real map has a non-zero position at
  or near its very first SNP, so on normal data this check returns almost
  immediately and costs essentially nothing. It only scans the whole array in the
  genuinely all-zero case, which is exactly the case that needs the full scan.

Why this matters: without a genetic map, every genetic position is zero, so the
distance from any anchor is always zero, and the *only* cuts left are chromosome
changes. The walk would then collapse to one block per chromosome. That breaks the
uncertainty estimate — a subset drawn from a single chromosome would contain just
one block, which cannot yield a valid standard error. Section 4 describes the
fallback that avoids this.

---

## 4. `assign_blocks` — guards and dispatch

`assign_blocks` is the public entry point. It validates its inputs, decides which
of the two block-drawing modes to use, and calls `block_walk`. It does no walking
itself.

### The block-width guard

Before anything else, the block width is checked, and an illegal width returns an
empty partition (zero blocks) rather than producing a wrong one. The width feeds
directly into the distance comparison, so a bad value does not crash — it silently
produces a *wrong* partition, which is worse. There are three bad cases, each
failing in its own way:

- **Zero or negative width.** The distance from the anchor is always at least
  zero, so the "greater than or equal" test trips on every single SNP. Every SNP
  becomes its own block — a silent over-partition that still looks plausibly dense.
- **A "not a number" width.** Any comparison against a NaN is false, so the
  distance test never fires. No interior cut ever happens and an entire chromosome
  collapses into one block — a silent merge.
- Both defeat the purpose of matching the reference.

The guard is written as `!(width > 0.0)`. This single test rejects all three at
once: zero fails `> 0`, every negative fails `> 0`, and NaN fails `> 0` (any
comparison with NaN is false, so its negation is true). Writing it this way,
rather than as `width <= 0.0`, is what lets the one check also catch NaN.

This is the only place today where an illegal width can be caught, since the
configuration-validation layer that would normally reject it up front does not
exist yet. The chosen failure — an empty partition with zero blocks — is the same
well-defined, harmless result that empty input produces.

### Defensive handling of mismatched input lengths

The chromosome, genetic-position, and physical-position arrays are supposed to be
parallel, with one entry per SNP. If two of them disagree in length, that is a
programming error upstream — but rather than risk reading past the end of the
shorter one, the code takes the shorter length as the SNP count. An honest caller
always passes equal lengths, so this never changes a real result; it only bounds
the damage from a bug. Empty input (no SNPs) returns an empty partition.

### Choosing the fallback: base-pair blocks

After the guards, `assign_blocks` decides whether to draw blocks by genetic
distance (the normal case) or by physical distance (the fallback). The fallback is
taken only when **all** of these hold:

1. A usable physical-position axis was supplied — it is at least as long as the
   SNP count and comes with a positive window.
2. The genetic-position column is entirely zero (no genetic map — see section 3).
3. The physical-position axis is **not** itself all zero. An all-zero physical
   axis could not anchor blocks any better than the genetic one, so there is
   nothing to gain and the code stays on the genetic walk.

When the fallback fires, the code prints a warning to standard error — the exact
same message text ADMIXTOOLS 2 prints in this situation — once per call, and then
runs the identical walk over the physical positions with the physical window
(2 megabases by default). It uses raw base-pair positions directly, which keeps
the arithmetic in exact integer-valued numbers and makes the partition robust and
matched to the reference.

Crucially, a dataset that *does* have a real genetic map short-circuits the
all-zero check on its very first SNP and takes the ordinary genetic walk
completely unchanged. The fallback is a strict no-op for real-map data, which is
why adding it did not disturb existing reference-parity results.

### The normal path

If the fallback conditions are not met, `assign_blocks` runs the walk over the
genetic positions with the genetic block width. This is the common case for
datasets that carry a real linkage map.
