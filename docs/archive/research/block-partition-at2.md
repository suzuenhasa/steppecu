# Block-jackknife partition reconciliation: steppe vs ADMIXTOOLS (AT2)

**Headline.** steppe `assign_blocks` floor-bins SNP genetic positions into a FIXED
0.05-Morgan grid (`floor(cm/0.05)`) and cuts on every grid-line crossing; AT2
(`setblocks`, qpsubs.c:1698) walks SNPs sequentially and cuts when the cumulative
genetic distance FROM THE BLOCK-START SNP reaches `blgsize`, re-anchoring to the
actual SNP at each cut. On the Haak v66 union (264544 SNPs, blgsize 0.05 Morgan,
maxmiss=0) this is steppe **718** vs AT2 **709** — a +9 boundary-convention excess,
not a data/filter bug. **The fix is a drop-in rewrite of the single
`assign_blocks` loop (§4); it is CONTAINED — it touches no AT2-generated golden or
fixture (those are read, never recomputed by `assign_blocks`). RECOMMEND PROCEED.**

This spec is verified from THREE concordant sources (DReichLab C, admixtools R
2.0.10, admixtools Rcpp) and empirically reproduced on the real Haak `.snp` on the
box. Uncertainty is flagged explicitly in §3 and §6.

---

## 1. THE AT2 ALGORITHM — the exact block-cut rule

### 1a. Canonical C (DReichLab AdmixTools), `setblocks()`

`/workspace/AdmixTools_src/src/qpsubs.c:1698-1759` (verified verbatim on the box,
2026-06; the build of convertf v8621 came from this tree). Called by qpAdm at
`qpAdm.c:534`. Verbatim core:

```c
double
setblocks (int *block, int *bsize, int *nblock, SNP ** snpm, int numsnps,
	   double blocklen)
{
  int n = 0, i;
  int chrom, xsize, lchrom, olds;
  double fpos, dis, gpos;
  ...
  lchrom = -1;
  xsize  = 0;
  n      = 1;                 // NB: C counter is 1-BASED internally (see note below)
  fpos   = -1.0e20;           // sentinel: first real SNP always opens a block
  for (i = 0; i < numsnps; i++) {
    cupt = snpm[i];
    cupt->tagnumber = -1;
    if (cupt->ignore) continue;     // skipped SNPs do NOT participate (pre-filter)
    if (cupt->isfake) continue;
    chrom = cupt->chrom;
    gpos  = cupt->genpos;
    dis   = gpos - fpos;            // distance from the BLOCK-START snp's genpos
    if ((chrom != lchrom) || (dis >= blocklen)) {   // CUT: chrom change OR cum >= blgsize
      if (xsize > 0) {             // emit the just-closed block (post-loop twin below)
        ybsize[n] = xsize;
        if (block != NULL) block[n] = olds;
        if (bsize != NULL) bsize[n] = xsize;
        ++n;
      }
      lchrom = chrom;
      fpos   = gpos;               // RE-ANCHOR to THIS snp's genpos (remainder rolls fwd)
      olds   = i;
      xsize  = 0;
    }
    cupt->tagnumber = n;
    ++xsize;
  }
  if (xsize > 0) {                 // trailing short block emitted AS-IS
    ybsize[n] = xsize;
    if (block != NULL) block[n] = olds;
    if (bsize != NULL) bsize[n] = xsize;
    ++n;
  }
  *nblock = n;
  ...
}
```

The cut test is line `:1728`; the re-anchor `fpos = gpos` is line `:1738`.

### 1b. admixtools R 2.0.10 — `get_block_lengths()` (THE golden generator)

`uqrmaie1/admixtools@master R/resampling.R` (doc rule at `:241`, pure-R loop
`:278-300`, dispatch `:256`). Provenance confirmed: installed `admixtools` on box =
**2.0.10**, the exact version in the Haak cache metadata (§ Source citations). It is a
line-for-line mirror of the C walk:

```r
fpos = -1e20; lchrom = -1; xsize = 0; n = 0; bsize = c()
for (i in 1:nrow(dat)) {
  chrom = numchr[i]; gpos = dat[[distcol]][i]; dis = gpos - fpos
  if ((chrom != lchrom) || (dis >= blgsize)) {
    if (xsize > 0) { bsize[n+1] = xsize; n = n+1 }
    lchrom = chrom; fpos = gpos; xsize = 0       # anchor reset to the current SNP
  }
  xsize = xsize + 1
}
```

By default `get_block_lengths(cpp = TRUE)` dispatches to the compiled
`cpp_get_block_lengths()` (`src/cpp_resampling.cpp:9-39`, cut at `:23`) — the C++
default path that actually generated the golden; it returns the identical partition
to the R loop (verified on box). `distcol = 'cm'` (the genetic-map column) when
`blgsize < 100`. `afs_to_f2_blocks` calls `get_block_lengths(afdat$snpfile[sp,],
blgsize)` with `sp = polymorphic` — i.e. on the POST-filter SNP set.

### 1c. C-vs-R difference (none material; one cosmetic + one structural note)

- **Algorithm: IDENTICAL.** Same cut test `(chrom != lchrom) || (dis >= blgsize)`,
  same `>=`, same `fpos = gpos` re-anchor, same `fpos = -1e20` sentinel, same
  per-chromosome reset, same trailing-block emit. No behavioral difference.
- **Cosmetic:** the C counter `n` starts at **1** (1-based) with an unused `[0]` slot;
  R starts `n = 0`. The RESULTING block COUNT is the same — `*nblock` is the number of
  emitted blocks. Do NOT copy the `n = 1` start into steppe (steppe is 0-based dense;
  see §4). This is a C indexing convention, not an off-by-one in the partition.
- **Structural:** the C walks the in-memory `snpm[]` AFTER `cupt->ignore` / `isfake`
  filtering (cut SNPs are `continue`-skipped, so the walk sees only KEPT SNPs); the R
  walks `afdat$snpfile[polymorphic, ]` (already filtered). Both walk the KEPT set in
  file order. steppe must likewise feed `assign_blocks` the post-filter (chrom,
  genpos) in file order — which it already does.

**Key invariant (all three):** `fpos` is the genetic position of the block's FIRST
SNP. A cut opens only when cumulative distance *from that anchor* reaches `blgsize`;
the sub-blgsize remainder at each cut rolls FORWARD into the next block's anchor.
Blocks are anchored at actual SNP positions, NEVER at fixed grid multiples, and a
block spans `>= blgsize` (can be wider) except the trailing/short-chrom remnant.

---

## 2. STEPPE ALGORITHM — the exact `assign_blocks` rule

`src/core/domain/block_partition_rule.cpp:56-80` (the loop) +
`block_partition_rule.hpp:77-79` (`block_of`):

```cpp
// hpp:77-79 — the FIXED-grid per-SNP primitive
[[nodiscard]] inline int block_of(double genpos_morgans, double block_size_morgans) {
  return static_cast<int>(std::floor(genpos_morgans / block_size_morgans));
}

// cpp:52-80 — assign_blocks loop (state: prev_chrom, prev_local_bin, global)
for (long s = 0; s < M; ++s) {
  const int    c     = chrom[us];
  const int    local = block_of(genpos_morgans[us], block_size_morgans);  // floor(pos/0.05)
  const bool open_new = (s == 0) || (c != prev_chrom) || (local != prev_local_bin);
  if (open_new) ++global;
  out.block_id[us] = static_cast<int>(global);
  prev_chrom = c; prev_local_bin = local;
}
out.n_block = static_cast<int>(global) + 1;   // dense 0..n_block-1
```

steppe assigns each SNP to the ABSOLUTE grid cell `floor(cm / 0.05)` and opens a new
block whenever the chromosome changes OR the floor-cell index changes from the
previous SNP. The anchor is the FIXED grid line `k*0.05`, never reset to a SNP
position. Interior empty cells are absorbed (dense ids). The fail-fast width guard
(`cpp:35-46`) and the empty-input guard are correct and stay.

**Empirically (box probe, §3): steppe's per-chromosome block count ==
`n_distinct(floor(cm/0.05))` within the chromosome == number of OCCUPIED 0.05-Morgan
grid cells. Confirmed `st_block_count == occupied_cells` for ALL 22 chromosomes.**

Units are already correct (NOT the bug): R `io.R` assigns EIGENSTRAT `.snp` col 3
directly to `cm` with NO ÷100, so col 3 is Morgans and `blgsize = 0.05` compares
against Morgans — exactly as the C compares `gpos` (= `cupt->genpos`, Morgans) against
`blocklen`. steppe's `genpos_morgans` is the same map in Morgans and
`block_size_morgans = 0.05`. (The cM→Morgan ÷100 happens only on the *config*
`block_size_cm` via `block_size_cm_to_morgans`, hpp:92-94 — correct, single-homed.)
The bug is the BINNING RULE, not units.

---

## 3. THE PRECISE DIFFERENCE — what makes 718 vs 709

Both rules cut on every chromosome boundary identically (22 chrom cuts). The
divergence is entirely WITHIN chromosomes, and it is a genome-wide per-chromosome
**phase + partial-end-cell** effect (NOT localized to one chromosome):

1. **Phase.** AT2 anchors each block at the first SNP of that block; steppe anchors at
   the absolute grid line `k*0.05`. AT2 block starts drift to wherever SNPs fall
   (observed start spacing: min exactly 0.05, median ~0.0505-0.0512 — AT2 packs
   slightly MORE than blgsize per block because the remainder rolls forward), while
   steppe starts are pinned to `k*0.05`.
2. **Partial end cells.** steppe forces the first and last partial grid cell of every
   chromosome to be a full block. AT2's last block on a chromosome is whatever
   remainder is left (observed tiny chr-end remnant spans, e.g. chr1 last block
   0.0092, chr15 0.0081) — kept as its own block but NOT adding a grid cell beyond
   the walk count. Net: steppe tends to emit ~+1 block per chromosome.
3. **Empty interior cells (the offset).** Where SNPs are sparse, steppe ABSORBS empty
   grid cells (dense-id policy) while AT2 makes one long block. This REMOVES some
   steppe blocks (e.g. chr13: occupied 23 < full grid span 25). This is why the net
   excess is small (~+1/chrom minus absorbed empties), not ~+22.

Every steppe-extra cut is a **pure floor-grid artifact**: at all 645 steppe-extra cut
points on the probe set, the cM gap to the previous SNP was `< blgsize` (range
1e-6 .. 0.0337, median ~0.0011) with grid jump exactly 1 — NONE was a real
`>= blgsize` gap. steppe splits blocks AT2 deliberately keeps merged, purely because
a `k*0.05` grid line falls between two adjacent SNPs. Worked chr8 example: at SNP idx
1721 (cm 0.150104), AT2 keeps it in block 3 (block started cm 0.100985, dis 0.049119
< 0.05) while steppe cuts because `floor` crossed the 0.150 grid line.

### Per-chromosome diff (box reproduction)

Reproduced by rebuilding the aftable from the real geno
(`/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.{geno,snp,ind}`, the same source
as both caches), applying maxmiss=0 / auto_only / poly, then running BOTH partitions
on the identical kept (chrom, cm).

> **FLAGGED UNCERTAINTY — SNP count provenance.** A fresh `extract_f2` /
> aftable-filter pass kept **286646** polymorphic SNPs → **AT2 687 / steppe 712 (net
> +25)**, NOT the golden's 264544 / 709 / 718. The 22102-SNP gap is a
> data-provenance difference (the golden cache was built with an additional SNP
> restriction not captured by the metadata's documented filters — a `keepsnps` /
> outgroup / panel-intersection restriction). This does NOT change the algorithm
> conclusion: the MECHANISM and per-chromosome PATTERN are identical on both sets, and
> `st_block_count == occupied_grid_cells` holds exactly on both. The absolute 709 vs
> 718 split is the GOLDEN (cache_metadata n_blocks=709; steppe meta n_block=718); the
> +25 table below is the larger probe set. Treat the per-chromosome SHAPE as
> load-bearing and the 709-vs-718 golden as the parity target.

| chr | nsnp  | span(cM) | AT2 | steppe | occ_cells | excess |
|-----|-------|----------|-----|--------|-----------|--------|
| 1   | 14550 | 2.836    | 55  | 58     | 58        | **+3** |
| 2   | 15638 | 2.677    | 52  | 54     | 54        | **+2** |
| 3   | 14968 | 2.232    | 43  | 45     | 45        | **+2** |
| 4   | 8926  | 2.142    | 42  | 43     | 43        | +1     |
| 5   | 9166  | 2.009    | 40  | 41     | 41        | +1     |
| 6   | 18189 | 1.916    | 38  | 39     | 39        | +1     |
| 7   | 29629 | 1.871    | 38  | 38     | 38        | 0      |
| 8   | 29750 | 1.680    | 33  | 34     | 34        | +1     |
| 9   | 23103 | 1.661    | 33  | 34     | 34        | +1     |
| 10  | 24249 | 1.809    | 36  | 37     | 37        | +1     |
| 11  | 25801 | 1.582    | 32  | 32     | 32        | 0      |
| 12  | 8874  | 1.745    | 34  | 35     | 35        | +1     |
| 13  | 5670  | 1.243    | 20  | 23     | 23        | **+3** |
| 14  | 11058 | 1.139    | 23  | 24     | 24        | +1     |
| 15  | 4479  | 1.218    | 24  | 26     | 26        | **+2** |
| 16  | 4455  | 1.334    | 26  | 27     | 27        | +1     |
| 17  | 4711  | 1.271    | 25  | 26     | 26        | +1     |
| 18  | 6620  | 1.174    | 23  | 24     | 24        | +1     |
| 19  | 3256  | 1.077    | 21  | 22     | 22        | +1     |
| 20  | 11659 | 1.081    | 21  | 22     | 22        | +1     |
| 21  | 6784  | 0.613    | 13  | 13     | 13        | 0      |
| 22  | 5111  | 0.722    | 15  | 15     | 15        | 0      |
| **Σ** |     |          | **687** | **712** | **712** | **+25** |

Biggest contributions on the longest/densest chromosomes (chr1 +3, chr13 +3,
chr2/3/15 +2); 4 chromosomes (7, 11, 21, 22) net zero (absorbed empties exactly
offset partial-end-cells). The +9 on the golden 264544 set is the SAME effect at
smaller magnitude (fewer SNPs → more interior empties absorbed → more per-chr +1s
cancelled). The golden `block_lengths_f2.rds` (709 ints, Σ=264544) corroborates the
walk: it contains tiny chr-end / sparse-region remnant blocks — values 9, 9, 29, 6,
50, 62, 47, 64, 42, 74, 72, 85, 87, 95, 99, 100, 101, 103 — exactly the SNP-anchored
remainders the walk produces and a fixed grid cannot.

---

## 4. THE RECONCILIATION SPEC — make `assign_blocks` walk like AT2

Replace the fixed-grid floor loop with the SNP-anchored sequential walk. Drop-in for
the loop body in `src/core/domain/block_partition_rule.cpp:52-80` (keep the guards at
`:35-46` UNCHANGED):

```cpp
// AT2 setblocks() convention (qpsubs.c:1698-1759): SNP-anchored cumulative walk.
// A new block opens on a chromosome change OR when the cumulative genetic distance
// from the block's FIRST snp reaches block_size_morgans; the anchor re-sets to the
// SNP that opens the block, so the sub-blgsize remainder rolls FORWARD.
double fpos       = -1.0e20;  // genpos of current block's first SNP; sentinel forces first cut
int    prev_chrom = -1;       // sentinel: any real chrom differs -> first SNP opens block 0
long   global     = -1;       // dense 0-based block counter; first SNP bumps to 0

for (long s = 0; s < M; ++s) {
    const std::size_t us  = idx(s);
    const int         c   = chrom[us];
    const double      pos = genpos_morgans[us];
    // Cut on chrom change OR cumulative distance from the anchor >= blocklen (>=, not >).
    if (c != prev_chrom || (pos - fpos) >= block_size_morgans) {
        ++global;
        fpos       = pos;     // RE-ANCHOR to this SNP (NOT to fpos + blocklen / grid line)
        prev_chrom = c;
    }
    out.block_id[us] = static_cast<int>(global);
}
out.n_block = static_cast<int>(global) + 1;  // dense 0..n_block-1
```

Implementer notes (each is load-bearing for bit-tight parity):

- **`>=` not `>`.** Matches C `dis >= blocklen` and R `dis >= blgsize`. A SNP exactly
  `blgsize` from the anchor CUTS. Port verbatim.
- **Re-anchor to `pos`, NOT to `fpos + blocklen` and NOT to the grid line `k*0.05`.**
  Re-anchoring to a grid line reintroduces the steppe bug; the entire point is that
  the remainder rolls forward.
- **Sentinels `prev_chrom = -1`, `fpos = -1e20` replace the `s == 0` special case.**
  The first SNP always opens block 0 because its chrom differs from -1. Drop the
  `(s == 0)` term. CAVEAT: steppe chrom codes are 1..24 (AT2 1..22), so -1 is a safe
  sentinel; if steppe could ever emit a real chrom code of -1, use a separate
  `bool first = true` instead. (Today it cannot — safe.)
- **Drop `prev_local_bin` and the `block_of` call from this loop.** `block_of`
  (hpp:77-79) and `block_size_cm_to_morgans` (hpp:92-94) stay as primitives;
  `block_of` becomes UNUSED by `assign_blocks` — keep it (still a valid primitive) or
  remove if no other caller. NOTE the M3 unit test still references `block_of`
  directly (test lines 99, 137) — see §5; those test lines become moot, not the
  header.
- **Drop the negative-position special case.** The walk has no negative-bin concept
  (positions are `>= 0` within a chromosome; the sentinel handles the first SNP). The
  current "negative floors to -1 own bin" test (lines 132-143) is rewritten for the
  walk semantics.
- **Postcondition is preserved.** Output is still dense, non-decreasing,
  per-chromosome-contiguous, so `block_ranges()` (hpp:213-271) and its fail-fast
  contract are UNAFFECTED. The function stays a pure, launch-order-independent
  function of (chrom, genpos) in file order — the §8 DRY invariant and §12
  determinism hold. This is `numblocks()`-equivalent (qpsubs.c:1763).

---

## 5. THE BLAST RADIUS — CONTAINED, not a ripple

### 5a. MUST CHANGE — steppe-computed logic + assertions

- **`src/core/domain/block_partition_rule.cpp:52-80`** — the loop (the fix, §4).
- **`src/core/domain/block_partition_rule.hpp:119-130`** — the rule's doc comment
  ("track the previous SNP's local floor-bin… interior empty bins absorbed… negative
  positions get their own block") describes the OLD rule; rewrite to the walk.
- **`tests/unit/test_block_partition.cpp`** — re-author for the walk:
  - `kExpectedNBlock = 757` (line 283) is the FULL v66 chr1-24 steppe-COMPUTED count
    under the OLD rule; it WILL change and MUST be re-derived on the box under the new
    rule (the header narrative at lines 227-232 — "757 = plan's 756 + Y chr 24" — and
    the printf at lines 292-293, 305-309 update with it). UNMEASURED today (see §6).
  - the `block_of` sanity-pins (lines 99, 137) and the negative-bin case
    (lines 132-143) become moot under the walk; rewrite the synthetic cases to the
    walk semantics. The width-guard verdict cases (lines 154-186) and empty-input
    case (line 150) are unchanged (the guards stay).
- **Count literals used as VRAM/launch SIZING magnitudes (correctness-neutral, but
  update for accuracy):**
  - `tests/unit/test_vram_budget.cpp` — `757` at lines 58, 64-69, 78-81, 91, 114,
    153, 175.
  - `tests/unit/test_launch_config.cpp:179` — `max_blocks_per_chunk(..., 757, ...)`.
  - `tests/reference/test_filter_oracle.cu:450-451` — printed "chr1-24=>757,
    chr1-23=>756" narrative.
  - `tests/reference/bench_f2_multigpu.cu:11` — "n_block ~ 757" comment.
  These are sizing magnitudes, not parity gates; they do not gate goldens.

### 5b. DOCS to update (the 757/756/719 narrative)

`ROADMAP.md:73,74,78,178`; `TODO.md:5,11,62`;
`docs/.../cleanup/include-config.md:69,237,255`; `cleanup/device-backend.md` &
`core-internal-f2_estimator.md:52` (sizing magnitudes); `studies/haak2015.md:135-198`
& `haak2015-at2-reference.md:12` (these already track this as the open follow-up —
mark RESOLVED). `tests/cli/test_cli_extract_qpadm.cpp:35` references the prior
"719-vs-710 residual" — update its comment.

### 5c. MUST NOT CHANGE — AT2 goldens/fixtures are UNAFFECTED (confirmed)

Every committed block count is AT2-GENERATED and READ by steppe, never recomputed by
`assign_blocks`; the fit math consumes the fixture's existing block axis, so fit
goldens stay bit-exact:

- `tests/reference/goldens/at2/golden_fit1_NRBIG.json:51` (`n_block: 701`),
  `golden_fit0.json` (710), `golden_fitNA.json` (710/709), `golden_rot.json` (702) —
  all carry AT2's own counts, read not recomputed.
- all `tests/reference/.../fixtures/*.bin`.
- `tests/reference/test_qpadm_missing_block.cu:148-149` (710/709 from the AT2
  fixture).
- box `/workspace/data/haak/at2_f2/cache_metadata.json` (`n_blocks: 709`,
  `blgsize: 0.05`, `n_snps: 264544`, `maxmiss: 0`, `admixtools_version: 2.0.10`) —
  the standing empirical anchor, an AT2 artifact (verified this pass).
- structurally-only tests (`test_block_ranges`, `test_shard_plan`,
  `test_f2_blocks_multigpu`, `test_f2_partials_validate`) assert NO count →
  unaffected.

### 5d. Single source — no second copy to change

`assign_blocks` is the one home (`block_partition_rule.cpp`); both `io` and the
CUDA/CPU backends consume `block_id` from it via `block_ranges` (the §8 DRY
invariant). Confirm during implementation that no kernel re-derives `floor(pos/blg)`
(the CUDA path `f2_blocks_kernel.cu` / `cuda_backend.cu` should derive layout from
`block_ranges(block_id, ...)`, not recompute bins). If any does, it changes too.

---

## 6. GO / NO-GO

**GO — recommend PROCEED. This is a clean, bounded change.** It does NOT ripple into
any validated precompute golden or AT2 fixture (§5c), so it needs no sign-off to
regenerate goldens — the AT2 goldens are the parity TARGET this change moves steppe
TOWARD. The change is confined to one loop in `assign_blocks`, its M3 unit test
(re-derive the steppe-computed counts), a handful of correctness-neutral sizing
literals, and docs.

Intended consequences (the parity fix, expected and desired):

- steppe Haak block count 718 → **709**; steppe f2-cache `n_block` for this dataset
  should read 709 and the per-block `block_sizes` should match AT2
  `block_lengths_f2.rds` ELEMENT-WISE (the tightest parity gate — 709 ints summing to
  264544, containing the tiny chr-end remnants 9, 9, 29, 6, 50, 62, 47, 64, 42, 74…
  that the walk produces and a grid cannot).
- this flows into the jackknife SE/p and the ~0.5% qpAdm weight wobble (Q → GLS) —
  the whole reason for the change.

**Two items to MEASURE on the box AFTER the change (do not hardcode blind):**

1. The new full-v66 steppe count under the walk (replaces `kExpectedNBlock = 757` at
   `test_block_partition.cpp:283`). UNMEASURED — no modified code was run this pass.
2. One confirming `extract_f2` run on the Haak set to verify steppe emits 709 and the
   element-wise `block_sizes` == `block_lengths_f2.rds` match (cheap, closes the
   "fixed `assign_blocks` not compiled/run" gap).

**Flagged uncertainties (do not block GO):**

- The 264544-SNP golden set was not fully reconstructed from documented filters (a
  fresh pass kept 286646; +22102 is an undocumented SNP restriction). Mechanism and
  per-chromosome pattern are identical on both sets; the 709-vs-718 is the golden
  anchor. (§3)
- The exact 709 was reproduced on the AT2 side from its RDS/code; the fixed steppe
  `assign_blocks` was not compiled/run (item 2 above).

---

## Source citations

- **AT2 C convention:** `/workspace/AdmixTools_src/src/qpsubs.c:1698-1759`
  (`setblocks`; cut `:1728`, re-anchor `:1738`), `:1763` (`numblocks`); called by
  qpAdm at `qpAdm.c:534`. (Read verbatim on box this pass.)
- **AT2 R (golden generator, v2.0.10):** `get_block_lengths` (uqrmaie1/admixtools
  `R/resampling.R`: doc `:241`, dispatch `:256`, loop `:278-300`); Rcpp default
  `cpp_get_block_lengths` (`src/cpp_resampling.cpp:9-39`, cut `:23`);
  `afs_to_f2_blocks` calls it on the polymorphic post-filter set. Refs:
  https://uqrmaie1.github.io/admixtools/reference/extract_f2.html ,
  https://github.com/uqrmaie1/admixtools .
- **Golden metadata:** `/workspace/data/haak/at2_f2/cache_metadata.json`
  (`n_blocks:709, blgsize:0.05, n_snps:264544, maxmiss:0,
  admixtools_version:2.0.10`); block sizes
  `/workspace/data/haak/at2_f2/block_lengths_f2.rds` (709 ints, Σ=264544).
- **steppe current rule:** `src/core/domain/block_partition_rule.cpp:52-80`,
  `block_partition_rule.hpp:77-79` (`block_of`), `:92-94` (`block_size_cm_to_morgans`),
  `:119-130` (rule doc), `:213-271` (`block_ranges`, unaffected).
- **steppe golden output (current, pre-fix):**
  `/workspace/data/haak/steppe_f2_fixed/meta.json` & `steppe_f2_verdict/meta.json`
  (`n_block:718, blgsize_cm:5, n_snp_kept:264544`).
- **steppe computed counts:** `tests/unit/test_block_partition.cpp:227-233,283`
  (chr1-24=757, chr1-23=756, autosome=719); `ROADMAP.md:73-74`.
