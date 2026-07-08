# `steppe roh` reference

## 1. What it is

`steppe roh` finds **runs of homozygosity (ROH)** in a single ancient genome:
long stretches of chromosome where the person inherited the *same* haplotype from
both parents, so there is no variation across the whole tract. Long ROH are the
signature of recent shared ancestry between someone's parents — close-kin
marriage, a small isolated population, a founder bottleneck. Short, common ROH are
just the background of any human genome. `steppe roh` scans the target's genome
against a panel of phased reference haplotypes, calls the homozygous-by-descent
tracts, and hands you back a table of segments plus a per-individual summary of how
much ROH sits above each length cutoff.

The real question it answers: *does this one ancient individual carry the genomic
signature of a small or inbred population — and how much?*

This is the GPU port of **hapROH** (Ringbauer et al. 2021, the `hapsburg` package),
run faithfully in its production `e_model=haploid` shape — not a simplified
stand-in. It is the last face of steppe's ancient-DNA relatedness family, and it
reuses the same phased-panel EIGENSTRAT reader as `steppe paint` and the same
Li-Stephens forward-backward substrate as `steppe ibd`.

## 2. The method, in one paragraph

hapROH copies the target genome, site by site, against a panel of phased reference
haplotypes using a copying hidden Markov model. There are `K+1` hidden states:
state 0 is the **non-ROH background** (the target's allele is drawn from the
Hardy-Weinberg allele frequency), and states `1..K` each say "here the target is
copying reference haplotype `k`" — that is where a ROH lives, `K = 2 × n_ref`
haplotypes (about 5008 for the full phased 1000-Genomes panel). At each site the
emission compares the target's allele to the copied reference haplotype (a
match/mismatch with error rate `e_rate = 0.01`) for the copying states, and to the
panel allele frequency for the background state. The transitions are a
symmetry-collapsed rate table — a small rate of jumping *into* a ROH (`roh_in = 1`
per Morgan), a larger rate of jumping *out* (`roh_out = 20`), and a rate of
switching which reference haplotype is copied while staying inside a ROH
(`roh_jump = 300`) — turned into per-gap transition probabilities from the genetic
map. A forward-backward pass gives, at every site, the posterior probability that
the target is in *some* copying state (i.e. in a ROH). steppe runs that
forward-backward on the GPU (one block per target), then post-processes the
posterior into called segments: threshold each site at `cutoff_post = 0.999`, form
contiguous runs, drop runs shorter than the length floor, and merge runs separated
by a tiny gap (the hapsburg `min_len1 = 0.02 M` / `min_len2 = 0.04 M` merge gates).
Finally it rolls the segments up per individual into the summary table.

Because `K × M` (haplotypes × sites) is roughly 2 GB per target, the GPU kernel
streams the forward pass with checkpoint/recompute rather than holding the whole
lattice — closer in shape to `steppe paint` than to `steppe ibd`. The numerics are
ported to match production hapsburg: the column-0 lattice cell is prior-only with
no emission (matching hapsburg's `cfunc.pyx`), the rescale is native FP64
per-column, and the REF/ALT polarity of every intersected site is asserted between
target and panel (a consistent strand flip is applied; a genuine mismatch is
dropped, not silently mis-copied).

## 3. What it matches (the gate)

`steppe roh` was gated on box5090 (Release, sm_120, `-Werror`, real data) against
**pip hapROH / hapsburg**, using a 912 MB Zenodo 1000G-1240K phased reference panel
(subset to 3 chromosomes, ~167,745 sites × 5008 haplotypes — disk-tight, so a
documented subset) copied against real AADR v66 ancients.

The scored positive control was **I1178.AG** — an Israel Late Chalcolithic
individual with a genuinely high ROH burden (a 91 cM maximum run). Against pip
hapROH on that data, steppe was **byte-exact**:

- per-bin total ROH (`sum_roh` for `>4 / >8 / >12 / >20` cM):
  **158.9 / 158.9 / 149.1 / 149.1 cM**, identical in every bin
- **4 vs 4** called segments, with boundaries byte-identical to hapROH's own raw
  `roh.csv`
- segment overlap: **Jaccard = recall = precision = 1.000**

That match was confirmed by an independent fresh steppe run — a faithful-port
result, not a copied output file. By nsys the ROH forward-backward kernel
(`roh_fb_kernel`) is 100% of GPU time (there is no host forward-backward loop).
Full test suite: ctest 107/107, including the new `roh_model_unit`,
`roh_segments_unit`, and `cli_roh` tests.

## 4. Speed

**Not measured.** This is the one relatedness-family face with no defensible
wall-clock number. The 912 MB real gate panel was cleaned off the disk-tight box
after the gate (panels are not re-downloaded), and no steppe wall-clock was
recorded at the gate itself — the commit notes only that `roh_fb_kernel` was 100%
of GPU time, not how long it ran. Rather than quote the tiny synthetic `cli_roh`
test fixture (which is not the real-panel run) or fabricate a figure, the honest
report is: unmeasured.

The reference tool's wall-clock was likewise **not separately timed at the gate**.
For context, hapsburg runs a single-threaded numpy forward-backward; at the gate
its low-ROH control run (see §8) had still not finished after a long grind while
steppe's GPU run was long done — but that is an observation, not a timed
comparison, so no speedup is claimed.

## 5. Inputs

Two EIGENSTRAT genotype triples (`.geno` / `.snp` / `.ind`), read through the same
front-end as `steppe paint`:

- **`--prefix`** *(required)* — the **target**: one ancient genotype triple
  (pseudo-haploid 1240K), the individual you are testing for ROH.
- **`--ref-panel`** *(required)* — the **phased reference-haplotype panel** triple,
  where each column is one phased haplotype. This panel is staged from the hapROH
  HDF5 panel to EIGENSTRAT by the helper script
  `tools/roh_panel_hdf5_to_eigenstrat.py` (a one-time staging step — steppe does
  not read HDF5 at runtime). The genetic map is read from the panel `.snp` Morgan
  column, so no separate map file is needed.

The two triples are intersected onto a common site axis by `(chromosome, physical
position)`, with the REF/ALT polarity contract applied per site.

Optional subsetting:

- **`--samples`** — a file of target population labels (one per line) to restrict
  to; default is every individual in the target triple.
- **`--exclude-pops`** — a comma-separated list of panel population labels to drop
  from the reference haplotypes (hapROH's `exclude_pops`), e.g. to remove the
  target's own population from the panel.
- **`--n-ref`** — cap the panel to this many reference *individuals*
  (`K = 2 × n-ref` haplotypes); default `0` keeps every panel column.

## 6. Outputs

Two tab- (or comma-) separated tables.

**Per-segment table** (`--out`, default stdout), one row per called ROH segment:

```
iid  ch  Start  End  StartM  EndM  length  lengthM  lengthCM  StartBP  EndBP
```

`iid` is the target label, `ch` the chromosome, `Start`/`End` are SNP indices in
the target's covered-site order, `StartM`/`EndM`/`lengthM` are genetic
positions/length in Morgan, `lengthCM` the length in centimorgans, `length` the SNP
count, and `StartBP`/`EndBP` the physical bounds.

**Per-individual summary** (`--summary`, default `<out>.summary`, or a note on
stderr if neither `--out` nor `--summary` is set), one row per target:

```
iid  max_roh  sum_roh>4  n_roh>4  sum_roh>8  n_roh>8  sum_roh>12  n_roh>12  sum_roh>20  n_roh>20
```

`max_roh` is the longest single ROH (cM); each `sum_roh>k` / `n_roh>k` pair gives
the total ROH length (cM) and segment count above length bin `k`, for the bins set
by `--roh-bins` (default `4,8,12,20` cM). Every selected target gets a row — an
individual with no called ROH emits an all-zeros row rather than being dropped, so
a clean low-ROH result is explicit in the output.

## 7. The CLI

The gate-shape run (target triple + staged panel triple, the 1000G default
`n-ref`):

```
steppe roh \
  --prefix I1178.AG \
  --ref-panel 1000G_1240K.phased \
  --n-ref 2504 \
  --device 0 \
  --out roh.csv
```

Key flags beyond the inputs in §5:

- **`--device`** *(default auto)* — CUDA device ordinal.
- **`--out`** *(default stdout)* — per-segment table path.
- **`--summary`** *(default `<out>.summary`)* — per-individual summary path.
- **`--roh-bins`** *(default `4,8,12,20`)* — the summary length bins, a comma list
  of cM values.
- **`--format tsv|csv`** *(default `tsv`)* — output field separator.

The HMM knobs — `--e-rate` (0.01), `--roh-in` (1), `--roh-out` (20), `--roh-jump`
(300), `--in-val` (1e-4), `--cutoff-post` (0.999) — default to the locked hapsburg
`hapsb_ind` values and normally need no touching; they exist so the model is
reproducible, not so you tune it.

## 8. Honest caveats

- **The low-ROH control is confirmed correct, but not yet cross-checked against
  pip hapROH.** On the negative control **CL121.AG**, steppe correctly calls **0
  ROH**. That result awaits its pip-hapROH confirmation — at the gate hapsburg's
  single-threaded numpy forward-backward was still grinding on it and had not
  finished, so the byte-exact concordance in §3 is established for the *high*-ROH
  positive control (I1178.AG) only. steppe's zero-ROH call on the low-ROH control
  is correct on its face but not yet pinned to a finished reference run.
- **`--n-ref` above ~5000 is rejected up front.** The column-0 forward prior is
  `1 − K × in_val` (with `K = 2 × n_ref`), which turns negative once
  `K × in_val ≥ 1` — i.e. `n_ref ≥ 1/(2 × in_val)`, about 5000 at the default
  `in_val = 1e-4`. hapsburg carries the same latent `1 − K × in_val` and silently
  emits a broken negative prior there; steppe instead **fails fast** with a clear
  message telling you to lower `--n-ref` or `--in-val`. The 1000G default
  `n_ref = 2504` (`K = 5008`, product `0.5008`) is safely inside the valid range.
  The same guard is re-checked against the *actual* selected haplotype count for an
  all-panel run (`--n-ref 0`) whose panel happens to be very large.
- **It needs a phased reference panel and a genetic map.** ROH calling is only
  meaningful against phased reference haplotypes with a per-site Morgan position;
  the panel `.snp` supplies both. Feed the staged hapROH panel (or an
  equivalently-formatted one), not an arbitrary genotype set.
- **Sites must intersect and agree on polarity.** A target site is dropped if it
  has no `(chrom, pos)` match in the panel, or if its REF/ALT can be reconciled
  with the panel's neither directly nor as a consistent flip. The run prints the
  kept-site count and the `no_match` / `polarity` drop counts — read them.
- **This is a per-individual method.** hapROH characterises one genome's own
  homozygosity from a reference panel; it is not a pairwise relatedness caller.
  For "who is related to whom," use `steppe ibd` (ancIBD) or `steppe readv2`.

## 9. Under the hood

The CPU backend is a **reference oracle only** — used to validate the GPU, never as
a user runtime. When a CUDA device is visible steppe runs the forward-backward on
the GPU (`roh_fb_kernel`, block-per-target, native-FP64 per-column rescale,
checkpoint/recompute at stride `ceil(sqrt(M))` because `K × M` is too large to hold
resident); with no device visible it falls back to the CPU oracle so the command
still works in a test/CI setting. The forward-backward runs in native FP64.

Source: `src/app/cmd_roh.cpp` (CLI + driver), `src/core/stats/roh_model.hpp`
(emission + transition rate table + the `roh_prior_valid` guard),
`src/core/stats/roh_fb.hpp` (the forward-backward interface),
`src/core/stats/roh_segments.{hpp,cpp}` (segment calling + summary),
`src/device/cuda/roh_fb_kernel.cu` (the GPU kernel). Committed in 5f80ea9.
