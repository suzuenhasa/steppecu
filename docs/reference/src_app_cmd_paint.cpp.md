# `cmd_paint.cpp` reference

## 1. Purpose

`src/app/cmd_paint.cpp` is the front end for `steppe paint` — the command that
drives the Li-Stephens haplotype-copying forward-backward. It is the glue between
the two genotype triples a user hands in and the batched forward-backward that
lives in the backend (`ComputeBackend::ls_paint_coancestry` on the GPU CUDA
backend, or the CPU reference oracle when no device is visible). The file itself
runs no scan: it reads and canonicalizes the panels, decodes them to haploid
allele bytes, builds the model inputs (recombination probabilities, the copying
prior, the emission rate, the genetic weights), calls the backend wave by wave,
folds the small returned accumulators into the shape the user asked for, and
prints them.

Two faces share this one command:

- **`paint`** (the default) — the *coancestry* face. It collapses the whole SNP
  axis into a small per-recipient × per-donor summary: how many copied chunks and
  how much genetic length each donor (or donor label) contributed. This is the
  ChromoPainter-style output.
- **`localanc`** (`--face localanc`) — the *local-ancestry* face. It keeps the SNP
  axis and reports, at every marker, the posterior probability that the recipient
  copied from each ancestry label. This is the per-position ancestry call a
  FLARE/RFMix-style aligner keys on.

The load-bearing contract, enforced before any compute, is that the input is
**pre-phased haploid data**. steppe builds no phaser; each haploid column in a
triple *is* one haplotype (one donor, or one recipient). A diploid triple is
refused up front (section 4).

---

## 2. What a paint run reads

`steppe paint` takes two genotype triples, each a `PREFIX.{geno,snp,ind}`:

| Flag | Role |
|---|---|
| `--prefix` | The phased **recipient** haplotypes — the ones being painted. |
| `--donors` | The phased **donor** panel — the library recipients copy from. |

Both are required; a missing prefix is an invalid-configuration exit with a clear
message naming which one. Each is resolved to a triple and read through a
host-only `CpuBackend` used purely as the io / transpose / ploidy-detect oracle —
the front end needs *a* backend to canonicalize a tile, but this read backend is
not the one that runs the forward-backward. The read keeps **every** individual
(`all_individuals()` — a `MinN` selection with a floor of 1), because in a phased
panel every haploid column is a haplotype worth keeping, not a member to be
filtered.

When `--prefix` and `--donors` name the **same** triple, that is the panel-vs-self
(all-vs-all) painting case: the two reads use identical selections so individual
order lines up, recipient `r`'s own copy is donor `r`, and that self donor is
zeroed out of the copying prior (leave-one-out — section 6).

---

## 3. Building the forward-backward model inputs

Once both panels are read and validated, the command assembles the plain arrays
the backend forward-backward consumes. Everything here is host arithmetic from
`core/stats/li_stephens.hpp`:

- **`donors` / `recips`** — each panel decoded to a flat, haplotype-major buffer of
  allele bytes `allele[g*M + l]` in `{0, 1, missing}` (`decode_haplotypes`). Every
  individual is one haploid column; the layout is exactly what the FB expects
  (donor-major for the donor library, recipient-major for the recipients it
  batches over).
- **`rho`** — the per-SNP recombination ("switch") probability
  (`build_recomb_probs(chrom, gpos, Ne, K)`). `rho[0]` is 1.0 (the first column has
  no predecessor, so it is fully unlinked and `alpha_0 = pi · e_0`), and any
  chromosome change resets `rho` to 1.0 (unlinked across the boundary).
- **`mu`** — the per-SNP emission ("miscopy") rate, a constant `theta_val` at every
  marker. With `--theta auto` this is the Watterson rate over `K` donors
  (`watterson_emission_rate(K)`); a user `--theta` overrides it.
- **`w`** — the per-SNP genetic-length weight (`build_genetic_weights`), the
  Morgans each marker stands for, used by the coancestry face to turn the copying
  marginal into an expected copied *length*.
- **`pi_all`** — the per-recipient copying prior over the `K` donors (section 6).

The panel size `K` is the donor count; `Nrec` is the recipient count; `M` is the
shared marker count. Before any of this the command insists the recipient and
donor `.snp` sets **match** — same `M`, and the same chromosome and genetic
position at every index — because the copying model only makes sense when donors
and recipients are described on one common map. A mismatch is an
invalid-configuration exit naming the offending marker index.

---

## 4. The contracts checked before compute

`validate_paint_request` (in `li_stephens_validate.cpp`) runs on the gathered
scalar facts *before* a single kernel launches — the same validate-once, fail-fast
posture the config builder and the f-stat sweep cap use. It enforces the five
model contracts:

1. **Phased/haploid input.** The command counts, across both triples, how many
   samples auto-detect as diploid (`count_diploid` — a heterozygous code was seen
   in the detection window). That count must be **zero**. A diploid triple is
   rejected with a "phase first" error, because the model copies phased
   haplotypes and steppe ships no phaser.
2. **A usable genetic map.** The `.snp` map must be present and monotonic within
   each chromosome; a missing map is a hard error unless `--bp-fallback` is
   explicitly opted into.
3. **A coherent self-copy policy.** When the donor panel is the recipient panel
   (panel-vs-self), self-copy must be off, or the copying diagonal degenerates
   into a self-match.
4. **Legal scalars.** `Ne > 0`, a legal (or auto) `theta`, `recip-batch >= 1`.
5. **A cost guard.** The `O(N·K·M)` work and the per-wave forward-table footprint
   are estimated up front and refused past a cap unless `--sure` lifts it.

A failed check is an invalid-configuration exit carrying the validator's one-line
reason.

---

## 5. Why the forward-backward never blows up memory

The command drives the backend one **recipient wave** at a time — `wave` (or
`wave_la` for localanc) recipients per call, `wave = max(1, recip_batch)`. This is
the visible half of the engine's core trick: the backend runs one thread block per
recipient in the wave, and inside each block it fuses the donor reduction, the
rank-1 recombination update, and the emission into a single per-column scan with
native-FP64 rescaling. The full `K × M` copying posterior γ for a recipient — the
`O(K·M)` alpha table — is **never resident** and **never leaves the device**;
checkpoint/recompute rebuilds the slices it needs during the backward pass.

What crosses back to the host is only the small, already-reduced accumulator:

- coancestry: an `N × K` chunk-count array and an `N × K` chunk-length array;
- localanc: an `N × M × P` per-SNP, per-label posterior.

So this file's host buffers scale with the *summaries*, not with the
recipient-times-donor-times-marker product the scan touches internally. The
command copies each wave's slice into the right offset of the full result and
moves on.

The backend used for the actual run is chosen after validation: the CUDA backend
on the requested device when a GPU is visible (`visible_device_count() > 0`),
otherwise the CPU reference oracle. Either way the run is **native FP64** — the
scalar rescaled scan and its reductions want real double precision, and there is
no matrix multiply here for the emulated-FP64 (Ozaki) default to help with. A
device-init failure is a clean io-error exit.

---

## 6. The copying prior and leave-one-out

`pi_all` is one `K`-length prior per recipient, laid out `Nrec × K`. Each row comes
from `build_uniform_pi(K, self_r)`:

- The normal case is uniform, `1/K` per donor.
- In the panel-vs-self case with self-copy off (`loo` true — donors are the
  recipient panel and the user did not ask for self-copy), recipient `r`'s own
  donor index `r` is zeroed and the remaining `K-1` donors share `1/(K-1)`. That
  is leave-one-out: a haplotype is not allowed to copy itself.

Building a full per-recipient prior array (rather than one shared vector) is what
lets the leave-one-out self index differ from wave to wave — recipient `r`'s zeroed
column is at position `r`, so the batched backend can slice `pi_all` by wave and
each recipient still leaves out the right donor.

---

## 7. Donor labels — the ancestry partition

Both faces need to know which *ancestry* each donor haplotype belongs to. The
default label for a donor is its population label from the `.ind`/tile
(`per_individual_labels` — individual `k` inherits the label of the population
whose offset range contains it). A `--labels` file overrides this: one label per
donor **haplotype column**, in `.ind` order. The file is read leniently (blank
lines skipped; a second whitespace token is taken as the label, else the first),
but its count must equal `K` exactly — and the error message is explicit that for
phased diploid donors that means *two* identical entries per individual, one per
haplotype column. A count mismatch is an invalid-configuration exit.

From those labels the command derives the distinct label set in first-appearance
order (`group_labels`, size `P`) and a `label → column` map (`group_index`). `P` is
the number of ancestry columns the summaries collapse to.

---

## 8. The localanc face (Phase 3)

When `--face localanc`, the command keeps the SNP axis. It builds `donor_group[k]`
— each donor's ancestry-label index in `[0, P)` — and hands it to
`ls_localanc(...)` as a compute input, wave by wave. The backend, for every marker,
sums the copying posterior γ over all donors sharing a label, giving an `N × M × P`
per-position ancestry posterior; the `K × M` γ never leaves the device, only the
`N × M × P` result returns. A non-`Ok` status from the backend is an
invalid-configuration exit; a thrown exception is an io-error exit.

The emit stage writes the posterior in long format, one row per
(recipient, SNP, ancestry label), carrying the marker coordinates a downstream
aligner keys on — `snp_id`, `chrom`, `pos_bp`, and `genpos_cM` (Morgans scaled to
centimorgans) — so no external join file is needed. CSV/TSV and JSON are both
supported; the JSON nests SNPs under each recipient and labels under each SNP.

---

## 9. The coancestry face (Phase 2, the default)

The default `paint` face collapses the SNP axis. It runs the backend
`ls_paint_coancestry(...)` wave by wave, accumulating two `Nrec × K` arrays:
`chunkcounts` (the expected number of copied chunks from each donor) and
`chunklengths` (the expected total copied length, in Morgans, integrated against
the section-3 genetic weights `w`). Again only these small `N × K` accumulators
come back; the `K × M` posterior stays on the device.

By default the output is aggregated to the **ancestry-label** level: the `N × K`
per-donor accumulators are folded down to `N × P` per-label totals
(`g_cnt` / `g_len`) using `group_index`. With `--full` the per-donor `N × K`
detail is emitted directly instead — one column per donor haplotype rather than
one per label.

Emit is long format again — one row per (recipient, donor-or-label) — with
`expected_chunks` and `expected_length_cM` (lengths scaled from Morgans to
centimorgans). The column header switches between `donor` and `donor_label`
depending on `--full`. CSV/TSV and JSON are both supported.

---

## 10. Invariants and edge cases

- **Phased input is mandatory.** Any diploid sample across either triple is a hard
  refusal (section 4). This is the whole model's precondition, checked before
  compute, never worked around.
- **Shared marker set.** Recipients and donors must agree on `M` and on every
  marker's chromosome and genetic position; the command verifies this index by
  index and refuses a mismatch.
- **Native FP64 throughout the run.** The rescaled scan and its reductions run in
  real double precision on whichever backend does the work; the emulated-FP64
  default is matmul-only and does not apply here (there is no GEMM).
- **Posterior normalization is the backend's job.** The forward-backward
  per-column rescaling keeps the copying posterior summing to one at each marker;
  this front end folds already-normalized γ into the summaries and never
  renormalizes.
- **Chromosome boundaries are unlinked.** `rho` and the genetic weights both reset
  at a chromosome change, so a recipient never "copies across" a boundary.
- **Degenerate priors.** Leave-one-out with `K = 1` would leave a recipient with no
  legal donor; the validator's scalar and self-copy checks (section 4) are what
  keep the prior well-formed before the backend ever sees it, so an
  all-zero/degenerate copying column doesn't reach the scan.
- **No device, no problem for correctness.** With no visible GPU the run falls back
  to the CPU reference oracle — the same exact, per-column-rescaled, native-FP64
  forward-backward, just not batched on hardware. Results match; only throughput
  differs.
- **Exit codes.** Missing/malformed inputs, a marker-set mismatch, a failed
  validator check, a bad `--labels` count, or a non-`Ok` backend status all exit
  invalid-configuration; an io/read/device failure exits io-error; a clean run
  exits success. Each carries a one-line diagnostic on standard error naming what
  went wrong.
