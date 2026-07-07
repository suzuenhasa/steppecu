# `li_stephens_validate.cpp` reference

## 1. Purpose

`src/core/stats/li_stephens_validate.cpp` is the up-front gatekeeper for a
`steppe paint` run — the Li-Stephens haplotype-copying forward-backward. Before a
single GPU kernel launches, before an alpha table is allocated, this file looks at
what the user asked for and either says "yes, this request is well-formed, go" or
stops with one clear sentence about what's wrong.

It is host-pure and CUDA-free by design. There is no device code here, no kernel,
no `RunConfig` merge — just a `PaintRequest` (a bag of scalar facts the `paint`
command has already resolved) plus the parsed `.snp` table. That plainness is the
whole point: the validator unit-tests with no files and no GPU, and it runs the
same "validate once, fail fast" posture the config builder and the f-stat sweep's
maxcomb cap already use. Nothing expensive happens until every contract below
holds.

The one public entry point is `validate_paint_request(req, snp, err)`. It returns
`Status::Ok` on a good request, or `Status::InvalidConfig` with a human-readable
reason written into `err`. It never throws.

---

## 2. Why paint needs a gatekeeper at all

The copying model is unforgiving about its inputs in ways a user can't always see
coming, and the failures it produces without a guard are quiet and awful:

- Feed it a diploid (unphased) genotype and the "haplotype" it copies is a fiction.
- Feed it a broken genetic map and the recombination probabilities between SNPs are
  garbage — silently.
- Let a haplotype copy itself in an all-vs-self panel and the copying diagonal
  swamps everything with a degenerate self-match.
- Point it at a panel too big and it will happily try to do a trillion state
  updates, or ask for a forward table that won't fit in memory.

Every one of those is cheaper to catch here, as a sentence on the terminal, than as
a wrong number three hours into a run. So this file enforces five contracts (the §3
contracts from the engine scope doc), and the ordering below is deliberate: the
numeric knobs get sanity-checked early so the later map and cost math can lean on
them being finite and positive.

---

## 3. Contract 1 — phased, haploid input

This is the load-bearing dependency of the whole engine. The Li-Stephens model
copies **phased haplotypes**, and steppe builds no phaser. So if any sample across
either triple decodes as diploid — meaning a heterozygous call was actually seen —
the request is rejected outright.

The command counts those samples into `req.n_diploid_samples`, and any positive
value is a hard stop. The error names the count and tells the user exactly what to
do: phase upstream with SHAPEIT or Beagle so each haplotype becomes its own haploid
column of the genotype triple (two haploid columns per diploid). There is no
"proceed anyway" override for this one — a copying model over unphased data isn't a
degraded result, it's a meaningless one.

---

## 4. Contract 4 — the numeric knobs (checked first)

Although it's "contract 4" in the spec, the panel geometry and numeric knobs are
validated **before** the map, because the map and cost arithmetic downstream assume
these are already finite and positive. The checks:

- **At least one recipient and one donor.** `--prefix` resolving to zero recipient
  haplotypes, or `--donors` resolving to zero donors, is an empty job — refused.
- **`Ne > 0` and finite.** The effective population size drives the recombination
  probabilities; a non-positive or non-finite `Ne` would poison them.
- **`theta` legal or auto.** If the user pinned a fixed emission rate rather than
  letting it auto-derive (Watterson's theta over K donors), it must be a finite
  value in `[0, 1]` — it's a probability. When `theta_auto` is set the fixed value
  is ignored and this check is skipped.
- **`recip-batch >= 1`.** The wave size can't be zero or negative.

These are small, but they're the ones that turn a later multiply or a `log` into a
NaN if they slip through, so they go first.

---

## 5. Contract 2 — the genetic map: present and monotonic

Recombination between adjacent SNPs is a function of genetic distance, so paint
needs a real centimorgan map, and it needs that map to make sense.

**Present.** The `.snp` table's `genpos_morgans` column is the map. There's a
subtlety in detecting its absence: a `.snp` file with no genetic-position column
doesn't fail to parse — the io leaf just derives an all-zero `genpos_morgans`. A
genuine map is a strictly increasing cM run within each chromosome, so an all-zero
column is unambiguously "no map." The local `map_absent()` helper is exactly that
test: any non-zero value means a map is present. An empty or zero-SNP table is
rejected up front regardless.

When the map is absent, the default is a hard error — paint tells the user the
genpos column (column 3, Morgans) is missing and that a real cM map is required.
The one escape hatch is `--bp-fallback`, which the user must opt into explicitly: it
approximates recombination from a fixed base-pair window instead. And even that
isn't a free pass — if bp-fallback is requested but the `.snp` has no physical
positions either, there's no scale left to derive a recombination rate from, and
that's its own error.

**Monotonic within each chromosome.** A present map must be non-decreasing as you
walk consecutive SNPs on the same chromosome. A *decreasing* cM run is the classic
map-merge bug — two maps stitched together at a chromosome join, producing a
backward step. The check compares only consecutive same-chromosome SNPs (the map
naturally resets across a chromosome boundary, so a cross-boundary "decrease" is
expected and skipped). When `chrom` labels are available it uses them to draw those
boundaries; when they aren't, it treats the whole table as one run. The moment it
finds a backward step it reports the SNP index, the chromosome, and the exact
`before -> after` Morgan values, so the user can find the bad join in their map file.

---

## 6. Contract 3 — self-copy / leave-one-out coherence

When the donor panel is a superset of the recipients — the ChromoPainter
all-vs-self painting shape, where haplotypes are painted against a panel that
includes themselves — a haplotype must not be allowed to copy itself. If it could,
the copying diagonal would be a degenerate self-match that dominates the posterior:
the best donor for a haplotype is always itself.

So the locked policy is: in that superset case, leave-one-out is required, which
means `--self-copy` must be **off**. The validator catches the one incoherent
combination — `donors_superset_recipients` true *and* `self_copy` true — and
refuses it, telling the user to turn `--self-copy` off. (When the donor and
recipient sets are disjoint there's no self to match, so `--self-copy` is free to be
whatever the user likes; this guard only fires on the superset overlap.)

---

## 7. Contract 5 — the cost guard

The last two checks are about not letting the user accidentally start a job that
never finishes or never fits.

**Total work.** The forward-backward does on the order of `N × K × M` state
updates — recipients times donors times SNPs. That product is computed in `double`
(so a large panel can't overflow the multiply) and compared against
`kLsMaxWorkStates` (currently `1e12`, from `include/steppe/config.hpp`). Past the
cap, the request is refused with the actual `N`, `K`, `M` spelled out and a pointer
to `--sure`, which lifts the cap for users who genuinely mean to run a long job.

**Forward-table footprint.** Even a job under the work cap can ask for a forward
table too big to hold. The GPU engine keeps one wave's alpha table resident —
`recip_batch × K` doubles at a time — and that must fit. The batch is clamped to the
recipient count (you never allocate a wider wave than there are recipients), the
byte count is computed, and if it exceeds `kLsMaxAlphaFootprintBytes` (currently
4 GiB) the request is refused with advice to lower `--recip-batch`. This is the knob
that keeps the checkpoint/recompute design honest: the full `O(K × M)` alpha table
is never resident — only the current wave's `recip_batch × K` slice is — and this
guard is what enforces that the slice stays within budget.

Note that `--sure` lifts *only* the work cap, not the footprint cap. Being willing
to wait a long time doesn't make an oversized allocation fit, so the footprint check
has no override — the fix is always to shrink the batch.

---

## 8. Contracts vs. the numerics

Worth being clear about what this file does and does not touch. It validates the
*request* — shapes, knobs, map sanity, policy coherence, cost. It does not do any of
the copying math. The forward-backward recurrence itself, the per-column rescaling,
the native-FP64 scan and reductions, the sum-to-one posterior, the checkpoint /
recompute that keeps the alpha table off-resident — all of that lives in the
CpuBackend reference oracle and the GPU engine, downstream of a clean bill of health
from here.

(One precision note that belongs with the engine, not this validator: the copying
scan and its reductions run in **native FP64**. The emulated-FP64 Ozaki default is a
matmul-only technique, and there is no GEMM anywhere in the Li-Stephens path — it's a
scan with rank-1 recombination and a donor-reduction, not a matrix multiply — so the
emulated path simply doesn't apply here.)

---

## 9. Edge cases and behaviour worth knowing

- **`err` is always cleared first.** Every call starts by clearing the out-param, so
  a stale message from a previous call can't leak into a success.
- **First failure wins.** The checks run in a fixed order and return on the first
  contract that fails — the user fixes one thing, re-runs, and sees the next, rather
  than getting a wall of possibly-cascading errors.
- **Zero-SNP / empty map** is rejected in the map section before any monotonicity
  walk, so the loop never indexes an empty table.
- **Missing `chrom` labels** don't break the monotonicity check — it degrades to
  treating the table as a single chromosome rather than throwing.
- **`--sure` is narrow.** It lifts the work cap and nothing else — not the footprint
  cap, not the phasing contract, not the map requirement.
- **Everything is data-in, status-out.** No I/O, no allocation, no device — which is
  exactly why it's the cheap, fast, fully unit-testable first line of defence for the
  whole paint engine.
