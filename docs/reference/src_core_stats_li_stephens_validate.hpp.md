# `li_stephens_validate.hpp` reference

## 1. Purpose

`src/core/stats/li_stephens_validate.hpp` (with its `.cpp`) is the one place that
checks a `steppe paint` request is well-formed **before** any of the heavy
Li-Stephens work begins. `steppe paint` runs a GPU haplotype-copying
forward-backward: every recipient haplotype is explained as a mosaic of copies from
a panel of donor haplotypes, and the copying posterior it produces feeds either the
coancestry paint face or the local-ancestry face. That engine is expensive and it
leans on several assumptions about its inputs — the biggest being that the input is
already phased into haploid columns, because steppe ships no phaser of its own. This
validator makes every one of those assumptions explicit and refuses the run up front
if any is broken.

It is deliberately host-pure and CUDA-free. It touches no GPU, launches no kernel,
and never sees the forward-backward math — that lives in `core/stats/li_stephens.*`
(the recurrence and the CPU reference oracle) and `device/cuda/li_stephens_fb_kernel.cu`
(the batched GPU scan). This file operates only on already-parsed, plain-data facts:
a `SnpTable` (the SNP/genetic-map columns) plus a small bag of scalar request facts.
That keeps it the same **validate-once, fail-fast** shape the rest of steppe already
uses — `ConfigBuilder::build()` and the f-stat sweep's `maxcomb` cap have exactly
this posture — and it makes the whole check trivially unit-testable with no files and
no device. The `paint` command is what adapts a resolved `RunConfig` plus the two
read genotype triples onto the `PaintRequest` this validator consumes.

The single public entry point is:

```
[[nodiscard]] Status validate_paint_request(const PaintRequest& req,
                                            const io::SnpTable& snp,
                                            std::string& err);
```

It returns `Status::Ok` on a good request, or `Status::InvalidConfig` with a clear
one-line reason written into `err`. It **never throws** — a bad paint request is a
configuration outcome the caller reports and exits on, not an exception.

---

## 2. `PaintRequest` — the plain-data facts a paint run must satisfy

The command gathers the scalar facts of a request into a `PaintRequest` after it has
resolved the recipient and donor triples (their individual and SNP tables). Keeping
it plain data — no io handles, no device state — is what lets the validator be unit
tested standalone.

| Field | Meaning |
|---|---|
| `Ne` | Effective population size for the recombination scale. Must be finite and `> 0`. Default `20000`. |
| `theta` / `theta_auto` | The per-site mutation/emission rate. When `theta_auto` is true, the engine derives Watterson's theta over the K donors; otherwise `theta` is a fixed rate that must be finite and in `[0, 1]`. |
| `self_copy` | The user's `--self-copy`. `true` means a haplotype is allowed to copy itself (no leave-one-out). |
| `recip_batch` | How many recipients ride in one GPU wave. Must be `>= 1`. Default `256`. It bounds the resident forward-table footprint (section 5). |
| `allow_bp_fallback` | Whether recombination may fall back to a fixed base-pair window when the genetic map is absent (`--bp-fallback`). |
| `n_recipients` / `n_donors` | Panel geometry — the N recipient haplotypes and K donor haplotypes. Each must be `>= 1`. |
| `n_diploid_samples` | How many samples across **both** triples decoded as diploid (a heterozygous call was seen). Must be **zero** — the whole phased-input contract turns on this. |
| `donors_superset_recipients` | True when the donor panel is a superset of the recipients (all-vs-self "panel-vs-self" painting, the ChromoPainter shape). Governs the self-copy decision. |
| `sure` | The `--sure` posture — lifts the work-cost cap for a knowingly long job. |

---

## 3. The five contracts, in the order they're checked

The validator enforces five contracts. It checks them in a specific order chosen so
that later arithmetic is always safe (panel geometry and numeric knobs are validated
before any cost math runs on them). On the first failure it returns immediately with
a written reason.

### Contract 1 — phased / haploid input (the load-bearing one)

The copying model consumes phased haplotypes, one haploid column per haplotype, and
**steppe builds no phaser**. So if `n_diploid_samples > 0` — any heterozygous call
was seen anywhere in either triple — the request is rejected with a "phase first"
error that names the count and points at SHAPEIT/Beagle (which turn each diploid into
two haploid columns). This is checked first because it is the dependency the entire
model rests on: unphased input isn't a degraded run, it's a category error.

### Contract 4 — panel geometry and numeric knobs (checked early)

Although it's contract "4" by the scope numbering, it runs second because the later
cost math multiplies these values and must not do so on garbage:

- `n_recipients >= 1` (else `--prefix` resolved zero recipient haplotypes).
- `n_donors >= 1` (else `--donors` resolved zero donor haplotypes).
- `Ne` finite and `> 0`.
- If `theta` is not `auto`, it must be finite and in `[0, 1]`.
- `recip_batch >= 1`.

### Contract 2 — the genetic map is present and monotonic

Recombination in the Li-Stephens model is driven by genetic distance, so the `.snp`
column-3 genetic map (in Morgans) matters:

- An empty SNP table (`snp.count == 0` or no `genpos_morgans`) is a hard error.
- **Map-absent detection** is a simple, robust rule: a `.snp` with no genpos column
  parses to an all-zero `genpos_morgans` (the io leaf derives nothing), and a real
  map is a strictly increasing cM run per chromosome — so *all-zero means "no map"*.
  When the map is absent, that's a hard error **by default**, telling the user to
  supply a real cM map or opt into `--bp-fallback`. If `--bp-fallback` is set, the
  validator then insists physical positions exist to serve as the recombination
  surrogate — with neither a cM map nor bp positions there is simply no scale to
  derive.
- **Monotonicity**: a present map must be non-decreasing *within* each chromosome.
  The classic map-merge bug produces a decreasing step at a chromosome join, so the
  check only compares consecutive SNPs on the **same** chromosome (the map resets
  across a boundary, and comparing across one would false-positive). On a violation
  the error names the chromosome, the SNP index, and the offending
  `before -> after` Morgan values.

### Contract 3 — self-copy / leave-one-out coherence

When the donor panel is a superset of the recipients (`donors_superset_recipients`),
a recipient can find *itself* in the donor panel. If `--self-copy` were also on, the
copying posterior would collapse onto that perfect self-match — a degenerate diagonal
that tells you nothing. The locked decision is that leave-one-out is required in that
geometry, so `donors_superset_recipients && self_copy` is rejected, asking the user
to turn `--self-copy` off. (Disjoint donor/recipient panels don't hit this — there's
no self to copy.)

### Contract 5 — the cost guard

See section 5.

---

## 4. Why native FP64, and where the precision actually lives

A note for anyone reading this alongside the precision policy: the Li-Stephens engine
runs its forward-backward scan and its reductions in **native FP64**, not the
emulated-FP64 (Ozaki) default. That default is a *matmul-only* trick — it accelerates
the big GEMMs and SYRKs of the f-stat / fit paths — and there is **no GEMM anywhere in
the copying model**. The forward-backward is a sequential per-column scan with a
rank-1 recombination update, and the coancestry / local-ancestry sinks are plain
reductions; none of that is a matrix multiply, so there is nothing for emulation to
stand in for. Native FP64 is the right and only choice here.

This validator itself does no FP64 compute — it's scalar host checks — but it is the
gate that guarantees the engine downstream gets the well-formed, phased, mapped input
that its native-FP64 per-column rescaling needs to stay in range and sum to one.

---

## 5. The cost guard math

The forward-backward does **O(N·K·M)** state updates — N recipients times K donors
times M streamed SNPs — and each GPU wave holds a `recip_batch·K` forward table (the
alpha table) resident in FP64. Both are capped up front, with the same `--sure`
override the f-stat sweep uses:

- **Work cap.** `work = n_recipients · n_donors · snp.count`, computed in `double` so
  a huge panel can't overflow the product. If `work > kLsMaxWorkStates` (`1e12`, a
  "minutes not hours" envelope) and `--sure` was **not** passed, the run is refused
  with an error spelling out N, K, M and the cap, inviting `--sure` for a knowingly
  long job.
- **Resident forward-table cap.** The per-wave alpha footprint is
  `min(recip_batch, n_recipients) · n_donors · sizeof(double)` bytes — the batch is
  clamped so a batch larger than the recipient count doesn't over-report. If that
  exceeds `kLsMaxAlphaFootprintBytes` (4 GiB, a few-GB resident wave) it's a **hard**
  error (no `--sure` escape) asking for a smaller `--recip-batch`, because unlike the
  work cap this one is a real VRAM ceiling, not just a patience one.

Both constants live in `include/steppe/config.hpp`.

---

## 6. Edge cases and testability

- **Never throws.** Every failure path routes through the small local `fail` helper,
  which writes `err` and returns `Status::InvalidConfig`. `err` is cleared at entry,
  so a successful validate leaves it empty.
- **Empty SNP table** is caught explicitly rather than silently treated as "no map".
- **All-zero genetic map** is the map-absent signal, not a monotonicity violation —
  the absent-map branch fires first, so an all-zero map never reaches the monotonic
  loop.
- **Missing chromosome column.** If `snp.chrom` isn't the same length as the SNP
  count, the monotonic check treats all SNPs as one run (it can't tell chromosomes
  apart), which is the conservative choice — it may flag a legitimate cross-chromosome
  reset, but it never lets a real intra-chromosome decrease slip through.
- **Batch larger than the panel** is clamped before the footprint math, so it's never
  over-charged.
- **Plain-data by design.** Because `PaintRequest` and `SnpTable` are just numbers and
  vectors, the whole matrix of failures — unphased input, absent map, non-monotonic
  map, self-copy-on-superset, over-cap work, over-cap footprint — is exercised in
  `tests/unit/test_li_stephens_validate.cpp` with no files and no GPU.
