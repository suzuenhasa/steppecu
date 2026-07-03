# `f2_block_kernel.cu` reference

## 1. Purpose

`src/device/cuda/f2_block_kernel.cu` is the GPU implementation of the core f2
statistic. An f2 value measures how far apart two populations are in allele
frequency, corrected for the noise of finite sampling. Computing every pairwise f2
between a set of populations, over a whole panel of SNPs, is the single most
expensive step in the pipeline.

The naive way to compute it — form the product of every population pair at every
SNP, giving a three-dimensional block of size (SNPs × populations × populations),
then sum along the SNP axis — is far too much memory and work. This file instead
casts the whole computation as **three ordinary matrix multiplications** framed by
**two custom kernels**, so the heavy lifting runs on the highly optimized GPU
matrix-multiply library and the three-dimensional block is never built.

This translation unit is private to the GPU backend. It owns exactly three public
entry points and nothing else:

1. `launch_f2_feeder` — the *feeder*, a single elementwise sweep over the decoded
   input tile that produces the three matrices the multiplications consume.
2. `run_f2_gemms` — the three library matrix multiplications, run under the chosen
   precision policy.
3. `launch_assemble_f2` — the *assemble* step, which reads the three multiplication
   outputs and combines them into the final f2 values.

The per-element math in the feeder and the assemble step is expressed through small
shared helper functions (`het_correction`, `assemble_f2_numerator`, `finalize_f2`)
that the CPU reference path also calls. Sharing those helpers is deliberate: it
guarantees the GPU and the CPU reference apply the exact same formula element by
element, so the two can never quietly disagree on the definition of f2.

---

## 2. The three-GEMM reformulation

The building blocks come out of the genotype decode as three matrices, one value
per (population `i`, SNP `s`) cell:

- `Q_raw` — the allele-frequency estimate for population `i` at SNP `s`.
- `V_raw` — a per-cell weight that is nonzero exactly when population `i` has usable
  (non-missing) data at SNP `s`.
- `N_raw` — the number of non-missing haploid samples behind that estimate, used to
  size the sampling correction.

From these the feeder builds:

- `Q` (masked) — `Q_raw` where the cell is valid, and `0` where it is not.
- `V` — `1` where valid, `0` where not (a plain validity indicator).
- `S` — a stacked matrix holding, per cell, both the squared frequency `Q²` and the
  per-cell heterozygosity correction.

The three multiplications then are:

| Product | Formula | What each output cell means |
|---|---|---|
| `G = Q · Qᵀ` | `G(i,j) = Σ_s Q(i,s)·Q(j,s)` | The cross term: sum over SNPs of the two populations' frequencies multiplied together. |
| `Vpair = V · Vᵀ` | `Vpair(i,j) = Σ_s V(i,s)·V(j,s)` | The count of SNPs where **both** `i` and `j` have usable data — the joint-valid count. |
| `R = S · Vᵀ` | see below | Two stacked results at once: the masked squared-frequency sums and the masked heterozygosity-correction sums. |

Because `Q` and `S` are zero-filled wherever a cell is invalid, and `V` marks
validity, every sum above is automatically restricted to the SNPs the two
populations actually share. This is what lets a single dense multiplication
reproduce the "use only SNPs present in both populations" rule without any
per-pair masking logic.

The assemble step then forms, for each pair `(i,j)`:

- the numerator, which combines `Σ Q(i)²`, `Σ Q(j)²`, the cross term `G(i,j)`, and
  the two heterozygosity corrections into the classic `Σp_i² + Σp_j² − 2Σp_i·p_j`
  difference, minus the sampling corrections;
- divided by `Vpair(i,j)`, the joint-valid count.

`Vpair` is kept around after this divide because it is also the natural weight for
the later block-jackknife uncertainty estimate.

---

## 3. The feeder kernel

The feeder (`f2_feeder_kernel`, launched by `launch_f2_feeder`) is one sweep over
the decoded input tile. Each GPU thread owns one (population `i`, SNP `s`) cell and,
from that one cell, writes all three multiplication inputs:

- `valid` is true when `V_raw` is nonzero at the cell.
- `Q_masked` gets the raw frequency where valid, else `0`. Zero-filling here is what
  makes the masked multiplication reproduce the pairwise-complete reference: at an
  invalid cell both `Q²` and the cross term vanish.
- `V_out` gets `1` where valid, else `0`.
- `S` gets `Q²` in its top block and the heterozygosity correction in its bottom
  block.

The heterozygosity correction is computed by the shared `het_correction` helper,
which takes the raw frequency, the per-SNP sample count `N`, and the validity flag,
and returns `0` at an invalid cell. The sample size is always the real per-SNP `N`,
never a hardcoded panel-wide number — an earlier prototype divided by a fixed
constant, and that shortcut is not carried here.

All of the feeder's arithmetic is native double precision. The feeder is limited by
memory bandwidth, not by arithmetic, so reduced-precision math would buy nothing
here.

### Memory-coalescing layout

The input and output matrices are column-major and shaped (populations × SNPs), so
the cell at (population `i`, SNP `s`) lives at offset `i + P·s`. That means the
addresses that are next to each other in memory differ by **population**, not by
SNP.

To make each group of 32 neighboring threads (a warp) read and write a contiguous
run of memory, the kernel maps the fast thread index onto the **population** axis
and the slow thread index onto the **SNP** axis, inside a square 16×16 block. So a
warp's consecutive lanes vary `i`, its loads and stores land at consecutive
addresses `i + P·s`, and the hardware coalesces them into wide bursts instead of a
scattered, population-strided pattern.

This is purely a thread-to-cell remapping. The block still covers the same 16×16
tile of SNPs and populations, the per-cell math is identical, and the addresses
written are identical — so it does not change any reported result. Crucially, it is
done *without* moving the SNP count off the grid's `x` axis (see section 9 for why
that axis assignment is mandatory).

---

## 4. The three GEMMs

`run_f2_gemms` issues the three matrix multiplications from section 2 through the
GPU matrix-multiply library, using the compute type chosen by the precision policy
(sections 6–8). It does **not** rebind the library handle's stream on each call: the
stream and the emulated-precision scratch workspace are bound once when the backend
is constructed. Rebinding the stream per call would silently reset that workspace
back to a default pool before every multiplication, which would break the
run-to-run reproducibility guarantee.

The three calls, in order:

| Output | Shape | Operation |
|---|---|---|
| `G` | P × P | `Q · Qᵀ` |
| `Vpair` | P × P | `V · Vᵀ` |
| `R` | 2P × P | `S · Vᵀ`, where `S` is the stacked (2P × SNPs) matrix; the top P rows come out as the squared-frequency sums and the bottom P rows as the heterozygosity-correction sums. |

### The SNP-count guard

The contraction length of these multiplications is the SNP count `M`. The library
call used here takes that length as a signed 32-bit integer. `M` is stored as a
wider `long` because this path uploads every SNP at once without tiling, so a panel
with more than about 2.1 billion SNPs would overflow the 32-bit field and either be
rejected or silently contract over the wrong number of SNPs.

The only caller checks `M ≤ INT_MAX` and throws a typed error *before* reaching this
function, and a debug-build assertion re-checks the same bound right at the point of
narrowing, so the invariant is pinned at both ends.

---

## 5. The assemble kernel

The assemble kernel (`assemble_f2_kernel`, launched by `launch_assemble_f2`) reads
the three multiplication outputs and produces the final f2 matrix. Each thread owns
one output cell `(i,j)`. It reads:

- `G(i,j)` — the cross term;
- `Vpair(i,j)` — the joint-valid count (the divisor);
- from the stacked `R`: the squared-frequency sum for `i` and for `j`, and the
  heterozygosity-correction sum for `i` and for `j`.

Because `R` is stacked with its two halves interleaved by row, the "for `i`" and
"for `j`" values are read from transposed positions — the `i` values from column
`j`, the `j` values from column `i`. Index arithmetic uses wide (`size_t`) offsets,
which is mandatory once the population count exceeds roughly 32,000 and costs
nothing below that.

It then calls the shared `assemble_f2_numerator` and `finalize_f2` helpers — the
same ones the CPU reference uses — to form the numerator and divide by `Vpair`.

**This kernel is always native double precision, in every precision mode.** The
numerator is a subtraction of large, nearly equal quantities (`Σp_i² + Σp_j² −
2Σp_i·p_j`), which is a catastrophic-cancellation step: the leading digits cancel
and the answer lives in the low-order digits. Emulated precision cannot recover
those digits, so the precision policy deliberately governs only the three big
multiplications and never this final combine.

### Memory access here is a small, accepted cost

The two transposed reads (`i` values pulled from column `j`) are strided rather than
contiguous, so they do not coalesce. This is left as-is on purpose: `R` is the small
(2P × P) buffer, this kernel sits off the multiplication-dominated critical path,
and it must stay native double precision for the cancellation. Reworking the access
pattern with shared-memory tiling would need two off-diagonal tiles for a bounded,
off-critical-path gain, so it is not done.

---

## 6. The precision policy

The three multiplications are the expensive part, and they run under a measured
precision policy. There are three modes:

- **Emulated double precision (the default).** Double precision is emulated from
  fixed-size integer slices. On real data this was measured at 7 to 17 times faster
  than native double precision, at essentially the same accuracy, and the lead grows
  as the population count rises. The number of mantissa bits kept is set from the
  precision configuration; the default (40 bits) is about as accurate as native
  double precision, while 32 bits is faster and slightly less accurate.
- **Native double precision.** This is both the gold-standard reference the emulated
  mode is validated against, and the fallback.
- **TF32 tensor-core arithmetic.** A lower-precision screening mode. It is *not*
  wired into this double-precision-storage multiplication path — here it is treated
  as native double precision, so a screening request never silently degrades the
  numbers this path reports. A dedicated TF32 path is a later addition.

### Fixed, not dynamic, mantissa

The mantissa bit count is always pinned to a **fixed** value. The matrix-multiply
library also offers a *dynamic* mode that picks the bit count automatically per
operation, and that mode is the default when nothing pins it. On real data's wide
range of magnitudes, dynamic control overshoots to roughly 60 bits and collapses
back to the speed of native double precision — all cost, no benefit. This file
therefore always engages fixed control explicitly and never lets the dynamic mode be
selected. (An earlier prototype comment concluded the multiplications had to stay
native double precision; that predates the real-data measurement that made emulated
precision the default, and it is not carried here.)

### The `STEPPE_HAVE_EMU_TUNING` build switch

Pinning the mantissa to a fixed value requires a tuning API that is only present on a
recent enough toolkit. That API is gated behind the `STEPPE_HAVE_EMU_TUNING` compile
flag, which **defaults off** so the file still compiles on a stock toolkit. The two
load-bearing pieces — asking for an emulated compute type and an emulated math mode
— are unconditional; only the fixed-slice pinning needs the flag. The real GPU build
turns it on with `-DSTEPPE_HAVE_EMU_TUNING=1`.

---

## 7. Deciding whether emulation is honorable

`emulation_honorable` is the single predicate that decides whether a request for
emulated double precision can actually be delivered as the fast, fixed-slice path.

The rule is: emulated precision is honorable **only** when the request asks for it
*and* the build carries the fixed-slice tuning API. Without that API, the fixed-slice
pin cannot be applied, so the library would fall back to its own dynamic-mantissa
default — the exact rejected mode from section 6 that costs the same as native
double precision. Rather than run that trap silently while still labeling the result
"emulated," the predicate reports the request as **not honorable**, and the caller
downgrades it to native double precision.

Both halves of the policy — the math-mode engagement and the compute-type selection
(section 8) — consult this one predicate. That is what keeps them from ever
disagreeing: a request downgraded for one is downgraded for the other in lockstep.
It is the single source of truth for the emulated-precision honorability decision.

### The one-time downgrade notice

When emulated precision is requested but the build cannot honor it, the downgrade to
native double precision is announced through the library's warning channel **at most
once per process**, tagged as a capability downgrade so it is observable without
spamming every call. A thread-safe one-shot guard enforces the once-per-process
limit, which matters because more than one host thread may reach this path. The
notice makes clear that the precision actually delivered is native double precision,
not emulated. The helper that emits it is compiled only in the no-tuning build — the
only build where the downgrade can happen — so the normal (tuning-on) build carries
no unused code.

---

## 8. Mapping precision to cuBLAS: compute type and math mode

Two small functions turn the typed precision setting into the concrete library
calls, and both route through `emulation_honorable` so they can never diverge.

`f2_compute_type` picks the compute type for the multiplications:

| Request | Compute type used |
|---|---|
| Emulated, honorable | the fixed-point emulated double-precision type |
| Emulated, not honorable | native double precision (downgraded — tuning absent) |
| Native double precision | native double precision |
| TF32 | native double precision (screening mode is not wired into this path) |

`engage_f2_precision` configures the library handle to match:

- For an **honorable** emulated request, it sets the emulated math mode, forces
  emulation on even for small multiplications, pins fixed-slice mantissa control, and
  sets the mantissa bit count from the configuration. Setting fixed control is the
  whole point — dynamic control would lose the speed win.
- For **native double precision**, and for an emulated request the build cannot
  honor, it sets strict native double-precision math (no tensor-core shortcuts) and,
  on the no-tuning build, emits the one-time downgrade notice. It deliberately does
  **not** set the emulated math mode in the unhonorable case: doing so would engage
  the rejected dynamic-default emulation while the compute type stayed native, making
  the two halves incoherent.

Both functions are shared by the single-block path here and the batched/grouped path
elsewhere, so the compute-type and math-mode mapping lives in exactly one place.

---

## 9. Launch wrappers and grid geometry

Host code never issues raw kernel launches; the three launch wrappers do. They pull
their block and grid arithmetic from a single launch-configuration helper rather than
open-coding the ceiling-division math, and every kernel uses the same 16×16 (256
thread) square block.

The grid orientation for the feeder is a hard constraint. The grid's `x` axis is the
only one that can span more than about 2.1 billion; the other axes are capped at
65,535. The SNP count can run well past that cap — it already exceeds 65,535 at
roughly a million SNPs — so the **SNP count must ride the `x` axis**. The population
count, which is at most a few thousand, safely rides the `y` axis, and an assertion
in the launch helper enforces that it fits. The in-block thread-to-cell transpose
described in section 3 achieves coalesced access *without* moving the SNP count off
the `x` axis, so the coalescing fix and the axis constraint coexist.

The assemble launch is simpler: its output is the square (P × P) f2 matrix, so the
population count rides both grid axes, and the same "P fits the capped axis"
assertion applies.
