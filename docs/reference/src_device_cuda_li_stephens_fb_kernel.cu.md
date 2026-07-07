# `li_stephens_fb_kernel.cu` reference

## 1. Purpose

This file is the GPU heart of `steppe paint`: the Li-Stephens haplotype-copying
forward-backward, run in batch on the device. It is the Phase-1 engine that all
three paint faces are built on. Give it a panel of donor haplotypes and a set of
recipients, and for every recipient it works out the *copying posterior* — for
each SNP, how strongly each donor is the one this recipient was copying from at
that position.

The Li-Stephens model treats a recipient haplotype as a mosaic stitched together
from the donors: you copy along one donor for a while, then recombination flips
you onto another, and each SNP either matches the donor you're copying (likely)
or mutates away from it (rare). The forward-backward makes that intuition exact —
it hands back the posterior probability `gamma_l(k)` that donor `k` is the source
at SNP `l`, having accounted for the whole rest of the chromosome on both sides.

One kernel, three jobs. The same recurrence powers:

- **GATE mode** (Phase 1) — write the full `K × M` posterior out so it can be
  diffed against the CPU oracle. This is the parity gate; it is not how a real
  paint run is meant to be spent.
- **PAINT mode** (Phase 2) — fold the posterior straight into ChromoPainter's two
  coancestry summaries (chunkcounts + chunklengths), never storing the full
  posterior.
- **LOCALANC mode** (Phase 3) — collapse the posterior per SNP down onto a handful
  of ancestry labels, giving a per-position local-ancestry posterior.

Which job runs is chosen entirely by *which output pointer is non-null* — see
section 8.

One hard contract up front: **the input must be pre-phased haploid haplotypes.**
Steppe builds no phaser. The recipient and every donor row are single haplotypes;
missing alleles are allowed (they become uninformative emissions), but this
engine does not phase diploid genotypes for you.

---

## 2. One block per recipient, and why that shapes everything

The launch is `<<<n_recip, 256>>>`: **one CUDA block per recipient**, 256 threads
per block, donors spread across the threads with a grid-stride loop over `K`.

That choice is the whole design. The forward-backward is *sequential in `M`* — SNP
`l` can't start until SNP `l-1` is done — but it is *embarrassingly parallel across
donors* within a column. By pinning one recipient to one block, the entire
left-to-right scan lives inside a single block. The per-column barrier is just
`__syncthreads()`. There is no grid-wide synchronization, no per-column kernel
relaunch, and no global memory round-trip between columns. Recipients are fully
independent, so the grid scales them out trivially.

The block size (`kBlock = 256`) is a power of two on purpose: the shared-memory
tree reduction in `block_reduce_sum` halves the active range each step and needs
that. Threads with `tid >= K` are real threads that just contribute `0.0` to every
reduction and skip the per-donor stores — they still have to call every
`block_reduce_sum`, because that helper is a block-wide *collective* (it has
`__syncthreads()` inside), and a thread that skipped one would deadlock the rest.

---

## 3. The forward recurrence, rescaled every column

The forward pass builds `alpha_l(k)` — the probability of the recipient's alleles
up to SNP `l` *and* copying donor `k` at `l`. Two device helpers do it, and they
mirror the CPU oracle (`cpu_backend.cpp`) line for line so the GPU and CPU agree:

- **`forward_init`** handles column 0: `alpha_0(k) = pi[k] * e_0(k)`, then rescale
  the whole column by its sum.
- **`forward_step`** handles every column `l >= 1`. The Li-Stephens transition is
  "stay on the same donor with probability `1 - rho_l`, or recombine (probability
  `rho_l`) onto a fresh donor drawn from the prior `pi`." Naively that's a `K × K`
  transition, but it collapses to a rank-1 update: the recombination contribution
  is the same `pi[k] * prev_sum` for every `k`, where `prev_sum` is just the sum of
  the previous (already-normalized) alpha column. So each column is
  `alpha_l(k) = e_l(k) * ( (1 - rho_l) * alpha_{l-1}(k) + rho_l * pi[k] * prev_sum )`,
  which is `O(K)` per column, not `O(K^2)`.

The emission `e_l(k)` (`ls_emission`) is Watterson's: if either the recipient or
the donor allele at `l` is missing (byte `> 1`) the emission is an uninformative
`1.0`; otherwise a match scores `1 - mu[l]` and a mismatch scores `mu[l]`.

**Why rescale every column.** The forward-backward is a long product of
sub-one probabilities, so over thousands of SNPs `alpha` would silently underflow
to zero. Instead of log-space, this engine follows kalis's convention: normalize
each column by its own sum as it's produced. The posterior `gamma = alpha*beta`
normalized per column is invariant to those per-column scalings, so the answer is
unchanged — only the bookkeeping stays in range.

`rho[0]` is never used (there's no transition into the first column); the launch
wrapper documents it as ignored.

---

## 4. Precision: native FP64, not the emulated default

This kernel runs the whole recurrence in **native FP64**, and that is a
deliberate departure from steppe's usual precision policy. The house default is
emulated-FP64 (Ozaki), but that default is *matmul-only* — it buys its accuracy
inside GEMMs. **There is no GEMM here.** The forward-backward is a chain of
elementwise rank-1 updates and reductions, so the emulated path has nothing to
act on. Native FP64 is the right and only tool: the scan, every emission, every
transition, and every tree reduction are plain `double`.

The reductions (`block_reduce_sum`) are shared-memory tree reductions in `double`.
Reduction order is not load-bearing here because `gamma` is scale-invariant per
column — a slightly different summation order rescales a column by a factor that
divides straight back out of the posterior.

If you're writing or reading a precision note about this file: say native FP64 for
the scan and the reductions, and say *why* (no matmul to emulate). Do not describe
it as the emulated-FP64 default.

---

## 5. The checkpoint / recompute memory scheme

The naive forward-backward keeps the entire `K × M` alpha table resident so the
backward pass can read it. For a real panel that table is enormous. This kernel
**never** materializes it. Instead it always runs a checkpoint/recompute scheme
with stride `C = ceil(sqrt(M))` (giving `nck = ceil(M/C)` blocks — the
`sqrt(M)`-memory / `sqrt(M)`-recompute sweet spot).

- **Forward sweep** runs left to right on two ping-pong columns (`alphaA`/`alphaB`)
  and stores *only* the normalized alpha column at each checkpoint — `nck * K`
  doubles, not `K * M`.
- **Backward sweep** walks block by block, right to left. For each block it reloads
  that block's stored checkpoint and *recomputes* the `C`-column forward tile using
  the **same** `forward_step` the forward sweep used. Then it descends the block,
  computing `gamma_l` and stepping `beta` down.

The recompute is **bit-identical** to the original forward sweep, and that's the
subtle invariant that makes this safe. The complete Markov state carried from one
column to the next is just the normalized alpha column; the checkpoint *is* that
state. Recomputing from it replays the identical FP64 instruction stream — same
reductions, same operations, same rounding — so the recomputed alpha equals what
the forward sweep produced. That is exactly why `forward_step` is factored out as
one function called from both places: the two callers must run the same code.

The l=0 init is never recomputed (a block's checkpoint 0 already *is* the
normalized `alpha_0`); only steps `1..len-1` inside a tile replay `forward_step`.

---

## 6. The backward pass and the posterior

`beta_l(k)` is the mirror image of alpha — the probability of the recipient's
alleles *after* SNP `l`, given you're copying donor `k` at `l`. It's seeded
`beta_{M-1} = 1` before the rightmost block runs, then stepped right to left with
the same rank-1 collapse (a shared term `T = sum_k pi[k] * e_l(k) * beta_l(k)`),
rescaled every column just like alpha. The index bookkeeping matches the CPU
oracle: producing `beta_{l-1}` uses `rho[l]` and `e_l`.

The posterior at each column is `gamma_l(k) = alpha_l(k) * beta_l(k) / denom`,
where `denom` is the column's `sum_k alpha_l(k) * beta_l(k)`. **Each column sums to
exactly 1** (it's a normalized posterior over which donor was being copied) — that
sum-to-one is the invariant to lean on when sanity-checking output, and it holds in
every mode except the degenerate-column guard below.

There's a small ordering rule worth naming: within the block descent the kernel
reads `beta_cur` to form `gamma_l`, then a `__syncthreads()`, *then* steps beta —
because the beta step overwrites the ping-pong buffer the gamma read is still
using. And when `l == 0` it breaks before stepping: `beta_{-1}` doesn't exist.

---

## 7. The degenerate-column guard

Every normalization in this file — alpha init, each forward step, each beta step,
and the gamma denominator — divides by a column sum `s` with the identical guard:
`inv = (s > 0.0) ? 1.0/s : 0.0`. If a column's sum is zero (it can happen when a
run of missing data collides with an all-mismatch emission, or a degenerate prior),
the reciprocal would be a division by zero. The guard makes that column **all
zeros** instead of NaN/Inf. So the one exception to "columns sum to 1" is a
genuinely degenerate column, which sums to 0 — a defined, contained outcome that
never poisons the rest of the scan with a NaN. The paint switch term (section 8)
carries the same guard on its own denominator (`pi_k = 0` or `rho = 0` -> 0).

---

## 8. Three output modes, gated by output pointers

The kernel takes three candidate output pointers — `gamma_all`, `acc_cnt_all`,
`post_all` — and **exactly one is non-null** on any launch (the host orchestrators
guarantee it; an `assert` checks it). The branch is always on *the output pointer*,
never on a `!paint` shorthand, so a null-derived pointer can never be dereferenced
on a path that doesn't own it. On the paths that don't use `gamma`, the `gamma`
pointer is left genuinely `nullptr` (no `nullptr + offset` UB).

### GATE mode — `gamma_all != nullptr`

Stores the full `K × M` posterior, donor-major, exactly as Phase 1 defined it.
This is the parity artifact diffed against the CPU oracle. It is *not* the
steady-state design — a real paint run should never pay for a resident `K × M`
posterior. The note at the top of the file says so.

### PAINT mode — `acc_cnt_all != nullptr`

The ChromoPainter coancestry sink. It allocates **no** `K × M` posterior; instead,
as each `gamma_l(k)` is formed it folds it online into two per-recipient `N × K`
accumulators:

- **chunklengths** (`acc_len`): `+= gamma_l(k) * w[l]`, where `w[l]` is the SNP's
  genetic-length weight — expected copied length (in Morgans) from each donor.
- **chunkcounts** (`acc_cnt`): the expected number of copied *chunks* from each
  donor. This needs a "switch" term — the probability that a *new* copying segment
  from `k` started at `l` — which uses the previous normalized forward column
  `alpha_{l-1}(k)`: `sw = gamma_l(k) * rho_l * pi_k / ((1-rho_l)*alpha_{l-1}(k) + rho_l*pi_k)`.
  At `l == 0` there's no previous column, so the whole of `gamma_0` is the initial
  chunk (`sw = gamma_0(k)`).

Getting `alpha_{l-1}` inside the recomputed backward tile is the fiddly part: it's
tile column `j-1` when we're mid-tile, but at a block's *first* column (`j == 0`,
`b0 > 0`) column `l-1` lives in the *previous* tile, which isn't resident. That's
what the **companion checkpoints** are for: `checkpts_prev` stores the normalized
alpha at column `bi*C - 1` for each block, captured during the forward sweep, so the
switch term at a block head has its `alpha_{l-1}` on hand. No atomics are needed —
donor `k` is owned by exactly one thread within the block (`k = tid, tid+blockDim,
...`), so `acc_*[rid*K + k]` has a single writer.

### LOCALANC mode — `post_all != nullptr`

The local-ancestry sink. Also allocates no `K × M` posterior. Each donor carries an
ancestry-label index `donor_group[k]` in `[0, P)`, and for each SNP the kernel sums
`gamma_l` over the donors within each label:
`post[(rid*M + l)*P + g] = sum_{k: group(k)==g} gamma_l(k)`. That's `P` masked block
reductions per column (`P` is a small handful of labels). Every thread — including
`tid >= K` and threads whose donor is a different label, contributing `0.0` — must
call every `block_reduce_sum`, since it's a block collective. No switch term, no
genetic weight, no companion checkpoint (`checkpts_prev` is null on this path). When
a column is degenerate (`denom == 0`), `ginv == 0` and that column's posterior is
all zeros — the same guard as everywhere else; that column sums to 0, not 1.

---

## 9. The launch wrapper

`launch_ls_forward_backward` is the thin host entry the backend calls (declared in
the companion `.cuh`, which is private to `steppe_device` because it names
`cudaStream_t`). It early-returns on an empty problem (`K <= 0 || M <= 0 ||
n_recip <= 0`), launches `<<<n_recip, kBlock, 0, stream>>>`, and runs the standard
`STEPPE_CUDA_CHECK_KERNEL()` error check. All the scratch buffers (the ping-pong
alpha/beta columns, the recompute tile, the checkpoint arrays) are owned and sized
by the backend and passed in as device pointers; the trailing sink-selecting
pointers default to null so a plain gate-mode call stays short. See the `.cuh` for
the full buffer table and sizes.

---

## 10. Edge cases and invariants at a glance

- **Pre-phased haploid input only.** Recipient and donor rows are single
  haplotypes; steppe does no phasing.
- **Native FP64** for the scan and reductions (section 4) — never the emulated
  default.
- **Every posterior column sums to 1**, except a degenerate (all-zero) column
  under the guard (section 7).
- **Bit-identical forward recompute** underwrites the checkpoint scheme (section
  5): the same `forward_step` is the only code that advances a column, from either
  caller.
- **Exactly one output mode** per launch, selected by a non-null output pointer;
  the assert enforces it (section 8).
- **No atomics** in any sink — per-donor accumulators have a single writer thread;
  the localanc per-label reductions are block collectives, not scattered writes.
- **Empty problem is a clean no-op** — the wrapper returns before launching.
