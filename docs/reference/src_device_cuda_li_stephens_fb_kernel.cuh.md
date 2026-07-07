# `li_stephens_fb_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/li_stephens_fb_kernel.cuh` declares one function —
`launch_ls_forward_backward` — the single seam between the CUDA backend and the
GPU Li-Stephens copying forward-backward kernel that powers `steppe paint`. It is
a launch-wrapper header: it names `cudaStream_t`, so it stays private to the
`steppe_device` target, and it carries the full buffer contract in its comments.
The kernel body and the `<<<...>>>` launch live in the paired
`li_stephens_fb_kernel.cu`; this header is what the backend `#include`s to reach
them.

The kernel is a batched Li-Stephens haplotype-copying forward-backward. Given a
panel of donor haplotypes and one or more recipient haplotypes, it computes, for
each recipient, the per-site copying posterior — the probability that recipient
site `l` was copied from donor `k`. That posterior is either handed back in full
(the parity gate) or folded on the fly into one of two summaries: the ChromoPainter
coancestry counts (`steppe paint`) or a per-SNP ancestry posterior (`steppe
localanc`).

The one hard input requirement, baked into the whole design: **the input must be
pre-phased haploid data.** Steppe ships no phaser. Recipient and donor alleles
arrive as one byte per site, `0` or `1`, with any byte `> 1` read as missing.

---

## 2. One CUDA block per recipient

The launch is `ls_fb_kernel<<<n_recip, kBlock, ...>>>` — the grid has one block
per recipient (`blockIdx.x = rid`) and `kBlock = 256` threads. That shape is the
core design decision, and it falls straight out of the math: the forward-backward
is a strictly sequential scan over the `M` sites (each column depends on the one
before it), but it is *embarrassingly parallel across recipients*, and within a
single recipient the work at each column is a parallel reduction over the `K`
donors.

So the sequential dimension (`M`) is confined to a block, where a plain
`__syncthreads()` is the per-column barrier — no grid-wide synchronization, no
per-column kernel relaunch. The donor dimension (`K`) is spread over the block's
threads with a grid-stride loop (`k = tid; k < K; k += blockDim.x`), and every
per-column sum is a shared-memory tree reduction (`block_reduce_sum`). `kBlock` is
a power of two because the tree reduction requires it. Threads with `tid >= K`
still participate in every reduction contributing `0.0` — the reduction is a
barrier-collective, so *every* thread must call it every time or the block
deadlocks.

All buffers are device pointers owned by the backend. The kernel allocates
nothing; the host sizes and hands it every scratch region it needs.

---

## 3. The forward-backward recurrence

The kernel implements the standard Li-Stephens copying HMM, matching the
`CpuBackend::ls_forward_backward` reference oracle instruction for instruction
(the CPU version is the exact, per-column-rescaled scalar oracle the GPU is gated
against). The three pieces:

**Emission** `e_l(k)` (Watterson): if either the recipient allele or donor `k`'s
allele at site `l` is missing (byte `> 1`), the site is uninformative and the
emission is `1.0`. Otherwise a match emits `1 - mu[l]` and a mismatch emits
`mu[l]`, where `mu[l]` is the per-site copying-error rate.

**Forward** `alpha_l(k)`, left to right. Column 0 is `alpha_0(k) = pi[k] *
e_0(k)`, where `pi[k]` is the recipient's copying prior over donors. Each later
column collapses the transition into a rank-1 update:

```
alpha_l(k) = e_l(k) * [ (1 - rho[l]) * alpha_{l-1}(k)  +  rho[l] * pi[k] * sum_j alpha_{l-1}(j) ]
```

`rho[l]` is the recombination (switch) probability at the gap before site `l`
(`rho[0]` is unused). The `(1 - rho)` term is "stay on the same donor"; the
`rho * pi` term is "switch, drawing a new donor from the prior." Because
`alpha_{l-1}` is kept normalized, `sum_j alpha_{l-1}(j)` is the whole switch mass
— that single reduction is what makes each column `O(K)` instead of `O(K²)`.

**Backward** `beta_l(k)`, right to left, seeded `beta_{M-1} = 1`, using the
mirror rank-1 term `T = sum_k pi[k] * e_l(k) * beta_l(k)`.

**Posterior** `gamma_l(k) = alpha_l(k) * beta_l(k) / denom`, normalized so each
site column sums to 1 — this is the copying posterior, the probability recipient
site `l` copied from donor `k`.

Every column of `alpha`, `beta`, and `gamma` is rescaled by its own column sum.
That per-column rescaling is not optional cosmetics: the forward-backward is a
long product of sub-one probabilities that would underflow to zero over thousands
of sites, so each column is renormalized to keep it in range. `gamma` is
per-column scale-invariant, which is also why the reduction *order* is not
load-bearing — the ratio comes out the same however the block sums it.

---

## 4. Checkpoint / recompute — why the K×M table is never resident

A naive forward-backward stores the full `alpha` table (`K * M` doubles per
recipient) so the backward pass can read it back. For a real donor panel that is
enormous and would cap how many recipients fit on the GPU at once. This kernel
never materializes it.

Instead it uses an always-on checkpoint/recompute scheme with stride
`C = ceil(sqrt(M))` and `nck = ceil(M / C)` blocks:

- The **forward sweep** runs the full scan but stores only the normalized `alpha`
  column at each checkpoint (`nck * K` doubles), ping-ponging two working columns
  (`alphaA`/`alphaB`) for everything in between.
- The **backward sweep** walks the checkpoint blocks right to left. For each
  block it reloads the stored checkpoint and *recomputes* the `C`-column forward
  tile by replaying the exact same single-column `forward_step`, then descends
  through the block computing `gamma` while stepping `beta`.

The replay is bit-identical to the original forward sweep, and that is the crux
of the correctness argument. The complete Markov state carried column to column is
the normalized `alpha` column — nothing else. Recomputing forward from a stored
normalized checkpoint replays the identical FP64 operations, so the recomputed
tile equals the sweep's original values to the last bit. `forward_init` and
`forward_step` are factored out precisely so the forward sweep and the backward
recompute call *the same instruction stream*.

The memory bill drops from `O(K * M)` to `O(nck * K + C * K)` per recipient — the
`sqrt(M)` stride balances the two terms.

---

## 5. Three output modes, selected by null pointers

The kernel has one body and three output modes. Exactly one output pointer is
non-null on any launch; the host orchestrators guarantee it and the kernel
asserts it. The branches gate on the *output pointer being non-null*, never on a
negated flag, so a null-derived pointer is never formed or dereferenced on a path
that doesn't own it.

**Gate mode** (`d_gamma` non-null). Stores the full `K * M` copying posterior,
donor-major, each column normalized — byte-identical to the Phase-1 posterior.
This is the parity path validated against the CPU oracle. It is deliberately *not*
the steady-state design: materializing the whole posterior is only for the gate.
The two summary faces below fold `gamma` away as it is produced.

**Paint mode** (`d_acc_cnt` non-null). The ChromoPainter coancestry sink. It
allocates no `K * M` posterior; it folds each `gamma_l(k)` online into two
per-recipient `N * K` accumulators — `d_acc_cnt` (expected number of copied chunks
from each donor) and `d_acc_len` (expected copied genetic length in Morgans,
weighted by the per-site genetic-length weight `d_w`). The chunk-count "switch"
term needs the previous normalized forward column `alpha_{l-1}`: inside a
recomputed tile that is the tile column `j-1`; at a block's leading column it is
the companion checkpoint `d_checkpts_prev` (the normalized `alpha` at column
`bi*C - 1`, stashed during the forward sweep alongside the ordinary checkpoint).
Each donor `k` is written by exactly one thread within the block, so the
accumulators need no atomics.

**Localanc mode** (`d_post` non-null). The per-SNP ancestry sink. Each donor
carries an ancestry-label index `d_donor_group[k]` in `[0, P)`. For each site the
kernel does `P` masked block reductions of `gamma_l` over the donors sharing each
label, writing `d_post[(rid*M + l)*P + g]` — the per-site posterior mass on
ancestry `g`. There is no switch term, no genetic weight, and no companion
checkpoint on this path. As in paint mode, every thread (including `tid >= K` and
label-mismatched threads, contributing `0`) must call every masked reduction, or
the collective barrier deadlocks.

---

## 6. Contracts and invariants

- **Pre-phased haploid input.** Recipient and donor alleles are one byte per site,
  `0`/`1`, donor panel donor-major (`K` rows of `M`), recipient recipient-major.
  Any byte `> 1` is missing and yields a `1.0` (uninformative) emission for that
  site. There is no phasing step anywhere in steppe.
- **Native FP64, not emulated.** The entire recurrence and every reduction run in
  native double precision. This is a deliberate departure from steppe's
  matmul-heavy emulated-FP64 (Ozaki) default — there is *no* GEMM here to emulate.
  The forward-backward is a long product of sub-one probabilities whose per-column
  rescaling and cancellation-sensitive sums want native FP64. Any precision doc
  should state this plainly: emulated-FP64 is for matmul; this kernel is native.
- **Columns sum to 1.** Every `alpha`, `beta`, and `gamma` column is normalized by
  its own sum. `gamma`'s per-column sum-to-1 is the posterior contract downstream
  code relies on; it is what makes reduction order immaterial.
- **Exactly one live output.** `{d_gamma, d_acc_cnt, d_post}` — precisely one is
  non-null per launch (host-guaranteed, kernel-asserted). Pass `nullptr` for the
  other two.
- **Bit-identical recompute.** The backward recompute replays the forward step
  from a normalized checkpoint and reproduces the original tile exactly, which is
  what lets the `O(K*M)` table stay off the device without any numeric drift.
- The stride/block sizing (`C = ceil(sqrt(M))`, `nck = ceil(M/C)`) and all scratch
  buffers are the host's responsibility; the kernel trusts the sizes it is given.

---

## 7. Edge cases

- **Degenerate columns.** If any column sum is `0` (or a `gamma` denominator is
  `0`) — an all-missing site, or a prior/recombination combination that zeros a
  column — the reciprocal is clamped to `0.0` rather than dividing by zero. That
  column then sums to `0`, not `1`: an honest "no information here" signal instead
  of a NaN. The same guard fires identically in gate, paint, and localanc modes
  (in localanc, a degenerate column makes the whole `post` row zero). This mirrors
  the CPU oracle's guard site for site.
- **Paint switch-term guard.** The chunk-count switch term divides by
  `(1 - rho[l]) * alpha_{l-1}(k) + rho[l] * pi[k]`; when that denominator is `0`
  (e.g. `pi[k] = 0` under a self-excluded prior, or `rho = 0`) the contribution is
  clamped to `0`. The `l = 0` initial column contributes its whole `gamma` as the
  first chunk, with no switch term.
- **Empty problem.** `launch_ls_forward_backward` returns immediately, launching
  nothing, if `K <= 0`, `M <= 0`, or `n_recip <= 0`.
- **`rho[0]` is unused.** The first site has no preceding gap, so `rho[0]` is
  ignored; the forward scan starts recombination at column 1.
