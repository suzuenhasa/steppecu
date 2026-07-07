# `cuda_backend_li_stephens.cu` reference

## 1. Purpose

`src/device/cuda/cuda_backend_li_stephens.cu` is the GPU-backend translation unit
that holds three CUDA member functions of `CudaBackend`, all built on the one
Li-Stephens haplotype-copying forward-backward kernel that lives next door in
`li_stephens_fb_kernel.cu`:

1. **`ls_forward_backward`** — the Phase-1 parity gate. Runs the forward-backward
   for a single recipient and returns the full copying posterior `gamma` (`K*M`,
   donor-major). This is the byte-for-byte checkable path against the CpuBackend
   oracle.
2. **`ls_paint_coancestry`** — the Phase-2 `steppe paint` coancestry face. Runs the
   same forward-backward, batched over `N` recipients, but never materializes the
   `K*M` posterior — it folds each column of `gamma` online into two small
   per-donor summaries (chunkcounts and chunklengths).
3. **`ls_localanc`** — the Phase-3 local-ancestry face. Again the same
   forward-backward, batched over `N` recipients, folding `gamma` per-SNP into a
   per-label ancestry posterior — again without ever holding the `K*M` table.

Each of these is a thin **host orchestrator**: it validates the shape, sizes and
allocates the device buffers, uploads the inputs, launches
`launch_ls_forward_backward` in the right mode, copies the (small) result back, and
returns. All of the actual math — the recurrence, the rescaling, the reductions —
happens inside the kernel. This file is the plumbing that feeds it and drains it.

The heavy design story (the FB recurrence, checkpoint/recompute, the three output
modes) is documented with the kernel itself in
`docs/reference/src_device_cuda_li_stephens_fb_kernel.cu.md`. This doc covers the
host side: what gets allocated, why the buffer shapes differ between the three
faces, and the contracts each entry point honors.

---

## 2. What Li-Stephens copying is (the one-paragraph version)

The Li-Stephens model treats a recipient haplotype as an imperfect mosaic copied
from a panel of `K` donor haplotypes. As you walk left-to-right along the `M` SNPs,
a hidden "which donor am I copying right now" state drifts: with recombination
probability `rho[l]` it jumps (re-drawn from the copying prior `pi`), otherwise it
stays put, and at each site it emits the recipient allele with a small mutation
rate `mu[l]` (a mismatch costs `mu`, a match costs `1-mu`). The forward-backward
computes, for every SNP `l` and every donor `k`, the posterior probability
`gamma_l(k)` that donor `k` was the copied source at that site. That posterior is
what all three faces here consume.

**The input must be pre-phased haploid data.** steppe ships no phaser; the recipient
and donor byte arrays are already-resolved haplotypes. Alleles are `0`/`1`; any byte
`> 1` is treated as missing and makes the emission uninformative (`1.0`).

---

## 3. The checkpoint stride, computed identically on all three paths

Every one of the three functions opens with the same three lines:

```
C   = ceil(sqrt(M))     // clamped to >= 1
nck = ceil(M / C)       // number of checkpoint blocks
```

`C` is the checkpoint stride and `nck` the number of checkpoint columns the forward
sweep will store. This is the memory trick that makes the whole engine tractable:
the forward pass keeps only `nck*K` doubles (one normalized alpha column every `C`
SNPs) instead of the full `K*M` alpha table, and the backward pass recomputes each
`C`-column tile on the fly from the nearest checkpoint. `sqrt(M)` is the stride that
balances the two costs (storage `~sqrt(M)*K`, recompute work `~sqrt(M)` per column).

The scheme is **always on** — there is no "small model, keep the table resident"
shortcut. Even the tiny `M=256` golden runs the recompute path, so the path that
ships is the path that is tested. All three functions compute `C`/`nck` the same
way so the gate, the paint sink, and the localanc sink share bit-identical
checkpoint boundaries.

---

## 4. `ls_forward_backward` — the Phase-1 gate

`ls_forward_backward(recipient, donors, pi, rho, mu, K, M, precision)` runs one
recipient (`n_recip = 1`) and returns an `LsPosterior` carrying the full `K*M`
`gamma`, donor-major (`gamma[k*M + l]`), with each SNP column normalized to sum to
1.

The Phase-1 override deliberately drives exactly one recipient per call — it matches
the CpuBackend signature so the two can be compared column-for-column. The kernel
itself is already batch-ready over its block axis (`blockIdx.x` = recipient); this
entry point just launches a one-block grid.

What it allocates (all device-side, backend-owned):

- `dRecip` (`M`), `dDonors` (`K*M`) — the uploaded haplotypes.
- `dPi` (`K`), `dRho`/`dMu` (`M`) — the model parameters.
- `dGamma` (`K*M`) — the output posterior. **This is the one path that allocates the
  full posterior**; the paint and localanc faces never do.
- `dCheck` (`nck*K`) — the forward checkpoints.
- `dAlphaA`/`dAlphaB` (`K` each) — the forward ping-pong columns.
- `dAlphaBlk` (`C*K`) — the recomputed backward tile.
- `dBetaA`/`dBetaB` (`K` each) — the backward ping-pong columns.

The five inputs upload asynchronously on the backend stream, the kernel launches in
GATE mode (all the trailing paint/localanc pointers left at their defaults), the
`K*M` gamma copies back, and a single `cudaStreamSynchronize` closes the round-trip
before the result is returned.

---

## 5. `ls_paint_coancestry` — the Phase-2 paint face

`ls_paint_coancestry(recipients, donors, pi, rho, mu, w, K, M, N, precision)` is the
`steppe paint` coancestry sink. It runs the same forward-backward but batched over
`N` recipients (one CUDA block each, all sharing the single donor panel), and
returns an `LsCoancestry` with two `N*K` recipient-major summaries:

- **`chunklengths[r*K + k] = sum_l gamma_l(k) * w_l`** — the expected genetic length
  (Morgans) that recipient `r` copies from donor `k`, weighted by the per-SNP
  genetic-length weight `w`.
- **`chunkcounts[r*K + k] = gamma_0(k) + sum_{l>=1} switch_l(k)`** — the expected
  number of copied chunks attributed to donor `k`. The `switch_l` term is the
  posterior probability that a fresh recombination landed on donor `k` at site `l`;
  the `l=0` term is the initial chunk.

The load-bearing point: **the `K*M` posterior is never allocated.** The kernel folds
each `gamma_l(k)` straight into these two `N*K` accumulators as it descends the
backward pass, and only the accumulators (`N*K` doubles apiece — tiny) come back to
host. For a real run with thousands of donors and hundreds of thousands of SNPs, the
full posterior would be enormous; this is the design that lets paint scale.

Extra buffers this face needs beyond the gate set:

- `dW` (`M`) — the genetic-length weight, uploaded and passed live.
- `dAccCnt`/`dAccLen` (`N*K` each) — the two output accumulators.
- `dCheckPrev` (`N * nck*K`) — the **companion checkpoints**: the normalized alpha
  at column `bi*C - 1`, the column just *before* each checkpoint. The chunkcount
  switch term at a tile's first column needs the previous column's forward vector,
  which isn't inside that tile — the companion checkpoint supplies it. (This is the
  one scratch buffer localanc does not need, because localanc has no switch term.)
- The per-recipient scratch (`dCheck`, `dAlpha*`, `dBeta*`) is all sized `* N`.

The kernel launches with `d_gamma = nullptr` and the paint pointers non-null, which
is how it selects the coancestry sink (section 7).

---

## 6. `ls_localanc` — the Phase-3 local-ancestry face

`ls_localanc(recipients, donors, pi, rho, mu, donor_group, K, M, N, P, precision)`
is the `steppe paint --face localanc` output. Same batched forward-backward, but
instead of per-donor summaries it produces a per-SNP posterior over `P` ancestry
**labels**. Each donor carries a label in `donor_group` (an index in `[0, P)`), and
the sink sums `gamma` over the donors sharing each label:

```
post[(r*M + l)*P + g] = sum_{k : group(k) == g} gamma_l(k)
```

The returned `LsLocalAncestry.post` is `N*M*P`, laid out with the label index
fastest. On an informative column each SNP's `P` values sum to 1 (the labels
partition the donors and `gamma` is column-normalized); a degenerate column whose FB
denominator underflowed to 0 has its whole column zeroed instead (section 8).

Buffers, relative to the gate/paint sets:

- `dGroup` (`K`) — the per-donor ancestry-label indices.
- `dPost` (`N*M*P`) — the per-SNP posterior output. This is the only output; the
  `K*M` gamma is again never allocated.
- The same `dCheck` / `dAlpha*` / `dBeta*` scratch as coancestry, **minus** the
  companion-checkpoint buffer — localanc has no switch term, so `d_checkpts_prev` is
  passed null.

The launch call passes `d_gamma`, `d_acc_cnt`, `d_acc_len`, and `d_checkpts_prev`
all null, and hands over the trailing three localanc parameters (`dGroup`, `dPost`,
`P`) as the live output — which is how the kernel selects the localanc sink.

---

## 7. One kernel, three modes, selected by null pointers

All three functions call the exact same `launch_ls_forward_backward`. The kernel
picks its behavior from **which output pointer is non-null**, and the host
orchestrators guarantee that exactly one of `{d_gamma, d_acc_cnt, d_post}` is live:

| Function | Live output | Nulled |
|---|---|---|
| `ls_forward_backward` | `d_gamma` | paint + localanc pointers |
| `ls_paint_coancestry` | `d_acc_cnt` / `d_acc_len` (+ `d_w`, `d_checkpts_prev`) | `d_gamma`, localanc pointers |
| `ls_localanc` | `d_post` (+ `d_donor_group`, `P`) | `d_gamma`, `d_acc_cnt`, `d_checkpts_prev` |

Gating on the output pointers (never on a `!paint` flag) is what keeps the gate path
byte-identical across the three faces: the paint and localanc sinks fold the same
`gamma` values the gate would have stored, they just fold instead of store. So the
Phase-1 parity check over `gamma` also validates the numbers Phases 2 and 3 build
on. The kernel `assert`s the exactly-one-output invariant.

---

## 8. Precision and the degenerate-column guard

**The recurrence runs in native FP64 — not the emulated-FP64 default.** The
`precision` argument is accepted (for interface uniformity) and immediately
`(void)`-cast on all three paths. This is deliberate and correct: steppe's
emulated-FP64 (Ozaki) default is a *matmul* optimization, and there is no GEMM
anywhere in the Li-Stephens engine. The forward-backward is a long sequential
product of sub-one probabilities that would underflow without the per-column
rescaling the kernel does, and the sinks are plain reductions. Both belong to the
native-FP64 carve-out, so emulation would be the wrong tool, not merely a slower
one. The `(void)precision` lines carry a comment saying exactly this.

The **degenerate-column guard** runs throughout the kernel: whenever a normalizing
sum underflows to `0` (an all-missing column, or one whose parameters zero it out),
the reciprocal is set to `0` rather than dividing by zero. The visible consequence
is that such a column sums to `0` instead of `1` — the posterior is honestly zeroed
rather than filled with NaNs. The paint and localanc sinks inherit the same guard
(a zeroed `gamma` folds nothing into the accumulators).

---

## 9. Contracts and invariants

- **Pre-phased haploid input.** Recipient and donor bytes are resolved haplotypes;
  `0`/`1` are alleles, any byte `> 1` is missing (uninformative emission).
- **Column normalization.** Each SNP column of `gamma` sums to 1 on an informative
  column (0 on a degenerate one). The localanc posterior inherits this per SNP over
  its `P` labels.
- **`rho[0]` is unused.** There is no transition into the first column; the forward
  init uses only `pi` and the emission.
- **Native FP64 everywhere** (section 8).
- **The `K*M` posterior is resident only on the gate path.** Both production faces
  (paint, localanc) fold `gamma` online and return only small summaries. The
  resident `dGamma` exists purely for the Phase-1 parity gate — it is not the
  steady-state design.
- **Exactly one live output.** The host guarantees, the kernel asserts (section 7).
- **Layouts.** `donors` is donor-major (`K*M`) and shared by all recipients;
  `recipients` and `pi` are recipient-major; `gamma` is donor-major (`gamma[k*M+l]`);
  `chunkcounts`/`chunklengths` are recipient-major `[r*K+k]`; `post` is
  `[(r*M+l)*P+g]` with the label fastest.

---

## 10. Edge cases

- **Empty shape.** Any of `K <= 0` or `M <= 0` (plus `N <= 0` for the batched faces,
  `P <= 0` for localanc) returns an `Ok` result with empty vectors — no allocation,
  no launch. Zero work is a clean no-op, not an error.
- **`C` clamped to `>= 1`.** `ceil(sqrt(M))` can never be `0` for `M >= 1`, but the
  clamp is there defensively so the checkpoint arithmetic and the `l % C` stride are
  always well-formed.
- **Degenerate columns** are zeroed, not NaN'd (section 8), so a recipient full of
  missing sites yields an honest all-zero posterior rather than poisoning the run.
- **Single synchronize per call.** Each function issues its uploads, launch, and
  download on the backend stream and then does exactly one
  `cudaStreamSynchronize` — the results are only read after that barrier.
