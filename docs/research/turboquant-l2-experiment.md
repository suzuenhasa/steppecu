# TurboQuant + L2-residency for steppe: honest assessment & experiment design

Status: research note (read-only on code). Author lens: lead-researcher synthesis of three
sub-agent lenses (TurboQuant identity, steppe sacred-vs-slack map, L2 fit math).
Date: 2026-06.
Verdict in one line: **a lossy quantization can NEVER touch the f2 values (§12), but it has
one honorable home — an approximate, L2-resident pre-screen of the not-yet-built S8 rotation
model search that prunes candidates, then exact-recomputes survivors in EmulatedFp64/Fp64.
This is a MEDIUM-priority, well-scoped experiment that is gated behind S8 existing at all,
and it ranks BELOW M5 streaming and the multi-GPU rotation fit on the critical path.**

---

## 1. Honest framing — what TurboQuant actually is

CONFIDENCE: **HIGH** on identity and method; MODERATE on the exact per-coordinate quantizer
internals (abstract-level description verified verbatim; the PDF body was not text-extractable).

TurboQuant is a **2025 Google Research paper + algorithm** — "TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate" (Zandieh, Daliri, Hadian, Mirrokni;
arXiv:2504.19874; accepted ICLR 2026). It is a **vector-quantization (VQ) method**, not a
library or a hardware feature, and not anything NVIDIA ships. Method, verified from the abstract:

1. **Random rotation** of each input vector (JL-style orthogonal transform) → spreads outliers,
   forces a predictable, concentrated **Beta distribution** on coordinates.
2. High-dim rotated coordinates are near-i.i.d., so apply a **simple optimal per-coordinate
   scalar quantizer** — cheap, no codebook training, **data-oblivious / online** (no fitting pass).
3. Two variants: **TurboQuant-MSE** (MSE-optimal scalar quantizer) and **TurboQuant-prod**
   (adds a 1-bit Quantized-JL transform on the residual to yield an **unbiased inner-product**
   estimator — the MSE variant alone biases inner products).
4. Proven near-optimal: within a small constant (≈2.7×) of the information-theoretic distortion
   lower bound across bit-widths and dimensions.

Stated wins (primary target = **LLM KV-cache** quantization): "absolute quality neutrality at
3.5 bits/channel, marginal degradation at 2.5 bits/channel"; Google's blog claims ≥6× KV memory
reduction and "up to 8× over 32-bit unquantized keys on H100." Secondary target: nearest-neighbor
/ MIPS vector search, beating product quantization on recall with ~zero indexing time.

**Crucial honesty caveat — the L2 framing is NOT TurboQuant's claim.** Neither the paper nor
the blog mentions L2 cache or on-chip residency. TurboQuant's stated win is *memory footprint*
(fit a bigger KV cache) and *bandwidth* (move ~3.5 bits instead of 16/32). The "keep it in L2"
idea is **our application of the general methodology** (quantize-to-shrink so the working set is
cache-resident), not a property of TurboQuant. For steppe, what we are really evaluating is:

> **the GENERAL methodology = "quantize a working set so a re-read-heavy approximate pass runs
> L2-bandwidth-bound instead of VRAM-bound," with TurboQuant as the specific quantizer if and
> only if the screening metric is inner-product-like (then TurboQuant-prod's unbiased estimator
> is the right tool; the random rotation also tames the dynamic range that wrecked steppe's
> rejected dynamic-mantissa Ozaki path).**

---

## 2. The precision verdict — sacred vs. slack (the §12 crux)

The §12 PARITY LAW is a `memcmp` bit-identity contract: steppe's whole value proposition is
FP64-accurate f-stats bit-identical to ADMIXTOOLS 2 / a native-FP64 oracle. A ~2.5–3.5-bit lossy
VQ with a 2.7×-of-optimal distortion floor (and inner-product bias in the MSE variant) is
**categorically incompatible** with any value that must stay bit-identical.

### SACRED — no lossy method, ever
- **`f2_blocks[i + P·j + P·P·b]`** — the bias-corrected f2 tensor, FP64 in every precision mode
  (fstats.hpp:50). The cacheable AT2-compatible interchange artifact; the thing the oracle diff
  runs against.
- **`vpair[...]`** — the per-block pairwise-valid SNP count, the S4 jackknife weight
  (fstats.hpp:54-60). Must *compose* with S4 weighting, not double-normalize.
- **The native-FP64 cancellation assemble** — `f2_num = R(i,j)+R(j,i) − 2·G(i,j) − H(i,j) − H(j,i)
  = Σ(p_i−p_j)²`, a difference of large like-magnitude sums (catastrophic cancellation). Held
  native FP64 in EVERY mode (f2_block_kernel.cu assemble; f2_estimator.hpp numerator/divide).
  §12's reason is decisive: emulation "faithfully computes the matrix product, it cannot recover
  significant bits annihilated in a prior subtraction." Quantizing operands feeding this
  subtraction bakes the error into the surviving content — unrecoverable, and it would corrupt
  the sacred `f2_blocks` output.
- **The multi-GPU combine order** — partials summed in fixed `g=0..G-1` device order, never NCCL
  AllReduce (order varies with G → breaks parity).

### ALREADY OPTIMALLY "QUANTIZED" — and not lossy at the output (do not double-dip)
The three f2 GEMMs (`G=Q·Qᵀ`, `Vpair=V·Vᵀ`, `R=[Q²;Hc]·Vᵀ`) already run a *quantized* matmul:
`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`, the **Ozaki fixed-slice** scheme pinned at
`EmulatedFp64{mantissa_bits=40}` (config.hpp:188-195). This is an integer/fixed-point
decomposition that *re-derives* full FP64 accuracy (40-bit → 2.2e-11 worst-case f2 error ≈ native,
at 7–17× native throughput). It is **lossless at the f2 output** by validation. **Do not conflate
this with TurboQuant** — Ozaki slicing is exact-by-construction; TurboQuant is the opposite (lossy
distortion-rate VQ). There is no further headroom to quantize the GEMM operands without leaving
the validated 40-bit regime; the only remaining FP64-path knob is the documented 32-bit step
(8.6e-9), already exposed as a tunable. Note the honorable-or-downgrade guard
(`emulation_honorable` / `engage_f2_precision`): if the FIXED-slice API is absent the path
downgrades to native FP64 with a logged tag rather than silently running lossy. **Any future
quantization knob MUST follow this template: honorable-or-downgrade, never silently lossy under
the reported tag.**

### SLACK — the one honorable home for a lossy method
**The fit / rotation model search (S8). It is the architecturally-sanctioned home, and it does
not exist in source yet** (confirmed: `src/core/fstats/` contains only f2 precompute — no
`qpadm`, `ranktest`, `gls_solve`, `nested_models`, or model-search file; S3–S8 are planned).
The architecture already reserved the exact mechanism:
- `Precision::Kind::Tf32` — "model-space SCREENING / ranking ONLY … NEVER bit-compared to
  ADMIXTOOLS 2 goldens and never emitted as a reported est/se/z/p without recomputation in
  EmulatedFp64/Fp64" (config.hpp:180-184).
- `DeviceConfig::search_streams = 4` — "Throughput-only lanes for the model-space search (S8),
  where results are recomputed in EmulatedFp64/Fp64 before any reported number" (config.hpp:222).

A TurboQuant-style lossy screen slots in here **verbatim**: rank/prune the many left/right
rotation configurations in low precision (quantized f2 subsets), then EXACTLY recompute only the
survivors in EmulatedFp64{40}. **§12 is preserved structurally** — the exact recompute is the
source of truth; the screen only changes *which* models get recomputed, never a reported number.
S8 is also embarrassingly parallel with zero cross-GPU traffic (§11.4), so it composes with
multi-GPU sharding.

---

## 3. The L2 fit — measured sizes and which structure becomes resident

### Measured L2 (sm_120 Blackwell)
| GPU | L2 total | Persisting cap (≈75%, confirm at runtime) | Mem BW |
|---|---|---|---|
| **RTX 5090** (GB202, cut, 170 SM) | **96 MB** | ~72 MB | ~1.79 TB/s |
| **RTX PRO 6000 Blackwell** (GB202, 188 SM) | **128 MB** | ~96 MB | ~1.79 TB/s |
| full GB202 reference | 128 MB | — | — |

Persistence mechanism (works on Blackwell as on Ampere/Ada): set-aside via
`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, N)`, then mark a range with a stream
access-policy window (`accessPolicyWindow{base_ptr, num_bytes, hitRatio,
hitProp=cudaAccessPropertyPersisting}`). Read the real cap from
`cudaDeviceProp.persistingL2CacheMaxSize` / `accessPolicyMaxWindowSize`; rule to avoid thrash:
`hitRatio · num_bytes ≤ persistingL2CacheMaxSize`. **L2 persistence only helps a kernel that is
VRAM-bandwidth-bound AND re-reads the pinned data.** It does nothing for compute-bound kernels,
stream-once data, or working sets that overflow L2.

### Which steppe structure benefits (quantified) — and which do NOT
- **Per-block f2/Vpair slab (P²):** P=256→512 KB, P=512→2 MB, P=1024→8 MB. A single slab fits L2
  trivially, but the assemble kernel reads each slab **once** and writes once — no reuse — so
  pinning buys nothing. **NO win.**
- **Whole resident `f2_blocks` tensor (P²·n_block, n_block=757):** P=64→24 MB (fits!), P=128 f2
  alone→95 MB (borderline on 5090), f2+vpair→189 MB (overflows both), **P≥256 overflows L2 even at
  INT8** (P=512 INT8 ≈ 378 MB). Whole-tensor residency is only realistic for small-P studies —
  exactly the device-resident regime already won (3.9×). **Narrow / mostly NO.**
- **GEMM operand tiles (Q/V/S per block):** a single block's operands fit L2 (13–79 MB to
  P≈1024), but they are streamed **once** per block; cuBLAS already tiles them through L1/SMEM/
  registers and the s-axis is read once. No inter-call reuse to capture, and they are FP64
  parity-critical anyway. **NO win — do not pin.**
- **Rotation-fit pair-subsets (the target):** each qpAdm fit reads f2 over only the model's k
  pops (low-tens), across all n_block=757 blocks:

  | k pops | FP64 subset | FP32 | INT8 |
  |---|---|---|---|
  | 8  | 379 KB | 189 KB | 47 KB |
  | 16 | 1.48 MB | 757 KB | 189 KB |
  | 24 | 3.33 MB | 1.66 MB | 426 KB |
  | 32 | 5.91 MB | 2.96 MB | 757 KB |

  A single model's subset is tiny — trivially L2-resident even at FP64. **The leverage is BREADTH
  across the S8 search**: thousands of candidate models, adjacent ones sharing overlapping pop
  subsets. Quantization **multiplies how many candidates' subsets co-reside in L2**: FP64→FP32 =
  2×, FP64→INT8 = 8×. With ~72–96 MB persisting L2 and ~50–760 KB INT8 subsets, you can keep the
  f2-subsets for **hundreds of candidate models L2-resident at once** for an approximate pre-screen
  — VRAM-round-trip-free. This is the only structure where both (a) re-read/breadth reuse exists
  and (b) quantization is parity-safe (it feeds only the screen).

---

## 4. THE EXPERIMENT — minimal, measurable, parity-gated

PRECONDITION: this requires S8 (the rotation model search) to exist in at least skeletal form.
Until then the experiment is **a prototype harness on synthetic/representative f2_blocks**, not a
production integration. Phase it.

### What to build
A two-tier model-screen prototype over a fixed `f2_blocks` tensor (real or representative;
P≈64–256, n_block≈757, a candidate-model list of C models, C ~ 10³–10⁴):

1. **Quantizer (the lossy bit):** build a compact screening copy of each candidate's f2 sub-tensor.
   Start with the **simplest defensible quantizer first** (FP32 truncation, then INT8 per-block
   affine), and only escalate to **TurboQuant-prod** (random-rotation + per-coordinate scalar +
   1-bit QJL residual, unbiased inner-product) if (a) the screening metric is inner-product-like
   and (b) plain FP32/INT8 ranking recall is inadequate. Rationale: TurboQuant earns its keep when
   the dynamic range is wide and inner-product bias matters — its random rotation is the same
   dynamic-range-taming move that the rejected dynamic-mantissa Ozaki path lacked — but it adds
   complexity that a plain scalar quantizer may not need at the bit-widths in play.
2. **L2-resident screen:** pin the quantized subsets with an access-policy window
   (`hitProp=Persisting`, `hitRatio·num_bytes ≤ persistingL2CacheMaxSize`), run the approximate
   qpAdm rank/feasibility metric (GLS residual / rank-test surrogate) over all C candidates,
   producing an **ordering + a survivor set** (top-S, or all above a feasibility threshold).
3. **Exact recompute (the source of truth):** recompute the S survivors in EmulatedFp64{40}
   (and native FP64 where §12 mandates), emit est/se/z/p ONLY from these.

### Metrics
- **Throughput:** candidate-screens/sec for the quantized-L2 path vs. (a) a full-FP64 fit of every
  candidate, and (b) an unpinned (no access-policy window) quantized path. Report the two ratios
  separately — they isolate the *quantization* win from the *L2-residency* win.
- **L2 behavior (ncu, Release build):** `lts__t_sector_hit_rate`, `dram__bytes_read`, and achieved
  vs. peak DRAM throughput; confirm the screen is L2-bandwidth-bound (high L2 hit rate, low DRAM
  bytes) not VRAM-bound, and that pinning measurably raises the hit rate.
- **Screen quality (recall):** does the quantized screen's survivor set CONTAIN the true top-S
  models that exact FP64 ranking would have selected? Report recall@S and the rank correlation
  (Spearman) of the quantized ordering vs. the exact ordering.

### THE PARITY GATE (non-negotiable)
- The **reported numbers** (est/se/z/p for every model that survives and is recomputed) MUST be
  bit-identical to the §12 single-tier EmulatedFp64/Fp64 result for that same model — `memcmp`,
  not tolerance. The screen is allowed to change *which* models are recomputed; it is NEVER allowed
  to change a recomputed value. This is enforced because the exact recompute is independent of the
  screen — the screen only selects inputs.
- **Recall gate:** the quantized screen must retain the true top-S models with recall ≥ a fixed
  threshold (propose **recall@S ≥ 0.999** on a representative candidate set, i.e. it essentially
  never drops a model that exact ranking would have kept). A missed true-positive is a *correctness*
  failure of the screen even though §12 (which governs reported values) is untouched. Mitigation if
  recall is short: widen the survivor set (recompute more) — self-correcting, at a throughput cost.

### Expected payoff
- Quantization shrink: FP64→INT8 = 8× more candidates co-resident; FP32 = 2×.
- L2-residency: screen reads hit on-chip L2 (multi-TB/s) instead of the 1.79 TB/s DRAM ceiling,
  for the re-read-heavy breadth pass. **Plausible: a low-single-digit to ~mid-single-digit×
  speedup on the SCREEN phase only** (not the whole fit — the exact recompute of survivors is
  unchanged FP64 work). The end-to-end S8 win depends on the screen rejecting a large fraction of
  candidates cheaply, i.e. survivor-fraction ≪ 1.

### Failure modes (be honest)
- **No reuse / no breadth:** if S8 ends up evaluating few candidates, or each only once, there is
  no re-read for L2 to capture → no win. The whole idea is contingent on a *large* candidate space.
- **Working set overflows persisting L2:** if the co-resident subset set exceeds ~72–96 MB it
  thrashes and the win evaporates; the `hitRatio·num_bytes` budget must be respected.
- **Screen is compute-bound, not bandwidth-bound:** if the approximate GLS/rank metric is FLOP-heavy
  per candidate, L2 bandwidth is not the bottleneck and pinning does nothing (the high-P f2 GEMMs
  are already compute-bound at intensity ≈ P/8 — but the *screen* metric over a tiny k-pop subset is
  the opposite; verify with ncu).
- **Recall too low:** if INT8 ranking drops true survivors below the recall gate, fall back to FP32
  (2× instead of 8×) or widen survivors — either erodes the payoff.
- **Premature build:** integrating before S8 exists means optimizing a phase that may not look like
  the prototype assumes.

### GO / NO-GO
- **GO** iff, on a representative candidate set: (1) the parity gate holds (recompute bit-identical),
  AND (2) recall@S ≥ 0.999, AND (3) the quantized-L2 screen is ncu-confirmed L2-bandwidth-bound with
  ≥2× screen-phase throughput over full-FP64-fit-every-candidate, AND (4) survivor-fraction is small
  enough that the end-to-end S8 fit improves by a margin worth the complexity (propose ≥1.5×
  end-to-end on the fit phase).
- **NO-GO / shelve** if any of: screen not bandwidth-bound (pinning is a no-op), recall can't clear
  the gate without recomputing most candidates (no pruning benefit), or S8's candidate space turns
  out small.

---

## 5. Blunt bottom line

**Promising but secondary — not a distraction, not a top lever.** It is a genuinely correct,
parity-safe idea with a real (if modest) payoff, and the architecture *already reserved the exact
slot* for it (`Tf32` screening + `search_streams` + the recompute-survivors contract). That is the
honest good news.

But weigh it against the known levers:
1. **M5 streaming (large-P precompute)** — the actual blocker for P≥256 (the 3.57 GB resident
   tensor overflows VRAM); without it, large studies don't run at all. **Highest priority.**
2. **The multi-GPU rotation fit (S8 itself)** — the embarrassingly-parallel, throughput-bound phase;
   building it and sharding it across GPUs is the big, unconditional throughput win. **High
   priority — and it is the PREREQUISITE for the TurboQuant idea, which is a refinement *inside* S8.**
3. **TurboQuant-L2 screen** — a **refinement of an optimization (the screen) of a phase (S8) that
   does not exist yet.** It cannot precede S8; its payoff is bounded to the screen sub-phase; and a
   plain FP32/INT8 quantizer may capture most of the win without TurboQuant's machinery (only
   escalate to TurboQuant-prod if recall demands it). **Medium priority, correctly *after* S8 lands.**

Ranking: **M5 ≳ multi-GPU S8 ≫ TurboQuant-L2 screen.** Recommendation: **do NOT invest now.** Park
this note. When S8 is being built, revisit it as a scoped screen-phase optimization, and start with
the cheapest quantizer (FP32, then INT8) before reaching for TurboQuant — keep the random-rotation/
unbiased-inner-product variant in reserve for the specific case where the screening metric is
inner-product-like and plain scalar quantization can't clear the recall gate.

---

## Sources
- TurboQuant: arXiv:2504.19874 (abstract quoted verbatim); ICLR 2026 OpenReview id=tO3ASKZlok;
  Google Research blog "TurboQuant: redefining AI efficiency with extreme compression."
- L2 sizes: TechPowerUp / Tom's Hardware (RTX 5090 96 MB, GB202 128 MB); CpuTronic + NVIDIA RTX
  PRO 6000 Blackwell page (128 MB L2, 188 SM, 96 GB GDDR7, ~1.79 TB/s); NVIDIA RTX Blackwell GPU
  Architecture whitepaper.
- L2 persistence: CUDA Programming Guide §4.13 (L2 Cache Control); NVIDIA Blackwell Tuning Guide;
  Lei Mao "CUDA L2 Persistent Cache."
- steppe code (read-only): architecture.md §5 / §11.4 / §12; src/device/cuda/f2_block_kernel.cu
  (3 GEMMs + emulated-FP64 engagement + native assemble); src/core/internal/f2_estimator.hpp
  (cancellation assemble + finalize); include/steppe/config.hpp:149-227 (Precision / Tf32 /
  search_streams); include/steppe/fstats.hpp:47-78 (F2BlockTensor, FP64-in-every-mode);
  docs/cleanup/m4.5/perf-discovery.md (GEMM-bound profile, 3.57 GB resident tensor).
```

