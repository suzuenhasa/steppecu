# TurboQuant + L2-residency on the RTX 5090 — Feasibility (research fan-out)

*Output of the `research-turboquant-l2` workflow (5 web/code angles + adversarial critic + synthesis). Honest, quantified; respects the §12 parity law. Supersedes the speculative design notes in `turboquant-l2-experiment.md`.*

---

Both critic corrections are verified against source: the repo's real measured single-GPU number is **351 models/sec** on the 84-model validated set (fit-engine.md:377), with the scale-N sweep explicitly RETRACTED; and the gather (qpadm_fit_kernels.cu:865) is the catastrophic-cancellation 4-tap subtraction reading the resident f2 once into `dX`. I have what I need to synthesize honestly.

---

# TurboQuant + L2-Residency for steppe on the RTX 5090 — Feasibility Report

**Verdict in one line:** The L2-residency mechanism is **NO-GO** — it attacks a cost that is ~0.01–0.04% of the per-model wall on a kernel that already fits L2 at full FP64. The *separable* screen-then-verify architecture is a real idea, but its lever is compute, not L2, and one ncu run on box5090 should formally close the L2 question before any code is written.

---

## 1. What TurboQuant is

TurboQuant (arXiv:2504.19874, Zandieh/Daliri/Hadian/Mirrokni, Google Research; ICLR 2026, OpenReview `tO3ASKZlok`) is a **data-oblivious, training-free codec for high-dimensional Euclidean vectors**. The algorithm is: apply a random rotation `y = Πx` (which drives each rotated coordinate toward a Beta/`N(0,1/d)` distribution, taming dynamic range), then quantize each coordinate to the nearest of `2^b` precomputed Lloyd-Max centroids, and rotate back on dequant [HIGH, ar5iv Algorithm 1]. It comes in two variants: **TurboQuant-MSE** (MSE-optimal, but *biases* inner products) and **TurboQuant-prod** (adds a 1-bit Quantized-JL residual pass to make the inner-product estimator *unbiased*, with variance shrinking ∝1/d) [HIGH, ar5iv Algorithm 2 + Theorem 2]. Its headline targets are LLM KV-cache and ANN/embedding vectors — every reference use is a *streaming codec on independent vectors* whose cost amortizes only under heavy re-read (each quantized key is read on every subsequent attention step) [HIGH]. Provable distortion is within ≈2.7× of the `4^−b` information-theoretic floor; measured 3.5 bits/channel = full-precision-neutral on LongBench, ~95% ANN Recall@10 at 4-bit [HIGH on paper tables]. **Caveat (critic-verified):** the paper gives no explicit recall@10 numeric annotation (Figure 5 is comparative-only), no CUDA kernel, and ran on a single A100 — the "8× on H100" is a secondary Google-blog footprint framing, not an experiment [HIGH]. There is **no shipped CUDA**; reference impls are NumPy/PyTorch with experimental Triton, and the rotation is dense O(d²)-per-vector (fast-Hadamard is unimplemented future work in every repo) [HIGH].

## 2. How L2-residency works on the RTX 5090

- **L2 size:** 96 MB on the 5090 (170 of 192 SMs enabled; the GPC fuse-off cuts L2 — full GB202 = 128 MB) [HIGH, Tom's Hardware / TechPowerUp / chips-and-cheese].
- **Bandwidth prize:** GDDR7 DRAM = 1792 GB/s (≈1.79 TB/s); measured Blackwell L2 ≈ **8.7 TB/s** with ~130 ns hit latency vs ~329 ns VRAM latency — so an L2 hit is worth **~4.9× the DRAM read rate / ~2.5× lower latency** [HIGH, chips-and-cheese on GB202 silicon; the 8.7 TB/s is the PRO 6000 — the 5090's cut L2 is inferred slightly lower but same ~5× ratio class].
- **Persisting-L2 mechanism (CUDA 13):** `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, N)` carves a set-aside; you mark a region with `cudaAccessPolicyWindow{base, bytes, hitRatio, hitProp=cudaAccessPropertyPersisting}` and attach it to a stream (`cudaStreamSetAttribute`) or graph node. The practical carve-out caps at **~2/3–3/4 of total L2** (so ~64–72 MB on the 5090, *read at runtime* via `cudaDeviceProp::persistingL2CacheMaxSize` — not yet measured in steppe), governed by the thrash rule `hitRatio·num_bytes ≤ set-aside` [HIGH, CUDA 13.1 Programming Guide §4.13].
- **Realistic win size:** Even in the *ideal* memory-bound-reuse case, the published persisting-L2 win is **~20%** (Lei Mao microbenchmark), not the raw ~5× ratio — because it accelerates *re-read hits*, while first-touch misses and compute are unaffected [FACT, Lei Mao].

**The load-bearing precondition:** persisting-L2 helps only a kernel that is (a) DRAM-bandwidth-bound AND (b) re-reads the pinned data. steppe's fit satisfies neither (§3).

## 3. Would it speed up steppe? — the honest bottleneck analysis

**The anchor, stated honestly (critic correction):** The only *clean single-GPU* measurement in the repo is **351 models/sec on the 84-model validated set** (fit-engine.md:377, ≈2.85 ms/model). The "530/sec @ 9086 models" from the prompt is *not* a repo measurement — the scale-N sweep is explicitly `[synthetic scale-N RETRACTED — commit 2a0c020]`, and "9086 models" appears only in the multi-GPU host-bounce context. Using the slower, real 351/sec makes every fraction below *more* lopsided, so the verdict is robust to which anchor you pick.

**Three independent walls, any one fatal:**

**Wall A — the per-model f2 working set already fits L2 at full FP64.** The gather (`assemble_f4_gather_models_kernel`, qpadm_fit_kernels.cu:839–866) reads only the `(nl+1)×(nr+1)` f2 submatrix across all `nb=757` blocks. Unique DRAM bytes moved per model = `(nl+1)(nr+1)·nb·8` ≈ **106 KB** (golden shape nl=2,nr=5) to **~390 KB** (nl=5,nr=10). Even cache-line-inflated for L2 *occupancy* (128 B/line, strided), the footprint is ~7.6 MB (18 pops) to ~19 MB (38 pops) — all ≪ the 64–72 MB set-aside. **Quantization shrinks nothing that isn't already resident** [HIGH, code+arithmetic].

> *Critic note on the numerator:* the angle reports disagree by ~8–60× on per-model f2 bytes (106 KB vs 0.87 MB vs 19 MB). The discrepancy is real and worth flagging: 106 KB is correct **DRAM bytes moved** (unique cells `(nl+1)(nr+1)·nb`); 0.87 MB over-counts by using `k²≈144` instead of `(nl+1)(nr+1)=18`; 19 MB is correct **L2 occupancy** after cache-line inflation. All three round to "negligible," but they are not corroborating each other's arithmetic.

**Wall B — f2 DRAM traffic is a sub-percent fraction of the wall; Amdahl caps the win near 1×.** At 351 models/sec (2.85 ms/model) and the unique f2 footprint at 1.79 TB/s DRAM: 106 KB → 0.06 µs (**0.002%**); 390 KB → 0.22 µs (**0.008%**). Even the cache-line-inflated 19 MB at full DRAM rate is 10.7 µs = **0.38%**. Removing *all* f2 HBM traffic — the absolute maximum TurboQuant→L2 could deliver — yields an Amdahl ceiling of **~1.004× at worst-case occupancy, ~1.0001× on bytes-moved** [HIGH, arithmetic]. (The cost-benefit report's "1.0005×" is the right order of magnitude but is false precision on an unverified denominator — present it as "≤~1.004×, sub-percent.")

**Wall C — the rotation is launch/serial-LA-bound, which L2 cannot touch.** `f2_device()` is passed to *exactly one* kernel (the gather); every downstream stage reads only tiny per-model arenas (`dX`/`dLoo`/`dQinv`/`dXtau`), never f2 [HIGH, grep verified]. The 2.85 ms/model is dominated by **single-thread-per-model serial linear algebra**: `qpadm_fit_models_kernel` (one thread/model: ALS loop + rank sweep + popdrop), and the dominant `qpadm_loo_models_kernel` which fans `B·nb` threads each running a full `dev_als_weights` (a ≤60-sweep Jacobi-SVD seed + ALS + weight solve) — i.e. **~757 serial LOO refits per model** for the SE [HIGH, code; critic independently verified the 60-sweep Jacobi seed at :125 and the 757-per-model LOO at :1085]. Plus a per-chunk launch tax (column-by-column `nrhs=1` potrsBatched loop + 3 stream syncs). This is a compute/occupancy/launch regime; pinning f2 in L2 is a no-op for kernels that never read f2.

**Where it does NOT help (the L2 framing):** the entire premise — "shrink f2 to be L2-resident → HBM-traffic-free screen" — is moot. The f2 is already resident at FP64, its traffic is sub-percent, and the bottleneck is elsewhere.

**Where a screen COULD help (separable, compute-lever):** If ~70–99%+ of rotation models are infeasible (qpAdm literature: FNR 89.9–99.8%, FDR 72.5–100% for proximal rotating — only consistent with a small feasible minority) [HIGH that the majority is rejected; MODERATE on a single number], a *cheap screen that skips the full FP64 SVD/Cholesky/757-LOO chain* on the obvious-infeasible majority has a real Amdahl ceiling of **~3–5×** (`1/(0.1 + screen_cost)`), eroded by recall-driven survivor widening [MODERATE — depends on building a screen both ≫cheaper and safe]. That win comes from **fewer/cheaper fits**, not faster f2 reads. TurboQuant is *incidental* here — relevant only because the GLS residual is quadratic/inner-product-shaped, so an error-bounded unbiased estimator (TurboQuant-prod / RaBitQ-style) could set a provably safe prune threshold.

## 4. How it would work in steppe IF worth it (screen-then-verify, parity-safe)

This is the *only* §12-legal shape, and it is well-precedented (it is structurally IVF-PQ/ScaNN/RaBitQ with rerank — coarse-quantize-then-rerank is the canonical two-stage ANN architecture; cuVS uses `refine_ratio=2`) [HIGH on the pattern]:

- **Screen pass:** a cheap, low-precision metric ranks/prunes candidate models. To be parity-relevant it must use an **error-bounded unbiased** estimator (TurboQuant-prod, *not* plain PQ which is biased and has >50% distance error on hard data) [HIGH].
- **Parity-safe error budget (§12):** prune on a **confidence-padded lower bound**, never the point estimate — "reject only if `est − ε > threshold`" where `ε ~ O(1/√D)` is the quantizer's high-probability bound and steppe's effective `D = k·n_block` is in the hundreds–thousands (favorable, small ε). This converts the recall gate into a *provable* false-negative probability δ [HIGH on mechanism; MODERATE that ε for steppe's chi-square/quadratic metric needs re-derivation, not direct import].
- **Critical constraint:** the qpAdm feasibility boundary is *statistically noisy* (poor P-value/optimality correlation). The screen must prune **only the clearly-infeasible tail**, padded well away from p≈0.01; every near-boundary model goes to FP64 verify. The screen rejects the obvious ~90%; it must never adjudicate the marginal ~10%.
- **Verify stage (unconditionally §12-clean):** survivors recompute through the exact `assemble_f4 → run_impl` chain in **EmulatedFp64{40}** (`model_search.cpp:31`). The screen only selects *which* models enter that chain; the reported est/se/z/p come solely from the FP64 recompute [HIGH, code].
- **Parity blocker on f2 itself:** the gather's 4-tap subtraction `0.5·(f2(Li,R0)+f2(L0,Rj)−f2(L0,R0)−f2(Li,Rj))` is catastrophic-cancellation-sensitive and held native FP64 in *every* mode (qpadm_fit_kernels.cu:865, verified). Quantizing f2 directly is parity-illegal even as a screen *input* — the screen must derive its own approximate operands, adding a whole second pipeline.

**Honest cost accounting for this design:** TurboQuant's O(d²) rotation is *per-screen* (applied to each candidate's re-derived operands), not one-time; and the screen adds a full approximate fit pipeline *alongside* the existing FP64 fit. Because the FP64 fit isn't bandwidth-bound, the screen only buys you anything by **avoiding** the expensive fit on rejected models — pure compute-rejection economics, zero L2 involvement.

## 5. Verdict

| Half of the idea | Verdict |
|---|---|
| **"Quantize f2 → L2-resident → HBM-traffic-free screen"** (the L2-residency mechanism) | **NO-GO — shelve permanently.** Three independent walls (already L2-resident at FP64; sub-percent DRAM traffic, Amdahl ≤~1.004×; launch/serial-LA-bound). Robust to the anchor. |
| **"Screen-then-verify to skip FP64 fits on the infeasible majority"** (compute-rejection) | **EXPERIMENT-FIRST.** Separable, well-precedented, §12-safe with an error-bounded estimator. Plausible ~3–5× ceiling, but contingent on a screen ≫cheaper than the full fit *and* a provable recall gate. TurboQuant is incidental, not load-bearing. |

**Overall: NO-GO on TurboQuant→L2 as framed.** The L2 angle is a solution looking for a problem. Keep TurboQuant only as a filed note for the compute-rejection screen.

**The single smallest experiment that decides it (no new code):** one ncu memory-workload run on box5090, Release build, on `test_qpadm_rotation` with the real fixture:

```
ncu --set full --section MemoryWorkloadAnalysis \
    -k regex:"assemble_f4_gather_models|qpadm_fit_models|qpadm_loo_models" \
    tests/reference/test_qpadm_rotation     # fixtures/f2_rot.bin, Release
```

Read three counters: `dram__bytes_read.sum`, `lts__t_sector_hit_rate.pct`, and `sm__throughput` vs `dram__throughput`. **Prediction (from structure+arithmetic, ~3 orders of magnitude of margin):** the gather is already near-100% L2-hit with negligible DRAM bytes; the fit/LOO kernels are compute/latency-bound at low occupancy (the single-thread-per-model frame). If f2-attributable `dram__bytes_read` is <1% of the run and the gather is L2-hit-bound, update `docs/research/turboquant-l2-experiment.md` from "MEDIUM-priority, gated behind S8" to **"REFUTED on built S8: not f2-HBM-bound (Amdahl ≤~1.004×), per-model f2 already L2-resident at FP64; shelve the L2 angle; re-scope any screen work to compute-rejection."** Until that profile exists, the honest claim is "very likely refuted by structure + arithmetic," not "proven." The profile will instead point at the real lever: warp-cooperative per-model LA, CUDA-graph capture per (nl,nr,r) bucket, and reworking the 757-per-model LOO-SE batching — none involve quantization or L2.

## 6. Sources

- TurboQuant: [arXiv:2504.19874 abstract](https://arxiv.org/abs/2504.19874) · [ar5iv full-text](https://ar5iv.labs.arxiv.org/html/2504.19874) · [OpenReview tO3ASKZlok](https://openreview.net/forum?id=tO3ASKZlok)
- TurboQuant reimpls: [OmarHory/turboquant](https://github.com/OmarHory/turboquant) (PyTorch+Triton, 1.85× attn) · [yashkc2025/turboquant](https://github.com/yashkc2025/turboquant) (NumPy, dense-QR) · [scos-lab/turboquant](https://github.com/scos-lab/turboquant) · [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) (3-bit recall caveat)
- RTX 5090 / GB202 hardware: [Tom's Hardware L2](https://www.tomshardware.com/pc-components/gpus/nvidias-rtx-5090-5080-reportedly-have-the-same-l1-cache-size-per-sm-compared-to-rtx-4090-4080) · [chips-and-cheese Blackwell (8.7 TB/s L2, 130/329 ns)](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu) · [chips-and-cheese H100 L2 vs VRAM](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth) · [RTX 5090 1792 GB/s](https://www.runpod.io/articles/guides/nvidia-rtx-5090) · [vast.ai 5090 specs (96 MB L2)](https://vast.ai/article/nvidia-geforce-rtx-5090-specs-everything-you-need-to-know)
- CUDA persisting-L2: [CUDA 13.1 Programming Guide §4.13 L2 Cache Control](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/l2-cache-control.html) · [Lei Mao — CUDA L2 Persistent Cache (~20% win, 2/3 carve-out)](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/)
- Screen-then-rerank precedent: [NVIDIA cuVS IVF-PQ deep dive (refine_ratio=2)](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/) · [Google ScaNN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) · [RaBitQ arXiv:2405.12497 (unbiased, O(1/√D) bound)](https://arxiv.org/pdf/2405.12497) · [RaBitQ GitHub](https://github.com/gaoj0017/RaBitQ)
- qpAdm rotation statistics: [PMC10614728 (FNR 89.9–99.8%, FDR 72.5–100%)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614728/) · [PMC12118350 (positive-model definition)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12118350/) · [Genetics iyaf047 (P-value/optimality)](https://academic.oup.com/genetics/article/230/1/iyaf047/8102970)
- steppe code (absolute paths): `/home/suzunik/steppe/src/device/cuda/qpadm_fit_kernels.cu` (gather :839–866 = the 4-tap cancellation read; LOO 757/model :1085; Jacobi seed :125) · `/home/suzunik/steppe/src/device/cuda/cuda_backend.cu:2177` (batched fit; column-wise potrs loop) · `/home/suzunik/steppe/src/core/qpadm/model_search.cpp:31` (EmulatedFp64{40} verify), `:147` (host-bounce TODO) · `/home/suzunik/steppe/docs/design/fit-engine.md:377` (REAL 351 models/sec + RETRACTED scale-N) · `/home/suzunik/steppe/docs/research/turboquant-l2-experiment.md` (the doc to re-scope)
