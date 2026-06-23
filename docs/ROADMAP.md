# `steppe` — Roadmap

Companion to [`architecture.md`](architecture.md). That doc says **how** the system is structured; this one says **in what order** we build it and **what is already proven**. Read both before writing code.

---

## 0. Status — what is validated (2026-06-21)

> **Where we are now (one line):** Phase-1 precompute (S0–S2) is DONE through M5 (device-resident output + out-of-core streaming; full-autosome P=2500 ~51.5 s on one 5090), and the Phase-2 **qpAdm FIT engine (S3–S8) BACKEND is now FINISHED and golden-gated on the GPU** against real-AADR ADMIXTOOLS-2.0.10 goldens — the 6 backend-finish items (F1–F6, step 1 of the backend-first build sequence) all landed (`e8430a2`…`2496a14`), closing NA handling, the `ChisqUndefined` outcome, the domain-outcome acceptance gate, the `run_qpwave` golden, and the determinism widen — see §3. The codebase-wide **big refactor is complete** (`docs/cleanup/bigrefactor/`, bit-identical, no perf regression). What is *not* built — **the honest next work (step 2)**: a CLI, Python bindings, then (step 3) standalone f-stat entry points each with its own CLI/bindings, plus qpfstats and precompute M6/M7 — see the **Phase 2.5** (§3) for the honest remaining work.

### 0.1 The f2 kernel — the original uncertainty, settled (2026-06-15)

The hardest, most *uncertain* part — the f2 compute kernel and its precision/throughput on **real** data — is settled. Measured on real AADR v66 HO data, 2× RTX 5090, CUDA 13:

- **f2 = 3-GEMM reformulation:** `G = Q·Qᵀ`, `Vpair = V·Vᵀ`, `R = [Q² ; Hc]·Vᵀ`, then `f2 = (R_diag + R_diagᵀ − 2G − H − Hᵀ) / Vpair`. ✅
- **Precision policy (MEASURED on real data, not synthetic):**
  | mode | speed vs native FP64 | worst-case f2 error | use |
  |---|---|---|---|
  | native FP64 | 1× | 1e-11 | exact baseline / fallback |
  | **fixed-slice Ozaki, 32-bit mantissa** | **8.5–17.5×** (grows with P) | 8.6e-9 | reported-stat precompute (max speed) |
  | **fixed-slice Ozaki, 40-bit** | 7.2–13.3× | 2.2e-11 (≈ native) | reported-stat precompute (safe default) |
  | fixed-slice Ozaki, 48-bit | 6.1–10.3× | 1e-12 | exceeds native |
  | Ozaki **dynamic** mantissa (~60-bit) | ~1× (**no win — TRAP**) | 1e-11 | ❌ rejected |
  | TF32 | ~64× | 7.7e-2 | screening/ranking ONLY |
  | FP32 / BF16×9 | ~30× | 1.6e-3 | screening only (= FP32) |
- **Speedup scales with population count** (arithmetic intensity ≈ P/8): P=768 → 8.5×, P=2416 → 14.5×, P=4266 → 17.5× (fixed-32b).
- **Scale demonstrated:** full all-pairs f2 over **all 4,266 AADR populations** (100k SNPs) in **489 ms** (fixed-32b) vs **8.56 s** native FP64 — vs ADMIXTOOLS 2's *hours*.

> **Load-bearing cautionary tale (never unlearn this):** synthetic uniform data showed dynamic Ozaki winning **8×**; real AADR data (wide dynamic range from variable per-pop sample sizes + missingness) showed **parity (1.0×)**. **Precision/throughput claims are benchmarked on REAL data only — never synthetic.** The win came from *capping* the mantissa bits (fixed slices), which dynamic mode squandered.

This is ~the kernel. It is **not** the full `f2_blocks` cache — see §3.

---

## 1. The spike → production discipline

Everything under `experiments/` is **throwaway spike code**: monolithic single files, magic numbers, no RAII, no config, and the 3-GEMM pattern / `CUDA_CHECK` / loader **duplicated across three `.cu` files**. It answered the science; it is **not** production and must not be extended in place.

Production follows `architecture.md`: layer split (`api → core → device`, `io` leaf), compiler-enforced (CUDA `PRIVATE` to `steppe_device`), RAII everywhere, typed immutable config, DRY helpers, a CPU-reference oracle for every kernel. **Milestone 0 is "lift the spike into that structure," not "keep going."**

---

## 2. Sequencing & dependencies

Dependency chain for the cache:

```
genotypes → decode → allele-freq (Q,V,N) → SNP→block → per-block f2 (3-GEMM) → f2_blocks
```

- Allele-freq is **upstream** of f2 but **decoupled** from it by the **Q/V/N contract** + the `ComputeBackend` seam. So we lock the f2 *structure* against that contract first; the numpy `build_tgeno_matrix.py` remains the Q/V/N producer until M1 replaces it with a GPU decoder.
- **Decide early (M0), don't build:** the Q/V/N semantics, in particular **pseudo-haploid** handling for ancient samples — it changes how `N` (non-missing haploid count) is computed, not the contract.
- Therefore: **structure-lift (M0) → data front-end (M1) → block structure (M3) → per-block f2 (M4).** Alleles are the immediate next step, not a blocker.

### The Q/V/N contract (seam between the data front-end and the f2 kernel)
Three column-major `[P × M]` arrays (leading dim `P`; element `(pop i, snp s)` at `i + P·s`), per SNP-block:
- **Q** — frequency of the fixed reference allele in `[0,1]` (zero-filled where invalid).
- **V** — validity mask (`1.0` if the pop has a non-missing genotype at that SNP, else `0.0`).
- **N** — **non-missing haploid count**: `2 × non-missing diploids` *or* `1 × non-missing pseudo-haploids` (the ancient-DNA case — **must** be honored).
- Block membership comes from the shared `block_partition_rule` (cM-based). This contract is stable regardless of decode method, ploidy, or precision mode.

---

## 3. Phases & milestones

### Phase 0 — Scaffold & standards  (`architecture.md` §17)
Repo skeleton, CMake + presets, RAII wrappers (`DeviceBuffer<T>`, `Stream`, handle wrappers), `ComputeBackend` interface + CPU/CUDA stubs, CI gates (clang-tidy/format/sanitizer/arch-grep), and **`include/steppe/config.hpp`** carrying the precision knob (incl. the fixed-slice mantissa-bits, see §4).

### Phase 1 — `f2_blocks` cache (precompute engine)  (M0–M5 ✅ DONE; M6–M7 remaining)
> **Substantially complete through M5.** M0–M4 (3-GEMM kernel → loader → filters → per-block tensor) + M4.5 multi-GPU + **device-resident output (`1f80c0c`)** + **M5 out-of-core streaming (`176a07d` tiered output + `c65179f` SNP-tile input)** are done and bit-identical. Full-autosome P=2500 completes on a single 32 GB 5090 in ~51.5 s. **The key lesson:** the precompute was host-result-bound — the perf win was getting the result OFF the CPU (device-resident output, ~4.3× at P=512) + streaming the input, **not multi-GPU.** Remaining: M6 (multi-dataset QC) + M7 (on-disk cache / FST). **The Phase-2 fit engine is now BUILT on top of this (see §3, Phase 2); the remaining precompute work (M6/M7) is tracked under Phase 2.5.**
- **M0 — Structure lift (no new functionality).** Port the validated 3-GEMM kernel + fixed-slice-Ozaki precision into the architecture layout: kernel → `src/device/cuda/f2_block_kernel.cu`; orchestration → `src/core/fstats/f2_from_blocks.cpp`; the f2 estimator as a shared `__host__ __device__` primitive (so CPU-ref and GPU can't diverge); promote **every** magic number to `config.hpp`/named constants (§4); dedup `CUDA_CHECK`/loader/GEMM-args into the §8 helpers; wrap all allocations in `DeviceBuffer<T>`; add the CPU-reference-equivalence test. Gate: GPU matches CPU reference at the tight tier; no magic numbers; no duplication.
- **M1 — Genotype decode + allele frequency. ✅ DONE** (`150bfb3`). New `src/io/` LEAF (TGENO/GENO header parse + `.geno`/`.snp`/`.ind` readers + plain tile struct) + shared `__host__ __device__` decode primitive (`core/internal/decode_af.hpp`) + `ComputeBackend::decode_af` seam (CPU-reference + CUDA `decode_af_kernel.cu` segmented reduction). **Correction:** the real file is **TGENO individual-major** with the **raw-value 2-bit mapping** (0/1/2 copies, 3 = missing) — *not* the binary mapping. Reproduces the numpy oracle **bit-for-bit (max|Δ| = 0 for Q/V/N)** on `derived_acc` (P=50, M=100k); decoded Q/V/N → `compute_f2` = M0's 1.085e-11. **No pseudo-haploids in v66 HO** (all diploid, `N = 2×count`); ploidy is a documented metadata parameter (default 2, never inferred from genotypes). 67.6 ms = 8.6e9 SNP·sample/s; compute-sanitizer clean.
- **M2 — Missingness + filters. ✅ DONE** (`1bbbad4`). Host-pure predicates in `src/io/filter/` (the M2 analogue of `f2_estimator.hpp`): MAF / `geno` / `mind` / include-exclude + flag-gated monomorphic / transversions-only / autosomes-only, all from `FilterConfig`. Pairwise-complete (V/N) is the default and the parity path — **no new f2 math, no kernel**. Proven on real AADR: no-op when default (f2 bit-identical to unfiltered); **drop-equals-mask** (dropping SNP columns ≡ zeroing `V`) bit-identical; keep-masks integer-exact vs an independent scalar oracle across 7 configs. **steppe full-v66 walk block counts (AT2 SNP-anchored `setblocks` rule): chr 1–22 = 711, chr 1–23 = 747, chr 1–24 = 748** (the pre-fix floor-grid rule gave 719/756/757); the AT2 *cache* parity target on the Haak 15-pop polymorphic union is **709** (264544 SNPs, see `docs/research/block-partition-at2.md`); `is_autosome` = 1–22. Strand-ambiguous = self-complementary A/T·C/G palindromes (zero in HO); the 18,534 transversions are a separate flag. Convention flags pinned for the AT2 golden: MAF pooled-across-samples; `geno` sample-axis missing fraction (AT2 `maxmiss` is population-axis).
- **M3 — SNP→block assignment. ✅ DONE** (`f7f31c6`; AT2-walk reconciliation per `docs/research/block-partition-at2.md`). `core/domain/block_partition_rule.{hpp,cpp}`: scalar `block_of` (kept as a primitive, no longer used by the walk) + `assign_blocks(chrom[], genpos_morgans[], blocksize) → BlockPartition{block_id[], n_block}` (one file-order pass: the AT2 `setblocks` SNP-anchored cumulative walk — a new block opens on a chromosome change OR when the distance from the block's FIRST SNP reaches blgsize, `>=` inclusive; the anchor re-sets to the opening SNP so the remainder rolls forward) + `block_size_cm_to_morgans` (the single cM↔Morgan site). Host-pure, shared by `io` and kernels. Real-AADR: all 584,131 SNPs (chr 1–24) → **748 blocks** (autosome chr 1–22 = 711, chr 1–23 = 747); the rule is deliberately **filter-agnostic** (the chromosome-range drop is `io` territory — see M2). The AT2 cache parity target on the Haak polymorphic union is **709** (`docs/research/block-partition-at2.md`). Negative chr17 positions + all-zero chr24 handled.
- **M4 — Per-block f2. ✅ DONE** (prerequisite of M4.5, which is bit-identical to this single-GPU path). The 3-GEMM kernel at **fixed-slice Ozaki (default 40-bit)**, **batched over blocks** (`cublasDgemmStridedBatched`) → `f2_blocks [P × P × n_block]`, carrying per-block `Vpair` (the S4 jackknife weight). M0 measured *one big GEMM*; M4 batches ~700 *small* per-block GEMMs — a different arithmetic-intensity regime, so M4 **opened with a throwaway spike** (`experiments/`) confirming the Ozaki 40-bit speedup + accuracy hold per-block (strided-batched vs loop-of-GEMMs vs one-big-then-rebin) on **real AADR**, before the design was committed. Validated against the CPU oracle per-block.
- **M4.5 — Single-node multi-GPU precompute (shard + parity combine). ✅ DONE** (`867a4bf`; nsys root-cause `165f655`) — **correct and bit-identical.** Shard SNP work across `DeviceConfig::devices` (≥2 GPUs — measured on rtxbox 2× RTX PRO 6000 Blackwell sm_120): SPMG — one host thread + per-device stream per GPU, `cudaSetDevice`, opportunistic peer access. Each device computes a **full-shape partial** `f2_blocks` + `Vpair` over its SNP range/tiles; the G partials are summed **once, in fixed device order** (`g = 0..G−1`, the `DeviceConfig::devices` order — *not* NCCL AllReduce, whose order varies with G and breaks parity, §12). Parity: **bit-identical across G and to the single-GPU reference** — `memcmp`-proven on **both** the host-staged and the device-resident P2P combine paths, on both datasets (`derived_acc` P=50 + `derived_full` P=768), production `EmulatedFp64{40}` (architecture.md §11.4, §12). **The honest multi-GPU verdict:** on the *precompute* multi-GPU is a **modest throughput layer, NOT the perf win** — it was measured *slower* than single-GPU until the data-bounce was fixed (nsys: a serial D2H/host tail, ~22–74% overlap), and even after the fix the gain is marginal. The real perf win came from getting the result **off the CPU** — see the device-resident note below. The original slowdown (a redundant second full ~7.14 GB Device→Host bounce — `compute_f2_blocks` D2H-copied each partial to host and freed its device buffers, forcing the combine to re-upload, place-add, then D2H again) was diagnosed by nsys (`docs/cleanup/m4.5/why-multigpu-slow.md`) and fixed by the **device-resident combine** (`867a4bf`): per-device compute leaves its partial **resident** (returns a move-only `DevicePartial` handle — no D2H, no free); the combine allocates one root result, D2D-copies the root partial and `cudaMemcpyPeer`s each peer partial straight into its **disjoint** block slice, then does one final D2H. **Multi-GPU's proper home is the Phase-2 fit ROTATION** (thousands of independent models, no combine), not the precompute. **Depends: M4**; **composes with M5** (each GPU streams its own range out-of-core).
- **Device-resident output. ✅ DONE** (`1f80c0c`) — **the real precompute perf win.** The precompute now returns a **DEVICE-RESIDENT handle** (`DeviceF2Blocks` — the result stays in VRAM); the host `F2BlockTensor` is an opt-in `.to_host()`. **The key lesson (`docs/cleanup/m4.5/why-d2h.md`):** the precompute was **HOST-RESULT-BOUND** — ~80% of the old wall was copying the 6.36 GB+ result to the CPU. Getting it off the CPU was the win, **not** multi-GPU: **measured P=512 device-resident ~673 ms vs ~2879 ms bulk-to-host = ~4.3×.**
- **M5 — Out-of-core streaming. ✅ DONE** (`176a07d` tiered output + `c65179f` SNP-tile input). **(a) Adaptive tiered output (NOT mandatory):** the result goes to the fastest tier it FITS, auto-selected from runtime free VRAM/RAM — VRAM-resident (small P, keeps the 4.3×) → host RAM (big box) → disk (laptop). **(b) SNP-tile input streaming:** per-block decode makes the GPU footprint `O(P·tile + P²)` **INDEPENDENT of M** (no `7·P·M` feeder wall). **Result: full-autosome (M=584131, n_block=748 under the AT2 walk) P=2500 COMPLETES on a single 32 GB RTX 5090 in ~51.5 s** (76 GB result streamed, GPU peak ~26 GB bounded), parity memcmp **bit-identical**. **Measured sweep (one 5090, streamed):** P=512 ~3.6 s, P=1000 ~10.4 s, P=1500 ~20.2 s, P=2000 ~34.0 s, P=2500 ~51.5 s. (This supersedes the pre-M5 `docs/cleanup/m4.5/scaling-sweep.md` claim that P=2500 OOMs on every path.)
- **M6 — QC / data-munging front-end.** Merge/harmonize multiple datasets (in-memory plan, no on-disk rewrite, no strand inference), transversions-only option.
- **M7 — On-disk cache + FST.** ADMIXTOOLS-compatible `f2_blocks` store; **FST as a cheap add-on output** of the same pass.
- **Validation:** parity vs ADMIXTOOLS 2 `extract_f2` on a pinned environment; reference-equivalence + golden tests.

### Phase 2 — qpAdm FIT engine (operates on `f2_blocks`)  ✅ **BACKEND FINISHED — golden-gated on the GPU**
> **Done — backend complete.** What §0 once called "THE NEXT REAL WORK (does not exist yet)" is built end-to-end AND **finished**: the qpAdm/qpWave fit chain S3 (f4 from f2) → S8 (model-space rotation) runs **on the GPU**, reading the **device-resident** `DeviceF2Blocks` (small P; never bounced through the host, `why-d2h.md`) and the M5 streamed-tile path for large P. The full design + the milestone build-order table live in **[`docs/design/fit-engine.md`](design/fit-engine.md)**. The public C++ API is `steppe::run_qpadm(...)` (single model), `steppe::run_qpwave(...)` (the standalone rank-sufficiency sweep), and `steppe::run_qpadm_search(...)` (the S8 rotation) in **`include/steppe/qpadm.hpp`**. Validated to **bit/tolerance parity against real-AADR ADMIXTOOLS-2 goldens**; the `CpuBackend` is the native long-double oracle (run under `STEPPE_THOROUGH`).

> **Backend-finish complete (F1–F6, step 1 of the backend-first sequence — `docs/design/fit-engine-finish-punchlist.md`, memory `build-sequence-backend-first`).** The gap audit's 6 FINISH-NOW items — *contract closure, not new math* — all landed, golden-gated on real AADR, CPU/GPU consistent, deterministic, wallclock unchanged:
> - **F1 — missing-block / NA handling (OQ-12)** (`2496a14`). steppe is pairwise-complete (NOT AT2 `maxmiss=0` global intersection), so a pair-block `Vpair==0` *can* occur on sparse AADR and was silently imputed `f2=0` (bias toward 0). Now implements AT2 `read_f2(remove_na=TRUE)`: **DROP** any block with a non-finite / `Vpair==0` pair before the LOO/jackknife (not impute-0), via the single shared host/device predicate `core::pair_block_is_missing` (`src/core/internal/f2_estimator.hpp`) — `CpuBackend` oracle + GPU (`f2_block_keep_kernel`; single-model AND S8 model-batched survivor compaction). A block is dropped only when *partially* covered, so legacy `maxmiss=0` goldens stay **byte-identical** (no-drop identity arm). NEW `golden_fitNA.json` + `test_qpadm_missing_block.cu` (real-AADR `maxmiss=0.99`, a sparse right pop → 1 real dropped block, real `Vpair==0`). NO synthetic data.
> - **F2 — M(fit-5) domain-outcome acceptance test** (`c8fe397`). NEW `tests/reference/test_qpadm_domain.cu` — degenerate REAL-AADR models (collinear left → `RankDeficient`; `fudge=0` singular Q → `NonSpdCovariance`; over-parameterized `dof≤0` → `ChisqUndefined`) asserted as **status values** (no crash / NaN) on BOTH `CpuBackend` + `CudaBackend`.
> - **F3 — `Status::ChisqUndefined` + `dof≤0` guard** (`ffdcba2`). New enumerator in `include/steppe/error.hpp` + a `dof≤0 ⇒ ChisqUndefined` guard on the HOST (`qpadm_fit.cpp`) AND the CUDA model-batched path (`cuda_backend.cu`) — was leaking NaN `p` with `status=Ok`.
> - **F4 — REAL AT2 `qpwave()` golden** (`6481dfa`). Pinned `tests/reference/goldens/at2/golden_qpwave.json` (admixtools 2.0.10 / R 4.3.3, real AADR `v66.p1_HO`) + NEW `test_qpwave_parity.cu` gating the first-class `run_qpwave` entry on BOTH backends (its no-target-prepend / `left[0]`-is-reference semantic was previously untested).
> - **F5 — remove the dead public `QpAdmOptions::constrained` field** (`e8430a2`). API hygiene: the field was never read in any solve path; non-negative *constrained* weights is a deliberate step-3 feature, not a backend item.
> - **F6 — widen the G1==G2 determinism memcmp** (`360e386`). `test_qpadm_rotation.cu` now compares the FULL `QpAdmResult` (`z`/`dof`/`est_rank`/`rank_*`/`rankdrop_*`/`popdrop_*` — previously a subset); no nondeterminism revealed.
>
> Default `ctest` is now 42 tests (`qpadm_domain` is #19). All gated against the genuine R `admixtools` goldens (`golden_fit0`/`fit1_NRBIG`/`rot`/`qpwave`/`fitNA`); default ctest + `STEPPE_THOROUGH` oracle green. **The fit-engine backend is now DONE.**

The fit milestones (all **BUILT** — see the milestone table at the foot of `fit-engine.md`):
- **M(fit-0) — contract + oracle scaffold.** The frozen `include/steppe/qpadm.hpp`; the `core/qpadm/` skeleton; the `CpuBackend` reference path; GPU-free, layering/allocation grep gates pass.
- **M(fit-1) — f4 + single GLS fit, ONE model.** S3 `assemble_f4` + S4 weighted `Q` (via `Vpair`) + S6 `opt_A`/`opt_B` ALS + the constrained weight solve, native FP64. **Correction to the old spec §5:243 "single Cholesky" framing:** AT2's weight fit is an **alternating-least-squares loop (default 20 iters, `fudge=1e-4` ridge)** refining a rank-`r` `X ≈ A·B` factorization; only the final constrained `(r+1)×(r+1)` solve is a single Cholesky (see `fit-engine.md` §0). Gate: `weight` matches the AT2 `qpadm()` golden at the tight tier.
- **M(fit-2) — rank test (S5) + p-values (qpWave/qpAdm).** The rank sweep `r=0..rmax` (`χ²`/`dof`/`p`), the AT2 `res$rankdrop` nested table and `res$popdrop` leave-one-left-source-out table, and `run_qpwave`. **Runs ON THE GPU** (f2 resident, on-device SVD/ALS/χ²). A **HYBRID dispatch**: the bit-parity envelope (nl≤5, nr≤10, r≤4) runs the on-device **Jacobi** small path; outside it (e.g. the NRBIG nr=39 case) runs the **cuSOLVER** large path (`gesvdj` for dims ≤32, `gesvd` for >32, native FP64). Gate: green vs `golden_fit0.json` `rankdrop`/`popdrop` (9-pop, nr≤32) AND vs `golden_fit1_NRBIG.json` (nr=39 large path); GPU == CpuBackend oracle.
- **M(fit-3) — block-jackknife SE (S7) + the opt-in `JackknifePolicy`.** `n_block` leave-one-out weight re-fits → `cov(wmat)` → `se`/`z`, batched over the block axis. Plus the rotation SE policy **`JackknifePolicy { None=0, FeasibleOnly=1, All=2 }`** on `QpAdmOptions` (default `All` ⇒ goldens unchanged; purely additive): pay the expensive LOO SE only for the feasible survivors worth reporting (≈64% of real rotation models are infeasible). Gate: green on the real 84-model rotation; the cheap point estimate is memcmp-identical across all three modes; survivor SE bit-identical to `All`.
- **M(fit-4) — CUDA backend, single GPU.** `CudaBackend` overrides of `assemble_f4`/`jackknife_cov`/`rank_test`/`gls_weights`; cuSOLVER + a search-stream pool on `PerGpuResources`; **EmulatedFp64{40}** on the S4 covariance SYRK/GEMM (native fallback via the one `emulation_honorable` predicate); deterministic cuSOLVER. Gate: GPU == CPU oracle at the `X`/`Q`/`w`/`χ²`/`se` seams; the AT2 goldens re-pass on the GPU path.
- **M(fit-5) — domain outcomes.** Rank-deficient / non-SPD-covariance / χ²-undefined returned as **per-model `status` values** (`RankDeficient`, `NonSpdCovariance`, `ChisqUndefined`), never exceptions — essential for the rotation, where one pathological model must not abort the batch. **Closed in the backend finish:** the `ChisqUndefined` enumerator + the `dof≤0` guard (F3 `ffdcba2`) and the acceptance test asserting all three outcomes on both backends (F2 `c8fe397`).
- **M(fit-6) — S8 model-space ROTATION.** `run_qpadm_search` (device-resident + host-oracle); `model_search_core` (the count-balanced model→device tiling) + `model_search` orchestrator; the **genuinely batched device path** `CudaBackend::fit_models_batched` — buckets models by `(nl,nr,r)` and fits each bucket of B models in ONE batched dispatch (batched f4 gather over the resident f2, `cublasDgemmStridedBatched` covariance, `potrfBatched`/`potrsBatched` SPD inverse, model-batched rank-sweep/weight/χ²/popdrop kernels, a `(model,block)`-grid LOO-SE kernel), NOT a per-model host loop (`batched_dispatch_count()` ≪ model count proves it). Gate: green vs the real-AADR 84-model rotation golden `golden_rot.json` (weights rtol 1e-5, f4rank EXACT, feasibility decision-match); **G=1 vs G=2 bit-identical and identically ordered**.

**Validation gate (real AADR only).** The fit outputs are gated against goldens generated by the genuine R `admixtools` package on real data — `tests/reference/goldens/at2/`: `golden_fit0.json` (9-pop single model), `golden_fit1_NRBIG.json` (nr=39 large path), `golden_rot.json` (84-model rotation), `golden_qpwave.json` (the standalone `qpwave()` rank sweep — F4), and `golden_fitNA.json` (the `maxmiss=0.99` missing-block drop — F1), all **admixtools 2.0.10 / R 4.3.3 / v66.p1_HO** (PACKEDANCESTRYMAP Human Origins panel; blgsize 0.05, fudge 1e-4, boot=FALSE). Run with the harness binaries `build-rel/bin/test_qpadm_parity <goldens-dir>`, `build-rel/bin/test_qpadm_rotation <goldens-dir>`, `build-rel/bin/test_qpwave_parity <goldens-dir>`, `build-rel/bin/test_qpadm_domain <goldens-dir>`, and `build-rel/bin/test_qpadm_missing_block <goldens-dir>` (default ctest ~1 min, 42 tests; `STEPPE_THOROUGH=1` adds the CpuBackend oracle + NRBIG full SE, ~56 s). See [`docs/RUN-GUIDE.md`](RUN-GUIDE.md) for the verified build/test commands.

> **Multi-GPU rotation is DEFERRED — RUN SINGLE-GPU (`TODO(multigpu-host-bounce)`).** On the consumer 5090s there is no GeForce P2P, so the one-time `f2` replication is a **~8.72 GB / ~3.8 s HOST BOUNCE** (`to_host`→re-upload). At 9086 real models that capped multi-GPU at only **G2/G1 ≈ 1.21×** (no 1.5× crossover). Parity still holds (84/84 real models == AT2; G=1==G=2 bit-identical), but **there is no multi-GPU rotation speedup to claim** — the payoff needs P2P hardware (RTX PRO 6000) or per-device precompute (`fit-engine.md` §1.6; SPEC §11.4). The throughput we *do* report is real-AADR single-GPU: the validated 84-model set at 351 (G=1) models/sec on box5090 (the earlier synthetic scale-N number was RETRACTED, `[[real-data-only-all-results]]`, commit 2a0c020).

Deferred/optional (after the fit, not on the critical path): the TurboQuant-L2 rotation screen (`docs/research/turboquant-l2-experiment.md`).

### The big refactor — ✅ COMPLETE  (`docs/cleanup/bigrefactor/`)
The codebase-wide cleanup that landed alongside the fit engine is done and audited: **3 HIGH** fixes (`block_sink` silent-corruption fail-fast `9dbc610`; `kQpMax` single-source `3beff6d`; the `opt_A`/`opt_B` ALS-ridge solve single-source `ed6cc44`), **14 MED** groups, **9 LOW** groups (held to `docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md`), and **2 device dedups** (`block_sink` StagingRing `b1bd620`; `qpadm_fit_kernels` twin-collapse `25c882a`). All **golden-gated on real AADR, bit-identical, wallclock unchanged** (qpadm_parity 0.86 s default / 27.3 s thorough; rotation 3.42 s; no perf regression). The precision policy is consistent (emulated-FP64 default + native fallback via the one `emulation_honorable` predicate; `precision-policy-consistency.md` = 0 deviations). Skipped by decision: the host/device pointer-wrapper tag-type and blanket `const __restrict__`.

### Phase 2.5 — productization & remaining modalities  ← **the honest "what is left"**
The fit-engine **backend is now finished** (Phase 2 above), but it is **not yet a product**. The build order follows the **backend-first sequence** (memory `build-sequence-backend-first`): step 1 (finish the backend) is **DONE**; step 2 (productize what exists) is **THE next work**; step 3 (new statistics, each shipped with its own access surface) follows. Prioritized from `docs/research/desirable-features-survey.md`:

- **STEP 2 (THE NEXT WORK) — CLI + Python bindings + tidy (CSV/Parquet) export, for the EXISTING backend.** Expose what is already built and finished — qpAdm fit / `run_qpwave` / the S8 model-space rotation / the f2 precompute. **Neither a CLI nor bindings exists yet** — the architecture `app/` is *planned*, not built (no `app/` or `bindings/` dir, no nanobind/CLI11 in CMake, no `main()`). nanobind Python per ADR-0002; `app/` is the CLI home. This is the delivery surface that makes the finished engine usable with no R/CRAN dependency chain. Effort: MEDIUM.
- **STEP 3 — standalone f-stat entry points, EACH shipped WITH its own CLI/bindings — f4 / D-stat / f3 / outgroup-f3 / f4-ratio / qpDstat.** Exposing batched all-quartet / all-trio scans over resident `f2_blocks` is the highest-ROI new statistic (competes directly with Dsuite). Per the build sequence, each new statistic ships **together with** its access surface (not backend-only). Effort: LOW–MEDIUM (math) + the step-2 binding pattern.
  - **f4 — DONE** (`271e302`): standalone `run_f4` (`src/core/qpadm/f4.cpp`), `steppe f4` CLI (`cmd_f4.cpp`), `steppe.f4` binding, `cli_f4` ctest gate, fixture-matched `golden_fit0_f4_readf2.csv`. Reuses `assemble_f4_quartets` + `jackknife_cov` (no dup).
  - **f3 / outgroup-f3 — DONE** (`cfa0d9d`): standalone `run_f3` (`src/core/qpadm/f3.cpp`) = the three-slab triple identity `0.5*(f2(C,A)+f2(C,B)-f2(A,B))`, `steppe f3` CLI (`cmd_f3.cpp`), `steppe.f3` binding, `cli_f3` ctest + 3 pytest gates, fixture-matched `golden_fit0_f3_readf2.csv` (28 outgroup-f3 + 8 admixture-f3). ONE new math seam (`assemble_f3_triples`: CpuBackend oracle + CUDA gather kernel); jackknife/LOO/z→p/CLI/binding all reused verbatim (no dup). Bit-tight parity (est/se/z/p rtol 1e-6; golden independently fixture-reverified to ~1e-14).
  - **NEXT: f4-ratio** (admixture proportion `f4(A,O;X,C)/f4(A,O;B,C)`), then D-stat / qpDstat.
- **qpfstats — cross-statistic smoothing solve for heavy missingness.** The documented AT2 remedy for aDNA's #1 false-positive driver; an over-determined regularized least-squares over the whole f2/f3/f4 system — a large batched solve that lands on steppe's existing cuSOLVER + emulated-FP64 seam. Not built. Effort: MEDIUM–HIGH.
- **Precompute M6 / M7.** M6 multi-dataset merge/harmonize (in-memory plan, no on-disk rewrite); M7 ADMIXTOOLS-compatible on-disk `f2_blocks` cache + FST as a cheap add-on of the same pass. Both pending.
- **Big-bet modalities (deferred, roadmap-fit not near-term):** DATES / admixture dating; qpGraph / admixture-graph search; genotype imputation. None built (`desirable-features-survey.md` §3).

> **Honesty guardrail.** Steps 2–3 are NOT built — do not claim a CLI, Python bindings, or standalone f-stat / qpfstats / DATES / qpGraph entry points exist. The fit-engine *backend* IS finished (F1–F6 git-verifiable: `e8430a2`…`2496a14`); productization is not. steppe's throughput figures are its own internal numbers, never benchmarked against AT2, and always on real AADR (never synthetic).

### Phase 3 — Interfaces
Folded into Phase 2.5 **step 2** above (now THE next work): CLI; nanobind Python bindings; scikit-build-core wheels.

---

## 4. Magic-number → config inventory  ("fix the numbers")

Every constant found in the spike, and where it lives in production. **No literal may survive M0 except true mathematical constants** (e.g. the `2` in `a²−2ab+b²`).

| spike literal (file) | meaning | production home |
|---|---|---|
| `q*(1-q)/198.0` (f2_timing) | hardcoded `N−1` for N=200 | **bug-shaped smell** — must use per-SNP `N`; never hardcode sample size |
| `max(N-1.0, 1.0)` (spike:709) | het bias-correction denom floor | shared `__host__ __device__` `f2_estimator` primitive |
| `{32,40,48}`, `{24,…,53}` | Ozaki mantissa-bit sweeps | `Precision::EmulatedFp64{ mantissa_bits }`, **default 40** |
| `dim3 block(16,16)` (spike:311) | kernel launch geometry | `core/internal/launch_config.hpp` (`grid_for`, occupancy) |
| `10.0 *` (spike:474) | accuracy verdict threshold | test tolerance config (`golden_rtol` tiers) |
| `1e-300`, `1e-12` | divide/relerr floors | named `kRelFloor` / `kAbsFloor` constants |
| `vp > 0.0 ? … : …` | Vpair==0 guard | keep (exact-integer count); `>= 1.0` is equivalent (cosmetic) |
| `(size_t)i + (size_t)j*P` | index cast | keep `size_t` (free, mandatory above P≈32k); *not* removed |
| block size `0.05` Morgans | jackknife block size | `RunConfig::block_size_cm` (default 5.0) |
| MAF / geno / mind | filters | `FilterConfig` (M2) |
| TGENO header `48`, `ceil(nsnp/4)` | packed-format constants | `io` format constants, derived from the header parse |
| device id, stream counts | resources | `DeviceConfig` (`devices`, `stream_count`, …) |

Precision config (extends `architecture.md` §9 — the new bit is the mantissa-bits knob):
```cpp
enum class PrecisionKind { Fp64, EmulatedFp64, Tf32 };
struct Precision {
    PrecisionKind kind = PrecisionKind::EmulatedFp64;
    int mantissa_bits  = 40;   // fixed-slice Ozaki only; 32=fast/8.6e-9, 40=native-grade, 48=exceeds.
                               // DYNAMIC mantissa is intentionally NOT offered (the parity trap, §0).
};
```

---

## 5. Cross-cutting standards (the "structure on point")

- **RAII everywhere** — no raw `cudaMalloc`/handle/workspace outside an owning wrapper (CI grep gate, `architecture.md` §2).
- **Layer split, not a feature folder** — f2 code is distributed `device`/`core`/`api` per §4; a flat `f2/` directory is forbidden because it breaks the compiler-enforced layering (CUDA is `PRIVATE` to `steppe_device`).
- **DRY helpers** — one `CUDA_CHECK`, one loader, one GEMM-arg wrapper, one `f2_estimator` primitive shared by CPU-ref and GPU.
- **CPU-reference oracle** — every statistic-bearing kernel diffed against an obviously-correct scalar/long-double reference; it validates *results*, not structure.
- **Precision is typed config** — fixed-slice Ozaki bit-count per the §4 table; dynamic-mantissa rejected.
- **No synthetic-data precision benchmarks** — ever (§0 cautionary tale).
- **No magic numbers** — see §4.
- **Test → commit between successes** — after each green milestone/step, commit; never commit a red or unverified state (see §6 for the message contract).

---

## 6. Definition of done (per milestone)

A milestone is done when: builds clean on the CUDA-13 × sm_120 matrix with warnings-as-errors; layering/RAII/allocation-allowlist grep gates pass; the new kernel has a CPU-reference-equivalence test passing at the tight tier; magic numbers are config/constants; compute-sanitizer (memcheck+racecheck) clean; and (for cache milestones) parity vs ADMIXTOOLS 2 within the documented tolerance tiers.

> **AT2-parity gate — now LIVE for the fit engine.** R + admixtools 2.0.10 (R 4.3.3) has since been installed and pinned goldens generated on real AADR (`tests/reference/goldens/at2/`, recording R/AT2 version, `blgsize`, `fudge`, `boot`, seed per §12). The **Phase-2 qpAdm fit outputs M(fit-0..6) are gated against these AT2 goldens** (`golden_fit0`/`golden_fit1_NRBIG`/`golden_rot`) — the GPU path matches them to bit/tolerance parity, with the CpuBackend as the native oracle under `STEPPE_THOROUGH` (§3, Phase 2). For the **precompute M1–M5**, the operative gate was (and remains) the **CPU reference oracle + property identities + internal-consistency + bit-identical-parity checks on real AADR** (never synthetic, §0): M1–M3 met it (decode reproduces the numpy oracle bit-for-bit; filters proven by drop-equals-mask + scalar-oracle exactness; block rule by the 748/711 (full-v66/autosome) AT2-walk internal-consistency counts, AT2 cache parity target 709 — see `docs/research/block-partition-at2.md`); M4.5 + M5 by memcmp bit-identity. AT2 `extract_f2` parity remains the hard acceptance criterion for the on-disk cache (M7), wired when M7 lands.

### Commit discipline — test → commit between successes
Every milestone (and every meaningful green step inside one) ends with a **commit**, taken only once its tests/verification pass on the box. Never commit a red or unverified state. Each commit message must carry:
1. **What changed and why** — the substantive summary.
2. **Timing / benchmark numbers** where relevant, with the configuration — e.g. `f2 EmulatedFp64{40b} = 244 ms @ P=2416,M=100k (11.3× vs native FP64); 32b = 190 ms (14.5×)`.
3. **Exact commands to build and run/reproduce** — the `nvcc`/CMake invocation, the test/run command, and the box (`ssh -i ~/.ssh/id_vastai -p <port> root@<host>`) when remote, so anyone can re-verify.

Per-milestone work goes on its own branch; messages end with the project's `Co-Authored-By` trailer. Code is authored locally and **verified on the remote box before the commit** (see the dev-process rule: nothing is built/run locally).
