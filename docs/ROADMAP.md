# `steppe` — Roadmap

Companion to [`architecture.md`](architecture.md). That doc says **how** the system is structured; this one says **in what order** we build it and **what is already proven**. Read both before writing code.

---

## 0. Status — what is validated (2026-06-15)

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

### Phase 1 — `f2_blocks` cache (precompute engine)  ← current focus
- **M0 — Structure lift (no new functionality).** Port the validated 3-GEMM kernel + fixed-slice-Ozaki precision into the architecture layout: kernel → `src/device/cuda/f2_block_kernel.cu`; orchestration → `src/core/fstats/f2_from_blocks.cpp`; the f2 estimator as a shared `__host__ __device__` primitive (so CPU-ref and GPU can't diverge); promote **every** magic number to `config.hpp`/named constants (§4); dedup `CUDA_CHECK`/loader/GEMM-args into the §8 helpers; wrap all allocations in `DeviceBuffer<T>`; add the CPU-reference-equivalence test. Gate: GPU matches CPU reference at the tight tier; no magic numbers; no duplication.
- **M1 — Genotype decode + allele frequency. ✅ DONE** (`150bfb3`). New `src/io/` LEAF (TGENO/GENO header parse + `.geno`/`.snp`/`.ind` readers + plain tile struct) + shared `__host__ __device__` decode primitive (`core/internal/decode_af.hpp`) + `ComputeBackend::decode_af` seam (CPU-reference + CUDA `decode_af_kernel.cu` segmented reduction). **Correction:** the real file is **TGENO individual-major** with the **raw-value 2-bit mapping** (0/1/2 copies, 3 = missing) — *not* the binary mapping. Reproduces the numpy oracle **bit-for-bit (max|Δ| = 0 for Q/V/N)** on `derived_acc` (P=50, M=100k); decoded Q/V/N → `compute_f2` = M0's 1.085e-11. **No pseudo-haploids in v66 HO** (all diploid, `N = 2×count`); ploidy is a documented metadata parameter (default 2, never inferred from genotypes). 67.6 ms = 8.6e9 SNP·sample/s; compute-sanitizer clean.
- **M2 — Missingness + filters. ✅ DONE** (`1bbbad4`). Host-pure predicates in `src/io/filter/` (the M2 analogue of `f2_estimator.hpp`): MAF / `geno` / `mind` / include-exclude + flag-gated monomorphic / transversions-only / autosomes-only, all from `FilterConfig`. Pairwise-complete (V/N) is the default and the parity path — **no new f2 math, no kernel**. Proven on real AADR: no-op when default (f2 bit-identical to unfiltered); **drop-equals-mask** (dropping SNP columns ≡ zeroing `V`) bit-identical; keep-masks integer-exact vs an independent scalar oracle across 7 configs. **AT2 autosome parity = chr 1–22 → 719 blocks** (vs chr 1–23 = 756, all = 757); `is_autosome` = 1–22. Strand-ambiguous = self-complementary A/T·C/G palindromes (zero in HO); the 18,534 transversions are a separate flag. Convention flags pinned for the AT2 golden: MAF pooled-across-samples; `geno` sample-axis missing fraction (AT2 `maxmiss` is population-axis).
- **M3 — SNP→block assignment. ✅ DONE** (`f7f31c6`). `core/domain/block_partition_rule.{hpp,cpp}`: scalar `block_of` (kept) + `assign_blocks(chrom[], genpos_morgans[], blocksize) → BlockPartition{block_id[], n_block}` (one file-order pass: per-chromosome reset + dense renumber of *occupied* bins, interior empties absorbed) + `block_size_cm_to_morgans` (the single cM↔Morgan site). Host-pure, shared by `io` and kernels. Real-AADR: all 584,131 SNPs (chr 1–24) → **757 blocks**; the rule is deliberately **filter-agnostic**, so autosome (chr 1–22) parity = **719** (the chromosome-range drop is `io` territory — see M2). Negative chr17 positions + all-zero chr24 handled.
- **M4 — Per-block f2. ← NEXT (spike-gated).** The 3-GEMM kernel at **fixed-slice Ozaki (default 40-bit)**, **batched over blocks** (`cublasDgemmStridedBatched`) → `f2_blocks [P × P × n_block]`, carrying per-block `Vpair` (the S4 jackknife weight). M0 measured *one big GEMM*; M4 batches ~700 *small* per-block GEMMs — a different arithmetic-intensity regime, so M4 **opens with a throwaway spike** (`experiments/`) confirming the Ozaki 40-bit speedup + accuracy hold per-block (strided-batched vs loop-of-GEMMs vs one-big-then-rebin) on **real AADR**, before the design is committed. Validated against the CPU oracle per-block.
- **M4.5 — Single-node multi-GPU precompute (shard + parity combine).** Shard SNP work across `DeviceConfig::devices` (≥2 GPUs — the box has 2× RTX 5090): SPMG — one host thread + per-device stream per GPU, `cudaSetDevice`, opportunistic peer access. Each device computes a **full-shape partial** `f2_blocks` + `Vpair` over its SNP range/tiles; the G partials are summed **once, host-side, in fixed device order** (`g = 0..G−1`, the `DeviceConfig::devices` order — *not* NCCL AllReduce, whose order varies with G and breaks parity, §12) and broadcast back. Parity: **bit-identical across G and to the single-GPU reference** (architecture.md §11.4, §12). **Depends: M4** (basic SNP-range shard needs only the per-block kernel); **composes with M5** (each GPU streams its own range out-of-core).
- **M5 — Out-of-core SNP-tile streaming.** Stream SNP tiles so VRAM holds only the current tile + the resident `f2_blocks` accumulator.
- **M6 — QC / data-munging front-end.** Merge/harmonize multiple datasets (in-memory plan, no on-disk rewrite, no strand inference), transversions-only option.
- **M7 — On-disk cache + FST.** ADMIXTOOLS-compatible `f2_blocks` store; **FST as a cheap add-on output** of the same pass.
- **Validation:** parity vs ADMIXTOOLS 2 `extract_f2` on a pinned environment; reference-equivalence + golden tests.

### Phase 2 — Fit engine (operates on cached `f2_blocks`)
Block jackknife → SEs/covariance; f3/f4 derivation; qpWave rank test (SVD); qpAdm GLS + p-values; multi-GPU model-space search.

### Phase 3 — Interfaces
CLI; nanobind Python bindings; scikit-build-core wheels.

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

> **AT2-parity gate is currently deferred (prerequisite not yet installed).** As of 2026-06-16 the GPU box has **no R and no `admixtools` package**, so ADMIXTOOLS 2 goldens do not exist yet. Until R + admixtools is installed and pinned goldens are generated (its own tracked task — small fixture, recording R version / `RNGkind` / AT2 version / `blgsize` / `boot` / seed per §12), the **operative gate for M1–M4 is the CPU reference oracle + property identities + internal-consistency checks on real AADR** (never synthetic, §0). M1–M3 met this gate (decode reproduces the numpy oracle bit-for-bit; filters proven by drop-equals-mask + scalar-oracle exactness; block rule by the 757/719 internal-consistency counts). AT2 parity remains a hard acceptance criterion for the on-disk cache (M7) and is wired in the moment the toolchain is present.

### Commit discipline — test → commit between successes
Every milestone (and every meaningful green step inside one) ends with a **commit**, taken only once its tests/verification pass on the box. Never commit a red or unverified state. Each commit message must carry:
1. **What changed and why** — the substantive summary.
2. **Timing / benchmark numbers** where relevant, with the configuration — e.g. `f2 EmulatedFp64{40b} = 244 ms @ P=2416,M=100k (11.3× vs native FP64); 32b = 190 ms (14.5×)`.
3. **Exact commands to build and run/reproduce** — the `nvcc`/CMake invocation, the test/run command, and the box (`ssh -i ~/.ssh/id_vastai -p <port> root@<host>`) when remote, so anyone can re-verify.

Per-milestone work goes on its own branch; messages end with the project's `Co-Authored-By` trailer. Code is authored locally and **verified on the remote box before the commit** (see the dev-process rule: nothing is built/run locally).
