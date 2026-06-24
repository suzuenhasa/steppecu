# Host-Compute Audit — every production path swept for CPU work violating the GPU-bound mandate

**Trigger.** The qpfstats block-jackknife (`matrix_jackknife_est_col`) was found running on the
host — a long-double per-block loop called **per popcomb** (~305k combos × 711 blocks at 40 pops
≈ 217M host iterations, GPU 50% idle / the 100-0 alternation), despite the standing GPU-only /
keep-on-device mandate. It passed the golden gate because the 9-pop golden is tiny (666 combos ×
666 blocks). This audit sweeps **every** production (`CudaBackend`) path for the same class of
violation before it bites at production / 40+ pop / sweep scale.

**Audited at** branch `phase2-fit-engine` @ `38674b3` (== main lineage). All file:line citations
were hard-verified against the real source at this commit — not the per-area auditors' notes.

**The offender pattern (a VIOLATION).** Host-side floating-point / statistical compute in the
production path (`src/core`, `src/app`, or `run_*` regardless of backend) that **scales with the
data** (M SNPs / n_block / N items / npopcomb / npairs / n_restart) — reductions, block-jackknife,
covariance, GLS, regression, normalization, het-correction, per-item/per-block/per-SNP/per-combo
arithmetic — that should be a `ComputeBackend` GPU kernel but is on the host, serializing with or
instead of the GPU.

**Exempt (not flagged).** (1) `CpuBackend` `src/device/cpu/*` — the test-only parity oracle;
(2) CLI/arg/config parsing; (3) one-time small integer index/setup-array builds; (4) unavoidable
genotype/f2 **read** orchestration (but the decode/2-bit-unpack/het-correction is flagged if it is
a slow host hot loop that competes with the GPU); (5) tiny final result formatting/emit.

**The discriminator.** Host arithmetic that scales with a data dimension AND serializes
against/replaces the GPU = VIOLATION. Integer setup / IO / formatting = OK.

---

## The exemplar (already being fixed — NOT counted as an active offender)

`src/core/stats/qpfstats.cpp` — the genotype-path joint-f2 smoother. The original host
`matrix_jackknife_est_col` + `f2blocks_pair_est` per-comb/per-pair long-double loops are **GONE
from the production driver**: `run_qpfstats` now calls the fused on-device seam
`be.qpfstats_blocks_smooth` (verified at `qpfstats.cpp:324`), which folds the reduce → per-comb
LOO jackknife → smoothing solve → per-pair recenter jackknife on-device, keeping
numsum/cnt/ymat/y/b resident in VRAM (the ~217M host iters + the ~5.2 GB D2H eliminated). One
per-area auditor flagged it as an active CRITICAL — that auditor read a **stale tree**; at `38674b3`
the host loops are removed and only the tiny scatter (`qpfstats.cpp:341-371`) remains.

This is the template every other offender below should follow: lift the host per-item/per-block
jackknife into a batched `ComputeBackend` virtual, native FP64 (the §12 cancellation carve-out),
inputs already device-resident.

---

## CRITICAL — bites at production / 40+ pop scale, like the qpfstats jackknife

These are the genotype-decode front-end host loops. They are the **highest-impact** active
offenders: they run on **every decode** in four tools (extract_f2, dstat, dates, qpfstats), scale
M×P, and gate the ~14 s 40-pop decode. The decode **arithmetic** is correctly on-device — but it is
bracketed by a forced D2H + host per-SNP reductions on the data the GPU just produced.

> **STATUS (Stage 1 landed `9ad33d9`, regime-A autosome-only: qpfstats + dstat).** The
> device-resident decode seam is built: `decode_af_compact_autosome` (cuda_backend.cu) keeps
> Q/V/N RESIDENT (no full D2H), runs an on-device per-SNP **autosome** keep-mask (INTEGER-EXACT
> chr in [1,22]), and stream-compacts Q/V onto the kept axis with `cub::DeviceSelect::Flagged`
> (1-D chrom/genpos, file-order preserved — the same idiom as `run_fstat_sweep_device`) +
> `cub::DeviceScan::ExclusiveSum` + a scan-keyed `[P×M]` column gather. `dstat_block_reduce` /
> `qpfstats_blocks_smooth` gained `DeviceDecodeResult` overloads that read the resident Q/V (no
> H2D re-upload). nsys on the 40-pop qpfstats confirms the ~1.1 GB Q/V/N D2H is GONE (D2H total
> 11.4 MB) and the Q/V H2D re-upload is GONE; the only remaining big H2D is the io-side .geno tile
> upload (out of scope). All genotype goldens held (STEPPE_THOROUGH 61/61; CpuBackend==CudaBackend).
> **Done for qpfstats + dstat (regime A).** STILL OPEN: **extract_f2 (regime B)** — the
> FP-sensitive pooled-MAF / maxmiss / allele-class filter + N compaction (Stage 2), and **dates**
> — the two-row gather + target-record device repack (Stage 3, M5). Those host loops remain.

| # | area | file:line | what | in_prod | scales_with | bites_at | severity |
|---|------|-----------|------|---------|-------------|----------|----------|
| C1 | decode front-end | `src/device/cuda/cuda_backend.cu:1237-1242` (D2H of `out.q/out.v/out.n` + `cudaStreamSynchronize`) | `decode_af` finalizes Q/V/N on the GPU then **unconditionally D2H copies the full [P×M] Q+V+N triple** to host `std::vector<double>` + a blocking sync. At 40 pops × ~1.15M SNPs ≈ 3·40·1.15M·8 B ≈ **1.1 GB D2H** per decode, round-tripping VRAM→host→VRAM (the result is the input to the f2/dstat/dates/qpfstats GPU compute that follows). Violates keep-on-device; structural cause of the ~14 s decode. Consumed at `extract_f2_core.cpp:148`, `dstat.cpp:234`, `dates.cpp:251`, `qpfstats.cpp:258`. | yes | P×M | 40+ pop (1.1 GB D2H/decode); invisible at 9-pop golden | **DONE (qpfstats+dstat, `9ad33d9`)** / open extract_f2+dates |
| C2 | decode front-end | `src/app/extract_f2_core.cpp:178-191` (maxmiss); `src/io/filter/snp_filter.cpp:79-107` (`derive_per_snp_summary`) | Two host hot loops over **M SNPs × P pops** on freshly-GPU-computed data: (a) per-SNP pop-coverage count of `dec.n==0` flipping `keep[s]`; (b) per-SNP pooled-MAF / missing-frac reduction `Σ_pop Q·N / Σ_pop N` feeding `snp_keep_decision`. Per-SNP-across-pops statistical reductions on the host, serial after the C1 D2H. (`snp_filter.cpp:163` notes `snp_keep_decision` was already made a device-shareable pure primitive "for the future M4.5 device kernel" — never built.) | yes | P×M | 40+ pop (~46M host iters/loop); golden ~10M hides it | **CRITICAL** — qpfstats/dstat AUTOSOME keep now on-device (`9ad33d9`); the FP-sensitive pooled-MAF/maxmiss (extract_f2 regime B) STILL OPEN (Stage 2) |

**Fix (one piece, serves C1+C2 and the MEDIUM consumers below).** Add a **device-resident decode
seam** mirroring the existing `DeviceF2Blocks` / `compute_f2_blocks_device` pattern: `decode_af`
leaves Q/V/N resident in VRAM and returns an opaque handle; a per-SNP keep-mask kernel (one
thread/warp per SNP reduces across P pops to emit pooled-MAF / missing-frac / pop-missing-fraction,
applies the already-pure `snp_keep_decision`, writes the keep bitmask on-device) + a CUB
`DeviceSelect`/gather stream-compaction produce the compacted resident Qk/Vk/Nk that
`compute_f2_blocks` / `dstat_block_reduce` consume directly. No D2H of the [P×M] arena at all.

---

## MEDIUM — bites at large (many-item / large-tail) scale, not at the golden

These are the **prime-suspect jackknife family** (the genotype-/ratio-path siblings of the
now-fixed qpfstats jackknife — same "host long-double per-item-per-block loop" shape, never lifted
to a backend virtual), the per-tool lockstep SNP-subset copies (the consumer side of the C1/C2
fix), and the dates host regressions. They are MEDIUM not CRITICAL because they are bounded by the
**user's explicit item list** (no C(P,k) combinatorial blow-up) or by a small bounded axis
(n_chrom≤22, n_target~100, the large-model tail), not by npopcomb.

| # | area | file:line | what | in_prod | scales_with | bites_at | severity |
|---|------|-----------|------|---------|-------------|----------|----------|
| M1 | f4-ratio | `src/core/qpadm/f4ratio.cpp:79-155` (`ratio_jackknife`), driven by the per-tuple loop `f4ratio.cpp:232-240` | The per-RATIO weighted block-jackknife wholly on the host: **3 long-double passes over all n_block** per tuple (survivor mask + totnum/totden; `est` diffsum/wmean; xtau var) producing alpha/se/z for one f4-ratio tuple. `assemble_f4_quartets` already D2H'd `x_blocks`/`x_loo` to host; the jackknife then runs on the CPU, GPU idle. The exact qpfstats `matrix_jackknife_est_col` pattern, per item. `build_tuple_names` (`cmd_f4ratio.cpp`) is row-aligned `--pop1..--pop5` / `--pops`×5 — **no sweep**, so N is the explicit tuple list. | yes | N (tuples) × n_block (~711) × 3 passes | large explicit f4-ratio screens (thousands × 711 = millions of host long-double iters); golden has few tuples | **MEDIUM** |
| M2 | qpDstat | `src/core/stats/dstat.cpp:70-157` (`dstat_jackknife`), driven by `dstat.cpp:293-300` | The num/den block-jackknife: per quadruple, **~4-5 host long-double passes over n_block** (survivor Σcnt, est_to_loo tot_num/tot_den, per-block R_b + accumulators, xtau var). Host loop `for k in 0..N-1` after the on-device `dstat_block_reduce` returns [N×n_block] numsum/densum/cnt. The file header itself names "the f4ratio.cpp ratio-jackknife FAMILY". **The `--all-quartets` D sweep does NOT reach here** (`cmd_qpdstat.cpp:246` routes to the on-device `run_fstat_sweep`); only the explicit normalized-D path. | yes | N (explicit quadruples) × n_block × ~5 passes | thousand-quadruple aDNA panels × 711 blocks; golden has few quads | **MEDIUM** |
| M3 | qpDstat | `src/core/stats/dstat.cpp:245-255` (autosome keep-mask + lockstep Qk/Vk subset) | Host per-SNP loop over all M decoded SNPs: chrom∈1..22 filter, then inner per-pop copy of `dec.q`/`dec.v` into Qk/Vk + chrom/genpos. Runs after the C1 D2H, rebuilds the resident Q/V on the CPU before re-uploading H2D for `dstat_block_reduce` (GPU decode → D2H → host repack → H2D → GPU reduce). Twin of `qpfstats.cpp:268-278`. | yes | M×P | real AADR (M~1.2M): M×P host copy + the D2H/H2D round-trip; negligible at golden | **DONE (`9ad33d9`)** — on the CUDA path the autosome keep + Q/V compaction are on-device (`decode_af_compact_autosome`); the host loop is now the CpuBackend-oracle-only branch |
| M4 | decode front-end | `src/app/extract_f2_core.cpp:212-226` (+ twins `dstat.cpp:245-255`, `qpfstats.cpp:268-278`, `dates.cpp:280-292`) | The lockstep SNP-subset stream-compaction: host loop over M SNPs copying each kept SNP's P-vector column `dec.q/v/n` into a dense kept-axis array — the same data just D2H'd, a second ~1 GB host gather. **Four independent copies** of the pattern across the genotype tools; the consumer side of the C1/C2 fix. | yes | P×M | 40+ pop (a 2nd ~1 GB host gather after the D2H); becomes free once the resident decode seam lands | **DONE (qpfstats+dstat twins, `9ad33d9`)** — scan-keyed `[P×M]` column gather + CUB Flagged, NO H2D re-upload; extract_f2 + dates twins STILL OPEN (Stage 2/3) |
| M5 | dates | `src/core/stats/dates.cpp:299-313` | Host nested loop re-packing target genotypes onto the kept-SNP axis: per target individual (n_target), per kept SNP (M_kept), read a 2-bit code via `io::code_in_byte` and re-insert with shift/OR. Runs to completion before any GPU work (input to `dates_curve`). Host bit-unpack+repack scaling M×n_target. | yes | M_kept × n_target | AADR autosome (M~1M × n_target~50-200 = 50-200M host bit-twiddles, pre-GPU, single-threaded); golden tiny | **MEDIUM** |
| M6 | dates | `src/core/stats/dates.cpp:107-161` (`fit_exp_decay`/`linfit_2x2`), called at `:433`/`:451` | Single-exp + affine decay fit: a **4000-point coarse host grid** over decay v, each point a `linfit_2x2` (two long-double passes over the windowed curve solving a 2×2 normal eq + RSS) + a 200-iter ternary refine. Host least-squares, run once for the full curve AND once per leave-one-chrom replicate (1 + n_chrom ≈ 23 calls). | yes | n_chrom (~22) × 4000 grid × window (~n_bin up to ~1000) | large-maxdis/many-chrom (~23×4000×1000 ≈ 9e7 host long-double FMA post-FFT); golden curve hides it | **MEDIUM** |
| M7 | qpAdm SE (single-model / large tail) | `src/core/qpadm/nested_models.cpp:16-45` (`sample_cov_diag`), called from `se_from_loo:72` | The S7 jackknife-SE variance reduction (`diag(cov(wmat))` of the nb×nl LOO-weight replicate matrix, long-double sum-of-squares per column) on the host. The per-block re-fits ARE on-device (`gls_weights_loo_batched`); only the final diag-of-covariance reduction is a host loop over nb. **The small-path rotation is exempt** (uses on-device `launch_qpadm_se_from_wmat_batched`); only the large-tail (nl>5 / nr>10 / r>4, > kQpMax*) routes through `se_from_loo`. **The on-device kernel already exists** (`launch_qpadm_se_from_wmat_batched`, cuda_backend.cu:4269) — wired only into the batched path. | yes | n_block × nl, per large-tail model | S8 rotation large-tail models; common small-path exempt; only O(nb) host after the GPU refits | **MEDIUM** |

**Fixes.** M1/M2 → a `ComputeBackend` ratio-/num-den block-jackknife virtual batched over N
(one warp/block per item over n_block), reading the already-device-resident `x_blocks`/`x_loo` (M1)
or `numsum`/`densum`/`cnt` (M2), returning only est/se/z — native FP64 (the host long-double cannot
be bit-reproduced; re-validate the goldens with the ascending-b operand order). M3/M4 → the
on-device keep-mask + CUB compaction from the C1/C2 fix (the shared device-compaction primitive,
replicated 4×). M5 → device gather kernel over (n_target × M_kept) consuming `tile.packed` + a
device kept-src index, no host repack. M6 → batch the 1+n_chrom curves × 4000-point v-grid as a
kernel (each (replicate, v) an independent 2×2 normal-equation least-squares, same shape as the
f2/cov SYRK), argmin per replicate on-device, keep only the ternary refine + λ-conversion on host.
M7 → wire the existing `launch_qpadm_se_from_wmat_batched` into the single-model path (a backend
`se_from_wmat` virtual, or fold the reduction into `gls_weights_loo_batched`), leaving `se_from_loo`
to D2H only the nl-length se vector.

---

## LOW — small fixed cost, off the data-scaling / parity-gated path

| # | area | file:line | what | in_prod | scales_with | bites_at | severity |
|---|------|-----------|------|---------|-------------|----------|----------|
| L1 | qpAdm rank diagnostic | `src/device/cuda/cuda_backend.cu:3383-3393` (`rank_Q` via `core::jacobi_svd` in the single-model `rank_sweep`) | Numerical rank of the m×m covariance Q on the host via the CPU-oracle `jacobi_svd` (O(m³) + a singular-value count), per single-model-path model. m=nl·nr is small+bounded; one fixed host SVD per model, never per-block/per-SNP. Observability-only (`model_well_determined.rank_Q`, explicitly NOT on the statistic path). | yes | m=nl·nr (bounded), per model | negligible even at rotation scale (batched path doesn't hit it; per-model tail is a small fraction) | **LOW** |
| L2 | decode front-end | `src/io/ploidy_detect.cpp:29-43` (`detect_sample_ploidy`), called `extract_f2_core.cpp:125` | AT2 pseudohaploid auto-detect: host loop over n_individuals × min(1000, nSNPs) unpacking 2-bit codes, bumping a sample to diploid on the first het. Reads the same packed tile about to be uploaded. **Fixed 1000-SNP window** → does NOT scale with M. | yes | N individuals × 1000 (fixed window) | large panel (thousands of individuals); bounded window keeps it a fraction of decode | **LOW** |
| L3 | qpGraph edge recovery | `src/device/cuda/cuda_backend.cu:2696-2700` (+ D=0 fast path `:2650-2654`) | One host `qpgraph_score()` eval per run for final edge-length + f3_fit recovery at the winning θ (dense host FP64: pppp O(npair²·nedge), cc O(npair·nedge²), LU/NNLS O(nedge³), res'·ppinv·res O(npair²)). **One eval per run** — the fleet optimizer itself is fully on-device. The on-device `d_qpgraph_score` already computes the same quantities; the winning restart's bl/f3_fit could be returned instead of recomputed. | yes | npair=C(npop,2), nedge, per run | not at golden; a few ms at 40 pops (one-time); bites HARD only if a future topology search loops `run_qpgraph` per candidate | **LOW** |
| L4 | qpGraph diagnostic | `src/core/qpadm/qpgraph_fit.cpp:135-149` | Worst-f3-residual-z scan: host loop over npair computing z[k]=(f_obs−f3_fit)/√Q[k,k], tracking max\|z\|. O(npair) per run, result formatting/diagnostics. | yes | npair=C(npop,2), per run | never meaningfully (~780 iters @ 40 pops, microseconds); only matters if multiplied by a future n_graph axis | **LOW** |

**Fixes.** L1 → accept as a bounded per-model diagnostic, or reuse the on-device `large_svd_V`
SVD already in the large-fit path. L2 → fold ploidy auto-detect into the decode launch as a cheap
prepass kernel (worth doing only as part of the resident-decode rework). L3 → return the winning
restart's bl/f3_fit from the fleet kernel instead of recomputing on the host. L4 → no fix needed
as a single-run diagnostic. **Forward risk (L3/L4):** if a future **n_graph topology search** ever
wraps `run_qpgraph` in a host loop over thousands of candidate graphs, the per-call basis
re-assembly + serial host edge-recovery eval become the qpfstats-class trap one level up — build
that axis **batched** (topologies fanned across the existing fleet thread axis, the f3 basis
assembled once and shared) from the start.

---

## CLEAN BILL — confirmed on-device / correct shape (no violation)

- **qpfstats production driver** (`qpfstats.cpp:324`): the fused on-device `be.qpfstats_blocks_smooth`
  (the exemplar fix landed); the smoothing solve (syrk/potrf/gemm/Dtrsm + NaN downdate) is cuBLAS/cuSOLVER.
- **The S8 rotation common path** — `fit_models_batched` (`cuda_backend.cu:3631-4094`): gather + loo/total
  + xtau kernels, `cublasDgemmStridedBatched` Q, cuSOLVER `potrfBatched`/`potrsBatched` Qinv, batched ALS,
  SE fully on-device (`launch_qpadm_loo_models_batched` + `launch_qpadm_se_from_wmat_batched`). The host
  two-pass survivor filter reads only already-D2H'd cheap per-model O(1) scalars. **This is the headline:
  the bulk of any real search is clean.**
- **The qpAdm orchestration** — `qpadm_fit.cpp` `run_impl`/`run_qpadm_impl`/`run_qpwave_impl` and
  `model_search.cpp` dispatch all compute through `ComputeBackend` virtuals; `jackknife.hpp`/`gls_solve.hpp`
  are thin host-pure seam forwarders.
- **`jackknife_cov`** (`cuda_backend.cu:2728-2868`): xtau kernel + `cublasDsyrk` Q (covariance over **all**
  blocks in one SYRK, not a per-block host loop) + cuSOLVER Cholesky Qinv. Shared by qpAdm and qpGraph.
- **`jackknife_diag` / f4 / f3 diagonal jackknife** (`cuda_backend.cu:2877-2927`): `launch_f4_xtau` +
  `launch_f4_diag_var` on-device; `f4.cpp:119-136` / `f3.cpp:116-133` per-k tails are O(N) scalar
  est/se/p formatting on already-reduced scalars (the correct contrast to the M1 f4ratio offender).
- **The f-stat sweep** (`run_f4_sweep`/`run_f3_sweep`, `cuda_backend.cu:1962-2199`): on-device unrank
  (`sweep_unrank_*_kernel`) + gather + loo/total + xtau + diag_var + z-filter + `cub::DeviceSelect::Flagged`
  compaction + bounded device top-K reservoir. No host enumeration, no host per-item arithmetic. **The
  MEMORY note "fstat-sweep CPU-bound ~40k/s" is STALE** — that was the pre-fix state; the current code
  (cuda_backend.cu:1867) is the explicit fix. (One non-violation: a per-chunk `cudaStreamSynchronize` +
  single int readback serializes the host once per chunk — a pipeline-overlap nicety, O(chunks) which is a
  handful per sweep, ZERO host FP compute. Not flagged.)
- **`dstat_block_reduce`** (`cuda_backend.cu:1253-1306` → `dstat_kernel.cu`): the per-SNP D reduction is
  on the GPU, batched over all N quadruples, copies back only the tiny [N×n_block] numsum/densum/cnt.
- **`decode_af` arithmetic** (`decode_af_kernel.cu:100-168`): the 2-bit unpack + per-SNP AF reduction +
  het-correction are a real GPU kernel (segmented reduction, coalesced reads, shared-mem transpose). The
  het-correction (`f2_estimator.hpp`) is `__host__ __device__` scalars invoked inside the GPU f2 GEMM
  feeder, not a host loop. **The arithmetic is clean — only the surrounding D2H + host filter (C1/C2) is not.**
- **qpGraph fleet** (`qpgraph_fit_fleet`, `cuda_backend.cu:2591-2704`): the whole multistart × maxit
  projected-Newton loop on-device (one thread/restart), per-iteration objective `d_qpgraph_score` in-kernel
  (fill_pwts, GLS design, trace-ridge, native NNLS, LU, res'Qinv·res); only per-restart {θ,score} returns.
- **dates LD engine** (`dates_curve`, `cuda_backend.cu:1320-1510`): per-sample weight/residual,
  grid-scatter, batched-over-chrom cuFFT autocorrelation, power/cross-power/extract-lags/re-bin are all
  kernels; host drives only the ~100-sample loop + tiny per-sample scalar D2H. **The FFT core is flat in M;
  the ~10¹² SNP-pair object is never formed.** (Only the M5/M6 host loops before/after it are flagged.)
- **Setup/IO/formatting (exempt, verified):** `build_popcomb_and_design` (O(npopcomb) integer/coeff,
  data-independent, once — large at 40 pops but setup, candidate for future on-device generation);
  `assign_blocks`/`block_ranges`/`block_lengths` (the shared integer block-layout primitive);
  `geno_reader.cpp` `read_tile` (byte-gather, no decode); `parse_qpgraph` (one-time index/path tables);
  the flat quad/tuple index builds in dstat/f4/f3/f4ratio; `assemble_result` per-model O(rmax+nl) formatting;
  `ranktest.cpp` `reduce_rows` (integer gather, m bounded); `fstat_sweep.cpp` p-loop over ≤top_k survivors.
- **All `CpuBackend` `src/device/cpu/*` host loops** (`gls_weights_loo_batched`, `dstat_block_reduce`,
  `matrix_jackknife_est_col`, `qpgraph_score`, `dates_curve`, ...) — the test-only parity oracle, EXEMPT.

---

## Recommended fix order

The qpfstats jackknife is already in flight (the exemplar — the `qpfstats_blocks_smooth` fused
seam). After it:

1. **C1 + C2 + M3 + M4 (one shared piece): the device-resident decode seam.** Highest leverage —
   `decode_af` keeps Q/V/N resident in VRAM (no [P×M] D2H), an on-device per-SNP keep-mask kernel
   does the maxmiss/pooled-MAF reductions, CUB `DeviceSelect` does the lockstep compaction. This
   single seam removes the ~1.1 GB D2H + the ~46M-iter host reductions + the four lockstep-subset
   copies across extract_f2 / dstat / dates / qpfstats at once. It is the structural cause of the
   ~14 s 40-pop decode and the closest analog to the qpfstats keep-on-device violation.
2. **M1 + M2 (shared kernel family): the ratio / num-den block-jackknife backend virtual.** The
   genotype-/ratio-path siblings of the qpfstats jackknife — same shape, inputs already
   device-resident from `assemble_f4_quartets` / `dstat_block_reduce`. One batched per-item LOO
   jackknife kernel (native FP64, ascending-b operand order, re-validate goldens) serves both;
   eliminates their D2Hs too.
3. **M7: wire the existing `launch_qpadm_se_from_wmat_batched` into the single-model path.** Cheapest
   real win — the on-device kernel already exists, it just isn't called from `se_from_loo`.
4. **M5 + M6: the dates host loops.** M5 (target repack → device gather) and M6 (the 4000-point
   exp-fit grid → batched least-squares kernel). Bounded axes (n_target, n_chrom), lower urgency.
5. **L1-L4: defer / tidy-up.** Bounded per-model/per-run costs off the data-scaling path. L3/L4
   matter only if a future qpGraph topology-search axis is built — build it **batched** when it lands.
