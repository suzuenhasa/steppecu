# Code inventory for the new tools (qpfstats / qpGraph / DATES)

Status: research only, no implementation. Branch `phase2-fit-engine`.
Scope: map every reusable steppe seam the three remaining new-machinery tools
(qpfstats, qpGraph, DATES) can stand on, and enumerate the gaps each one opens.

THE GOVERNING LESSON (foregrounded in every design doc): we just shipped a
CPU-bound f-stat sweep by accident — host enumeration/filter/IO pegged one CPU
core at GPU-0% — rolled it back, and rebuilt it GPU-bound (on-device unrank +
CUB compaction, `cuda_backend.cu:1470`). Every new design here is GPU-FIRST and
must be provably GPU-BOUND: the only host work is sizing, dispatch, and the
small final result. Wherever a step *looks* like it needs the CPU (a loop over
items / SNPs / SNP-pairs / iterations, a host-side solve/sort/fit, a large host
materialization), the algebra is reformulated so it does not.

All `file:line` citations below were re-verified by reading the files at HEAD
(`605ef6f`). API/reference checks noted inline.

---

## A. The solve / precision-promotion seam (qpfstats + qpGraph reuse directly)

1. **`ComputeBackend` abstract seam** — `src/device/backend.hpp:432`. The single
   CUDA-free DI door between host `core` and the GPU; `core` never issues a
   GEMM/SVD/Cholesky itself. Two impls (CudaBackend, CpuBackend oracle). Every
   new tool adds ONE new virtual here following the established
   `assemble_*` / `*_sweep` pattern, keeping CUDA private to `steppe_device`.

2. **`set_solve_precision` promotion knob** — base no-op `backend.hpp:643`; CUDA
   override `cuda_backend.cu:1411` (records `solve_precision_`). The per-stage
   seam that promotes the small ill-conditioned solves from default native-FP64
   to emulated-FP64. Reusable as the qpfstats Cholesky-promotion knob and
   qpGraph's per-edge GLS-solve knob.

3. **`CusolverMathModeScope` + `engage_solver_precision`** — `handles.hpp:535`
   (class) and the factory around `handles.hpp:627`. RAII get/set/restore of
   `cusolverDnSetMathMode`, routed through the ONE `emulation_honorable`
   predicate. VERIFIED: `STEPPE_HAVE_CUSOLVER_FP64_EMULATED` is `1`/`0`-gated at
   `handles.hpp:85/87` and the box ships it OFF — CUDA 13.0 cuSOLVER has no
   FP64-emulated math mode, so the seam is LIVE but degrades to native today
   (`handles.hpp:478-495`). `MathModeScope` (cuBLAS analogue) `handles.hpp:263`.

4. **RAII handle wrappers** — `CublasHandle handles.hpp:101`,
   `CusolverDnHandle handles.hpp:329`, `GesvdjInfo` (around `handles.hpp:427`).
   Created-once-reused; any new device kernel needing cuBLAS/cuSOLVER reuses
   these.

5. **The wired cuSOLVER/cuBLAS solve primitives** (all VERIFIED in
   `cuda_backend.cu`): single `cublasDsyrk` (Q covariance, `:2263`),
   `cusolverDnDpotrf`/`Dpotri` (SPD Qinv, `:2299`/`:2312`); and the MODEL-BATCHED
   path `cublasDgemmStridedBatched` (`:3310`), `cusolverDnDpotrfBatched`
   (`:3342`) + `cusolverDnDpotrsBatched` (`:3370`). qpfstats' per-block solve
   and qpGraph's per-(graph,restart) edge solve are exactly potrf/potrs + gemm,
   batchable across blocks/edges via these proven primitives.

## B. Small dense linear algebra (host oracle)

6. **`small_linalg.hpp`** — `src/core/internal/small_linalg.hpp`: header-only,
   CUDA-free, native-FP64. `solve()` (`:145`, LU partial-pivot = R's `solve`),
   `inverse()` (`:165`), `jacobi_svd()` (`:204`), `lu_factor`/`lu_solve_rhs`
   (`:110`/`:89`). The CpuBackend ORACLE the GPU is diffed against for EVERY new
   tool. Small dense only (O(n^3)) — that IS the oracle property.

7. **`pchisq.hpp`** — `pchisq_upper` (`src/core/internal/pchisq.hpp:88`,
   regularized incomplete gamma). qpGraph per-fit / overall p-values; qpfstats
   GOF p. (`f4_two_sided_p` in `src/core/qpadm/f4.cpp` for normal-tail z->p.)

## C. The f2 cache (qpfstats + qpGraph read this; DATES does NOT)

8. **`F2BlockTensor`** — `include/steppe/fstats.hpp` (around `:47`). The
   cacheable ADMIXTOOLS-compatible per-block f2 `[P x P x n_block]` + per-block
   vpair (S4 jackknife weight) + block_sizes; FP64 in every precision mode.
   qpfstats and qpGraph are BOTH pure functions of this tensor — every f-stat is
   a linear combo of f2 entries — consumed with zero new precompute, exactly
   like `run_f4`/`run_f3`/qpadm.

9. **`DeviceF2Blocks`** (device-resident handle) — `compute_f2_blocks_device`
   `backend.hpp:509`; `f2_device()`/`vpair_device()` accessors used at
   `cuda_backend.cu:1437`; `upload_f2_blocks_to_device` /
   round-trip in `device/device_f2_blocks.hpp`. qpfstats/qpGraph read the
   resident f2 with zero D2H.

10. **The f2-dir reader** — `read_f2_dir` (`src/app/f2_dir_io.hpp`, around
    `:63`) parses `<dir>/f2.bin` into a host `F2BlockTensor` + pop_labels.
    New CLI subcommands load the cache identically to `cmd_f4`.

11. **`f2_estimator.hpp`** — `src/core/internal/f2_estimator.hpp`: `het_correction`,
    `f2_term`, `assemble_f2_numerator`, `finalize_f2`, `pair_block_is_missing`
    (STEPPE_HD host+device shared) — the underlying f2 math if a new tool ever
    needs raw f2 assembly.

12. **`block_ranges` / `assign_blocks`** — `src/core/domain/block_partition_rule.hpp`
    `assign_blocks` (`:162`, AT2 `setblocks()` SNP-anchored cumulative walk by
    Morgans, cited to DReichLab `qpsubs.c`), `block_ranges` (`:179`/`:224`).
    qpGraph/qpfstats inherit blocks via the f2 cache's block_sizes; DATES needs
    it on its own genotype SNP axis (and uses chromosome as the jackknife block).

## D. The genotype decode front-end (DATES reuses; qpDstat already does)

13. **`decode_af` virtual** — `backend.hpp:615` (`DecodeResult = Q/V/N` col-major
    `[P x M]`, struct `backend.hpp:304-315`); shared math
    `core/internal/decode_af.hpp`. `DecodeTileView backend.hpp:273` (CUDA-free
    packed-tile + pop partition + per-sample ploidy view). DATES shares exactly
    this front-end. NOTE the CUDA `decode_af` D2H's Q/V/N to host at
    `cuda_backend.cu:1215-1219` — DATES needs a device-RESIDENT variant (its only
    front-end change; see the DATES doc).

14. **The io readers** — `io::GenoReader::read_tile` (`src/io/geno_reader.hpp`,
    around `:79`, tiled packed-byte gather, TGENO individual-major),
    `read_snp` -> `SnpTable` with `genpos_morgans` (`src/io/snp_reader.hpp:41`)
    AND physpos, `read_ind` -> `IndPartition` (`PopSelection::Explicit` subsets
    to only the needed pops). `genpos_morgans` is the DATES grid-index input.

15. **The WORKED genotype-stat exemplar** — `run_dstat`,
    `src/core/stats/dstat.cpp:185-285`: `GenoReader` -> `read_ind(Explicit{pop_union})`
    -> `read_snp` -> `read_tile` -> `decode_af` -> autosome keep-mask + lockstep
    Q/V/genpos subset (`:240-254`, the `genpos_kept` DATES down-payment, VERIFIED)
    -> `assign_blocks` over the kept axis (`:267`) -> `dstat_block_reduce`
    (`:285`). The COPY-THIS template for DATES' front-end. `dstat_block_reduce`
    virtual (`backend.hpp:774`) is the genotype-path back-end DIVERGENCE point
    (SNP-tile-streamed segmented reduction, NOT the f2 GEMM) — the structural
    model DATES follows with its own (pairwise) kernel.

## E + F. The genotype-stat seam and the jackknife (all three reuse)

16. **`run_dstat`** — `include/steppe/dstat.hpp`, impl `dstat.cpp:162`. The
    genotype sibling of `run_f4` (does NOT read the f2 cache). `dstat_jackknife`
    (`dstat.cpp:70-157`) num/den block-jackknife.

17. **`jackknife_cov`** — `backend.hpp:791` (S4 weighted block-jackknife producing
    dense `Q[m x m]` + `Qinv`; CUDA path `cublasDsyrk`+`potrf`/`potri`). Driver
    `src/core/qpadm/jackknife.hpp:19`. `JackknifeCov` POD `backend.hpp:130`.
    This produces exactly the qpGraph `ppinv = solve(f3_var)` (m = npair).

18. **`jackknife_diag`** — `backend.hpp:809` (DIAGONAL-only var, O(m) memory — the
    OOM fix at sweep scale; same xtau seam minus the dense SYRK+Cholesky).
    `JackknifeDiag` POD `backend.hpp:163`. Any per-item SE in a new tool.

19. **f4ratio's per-block-RATIO jackknife** — `src/core/qpadm/f4ratio.cpp`. The
    delete-one ratio statistic via the `est_to_loo` replicate seam with an
    explicitly-written ratio xtau. VERIFIED IMPORTANT CAVEAT
    (`f4ratio.cpp:28-31`): it deliberately does NOT route through `jackknife_cov`
    because AT2's `jack_mat_stats` uses a *different* xtau decomposition (`tot*h`
    leading minus `est`) than `jackknife_cov` (`x_total*h` minus `tot_line_`).
    The reusable part is the `est_to_loo` replicate seam; the ratio xtau is
    written explicitly. This is THE template for jackknifing any
    nonlinear-of-f-stats functional (qpGraph fits ratios/products of f-stats).

## G. The rotation / model-batch engine (qpfstats joint multi-model, qpGraph multi-edge)

20. **`run_qpadm_search` orchestrator** — `src/core/qpadm/model_search.cpp` (the
    S8 rotation): partitions a model list by `model_in_small_path`, routes
    small-path -> device-BATCHED `fit_models_batched`, large/>32 tail -> per-model
    `fit_models_batched_default`. The proven "bucket by shape, batch the bucket,
    per-model only the tail" pattern — exactly the qpfstats many-block and qpGraph
    many-(graph,restart) batching shape.

21. **`fit_models_batched` virtual** — `backend.hpp:954` (BATCHED fit over many
    same-shape models against ONE resident f2 in ONE dispatch). CUDA override is
    the genuine model-batched device path (`cuda_backend.cu:3075+`:
    strided-batched gemm Q, potrfBatched/potrsBatched Qinv, one-thread-per-model
    kernel). `batched_dispatch_count` (`backend.hpp:996` / `cuda_backend.cu:1419`)
    PROVES it ran batched, not as a host loop. THE GPU-batched-not-host-loop
    contract every new tool must meet.

## H. The GPU-bound sweep (the on-device enumerate+compute+filter+compact template)

22. **The CUB sweep pipeline** — `cuda_backend.cu:60` (`cub/device/device_select.cuh`),
    `:64` (`device_radix_sort.cuh`). `run_fstat_sweep_device` (`:1470`): on-device
    UNRANK (combinatorial-number-system kernel `sweep_unrank`
    `qpadm_fit_kernels.cu:697`, replacing host enumeration) -> gather+loo+xtau+diag
    -> `|z|` filter -> `cub::DeviceSelect::Flagged` stream-compact (two-call
    temp-storage idiom, `:1636`/`:1735`) -> `cub::DeviceRadixSort::SortPairsDescending`
    bounded top-K reservoir (`:1640`/`:1656`) -> D2H only survivors. Drivers
    `backend.hpp:831` `f4_sweep` / `:842` `f3_sweep`; host driver
    `src/core/qpadm/fstat_sweep.cpp` (NO host enumerate/filter/loop). THE template
    that prevents the CPU-bound relapse — DATES' SNP-pair enumeration and
    qpGraph's many-topology screen must be on-device-unranked + CUB-compacted
    exactly like this. CUB is already wired; VERIFY CUB API against CUDA 13.x /
    CCCL 3.x docs (the two-call temp-storage idiom and `SortPairsDescending`
    signature are CCCL-3.x stable but confirm before coding).

## I. CLI / binding scaffold

23. **CLI command pattern** — `src/app/cmd_f4.cpp` (`run_f4` chain at `:113-173`):
    `read_f2_dir` -> `PopResolver` name->index (`src/app/pop_resolver.hpp`) ->
    `build_resources` -> `upload_f2_blocks_to_device` -> `run_f4(DeviceF2Blocks,...)`
    -> emit. Nine `cmd_*.cpp` exist (extract_f2/f3/f4/f4ratio/fstat_sweep/qpadm/
    qpdstat/qpwave/rotate). `cmd_qpdstat.cpp` is the genotype-path exemplar (no
    f2-dir). Each new tool gets a `cmd_*.cpp` on this chain; `pop_resolver` +
    `result_emit` reused verbatim.

24. **Python bindings** — `bindings/module.cpp` (nanobind `steppe._core`,
    MARSHALLING ONLY): `F2Handle` (host tensor + pop labels + cached Resources,
    precompute-once/fit-many) at `:69`, `read_f2` loader at `:343`; already binds
    `run_f4`/`run_f3`/`run_f4ratio`/`run_dstat`/`run_qpadm`/`run_qpwave`/
    `run_qpadm_search`. Each new tool exposes `run_<tool>` the same way; results
    are host-side (no `DeviceF2Blocks` crosses into Python).

---

## The gaps — what steppe LACKS (all VERIFIED absent)

`grep -rniE 'cufft|nlopt|lbfgs|levenberg|marquardt|gauss-newton|nelder|simplex|
projected.gradient|graph_to_pwts|fill_pwts|exponential.decay|weighted.ld|
autocorrel'` over `src/` + `include/` returns EMPTY (only false positives like
"become"/"outcome" excluded). Confirmed.

1. **NO general nonlinear optimizer** (blocks DATES; partially blocks qpGraph).
   The ONLY optimizer in the tree is the qpAdm-SPECIFIC bilinear ALS
   (`dev_opt_A_core`/`dev_opt_B_core` + the constrained weight solve `dev_solve`,
   `qpadm_fit_kernels.cu:97-460`) — hardcoded to the rank-r factorization
   X ~= A.B, NOT a general fitter. DATES ends in a nonlinear exponential-decay
   fit `A(d) ~= A*exp(-lambda*d)+c`; qpGraph minimizes a GLS objective over
   admixture weights. NEEDED: a small nonlinear least-squares / projected-gradient
   solver. GPU-BOUND design risk: a host iterate-to-convergence loop per
   item/edge is the CPU-bound failure mode — must be batched (one thread/block per
   fit, on-device residual+gradient), like `fit_models_batched`
   (`cuda_backend.cu:3075+`). For DATES the algebra makes it tiny-host (fit on a
   ~1000-point reduced curve, not on data) — see the DATES doc.

2. **NO FFT** (blocks DATES at scale). No cufft include anywhere. VERIFIED the
   device link line is `CUDA::cudart CUDA::cublas CUDA::cusolver` only
   (`src/device/CMakeLists.txt:56`) — `CUDA::cufft` NOT linked. VERIFIED
   `libcufft.so` IS present in CUDA 13.0 on box5090 (`libcufft.so.12.0.0.61`), so
   the dependency is satisfiable. DATES/ALDER make the naive O(M^2) SNP-pair
   covariance tractable via FFT of a genetic-map-binned grid array. NEEDED:
   `CUDA::cufft` (D2Z/Z2D batched). GPU-BOUND design risk: a host SNP-pair loop is
   an instant CPU-bound relapse — the pairwise sum is reformulated as an
   on-device autocorrelation (see the DATES doc).

3. **NO genetic-map / LD engine, no per-SNP-pair machinery** (blocks DATES).
   Every existing stat is per-SINGLE-SNP then per-block-reduced; there is NO
   per-SNP-PAIR primitive. `genpos_morgans` IS read and now retained through the
   genotype path (`dstat.cpp:240-254`); `assign_blocks` bins by Morgans — but
   there is NO weighted-LD / ancestry-covariance-vs-genetic-distance kernel.
   `physpos` is parsed then discarded. NEEDED: a new genotype-path kernel
   computing the ancestry covariance between SNP pairs binned by genetic
   distance. Structurally NEW; the dates seam doc warns NOT to design it from D's
   per-single-SNP shape.

4. **(qpGraph-specific) NO graph/topology data structure or graph-search**.
   qpGraph fits an admixture-graph DAG. steppe has no graph representation, no
   topology enumeration/scoring. The per-fit GLS solve REUSES the existing
   potrf/potrs+gemm seam; the graph object + path algebra + (optional) topology
   search are new host+device machinery.

---

## Summary by tool

- **qpfstats — LOWEST new-machinery.** Pure function of the f2 cache. Reuses an
  `assemble_f4_quartets`-style combine + `jackknife_cov` + the cuSOLVER
  potrf/potrs solve seam + the rotation batching. The "joint" estimator is a
  larger covariance solve (already have `cublasDsyrk` + potrfBatched/potrsBatched)
  + emulated-FP64 promotion. New code: a design-matrix fill, a shared-factor
  batched-solve arrangement (+ its oracle), a NaN-block downdate handling, an
  allsnps-genotype numerator extension, a scatter/recenter + run/emit/binding.
  No optimizer/FFT gap.

- **qpGraph — MEDIUM-HIGH.** Reuses f2 cache + GLS solve primitives + jackknife
  (incl. the f4ratio nonlinear-functional jackknife template). GAPS: a general
  bounded nonlinear optimizer (gap #1) + a graph/path-algebra data structure +
  optional topology search (gap #4). Per-edge linear algebra is reusable; the
  outer fit loop and graph object are new.

- **DATES — HIGHEST.** Reuses ONLY the decode front-end + the jackknife. GAPS:
  all three big ones — FFT (gap #2), the nonlinear decay fitter (gap #1, but tiny
  by reformulation), and the per-SNP-pair genetic-distance-binned
  ancestry-covariance LD engine (gap #3). Genuinely new machinery.

THE GPU-BOUND MANDATE for all three: the CPU-bound failure mode is a host loop
over items/SNP-pairs/edges/iterations or host-side solves/sorts/fits. The
in-tree defenses are (a) the on-device unrank + CUB compaction + bounded
radix-sort reservoir sweep (`cuda_backend.cu:1470`) and (b) the model-BATCHED
one-dispatch-per-bucket fit with `batched_dispatch_count` proof
(`cuda_backend.cu:3075+`). DATES' pairwise enumeration, qpGraph's per-edge fits
and many-topology screen, and qpfstats' per-block solves must be built on these,
not host loops.

PRECISION: default `EmulatedFp64{40}` for matmul-heavy work (xtx, x'.y, the
ppwts'.ppinv.ppwts GEMMs); native-FP64 carve-out for cancellation-sensitive
f-stat combines and the small ill-conditioned solves; promotable via
`set_solve_precision` / `CusolverMathModeScope` (commit 8d4f22f, fit-engine.md).
NOTE cuSOLVER has NO FP64-emulated math mode in CUDA 13.0 — the seam is live but
degrades to native today.

VERIFICATION BASIS: read every cited steppe file at HEAD. Box5090 confirmed to
hold `/workspace/AdmixTools_src/qpfs.pdf`, `src/qpfstats.c`, `src/qpGraph.c`, and
`libcufft.so` (CUDA 13.0). No DATES/ALDER source on the box — DATES algebra is
grounded in the papers (Loh 2013, Chintalapati 2022) and would need cloning
`github.com/priyamoorjani/DATES` for byte-parity. FLAG: qpGraph's exact optimizer
class differs between the R port (qpgraph.R, L-BFGS-B) and the C source
(qpGraph.c, a custom routine — no lbfgs/nelder/powell symbols found); confirm
which the goldens come from before designing the fitter.
