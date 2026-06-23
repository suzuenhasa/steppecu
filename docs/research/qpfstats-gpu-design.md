# qpfstats — GPU-shape-first design

Status: research only, no implementation. Branch `phase2-fit-engine`.
Effort: MEDIUM-HIGH (on-seam, but more new code than f4/f3; ~55% reuse / 45% new).

qpfstats = joint full-basis least-squares smoothing of all f2/f3/f4 over the
outgroup f(O;A,B) basis. The output is a *smoothed* per-block f2 tensor that the
downstream qpAdm / qpWave / qpGraph consume — its whole reason to exist is the
`allsnps:YES` case where different stats use different SNP sets and the exact
f-stat linear identities break.

## 0. The CPU-bound failure mode — designed out FIRST (the lesson)

THE SWEEP-DISASTER ANALOGUE IS REAL AND SPECIFIC HERE. AT2's own
`qpfstats_regression` (R `io.R:3898-3925`) is LITERALLY a host loop over
`nblocks` (chunked at 64) doing per-block triangular solves and per-block
Cholesky downdates — a CPU loop over ~1300 jackknife blocks, each a dense solve
at `npairs` up to thousands. Transliterating that loop (or looping the host
`small_linalg.hpp` solve per block) is EXACTLY the CPU-bound trap that sank the
first f-stat sweep: one core pegged, GPU idle, scaling death as `npop` and
`nblocks` grow. A second latent trap: AT2 itself warns the long-format
`(npopcomb x nblocks)` frame "peaks at hundreds of GB" (`io.R:4002-4013`) — steppe
must NEVER host-materialize that frame.

PREVENTION (by design, not discipline):
- The per-block solve is REFORMULATED as ONE shared-factor batched solve:
  `A_shared = x'x + ridge*I` factored ONCE (`cusolverDnDpotrf`), then the
  per-block `b` is a SINGLE batched back/forward solve over the whole
  `[npairs x nblocks]` RHS (`cusolverDnDpotrsBatched` / `cublasDtrsm`). There is
  NO host per-block loop. Same batched-cuSOLVER primitive already proven at
  `cuda_backend.cu:3342`/`:3370`.
- Numerator matrices are produced by the existing on-device
  `assemble_f4_quartets` / `dstat_block_reduce` kernels (per-SNP reduction stays
  on-device; only the small `[npopcomb x nblocks]` result lands).
- Missing/NaN blocks are GROUPED by missingness mask (CUB segmentation) and
  re-factored in batches, NOT branched per-block on the host.
- `ymat` is never the long frame — it is the compact `t(numer)` the device
  already holds.

Net: the only host work is sizing/dispatch + the final small output copy. The
CPU-bound failure mode is structurally impossible because no per-item /
per-block / per-SNP loop ever runs on the host.

## 1. The algebra (verified)

THE BASIS (`qpfs.pdf`, N. Patterson 2020-06-18, on box5090). F-statistics form a
linear vector space of dim `n(n-1)/2`. The basis is all `f(O;A,B)` with O a fixed
outgroup: `n-1` f2 + `(n-1)(n-2)/2` f3 = `n(n-1)/2`. Any f-stat
`f_i = sum_j c_ij b_j` with `c_ij` FIXED, data-independent — EXCEPT under
`allsnps:YES` (different SNP sets per stat) the identity breaks, which is the
point of the tool.

THE MODEL (`qpfs.pdf` §2). Write `f_hat_i = sum_j c_ij b_j + sigma_i n_i`,
`n_i` unit-variance noise, `sigma_i` from block jackknife; set
`c'_ij = c_ij/sigma_i` and solve the overdetermined `c'_ij b_j ~= f_hat_i/sigma_i`
by weighted least squares; covariance on `b_hat` from a second jackknife pass.

THE AT2 R IMPLEMENTATION (the production reference; ridge replaces explicit
sigma-weighting). `construct_fstat_matrix` (`io.R:3745-3767`): each design row
sets `x[i, idx(p1,p4)]=+1`, `x[i,idx(p2,p3)]=+1`, `x[i,idx(p1,p3)]=-1`,
`x[i,idx(p2,p4)]=-1` (doubled then /2 for a pure f2) — each f-stat as its f2-basis
expansion via `f4(A,B;C,D)=0.5[f2(A,D)+f2(B,C)-f2(A,C)-f2(B,D)]`.
`qpfstats_regression` (`io.R:3849-3953`): `A_shared = crossprod(x) + diag*ridge`
(ridge `0.00001`); `L = chol(A_shared)`; per block `b = backsolve(L, forwardsolve(t(L), crossprod(x, ymat[,b])))`;
NaN blocks downdate `A_b = A_shared - crossprod(x[nan_rows,])` then generic solve.
Output (`io.R:4038-4051`): scatter `b` into `f2blocks[npop,npop,nblocks]`,
recenter `f2blocks2 = f2blocks - f2(f2blocks)$est + bglob`.

PARITY DIVERGENCE — VERIFIED in the C source on box5090. `qpfstats.c` defaults
`doscale=YES` (`lambdascale`, lines 107, 575-590: f-stats scaled so least-squares
matches fst) and `lsqmode=NO` (line 82), with `diag`/`f2diag` regularization
(76, 129) and `f2weight` only in lsqmode (83). The R `qpfstats()` shown does ridge
only with no visible `doscale`. THIS IS A REAL BYTE-PARITY RISK: the C tool's
`doscale` rescales f2/f2sig before the solve — whichever produced the goldens must
be matched exactly.

GPU REFORMULATION: the per-block normal-equations solve is one shared SPD
Cholesky factor reused across ALL blocks -> `cusolverDnDpotrf` once + a
batched/streamed triangular solve over the `[npairs x nblocks]` RHS in ONE shot;
the NaN-block downdate becomes a low-rank Cholesky-update kernel or a small
grouped re-factor.

## 2. GPU-first, GPU-bound pipeline (single-GPU --device 0)

1. **Per-block numerator matrix** `ymat = t(numer)` `[npairs_obs x nblocks]`,
   built ON-DEVICE. f2-PATH (smoothing the f2 cache): reuse the device-resident
   `assemble_f4_quartets` (`cuda_backend.cu` `:1960`+, launcher
   `:2018`) over `DeviceF2Blocks`. allsnps-genotype path (the real reason
   qpfstats exists): the per-block numerator + per-pair valid-SNP count come from
   the `dstat_block_reduce` seam (`backend.hpp:774`) generalized to emit the
   f4-numerator `sum(a-b)(c-d)` per `(popcomb,block)` — output is the small
   `[npopcomb x nblocks]` matrix, never the per-SNP table.

2. **Design matrix** `x [npopcomb x npairs]` — a tiny on-device fill kernel (one
   thread per popcomb writes its 4 +/-1 coefficients), or host-built once (it is
   O(npopcomb*4) integers, data-independent) and uploaded. Trivial either way.

3. **The solve (the core).** `A_shared = x'x + ridge*I` via ONE `cublasDsyrk`;
   ONE `cusolverDnDpotrf` factor; then per-block `b` is a SINGLE batched
   back/forward solve: `cusolverDnDpotrsBatched` / `cublasDtrsm` over the full
   `[npairs x nblocks]` RHS `= x' . ymat` (one `cublasDgemm`). NaN/missing-block
   downdates handled by grouping blocks by missingness mask (CUB segmented) + a
   per-group re-factor, OR a rank-k Cholesky downdate kernel — NOT a host loop.
   EXACT existing seam: `cuda_backend.cu:3342`/`:3370` already runs
   `cusolverDnDpotrfBatched`+`cusolverDnDpotrsBatched` for the S4 batched Qinv;
   qpfstats reuses the same primitive with a shared-factor batch structure.

4. **Output** smoothed f2_blocks + its jackknife covariance: scatter `b` into the
   `[P x P x nblocks]` tensor on-device (a gather/scatter kernel like the f2
   assemble), recenter (one axpy), and run the EXISTING `jackknife_cov`
   (`backend.hpp:791`) over the smoothed tensor for the covariance pass. Result
   stays a `DeviceF2Blocks` handle, consumed resident — zero D2H except the small
   final output.

HOST SEAM (small only): `npairs` (<= `n(n-1)/2`, ~hundreds for n<=30), `nblocks`
(~1300), the final smoothed tensor + covariance. Everything O(npopcomb*nblocks)
or O(per-SNP) stays on-device.

PRECISION: `A_shared=x'x` and `x'.ymat` are matmul-heavy -> `EmulatedFp64{40}`
default (`engage_f2_precision` / `MathModeScope`, `cuda_backend.cu:2262`/`:3309`).
The Cholesky factor+solve is the ill-conditioned solve -> DEFAULT native FP64 but
PROMOTABLE via `set_solve_precision` -> `engage_solver_precision` ->
`CusolverMathModeScope` (`cuda_backend.cu:1411`, `:2296`, `:3342` region) for the
S8-scale throughput wall, validated per-stage vs the native oracle. `ridge`
regularizes the `x'x` conditioning.

## 3. Reuse inventory (all VERIFIED)

- SOLVE-PRECISION SEAM: `set_solve_precision` (`backend.hpp:643`; CUDA
  `cuda_backend.cu:1411`); `engage_solver_precision`/`CusolverMathModeScope`
  (`handles.hpp:535`) at the solve sites.
- BATCHED cuSOLVER SOLVE: `cusolverDnDpotrfBatched`+`cusolverDnDpotrsBatched`
  already wired for S4 Qinv, `cuda_backend.cu:3342`/`:3370` (the qpfstats
  shared-factor batched-solve clones this); single potrf/potri `:2299`/`:2312`.
- `small_linalg.hpp`: `solve()` (`:145`), `inverse()` (`:165`), `jacobi_svd`
  (`:204`) — the host ORACLE for the qpfstats solve parity.
- PER-BLOCK NUMERATOR: `assemble_f4_quartets` device-resident
  (`cuda_backend.cu:1960`+); `assemble_f3_triples` (`backend.hpp:728`); host
  oracle doors `backend.hpp:705`/`:740`.
- GENOTYPE allsnps PER-BLOCK REDUCTION: `dstat_block_reduce` (`backend.hpp:774`),
  consumed `dstat.cpp:285`.
- JACKKNIFE: `jackknife_cov` (`backend.hpp:791`; driver `jackknife.hpp:19`);
  `JackknifeCov` POD (`backend.hpp:130`), `JackknifeDiag` (`backend.hpp:163`).
- f2 CACHE: `DeviceF2Blocks` (`device_f2_blocks.hpp`), `compute_f2_blocks_device`
  (`backend.hpp:509`); `F2BlockTensor` (`include/steppe/fstats.hpp`). Vpair
  per-pair valid counts retained — the allsnps missingness input.
- PRECISION for matmul: `engage_f2_precision` + `emulation_honorable` +
  `MathModeScope` (`cuda_backend.cu:2262`/`:3309`).
- CLI/BINDING: `run_f4`/`run_f3`/`run_f4ratio` siblings + `cmd_f4.cpp` chain +
  `bindings/module.cpp`.

## 4. New machinery (small, all on existing seams)

1. `construct_fstat_matrix` kernel/host-fill: the `[npopcomb x npairs]` design
   matrix from the +/-1, /2 f4-identity coefficients. Tiny; data-independent.
2. SHARED-FACTOR BATCHED SOLVE driver: `cublasDsyrk(x'x)+ridge*I` -> ONE
   `cusolverDnDpotrf` -> `cusolverDnDpotrsBatched`/`cublasDtrsm` over
   `RHS=x'.ymat`. A NEW arrangement of EXISTING potrf/potrs primitives — a new
   backend virtual (e.g. `qpfstats_solve`) + its CPU oracle via small_linalg
   chol+solve.
3. NaN/missing-block DOWNDATE handling: group blocks by missingness mask (CUB
   segmented sort/scan) -> per-group re-factor, OR a rank-k symmetric
   Cholesky-downdate kernel. The only nontrivial new GPU code (the long pole).
4. The allsnps-genotype numerator emission: extend `dstat_block_reduce` (or a
   sibling) to emit per-`(popcomb,block)` f4 numerators + per-pair valid-SNP
   counts when stats span different SNP sets. (Pure-f2-cache mode reuses
   `assemble_f4_quartets` verbatim — zero new math there.)
5. OUTPUT scatter+recenter kernel (`b` -> `[P x P x nblocks]`, subtract
   `f2(f2blocks)$est`, add `bglob`) + `run_qpfstats` + result struct + emitter +
   golden + binding.
6. The `doscale`/`lambdascale` rescaling IF the goldens use the C tool's default
   (parity-load-bearing scalar prep, not new machinery, but must be matched).

NOT NEEDED: no new jackknife, no new SVD, no new optimizer, no new IO, no per-SNP
table. `ridge` is a scalar constant.

## 5. Effort + ordering

MEDIUM-HIGH (~55% reuse / 45% new). Reuse: the entire batched cuSOLVER
potrf/potrs primitive + precision seam, jackknife_cov, DeviceF2Blocks,
assemble_f4_quartets/dstat_block_reduce numerator producers, small_linalg oracle,
CLI/binding scaffold. New: the design-matrix fill, the shared-factor batched-solve
arrangement + its oracle (the bulk), the NaN-block downdate kernel (the one hard
new kernel), the allsnps genotype-numerator extension, the scatter/recenter +
run/emit/golden. Schedule AFTER f4/f3 (it consumes the same numerator + jackknife
seams), per `standalone-fstats.md`. The downdate kernel is the long pole.

## 6. Risks + parity

PARITY ORACLES: (a) small_linalg chol+solve as the host oracle for
`qpfstats_solve`, diffed vs AT2's `qpfstats_regression` on REAL AADR (box has the
R + `qpfstats.c`). (b) The genotype-path `allsnps:YES` is the HARD case —
different SNP sets per stat means the per-block numerator + per-pair valid count
must match AT2's `f4blockdat_from_geno(allsnps=TRUE, return_matrices=TRUE)`
exactly; the f2-cache path is easier but is NOT the full method. (c) `ridge`
(`0.00001`) must match byte-for-byte. (d) `doscale=YES`/`lambdascale`
(`qpfstats.c:107,575-590`) materially rescales the inputs — confirm whether the
goldens use the C tool (doscale ON) or the R path.

OPEN QUESTIONS: (1) qpfs.pdf §2 describes sigma_i (jackknife-SE) WEIGHTING while
the R io.R path does ridge only — verify whether the C `qpfstats.c` does the
sigma-weighting/scaling or only ridge (lsqmode/doscale/diag knobs at C lines
82,107,76). (2) all-NaN-block -> b=0 bias (io.R warns) — decide steppe policy
(drop vs zero) and match AT2's default. (3) which precision tier the goldens were
generated under.

HARDEST PART: the missing-block Cholesky-downdate as a batched GPU kernel without
falling back to a host per-block loop (the CPU-bound trap).

VALIDATION: REAL AADR only. AADR TGENO goldens are known-corrupt (memory) —
regenerate via convertf>=8.0.0 TGENO->PA then AT2 qpfstats.

## Sources

PUBLISHED: `qpfs.pdf` (N. Patterson 2020-06-18, /workspace/AdmixTools_src/, read
via pdftotext per prior research) — basis dim `n(n-1)/2`, eq(1), sigma-weighted
LSQ, 2-pass jackknife. README.qpfstats (DReichLab) — allsnps:YES rationale,
scale:YES. AT2 R io.R: `qpfstats()` `:3979-4052`, `qpfstats_regression()`
`:3849-3953`, `construct_fstat_matrix()` `:3745-3767`,
`f4blockdat_from_geno(return_matrices=TRUE)` `:3276,4014`. DReichLab C
`/workspace/AdmixTools_src/src/qpfstats.c` (VERIFIED on box5090: `doscale=YES`
:107, `lsqmode=NO` :82, `diag`/`f2diag` :76/:129, `f2weight` :83, lambdascale
:575-590). STEPPE CODE (read at HEAD): `backend.hpp:643,774,728,705,509,130,163`;
`cuda_backend.cu:1411,2262,2296,2299,2312,3309,3342,3370,1960,2018`;
`small_linalg.hpp:145,165,204`; `jackknife.hpp:19`; `dstat.cpp:285`;
`handles.hpp:535`. Prior research: `standalone-fstats.md` (qpfstats MEDIUM-HIGH,
on the cuSOLVER/emulated-FP64 seam).
