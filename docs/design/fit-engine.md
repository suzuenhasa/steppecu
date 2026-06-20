# qpAdm FIT ENGINE — design + milestone build-order + first-milestone frozen contract

**Status:** DESIGN (Phase 2, S3–S8). Nothing here is built. The S0–S2 precompute is BUILT through
M5 on `main` and produces a device-resident `steppe::device::DeviceF2Blocks`
(`src/device/device_f2_blocks.hpp`); this document designs the engine that consumes it.

**Scope.** The qpAdm/qpWave fit engine: stages S3 (f4 from f2) → S8 (model-space search), per
`docs/architecture.md` §5. This is a contracts/architecture deliverable so the milestone build
(Contracts → per-file Implement → Build → Verify) can start from a frozen first-milestone contract.
No production code is written here.

**Provenance tags used throughout.**
- **[SPEC]** — mandated by `docs/architecture.md` (section cited).
- **[AT2]** — grounded in ADMIXTOOLS 2 source / qpAdm literature (source cited).
- **[PROPOSAL]** — a synthesis decision proposed here, to be ratified when the contract is frozen.

This document synthesizes four review lenses (algebra-dataflow, numerics-precision, seam-rotation,
at2-goldens-validation). Where the lenses disagree, the disagreement is recorded as an open question
in §6 and a default is proposed; nothing contradictory is silently resolved.

---

## 0. The one load-bearing correction (read first)

The workflow brief and the original spec §5 S6 note (architecture.md:243) say qpAdm's GLS weight
solve is **"a single Cholesky `potrf` + solve, NOT an iterative sweep; any earlier '~20 sweeps'
language was wrong."** Two of the four lenses (numerics-precision, at2-goldens-validation) found this
is **inconsistent with the ADMIXTOOLS 2 master source**: `qpadm()` extracts weights from a rank-`r`
factorization `X ≈ A·B` that is refined by an **alternating-least-squares loop (`opt_A`/`opt_B`,
default 20 iterations) with a `fudge = 1e-4` diagonal ridge**, and only the *final constrained
weight extraction* (`solve(rhs, lhs)` on a `(r+1)×(r+1)` SPD system) is a single Cholesky.
[AT2 `qpadm.R`: https://github.com/uqrmaie1/admixtools/blob/master/R/qpadm.R]

Because the acceptance gate is **bit-tier parity against AT2 goldens** [SPEC §12, §13], this is the
single largest design risk in the fit engine and a contract-blocking decision (OQ-1, §6). The default
proposed here is **reproduce AT2's ALS verbatim in native FP64** for `est`/`chisq`/`w` parity, and
correct the spec §5:243 text. The "single Cholesky" framing survives only as the description of the
final constrained solve and of the `Q` factorization, not of the whole weight fit.

---

## 1. THE FIT-ENGINE ARCHITECTURE

### 1.1 What the precompute hands the fit (the frozen input seam)

The precompute→fit handoff is `steppe::device::DeviceF2Blocks` (`src/device/device_f2_blocks.hpp`),
a move-only CUDA-free opaque handle owning two co-resident FP64 tensors in VRAM on one device:

- `f2_device()` → `const double*`, layout `[P × P × n_block]`, element `(i,j,b)` at flat index
  `i + P·j + P·P·b` (block-major outer, column-major within a slab). **[SPEC §11.1; device_f2_blocks.hpp:56-59]**
- `vpair_device()` → `const double*`, same shape: per-block pairwise-valid SNP count, the **S4
  weighted-block-jackknife weight** [SPEC §5 S2 caveat (a); fstats.hpp:54-60].
- `block_sizes` (host `std::vector<int>`, length `n_block`), `P`, `n_block`, `device_id` (plain host
  scalars; the handle is CUDA-free so `core` can hold and forward it).
- `to_host()` → host `F2BlockTensor` (`include/steppe/fstats.hpp`) is the ONLY D2H; used by the CPU
  oracle path, the parity test, and bindings.

**Representation finding (algebra-dataflow lens, HIGH).** `f2[i,j,b]` is the **AT2-unbiased per-block
point estimate** (per-block *total*), NOT a leave-one-out replicate. The workflow brief's phrase
"leave-one-out jackknife tensor" describes the *use*, not the *storage*. The totals→LOO conversion
(AT2 `est_to_loo`) is **fit-engine work in S4**, not precompute output. [AT2 `resampling.R`
`est_to_loo`; fstats.hpp:48-52] This is OQ-2 (§6) — confirm nothing already pre-applied LOO (it has
not, per fstats.hpp).

The diagonal `f2(i,i)` carries `−2·hc_i` (within-pop het), never consumed by f3/f4
(off-diagonal only) but kept consistent across backends [fstats.hpp:42-46; backend.hpp:50-59].

### 1.2 The qpAdm algebra (what each stage computes), cited

**Index convention [AT2 `qpadm.R`].** `qpadm()` prepends the target to left: `left = c(target,
sources...)`, so `L_0 = target`, `L_1..L_{nl} = sources` with `nl = length(left) − 1`; `R_0` is the
first right pop, `R_1..R_{nr}` the rest with `nr = length(right) − 1`.

**S3 — f4 matrix X from f2** [AT2 `f2blocks_to_f4blocks`, `qpadm.R`; SPEC §5:219]. Per block `b`:
```
X[i,j,b] = f4(L_0, L_i; R_0, R_j)
         = ½·( f2(L_i,R_0) + f2(L_0,R_j) − f2(L_0,R_0) − f2(L_i,R_j) )[b]
```
Shape `(nl) × (nr)` per block; vectorized `m = nl·nr`; the per-block tensor is `X_blocks [m × n_block]`
[SPEC §5:219]. This is a **pure linear gather+combine of four f2 slabs** — four strided gathers + an
axpy, NOT a true GEMM (matters for precision, §1.4, and for the seam, §1.3).

**S4 — block jackknife → covariance Q** [AT2 `jack_pairarr_stats`, `resampling.R`; SPEC §5:220, §11.2].
The vectorized per-block f4 `[m × n_block]` is jackknifed:
1. totals `tot[k] = weighted.mean(X_blocks[k,:], 1 − bl/n)`, `n = Σ block_lengths`.
2. LOO per entry per block via `est_to_loo`: `loo[k,b] = (tot[k] − X[k,b]·rel_b)/(1 − rel_b)`,
   `rel_b = bl_b/Σbl`.
3. pseudo-deviation with `h_b = n/bl_b`:
   `xtau[k,b] = ( est[k]·h_b − loo[k,b]·(h_b−1) − tot[k] ) / sqrt(h_b − 1)`.
4. covariance `Q = xtau · xtauᵀ / n_block` — an `m × m` matrix (the Busing/Patterson weighted
   block-jackknife form).
5. regularize + invert: `diag(Q) += fudge·tr(Q)` (`fudge = 1e-4`), `qinv = solve(Q)`.

Note `Q` is `m×m` with `m = nl·nr`. With thousands of right pops `m` is large (e.g. `nl=4, nr=40
⇒ m = 117`, `Q` is `117×117`); the §5:222 "low-double-digit" characterization holds only at small
`nr`. At-scale `m` drives the per-model Cholesky/inverse cost (OQ-6, §6).

**S5 — rank test / qpWave (SVD)** [AT2 `qpadm.R`; SPEC §5:221, §5:231]. Initialize a rank-`r`
factorization from `svd(X)`: `B = Vᵀ[:,1:r]`, `A = X·Bᵀ`, then refine by Q-weighted ALS. The test
statistic is the GLS residual quadratic form:
```
E = X − A·B            (m-vector when vectorized)
χ² = vec(E)ᵀ · Q⁻¹ · vec(E)
dof = (nl − r)·(nr − r)
p = pchisq(χ², dof, lower.tail = FALSE)
```
qpWave sweeps ranks `r = 0 … min(nl,nr)−1`; the inferred number of admixture waves is the smallest
non-rejected `r`. **Batched-SVD limit [SPEC §5:231]:** `gesvdjBatched` requires `m,n ≤ 32`; with
thousands of rights `nr > 32` is common → per-model `gesvd` fallback. **Determinism [SPEC §12]:**
`gesvd` is deterministic only for `m ≥ n` — orient `X` so rows ≥ cols; `gesvdjBatched`/`gesvdj` are
covered; `gesvda` is **excluded** (screening-only). **GPU dispatch (BUILT, the FROZEN CONTRACT §0):**
the GPU `CudaBackend` now EXECUTES arbitrary `nl`/`nr` on the device. A model inside the bit-parity
envelope (`nl≤5, nr≤10, r≤4`) runs the UNCHANGED deterministic on-device Jacobi (byte-for-byte 9-pop
golden parity). A model outside it runs the cuSOLVER LARGE path: `large_svd_V` (`cusolverDnDgesvdj`
for both dims ≤ 32, `cusolverDnDgesvd` for > 32) with `X` oriented **rows ≥ cols** (for the common
`nr ≥ nl` case hand `Xᵀ`, then `U(Xᵀ) = V(X)`), native FP64, single statistic stream, plus a dynamic
**VRAM** scratch buffer (sized at runtime from `nl·nr·m·r`) replacing the per-thread fixed local arrays
— so the ALS rank-`r` opt + the `Q⁻¹` quadratic form run for any `m = nl·nr`. The nr>32 NRBIG fixture
(nr=39) now RUNS ON THE GPU and GATES against its golden; the prior "PENDING seam" is removed. The SVD
±sign ambiguity is harmless (ALS + chisq are sign-invariant), so the large path is golden-tier-correct
without bit-parity to the Jacobi.

**S6 — GLS weights** [AT2 `qpadm_weights`, `qpadm.R`; SPEC §5:243 — but see §0]. From the refined `A`
(`nl × r`): `x = t(cbind(A, 1))` (`(r+1) × nl`), `y = c(rep(0,r), 1)`, `rhs = xᵀx`,
`diag(rhs) += fudge·tr(rhs)`, `lhs = xᵀy`, `w = solve(rhs, lhs)`, `weights = w/Σw`. The constrained
solve is the literal "single Cholesky" of §5:243; the surrounding `opt_A`/`opt_B` ALS is iterative (§0).

**S7 — standard errors + p** [AT2 `get_weights_covariance`, `qpadm.R`; SPEC §5:223]. For each LOO block
`b`, re-run `qpadm_weights` on `X_blocks[:,b]`-LOO **reusing the full-data `qinv`** (parity pin — AT2
does NOT re-invert per replicate), giving `wmat[b,:]`; `se = sqrt(diag(cov(wmat)))` with the jackknife
scaling; drop non-finite replicates (`finreps`). `z = weight/se`. So S7's per-model cost ≈ `n_block ×`
the weight solve (OQ-7, §6 — must be a batched device op, not `n_block` host launches).

**S8 — model-space search.** Each model is a self-contained S3→S7 chain over its own pop-subset
gather from the shared resident `f2_blocks`; the rotation shards models across GPUs (§1.6).

### 1.3 Layering (where the fit engine lives) — [PROPOSAL grounding SPEC §4, §5, §17 item 11]

New tree, mirroring the precompute split. `core/qpadm/*` is host-pure C++; **every device op routes
through `ComputeBackend`** (`src/device/backend.hpp`) — `core` never issues a GEMM/SVD/Cholesky
[SPEC §2, §4, §5 table]. CUDA is `PRIVATE` to `steppe_device`, so a CUDA leak here fails the host-only
compile (the §4 layering proof).

```
include/steppe/qpadm.hpp           PUBLIC, CUDA-FREE: QpAdmModel, QpAdmResult, QpWaveResult,
                                    QpAdmOptions; the C++ value shapes. (C-ABI shim per §16/OQ-9.)
src/core/qpadm/
  f4_matrix.{hpp,cpp}              S3: assemble X[m×n_block] from DeviceF2Blocks (orchestration;
                                    the 4-slab gather dispatched via ComputeBackend).   [SPEC §5:219]
  jackknife.{hpp,cpp}             S4: totals + LOO + weighted Q[m×m]; est_to_loo + xtau.  [SPEC §5:220]
  ranktest.{hpp,cpp}              S5: SVD of X → rank, χ², dof.                           [SPEC §5:231]
  gls_solve.{hpp,cpp}             S6: opt_A/opt_B ALS + constrained weight solve.         [SPEC §5:243]
  nested_models.{hpp,cpp}         S7: SE via n_block weight re-fits; p-value; rank-drop.  [SPEC §5:223]
  qpadm_fit.{hpp,cpp}             single-model orchestrator: S3→S7 over ONE device, ONE model.
  model_search.{hpp,cpp}          S8: the rotation — work-queue across Resources::gpus, re-sort.
  model_search_core.{hpp,cpp}     HOST-PURE, CUDA-FREE, GPU-free-testable heart (queue + re-sort
                                    + exception discipline) — mirrors f2_blocks_multigpu_core.hpp.
src/app/commands/qpadm.cpp        CLI wiring; app is the only layer that bridges io→compute.  [SPEC §4]
```

The `model_search_core` split is the proven pattern: it lets a GPU-free host `.cpp` test drive the
rotation against a **fake `ComputeBackend`** with no GPU, exactly as
`tests/unit/test_f2_blocks_multigpu.cpp` drives `compute_multigpu_partials`
(`f2_blocks_multigpu_core.hpp:18-26`).

### 1.4 The precision map (§12) — precision follows conditioning, not shape

**[SPEC §12, §9]** Precision is assigned by the conditioning of the operation. The fit now follows the
**SAME policy as the f2 precompute** (one policy, not two): the default for every fit stage is
`EmulatedFp64{kDefaultMantissaBits}` (= `EmulatedFp64{40}`, the f2 default; Ozaki fixed-slice,
~2.2e-11), and a stage that cannot honor it **falls back internally** to native FP64. Concretely:
the well-conditioned matmul-heavy covariance SYRK **engages the default** (auto-native fallback via
`emulation_honorable` when the build/device cannot honor it — the SAME predicate the f2 GEMMs use);
the cancellation-prone elementwise math (the f4 four-slab combine, the `xtau` centering) is held
**native FP64 ALWAYS** by a fixed carve-out (emulation faithfully forms a product but cannot recover
bits a prior subtraction annihilated — exactly the f2-numerator carve-out); the small ill-conditioned
cuSOLVER solves **default native** and are **promotable** to `EmulatedFp64` under the validated
per-stage seam (they currently fall back because CUDA 13.0 / cuSOLVER 12.0 exposes no FP64-emulated
cuSOLVER mode — see the seam note below); TF32 is screening-only. `qpadm_fit.cpp` passes the unified
`EmulatedFp64{kDefaultMantissaBits}` default into the backend virtuals, and `QpAdmResult::precision_tag`
reports what ACTUALLY ran on the SYRK (`EmulatedFp64` iff emulation was honorable, else `Fp64` — queried
via `capabilities().emulated_fp64_honorable`, not hardcoded).

| Stage | Operation | Precision | Primitive / note |
|---|---|---|---|
| S3 f4 assembly | `X = ½(f2+f2−f2−f2)` four-slab gather | **native Fp64 (cancellation carve-out, ALWAYS)** | gather + axpy; a *second* subtraction after f2 (catastrophic cancellation), held native regardless of the requested precision — the f2-numerator carve-out [SPEC §5:219; OQ-5] |
| S4 covariance Q | `xtau` (centering = a difference) | **native Fp64 (cancellation carve-out, ALWAYS)** | difference of like-magnitude replicates; held native regardless of the requested precision |
| S4 covariance Q | `Q = xtau·xtauᵀ/n_block` (SYRK) | **EmulatedFp64{40} by default (auto native fallback)** | well-conditioned accumulate-many; the canonical §12 covariance-SYRK case; legacy `cublasDsyrk` driven by the handle MATH MODE, engaged via `MathModeScope` + `engage_f2_precision` (capture/apply/restore — no leak to the SPD inverse), routed through the SAME `emulation_honorable` predicate the f2 GEMMs use. LANDED: the fit DEFAULTS this emulated like f2; falls back native when not honorable |
| S5 rank SVD | `svd(X)` + `E'Q⁻¹E` | **native Fp64** (default; promotable) | small, near rank-deficiency, oracle-grade [SPEC §12]; deterministic cuSOLVER, single statistic stream |
| S5 `Q` factor/inverse | factor/invert `Q` | **native Fp64** (default; promotable) | `potrf`/`potri`; explicit `Q⁻¹` if parity needs it (OQ-3, §6); the cuSOLVER math mode is engaged via `CusolverMathModeScope` (default native) |
| S6 weights | `opt_A`/`opt_B` Kronecker GLS + constrained solve | **native Fp64** (default; promotable) | tiny + ill-conditioned; `potrf`/`potrs` on the ridge-regularized systems |
| S7 SE | per-LOO-block weight re-solves; `cov(wmat)` | **native Fp64** solves (default; promotable); cov accumulation may be EmulatedFp64 | reuse S5/S6 primitives |
| S8 screening | rank/feasibility ranking only | **Tf32 allowed**, recomputed in EmulatedFp64/Fp64 before any reported est/se/z/p [SPEC §9, §12] | never emits a reported number |

**Covariance-SYRK precision (LANDED this change).** The S4 covariance SYRK is no longer hardcoded
native — `qpadm_fit.cpp` now passes the unified `EmulatedFp64{40}` default and `jackknife_cov` engages
it on the `cublasDsyrk` via a `MathModeScope` (capture the handle's current math mode) bracketing
`engage_f2_precision` (apply the emulated/fixed-slice mode, or PEDANTIC-native when not honorable),
restoring the captured mode at scope exit so the emulated mode never leaks into the subsequent native
SPD inverse. At single-model scale this is a **policy/consistency change, not a speedup** (xtau is
`n_block × m` = tiny, so emulation gives ~no single-model benefit); the payoff is (a) the fit policy
now MATCHES f2, and (b) S8 inherits it — when the SYRK is batched across thousands of rotation models it
is already on the tensor-core emulated path. The af6a8c2 golden re-passes within tier with the SYRK
emulated (the SYRK is well-conditioned accumulate-many and emulated{40} is ~2.2e-11).

**Determinism contract [SPEC §12, §9].** Statistic-bearing GEMM/SYRK/SVD/Cholesky run on the **single
statistic stream** (`stream_count = 1`; cuBLAS reproducibility voids under concurrency).
`EmulatedFp64` requires an explicit `cublasSetWorkspace` workspace or `build()` rejects
`deterministic && EmulatedFp64` (§9:635, §12). `cusolverDnSetDeterministicMode` on the statistic
solver handle; CI asserts the rank-test routine is in the covered set. The `search_streams = 4` lanes
(§9:585) are throughput-only — any TF32-screened lane's survivors are recomputed before reporting.

**Solve-precision promotion seam [ROADMAP §6; landed this commit].** Native FP64 is free for ONE
model, but the S8 model rotation runs *millions* of small SVD/Cholesky/GLS solves where native FP64 on
Blackwell is a tensor-core-throughput wall (native FP64 ≈ a small fraction of the emulated-FP64/Ozaki
tensor-core rate). So the small solves are **default native but PROMOTABLE to `EmulatedFp64` under a
validated per-stage policy**: the CUDA backend's `CusolverDnHandle` exposes `cusolverDnSetMathMode`
through a scoped RAII (`CusolverMathModeScope` / `engage_solver_precision`, src/device/cuda/handles.hpp),
mirroring the cuBLAS `MathModeScope`/`engage_f2_precision` f2 path and routing the promote/native
decision through the SAME `emulation_honorable` predicate. The per-stage knob is
`ComputeBackend::set_solve_precision` (default native ⇒ the scope targets native and the af6a8c2 golden
parity is byte-for-byte unchanged). **Native FP64 remains the oracle/fallback**, and any promotion is
validated against it per stage at S8 scale before becoming a default. As of this commit the SEAM
exists (real `cusolverDnSetMathMode` engage+restore at the S4 SPD-inverse solve); the *default* is still
native, so nothing in the parity tier changes. NOTE: CUDA 13.0 / cuSOLVER 12.0 `cusolverMathMode_t`
exposes no FP64-emulated mode yet (only `CUSOLVER_DEFAULT_MATH` and `CUSOLVER_FP32_EMULATED_BF16X9_MATH`);
the seam is forward-compatible — an honorable EmulatedFp64 solve request DEGRADES to native with a
one-shot capability tag until a newer cuSOLVER adds the FP64-emulated mode (then the same code promotes
with no change).

### 1.5 Public API (CUDA-free at the seam) — `include/steppe/qpadm.hpp` [PROPOSAL]

Internally a C++ value API; the installed/versioned boundary is the **C ABI with opaque handles**
[SPEC §16 ADR-0008]. The C++ shapes (frozen in M(fit-1), §3):

```cpp
namespace steppe {

struct QpAdmModel {                 // a model = target + left + right, as INDICES into the f2_blocks P axis
    int              target;        // L_0; the target population row/col index (0..P-1)
    std::vector<int> left;          // source pops L_1..L_nl  (name→index resolution is an app/binding concern)
    std::vector<int> right;         // reference/outgroup pops; right[0] is the fixed R_0
    int              model_index = -1;  // STABLE identity for the deterministic S8 re-sort
};

struct QpAdmResult {
    std::vector<double> weight;     // admixture weights w (len == left.size()), Σw = 1
    std::vector<double> se;         // jackknife SE per weight
    std::vector<double> z;          // weight / se
    double              p     = 0;  // tail p-value of the chosen-rank fit
    double              chisq = 0;
    int                 dof   = 0;
    std::vector<double> rank_p;     // qpWave nested rank-test p-values (rank 0..min-1)
    int                 est_rank = 0;
    Status              status   = Status::Ok;  // PER-MODEL value: Ok / RankDeficient /
                                                //   NonSpdCovariance / ChisqUndefined  [SPEC §10]
    Precision::Kind     precision_tag;          // which arithmetic produced this (Tf32 ⇒ provisional, §12)
    int                 model_index = -1;       // echoes the input for deterministic ordering
};

struct QpWaveResult { std::vector<double> rank_p; std::vector<int> dof; std::vector<double> chisq;
                      int est_rank = 0; Status status = Status::Ok; };

struct QpAdmOptions { bool   allow_negative_weights = false;
                      bool   constrained            = false;
                      double fudge                  = 1e-4;   // AT2 parity constant (NOT a magic number; OQ-4)
                      int    als_iterations         = 20; };  // AT2 opt_A/opt_B default (§0; OQ-1)

// Single model (also the unit the search dispatches), reading DEVICE-RESIDENT f2_blocks:
[[nodiscard]] QpAdmResult run_qpadm(const device::DeviceF2Blocks& f2,
                                    const QpAdmModel&, const RunConfig&, device::Resources&);

// qpWave-only rank test:
[[nodiscard]] QpWaveResult run_qpwave(const device::DeviceF2Blocks& f2,
                                      std::span<const int> left, std::span<const int> right,
                                      const RunConfig&, device::Resources&);

// S8 ROTATION — the embarrassingly-parallel batch:
[[nodiscard]] std::vector<QpAdmResult> run_qpadm_search(const device::DeviceF2Blocks& f2,
                                      std::span<const QpAdmModel>, const RunConfig&, device::Resources&);

}  // namespace steppe
```

Key seam decisions [PROPOSAL]:
- **Input is `DeviceF2Blocks` (device-resident), not `F2BlockTensor`** — the fit reads `f2_device()`/
  `vpair_device()` in VRAM, zero D2H, honoring "keep f2_blocks GPU-resident" [SPEC §5:245, §11.1]. A
  `const F2BlockTensor&` overload exists for the CPU/host oracle path and Python (uploads once via
  `upload_f2_blocks_to_device`).
- **Models reference populations by index** into the P axis — no strings at the compute seam. Each
  per-model fit is then a **pop-subset gather** of rows/cols `{target} ∪ left ∪ right` from the
  resident slabs.
- **Domain outcomes are per-model `status` values, not exceptions** [SPEC §10] — essential: in a
  search of thousands of models, many *will* be rank-deficient and the search must record-and-continue.

### 1.6 The S8 rotation / multi-GPU model [SPEC §11.4 point 2; PROPOSAL for the mechanism]

This WAS framed as multi-GPU's decisive payoff. **MEASURED REALITY (real AADR, P=600; commit 2a0c020 /
`TODO(multigpu-host-bounce)`):** on the consumer 5090s the rotation is **host-bounce-capped** — the
one-time `f2` replication is ~8.72 GB / ~3.8 s through host (no GPU↔GPU P2P on GeForce), so G2/G1 only
reached **~1.21× at 9086 real models** (no 1.5× crossover in range). **RUN THE FIT/ROTATION SINGLE-GPU
on box5090.** Multi-GPU's real payoff needs P2P hardware (rtxbox) or the deferred **per-device-precompute**
fix (each GPU builds its own f2 from the genotype stream → zero cross-GPU transfer) [SPEC §0, §11.4].

- **Replication precondition.** `f2_blocks` must be resident on every device in `Resources::gpus`.
  G==1 is trivial. For G≥2, broadcast the resident `DeviceF2Blocks` to each device once (NCCL
  Broadcast or `cudaMemcpy`/`cudaMemcpyPeer` — order-independent, parity-safe). This is the only
  cross-GPU traffic in S8, one-time — **but NOT "off the critical path" on no-P2P cards:** MEASURED
  ~8.72 GB / ~3.8 s host bounce at P=600 on the 5090s (the cold cost dominates until impractical N).
  On P2P hardware `cudaMemcpyPeer` avoids host; the deferred fix is per-device precompute (no
  replication at all) — `TODO(multigpu-host-bounce)` [SPEC §11.4].
- **Dispatch = dynamic atomic work-queue over model indices** [SPEC §11.4: "a dynamic atomic
  work-queue over model indices (load-balances uneven per-model cost)"]. Concretely, mirroring
  `compute_multigpu_partials`: one `std::jthread` per device (`cudaSetDevice`-bound), a single
  `std::atomic<int> next_model` the workers `fetch_add` to claim work — dynamic because per-model cost
  varies wildly (an `nr > 32` model drops to the per-model `gesvd` fallback, far more expensive than a
  batched one [SPEC §5:231]). Each worker runs `run_qpadm` wholly on its own device → **zero inter-GPU
  traffic**, internal reductions stay single-GPU and inherit single-GPU parity [SPEC §11.4, §12]. Each
  worker writes only `results[model_index]` (pre-sized vector, distinct elements, no aliasing) — the
  race-free discipline of `f2_blocks_multigpu_core.hpp:79-88`. `exception_ptr` per worker, rethrow the
  lowest-index *fault* after join (domain outcomes are not faults; they go in `status`).
- **Per-model CUDA graph capture** [SPEC §5:245, §11.3]. The fit replays an identical kernel sequence
  thousands of times; capture once per device, `cudaGraphExecUpdate` for param-only changes between
  models of the same shape. Bucket models by `(nl, nr)` shape so a graph is reused within a bucket
  (analogous to the precompute's block-size bucketing). Fallback-path (`nr > 32`) models may defeat
  capture and run eagerly (OQ-8, §6).
- **`search_streams` lanes** [SPEC §9:585]. The per-model SVD fallback is launch-bound; keep
  `search_streams = 4` independent fit lanes per device to hide per-launch latency. These are
  throughput-only; the single-statistic-stream rule still binds any reported number.
- **Deterministic re-sort** [SPEC §11.4, §18]. `model_index` is the result slot; the pre-sized-slot
  write *is* the re-sort, so the returned ordering is identical regardless of which GPU/lane/wall-clock
  produced which model — the §18 multi-GPU-parity DoD bullet.

`cuSOLVERMp` stays deferred: `Q`/`X` are low-double-digit; the right parallelism is across many
independent small fits, not within one distributed solve [SPEC §11.4]. Multi-node is out of scope.

**Resources extension (prerequisite, OQ-10, §6).** Today `PerGpuResources`
(`src/device/resources.hpp:128-142`) carries only `device_id`, `backend`, `caps`; cuSOLVER and the
search-stream pool are explicitly deferred ("cuSOLVER lands with the S4–S8 fit engine, a later
workflow"; resources.hpp:119-122). The spec §9 sketch (architecture.md:618-626) *does* list
`CusolverDnHandle solver` and a `StreamPool streams`. The fit engine's first device-side job is to
extend `PerGpuResources` with the per-device cuSOLVER handle + search-stream pool.

### 1.7 New `ComputeBackend` methods (the device primitives the fit needs) [PROPOSAL]

All added to the CUDA-free `backend.hpp` seam as **non-pure virtuals that throw in the base** (the
established pattern at backend.hpp:281-291), so `CpuBackend` overrides with Eigen and `CudaBackend`
with cuSOLVER/cuBLAS. All consume `DeviceF2Blocks` device pointers (zero D2H):

- `assemble_f4(const DeviceF2Blocks&, left_idx, right_idx) → X_blocks [m × n_block]` (device-resident) — S3.
- `jackknife_cov(X_blocks, vpair, block_sizes) → {x_total[m], Q[m×m]}` — S4 (LOO + weighted SYRK).
- `rank_test(X, Qinv, r) → {chisq, dof, U, S, V}` — S5.
- `gls_weights(X, Qinv, r, opts) → {w, A, B}` — S6 (the ALS + constrained solve).

Each takes a `Precision` that governs **only** the matmul sub-steps; the SVD/Cholesky/weight solves
**default native FP64** internally and are **promotable** to `EmulatedFp64` via the per-stage
`set_solve_precision` seam (§1.4; ROADMAP §6) under a policy validated against the native oracle. The
matmul `Precision` passed here does not govern the solve conditioning (the §12 conditioning rule is a
backend invariant); native FP64 stays the default and the oracle/fallback.

---

## 2. MILESTONE BUILD-ORDER (smallest-first; dependency order; each with its acceptance gate)

Each milestone is buildable as Contracts → per-file Implement → Build → Verify. Correctness before
speed [SPEC §2]: the native-FP64 CPU oracle is built first and is the seam every GPU path is diffed
against [SPEC §13, §18]. **Nothing below is built.**

| Milestone | Builds | Depends on | Acceptance gate |
|---|---|---|---|
| **M(fit-0) — contract + oracle scaffold** | `include/steppe/qpadm.hpp` (frozen, §1.5); the `core/qpadm/` file skeleton; `CpuBackend` reference path for S3 + S6 only (S4 unweighted Q, full-rank assumed); `model_search_core` host-pure stub. | precompute (DONE); `DeviceF2Blocks`. | Compiles GPU-free; layering grep + allocation-allowlist pass [SPEC §18]; the frozen header reviewed and locked. |
| **M(fit-1) — f4 + single GLS fit, ONE model** | S3 `assemble_f4` + S4 (weighted `Q` via `Vpair`) + S6 (`opt_A`/`opt_B` ALS + constrained weights) on `CpuBackend`, native FP64, ONE model, full rank `r = nl−1`. (THE FIRST-MILESTONE CONTRACT, §3.) | M(fit-0). | `weight` matches an **AT2 `qpadm()` golden** within the tight tier (`rtol ~1e-6`); `X` and `Q` match the CPU oracle at the seam [SPEC §12, §13]. |
| **M(fit-2) — rank test (S5) + p-values** — **BUILT** (CPU oracle + GPU deliverable; BOTH the nr≤32 9-pop AND the nr=39 NRBIG GPU GATEs green) | `rank_sweep` virtual (backend.hpp): the qpWave/qpAdm rank sweep r=0..rmax (`χ²`, `dof`, `p`), the AT2 `res$rankdrop` nested table (`f4rank`/`chisqdiff`/`dofdiff`/`p_nested`), the AT2 `res$popdrop` leave-one-LEFT-SOURCE-out table (verbatim `admixtools::drop_pops`: SUBSET the rows of the already-computed `f4_est` + SUBSET the FUDGED `qinv[ind,ind]`; NO re-gather/re-jackknife — the host orchestrator `ranktest.cpp` is backend-agnostic so the SAME path serves the CPU oracle AND the GPU deliverable), and `run_qpwave` (no target prepend). `rank_alpha` (default 0.05) names the rank decision (`f4rank` = smallest r with p>alpha). **GPU `CudaBackend::rank_sweep` (THE deliverable, BUILT): uploads x_total+Qinv ONCE, builds dXmat ONCE, then per r runs the fit (seed→ALS→weights+chisq) and reads back `chisq(r)` — the rank test RUNS ON THE GPU (f2 resident, on-device SVD/ALS/chisq, no host round-trip); the host forms dof/p/rankdrop/f4rank; `rank_Q` is `core::jacobi_svd(Q)` host-side (observability, bit-identical to the oracle).** **DISPATCH (the FROZEN CONTRACT §0 HYBRID): a model inside the bit-parity envelope (nl≤5, nr≤10, r≤4) runs the UNCHANGED on-device Jacobi small path; a model outside it (e.g. NRBIG nr=39) runs the cuSOLVER LARGE path — `large_svd_V` (cuSOLVER `gesvdj` for both dims≤32, `gesvd` for >32, oriented rows≥cols for determinism, native FP64) + a tiny seed-from-V kernel + VRAM-scratch (dynamically sized from nl/nr/m/r) `*_large` ALS/weight/chisq kernels (no per-thread fixed local arrays). `RankSweep.svd_path` REPORTS the EXECUTED routine (1 = gesvdj/Jacobi when nl,nr≤32; 2 = gesvd when >32).** The SVD V-column ±sign ambiguity is HARMLESS (ALS + the chisq quadratic form are sign-invariant), so the large path holds the golden chisq tier without bit-parity to the Jacobi. | M(fit-1) + M(fit-4). | **GPU GATE green** vs `golden_fit0.json` `res$rankdrop`/`res$popdrop` (the 9-pop nr≤32 small path) RUNNING ON THE GPU: `f4rank`=1 EXACT, `dof`/`dofdiff` EXACT, `chisq`/`chisqdiff` tight (rtol 1e-6; measured |Δ|≤7e-11), `p`/`p_nested`/rank-decision loose (rtol 1e-3), rankdrop+popdrop tables row-for-row; GPU == CpuBackend oracle to 1e-9; `svd_path`=1, `rank_Q`=10. **nr>32 NRBIG GPU GATE green (the LARGE path): golden PINNED — `golden_fit1_NRBIG.json` + `fixtures/f2_fit1_NRBIG.bin` (REAL AADR v66.p1_HO, same `geno_sha256` 7af8c2f5…, admixtools 2.0.10, R 4.3.3; target=England_BellBeaker, left=2, right=40 ⇒ nr=39 > 32, P=43, nb=701; `f4rank=1`). The GPU rank_sweep RUNS ON THE GPU (f2 RESIDENT `devbig`, cuSOLVER gesvd large path) and matches the golden: `f4rank`=1, rankdrop `dof` 38/78, `dofdiff` 40, `chisq` 52.7043/190.836 (rtol 1e-6), `chisqdiff`, popdrop 00/01/10; `svd_path`=2 (gesvd EXECUTED); GPU == CpuBackend oracle localizer ≤1e-7.** [SPEC §12] |
| **M(fit-3) — jackknife SE (S7)** | S7: `n_block` LOO weight re-fits reusing full `qinv`; `cov(wmat)` → `se`, `z`; `finreps` non-finite drop; batched over the block axis (not `n_block` host launches). | M(fit-2). | `se`/`z` in the loose tier (`rtol` **derived** from observed jackknife spread + margin, recorded in golden meta, not a magic constant) [SPEC §12, §18]. |
| **M(fit-4) — CUDA backend, single GPU** | `CudaBackend` overrides of `assemble_f4`/`jackknife_cov`/`rank_test`/`gls_weights`; `PerGpuResources` extended with cuSOLVER + search-stream pool; EmulatedFp64{40} on the S4 SYRK only; deterministic cuSOLVER + single statistic stream. | M(fit-3); resources extension (OQ-10). | GPU == CPU oracle at the `X`/`Q`/`w`/`χ²`/`se` seams within tier; AT2 golden re-passes on the GPU path; `cublasSetWorkspace` present under `deterministic` [SPEC §12, §18]. |
| **M(fit-5) — domain outcomes** | Rank-deficient / non-SPD / χ²-undefined returned as per-model `status` values; `STEPPE_ERR_*` taxonomy. | M(fit-4). | Domain-outcome test: a deliberately collinear model returns `RankDeficient` as a value, not a crash [SPEC §10, §13]. |
| **M(fit-6) — S8 rotation, G==1 then G==2** — **BUILT** (the model-space ROTATION on the GPU(s), GENUINELY BATCHED + multi-GPU sharded, vs a REAL AADR AT2 rotation golden; full ctest green) | `run_qpadm_search` x2 (device-resident + host-oracle, `include/steppe/qpadm.hpp`); `core/qpadm/model_search_core.{hpp,cpp}` (`plan_model_shards` — the count-balanced CONTIGUOUS model→device tiling, the pre-sized-slot re-sort is implicit); `core/qpadm/model_search.{hpp,cpp}` (the orchestrator: G==1 fast path; G≥2 jthread-per-device fan-out, one-time `f2` broadcast via `to_host`→`upload_f2_blocks_to_device`, per-worker `exception_ptr` + lowest-g rethrow, race-free pre-sized-slot writes; `fit_shard` PARTITIONS each device's models into the SMALL-path bucket → the device-BATCHED `be.fit_models_batched` virtual, and the >32/large tail → the per-model `fit_models_batched_default`). **THE BATCHED DEVICE PATH (the deliverable):** `CudaBackend::fit_models_batched` (cuda_backend.cu) buckets the models by `(nl,nr,r)` and fits each bucket of B models in ONE batched dispatch — NOT a per-model loop: a `(k,b,MODEL)`-grid f4 gather reading the RESIDENT f2 with per-model `d_left`/`d_right` index arenas (`launch_assemble_f4_gather_models_batched`) + model-batched loo/total/xtau; the covariance Q = xtau·xtauᵀ/nb via `cublasDgemmStridedBatched` (ENGAGES `precision` — emulated{40} default, the SAME `engage_f2_precision` math-mode scope the f2/jackknife SYRK uses, now strided across the model axis); the per-model SPD inverse Qinv via cuSOLVER `potrfBatched` + column-wise `potrsBatched` (vs a batched identity; per-model devInfo>0 ⇒ NonSpdCovariance); the rank-sweep + constrained weight solve + chisq + popdrop via a MODEL-batched kernel (`qpadm_fit_models_kernel`, one thread per model — the proven `loo_batched_kernel` lift); and the LOO SE via a `(model,block)`-grid kernel (`qpadm_loo_models_kernel`, B·nb parallel threads) + a deterministic variance reduction (`qpadm_se_from_wmat_kernel`, fixed op order ⇒ G=1==G=2 bit-identical, NO atomics). VRAM-budgeted chunking (`free_vram_bytes` − f2 − headroom). `batched_dispatch_count()` (one per bucket chunk) is the observability that the test asserts is ≪ the model count (PROVES batched, not a host loop). **POPDROP RANK FIX (shared M(fit-2) path):** `popdrop_one` fits each (sub-)model at its FULL rank `f4rank = len(surv)−1` (the AT2 popdrop column), NOT the rank-DECISION `rs.f4rank`. | M(fit-5). | **GPU GATE green** vs the REAL AADR AT2 rotation golden `golden_rot.json` + `fixtures/f2_rot.bin` (v66.p1_HO, admixtools 2.0.10, R 4.3.3; target=England_BellBeaker, an 8-pop source POOL, the 6-pop nr≤32 right set; all 28 two-source + 56 three-source subsets = 84 models, boot=FALSE; 30 feasible): per-model **weights rtol 1e-5+atol 1e-5 (the WELL-DETERMINED feasible set matches at max rel-Δ 2.32e-7; the looser tier only covers 2 pathological INFEASIBLE extrapolations — both agree infeasible), p loose (rtol 1e-3), feasible DECISION-match, f4rank EXACT** — 0 f4rank mismatches, 0 feasible mismatches; runs GENUINELY BATCHED on the CudaBackend (precision_tag EmulatedFp64, f2 RESIDENT, `batched_dispatch_count`=2 buckets for the 84-model set, NOT 84 per-model launches, NOT the CpuBackend); **G=1 vs G=2 results BIT-IDENTICAL AND identically ordered** (the determinism gate); **no regression** (9-pop + NRBIG single-model goldens pass via a 1-element `run_qpadm_search` AND `run_qpadm`; NRBIG nr=39 is the LARGE tail → the per-model device path); `model_search_core` GPU-free unit test. **Throughput (box5090, 2x RTX 5090, RELEASE):** 84-model validated set 351 (G=1) / 894 (G=2) models/sec; **[synthetic scale-N RETRACTED — see [[real-data-only-all-results]] / commit 2a0c020]**. On REAL AADR (P=600, real k-subset models) the one-time f2 replication is a ~8.72 GB / ~3.8 s HOST BOUNCE (no P2P on the 5090s), so multi-GPU reached only **G2/G1 ≈ 1.21× at 9086 real models, no 1.5× crossover** — host-bounce-capped ⇒ **RUN SINGLE-GPU**; the multi-GPU payoff is DEFERRED (`TODO(multigpu-host-bounce)`, needs P2P HW or per-device precompute). Parity stands (84/84 real models == AT2) and G=1==G=2 bit-identical. [SPEC §11.4, §18] |

**The CPU oracle (M(fit-1)) needs nothing new from the device layer** — it is layering-legal and
GPU-free-testable [SPEC §2, §13]. The device path (M(fit-4)+) is diffed against this oracle at the
`X`/`Q`/`w`/`χ²` seams [SPEC §12, §13].

---

## 3. THE FIRST-MILESTONE FROZEN CONTRACT — M(fit-1)

**Goal.** A single-model qpAdm fit on the `CpuBackend` reference, native FP64, no multi-GPU, no model
search, no missing blocks — exercising S3 → S4 → S6 → S7 against one AT2 golden. The smallest slice
that freezes the input/output structs and the f2→f4→Q→weights dataflow that S8 later parallelizes.

### 3.1 Files / types / signatures (concretely buildable)

**Public header `include/steppe/qpadm.hpp`** — the C++ value shapes `QpAdmModel`, `QpAdmResult`,
`QpAdmOptions` of §1.5, plus `run_qpadm(const device::DeviceF2Blocks&, const QpAdmModel&, const
RunConfig&, device::Resources&)` and the `const F2BlockTensor&` host-oracle overload. CUDA-free,
indices-not-names. (C-ABI shim deferred until a later milestone; OQ-9.)

**`src/core/qpadm/f4_matrix.{hpp,cpp}`** — `assemble_f4_blocks(host F2BlockTensor or DeviceF2Blocks,
left_idx, right_idx) → X_blocks[m × n_block]` + the `tot` weighted-mean point estimate `X[m]`. CPU
oracle = a host loop over the four-slab combine (§1.2 S3); native FP64.

**`src/core/qpadm/jackknife.{hpp,cpp}`** — `jackknife_cov(X_blocks, vpair-or-block_sizes weight) →
{x_total[m], Q[m×m]}`: `est_to_loo` → `xtau` → `Q = xtau·xtauᵀ/n_block`; `diag(Q) += fudge·tr(Q)`
(§1.2 S4). Native FP64.

**`src/core/qpadm/gls_solve.{hpp,cpp}`** — `gls_weights(X, Q, r, QpAdmOptions) → {w[nl], A, B, chisq}`:
`svd(X)` seed (host Eigen), `opt_A`/`opt_B` ALS (default 20 iters, `fudge` ridge), constrained
weight solve, `χ² = vec(E)ᵀ Q⁻¹ vec(E)` (§1.2 S5/S6). Native FP64. (S5 here is the seed+ALS for the
chosen rank `r = nl−1`; the full qpWave rank sweep is M(fit-2).)

**`src/core/qpadm/nested_models.{hpp,cpp}`** — S7 SE: per-LOO-block weight re-solve reusing full
`qinv`, `cov(wmat)` → `se`, `z`; `dof`, `p = pchisq(χ², dof)`. Native FP64.

**`CpuBackend`** — overrides for the four new `ComputeBackend` virtuals using Eigen (`assemble_f4`,
`jackknife_cov`, `rank_test`, `gls_weights`); the GPU backend's overrides are M(fit-4).

### 3.2 What it reads from `DeviceF2Blocks`

For model `(target, left[], right[])`: resolve `L = [target, left...]` (`nl+1` ids), `R = right`
(`nr+1` ids). Gather only `f2[idx(L), idx(R), :]`, `f2[idx(L), R_0, :]`, `f2[L_0, idx(R), :]`,
`f2[L_0, R_0, :]` across all `n_block` slabs (≤ `(nl+1)(nr+1)` distinct f2 scalars per slab) plus
`block_sizes` (the jackknife `block_lengths`; OQ-3). For the oracle path, `.to_host()` once and read
the host `F2BlockTensor`; no `vpair` per-pair divide is redone (f2_blocks already holds the divided
per-block estimate).

### 3.3 Numerics + precision

Entirely **native FP64** (the cleanest parity baseline; what §12's release gate "match the oracle to
all reported digits" is anchored on). `fudge = 1e-4` and `als_iterations = 20` are AT2 parity
constants, named in `QpAdmOptions`, recorded in the golden metadata (OQ-1, OQ-4). No EmulatedFp64,
no TF32, no batched SVD in this slice — those are throughput layers over a correct native core.

### 3.4 The AT2 golden it is validated against + tolerance

**Fixture / model.** AT2 reference model #2: **qpAdm 2-way well-determined** — 1 target, 2 left, ~4
right, `boot = FALSE` (delete-1 jackknife ⇒ deterministic, no RNG → sidesteps the §12
RNG-fragility for the first milestone). Dataset: the small committed AADR subset already used for the
f2 seam (architecture.md:184), reduced to ~tens of pops × enough SNPs for `B ≈ 50–100` blocks.

**Gate (two-tier tolerance, §12).** `|a−b| ≤ atol + rtol·|b|`:
- `weight` — **tight** (`rtol ~1e-9..1e-6`, `atol ~1e-12`).
- `chisq` — tight (computed but the rank sweep is M(fit-2)); `dof` — exact integer.
- `se`/`z`/`p` — **loose** (`rtol ~1e-3`, derived per-fixture from observed spread, recorded in meta).
- `X`, `Q` — diffed against the CPU oracle at the seam (tight) [SPEC §13].

The §13 domain-outcome stub (a rank-deficient model returns `RankDeficient` as a value) lands here so
the error path is in the contract from day one.

---

## 4. THE AT2-GOLDENS PLAN

### 4.1 What to pin (two families)

**Family A — `f2_blocks` parity (precondition gate).** Before any fit golden runs, assert steppe's
`f2_blocks`/`Vpair` for the fixture match AT2's `extract_f2` output bit-for-bit within the tight tier
(the existing M0/M5 trust seam, architecture.md:184). Dump AT2's `f2_blocks` + `Vpair` as a flat
`[P×P×B]` artifact at `tests/golden/f2/<dataset>.bin` so a fit-golden failure is unambiguously a fit
bug, not an f2 regression. **[at2-goldens lens; SPEC §12]**

**Family B — fit-engine goldens (the new surface).** A small matrix of reference models, each pinned
with its full §12 metadata:

| Quantity | AT2 source | Tier [SPEC §12] | Compare |
|---|---|---|---|
| `weights$weight` | `qpadm()$weights` | **tight** `~1e-9..1e-6` | tolerance |
| `weights$se`, `$z` | `qpadm()$weights` | **loose** `~1e-3` (derived) | tolerance |
| `rankdrop$chisq` | `qpadm()$rankdrop` | tight | tolerance |
| `rankdrop$dof` | `qpadm()$rankdrop` | exact | `==` |
| `rankdrop$p`, `$p_nested` | `qpadm()$rankdrop` | loose | tolerance |
| `popdrop$feasible` | `qpadm()$popdrop` | exact bool | `==` |
| qpWave `rankdrop$p` | `qpwave()$rankdrop` | loose (rank decision) | tolerance |
| rank decision @ α | derived | loose, documented; exact-int after thresholding | `==` |
| intermediate `Q` | AT2 `f4` covariance | tight | tolerance (localizes the double-normalization trap, OQ-3) |

**Reference model set [PROPOSAL]:** (1) qpWave 2-way feasible; (2) qpAdm 2-way well-determined (the
M(fit-1) fixture); (3) qpAdm 3-way (exercises `nr` near the gesvdjBatched=32 boundary); (4)
rank-deficient/clade (returns `STEPPE_ERR_RANK_DEFICIENT` as a value); (5) infeasible weights (AT2
`feasible=FALSE`, weight outside [0,1]); (6, stretch, not first-milestone) a `boot=20` bootstrap-SE
variant exercising the RNG-fragile path.

### 4.2 Generation recipe + which box

Golden generation is **CPU-only R** — it does NOT need a GPU box. R + admixtools is not installed
locally; the `rtxbox` SSH host key rotated (ephemeral vast.ai IP per memory). Generate on any box with
R, record the env, commit the artifacts. **[at2-goldens lens]**

```r
# tools/generate_goldens.R  (run on any CPU box with R >= the pinned version)
remotes::install_github("uqrmaie1/admixtools@<PINNED_SHA>")   # pin the exact AT2 commit
library(admixtools)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")          # recorded
set.seed(42)                                                   # only matters for boot=TRUE
extract_f2(pref="<fixture>", outdir="<dir>", blgsize=0.05, maxmiss=0,
           adjust_pseudohaploid=TRUE, apply_corr=TRUE, auto_only=TRUE, overwrite=TRUE)
f2b <- f2_from_precomp("<dir>")
res <- qpadm(f2b, left=L, right=R, target=T, boot=FALSE, fudge=1e-4)   # deterministic
qpw <- qpwave(f2b, left=L, right=R, boot=FALSE)
# dump f2_blocks + Vpair (Family A) and res$weights / res$rankdrop / res$popdrop (Family B)
```

[Sources: extract_f2 ref https://uqrmaie1.github.io/admixtools/reference/extract_f2.html ;
qpadm ref https://uqrmaie1.github.io/admixtools/reference/qpadm.html]

### 4.3 §12-mandated metadata (the golden refuses to run without it)

`tests/golden/qpadm/<dataset>/meta.json` records the six §12-required keys: **R version, `RNGkind`,
ADMIXTOOLS 2 version (pinned commit SHA), `blgsize` (=0.05 Morgans), `boot` (FALSE / integer N)**,
plus the seed, the dataset hash, the L/R/target lists, and the per-fixture **derived loose `rtol`**
(from the measured jackknife-replicate spread + margin). The harness fails fast if any of the six is
missing [SPEC §12, §18]. AT2 has no `seed` argument and cross-version regeneration drifts — a version
bump is a deliberate golden-regeneration event, not an auto-update.

### 4.4 The tolerance gate (comparator)

One comparator `assert_at2_parity(steppe_result, golden, tiers)` implementing `|a−b| ≤ atol +
rtol·|b|` [SPEC §12] with three classes: **exact** (`dof`, `feasible`, rank decision after
thresholding) `==`; **tight** (`weight`, `chisq`, EmuFp64 covariance vs native oracle) `rtol ~1e-6`,
`atol ~1e-12`; **loose** (`se`, `z`, `p`, anything from a non-deterministic cuSOLVER routine or
TF32) `rtol ~1e-3` read from `meta.json` (derived, not hard-coded; §18).

---

## 5. SUMMARY OF SPEC-MANDATED VS PROPOSED

**Spec-mandated [SPEC]:** the S3–S8 stage ownership + `ComputeBackend` routing (§5 table); the f4
matrix shape `[(nL−1)(nR−1) × n_block]` and `Q [m×m]` (§5:219/§5:222); the batched-SVD limit + per-model
`gesvd` fallback + `m>n` determinism orientation (§5:231, §12); the precision map (native FP64 elementwise
+ ill-conditioned solves; EmulatedFp64 covariance SYRK; TF32 screening-only) (§9, §12); single statistic
stream + deterministic cuSOLVER + `cublasSetWorkspace` (§12); the S8 dynamic work-queue, zero inter-GPU
traffic, deterministic re-sort, bit-identical-across-G DoD (§11.4, §18); the per-model domain-outcome
status taxonomy (§10); the two-tier tolerance with derived loose `rtol` and the six metadata keys (§12);
the C-ABI public boundary (§16); `Vpair` as the S4 jackknife weight (§5 S2 (a)).

**Proposed here [PROPOSAL]:** the `core/qpadm/` file layout and the `model_search_core` split; the four
new `ComputeBackend` virtuals; the `QpAdmModel`/`QpAdmResult`/`QpAdmOptions` shapes and the
indices-not-names + device-resident-input seam; the milestone build-order and acceptance gates; the
reference-model set; reproducing AT2's `opt_A`/`opt_B` ALS verbatim (the §0 default).

**Grounded [AT2]:** every formula in §1.2 (f4 combine, `est_to_loo`/`xtau`/`Q`, `E'Q⁻¹E`, `dof`,
`qpadm_weights`, `get_weights_covariance`, `fudge`, the 20-iter ALS) — cited to `qpadm.R` /
`resampling.R` and the qpWave/qpAdm references.

---

## 6. OPEN QUESTIONS + RISKS (for the contract freeze)

- **OQ-1 [CRITICAL — blocks freezing S6]. AT2's weight solve is iterative.** `opt_A`/`opt_B` (20
  iters + `fudge` ridge) contradicts spec §5:243 "single Cholesky, NOT iterative" (§0). Decide:
  (a) reproduce AT2's ALS verbatim for bit-parity on goldens [proposed default], or (b) prove a
  closed-form rank-`r` GLS equivalent and accept a looser weight tolerance. The acceptance gate is AT2
  goldens (§12), so (a) is safe — but the spec §5:243 text must be corrected. [numerics + at2-goldens
  lenses; AT2 `qpadm.R`]

- **OQ-2 [HIGH]. Representation: totals, not LOO.** `f2_blocks` stores per-block *totals*; S4 must do
  the totals→LOO conversion (`est_to_loo`) + the `tot` weighted mean to match AT2 bit-for-parity.
  Confirm nothing pre-applied LOO (it has not, per fstats.hpp). [algebra lens]

- **OQ-3 [HIGH]. Block-weight semantics — `Vpair` vs `block_sizes`, and the double-normalization
  trap.** AT2's jackknife `block_lengths` is the **per-block SNP count** (`get_block_lengths`), a
  single scalar vector for the whole f4 array — NOT the per-pair `Vpair`. But spec §5 S2 (a) calls
  `Vpair` "the S4 weighted-block-jackknife weight." These differ. **Recommendation:** weight by
  `block_sizes[b]` (AT2-exact); keep `Vpair` only for missing-block exclusion (`est_to_loo_nafix`) and
  as a per-pair validity flag. The per-block divide (S2) and the S4 weighting must compose to AT2's
  `f2_blocks` definition **without double-normalizing**. Add an intermediate golden on `Q` itself so a
  double-normalization localizes to S4. [algebra + seam + numerics + at2-goldens lenses; SPEC §5 S2(a),
  §12]

- **OQ-4 [HIGH]. `fudge = 1e-4` is a load-bearing parity constant** inside `opt_A`/`opt_B`, the `Q`
  inversion, AND the weight normal-equations — its placement/order changes the conditioning of every
  solve. Pin as a named config constant (no magic number, §2), match AT2 exactly, record in golden meta.
  [numerics + at2-goldens lenses; AT2 `qpadm.R`]

- **OQ-5 [MEDIUM]. Precision of the f4 four-term difference.** `(a+b−c−d)/2` is a *second* subtraction
  after f2 (mild cancellation). **Recommendation:** native FP64 for S3 (trivially cheap, not a true
  GEMM, no emulation benefit); reserve EmulatedFp64 for the S4 SYRK only. Confirm. [numerics + algebra
  lenses; SPEC §12]

- **OQ-6 [MEDIUM]. At-scale `m` invalidates "low-double-digit."** With thousands of rights `m =
  (nl)(nr)` is hundreds–thousands; `Q` is `m×m`, `Q⁻¹` is `O(m³)`, VRAM `m²·8` (§11.2). Size the S8
  working set for the worst-case `m`; the SVD is definitely off the batched path. [numerics lens; SPEC
  §5:222, §11.2]

- **OQ-7 [HIGH for S8 throughput]. S7 SE cost asymmetry.** AT2's SE is `n_block` (≈700) weight
  re-fits per model. The per-model graph must include the **n_block-batched** LOO solves over the block
  axis (not host loops), or S8 throughput collapses. Confirm S7 is a batched device op. [seam lens; AT2
  `get_weights_covariance`]

- **OQ-8 [MEDIUM]. CUDA-graph capture vs `cudaGraphExecUpdate` topology limits.** Models of differing
  `(nl,nr)` or differing SVD path (batched vs `gesvd` fallback) are different topologies. Shape-bucket
  to reuse graphs; the bucket granularity + re-instantiate cost is unmeasured — fallback-path models
  (`nr > 32`) may defeat capture and run eagerly. [seam lens; SPEC §5:245, §11.3]

- **OQ-9 [MEDIUM]. C-ABI for a *vector* of results.** `run_qpadm_search` returns
  `std::vector<QpAdmResult>`; the C boundary (§16) needs an opaque `steppe_qpadm_results_t*` + index
  accessors. Cross-cutting; design once. Deferred past M(fit-1) (which freezes only the C++ shapes).
  [seam lens; SPEC §16]

- **OQ-10 [PREREQUISITE]. `Resources` lacks cuSOLVER + the search-stream pool.**
  `resources.hpp:119-122` defers them; spec §9 (architecture.md:618-626) lists them. Decide: extend
  the existing `PerGpuResources` (preferred, DRY) vs a parallel `FitResources`. Hard prerequisite for
  any device-side fit work (M(fit-4)+). [seam lens; SPEC §9]

- **OQ-11 [MEDIUM]. Device-resident vs streamed/tiered f2_blocks at large P.** At thousands of pops
  `f2_blocks` is tiered to host/disk (§11.1/§11.2) and not broadcast whole. Does S8 support a
  tiered/streamed input, or is the first cut resident-tier-only? **Recommendation:** resident-only for
  the first cut; state the boundary explicitly. [seam lens; SPEC §11.1, §11.2]

- **OQ-12 [LOW]. Missing-block handling.** AT2 `est_to_loo_nafix` excludes NA blocks; a
  `vpair[i,j,b]==0` makes that f4 entry NA for that block — must NOT impute 0 (biases toward 0,
  inflates variance). Decide whether the first milestone assumes no missing blocks (simplifies) or
  handles NA from day one. **Recommendation:** assume no missing blocks for M(fit-1); add NA handling
  before the at-scale search. [algebra lens; AT2 `resampling.R`]

- **OQ-13 [LOW]. `pchisq` parity.** The p tier is loose, but `pchisq` is a deterministic special
  function. Decide host implementation (Boost vs own) and whether `p` is compared at all or only the
  rank decision. **Recommendation:** compare `chisq` tight, `p` loose, rank decision exact. [at2-goldens
  lens]

---

## 7. SOURCES

- ADMIXTOOLS 2 `qpadm.R` (f4 matrix, SVD rank test, `qpadm_dof`, `E'Q⁻¹E`, `opt_A`/`opt_B`,
  `qpadm_weights`, `get_weights_covariance`, `qinv=solve(f4_var)`, `fudge`):
  https://github.com/uqrmaie1/admixtools/blob/master/R/qpadm.R
- ADMIXTOOLS 2 `resampling.R` (`jack_pairarr_stats`, `est_to_loo`, `est_to_loo_nafix`):
  https://github.com/uqrmaie1/admixtools/blob/master/R/resampling.R
- AT2 qpWave/qpAdm vignette + references: https://uqrmaie1.github.io/admixtools/articles/qpadm.html ,
  https://uqrmaie1.github.io/admixtools/reference/qpadm.html ,
  https://uqrmaie1.github.io/admixtools/reference/qpwave.html ,
  https://uqrmaie1.github.io/admixtools/reference/extract_f2.html
- Harney, Patterson et al. 2021, *Assessing the Performance of qpAdm* (Genetics 217:4):
  https://academic.oup.com/genetics/article/217/4/iyaa045/6070149
- Patterson qpWave/qpAdm documentation: https://gensoft.pasteur.fr/docs/AdmixTools/6.0/pdoc.pdf
- cuSOLVER `gesvdjBatched` m,n≤32 limit:
  https://forums.developer.nvidia.com/t/the-origin-of-the-m-32-and-n-32-limitations-in-gesvdjbatched/266434
- cuSOLVER 13.x deterministic-mode covered routines: https://docs.nvidia.com/cuda/cusolver/index.html
- Spec: `/home/suzunik/steppe/docs/architecture.md` §4, §5 (S3–S8, :219/:222/:231/:243), §9, §10,
  §11.1/§11.2/§11.4, §12, §13, §16, §17 item 11, §18.
- Seams: `include/steppe/fstats.hpp`, `src/device/backend.hpp`, `src/device/device_f2_blocks.hpp`,
  `src/device/resources.hpp`, `include/steppe/config.hpp`,
  `src/core/fstats/f2_blocks_multigpu_core.hpp`.
