# GROUP 12 — Launch config & indexing — ROLLUP

Tasks: 12.1 block dim not a multiple of 32 · 12.2 grid dim hardcoded vs ceil-div · 12.3 missing grid-stride loop in a one-elem-per-thread kernel · 12.4 launch config baked-in vs `cudaOccupancyMaxPotentialBlockSize` · 12.5 compute-cap / device-property assumptions hardcoded vs queried.

Scope = kernel. 13 in-scope units (all under `src/device/cuda/`).

## 1. Coverage

| Units in scope | Clean | With findings |
|----------------|-------|---------------|
| 13 | 10 | 3 |

Clean (10): block_sink, check, cuda_backend, device_buffer, device_f2_blocks, device_partial, f2_blocks_kernel, f2_blocks_out, p2p_combine, pinned_buffer.
- Most clean units have NO `__global__` kernels or `<<<>>>` launches (host-side transport / RAII / declaration-only TUs) — all five tasks vacuous.
- `cuda_backend` and `device_f2_blocks` explicitly verified 12.5: device is runtime-queried, not hardcoded.

With findings (3): decode_af_kernel, f2_block_kernel, qpadm_fit_kernels.

## 2. Counts by task + severity

Total findings: 6 (0 HIGH / 2 MED / 4 LOW).

| Task | HIGH | MED | LOW | Total |
|------|------|-----|-----|-------|
| 12.1 | 0 | 0 | 0 | 0 |
| 12.2 | 0 | 0 | 1 | 1 |
| 12.3 | 0 | 0 | 2 | 2 |
| 12.4 | 0 | 1 | 1 | 2 |
| 12.5 | 0 | 1 | 0 | 1 |
| **All** | **0** | **2** | **4** | **6** |

By severity: HIGH 0, MED 2, LOW 4.

## 3. Findings (HIGH first — none; then MED, then LOW)

### MED
- [12.5][MED] qpadm_fit_kernels.cu:1295,1309,1323,1344,1375,1387,1400,1549 — all 8 grid-stride launch wrappers clamp `gridDim.x` to the hardcoded literal `65535` (the legacy/pre-Fermi x ceiling, = the current y/z limit), never querying `cudaDevAttrMaxGridDimX`; the real sm_120 x max is 2^31-1, so the clamp is ~32000x too conservative. Not a correctness bug (each is a grid-stride loop, so residual work is still covered), but at the S8 rotation envelope (millions of model-batched threads) it forces long grid-stride spans instead of fanning across the device — real under-utilization on the target box. Suggested: query the attribute once (or a named constexpr = 2147483647) and clamp to that; keep the grid-stride net.
- [12.4][MED] qpadm_fit_kernels.cu:1293..1547 (15 sites) — every launch block size is a hand-picked baked-in literal (256/128/64), never derived from occupancy. The heaviest, high-register kernels `qpadm_fit_models_kernel` (block=64, :1358) and `loo_large_batched_kernel` (block=64, :1547) are exactly the ones register/local pressure bounds, so a wrong hand-pick spills or wastes an SM. Suggested: derive block size for the high-register kernels via `cudaOccupancyMaxPotentialBlockSize`; leave the bandwidth-bound elementwise gather/loo/xtau kernels at fixed 256/128.

### LOW
- [12.3][LOW] qpadm_fit_kernels.cu:1104,1358-1359 (qpadm_fit_models_kernel / launch_qpadm_fit_models_batched) — the only model-batched kernel that is NOT grid-stride: one-thread-per-model with `if (model >= n_models) return` and grid `(n_models+block-1)/block` with NO 65535 clamp, while every sibling IS grid-stride AND clamped. Never under-covers (grid sized from n_models); failure only at n_models > ~1.37e11 (unreachable). Finding is the inconsistency — the heaviest per-thread kernel uniquely relies on full-grid coverage with no fallback. Suggested: give it the same grid-stride form for uniformity / future-clamp robustness.
- [12.2][LOW] qpadm_fit_kernels.cu:1334 (launch_add_fudge_diag_models_batched) — grid dim is raw `n_models` (`<<<n_models, block>>>`), the deliberate one-block-per-model mapping required by the `__syncthreads()` per-block shared-memory trace; correct, not a ceil-div violation, and bounded under 2^31-1 at the S8 envelope. Suggested: if the 65535/maxGridDimX clamp is centralized (12.5), apply the same ceiling here for consistency.
- [12.4][LOW] f2_block_kernel.cu:319,379 — block dim baked into the fixed square `dim3(kCdivBlock,kCdivBlock)`=256 rather than derived from the occupancy API. Deliberate, named, project-wide square-f2-tile choice (launch_config.hpp:107-113); both kernels are bandwidth/cancellation-bound elementwise passes where occupancy tuning buys little. Hygiene only. Suggested: leave as-is; if ever revisited, derive once in launch_config.hpp, not per-TU.
- [12.3][LOW] decode_af_kernel.cu:106 — the x-axis (SNP/`M`) grid extent is the only launch axis lacking a fail-fast over-limit guard: y rides `grid_for(P, kDecodeBlockY)` (asserts ≤ kMaxGridY) and the M4 batch rides `grid_z_extent` (asserts ≤ kMaxGridZ), but `gridDim.x = cdiv(M, kDecodeBlockX)` has no assert it's ≤ kMaxGridX. Not a present-day bug (the `long`→`unsigned` cast can't truncate for the documented M envelope, or even M=2^31); the gap is the asymmetry — the one axis allowed to grow toward 2^31 lacks the sibling cap-assert, so an out-of-spec M (≥ ~6.9e10) would silently truncate instead of getting the precise "exceeds grid limit" message. Suggested: route the x extent through `grid_for`'s `max_grid` with kMaxGridX (or an analogous long-aware assert).

## 4. Cross-cutting patterns

1. **No correctness defects in Group 12.** Zero HIGH. Every elementwise/feeder kernel is one-elem-per-thread with a correct tail guard AND a grid sized from the input extent via the shared `cdiv`/`grid_for` launch-config home, so the grid always fully covers the input (12.3 "breaks when input exceeds the grid" never bites within the documented P/M/n_block envelope). The fit kernels add grid-stride loops on top. Block sizes are uniformly warp-multiples (256/128/64), including the one computed block which rounds up to a 32-multiple — so 12.1 is clean across all 13 units.

2. **All real findings are scale/utilization (MED) or consistency/hygiene (LOW), concentrated in `qpadm_fit_kernels`** (4 of 6 findings, both MEDs) — the S8-batched fit path. Two themes: (a) the device-grid-x ceiling is hardcoded `65535` (the legacy/y-z limit) and never queried against the sm_120 2^31-1 max [12.5 MED] — a throughput cap, not a correctness cap, but it bites exactly the production S8 rotation; (b) the high-register fit kernels use hand-picked block sizes instead of occupancy-derived ones [12.4 MED].

3. **Consistency gap within the fit wrappers**: most model-batched launchers are clamp-and-grid-stride, but two sites diverge — `qpadm_fit_models_kernel` (the heaviest, no stride, no clamp [12.3 LOW]) and `add_fudge_diag` (raw `n_models` grid [12.2 LOW]). Both are correct today; flagged so a future centralized maxGridDimX clamp doesn't leave them behind.

4. **Grid-axis limits as constants are intentional, not flagged as bugs.** decode_af / f2_block units correctly treat kMaxGridX=2^31-1 and kMaxGridY/Z=65535 as CC-invariant architectural constants (documented in launch_config.hpp) needing no `cudaGetDeviceProperties` query — the right call. The qpadm_fit 12.5 finding is distinct: it hardcodes the WRONG value (65535 on the x axis) AND never queries, which is the genuine gap.
