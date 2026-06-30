# 05 — cuda_backend.cu split plan (header extract + per-subsystem TUs)

## 1. Why / Constraint / Gate

**Why.** `src/device/cuda/cuda_backend.cu` is a **5,712-LOC single translation unit** whose
entire body is one inline class `class CudaBackend final : public ComputeBackend` (declared at
`:253`, spanning `:253–5649`). ~55 methods are defined **inline inside the class body** — there
are **0 out-of-line `CudaBackend::` definitions** and **no header declares the class**. A 5.7k-LOC
inline class is a navigability sink (no subsystem boundary, every reader scrolls the whole TU), a
portfolio liability (it reads like one giant blob, not an engineered backend), and a **serial build
bottleneck** — every edit anywhere recompiles all 5.7k LOC of host orchestration glue. Splitting
into a private class header plus per-subsystem `.cu` TUs gives navigable subsystem files, parallel
nvcc compilation, and surgical recompiles.

**Constraint.** Because the class is currently inline-only, you cannot "just move methods" — there
is no declaration to define against. The split is therefore two coupled mechanical transforms:
1. **Extract a private header** `src/device/cuda/cuda_backend.cuh` holding the class *declaration*
   (member signatures + the 3 nested types + the 17 private data members in byte-exact order), and
2. **Out-of-line every method body** to `CudaBackend::method(){...}`, then **distribute** those
   out-of-line definitions across per-subsystem `.cu` TUs that all compile into the **same**
   `steppe_device` target.

**Gate.** This is **pure source reorganization — behavior-neutral**. The `__global__` kernels are
already external in `*_kernel.cu` (this file is host-side launch/orchestration glue), so **no
device-link/RDC edges move**. The acceptance gate is **PARITY**: the real-AADR golden ctest must
stay **bit-identical** at every step. Nothing about codegen, math mode, precision policy, or
teardown order may change. Every step below is **golden-gated** — build Release + run the golden
ctest, diff hashes vs the STEP 0 baseline.

**Verified against source (this branch):** class span `:253–5649`; anon namespace opens `:139`,
closes `:148`; anon helpers `Bucket :154` / `BlockLayout :161` / `compute_block_layout :166` /
`size_buckets :185` / `fill_rankdrop :216`; nested types `ResidentBlocks :303` (public),
`SvdScratchSizes :4225` (public), `AssembleFlags :5386` (private); free functions
`make_cuda_backend :5659` / `visible_device_count :5673` / `device_fault_status :5692`.

---

## 2. Target file layout

### 2.1 `src/device/cuda/cuda_backend.cuh` (NEW — header only, not compiled)

`#pragma once`. PRIVATE-to-`steppe_device` CUDA header. Holds **only**:

- The class declaration `class CudaBackend final : public ComputeBackend` — all ~55
  overrides + private helpers **declared** (bodies removed/out-of-lined), in their current
  access-region order.
- The **3 nested types** kept in the class body (they appear in member signatures, so they cannot
  be demoted to a TU; preserve access region):
  - `struct ResidentBlocks { ... }` — **PUBLIC** (`:303`), return type of `run_f2_blocks_resident`.
  - `struct SvdScratchSizes { ... }` — **PUBLIC** (`:4225`), return of `large_svd_scratch_sizes`,
    param of `large_svd_V`.
  - `struct AssembleFlags { ... }` — **PRIVATE** (`:5386`), param of `assemble_result`, built in `fit_chunk`.
- **Two inline one-liners kept in the header** (hot, called from every TU; out-of-lining them would
  add a cross-TU call on every compute entry):
  - `void guard_device() const { STEPPE_CUDA_CHECK(cudaSetDevice(device_id_)); }` (`:5522`)
  - `[[nodiscard]] static int set_and_return_device(int)` (`:5530`) — ctor member-init hook; must be
    visible where the lifecycle TU's ctor is defined.
- **17 private data members in EXACT current order** (`:5543–5648`, teardown-order load-bearing —
  **DO NOT reorder**): `device_id_`; `Stream stream_`; `CublasHandle blas_`;
  `CusolverDnHandle solver_`; `DeviceBuffer<std::byte> workspace_`; `Precision solve_precision_`;
  `std::size_t batched_dispatch_count_`; `std::vector<double> tot_line_`;
  `PinnedRegistryCache pinned_in_`; `PinnedBuffer<double> stage_f2_`; `PinnedBuffer<double> stage_vpair_`;
  `DeviceBuffer<double> solver_work_`; `svd_s_`; `svd_u_`; `svd_vt_`; `svd_a2_`; `DeviceBuffer<int> svd_info_`.

**Header includes** (Group-4 interface headers naming member/return types + Group-5 RAII-wrapper
`.cuh` that *define* member types): `backend.hpp`, `device_partial.hpp`, `device_f2_blocks.hpp`,
`device_decode_result.hpp`, `stream_f2_blocks.hpp`, `f2_blocks_out.hpp`, `vram_budget.hpp`,
`steppe/config.hpp`, `steppe/fstats.hpp`, `cuda/handles.hpp` (CublasHandle/CusolverDnHandle),
`cuda/device_buffer.cuh` (DeviceBuffer<T>), `cuda/pinned_buffer.cuh` (PinnedRegistryCache/PinnedBuffer<T>),
`cuda/stream.hpp` (Stream), `cuda/check.cuh` (STEPPE_CUDA_CHECK for the inline guard). std:
`<vector>,<span>,<memory>,<optional>,<cstddef>,<cstdint>`.

> `cublas_v2.h` / `cusolverDn.h` / `cuda_runtime.h` arrive **transitively** via the RAII wrappers so
> member decls type-check. **Keep OUT of the header:** every `*_impl.cuh`, every `launch_*` kernel
> `.cuh`, `cub`/`cufft`, `backend_factory.hpp`, `resources.hpp`, and all `core/` host helpers — these
> travel with the per-subsystem TUs.

### 2.2 Shared internal helper header — NOT created

`src/device/cuda/cuda_backend_internal.cuh` is **not created** in this partition. Every file-local
anon-namespace helper has a **single owning TU**:
- `Bucket` / `BlockLayout` / `compute_block_layout` / `size_buckets` → used only by
  `run_f2_blocks_resident` + `stream_f2_blocks_impl` (**both** in the f2-blocks TU) → move into that
  TU's anon namespace.
- `fill_rankdrop` → used only by `rank_sweep` + `assemble_result` (**both** in the qpadm-fit TU) →
  move into that TU's anon namespace.

This header becomes necessary **only** if a later sub-split separates those co-located callers (see
§5 risk 4). Not taken here — the partition is deliberately **zero-shared-helper**.

### 2.3 The 9 TUs (8 new + `cuda_backend.cu` repurposed to lifecycle)

Each TU `#include "cuda_backend.cuh"` + its own kernel/`*_impl`/core headers. All join the **same**
`steppe_device` target.

#### TU-A — `cuda_backend.cu` (REPURPOSED → lifecycle/seam + factory) — ~180 LOC

| Method | Lines |
|---|---|
| `CudaBackend::CudaBackend(int)` ctor | `:277` |
| `capabilities()` | `:2523–2620` (body `:2543`) |
| `set_solve_precision()` | `:2637–2647` |
| `batched_dispatch_count()` | `:2649–2655` |
| `guard_device()` / `set_and_return_device()` | `:5522` / `:5530` (**stay inline in header**) |
| `[free fn] make_cuda_backend()` | `:5659` |
| `[free fn] visible_device_count()` | `:5673` |
| `[free fn] device_fault_status()` | `:5692–5710` |

Includes: `cuda_backend.cuh`; `device/backend_factory.hpp`; `device/resources.hpp` (Status);
`cuda/check.cuh` (CudaError/CublasError/CusolverError + STEPPE_CUDA_CHECK/WARN);
`cuda/f2_block_kernel.cuh` (`emulation_honorable` for `capabilities()`); `steppe/config.hpp`;
std `<optional>,<exception>`. `make_cuda_backend` does `make_unique<CudaBackend>` → needs the
**complete** class → home of the implicit virtual-dtor instantiation.

#### TU-B — `cuda_backend_f2_blocks.cu` (compute_f2 + resident/streamed block tensors) — ~1010 LOC

| Method | Lines |
|---|---|
| `compute_f2()` **[first non-inline virtual → vtable key-function home]** | `:311` |
| `compute_f2_blocks()` | `:445` |
| `compute_f2_blocks_device()` | `:459` |
| `compute_f2_blocks_resident()` | `:487` |
| `compute_f2_blocks_into()` | `:525` |
| `compute_f2_blocks_streamed()` | `:595` |
| `run_f2_blocks_resident()` | `:627` |
| `stream_f2_blocks_impl()` | `:871` |

Owns anon-namespace `Bucket`/`BlockLayout`/`compute_block_layout`/`size_buckets` (move here) +
method-local `struct Ring` (`:1084`). Uses members `stage_f2_`/`stage_vpair_`/`pinned_in_`.
Includes: `cuda_backend.cuh`; `cuda/f2_block_kernel.cuh`; `cuda/f2_batched_kernel.cuh`;
`cuda/block_sink.cuh`; `device/stream_f2_blocks.hpp`; `device/f2_blocks_out.hpp`;
`cuda/device_partial_impl.cuh`; `cuda/device_f2_blocks_impl.cuh`;
`core/domain/block_partition_rule.hpp`; `device/vram_budget.hpp`; `cuda/check.cuh`;
`core/internal/nvtx.hpp`; std `<algorithm>,<vector>,<stdexcept>,<cstring>,<cmath>`.

#### TU-C — `cuda_backend_decode.cu` (decode/transpose format-reader engine) — ~460 LOC

| Method | Lines |
|---|---|
| `decode_af_resident()` (private) | `:1209` |
| `decode_af()` | `:1264` |
| `detect_sample_ploidy_device()` | `:1302` |
| `transpose_to_canonical()` | `:1331` |
| `decode_af_compact_autosome()` | `:1392` |
| `decode_af_compact_filter()` | `:1517` |

Includes: `cuda_backend.cuh`; `cuda/decode_af_kernel.cuh`; `cuda/detect_ploidy_kernel.cuh`;
`cuda/transpose_canonical_kernel.cuh`; `cuda/decode_compact_kernel.cuh`;
`cuda/device_decode_result_impl.cuh`; `cub/device/device_scan.cuh` + `cub/device/device_select.cuh`;
`cuda/check.cuh`; std `<vector>,<cstdint>`.

#### TU-D — `cuda_backend_dstat.cu` (qpDstat block-reduce + shared ratio-block-jackknife) — ~320 LOC

| Method | Lines |
|---|---|
| `dstat_block_reduce_device()` (private core) | `:1677–1725` |
| `dstat_block_reduce(host-ptr)` | `:1727–1745` |
| `dstat_block_reduce(DeviceDecodeResult)` | `:1751–1759` |
| `f4ratio_blocks_jackknife(DeviceF2Blocks)` | `:3289–3405` |
| `dstat_blocks_jackknife(DeviceDecodeResult)` | `:3407–3502` |
| `f4ratio_blocks_jackknife(F2BlockTensor)` throw-twin | `:3504–3514` |
| `dstat_blocks_jackknife(host-ptr)` throw-twin | `:3516–3527` |

Includes: `cuda_backend.cuh`; `cuda/dstat_kernel.cuh`; `cuda/ratio_block_jackknife_kernel.cuh`
(`launch_ratio_block_jackknife` + `struct DRatioJackArray`); `core/domain/block_partition_rule.hpp`;
`cuda/device_decode_result_impl.cuh` + `cuda/device_f2_blocks_impl.cuh`; `cuda/check.cuh`;
`core/internal/nvtx.hpp`; std `<vector>,<stdexcept>`.
**CROSS-TU:** `f4ratio_blocks_jackknife(DeviceF2Blocks)` calls `device_survivor_blocks()` which is
**defined in TU-G** (member fn, header-declared → links).

#### TU-E — `cuda_backend_dates.cu` (DATES cuFFT autocorrelation LD engine) — ~245 LOC

| Method | Lines |
|---|---|
| `dates_curve()` | `:1771–1961` |
| `dates_repack()` | `:1963–1990` |
| `dates_fit()` | `:1992–2030` |

Includes: `cuda_backend.cuh`; `cuda/dates_kernel.cuh`; `<cufft.h>`; `cuda/handles.hpp` (CufftPlan +
CUFFT_CHECK); `cuda/check.cuh`; `core/internal/nvtx.hpp`; std `<vector>,<cstdint>`. Native-FP64
FFT/normal-eq carve-out; no BLAS/cuSOLVER.

#### TU-F — `cuda_backend_qpfstats.cu` (qpfstats smoothing-solve) — ~450 LOC

| Method | Lines |
|---|---|
| `qpfstats_smooth()` | `:2032–2226` |
| `qpfstats_blocks_smooth(host-ptr)` | `:2228–2266` |
| `qpfstats_blocks_smooth(DeviceDecodeResult)` | `:2268–2286` |
| `qpfstats_blocks_smooth_device()` (private core) | `:2288–2521` |

Uses members `blas_`/`solver_`/`solver_work_`/`solve_precision_` (all in header). Includes:
`cuda_backend.cuh`; `cuda/qpfstats_kernel.cuh`; `cuda/qpfstats_jackknife_kernel.cuh`;
`core/internal/qpfstats_jackknife.hpp`; `core/internal/small_linalg.hpp`; `cuda/f2_block_kernel.cuh`
(MathModeScope/engage_f2_precision + CusolverMathModeScope/engage_solver_precision/emulation_honorable);
`cuda/dstat_kernel.cuh`; `core/domain/block_partition_rule.hpp`; `cuda/device_decode_result_impl.cuh`;
`cuda/check.cuh`; `core/internal/nvtx.hpp`; `steppe/config.hpp`; std `<vector>,<cmath>,<stdexcept>`.

#### TU-G — `cuda_backend_fstats_assemble.cu` (f-stat sweep + f4/f3 assemble; `device_survivor_blocks` home; `tot_line_` writer) — ~750 LOC

| Method | Lines |
|---|---|
| `device_survivor_blocks()` (private) **— DEFINED HERE (single definition)** | `:2657–2685` |
| `run_fstat_sweep_device()` (private; k=3/k=4) | `:2687–3062` |
| `assemble_f4(DeviceF2Blocks)` | `:3064–3177` |
| `assemble_f4(F2BlockTensor)` throw-twin | `:3179–3192` |
| `assemble_f4_quartets(DeviceF2Blocks)` | `:3194–3287` |
| `assemble_f4_quartets(F2BlockTensor)` throw-twin | `:3529–3539` |
| `assemble_f3_triples(DeviceF2Blocks)` | `:3541–3635` |
| `assemble_f3_triples(F2BlockTensor)` throw-twin | `:3637–3647` |
| `f4_sweep()` | `:3950–3957` |
| `f3_sweep()` | `:3959–3964` |

**WRITES `tot_line_`** (read by TU-I). Sweep consts `kFstatDefaultSweepTopK`/`kSweepFilterTopK`/
`kSweepFilterMinZ` come from `config.hpp`/`backend.hpp` (header-supplied, not migrated). Includes:
`cuda_backend.cuh`; `cuda/f2_block_kernel.cuh`; `cuda/qpadm_fit_kernels.cuh` (assemble + sweep + f4
loo/xtau/diag_var launchers); `cuda/device_f2_blocks_impl.cuh`; `cub/device/device_select.cuh` +
`cub/device/device_radix_sort.cuh`; `core/domain/block_partition_rule.hpp`; `cuda/check.cuh`;
`core/internal/nvtx.hpp`; `steppe/config.hpp`; std `<cstdlib>,<vector>,<stdexcept>,<cmath>,<algorithm>`.

#### TU-H — `cuda_backend_qpgraph.cu` (qpGraph on-device fleet) — ~300 LOC — FULLY DECOUPLED

| Method | Lines |
|---|---|
| `qpgraph_fit_fleet()` | `:3649–3816` |
| `qpgraph_fit_fleet_batch()` | `:3818–3948` |

**ZERO cross-TU coupling** (uses only `guard_device`/`stream_`/`DeviceBuffer`); native-FP64
carve-out. Helper types `QpGraphDeviceTopo`/`QpGraphDeviceTopoView`/`ScratchLayout`/`make_layout`/
`kMaxThetaDev` are supplied by `cuda/qpgraph_fit_kernels.cuh` (NOT anon-namespace migration).
Includes: `cuda_backend.cuh`; `cuda/qpgraph_fit_kernels.cuh`; `cuda/check.cuh`;
std `<vector>,<stdexcept>,<string>`.

#### TU-I — `cuda_backend_qpadm_fit.cu` (qpAdm fit engine: jackknife/svd-large/rank/gls/se + batched fit) — ~1390 LOC (largest)

| Method | Lines |
|---|---|
| `jackknife_cov()` | `:3966–4116` |
| `jackknife_diag()` | `:4118–4177` |
| `model_fits_small_path()` static | `:4187–4195` |
| `gesvdj_applicable()` static | `:4197–4205` |
| `large_svd_scratch_sizes()` | `:4230–4261` |
| `large_svd_V(scratch-taking)` | `:4263–4349` |
| `large_svd_V(convenience)` | `:4351–4385` |
| `large_dbl_scratch()` static | `:4387–4403` |
| `large_int_scratch()` static | `:4404–4408` |
| `large_loo_dbl_refit()` static | `:4413–4419` |
| `large_fit_one()` | `:4425–4437` |
| `provides_rank_sweep()` | `:4442` |
| `rank_sweep()` | `:4465–4610` |
| `gls_weights()` | `:4616–4691` |
| `gls_weights_loo_batched()` | `:4699–4719` |
| `se_from_wmat()` | `:4733–4761` |
| `populate_loo_wmat_resident()` (private) | `:4771–4884` |
| `provides_batched_fit()` | `:4891` |
| `fit_models_batched()` | `:4916–4982` |
| `fit_one_bucket()` (private) | `:4991–5066` |
| `fit_chunk()` (private) | `:5071–5380` |
| `assemble_result()` (private) | `:5394–5508` |

Owns anon-namespace `fill_rankdrop` (move here) + method-local `struct Key` (`:4936`).
**READS `tot_line_`** (writer in TU-G). **CROSS-TU member calls:** `device_survivor_blocks` (TU-G) +
`capabilities` (TU-A). Uses members `solve_precision_`/`solver_work_`/`svd_s_`/`svd_u_`/`svd_vt_`/
`svd_a2_`/`svd_info_`/`blas_`/`solver_`/`batched_dispatch_count_`. Includes: `cuda_backend.cuh`;
`cuda/qpadm_fit_kernels.cuh`; `cuda/f2_block_kernel.cuh` (precision scopes); `cuda/device_f2_blocks_impl.cuh`;
`core/internal/pchisq.hpp`; `core/qpadm/qpadm_bounds.hpp`; `cuda/check.cuh`; `core/internal/nvtx.hpp`;
`steppe/config.hpp`; std `<vector>,<climits>,<cmath>,<limits>,<algorithm>,<stdexcept>`.

---

## 3. CMake change (`src/device/CMakeLists.txt`)

Inside `add_library(steppe_device STATIC ...)`, replace the single line

```cmake
    cuda/cuda_backend.cu          # implements ComputeBackend on the GPU
```

with the repurposed lifecycle TU + the 8 new per-subsystem TUs:

```cmake
    cuda/cuda_backend.cu                  # lifecycle/seam: ctor + capabilities + set/get-precision + factory/fault fns
    cuda/cuda_backend_f2_blocks.cu        # compute_f2 + resident/streamed f2 block tensors (owns Bucket/BlockLayout/Ring)
    cuda/cuda_backend_decode.cu           # decode/transpose format-reader engine (cub scan/select)
    cuda/cuda_backend_dstat.cu            # qpDstat block-reduce + shared ratio-block-jackknife
    cuda/cuda_backend_dates.cu            # DATES cuFFT host glue
    cuda/cuda_backend_qpfstats.cu         # qpfstats smoothing-solve
    cuda/cuda_backend_fstats_assemble.cu  # f-stat sweep + f4/f3 assemble (device_survivor_blocks home; tot_line_ writer)
    cuda/cuda_backend_qpgraph.cu          # qpGraph on-device fleet (decoupled)
    cuda/cuda_backend_qpadm_fit.cu        # qpAdm fit engine (svd-large + rank/gls/se + batched fit; owns fill_rankdrop/Key)
```

`cuda/cuda_backend.cuh` is **header-only — no `add_library` entry**. All 9 TUs join the **same**
`steppe_device` target → they inherit the existing PRIVATE link deps (`CUDA::cudart/cublas/cusolver/
cufft`), the PRIVATE `target_compile_definitions` (`STEPPE_HAVE_EMU_TUNING`, `STEPPE_NVTX`),
`CUDA_SEPARABLE_COMPILATION ON`, and `POSITION_INDEPENDENT_CODE ON` → codegen/macros/RDC stay
byte-identical. **No other CMake change.**

---

## 4. Safe ordering (header-extract first, then one TU at a time, each golden-gated)

The bit-exact real-AADR 9-pop golden ctest is the parity reference and **must never move**.

- **STEP 0 (baseline).** Build Release (`cmake --preset release`/`ci`) on the remote box, run the
  real-AADR golden ctest on the **current single-TU** file. **Record the golden hashes** — the
  parity reference for every later step.
- **STEP 1 (header extract ONLY — no redistribution).** Create `cuda_backend.cuh` with the class
  declaration (member sigs, the 3 nested types in their access regions, the 17 data members in
  EXACT order, inline `guard_device`/`set_and_return_device`, the Group-4/Group-5 includes). In
  `cuda_backend.cu`, replace the inline class body with `#include "cuda_backend.cuh"` and convert
  **every** method to an out-of-line `CudaBackend::method(){...}` **in place** (still one TU). Keep
  the anon helpers and the 3 free functions where they are. Build + golden. **This isolates "did
  header-extract + out-of-lining break anything" from "did redistribution break anything" — the
  critical de-risking gate.**
- **STEP 2.** Move the most-decoupled subsystem first → `cuda_backend_qpgraph.cu`. CMake. Build + golden.
- **STEP 3.** `cuda_backend_dates.cu`. Build + golden.
- **STEP 4.** `cuda_backend_decode.cu`. Build + golden.
- **STEP 5.** `cuda_backend_dstat.cu`. It calls `device_survivor_blocks`, still defined in the
  `cuda_backend.cu` remainder at this point → links fine. Build + golden.
- **STEP 6.** `cuda_backend_qpfstats.cu`. Build + golden.
- **STEP 7.** `cuda_backend_fstats_assemble.cu` — **move the `device_survivor_blocks` DEFINITION
  here** together with `run_fstat_sweep_device`/`f4_sweep`/`f3_sweep` + the 6 assemble methods + the
  `tot_line_` writers. After this move the only definition of `device_survivor_blocks` lives here;
  the dstat TU already calls it via the header decl. Build + golden.
- **STEP 8.** `cuda_backend_f2_blocks.cu` — move the 8 `compute_f2*` methods + the anon helpers
  `Bucket`/`BlockLayout`/`compute_block_layout`/`size_buckets` into this TU's anon namespace; the
  method-local `Ring` stays inside `stream_f2_blocks_impl`. This TU defines the first virtual
  override (`compute_f2`) → it becomes the **vtable key-function home**. Build + golden.
- **STEP 9.** `cuda_backend_qpadm_fit.cu` — move the 22 fit/svd methods + the anon helper
  `fill_rankdrop` into this TU's anon namespace + method-local `Key`. It reads `tot_line_` (writer
  now in TU-G) and calls `device_survivor_blocks` (TU-G) + `capabilities` (TU-A) cross-TU. Build + golden.
- **STEP 10 (final).** What remains in `cuda_backend.cu` IS the lifecycle/seam TU (ctor +
  capabilities + set_solve_precision + batched_dispatch_count + make_cuda_backend/
  visible_device_count/device_fault_status). Final build + full golden ctest; **diff hashes vs STEP
  0 — must be bit-identical.**

---

## 5. Cross-TU risks

1. **Private member fn shared across TUs — `device_survivor_blocks` (`:2657`).** DEFINED in TU-G,
   CALLED cross-TU by TU-D (`f4ratio_blocks_jackknife`) and TU-I (`fit_models_batched`). Safe
   because it is a **member fn** (external linkage, declared in `cuda_backend.cuh`), **NOT** a
   TU-local `static`. Keep **exactly one** definition. Same pattern: `capabilities()` defined in
   TU-A, called by TU-I (`fit_models_batched`/`fit_one_bucket` free-VRAM probe); `make_cuda_backend`
   needs the **complete** class.
2. **Shared mutable state = class members — MUST stay in `cuda_backend.cuh` (NOT file-statics).**
   - `tot_line_`: WRITER `assemble_f4`/`assemble_f4_quartets`/`assemble_f3_triples` (TU-G);
     READER `jackknife_cov`/`jackknife_diag` (TU-I).
   - `solve_precision_`: WRITER `set_solve_precision` (TU-A); READERS qpfstats (TU-F) + the
     qpadm-fit cuSOLVER solve sites `:4078`/`:5165` (TU-I).
   - `batched_dispatch_count_`: WRITER `fit_chunk`/`fit_models_batched` (TU-I); READER
     `batched_dispatch_count()` (TU-A).
   - Turning **any** of these into a TU-local static would silently break parity.
3. **Nested types in member signatures MUST live in the class decl in the header (preserve access).**
   `ResidentBlocks` PUBLIC (`:303`), `SvdScratchSizes` PUBLIC (`:4225`), `AssembleFlags` PRIVATE
   (`:5386`) — each used by a single subsystem but named in a member signature, so none can be
   demoted to a TU.
4. **Anon-namespace file-local helpers — single owner each → move to the owning TU's anon namespace;
   no shared internal header.** `Bucket`/`BlockLayout`/`compute_block_layout`/`size_buckets` → TU-B;
   `fill_rankdrop` → TU-I. **RISK:** if either TU is later sub-split so the callers separate, the
   helper MUST be promoted to `cuda_backend_internal.cuh` (`inline`/detail-namespace) — do NOT leave
   two anon copies that could drift. NOTE the qpgraph/ratio helper types
   (`QpGraphDeviceTopo`/`ScratchLayout`/`make_layout`/`kMaxThetaDev`/`DRatioJackArray`) and the
   sweep consts (`kFstatDefaultSweepTopK`/`kSweepFilterTopK`/`kSweepFilterMinZ`) are **NOT**
   cuda_backend.cu anon helpers — they come from `qpgraph_fit_kernels.cuh` /
   `ratio_block_jackknife_kernel.cuh` / `config.hpp` / `backend.hpp`, so they travel with the
   include, no migration.
5. **Data-member declaration order is teardown-order load-bearing.** `device_id_` first (its
   `set_and_return_device` initializer makes the device current before `blas_`/`workspace_`
   construct); `stream_` before `blas_`/`solver_` (destroyed AFTER them — their contexts bind to
   it); `workspace_` after `blas_` (frees first; `blas_` holds a non-owning ptr into it); the
   pinned/staging members last (destruct first). The header must reproduce the byte order EXACTLY —
   **reordering = UB at teardown.**
6. **Vtable / key-function emission.** Today all virtuals are inline → vtable is COMDAT in every TU.
   After out-of-lining, the vtable + typeinfo are emitted in the TU that defines the **first**
   non-inline virtual override = `compute_f2` (`:311`) → **TU-B**. Every other override MUST be
   defined in exactly one TU; if any override is accidentally left only-declared, the vtable/typeinfo
   won't be emitted → undefined-reference link error. The implicit virtual dtor needs every member
   type fully visible — guaranteed by the RAII-wrapper includes in the header.
7. **Template members `DeviceBuffer<double>/<int>` + `PinnedBuffer<double>`.** Header-defined
   templates, implicitly instantiated per-TU — no explicit instantiation, no ODR hazard. The
   implicit `CudaBackend` dtor instantiates in TU-A (via `make_unique`) and runs through the vtable;
   keep `device_buffer.cuh`/`pinned_buffer.cuh`/`handles.hpp`/`stream.hpp` fully visible in the
   header so all members are destructible there.
8. **Build invariants preserved by joining the SAME `steppe_device` target.** The PRIVATE
   `STEPPE_HAVE_EMU_TUNING`/`STEPPE_NVTX` compile-definitions and `CUDA_SEPARABLE_COMPILATION` are
   target-wide → every new TU inherits identical codegen. Only HOST orchestration moves (the
   `__global__` kernels are already external in `*_kernel.cu`, reached via `launch_*` host wrappers)
   → NO new device-link/RDC edges, no static-init-order concerns (the only namespace-scope objects
   are the stateless free functions; no global/static `CudaBackend` instance).

---

## 6. Per-task list (one task per step — runs through the fix-kimi machinery)

Each task: **files** + **what-moves** + **verify = Release build + real-AADR golden ctest hashes ==
STEP 0 baseline (bit-identical)**. Tasks are strictly sequential; do not start Tn+1 until Tn is
golden-green.

### T0 — baseline + header extract + in-place out-of-lining (de-risking gate)
- **Files:** create `src/device/cuda/cuda_backend.cuh`; edit `src/device/cuda/cuda_backend.cu`.
- **What moves:** (a) record STEP 0 golden hashes on the unmodified file. (b) Build `cuda_backend.cuh`
  = class declaration only (all ~55 member sigs with bodies removed; the 3 nested types
  `ResidentBlocks :303` / `SvdScratchSizes :4225` / `AssembleFlags :5386` in their access regions;
  the 17 data members `:5543–5648` in EXACT order; inline `guard_device :5522` + `set_and_return_device
  :5530`; Group-4/Group-5 includes per §2.1). (c) In `cuda_backend.cu`, replace the inline class body
  with `#include "cuda_backend.cuh"` and convert EVERY method to out-of-line `CudaBackend::method(){...}`
  **in place** — still one TU. Keep the 5 anon helpers (`:154/:161/:166/:185/:216`) and the 3 free
  functions (`:5659/:5673/:5692`) where they are. **No CMake change yet.**
- **Verify:** golden hashes == T0 baseline.

### T1 — extract `cuda_backend_qpgraph.cu` (most-decoupled first)
- **Files:** new `src/device/cuda/cuda_backend_qpgraph.cu`; edit `cuda_backend.cu`; edit
  `src/device/CMakeLists.txt`.
- **What moves:** `qpgraph_fit_fleet` (`:3649–3816`), `qpgraph_fit_fleet_batch` (`:3818–3948`) →
  new TU with includes per §2.3 TU-H. ZERO cross-TU coupling. Add the TU to `add_library(steppe_device …)`.
- **Verify:** golden == baseline.

### T2 — extract `cuda_backend_dates.cu`
- **Files:** new `cuda_backend_dates.cu`; edit `cuda_backend.cu`; edit CMake.
- **What moves:** `dates_curve` (`:1771`), `dates_repack` (`:1963`), `dates_fit` (`:1992`). Includes per TU-E.
- **Verify:** golden == baseline.

### T3 — extract `cuda_backend_decode.cu`
- **Files:** new `cuda_backend_decode.cu`; edit `cuda_backend.cu`; edit CMake.
- **What moves:** `decode_af_resident` (`:1209`), `decode_af` (`:1264`), `detect_sample_ploidy_device`
  (`:1302`), `transpose_to_canonical` (`:1331`), `decode_af_compact_autosome` (`:1392`),
  `decode_af_compact_filter` (`:1517`). Includes per TU-C (cub scan + select).
- **Verify:** golden == baseline.

### T4 — extract `cuda_backend_dstat.cu`
- **Files:** new `cuda_backend_dstat.cu`; edit `cuda_backend.cu`; edit CMake.
- **What moves:** `dstat_block_reduce_device` (`:1677`), `dstat_block_reduce(host-ptr)` (`:1727`),
  `dstat_block_reduce(DeviceDecodeResult)` (`:1751`), `f4ratio_blocks_jackknife(DeviceF2Blocks)`
  (`:3289`), `dstat_blocks_jackknife(DeviceDecodeResult)` (`:3407`), `f4ratio_blocks_jackknife
  (F2BlockTensor)` throw-twin (`:3504`), `dstat_blocks_jackknife(host-ptr)` throw-twin (`:3516`).
  `device_survivor_blocks` is **still defined in the `cuda_backend.cu` remainder** → links via the
  header decl. Includes per TU-D.
- **Verify:** golden == baseline.

### T5 — extract `cuda_backend_qpfstats.cu`
- **Files:** new `cuda_backend_qpfstats.cu`; edit `cuda_backend.cu`; edit CMake.
- **What moves:** `qpfstats_smooth` (`:2032`), `qpfstats_blocks_smooth(host-ptr)` (`:2228`),
  `qpfstats_blocks_smooth(DeviceDecodeResult)` (`:2268`), `qpfstats_blocks_smooth_device` (`:2288`).
  Includes per TU-F. Uses members `blas_`/`solver_`/`solver_work_`/`solve_precision_` (header).
- **Verify:** golden == baseline.

### T6 — extract `cuda_backend_fstats_assemble.cu` (moves `device_survivor_blocks` definition)
- **Files:** new `cuda_backend_fstats_assemble.cu`; edit `cuda_backend.cu`; edit CMake.
- **What moves:** `device_survivor_blocks` **DEFINITION** (`:2657–2685`) — after this, its **only**
  definition lives here; the T4 dstat TU links via the header decl. Plus `run_fstat_sweep_device`
  (`:2687`), `assemble_f4(DeviceF2Blocks)` (`:3064`), `assemble_f4(F2BlockTensor)` twin (`:3179`),
  `assemble_f4_quartets(DeviceF2Blocks)` (`:3194`), `assemble_f4_quartets(F2BlockTensor)` twin
  (`:3529`), `assemble_f3_triples(DeviceF2Blocks)` (`:3541`), `assemble_f3_triples(F2BlockTensor)`
  twin (`:3637`), `f4_sweep` (`:3950`), `f3_sweep` (`:3959`). This TU **writes `tot_line_`**.
  Includes per TU-G.
- **Verify:** golden == baseline.

### T7 — extract `cuda_backend_f2_blocks.cu` (vtable key-function home)
- **Files:** new `cuda_backend_f2_blocks.cu`; edit `cuda_backend.cu`; edit CMake.
- **What moves:** `compute_f2` (`:311`, first non-inline virtual → emits vtable+typeinfo here),
  `compute_f2_blocks` (`:445`), `compute_f2_blocks_device` (`:459`), `compute_f2_blocks_resident`
  (`:487`), `compute_f2_blocks_into` (`:525`), `compute_f2_blocks_streamed` (`:595`),
  `run_f2_blocks_resident` (`:627`), `stream_f2_blocks_impl` (`:871`). Move anon helpers `Bucket`
  (`:154`), `BlockLayout` (`:161`), `compute_block_layout` (`:166`), `size_buckets` (`:185`) into
  this TU's anon namespace; method-local `Ring` (`:1084`) stays inside `stream_f2_blocks_impl`.
  Includes per TU-B.
- **Verify:** golden == baseline.

### T8 — extract `cuda_backend_qpadm_fit.cu` (largest; moves `fill_rankdrop`)
- **Files:** new `cuda_backend_qpadm_fit.cu`; edit `cuda_backend.cu`; edit CMake.
- **What moves:** the 22 fit/svd methods `jackknife_cov` (`:3966`) → `assemble_result` (`:5394`)
  per the TU-I table. Move anon helper `fill_rankdrop` (`:216`) into this TU's anon namespace;
  method-local `Key` (`:4936`) stays inside `fit_models_batched`. READS `tot_line_` (TU-G writer);
  cross-TU calls `device_survivor_blocks` (TU-G) + `capabilities` (TU-A). Includes per TU-I.
- **Verify:** golden == baseline.

### T9 — finalize lifecycle TU + full diff
- **Files:** edit `cuda_backend.cu` (now the lifecycle/seam TU); edit CMake comment for the line.
- **What remains/moves:** confirm `cuda_backend.cu` holds exactly the ctor (`:277`), `capabilities`
  (`:2523`), `set_solve_precision` (`:2637`), `batched_dispatch_count` (`:2649`), and the 3 free
  functions `make_cuda_backend` (`:5659`) / `visible_device_count` (`:5673`) / `device_fault_status`
  (`:5692`); includes per TU-A. Nothing else should be left.
- **Verify:** full golden ctest; **diff all golden hashes vs T0 baseline — must be bit-identical.**

---

### Decision notes
- **9 TUs** (8 new + `cuda_backend.cu` repurposed to lifecycle), one per natural subsystem,
  **qpadm-fit and f2-blocks kept WHOLE**. This is the lowest-risk behavior-neutral layout: keeping
  qpadm-fit whole keeps `fill_rankdrop` TU-local; keeping f2-blocks whole keeps
  `Bucket`/`BlockLayout`/`compute_block_layout`/`size_buckets` TU-local → **no shared internal helper
  header at all** (cleanest possible). Trade-off: two heavy TUs (qpadm-fit ~1390 LOC, f2-blocks
  ~1010 LOC).
- The **one** worthwhile optional sub-split (if tighter LOC balance is preferred over the TU count):
  qpadm-fit → (A) jackknife + svd-large + rank/gls/se (~810 LOC) and (B) batched
  `fit_models_batched`/`fit_one_bucket`/`fit_chunk`/`assemble_result` (~570 LOC); its **only** cost is
  promoting `fill_rankdrop` to `cuda_backend_internal.cuh` (`inline`, pure parity-neutral host index
  math). Not taken here, to keep the partition zero-shared-helper.
- **Do NOT** sub-split f2-blocks (resident vs streamed) — that forces
  `Bucket`/`BlockLayout`/`compute_block_layout`/`size_buckets` into a shared header for no LOC win.
