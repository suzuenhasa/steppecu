# Review findings — src__device__cuda__device_f2_blocks

Files: src/device/cuda/device_f2_blocks.cu, src/device/cuda/device_f2_blocks_impl.cuh, src/device/device_f2_blocks.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (why clean), so the result is auditable:
- 4.2/4.6 index width (the real risk at scale): size() in device_f2_blocks.hpp:48-51
  widens EACH int operand to std::size_t BEFORE multiplying
  (static_cast<std::size_t>(P) * static_cast<std::size_t>(P) * static_cast<std::size_t>(n_block)).
  P*P*n_block reaches ~10^10 (> 2^31), but the product is formed in size_t, so no
  int overflow. Every consumer of the flat count (.cu:44 `total = size()`, .cu:74
  `total = out.size()`) takes that size_t value. No int global index is computed.
- 4.3 allocation sizing: bytes computed as `total * sizeof(double)` (.cu:54, .cu:85)
  with total already size_t — byte count, not element count, passed to cudaMemcpy.
  DeviceBuffer<double>(total) (.cu:83-84) takes an element count by design and the
  owner itself guards the n*sizeof(T) byte product (device_buffer.cuh ctor). No
  missing `* sizeof` and no element/byte confusion.
- 4.4/4.5 loops: none in any of the three files — no unsigned countdown, no
  signed/unsigned loop-bound compare. N/A.
- 4.1 float/double: all buffers/scalars are double (FP64 by design, §12 parity).
  No narrowing temp. N/A.
- 4.7 host/device pointer typing: D2H/H2D directions are explicit and correct
  (to_host = DeviceToHost; upload = HostToDevice); device pointers come from
  impl->f2.data() and host from std::vector::data(). The raw `double*` accessors
  lack a compile-time space tag, but that is the project-wide DeviceBuffer::data()
  convention (not introduced here) and not a numeric defect — not flagged.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (why clean), so the result is auditable:
- 2.1 Dropped archs: these three files are TU/header source, not CMake. No sm_/
  compute_ arch flags, no __CUDA_ARCH__ guards, no compute-capability literals
  anywhere in the unit. Arch lists live in CMake, not here. N/A.
- 2.2 Texture/surface REFERENCES: no `texture<...>`, no `surface<...>`, no
  cudaBindTexture*/cudaBindSurface*/cudaCreateTextureObject in any file. The unit
  touches no textures/surfaces at all. N/A.
- 2.3 Non-_sync warp intrinsics: there are NO kernels in this unit (no __global__,
  no __device__ code). No __shfl*/__ballot/__any/__all/__syncwarp usage — these are
  host-side TUs (special members, to_host D2H, upload H2D) plus the Impl struct. N/A.
- 2.4 cudaThreadSynchronize: absent. The only CUDA runtime calls are cudaGetDevice,
  cudaSetDevice, cudaMemcpy (with cudaMemcpyDeviceToHost/HostToDevice), and cudaFree
  (via DeviceBuffer dtor) — all current in CUDA 13, none deprecated/removed.
  device_f2_blocks.cu:50,52,57,59,78,80,86,88. No deprecated sync API.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (why clean), so the result is auditable:
- 3.1 Commented-out blocks: none. All comments in the unit are documentation —
  file-header banners (.cu:1-5, _impl.cuh:1-5, .hpp:1-6), Doxygen doc comments,
  the multi-line TODO(multigpu-host-bounce) design note (.hpp:86-91), and member
  banners. No statement/expression is commented out "just in case". The existing
  <!-- --> blocks in this findings file are reviewer notes, not source.
- 3.2 Unreachable code: no `#if 0`/`#if 0`-style guards anywhere. The two early
  returns are GUARDED and reachable — to_host() `if (total == 0 || !impl) return out;`
  (.cu:47) and upload_..._to_device() `if (total == 0) return out;` (.cu:75). No code
  follows an unconditional return/break/continue. N/A.
- 3.3 Unused symbols: every include is consumed — .cu: <cuda_runtime.h> (cudaMemcpy/
  cudaGetDevice/cudaSetDevice), <cstddef> (std::size_t), <memory> (std::make_unique
  .cu:82), check.cuh (STEPPE_CUDA_CHECK), device_buffer.cuh (DeviceBuffer),
  pinned_buffer.cuh (RegisteredHostRegion .cu:55-56), fstats.hpp (F2BlockTensor).
  _impl.cuh: device_f2_blocks.hpp + device_buffer.cuh both used in the Impl struct.
  .hpp: <cstddef>/<memory>/<vector>/fstats.hpp all used. No unused params (the two
  RAII guard structs read `prev`/`device_id`); no unused locals or static helpers.
- 3.4 Computed but unread: none. RegisteredHostRegion pin_f2/pin_vp (.cu:55-56) and
  the `G restore{prev}` guards (.cu:51, .cu:79) are RAII objects held for their dtor
  side effects (un-pin / restore device), which is a use — not dead stores. Every
  assigned scalar (`total`, `bytes`, `prev`, out.* shape fields) is subsequently read.
-->

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

<!--
Notes (why clean), so the result is auditable:
- 5.1 Unnamed literals: the only literals in the unit are structural, not tunable
  magic numbers. `0` is a default/identity (P=0, n_block=0 .hpp:40-41; prev=0 init
  .cu:49,77; the `total == 0` empty guard .cu:47,75; the `n_block < 0 ? 0` clamp).
  `-1` is the documented "no device" sentinel for device_id (.hpp:42). `sizeof(double)`
  (.cu:54,85) is the correct FP64 byte size, NOT a magic `8` — and double is parity-
  load-bearing (§12). None of these are knobs that want naming.
- 5.2 Hardcoded sizes/bounds: none. There are no fixed array sizes, no launch dims,
  no buffer capacities. Every size is DERIVED: total = size() = P*P*n_block
  (.cu:44,74 via .hpp:48-51), bytes = total*sizeof(double) (.cu:54,85). P / n_block /
  device_id are members or params, not baked-in bounds. Nothing here should be a
  param that is currently a literal.
- 5.3 Duplicated constants: no drift-prone duplication. The `n_block < 0 ? 0 : n_block`
  clamp recurs at .hpp:50, .cu:42, .cu:71, but it is a defensive guard idiom (clamp a
  possibly-negative shape to 0), not a shared dimension/size constant that could
  silently diverge — there is no launch-dim-vs-shared-mem-array pairing in this unit
  (no kernels), so the 5.3 correctness hazard (block dim duplicated in launch AND as
  an array size) does not arise. Could be factored to a helper for hygiene, but not a
  correctness risk; LOW-not-worth-flagging.
- 5.4 Hardcoded paths/IDs/device ids: none. No file paths, no string IDs. No device
  ordinal is hardcoded — device_id is a member/param (.hpp:42, .cu:68 param), the
  current device is read at runtime via cudaGetDevice(&prev) (.cu:50,78) and restored,
  never assumed to be 0. N/A.
- 5.5 Ambiguous 32: there is NO `32` (or any warp-size literal) anywhere in the unit;
  it contains no kernels, no launch configs, no warp-level intrinsics. N/A.
-->

## Group 6 — Naming

- [6.1][LOW] src/device/cuda/device_f2_blocks.cu:51,79 — the inline scope-restore RAII guard uses a single-letter struct name `G` with a single-letter member `d` (`struct G { int d; ~G() { (void)cudaSetDevice(d); } } restore{prev};`), opaque outside the two-line idiom. The instance `restore` and the captured `prev` are clear; only `G`/`d` are cryptic. Mirrors the same idiom in p2p_combine.cu, so it is consistent project-wide and a throwaway scope. Suggested: optional hygiene rename to e.g. `DeviceGuard`/`dev` (or share a single guard helper with p2p_combine.cu) — not load-bearing.

<!--
Notes (why otherwise clean), so the result is auditable:
- 6.1 Cryptic names: aside from the `G`/`d` guard above, every identifier is
  descriptive — `total`, `bytes`, `prev`, `pin_f2`, `pin_vp`, `out`, `host`. No
  `tmp`/`data2`/`arr`/`flag`. The loop-counter exemption is moot: there are no loops.
- 6.2 Misleading names: none. `size()`/`total`/`bytes` are honest (element count vs
  byte count are correctly distinguished — `total` is elements, `bytes = total *
  sizeof(double)`, .cu:54,85). `n_block` is a count and is used as a count; `block_sizes`
  is a per-block SNP-count vector and is documented as such (.hpp:44-45). `device_id`
  is an ordinal, named so. `f2_device()`/`vpair_device()` return device pointers as the
  suffix advertises. No count-that-is-an-index, no list-that-is-a-map.
- 6.3 Inconsistent conventions in one file: consistent. snake_case for locals/members
  (`n_block`, `block_sizes`, `device_id`, `pin_f2`, `pin_vp`), lowerCamel-free; the
  capital `P` is the project-wide population-count symbol (matches the public API and
  F2BlockTensor), not a stray convention. No `nBlock` vs `num_block` vs `n` mix; the
  Vpair tensor is uniformly `vpair`/`vp` (member `vpair`, accessor `vpair_device`, local
  `pin_vp`) with no `vPair`/`v_pair`/`variance_pair` drift in the same file.
- 6.4 Nonstandard abbreviations: the abbreviations used are standard/established —
  `prev` (previous device, idiomatic), `vp` (vpair, paired with `pin_f2`/`pin_vp`),
  `Impl` (pimpl, project-wide), `f2` (the f2-statistic tensor, domain term). None are
  invented or ambiguous in context.
-->

## Group 7 — Duplication

- [7.4][LOW] src/device/cuda/device_f2_blocks.cu:49-52,77-80 — the device-select-with-restore prologue is duplicated verbatim across `to_host()` and `upload_f2_blocks_to_device()`: `int prev = 0; STEPPE_CUDA_CHECK(cudaGetDevice(&prev)); struct G { int d; ~G() { (void)cudaSetDevice(d); } } restore{prev}; STEPPE_CUDA_CHECK(cudaSetDevice(device_id));`. The doc comments (.cu:36-38, 65-67, .hpp:67) confirm it also reproduces the same idiom in p2p_combine.cu:79-85, so the RAII guard `struct G` is copy-pasted at three+ sites. Suggested: fold into a single shared `ScopedDeviceSwitch`/`DeviceGuard` helper (one header) and replace the inline `struct G { int d; ~G(){...} } restore{prev};` at each site — collapsible boilerplate, parity-neutral.
- [7.1][LOW] src/device/cuda/device_f2_blocks.cu:39-62,68-91 — `to_host()` and `upload_f2_blocks_to_device()` are near-mirror copies differing only by transfer direction and the alloc side: both clamp the shape (`n_block < 0 ? 0`), copy `block_sizes`, compute `total = size()`, take the device guard, compute `bytes = total * sizeof(double)`, then issue two paired `cudaMemcpy` of `f2` and `vpair`. The f2/vpair pair-copy in particular is two structurally identical statements per function (.cu:57-60 D2H, .cu:86-89 H2D). Suggested: optional small helper for the paired f2+vpair memcpy (direction + dst/src as params) to remove the doubled `cudaMemcpy(... bytes, dir)` lines; the alloc/clamp scaffolding is light enough that the guard-helper (7.4) is the higher-value extraction.
- [7.2][LOW] src/device/device_f2_blocks.hpp:50, src/device/cuda/device_f2_blocks.cu:42,71 — the negative-`n_block` clamp `(n_block < 0 ? 0 : n_block)` is repeated at three sites (in `size()` it is `n_block < 0 ? 0 : n_block` inside the cast; in `to_host` and `upload` it sets `out.n_block`). Already noted under 5.3 as a defensive idiom, not a drift-prone shared constant. Suggested: a tiny `clamped_n_block()` (or normalize once at construction) would fold it; low value, no correctness risk.

Notes (why otherwise minimal), so the result is auditable:
- 7.3 Repeated sizeof/casts: `total * sizeof(double)` appears at .cu:54 and .cu:85, but each is the local byte count inside its own function and `sizeof(double)` is the parity-load-bearing FP64 size (§12), not a hoistable constant; the triple `static_cast<std::size_t>` in size() (.hpp:49-51) is a single widening expression, not duplicated. The byte-product would be subsumed by the 7.1 paired-memcpy helper. No standalone 7.3 fix.
- All Group 7 items are LOW: the unit has no kernels and no hot-path loops; the duplication is in cold host-side transport scaffolding (the one D2H site + its H2D inverse), so folding is hygiene, not a correctness or scale fix. The duplicated guard/clamp cannot diverge in a result-changing way (raw byte copies, §12 bit-faithful).

## Group 8 — Comments

No Group 8 issues found.

<!--
Notes (why clean), so the result is auditable:
- 8.1 Restating code: NONE. Every comment is a file-header banner (.cu:1-5,
  _impl.cuh:1-5, .hpp:1-6), a Doxygen doc comment, a member banner, or the design
  TODO. The inline comments at .cu:32-38, .cu:64-67, .hpp:61-71 explain WHY
  (parity-neutrality of pinned-vs-pageable, the device re-select/restore rationale,
  bit-faithfulness), not the trivial WHAT. No `x = y; // assigns y to x`-style
  restatement anywhere.
- 8.2 Stale comments: NONE. The cross-references are LIVE and EXACT, verified by
  reading the targets: ".cu:38 / .hpp:67 mirroring p2p_combine.cu:79-85" matches the
  DeviceGuard prologue at p2p_combine.cu:79-85, and ".cu:36 / .hpp:65 EXACTLY like
  p2p_combine.cu:185-186" matches the RegisteredHostRegion pinned-D2H at
  p2p_combine.cu:185-186. The TODO's forward pointer "see the full TODO on
  replicate_f2 in src/core/qpadm/model_search.cpp" (.hpp:91) resolves —
  replicate_f2 is at model_search.cpp:169 and the matching TODO(multigpu-host-bounce)
  is at :139/:235. The "THE ONLY D2H" claim is correctly SCOPED: .cu:32 says "in the
  device-resident pipeline" (true; the p2p_combine and no-peer-bounce D2H/H2D are the
  cross-card ASSEMBLY transport, an explicitly documented exception at .hpp:78-91, not
  the resident-output path) and .hpp:61 immediately qualifies with "(opt-in
  materialization)" plus the exact caller list at .hpp:67. Not contradictory, not
  stale.
- 8.3 Missing rationale: NONE material. The non-obvious choices ARE justified inline:
  why pin the host buffers (parity-neutral graceful degrade, .cu:34-36), why re-select
  and restore device (cudaMemcpy/cudaMalloc target the current device, .cu:36-38,
  .cu:64-67), why a raw byte copy (bit-faithful, preserves −0.0, .cu:65-66 / .hpp:84),
  why double (§12 parity, cited). The `n_block < 0 ? 0` clamp is a self-evident
  defensive guard needing no note. No bare magic constant lacks context (sizeof(double)
  is the parity FP64 size, contextualized).
- 8.4 Orphan TODO/FIXME/HACK: NONE. The only marker is TODO(multigpu-host-bounce)
  (.hpp:86-91) and it is the OPPOSITE of orphan — owner-tagged, carries MEASURED data
  (~8.72 GB / ~3.8 s cold on real AADR P=600, caps multi-GPU at ~1.21x), states the
  root cause (no P2P on consumer 5090s), the status (DEFERRED), the fix (per-device
  precompute), and a cross-pointer to the canonical TODO on replicate_f2 in
  model_search.cpp (verified present). No FIXME/HACK/XXX anywhere in the unit.
-->

## Group 9 — Constants & configuration

No Group 9 issues found.

<!--
Notes (why clean), so the result is auditable:
- 9.1 Should-be-const/constexpr left mutable: the local computed values are already
  const where they can be — `const std::size_t total` (.cu:44, .cu:74) and
  `const std::size_t bytes` (.cu:54, .cu:85). `int prev` (.cu:49, .cu:77) is
  INTENTIONALLY mutable: it is passed by address to cudaGetDevice(&prev) as an
  out-param, so it cannot be const. The RAII guard members (`restore{prev}`) capture
  by value. The public shape fields P/n_block/device_id/block_sizes (.hpp:40-46) are
  plain mutable data members BY DESIGN — the producers assign them after default
  construction (upload sets out.P/out.n_block/out.device_id/out.block_sizes at
  .cu:69-73; to_host fills out.P/out.n_block/out.block_sizes at .cu:40-42 on a fresh
  F2BlockTensor), so they must stay mutable. No constexpr candidates: every value is
  runtime-derived from shape (no compile-time constant is left as a runtime const).
  size()/empty() (.hpp:48-53) are already const-qualified member fns. N/A.
- 9.2 Tangled config: there are NO tunable knobs anywhere in the unit — no launch
  dimensions, no chunk/tile sizes, no thresholds, no precision-policy flags, no
  buffer capacities. The transport is straight paired cudaMemcpy (D2H in to_host,
  H2D in upload) with byte counts derived from shape. Nothing is buried in logic
  that wants surfacing to a file-top constant or config struct. N/A.
- 9.3 Positional booleans: NO boolean parameters in any signature in the unit.
  DeviceF2Blocks special members take no args; f2_device()/vpair_device()/to_host()
  take none; upload_f2_blocks_to_device(const F2BlockTensor& host, int device_id)
  (.hpp:92, .cu:68) takes a tensor ref + an int ordinal — no bools. No foo(true,false)
  call sites; the cudaMemcpy kind args are named enum constants
  (cudaMemcpyDeviceToHost / cudaMemcpyHostToDevice, .cu:58,60,87,89), not positional
  bools. N/A.
-->

## Group 10 — Initialization

No Group 10 issues found.

<!--
Notes (why clean), so the result is auditable:
- 10.1 Late/distant or uninitialized-then-assigned: every declaration sits at its
  first use and is initialized at the point of declaration. In to_host()
  (device_f2_blocks.cu): `F2BlockTensor out;` (.cu:40) is filled on the next 3 lines;
  `const std::size_t total = size();` (.cu:44); `int prev = 0;` (.cu:49) is
  value-initialized BEFORE the cudaGetDevice(&prev) out-param call (.cu:50) — defensive,
  correct; `struct G {...} restore{prev};` (.cu:51); `const std::size_t bytes` (.cu:54);
  `RegisteredHostRegion pin_f2/pin_vp` at first use (.cu:55-56). upload_..._to_device()
  mirrors this exactly: `DeviceF2Blocks out;` (.cu:69) filled immediately (.cu:70-73),
  `total` (.cu:74), `prev = 0` (.cu:77), guard (.cu:79), `bytes` (.cu:85). No scalar is
  declared uninitialized and assigned later; no declaration is hoisted away from use. N/A.
- 10.2 Zero-init assumptions that do not hold: NONE. The two default-constructed
  aggregates rely on EXPLICIT in-class member initializers, not implicit/relied-upon
  zero-init. `F2BlockTensor out;` (.cu:40) gets P=0/n_block=0 from NSDMIs
  (fstats.hpp:68,71) and empty vectors from std::vector's default ctor — and to_host
  then explicitly overwrites out.P/out.n_block/out.block_sizes (.cu:40-42) anyway.
  `DeviceF2Blocks out;` (.cu:69) gets P=0/n_block=0/device_id=-1 from NSDMIs
  (device_f2_blocks.hpp:40-42), block_sizes empty, impl null — and upload explicitly
  assigns P/n_block/device_id/block_sizes (.cu:70-73) and impl (.cu:82). The Impl's
  DeviceBuffer<double> f2/vpair are default-constructed (null ptr / size 0 via
  DeviceBuffer()=default) by make_unique<Impl>() (.cu:82) then explicitly assigned
  DeviceBuffer<double>(total) (.cu:83-84) — no read before that assignment. The RAII
  guard member `d` is aggregate-initialized from `prev` (restore{prev}), not left
  uninitialized. No padding/POD is memset-and-assumed; no allocation is assumed
  zero-filled (the resident buffers are fully overwritten by the H2D memcpy at
  .cu:86-89; in the degenerate total==0 path the buffers are never allocated and the
  early return at .cu:75 / .cu:47 yields a tensor whose vectors were size-0-resized).
  N/A.
-->

## Group 11 — Qualifiers & const-correctness

No Group 11 issues found.

<!--
Notes (why clean), so the result is auditable:
- 11.1 const __restrict__ on read-only kernel pointers: N/A — this unit contains NO
  __global__/__device__ kernels and no kernel pointer parameters at all. It is a
  host-side TU (out-of-line special members, the device-pointer accessors, the single
  to_host() D2H, the upload H2D inverse) + the Impl struct + the CUDA-free handle decl.
  The cudaMemcpy calls (.cu:57-60, .cu:86-89) are host-side runtime API, not kernels;
  there are no __restrict__-eligible read-only device pointer params to annotate.
- 11.2 Inconsistent __host__/__device__/__global__: NONE. No function in the unit
  carries an execution-space qualifier, and that is correct and consistent: the seam
  is CUDA-FREE BY DESIGN (.hpp:1, :20-24 name no CUDA type). The accessors f2_device()/
  vpair_device() (.cu:25-30), to_host() (.cu:39), upload_f2_blocks_to_device() (.cu:68),
  and the inline size()/empty() (.hpp:48-53) are all host-only; the borrowed double*
  they return are consumed by kernels in OTHER TUs. No same-symbol qualifier drift, no
  __global__ declared one place / called as host another. N/A.
- 11.3 Host/device helper duplicated instead of __host__ __device__: NONE. The only
  repeated micro-logic is the `n_block < 0 ? 0 : n_block` clamp (.hpp:50, .cu:42, :71)
  and `total * sizeof(double)` (.cu:54, :85) — both run HOST-side only; there is no
  device-side duplicate of either that a shared __host__ __device__ helper would
  unify. (Their host-side duplication is already captured as LOW hygiene under
  Group 7.2 / 7.3.) No host/device twin pair exists here.
- 11.4 Large by-value structs as kernel params (param-space limit / __grid_constant__):
  N/A — no kernels, so no kernel params. The one non-trivial parameter in the unit,
  upload_f2_blocks_to_device(const F2BlockTensor& host, int device_id) (.hpp:92, .cu:68),
  already passes the (potentially large) F2BlockTensor BY CONST REFERENCE, not by value;
  device_id is a scalar int. No by-value struct is handed to any launch. N/A.
- const-correctness (general): already correct. Every accessor is const-qualified
  (f2_device/vpair_device .cu:25,28 `const noexcept`; size/empty .hpp:48,53 `const
  noexcept`; to_host .cu:39 `const`). Return types are const double* (borrowed, read-
  only). The mutable public shape fields P/n_block/device_id/block_sizes (.hpp:40-46)
  are intentionally non-const so producers can assign post-default-construction
  (.cu:40-42, :69-73) — already analyzed and accepted under Group 9.1. No missing const
  on a read-only param/method, no const-cast. The `int prev` (.cu:49,:77) is correctly
  NON-const (taken by address as a cudaGetDevice out-param). Nothing to flag.
-->

## Group 12 — Launch config & indexing

No Group 12 issues found.

<!--
Notes (why clean), so the result is auditable:
- This unit contains NO kernels and NO kernel launches: there is no __global__, no
  __device__ code, and no `<<<grid, block>>>` launch syntax in any of the three files.
  It is host-side transport scaffolding (out-of-line special members + the device-
  pointer accessors + the single to_host() D2H + the upload_f2_blocks_to_device() H2D)
  plus the Impl struct (device_f2_blocks_impl.cuh) and the CUDA-free handle decl
  (device_f2_blocks.hpp). The only CUDA runtime calls are cudaGetDevice / cudaSetDevice
  / cudaMemcpy (device_f2_blocks.cu:50,52,57,59,78,80,86,88) plus cudaMalloc/cudaFree
  via DeviceBuffer. None of the five Group 12 hazards can arise:
- 12.1 Block dim not a multiple of 32: N/A — no block dim, no launch config anywhere.
- 12.2 Grid dim hardcoded vs (n+block-1)/block: N/A — no grid dim, no launch. Sizes
  here are byte/element counts for cudaMemcpy (bytes = total * sizeof(double),
  .cu:54,85), DERIVED from shape via size() (.hpp:48-51), not grid dims.
- 12.3 Missing grid-stride loop: N/A — there is no one-elem-per-thread kernel (no
  kernel at all). The full [P×P×n_block] tensor (which can exceed 2^31 elements at
  scale) is moved in ONE cudaMemcpy of `bytes` (.cu:57-60, :86-89); cudaMemcpy takes a
  size_t byte count and does not iterate a thread grid, so the "input exceeds the grid"
  failure mode does not apply. (The size_t-widened element count is already covered
  under Group 4.)
- 12.4 Baked-in launch config vs cudaOccupancyMaxPotentialBlockSize: N/A — no launch
  to size. No occupancy decision is made or needed in this TU.
- 12.5 Compute-cap / device-property assumptions hardcoded vs queried: NONE. No
  compute-capability literal, no cudaDeviceProp field, no sm_/arch assumption, no warp-
  size literal. The device is not assumed: the current device is QUERIED at runtime via
  cudaGetDevice(&prev) (.cu:50,78) and the target is the runtime `device_id` member/
  param (.hpp:42, .cu:68 param), selected via cudaSetDevice (.cu:52,80) and restored by
  the RAII guard. No device ordinal is hardcoded (never assumes device 0). N/A.
-->

## Group 13 — Error handling

No Group 13 issues found.

<!--
Notes (why clean), so the result is auditable:
- 13.1 Unchecked cuda* API return: NONE on the fault path. Every CUDA runtime call
  that can fail recoverably-into-a-fault routes through the throwing STEPPE_CUDA_CHECK
  (which throws a typed CudaError carrying file:line:function via source_location —
  check.cuh:151-155,214-215): cudaGetDevice (.cu:50, :78), cudaSetDevice (.cu:52, :80),
  the two D2H cudaMemcpy (.cu:57-60), the two H2D cudaMemcpy (.cu:86-89). The
  cudaMalloc behind DeviceBuffer<double>(total) (.cu:83-84) is checked inside the
  allowlisted owner (device_buffer.cuh:74) and its overflow-precheck throws a typed
  CudaError (device_buffer.cuh:66-73); cudaHostRegister behind RegisteredHostRegion
  (.cu:55-56) is intentionally NON-throwing graceful-degrade via STEPPE_CUDA_WARN
  (pinned_buffer.cuh:163-174) — a §11.4/§12 parity-neutral perf lever, correctly NOT a
  fault. The ONLY deliberately-discarded return is `(void)cudaSetDevice(d)` in the RAII
  restore-guard dtor (.cu:51, :79): this is the CORRECT idiom — destructors must be
  noexcept (architecture.md §7), so the device-restore status is swallowed exactly as
  DeviceBuffer/PinnedBuffer swallow teardown cudaFree/cudaFreeHost. Not a finding.
- 13.2 Unchecked launches (need cudaGetLastError + later sync): N/A — this unit has NO
  kernel launches (no `<<<...>>>`, no __global__) in any of the three files. It is
  host-side transport (special members + accessors + to_host D2H + upload H2D) plus the
  Impl struct + the CUDA-free handle. The transfers use SYNCHRONOUS cudaMemcpy (not
  cudaMemcpyAsync), whose status is returned by the call itself and is checked inline
  (.cu:57-60, :86-89), so no separate post-launch cudaGetLastError()/sync pairing is
  owed. STEPPE_CUDA_CHECK_KERNEL (check.cuh:251-256) is the launch-check home but is
  not needed here (nothing to launch).
- 13.3 Inconsistent checking: CONSISTENT. Both transport functions to_host() (.cu:39)
  and upload_f2_blocks_to_device() (.cu:68) use the IDENTICAL guarded prologue
  (cudaGetDevice + cudaSetDevice both STEPPE_CUDA_CHECK'd) and the IDENTICAL paired
  STEPPE_CUDA_CHECK'd cudaMemcpy, with the same `(void)cudaSetDevice(d)` discard in the
  guard dtor at both sites (.cu:51, :79). No call is guarded in one function and bare in
  the mirror. No mix of STEPPE_CUDA_CHECK vs raw status-ignore on the fault path.
- 13.4 Error-swallowing homegrown macro: NONE. STEPPE_CUDA_CHECK does the OPPOSITE of
  swallow — it throws a typed CudaError with the call site and the runtime's error
  name+string (check.cuh:51-67,151-155), never exit() and never silently continues.
  STEPPE_CUDA_WARN (the sibling) logs one line and YIELDS the status for the caller to
  branch on (check.cuh:174-189) — it does not hide the status; and it is not used in
  this unit's .cu anyway (it is reached only transitively via RegisteredHostRegion's
  intentional graceful degrade). No bare `if(err){}`-style suppressor, no macro that
  returns void on a discarded status, no homegrown CUDA_CHECK that masks failures. The
  unit imports the ONE canonical check home (check.cuh) and uses only it.
-->

## Group 14 — Memory: allocation & lifetime

No Group 14 issues found.

<!--
Notes (why clean), so the result is auditable:
- 14.1 Alloc/free mismatch: NONE. There is NO raw cudaMalloc/cudaFree/malloc/free/
  new[]/delete[] in any of the three files. Every device allocation goes through the
  allowlisted RAII owner DeviceBuffer<double> (the Impl's f2/vpair, _impl.cuh:15-16;
  constructed via DeviceBuffer<double>(total) at .cu:83-84), whose cudaMalloc
  (device_buffer.cuh:74) is paired with cudaFree in its own dtor (device_buffer.cuh:117)
  — same family, no cross-allocator free. Host destinations are std::vector<double>
  (out.f2/out.vpair, .cu:45-46) — standard-allocator new[]/delete[], never cudaFree'd.
  The in-place host pin uses RegisteredHostRegion (cudaHostRegister .cu:55-56) paired
  with cudaHostUnregister in ITS dtor (pinned_buffer.cuh:201) — again same family. No
  cudaMalloc-freed-with-free, no new[]/malloc-freed-with-cudaFree, no
  cudaHostAlloc-vs-free confusion. N/A.
- 14.2 Stream-ordered alloc on a hot path: N/A. The only allocations in the unit are
  the two DeviceBuffer<double>(total) in upload_f2_blocks_to_device (.cu:83-84), which
  use plain (device-synchronizing) cudaMalloc. But this is NOT a hot path: upload is the
  no-peer G>=2 cross-card ASSEMBLY transport, called once per device per precompute
  (model_search.cpp:190), explicitly DEFERRED/documented as the multi-GPU host-bounce
  cost (.hpp:86-91). to_host is likewise the opt-in, cold, "ONLY D2H" materialization
  (.cu:32, .hpp:61) used by the parity test / M7 disk cache / explicit CLI caller, NOT
  the fit handoff (cuda_backend.cu:277 "The hot path ... does NOT call this"). The S8
  millions-of-small-solves hot path reads f2_device()/vpair_device() in VRAM (no alloc).
  Promoting these cold transport allocs to cudaMallocAsync/pools would add a streaming
  dependency to a once-per-run path for no measurable win — not flagged.
- 14.3 Async/sync free pairing: N/A. There is NO cudaMallocAsync / cudaFreeAsync /
  cudaMemPool* anywhere in the unit (or in its DeviceBuffer owner). All allocation is
  synchronous cudaMalloc paired with synchronous cudaFree (device_buffer.cuh:74,117).
  No async-allocated pointer is freed with a plain cudaFree and vice versa. N/A.
- 14.4 Free before async work completes (use-after-free across streams): NONE.
  (a) The host PIN windows in to_host: RegisteredHostRegion pin_f2/pin_vp (.cu:55-56)
  are RAII guards whose cudaHostUnregister runs at function-scope exit (.cu:61). The
  two transfers they cover are SYNCHRONOUS cudaMemcpy (.cu:57-60), NOT cudaMemcpyAsync —
  each copy fully completes (host-blocking) before control returns, so the unregister
  never races a live DMA. (Contrast cuda_backend.cu:375-379, which uses cudaMemcpyAsync
  + an explicit trailing cudaStreamSynchronize precisely to keep its staging alive; the
  sync API here needs no such guard.) Verified there is no cudaMemcpyAsync in the unit
  (grep: only synchronous cudaMemcpy at .cu:57,59,86,88).
  (b) The escaping device buffers: the producer compute_f2_blocks_device
  (cuda_backend.cu:295-306) calls run_f2_blocks_resident, which DRAINS the producing
  non-blocking stream_ with cudaStreamSynchronize (cuda_backend.cu:679) BEFORE the
  f2/vpair DeviceBuffers move into the handle (cuda_backend.cu:303-304). So by the time
  any DeviceF2Blocks exists, all producer-stream writes are settled; to_host's
  default-stream cudaMemcpy (.cu:57-60) reads fully-written buffers — no read-before-
  write across streams. The buffers free only in ~Impl (when the moved-out handle dies),
  strictly AFTER every consumer (fit reads, the D2H copy) ran, per the §7 escape-and-
  outlive contract documented at .hpp:26-29.
  (c) upload: the H2D cudaMemcpy (.cu:86-89) is synchronous into freshly-allocated
  buffers on device_id; the data is in place before the handle is returned, and the
  buffers outlive into the returned handle. No premature free. N/A.
- 14.5 Missing frees on error paths: NONE. Both transport functions are fully RAII-
  protected on the throw path (STEPPE_CUDA_CHECK throws a typed CudaError). In to_host,
  if any cuda* check throws (cudaGetDevice/SetDevice .cu:50,52, the two cudaMemcpy
  .cu:57-60), stack unwinding runs ~RegisteredHostRegion (un-pins pin_f2/pin_vp),
  ~F2BlockTensor out (frees the host vectors), and the `G restore` guard (restores the
  device) — no orphaned pin, no leaked host alloc. In upload_f2_blocks_to_device, if the
  SECOND cudaMemcpy (.cu:88-89) throws AFTER out.impl was constructed (.cu:82) and both
  DeviceBuffers allocated (.cu:83-84), unwinding destroys `out` -> ~unique_ptr<Impl> ->
  the two ~DeviceBuffer<double> -> cudaFree both resident allocations. The
  make_unique<Impl> (.cu:82) is itself exception-safe (a throw there frees the Impl).
  No manual cleanup is owed because nothing is manually allocated; every owner frees in
  its dtor regardless of how the scope exits. N/A.
-->

## Group 15 — Memory: transfers

- [15.3][LOW] src/device/cuda/device_f2_blocks.cu:86-89 — asymmetry: `upload_f2_blocks_to_device`'s two H2D `cudaMemcpy` read from PAGEABLE host source (`host.f2.data()`/`host.vpair.data()`), while the mirror `to_host` PINS its D2H destinations via `RegisteredHostRegion` (.cu:55-56). In `replicate_f2` (model_search.cpp:178-193) the same `host` tensor is the source for all G-1 upload iterations (a stable, reused base pointer), so it is exactly the "stable, reused buffer" the PinnedRegistryCache rationale (pinned_buffer.cuh:213-245) says benefits from pinning, yet it stays pageable — at scale this is a multi-GB synchronous staged copy (the .hpp:86-91 TODO measures ~8.72 GB / ~3.8 s cold, P=600). Suggested: optionally wrap the H2D source pages in a `RegisteredHostRegion` per buffer (mirror .cu:55-56) — but note this whole no-peer host-bounce path is already DEFERRED (.hpp:86-91; real fix = per-device precompute that removes the transfer entirely), and the synchronous-`cudaMemcpy` single-shot register tax may not amortize, so this is a documented-tradeoff LOW, not a bug.

<!--
Notes (why otherwise clean), so the result is auditable:
- 15.1 memcpy in a loop that should be hoisted/batched/kept-resident: NONE in the unit.
  Neither to_host (.cu:39-62) nor upload_f2_blocks_to_device (.cu:68-91) contains a loop;
  each issues EXACTLY two synchronous cudaMemcpy (f2 + vpair) of the full [P×P×n_block]
  tensor in ONE shot per slab (.cu:57-60, :86-89) — already the maximally-batched single
  bulk transfer, not a per-block/per-element loop. The whole unit IS the kept-resident
  cure: DeviceF2Blocks holds f2/vpair in VRAM (impl, _impl.cuh:14-17), f2_device()/
  vpair_device() (.cu:25-30) hand the fit BORROWED device pointers so the S8 hot path
  reads VRAM with ZERO copy, and to_host is the single OPT-IN D2H (.cu:32, .hpp:61),
  explicitly NOT the fit handoff (cuda_backend.cu:277 "The hot path ... does NOT call
  this"). The one transfer that IS inside a caller loop — upload at model_search.cpp:190
  in the `for g` device loop — is the irreducible cross-card REPLICATION (one H2D per
  distinct target device, G-1 cards), NOT a redundant re-copy: each iteration targets a
  DIFFERENT device. Its paired source materialization (to_host) is correctly HOISTED out
  of that loop (model_search.cpp:179, guarded by need_host) and the result vector is
  pre-reserved (:184) so the move-only handles do not reallocate. No transfer that should
  be hoisted/batched is left in a loop; no resident value is needlessly round-tripped.
- 15.2 direction enum not matching the actual transfer: CORRECT at both call sites.
  to_host (.cu:57-60): dst = out.f2.data()/out.vpair.data() (host std::vector), src =
  impl->f2.data()/impl->vpair.data() (device DeviceBuffer) -> cudaMemcpyDeviceToHost —
  matches (device source, host dest). upload (.cu:86-89): dst = out.impl->f2.data()/
  vpair.data() (freshly-allocated device DeviceBuffer on device_id), src = host.f2.data()/
  host.vpair.data() (host std::vector) -> cudaMemcpyHostToDevice — matches (host source,
  device dest). No D2H labeled H2D or vice versa; no DeviceToDevice mislabel (there is no
  D2D here — cross-card transport goes host-side precisely because the no-P2P tier cannot
  do a D2D). Directions are exact.
- 15.3 pageable host memory for frequent transfers: the FREQUENT/hot transfers do not
  exist in this unit (the hot path is zero-copy VRAM reads via f2_device/vpair_device).
  The to_host D2H destinations ARE pinned for the window (RegisteredHostRegion pin_f2/
  pin_vp, .cu:55-56), the established §11.4/§12 parity-neutral graceful-degrade lever
  (pinned_buffer.cuh:126-150). The only pageable transfer is the upload H2D source above
  (recorded LOW); it is a COLD, once-per-device-per-precompute path (model_search.cpp:190)
  on the explicitly-DEFERRED multi-GPU host-bounce, where the per-call cudaHostRegister
  tax (~50-360 ms, pinned_buffer.cuh:216) plausibly does not pay back a single multi-GB
  synchronous copy that is never re-amortized — i.e. leaving it pageable is consistent
  with the PinnedRegistryCache "WHY NOT the D2H result destinations" reasoning
  (pinned_buffer.cuh:236-240), just applied to the H2D source. Not a HIGH/MED defect.
-->

## Group 16 — RAII: ownership & wrapper hygiene

- [16.2][LOW] src/device/device_f2_blocks.hpp:75,53,48-51 — the documented invariant `impl; // null iff empty()` is violated on a MOVED-FROM object. `DeviceF2Blocks(DeviceF2Blocks&&)`/`operator=` are `=default` (device_f2_blocks.cu:22-23), so the move resets `impl` to null (unique_ptr move) but leaves the moved-from shape scalars `P`/`n_block`/`device_id` at their old nonzero values (scalar move is a copy). Thus a moved-from object has `impl == nullptr` yet `empty()` (`n_block <= 0 || P <= 0`, .hpp:53) can still return false and `size()` (.hpp:48-51) nonzero — contradicting both the "null iff empty()" comment and the f2_device/vpair_device "null iff empty()" doc (.hpp:57-59). The accessors are still SAFE (f2_device/vpair_device guard on `impl ?`, .cu:26,29; to_host guards `!impl`, .cu:47), so no null-deref today, and no current caller queries empty()/size() on a moved-from handle (all moves feed return values / the `owned` vector, then the husk is destroyed — model_search.cpp:166, cuda_backend.cu:303-304). Latent, not a live bug. Suggested: either reset the shape scalars on move (user-written move that nulls P/n_block/device_id alongside impl) or weaken the doc to "impl null ⇒ no resident buffers (but a moved-from husk may report stale shape)"; the `=default` move is otherwise correct.

<!--
Notes (why otherwise clean), so the result is auditable:
- 16.1 Wrap EVERY resource: every resource at this seam is RAII-wrapped, none raw.
  The ONLY owned payload is the pimpl std::unique_ptr<Impl> (.hpp:75), and Impl holds
  two DeviceBuffer<double> (f2, vpair — _impl.cuh:15-16), the allowlisted move-only
  cudaMalloc/cudaFree owner (device_buffer.cuh:64-93,117). The transient host-pin in
  to_host is RegisteredHostRegion pin_f2/pin_vp (.cu:55-56), the allowlisted
  cudaHostRegister/cudaHostUnregister RAII guard (pinned_buffer.cuh:151-211). The
  device-context switch is wrapped by the inline `struct G { int d; ~G(){
  (void)cudaSetDevice(d);} } restore{prev}` RAII guard (.cu:51, :79) so cudaSetDevice
  is restored on EVERY exit (including the throw path). There are NO streams, events,
  graphs/graph-execs, texture/surface objects, memory pools, CUDA arrays, or library
  handles (cuBLAS/cuSOLVER) created in this unit — it issues only synchronous
  cudaMemcpy + cudaGet/SetDevice (.cu:50,52,57,59,78,80,86,88), none of which is an
  owned handle. No naked cudaMalloc/cudaHostAlloc/cuda*Create anywhere (grep-clean).
  Nothing leaks.
- 16.2 Move-only + null-on-move: COPY is deleted on DeviceF2Blocks (.hpp:36-37) and on
  every owner it touches (DeviceBuffer device_buffer.cuh:90-91; RegisteredHostRegion
  pinned_buffer.cuh:188-189) — no copyable freeing wrapper, so no double-free. The
  defaulted DeviceF2Blocks move nulls the moved-from `impl` (unique_ptr move) and
  empties block_sizes (vector move); the owned DeviceBuffer/RegisteredHostRegion both
  std::exchange their pointer to null on move (device_buffer.cuh:78-88;
  pinned_buffer.cuh:177-186) — moved-from handles are null, no double-free on the husk.
  The ONLY un-nulled-on-move state is the shape scalars (recorded LOW above) — a
  doc-invariant blemish, NOT a free-correctness bug (the freed resource, impl, IS
  nulled). The inline `struct G` guard is a function-local non-movable scope object
  (never moved), so it owns its restore exactly once.
- 16.3 Rule of five for a freeing dtor: COMPLETE and correct. DeviceF2Blocks declares
  all five — dtor, move-ctor, move-assign, deleted copy-ctor, deleted copy-assign
  (.hpp:32-37) — and defines the four non-deleted ones out-of-line `=default` in the
  .cu (.cu:20-23) precisely so unique_ptr<Impl> sees a COMPLETE Impl at the
  instantiation point (the pimpl-with-incomplete-type rule; Impl is defined in
  _impl.cuh:14-17, included by the .cu at line 6). DeviceF2Blocks itself frees nothing
  by hand — the freeing dtor is the IMPLICIT ~unique_ptr<Impl> -> ~Impl -> the two
  ~DeviceBuffer<double> -> cudaFree, so `~DeviceF2Blocks() = default` is right. The
  underlying freeing wrappers (DeviceBuffer, RegisteredHostRegion) each independently
  satisfy rule-of-five with a noexcept teardown (device_buffer.cuh:78-93;
  pinned_buffer.cuh:177-191). No half-implemented rule-of-five, no missing move that
  would force a copy of a non-copyable.
- 16.4 Single clear ownership; raw kernel ptrs non-owning: SINGLE owner = the pimpl
  unique_ptr (.hpp:75); the resident bytes are owned once, by Impl's two DeviceBuffer.
  f2_device()/vpair_device() return BORROWED `const double*` views (.cu:25-30, .hpp:58-59)
  documented "Borrowed device pointers" — non-owning, never freed by any consumer (the
  fit reads them in VRAM). No raw owning pointer escapes. The owning handle is NEVER
  passed by value: every external use takes `const DeviceF2Blocks&` (qpadm_fit.cpp:215,277;
  model_search.cpp:25,56,83,169,199; f4_matrix.hpp:24; backend.hpp:319,428,575,718) or a
  borrowed `const DeviceF2Blocks*` (model_search.cpp:165), and producers return BY VALUE
  via move (cuda_backend.cu:296, p2p_combine.cu:237, upload .cu:69) into the `owned`
  RAII vector (model_search.cpp:166) — no by-value owning-wrapper parameter anywhere.
  upload_f2_blocks_to_device takes the source by `const F2BlockTensor&` (.hpp:92, .cu:68),
  not by value. Clean.
- 16.5 Don't reinvent the wrapper: APPROPRIATE reuse, no reinvention. The handle uses
  std::unique_ptr<Impl> (.hpp:75) for the pimpl — the standard tool, not a hand-rolled
  owning-pointer-with-manual-delete. The device bytes use DeviceBuffer<double>
  (.cu:83-84), one of the three allowlisted cudaMalloc/cudaFree owners
  (device_buffer.cuh:5-7) — the project's canonical device-memory RAII, not a fresh
  cudaMalloc/cudaFree pair. The host pin uses RegisteredHostRegion (.cu:55-56), the
  canonical cudaHostRegister guard. The one bespoke wrapper, the inline `struct G`
  device-restore guard (.cu:51, :79), is GENUINELY needed (there is no std/thrust RAII
  for "save+restore current CUDA device") and is the established project idiom (mirrors
  p2p_combine.cu:79-85). thrust::device_vector is deliberately NOT used — the project
  standardizes on DeviceBuffer for the size-overflow precheck and §11.2 exact bytes()
  (device_buffer.cuh:17-22,99-108), a real reason to prefer the in-house owner. No
  wheel reinvented; the only LOW-grade duplication (the `struct G` guard repeated at
  three+ sites) is already captured under Group 7.4 as a hygiene fold, not a Group 16
  ownership defect.
-->

## Group 17 — RAII: lifetime & deleter pitfalls (CUDA-specific)

- [17.5][MED] src/device/device_f2_blocks.hpp:26-29,33 / src/device/cuda/device_f2_blocks.cu:21,82-84 — the resident f2/vpair DeviceBuffers can be cudaMalloc'd on a NON-current device (upload_f2_blocks_to_device sets the current device to `device_id` at .cu:80 before constructing both DeviceBuffer<double> at .cu:83-84), but `~DeviceF2Blocks() = default` (.cu:21) → ~unique_ptr<Impl> → ~DeviceBuffer simply calls `cudaFree(ptr_)` (device_buffer.cuh:117) with NO cudaSetDevice-to-alloc-device + restore. The handle moves into the `owned` vector (model_search.cpp) and is destroyed later in whatever device context happens to be current, which is NOT guaranteed to be `device_id`. The handle's own doc asserts this is safe because "cudaFree is pointer-device-aware" (.hpp:28) — which is TRUE for cudaFree itself (it frees the allocation regardless of current device), so this is NOT a live free-on-wrong-device bug today. But the device-correct-free guarantee is provided ENTIRELY by cudaFree's runtime semantics, not by the owner: DeviceBuffer is device-agnostic and records no alloc device, so the to_host()/upload prologue's set+restore guard (.cu:49-52, :77-80) covers only the malloc/memcpy, NOT the eventual free. This is the classic 17.5 pattern (free without restoring the alloc device); it is the canonical multi-GPU RAII hazard for the deferred no-peer path even though the current cudaFree contract masks it. Suggested: either keep relying on cudaFree's documented pointer-device-awareness but pin that assumption with a comment on the DeviceBuffer dtor / the §7 escape contract, or (more robust for any future per-device-pool / stream-ordered free) have the owner record its alloc device and set+restore in reset() — defense for the multi-GPU teardown path.

<!--
Notes (why otherwise clean), so the result is auditable:
- 17.1 Non-throwing dtor + teardown order: SATISFIED. ~DeviceF2Blocks (.cu:21) is
  =default and frees nothing by hand; the implicit teardown chain (~unique_ptr<Impl>
  -> ~Impl -> two ~DeviceBuffer<double>) ends in DeviceBuffer::reset() which CANNOT
  throw — a nonzero cudaFree status at teardown is routed to STEPPE_LOG_WARN, never
  thrown (device_buffer.cuh:111-125, explicitly "Destructor never throws ... fail-fast
  must not become fail-silent"). The transient host-pin teardown ~RegisteredHostRegion
  likewise swallows cudaHostUnregister into STEPPE_LOG_WARN (pinned_buffer.cuh:197-208).
  The two inline `struct G` device-restore guard dtors discard the status with
  `(void)cudaSetDevice(d)` (.cu:51, :79) — correct non-throwing teardown idiom. No dtor
  in the unit can throw during unwinding. Teardown-ORDER / context-gone-at-exit: NOT a
  concern here — DeviceF2Blocks is a move-only RAII handle created/moved/freed
  deterministically DURING the run (it flows producer -> `owned` vector ->
  fit -> scope-end, model_search.cpp / cuda_backend.cu:303-304), never a static/global
  with at-exit destruction after the CUDA primary context may already be torn down. So
  the "context may be gone at exit" guard the task warns about does not apply to this
  unit's objects (no static-duration DeviceF2Blocks / DeviceBuffer anywhere).
- 17.2 Deleter matches allocator: EXACT, all families paired. The resident bytes are
  cudaMalloc'd (device_buffer.cuh:74) and freed with cudaFree (device_buffer.cuh:117) —
  matched. The host pins are cudaHostRegister'd (pinned_buffer.cuh:164) and released
  with cudaHostUnregister (pinned_buffer.cuh:201) — matched. There is NO
  cudaMallocAsync/cudaFreeAsync, NO cudaMallocHost/cudaHostAlloc owned in THIS unit's
  paths (the in-place register guard is register/unregister, not alloc/free), and NO
  cudaMallocArray/cudaFreeArray (no CUDA arrays/textures). No cudaFree-on-malloc'd-host,
  no free()-on-cudaMalloc'd, no cross-family mismatch. The pimpl Impl itself is a plain
  host-heap object: make_unique<Impl> (.cu:82) -> default `delete` via ~unique_ptr<Impl>
  — single-object new/delete, the correct pairing (NOT delete[] — see 17.3).
- 17.3 unique_ptr<T[]> over cudaMalloc: N/A — does not occur. The ONLY std::unique_ptr
  in the unit is `std::unique_ptr<Impl> impl` (.hpp:75), a SINGLE-OBJECT (non-array)
  unique_ptr over a host-heap Impl from make_unique<Impl> (.cu:82); its default deleter
  is `delete` (not `delete[]`), which correctly matches make_unique's `new` — no array
  form, no UB. Crucially, NO unique_ptr is ever placed over a cudaMalloc'd device
  pointer here: the device bytes are owned by DeviceBuffer<double> (custom cudaFree
  reset(), device_buffer.cuh:117), NOT by a unique_ptr with std::default_delete that
  would call delete[]/delete on a device pointer (UB). The 17.3 footgun (unique_ptr<T[]>
  whose delete[] runs on a cudaMalloc pointer) is structurally absent.
- 17.4 RAII vs async lifetime (free/unregister at scope exit != async work done):
  N/A — no async lifetime hazard in this unit because EVERY transfer is SYNCHRONOUS
  cudaMemcpy, not cudaMemcpyAsync. (a) to_host: the two D2H are synchronous cudaMemcpy
  (.cu:57-60); each fully completes (host-blocking) before control reaches the function
  end where ~RegisteredHostRegion un-pins pin_f2/pin_vp (.cu:55-56 -> scope exit .cu:61).
  So the unregister can NEVER race a live DMA — the copy is already done. (Contrast the
  async staging in cuda_backend.cu, which needs an explicit cudaStreamSynchronize before
  its pinned staging dies; the sync API here owes no such guard.) (b) upload: the two
  H2D are synchronous cudaMemcpy (.cu:86-89) into freshly-allocated DeviceBuffers; the
  data is fully in place before the handle is returned/the locals leave scope — no free
  precedes an outstanding copy. (c) The resident DeviceBuffers free only in ~Impl when
  the moved-out handle dies, strictly AFTER the producer stream was drained
  (cuda_backend.cu cudaStreamSynchronize before the buffers move into the handle) and
  after every consumer ran — the §7 escape-and-outlive contract (.hpp:26-29). No
  free-before-async-completes, no use-after-free across streams. (Already cross-checked
  under Group 14.4.) No stream is tied to or owned by this unit.
-->

## Group 18 — Correctness traps (wrong numbers, not crashes)

No Group 18 issues found.

<!--
Notes (why clean), so the result is auditable:
- This unit contains NO CUDA kernels and NO device-side cooperative code: there is no
  __global__, no __device__ function, no `<<<grid, block>>>` launch, no shared memory
  (__shared__), no thread/block index (threadIdx/blockIdx/blockDim), no __syncthreads,
  no __syncwarp, and no warp intrinsics anywhere in the three files. It is host-side
  transport scaffolding (out-of-line special members .cu:20-23, the device-pointer
  accessors f2_device/vpair_device .cu:25-30, the single to_host() D2H .cu:39-62, the
  upload_f2_blocks_to_device() H2D .cu:68-91) + the Impl struct (_impl.cuh:14-17) + the
  CUDA-free handle decl (.hpp). The only CUDA runtime calls are cudaGetDevice /
  cudaSetDevice / cudaMemcpy (.cu:50,52,57,59,78,80,86,88). EVERY Group 18 task is a
  device-kernel hazard and is therefore structurally N/A:
- 18.1 Divergent __syncthreads() (barrier reached by only some threads): N/A — there is
  no __syncthreads/__syncwarp/barrier call in the unit at all, and no kernel to host one.
- 18.2 Missing __syncthreads (shared-mem RAW/WAR without a barrier): N/A — there is no
  __shared__ memory and no thread-to-thread shared communication anywhere; the two
  cudaMemcpy moves are single bulk host<->device transfers, not cooperative kernel writes.
- 18.3 Warp-synchronous assumptions (relying on lockstep without __syncwarp/_sync): N/A —
  no warp-level code, no implicit-lockstep assumption; nothing executes per-thread here.
- 18.4 Non-_sync warp intrinsics (__shfl/__ballot/__any/__all without explicit mask): N/A —
  none present (already confirmed clean under Group 2.3). No warp intrinsic of any kind.
- 18.5 Missing bounds guard (no if(idx<n) → OOB): N/A — there is no thread index to
  guard. The full [P×P×n_block] tensor (which can exceed 2^31 elements at scale) is moved
  in ONE cudaMemcpy of `bytes = total * sizeof(double)` (.cu:54,85), where `total` is the
  size_t-widened element count from size() (.hpp:48-51) — bounds correctness here is the
  byte-count derivation, already verified non-overflowing under Group 4. cudaMemcpy does
  not iterate a thread grid, so the "thread runs past n" failure mode does not arise. The
  degenerate path is guarded (total==0 || !impl → early return, .cu:47,75).
- 18.6 Cross-thread read without barrier/atomic: N/A — no kernel, no cross-thread data
  dependence; the synchronous cudaMemcpy fully completes before any subsequent read.
- 18.7 Order-dependent float reduction assuming a fixed thread-execution order: N/A —
  there is NO reduction in this unit (no sum/accumulate over threads, no atomicAdd, no
  cooperative combine). The transport is a verbatim raw byte copy (documented bit-faithful,
  preserves −0.0, .cu:65-66/.hpp:84), which performs no floating-point arithmetic at all,
  so the §12 fixed-order-determinism requirement is satisfied trivially (no FP op to order).
-->

## Group 19 — Performance: debug leftovers

No Group 19 issues found.

<!--
Notes (why clean), so the result is auditable:
- 19.1 Stray cudaDeviceSynchronize(): NONE. The only CUDA runtime calls in the unit are
  cudaGetDevice / cudaSetDevice / cudaMemcpy (device_f2_blocks.cu:50,52,57,59,78,80,86,88)
  plus the cudaMalloc/cudaFree behind DeviceBuffer and cudaHostRegister/Unregister behind
  RegisteredHostRegion. There is NO cudaDeviceSynchronize, NO cudaStreamSynchronize, and
  NO cudaThreadSynchronize anywhere in the three files — so no global serialization point
  was left from debugging. The cudaMemcpy calls are the SYNCHRONOUS variant (not
  cudaMemcpyAsync), which self-blocks by design (the to_host D2H / upload H2D must complete
  before the host buffer is read / the handle returns); that is a required correctness
  barrier on a cold opt-in path, not a stray debug sync. STEPPE_CUDA_CHECK_KERNEL (the
  NDEBUG-gated intentional debug sync called out by the task) is NOT used here — there is
  no kernel launch to check. N/A.
- 19.2 Leftover printf / #if 0: NONE. No printf / fprintf / std::cout / device printf in
  any file — the unit emits no diagnostic output. The ONLY preprocessor conditionals are
  the legitimate header include guards (_impl.cuh:6-7,20 #ifndef/#define ... #endif;
  device_f2_blocks.hpp:7-8,95 #ifndef/#define ... #endif). There is NO `#if 0`, no commented
  scaffolding-out block, no debug-only #ifdef DEBUG / #ifndef NDEBUG print (already
  cross-confirmed clean under Group 3 dead-code). N/A.
- 19.3 Redundant __syncthreads(): N/A — there are NO kernels in this unit (no __global__,
  no __device__ code), hence no __syncthreads / __syncwarp / barrier of any kind to be
  redundant (already noted under Group 18.1/18.2). Nothing to flag.
-->

## Group 20 — Performance: memory access

No Group 20 issues found.

<!--
Notes (why clean), so the result is auditable:
- This unit has NO CUDA kernels and NO device-side code: there is no __global__, no
  __device__ function, no `<<<grid, block>>>` launch, no threadIdx/blockIdx/blockDim,
  no __shared__ memory, and no per-thread global loads anywhere in the three files. It
  is host-side transport scaffolding (out-of-line special members device_f2_blocks.cu
  :20-23, the device-pointer accessors f2_device/vpair_device .cu:25-30, the single
  to_host() D2H .cu:39-62, the upload_f2_blocks_to_device() H2D .cu:68-91) + the Impl
  struct (_impl.cuh:14-17) + the CUDA-free handle decl (.hpp). The only CUDA runtime
  calls are cudaGetDevice / cudaSetDevice / cudaMemcpy (.cu:50,52,57,59,78,80,86,88)
  plus cudaMalloc/cudaFree via DeviceBuffer and cudaHostRegister/Unregister via
  RegisteredHostRegion. ALL THREE Group 20 tasks are device-kernel memory-access
  hazards and are therefore structurally N/A:
- 20.1 Uncoalesced global access (consecutive threads not hitting consecutive global
  addresses): N/A — there is no kernel and no thread grid, so there is no per-thread
  global-address stride to coalesce. The full [P×P×n_block] f2/vpair tensor is moved by
  ONE bulk synchronous cudaMemcpy of `bytes = total * sizeof(double)` over a single
  contiguous source->dest range (.cu:57-60 D2H, .cu:86-89 H2D); cudaMemcpy DMA is a
  contiguous copy of the whole region, not a thread-strided access pattern, so the
  coalescing failure mode cannot arise. The column-major [P×P×n_block] layout
  (i + P·j + P·P·b, .hpp:55-57) is consumed by kernels in OTHER TUs (the fit), not here.
- 20.2 Shared-memory bank conflicts: N/A — there is NO __shared__ memory anywhere in the
  unit (no kernel to declare it). No serializing shared-bank access pattern can exist.
- 20.3 Re-reading the same global value instead of caching in a register/shared: N/A —
  there is no device-side load at all to redundantly re-read. On the HOST side, the
  shape values that ARE read more than once are already cached in locals: `total =
  size()` is computed once and reused for the resize/byte calc (.cu:44, :74), `bytes =
  total * sizeof(double)` once per function then used for both the f2 and vpair copy
  (.cu:54 -> :57,:59; .cu:85 -> :86,:88), and `impl->f2.data()`/`impl->vpair.data()`
  /`out.f2.data()` etc. are each dereferenced once per copy. No global/device value is
  loaded twice in a thread where a register cache was owed (there is no thread). N/A.
-->

## Group 21 — Performance: occupancy & registers

No Group 21 issues found.

<!--
Notes (why clean), so the result is auditable:
- This unit has NO CUDA kernels and NO device-side code: there is no __global__, no
  __device__ function, no `<<<grid, block>>>` launch, no threadIdx/blockIdx/blockDim,
  no __shared__ memory, no warp intrinsics, and no register-hint pragmas/qualifiers
  anywhere in the three files. It is host-side transport scaffolding (out-of-line
  special members device_f2_blocks.cu:20-23, the device-pointer accessors
  f2_device/vpair_device .cu:25-30, the single to_host() D2H .cu:39-62, the
  upload_f2_blocks_to_device() H2D .cu:68-91) + the Impl struct (_impl.cuh:14-17) +
  the CUDA-free handle decl (.hpp). The only CUDA runtime calls are cudaGetDevice /
  cudaSetDevice / cudaMemcpy (.cu:50,52,57,59,78,80,86,88) plus cudaMalloc/cudaFree
  via DeviceBuffer and cudaHostRegister/Unregister via RegisteredHostRegion. ALL FOUR
  Group 21 tasks are device-kernel occupancy/register hazards and are therefore
  structurally N/A:
- 21.1 Warp divergence serializing a warp: N/A — there is no kernel and no warp; no
  per-thread branching exists. The only host-side branches are cold guards (the
  `total == 0 || !impl` early return .cu:47, the `total == 0` early return .cu:75, the
  `n_block < 0 ? 0` clamp .hpp:50/.cu:42,71, the `impl ?` accessor null-checks
  .cu:26,29) — host control flow, not a warp-divergent device branch. Nothing
  serializes a warp because no warp executes here.
- 21.2 Excessive shared memory cratering occupancy: N/A — there is NO __shared__
  declaration in the unit (no kernel to host one), so no per-block shared-memory
  footprint exists to limit occupancy. The resident DeviceBuffer<double> f2/vpair
  (_impl.cuh:15-16) are global-memory allocations, not shared memory.
- 21.3 Register spills (monolithic kernel spilling to local) — split candidate: N/A —
  there are NO kernels, so there is no register pressure, no local-memory spill, and
  no monolithic-kernel split candidate. The host functions (to_host, upload, the
  accessors) are tiny straight-line transport routines compiled for the host ABI, not
  device register-allocated code; "register spill" does not apply to them.
- 21.4 Missing register hints (#pragma unroll / __launch_bounds__ / __forceinline__):
  N/A and correctly ABSENT. There is no tight device loop to #pragma unroll (the only
  iteration over the [P×P×n_block] tensor is the single bulk cudaMemcpy, .cu:57-60,
  :86-89 — a DMA, not an unrollable per-element loop), no kernel launch to bound with
  __launch_bounds__, and no device function whose inlining __forceinline__ would steer.
  Adding any of these here would be a no-op at best (they are device-code attributes
  with no kernel to attach to). The task's "deliberately, not reflexively" guidance is
  satisfied trivially: the unit has no place a register hint could help.
-->

## Group 22 — Performance: compute & launch

No Group 22 issues found.

<!--
Notes (why clean), so the result is auditable:
- This unit has NO CUDA kernels, NO kernel launches, and NO loops: there is no
  __global__, no __device__ code, no `<<<grid, block>>>` launch syntax, and no for/
  while loop in any of the three files. It is host-side transport scaffolding
  (out-of-line special members device_f2_blocks.cu:20-23, the device-pointer accessors
  f2_device/vpair_device .cu:25-30, the single to_host() D2H .cu:39-62, the
  upload_f2_blocks_to_device() H2D .cu:68-91) + the Impl struct (_impl.cuh:14-17) + the
  CUDA-free handle decl (.hpp). The only CUDA runtime calls are cudaGetDevice /
  cudaSetDevice / cudaMemcpy (.cu:50,52,57,59,78,80,86,88) plus cudaMalloc/cudaFree via
  DeviceBuffer and cudaHostRegister/Unregister via RegisteredHostRegion. ALL FOUR
  Group 22 tasks are device-compute / launch-cost hazards and are therefore
  structurally N/A:
- 22.1 Atomics where a reduction/scan would be cheaper: N/A — there is NO atomic of any
  kind (no atomicAdd/atomicCAS/atomicMax/etc.) and NO reduction/scan anywhere in the
  unit. The transport is a verbatim raw byte copy (cudaMemcpy, bit-faithful, .cu:57-60,
  :86-89); it performs no accumulation, no combine, no per-thread contention. No atomic
  to demote to a tree/warp reduction.
- 22.2 Integer div/mod in loops (precompute / shift-mask power-of-two strides): N/A —
  there are NO loops and NO per-element index arithmetic in this unit. There is no `/`
  or `%` on any hot index at all: the only integer arithmetic is the one-shot byte-count
  `bytes = total * sizeof(double)` (.cu:54,85, a single MULTIPLY, computed once per
  function, not in a loop) and the size_t-widened element product in size()
  (P*P*n_block, .hpp:48-51, also a one-shot multiply). No division/modulo, no strided
  loop counter, so the "div/mod in a tight loop" cost does not arise.
- 22.3 Loop-invariant work / repeated index recomputation to hoist: N/A for loops (none
  exist); and the host-side values that ARE reused are already hoisted — `total =
  size()` is computed ONCE and reused for resize + byte calc (.cu:44 -> :45-46,:54;
  .cu:74 -> :85), `bytes` is computed ONCE per function then used for BOTH the f2 and
  vpair copy (.cu:54 -> :57,:59; .cu:85 -> :86,:88), and `prev`/`device_id` are read once
  each into the guard. No invariant is recomputed inside any repeated context (there is
  no repeated context). Nothing to hoist. (Already cross-checked under Group 20.3.)
- 22.4 Launch overhead / many small launches → fuse or CUDA-graph capture: N/A — there
  is NOTHING launched in this unit (no kernel, no `<<<...>>>`), so there is no per-launch
  cost to amortize and no fuse/graph-capture candidate. The two transport functions each
  issue EXACTLY two synchronous cudaMemcpy of the FULL [P×P×n_block] tensor in one shot
  (the maximally-batched single bulk transfer, not a flurry of small copies); the
  cross-card replication loop that calls upload (model_search.cpp:190) is in a DIFFERENT
  TU and runs one bulk H2D per distinct target device (G-1 cards), not many tiny launches
  — and that whole no-peer host-bounce path is the explicitly DEFERRED multi-GPU cost
  (.hpp:86-91), profiler-measured (~8.72 GB / ~3.8 s cold, P=600), where the fix is a
  per-device precompute (remove the transfer), not graph capture. CUDA-graph capture is a
  device-kernel-launch optimization with no launch in this unit to capture. The task's
  "only where the profiler confirms it" guard further blocks any speculative fuse here.
-->

