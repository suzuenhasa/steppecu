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


