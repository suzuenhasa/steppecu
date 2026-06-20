# Review findings — src__core__fstats__f2_blocks_multigpu

Files: src/core/fstats/f2_blocks_multigpu.cpp, src/core/fstats/f2_blocks_multigpu.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

Notes (verified clean, not findings):
- 4.2/4.6 (index width / overflow at scale): the only at-scale index arithmetic is the full-tensor sizing in the host-staged arm. `slab = (std::size_t)P * (std::size_t)P` (line 204) widens BOTH operands to size_t before the multiply, and `total = slab * (std::size_t)n_block` (line 205) is computed in size_t (slab is already size_t). For the max case P=2500, n_block=757 the product ~4.7e9 exceeds 2^31 but is held in size_t — correct. The `for (int b=0; b<n_block; ++b)` loop (line 372) keeps `b` tiny and widens via `(std::size_t)b` before indexing.
- 4.1 (float/double): orchestration-only file, no numeric literals or float math — N/A.
- 4.3 (allocation sizing): no cudaMalloc/new/DeviceBuffer; only std::vector resize/assign (element-counted, no sizeof confusion) — N/A.
- 4.4/4.5 (countdown / signed-unsigned compare): single ascending `int b < int n_block` loop; no unsigned countdown, no signed/unsigned mismatch.
- 4.7 (host/device pointer typing): host `.data()` pointers (line 212) flow only into the host-pure core combine; the device-resident path uses opaque typed handles (DevicePartial / DeviceF2Blocks). Space discipline is clean.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (verified clean, not findings):
- This is a CUDA-FREE, host-pure orchestration unit in `steppe::core` (is_cuda=false): it includes NO CUDA header (`<cuda_runtime.h>` etc. absent; confirmed in .cpp lines 29-51 and .hpp lines 22-28) and issues no CUDA API call — all device interaction is through the CUDA-free `ComputeBackend` seam and opaque handles (DevicePartial / DeviceF2Blocks). None of the Group 2 patterns can occur here by construction.
- 2.1 (dropped archs Maxwell/Pascal/Volta, sm_50/60/70, CUDA_ARCH): no arch flags, no CMake arch lists, no `__CUDA_ARCH__` guards in either file — N/A (those live in CMake/.cu units, not this host TU).
- 2.2 (texture/surface references): no `texture<>`/`surface<>`/`cudaBindTexture*`/`cudaBindSurface*` — none; this unit touches no texture memory.
- 2.3 (non-`_sync` warp intrinsics): no warp intrinsics at all (no `__shfl`/`__any`/`__all`/`__ballot`); host-only code, no `__device__`/`__global__` kernels.
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): no `cudaThreadSynchronize` (or any cuda* runtime call). The `cudaMemcpyPeer`/`cudaMemcpyPeerAsync`/`cudaDeviceEnablePeerAccess` tokens at .cpp lines 22, 109, 176 and .hpp line 17 are DESCRIPTIVE COMMENTS only, referencing the P2P transport implemented in the separate `device/cuda/p2p_combine.cu` unit — not API calls in this file, so not in scope here.

## Group 3 — Dead / commented-out code

- [3.4][LOW] src/core/fstats/f2_blocks_multigpu.cpp:358 — `out.P = P;` in compute_f2_blocks_multigpu_tiered is computed but never read: all three switch arms overwrite it unconditionally before any read (Resident `out.P = out.resident.P` line 386, HostRam `out.P = out.host.P` line 399, Disk `out.P = out.disk.P` line 419). The companion `out.n_block = ...` (line 359) is NOT dead — the HostRam arm only overwrites it under the `if (out.host.n_block >= 0)` guard (line 398), so its initial value is a live default. Suggested: drop the line 358 `out.P = P` (or keep both as intentional defensive defaults and leave as-is — it is harmless, parity-neutral hygiene only).

Notes (verified clean, not findings):
- 3.1 (commented-out code kept "just in case"): the file is heavily commented, but every `//` block is design/rationale PROSE (the §4 gate banner, parity argument, doc cross-references) — no statements or expressions are commented out anywhere in either file. None.
- 3.2 (unreachable code): no `#if 0`. Each `if (G == 1)` early-returns but the trailing code is reached for G != 1; the `if (use_p2p)` arm returns and the host-staged code below is reached when `!use_p2p`; the tiered `switch` covers all OutputTier enumerators and the trailing `return out` (line 424) is reachable. The `G >= 2` term in `use_p2p` (lines 163, 272) is documented as "dead-true" (the G==1 fast-path returned above), but that is an INTENTIONAL, parity-neutral predicate-completeness choice (cleanup X6, so the code matches the four-term gate documented across the seam) — not dead code, explicitly justified at lines 149-156; NOT flagged.
- 3.3 (unused symbols): all includes are used — `<cstddef>`(size_t), `<cstdlib>`(std::getenv lines 354/409), `<span>`(std::span), `<stdexcept>`(std::runtime_error), `<vector>`(std::vector); `block_partition_rule.hpp`(block_ranges/BlockRange line 367), `tier_select.hpp`(resolve_output_tier/free_host_ram_bytes/OutputTier lines 351-355), `stream_f2_blocks.hpp`(StreamTarget lines 393/412), `p2p_combine.hpp`(combine_f2_partials_resident_device), and every device handle/header has a use. All three entries use all of their params (resources, Q, V, N, partition, precision). No unused locals — `P`/`M`/`n_block`/`G`/`use_p2p`/`shards`/`shards_span`/`slab`/`total`/`free_vram`/`free_host`/`tier`/`path`/`ranges` are each read after assignment.

