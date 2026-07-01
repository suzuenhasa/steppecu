# src__device__f2_blocks_out
Files: /home/suzunik/steppe/src/device/f2_blocks_out.hpp, /home/suzunik/steppe/src/device/cuda/f2_blocks_out.cu
Subsystem: device-cuda

## Findings

### G3
- [G3.dead][LOW] src/device/cuda/f2_blocks_out.cu:176 — `return F2BlockTensor{};` after a `switch` whose `OutputTier` enumerators are all handled and each returns. The comment already flags it "unreachable". Kept as a defensive fall-through for the non-exhaustive-switch warning; harmless but is genuinely dead. Suggested: leave as-is (it is the standard -Wreturn-type guard), or replace with an `__builtin_unreachable()`/`std::unreachable()` to make intent explicit and silence any "code never executed" lint.

### G4
- [G4.bounds][LOW] src/device/cuda/f2_blocks_out.cu:77,97,117 — `read_block_to_host(int b, ...)` computes `off = slab * static_cast<std::size_t>(b)` and indexes `f2_dev + off` / `host.f2.data() + off` / a disk offset for an arbitrary caller-supplied `b` with NO check that `0 <= b < n_block`. The widening is correct (no int overflow at scale), but a negative `b` casts to a huge `std::size_t` and a too-large `b` reads past the slab — an OOB D2H / host read / file read. The header documents this as the fit's "tile reader" over a borrowed view (caller-trusted), so this is by-contract, not a live bug. Suggested: if the contract is to stay trust-based, leave it; otherwise add a debug-only `assert(b >= 0 && b < n_block)` at the entry to catch a bad tile index in tests.

### G8
- [G8.comment][LOW] src/device/cuda/f2_blocks_out.cu:108 — the pin-window comment cites a sibling line range "device_f2_blocks.cu:55-56" and the header comment at device_f2_blocks.hpp:69 cites "p2p_combine.cu:185-186"; such hard line-number cross-references are a known staleness trap (they silently rot when the cited file shifts). Not wrong today. Suggested: cite the function/symbol name rather than a line range, or accept the maintenance cost.

## Notes on N/A groups
G2 (no deprecated/removed APIs; only `cudaGetDevice`/`cudaSetDevice`/`cudaMemcpy`, all current). G5/G6/G7/G9/G10 clean (constants single-homed via `slab_elems`/`f2_disk_format.hpp`, no magic literals, no dup blocks, RAII fields default-init at decl). G11 clean (read-only kernel-ptr `__restrict__` N/A — no `__global__` kernels in this TU; the host helpers take `const F2DiskHeader&` / `const char*` correctly). G12/G18/G20/G21/G22 N/A (no device kernels — host-side `cudaMemcpy` only; the two copies are synchronous default-stream blocking, so the `RegisteredHostRegion` pin lifetime spanning both copies is correct). G13 clean (every `cuda*` call routed through STEPPE_CUDA_CHECK; the dtor-restore correctly uses the non-throwing STEPPE_CUDA_WARN with its `[[nodiscard]]` deliberately `(void)`-discarded). G14/G15 clean (no malloc/free here; the only D2H is the intentional single resident read-back, batched per-block by design, pinned). G16/G17 clean (FileCloser deleter gives DiskF2Blocks move-only + null-on-move + non-throwing freeing dtor; the DeviceGuard restores the prior device via the non-throwing warn so the dtor cannot throw).

No HIGH/MED issues found (groups checked: G2-G10, G11-G22).
