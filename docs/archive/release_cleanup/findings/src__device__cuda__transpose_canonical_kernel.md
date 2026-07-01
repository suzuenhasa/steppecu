# src__device__cuda__transpose_canonical_kernel
Files: /home/suzunik/steppe/src/device/cuda/transpose_canonical_kernel.cu, /home/suzunik/steppe/src/device/cuda/transpose_canonical_kernel.cuh
Subsystem: device-cuda

## Findings

### G12
- [G12.launch][LOW] transpose_canonical_kernel.cu:77-120 — `transpose_to_canonical_kernel` is a strict one-output-byte-per-thread kernel with no grid-stride loop; coverage relies entirely on the launch wrapper sizing `grid_x = cdiv(total, kTransposeBlock)` and the `STEPPE_ASSERT(grid_x <= kMaxGridX)` at .cu:135-138. That assert compiles out under NDEBUG (it routes through `STEPPE_ASSERT`/`STEPPE_DEBUG_ONLY`), so in a Release build a `total` whose `grid_x` exceeds `kMaxGridX` (2^31−1) would silently truncate at the `static_cast<unsigned>(grid_x)` on .cu:139 and under-cover the output, leaving uninitialized output bytes rather than failing. At documented scale (P~2500, full-M tile out_bytes ~1.46e5 ⇒ total ~3.65e8, grid_x ~1.4M) this is comfortably under the cap, and the contract documents tiling the output-byte axis (.cu:137-138, .cuh:on the wrapper) — so this is a latent robustness gap, not a present bug. Suggested: optional — add a grid-stride loop over `tid` (or a Release-surviving over-`kMaxGridX` guard) so coverage is correct even if the wrapper's debug-only assert is compiled out.
