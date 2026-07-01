# src__core__fstats__f2_partials_validate
Files: /home/suzunik/steppe/src/core/fstats/f2_partials_validate.hpp
Subsystem: core-stats

## Findings
No issues found (groups checked: G2-G10).

<!--
Notes (not findings; rationale for the clean verdict):
- G4 (scale/overflow): the P*P*n_block element count is widened correctly — line 142-143
  `slab = (size_t)P * (size_t)P` and line 173-174 `want_slabs = slab * (size_t)part.n_block`
  are computed in size_t before multiply, matching F2BlockTensor::size() (fstats.hpp:74-78).
  No int*int-before-widen at P~2500. The tiling accumulator `covered` is `long` (line 86) and
  compared against `static_cast<long>(n_block_full)` (line 93); span_blocks = b1-b0 (int block
  ids, max ~757) cannot overflow.
- G6/G8: field accesses verified against real defs — DeviceShard.b0/b1 (shard_plan.hpp:38-47),
  DevicePartial.P/n_block_local/b0/block_sizes (device_partial.hpp:40-47),
  F2BlockTensor.f2/vpair/block_sizes/P/n_block (fstats.hpp:47-71). Comments accurate, not stale.
- G3: no dead code; the detail::validate_partials_scaffold template + two public overloads are
  all reachable (called by the host and P2P combine TUs per the header doc). Includes all used.
- G9: knobs (P, n_block_full) are params, not buried; no positional-boolean calls.
- G10: `prefix` and `covered` declared at first use; no uninitialized-then-assigned.
- is_cuda=false: G11-G22 not applicable (header is CUDA-free by contract, no kernels).
-->
