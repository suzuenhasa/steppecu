# Review findings — src__device__shard_plan

Files: /home/suzunik/steppe/src/device/shard_plan.cpp, /home/suzunik/steppe/src/device/shard_plan.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Verified clean against the 4.1-4.7 checklist (with the FP64 + SCALE context):
- 4.1 float/double: no float/double arithmetic in this unit at all (pure integer index/SNP-count planning). N/A.
- 4.2 index width: SNP columns are carried as `long` end-to-end — DeviceShard::s0/s1 are `long` (shard_plan.hpp:41-42), sourced from ranges[].begin/.end which are `long` (block_partition_rule.hpp:152-153). DeviceShard::b0/b1 are `int` (shard_plan.hpp:39-40) but hold BLOCK ids, count O(1e3), so the static_cast<int>(...) at shard_plan.cpp:89-90,102-104 cannot overflow.
- 4.3 allocation sizing: no cudaMalloc/new; the only allocation is `std::vector<DeviceShard> plan(G)` (shard_plan.cpp:37) — element-count ctor, correct.
- 4.4 unsigned countdown: the two loops (shard_plan.cpp:53, 78) both count UP with std::size_t. No `for(unsigned i=n-1; i>=0; --i)` pattern.
- 4.5 signed/unsigned compares: loop bounds are size_t vs size_t throughout — `b < n_block`, `g + 1 < G`, `b + 1 < n_block` (shard_plan.cpp:53,78,84,86). No mixed-sign compare.
- 4.6 int overflow before widening: total_snps and device_snps are `long`; cdiv(total_snps, G_signed) deliberately routes through the cdiv(long,long) overload (shard_plan.cpp:63-65, launch_config.hpp:89-91), so (n + b - 1) is computed in long. total ≈ M (~6e5), no overflow. No int intermediate sits in a long-index expression.
- 4.7 host/device pointer typing: CUDA-free by contract, no raw pointers. N/A.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Verified clean against the 2.1-2.4 checklist. This unit is host-pure and CUDA-FREE by contract (shard_plan.cpp:5-6 includes only <cstddef>/<stdexcept>/<vector> + two core headers; shard_plan.hpp:22-26 includes only <cstddef>/<span>/<vector> + core/block_partition_rule.hpp). No <cuda_runtime.h>, no .cu code, no device entry points.
- 2.1 Dropped archs (Maxwell/Pascal/Volta, min sm_75): no CMakeLists, no nvcc -arch / -gencode / sm_* / compute_* flags or CUDA_ARCHITECTURES lists in either file. N/A.
- 2.2 Texture/surface REFERENCES removed in CUDA 12: no texture<...>, surface<...>, cudaBindTexture*, or tex1Dfetch in the unit. N/A.
- 2.3 Non-_sync warp intrinsics: no __shfl*/__ballot/__any/__all/__activemask (no device code at all). N/A.
- 2.4 cudaThreadSynchronize -> cudaDeviceSynchronize: no CUDA runtime calls whatsoever (the only synchronization-adjacent concept is the deterministic single-pass greedy loop, pure host code). N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Verified clean against the 3.1-3.4 checklist:
- 3.1 Commented-out code blocks: all comments in both files are explanatory documentation/rationale prose (header docs shard_plan.cpp:1-9 / shard_plan.hpp:1-18, inline rationale e.g. cpp:24-35,46-66,67-73,99-111). NONE is commented-out source kept "just in case". N/A.
- 3.2 Unreachable code: the two early returns (cpp:42-44 n_block==0, plus the G==0 throw cpp:27-31) guard degenerate cases; the non-degenerate path falls through normally. No code follows any return/break/throw unreachably; no #if 0. The final `return plan` (cpp:112) is the function tail. N/A.
- 3.3 Unused symbols: every include is used — cpp: <cstddef> (std::size_t), <stdexcept> (std::runtime_error cpp:28), <vector> (std::vector cpp:37), block_partition_rule.hpp (core::BlockRange in signature/ranges[b]), launch_config.hpp (core::cdiv cpp:65); hpp: <cstddef> (std::size_t), <span> (std::span:99), <vector> (std::vector:98), block_partition_rule.hpp (BlockRange:99). Both params (ranges, G) used. All locals read: n_block, plan, total_snps, G_signed, target_per_device, g, b0, device_snps, b1, more_devices_left, reached_target. No unused symbol.
- 3.4 Computed but unread: every assignment is subsequently read. device_snps reset to 0 (cpp:95) is read in the next iteration (cpp:79,85); all DeviceShard fields (b0/b1/s0/s1) are consumed by the orchestrator/parity test. No dead store.
-->

