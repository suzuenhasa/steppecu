# Review findings — src__core__internal__launch_config

Files: /home/suzunik/steppe/src/core/internal/launch_config.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Reviewed against all 4.1–4.7 tasks with the FP64 + SCALE context:
- 4.1 (float/double): N/A — the file is pure integer arithmetic + named integer
  constants; no FP literals or math.
- 4.2 / 4.6 (index width / overflow before widening): The header explicitly
  handles the SNP/M-scale axis. A `cdiv(long,long)` overload (lines 89–91) exists
  for the M axis, and the int `cdiv` (82–84) / `grid_for` (114–127) are documented
  as P-axis (≤2500) only. Verified at the call sites: f2_block_kernel.cu:320 and
  decode_af_kernel.cu:106 cast M to `long` before `cdiv`, so `n + b - 1` is a long
  add (no int overflow); `grid_for` is only ever passed P or s_pad (P-scale,
  fits int). The (n + b - 1) add in the int cdiv does not overflow for any
  documented/observed input (P, s_pad, kCdivBlock all small). Contract is correct
  and honored.
- 4.3 (allocation sizing): N/A — no cudaMalloc / DeviceBuffer / new in this file.
- 4.4 (unsigned countdown): N/A — no loops.
- 4.5 (signed/unsigned compares): The two asserts (122, 144) guard the value
  (`extent >= 0`; `n_in_group >= 1`) BEFORE the `static_cast<unsigned>` compare
  against the unsigned cap — correct, no signed/unsigned mismatch.
- 4.7 (host/device pointer typing): N/A — no pointers; all values are scalars
  returned by value.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Reviewed against all 2.1–2.4 tasks. The file is a pure STEPPE_HD constexpr
integer-math + named-constant header (CUDA-free-compilable per its own banner,
lines 36–40); it issues no CUDA runtime calls, declares no kernels, and pulls in
only host_device.hpp + config.hpp.
- 2.1 (dropped Maxwell/Pascal/Volta archs, sm_50/60/70 build flags, CMake arch
  lists): N/A — no build flags / CMake / arch lists in a .hpp. The grid-limit
  constants kMaxGridX/Y/Z (lines 65/68/73) and the §7 banner (lines 49–60) are
  documented as verified for CUDA 13.x / Blackwell sm_120 — current, not for a
  dropped arch.
- 2.2 (texture/surface references, texture<...>/cudaBindTexture*): N/A — no
  texture or surface usage anywhere in the file.
- 2.3 (non-_sync warp intrinsics, e.g. __shfl/__ballot/__any/__all): N/A — no
  warp intrinsics; the file is grid/block extent math only.
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): N/A — no CUDA runtime
  API calls of any kind.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Reviewed against all 3.1–3.4 tasks. The file is a pure STEPPE_HD constexpr
integer-math + named-constant header; every symbol it declares is exported and
consumed elsewhere (verified by grep across src/ tests/ include/).
- 3.1 (commented-out blocks kept "just in case"): N/A — every comment block
  (banner lines 1–40, §-divider headers, the [[maybe_unused]] rationale at
  116–120) is live documentation/rationale, not stashed code. No `// foo();`
  style dead statements anywhere.
- 3.2 (unreachable code; #if 0; code after return/break): N/A — no #if 0, no
  preprocessor dead branches; the four functions are single-return constexpr/
  inline bodies with no code after their `return`. The only #if is the standard
  include guard (41–42, 171).
- 3.3 (unused symbols — vars/params/includes/helpers):
  * Both includes are used: host_device.hpp provides STEPPE_HD/STEPPE_ASSERT
    (lines 82, 89, 114, 122, 143, 144), config.hpp provides kCdivBlock (the
    default arg at line 114).
  * All exported symbols have real call sites: cdiv(int/long) — shard_plan.cpp:65,
    decode_af_kernel.cu:106, f2_block_kernel.cu:320; grid_for — f2_block_kernel.cu
    :321/380/381, f2_blocks_kernel.cu:197/198, decode_af_kernel.cu:107; grid_z_extent
    — f2_blocks_kernel.cu:199; kMaxGridZ — vram_budget.hpp:152, cuda_backend.cu:828;
    kDecodeBlockX/Y — decode_af_kernel.cu:45/46/105/106. kMaxGridX/kMaxGridY are
    exercised by tests/unit/test_launch_config.cpp (static_asserts 85–87, runtime
    81–139) and kMaxGridY is the load-bearing default of grid_for's max_grid param
    (line 115) — not dead.
  * The `max_grid` param of grid_for (line 115) is read only inside the assert
    (line 122), which compiles out under NDEBUG; this is correctly annotated
    `[[maybe_unused]]` (line 115) with an explicit rationale — intentional, not
    an unused-param defect.
- 3.4 (computed but unread): N/A — `extent` (line 121) is both asserted (122) and
  returned (126); no other locals. All four functions return their computed value.
-->

