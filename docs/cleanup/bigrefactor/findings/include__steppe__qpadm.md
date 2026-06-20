# Review findings — include__steppe__qpadm

Files: /home/suzunik/steppe/include/steppe/qpadm.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Rationale (read-only confirmation, not findings):
- 4.1: Every floating field is `double` (fudge 1e-4 L60; rank_alpha/p thresholds
  L81,92; weight/se/z/p/chisq L125-132,137,147; QpWaveResult L216). Intentional
  per the FP64-by-design parity law — correct, not narrowing. N/A.
- 4.2: Population indices are `int` (QpAdmModel.target/left/right/model_index
  L109,113,116,118; std::span<const int> left/right L226-228,234-235). These are
  per-population indices in 0..P-1 (P<=~2500), NOT global offsets into the
  P*P*n_block f2 tensor (that offset is computed in the .cpp). `int` is correct.
- 4.3: No allocations — header of CUDA-free value types only. N/A.
- 4.4/4.5: No loops. N/A.
- 4.6: No index arithmetic in this header. N/A.
- 4.7: No raw pointers; device::DeviceF2Blocks / device::Resources are passed by
  ref via CUDA-free fwd-decls (L37-38,169,177,225,232). Host/device seam is clean.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Rationale (read-only confirmation, not findings):
- Unit is a PUBLIC, deliberately CUDA-FREE, standard-C++20 header (stated L11-15);
  no CUDA toolkit symbols by design (architecture.md §4 layering rule).
- 2.1 Dropped archs: header carries no CMake arch lists / sm_XX build flags. N/A.
- 2.2 Texture/surface references: no texture<...>, cudaBindTexture*, or surface
  refs anywhere (device handles are CUDA-free fwd-decls L36-39). N/A.
- 2.3 Non-_sync warp intrinsics: no warp intrinsics (__shfl/__ballot/__any/__all);
  host-only value types + free-function decls. N/A.
- 2.4 cudaThreadSynchronize: no CUDA runtime API calls at all. N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Rationale (read-only confirmation, not findings):
- 3.1 Commented-out blocks: none. All comments are explanatory documentation —
  the header banner (L1-22), field/enum doc-comments (e.g. L41-51, L56-99,
  L121-162), and entry-point docs (L164-236). No code is commented out.
- 3.2 Unreachable code: no `#if 0`; the only preprocessor directives are the
  include guard (L23-24 `#ifndef`/`#define`, L240 `#endif`). Declarations only —
  no function bodies, so no code after return/break. N/A.
- 3.3 Unused symbols: this is a PUBLIC API header — all enums/structs/free-function
  decls are the intended exported surface (JackknifePolicy L47, QpAdmOptions L56,
  QpAdmModel L106, QpAdmResult L124, run_qpadm L169/177, run_qpadm_search L191/200,
  QpWaveResult L211, run_qpwave L225/232). Every include is used: <span> (L191/200/
  226-228/234-235), <string> (std::string L149), <vector> (pervasive), config.hpp
  (Precision::Kind L159/220), error.hpp (Status L156/219), fstats.hpp (F2BlockTensor
  L177/200/232). No unused params (declarations carry no bodies). N/A.
- 3.4 Computed but unread: no statements in a header — only member default
  initializers (the intentional API defaults, e.g. L60/64/68/88/92/99/L118-161).
  Nothing is computed-then-discarded. N/A.
-->

