# Review findings — src__core__qpadm__ranktest

Files: /home/suzunik/steppe/src/core/qpadm/ranktest.cpp, /home/suzunik/steppe/src/core/qpadm/ranktest.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (not findings — context for why this is clean):
- 4.1: All math is intentional FP64 (weights/chisq/p/dof are double; NaN via std::numeric_limits<double>::quiet_NaN; popdrop_feasible compares against 0.0/1.0). No float temps, no narrowing in a parity-critical path. N/A per FP64 context.
- 4.2/4.6: This orchestrator operates on the ALREADY-REDUCED f4 covariance (m = nl*nr, population-scale), NOT the P*P*n_block f2 tensor or the M*P genotype matrix. The one large index — the Qinv/Q sub-block address at ranktest.cpp:70-72 (src = ind[a] + m_full*ind[b]) — is correctly widened: each operand is cast to std::size_t BEFORE the multiply/add (static_cast<std::size_t>(m_full) * static_cast<std::size_t>(ind[b])). The dst index (73-74) is likewise size_t-widened. The smaller indices (j + nr*ii at lines 49/53-54/62) stay within m_red ≤ m_full and fit int at qpAdm population scale.
- 4.3: No cudaMalloc/new; all allocation is std::vector::assign sized in element count (correct for std::vector). The Qinv/Q assign at lines 66-67 uses static_cast<std::size_t>(m_red)*static_cast<std::size_t>(m_red) — size_t multiply, no overflow before widening.
- 4.4: No unsigned countdown loops. The descending drop loop (148) uses signed int drop with drop >= 0 — terminates correctly.
- 4.5: Loop bounds are int-vs-int (lines 50/52/59/61/68-69) or size_t-vs-size_t (line 120 s < surv.size()); no signed/unsigned mismatch.
- 4.7: Host-pure file (is_cuda=false), CUDA-free per the header contract; no device pointers — 4.7 N/A.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (not findings — context for why this is clean):
- 2.1: No CMake arch lists or build/compile flags in this unit (ranktest.cpp/.hpp are source-only); no sm_50/60/70 references. N/A.
- 2.2: No texture/surface references (no `texture<...>`, no `cudaBindTexture*`, no surface refs) — host-pure file with no CUDA at all. N/A.
- 2.3: No warp intrinsics (no `__shfl*`/`__ballot*`/`__any*`/`__all*`), sync or non-sync. N/A.
- 2.4: No CUDA runtime calls at all (no `cudaThreadSynchronize`/`cudaDeviceSynchronize`). Includes are only <cmath>/<cstddef>/<limits>/<string>/<vector> + project headers; all work is host index arithmetic + std::vector, fit routed through the ComputeBackend seam. N/A.
-->

## Group 3 — Dead / commented-out code

- [3.4][LOW] ranktest.cpp:44 — `(void)m_full;` is a dead no-op suppression: `m_full` (declared line 42) IS read at line 71 (`static_cast<std::size_t>(m_full) * ...`), so the unused-variable cast is leftover and misleading (likely from a prior version where m_full was unused). Suggested: delete line 44.

<!--
Notes (not findings — context for why the rest is clean):
- 3.1: No commented-out CODE kept "just in case". Every comment block (file header 1-8, reduce_rows doc 34-37/57, popdrop_one doc 79-81, the AT2 rank-fit rationale 94-105, the inline weight/full-row notes 115/137/142-146) is explanatory contract/design documentation, not disabled source.
- 3.2: No unreachable code. No `#if 0`. No statements after a return/break; the `continue` at line 25 and the early `return false` at line 27 are normal control flow within loops.
- 3.3: No unused symbols. All includes are used: <cmath> (std::isnan L25), <cstddef> (std::size_t), <limits> (std::numeric_limits L117), <string> (row.pat is std::string, L86-87), <vector>, steppe/error.hpp (Status::Ok L119). All function params are used (be/x/cov/opts/precision threaded through; nl_full/drop/surv consumed). The anonymous-namespace helpers reduce_rows + popdrop_one are both called (L91, L140/L152).
- 3.4: Only the L44 (void)m_full above. Every other local is read: nr, nl_red, m_red, ind, r_fit, ri, rs, gw, row fields all consumed downstream.
-->

