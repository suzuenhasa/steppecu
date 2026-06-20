# Review findings — src__core__qpadm__qpadm_fit

Files: /home/suzunik/steppe/src/core/qpadm/qpadm_fit.cpp, /home/suzunik/steppe/src/core/qpadm/qpadm_fit.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

Notes (why clean, not padding):
- 4.1: All arithmetic is intentional FP64 (pchisq_upper, p/chisq, weight feasibility `v < 0.0 || v > 1.0` at .cpp:101, literals `0.0`/`1.0`). FP64-by-design; no wrong narrowing — N/A.
- 4.2/4.6: This file is the host-pure orchestrator; it issues NO global indexing into the f2 tensor or genotype matrix (delegated to assemble_f4/jackknife_cov/gls_weights/etc). The only indexing is `res.rank_p[(size_t)r]` (.cpp:130), where `r` is a small rank (<= nl-1, nl bounded by P). `res.dof = (nl - r) * (nr - r)` (.cpp:43) is at most ~P*P (~6.25M) in int32 — no overflow.
- 4.3: No cudaMalloc/new/DeviceBuffer; only std::vector reserve/assign/push_back (element-count APIs, correct).
- 4.4/4.5: Only loop is the range-for `for (const double v : w)` (.cpp:98) — no countdown. `rank_p.assign((size_t)r + 1, ...)` (.cpp:128) and the compare `(size_t)r < res.rank_p.size()` (.cpp:129) are guarded by `r >= 0` (.cpp:129); both sides size_t — no signed/unsigned underflow.
- 4.7: No raw pointers; types are ComputeBackend&, F4Blocks, std::span<const int>, std::vector, and the DeviceF2Blocks vs F2BlockTensor overloads carry the host/device space distinction through the type system already.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (why clean, not padding):
- This unit is the host-pure, CUDA-FREE orchestrator (qpadm_fit.hpp:4-6 "HOST-PURE, CUDA-FREE"). All GPU work (GEMM/SVD/Cholesky) is delegated through the CUDA-free ComputeBackend seam; the includes are only <cmath>/<cstddef>/<span>/<stdexcept>/<vector> and core/device headers (.cpp:5-22). No <cuda_runtime.h>, cuBLAS, or cuSOLVER includes.
- 2.1: No sm_/compute_ arch flags or CMake arch lists in this source unit (build flags live in CMake, not here) — N/A.
- 2.2: No texture<...>/surface<...> references, no cudaBindTexture*/tex1D/2D/3D — N/A (grep clean).
- 2.3: No warp intrinsics at all (no __shfl/__ballot/__any/__all/__activemask) — host code, N/A.
- 2.4: No cudaThreadSynchronize (or any cuda*Synchronize) — host code, N/A.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

Notes (why clean, not padding):
- 3.1: No commented-out code blocks "kept just in case". Every comment is explanatory prose (the precision policy at .cpp:45-67, the M(fit-2) build-order rationale at .cpp:132-147, the carve-out notes at .cpp:222-224, .cpp:236-238). The `(void)precision` text at .cpp:222 is a *citation* of cuda_backend.cu's carve-out, not commented-out code in this file.
- 3.2: No `#if 0` / `#ifdef`-disabled regions. No code after `return`/`break`. The three early `return res` (.cpp:74, .cpp:81) and the final `return res` (.cpp:178) all terminate their paths cleanly; the `case JackknifePolicy::All: default:` (.cpp:114-117) deliberately share one `break`. The `try { ... } catch (const std::runtime_error&)` (.cpp:148-175) catch body is intentionally empty (documented non-breaking absence of a backend rank_sweep override), not dead code.
- 3.3: All includes are used — <cmath> (std::isnan .cpp:99), <cstddef> (std::size_t .cpp:128-130), <span> (throughout), <stdexcept> (std::runtime_error .cpp:172), <vector> (throughout); the core/device headers each supply a symbol that is referenced (assemble_f4, gls_weights, jackknife_cov, se_from_loo, run_rank_sweep/run_popdrop, ComputeBackend/F4Blocks/JackknifeCov/GlsWeights, DeviceF2Blocks, Resources, Precision/kDefaultMantissaBits, Status, F2BlockTensor). Both anonymous-namespace helpers are used (qpwave_from_sweep .cpp:185 → .cpp:272; run_qpwave_impl .cpp:255 → .cpp:281/288). The hpp decls (pchisq_upper, left_with_target, run_impl) are all used. No unused params (each run_qpwave/run_qpadm overload uses `resources` to select the backend). MINOR doc-comment drift (not a code defect, not flagged): the ranktest.hpp include comment (.cpp:16) lists `run_qpwave_impl`, but run_qpwave_impl is actually defined locally in this file's anonymous namespace; ranktest.hpp only provides run_rank_sweep/run_popdrop.
- 3.4: No computed-but-unread values. Every `res.*` field assigned in run_impl/run_qpwave_impl is read via the returned QpAdmResult/QpWaveResult. Locals nl/nr/r (.cpp:39-41), cov (.cpp:70), gw (.cpp:77), se (.cpp:123), rs (.cpp:149), pd (.cpp:162), the `tag` (.cpp:263), and the weights_feasible lambda are all consumed.

