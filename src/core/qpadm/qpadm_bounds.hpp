// src/core/qpadm/qpadm_bounds.hpp
//
// The SINGLE SOURCE OF TRUTH for the qpAdm SMALL-path bit-parity envelope
// (architecture.md §2 DRY, §4 layering, §13). These named constants size the
// per-thread LOCAL-MEMORY arrays of the model-batched small-path fit kernels
// (qpadm_fit_kernels.cu) AND gate which models are routed to that path, both on
// the host core partition (model_search.cpp `model_in_small_path`) and the device
// backend (cuda_backend.cu `model_fits_small_path`).
//
// WHY ONE HOME (the drift-is-a-correctness-bug rule): the host gate and the kernel
// array bounds MUST agree exactly. A host gate WIDER than the kernel bounds admits
// an oversized model whose per-thread arrays then index past their fixed size — a
// DEVICE BUFFER OVERFLOW / UB. Defining the envelope here ONCE, and having all three
// sites reference it, makes that class of drift impossible: widening the gate without
// widening the kernel arrays is now a single edit that moves both together.
//
// CUDA-FREE (architecture.md §4): these are plain `constexpr int`, no CUDA header,
// so this leaf can be included by the CUDA-free core (model_search.cpp) AND by the
// device TUs (cuda_backend.cu, qpadm_fit_kernels.cu) without dragging CUDA into core.
// `constexpr int` is usable in C++ array-bound and template-non-type contexts, which
// is exactly how the kernel TU consumes them (e.g. `double xmat[kQpMaxM]`,
// `dev_als_weights<kQpMaxNl, kQpMaxNr, kQpMaxR>(...)`).
//
// WHY THESE VALUES (the per-thread local-memory budget). The model-batched small
// path runs ONE thread per model with the whole per-model fit working set in
// per-thread LOCAL memory (the dominant scratch is Wm[m*t] and coeffs[t*t]). CUDA
// reserves a kernel's per-thread local frame for the device's MAX resident-thread
// count, so an over-large fixed array trips cudaErrorMemoryAllocation at launch even
// single-threaded (MEASURED on box5090: a big-nr frame OOMs). The envelope below
// keeps the frame modest (m<=50, t<=40 ⇒ Wm<=50*40=2000 doubles=16 KB, coeffs<=
// 40*40=1600 doubles=12.5 KB). A model EXCEEDING the envelope is correct math, it
// just runs on the LARGE path instead (the cuSOLVER SVD + VRAM-scratch *_large
// kernels in cuda_backend.cu) — the host gate routes it there. The 9-pop golden
// (nl=2, nr=5, r=1 ⇒ m=10, t<=5) is far inside; NRBIG (nr=39) is outside ⇒ large path.

#ifndef STEPPE_CORE_QPADM_QPADM_BOUNDS_HPP
#define STEPPE_CORE_QPADM_QPADM_BOUNDS_HPP

namespace steppe::core::qpadm {

/// Max left sources (nl) the small-path kernels' per-thread arrays are sized for.
inline constexpr int kQpMaxNl = 5;
/// Max right outgroups (nr) the small-path kernels' per-thread arrays are sized for.
inline constexpr int kQpMaxNr = 10;
/// Max fit rank (r) the small-path kernels' per-thread arrays are sized for.
inline constexpr int kQpMaxR = 4;
/// Derived: max f4 matrix length m = nl*nr (the xmat / Qinv row count).
inline constexpr int kQpMaxM = kQpMaxNl * kQpMaxNr;  // 50
/// Derived: max ALS coeff dimension t = max(nl,nr)*r.
inline constexpr int kQpMaxT = (kQpMaxNl > kQpMaxNr ? kQpMaxNl : kQpMaxNr) * kQpMaxR;  // 40

/// The bit-parity small-path envelope predicate. A model with (nl, nr, r) inside the
/// kQpMax* bounds is fit by the model-batched small-path kernels (the rotation common
/// case + the 9-pop golden); outside ⇒ the cuSOLVER large path. This is the ONE home
/// of the predicate so the host core gate (model_search.cpp), the device backend
/// (cuda_backend.cu), and the kernel array bounds cannot drift apart.
constexpr bool model_fits_small_path(int nl, int nr, int r) {
    return nl <= kQpMaxNl && nr <= kQpMaxNr && r <= kQpMaxR;
}

/// The qpAdm chi-square DEGREES OF FREEDOM for a rank-r fit of an nl×nr f4 matrix:
/// dof(r) = (nl-r)·(nr-r). The SINGLE SOURCE of this formula so the per-rank sweep,
/// the popdrop dof fallback, and the result/rank-drop tables cannot drift from one
/// another (architecture.md §8 single-source; group-5 5.3). Called by the CpuBackend
/// oracle (cpu_backend.cpp), the CudaBackend (cuda_backend.cu rank_sweep /
/// assemble_result / popdrop), and the host ranktest fallback (ranktest.cpp). Plain
/// `constexpr int`, CUDA-free, so the device TUs and the host core share one copy.
constexpr int qpadm_dof(int nl, int nr, int r) {
    return (nl - r) * (nr - r);
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPADM_BOUNDS_HPP
