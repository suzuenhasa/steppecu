// src/device/cuda/sweep_unrank.cuh
//
// Reference: docs/reference/src_device_cuda_sweep_unrank.cuh.md
//
// The combinatorial-number-system unrank shared by the f-stat sweep and READv2.
// `sweep_unrank(r, P, k, c)` maps a flat rank r in [0, C(P,k)) to the k-subset
// c[0] < c[1] < ... < c[k-1] of {0..P-1}; `sweep_choose` is the C(n,kk) it walks.
// Extracted verbatim from qpadm_fit_kernels.cu so the qpAdm sweep and the READv2
// all-pairs mismatch kernel share one definition rather than a divergent copy.
//
// `static __device__ __forceinline__` gives each including CUDA TU its own internal-
// linkage copy — matching the original anonymous-namespace linkage — so this header
// is ODR-safe under separable compilation (CUDA_SEPARABLE_COMPILATION ON).
#ifndef STEPPE_DEVICE_CUDA_SWEEP_UNRANK_CUH
#define STEPPE_DEVICE_CUDA_SWEEP_UNRANK_CUH

#include <cuda_runtime.h>

namespace steppe::device {

// C(n, kk) in double (the sweep's rank arithmetic; identical to the qpAdm original).
static __device__ __forceinline__ double sweep_choose(int n, int kk) {
    if (kk < 0 || n < kk) return 0.0;
    double num = 1.0;
    for (int i = 0; i < kk; ++i) num *= static_cast<double>(n - i);
    double den = 1.0;
    for (int i = 1; i <= kk; ++i) den *= static_cast<double>(i);
    return num / den;
}

// Combinatorial-number-system unrank: r -> the k-subset c[0]<...<c[k-1] of {0..P-1}.
static __device__ __forceinline__ void sweep_unrank(long long r, int P, int k, int* c) {
    (void)P;
    for (int pos = k - 1; pos >= 0; --pos) {
        const int kk = pos + 1;
        int v = kk - 1;
        while (sweep_choose(v + 1, kk) <= static_cast<double>(r)) ++v;
        c[pos] = v;
        r -= static_cast<long long>(sweep_choose(v, kk));
    }
}

// O(1) closed-form unrank for the k=2 upper-triangular pair space (the READv2 case).
// The generic sweep_unrank does an O(P) linear scan per thread; for the all-pairs
// C(N,2) sweep that is O(N^3) overall. This closed form removes the scan: given the
// flat rank r over pairs (i<j) enumerated as sweep_unrank does — j is the outer
// combinatorial digit, i the residual — it returns i in c[0], j in c[1]. Derivation:
// pairs are grouped by j (j runs 1..N-1); the block for a given j starts at
// C(j,2)=j*(j-1)/2 and holds j pairs (i = 0..j-1). Solve C(j,2) <= r for the largest
// such j via the quadratic j = floor((1 + sqrt(1 + 8r)) / 2), then i = r - C(j,2).
static __device__ __forceinline__ void readv2_unrank_pair(long long r, int N, int* c) {
    (void)N;
    const double rd = static_cast<double>(r);
    // j = floor((1 + sqrt(1 + 8r)) / 2); nudge for FP rounding at block edges.
    int j = static_cast<int>((1.0 + sqrt(1.0 + 8.0 * rd)) * 0.5);
    while (static_cast<long long>(j) * (j - 1) / 2 > r) --j;
    while (static_cast<long long>(j + 1) * j / 2 <= r) ++j;
    const long long base = static_cast<long long>(j) * (j - 1) / 2;
    c[0] = static_cast<int>(r - base);  // i
    c[1] = j;                           // j  (c[0] < c[1])
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_SWEEP_UNRANK_CUH
