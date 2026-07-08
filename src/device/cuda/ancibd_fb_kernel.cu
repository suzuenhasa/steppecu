// src/device/cuda/ancibd_fb_kernel.cu
//
// ancIBD pairwise forward-backward GPU kernels (see the .cuh). The FB mirrors the
// host reference ancibd_fb_pair (core/stats/ancibd_fb.hpp) exactly — a 5-state
// scaled scan — but block-per-pair on device: one warp per pair, lanes 0..4 hold
// the five states, warp-shuffle reductions form the per-column normalizer and the
// backward IBD-mass, and the sequential column dependency is confined to the block
// (batching over pairs = blocks is the parallelism). Native FP64 throughout.
//
// Reference: docs/planning/ancibd-face-spec.md §3
#include <cstdint>

#include "device/cuda/ancibd_fb_kernel.cuh"

namespace steppe::device {

namespace {

constexpr int kWarp = 32;

// share(hx,hy,p) = hx*hy/(1-pt) + (1-hx)(1-hy)/pt, pt clamped — matches
// ancibd_model.hpp ancibd_share / emission.py HaplotypeSharingEmissions2.
__device__ inline double share_dev(double hx, double hy, double p, double p_min) {
    double pt = p < p_min ? p_min : (p > 1.0 - p_min ? 1.0 - p_min : p);
    return hx * hy / (1.0 - pt) + (1.0 - hx) * (1.0 - hy) / pt;
}

// Full-warp sum reduction (butterfly); inactive lanes must pass 0.
__device__ inline double warp_sum(double v) {
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return __shfl_sync(0xffffffffu, v, 0);  // broadcast lane-0 total
}

// -------- (1) hts derivation: LoadH5Multi2.get_haplo_prob ----------------------
__global__ void ancibd_derive_hts_kernel(const double* __restrict__ gp3,
                                         const std::uint8_t* __restrict__ phased2,
                                         double* __restrict__ hts, int n_sample, long M,
                                         double min_error) {
    const long total = M * static_cast<long>(n_sample);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x) {
        const long site = idx / n_sample;
        const int sample = static_cast<int>(idx % n_sample);
        const std::size_t gb = (static_cast<std::size_t>(site) * n_sample + sample) * 3;
        const std::size_t pb = (static_cast<std::size_t>(site) * n_sample + sample) * 2;
        const double gp0 = gp3[gb + 0];
        const double gp1 = gp3[gb + 1];
        const unsigned gt0 = phased2[pb + 0];
        const unsigned gt1 = phased2[pb + 1];
        double g01, g10;
        if (gt0 == 0u && gt1 == 1u) {
            g01 = gp1;
            g10 = 0.0;
        } else if (gt0 == 1u && gt1 == 0u) {
            g01 = 0.0;
            g10 = gp1;
        } else {
            g01 = gp1 * 0.5;
            g10 = gp1 * 0.5;
        }
        double a = gp0 + g01;
        double b = gp0 + g10;
        a = a < min_error ? min_error : (a > 1.0 - min_error ? 1.0 - min_error : a);
        b = b < min_error ? min_error : (b > 1.0 - min_error ? 1.0 - min_error : b);
        hts[(static_cast<std::size_t>(2 * sample + 0)) * M + site] = a;
        hts[(static_cast<std::size_t>(2 * sample + 1)) * M + site] = b;
    }
}

// -------- (2) 5-state scaled forward-backward, one warp per pair ---------------
__global__ void ancibd_fb_kernel(const double* __restrict__ hts, const double* __restrict__ p,
                                 const double* __restrict__ T, const int* __restrict__ pair_idx,
                                 long M, double in_val, double p_min, double* __restrict__ fwd0,
                                 double* __restrict__ cnorm, double* __restrict__ pibd) {
    const int pair = blockIdx.x;
    const int lane = threadIdx.x;                 // 0..31; states use lanes 0..4
    const bool active = lane < 5;

    const int i1 = pair_idx[2 * pair + 0];
    const int i2 = pair_idx[2 * pair + 1];
    // The pair's four haplotype rows (hap-row-major, row*M + site).
    const double* rowAa = hts + (static_cast<std::size_t>(2 * i1 + 0)) * M;
    const double* rowAb = hts + (static_cast<std::size_t>(2 * i1 + 1)) * M;
    const double* rowBa = hts + (static_cast<std::size_t>(2 * i2 + 0)) * M;
    const double* rowBb = hts + (static_cast<std::size_t>(2 * i2 + 1)) * M;

    double* my_fwd0 = fwd0 + static_cast<std::size_t>(pair) * M;
    double* my_c = cnorm + static_cast<std::size_t>(pair) * M;
    double* my_pibd = pibd + static_cast<std::size_t>(pair) * M;

    // Per-lane emission at a column: lane 0 -> 1; lanes 1..4 -> the four share configs.
    auto emit = [&](long l) -> double {
        if (lane == 0) return 1.0;
        const double pp = p[l];
        const double hAa = rowAa[l], hAb = rowAb[l], hBa = rowBa[l], hBb = rowBb[l];
        if (lane == 1) return share_dev(hAa, hBa, pp, p_min);
        if (lane == 2) return share_dev(hAa, hBb, pp, p_min);
        if (lane == 3) return share_dev(hAb, hBa, pp, p_min);
        if (lane == 4) return share_dev(hAb, hBb, pp, p_min);
        return 0.0;  // inactive lane
    };

    // ---- forward ----  fwd_j carried lane-local; only fwd0 + c stored resident.
    double fj = active ? (lane == 0 ? 1.0 - 4.0 * in_val : in_val) : 0.0;
    if (lane == 0) {
        my_fwd0[0] = fj;
        my_c[0] = 1.0;
    }
    for (long i = 1; i < M; ++i) {
        const double t00 = T[i * 9 + 0], t01 = T[i * 9 + 1];
        const double t10 = T[i * 9 + 3], t11 = T[i * 9 + 4], t12 = T[i * 9 + 5];
        const double stay = t11 - t12;
        const double fwd0_prev = __shfl_sync(0xffffffffu, fj, 0);  // lane-0 fwd[0]
        const double fl = 1.0 - fwd0_prev;
        const double e = emit(i);
        double temp;
        if (lane == 0)
            temp = e * (fwd0_prev * t00 + fl * t10);
        else if (active)
            temp = e * (fwd0_prev * t01 + fl * t12 + fj * stay);
        else
            temp = 0.0;
        const double ci = warp_sum(temp);
        fj = (ci > 0.0) ? temp / ci : 0.0;
        if (lane == 0) {
            my_fwd0[i] = fj;
            my_c[i] = ci;
        }
    }

    // ---- backward ----  bwd_j carried lane-local, init 1; posterior on the fly.
    double bj = active ? 1.0 : 0.0;
    for (long i = M - 1; i >= 1; --i) {
        if (lane == 0) my_pibd[i] = 1.0 - my_fwd0[i] * bj;  // post0 = fwd0[i]*bwd0[i]
        const double t00 = T[i * 9 + 0], t01 = T[i * 9 + 1];
        const double t10 = T[i * 9 + 3], t11 = T[i * 9 + 4], t12 = T[i * 9 + 5];
        const double stay = t11 - t12;
        const double e = emit(i);
        // fl = sum_{k=1..4} bwd_k * e_k  (lane 0 contributes 0)
        const double fl = warp_sum((lane >= 1 && lane <= 4) ? bj * e : 0.0);
        const double bwd0 = __shfl_sync(0xffffffffu, bj, 0);
        double ntemp;
        if (lane == 0)
            ntemp = bwd0 * t00 * 1.0 + fl * t01;  // e0 == 1
        else if (active)
            ntemp = 1.0 * bwd0 * t10 + fl * t12 + e * bj * stay;
        else
            ntemp = 0.0;
        const double ci = my_c[i];
        bj = (active && ci > 0.0) ? ntemp / ci : 0.0;  // now column i-1
    }
    if (lane == 0) my_pibd[0] = 1.0 - my_fwd0[0] * bj;
}

}  // namespace

void launch_ancibd_derive_hts(const double* d_gp3, const std::uint8_t* d_phased2, double* d_hts,
                              int n_sample, long M, double min_error, cudaStream_t stream) {
    if (n_sample <= 0 || M <= 0) return;
    const long total = M * static_cast<long>(n_sample);
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    ancibd_derive_hts_kernel<<<grid, block, 0, stream>>>(d_gp3, d_phased2, d_hts, n_sample, M,
                                                         min_error);
}

void launch_ancibd_fb(const double* d_hts, const double* d_p, const double* d_T,
                      const int* d_pair_idx, int n_sample, long M, int n_pair, double in_val,
                      double p_min, double* d_fwd0, double* d_c, double* d_pibd,
                      cudaStream_t stream) {
    (void)n_sample;
    if (n_pair <= 0 || M <= 0) return;
    // One block (one warp) per pair.
    ancibd_fb_kernel<<<n_pair, kWarp, 0, stream>>>(d_hts, d_p, d_T, d_pair_idx, M, in_val, p_min,
                                                   d_fwd0, d_c, d_pibd);
}

}  // namespace steppe::device
