// src/device/cuda/dates_kernel.cu
//
// The GPU kernels for DATES admixture dating: the elementwise, scatter, and
// reduce glue around the cuFFT autocorrelation (the transforms themselves are
// issued by the backend). The kernel bodies live only in this TU; the backend
// reaches them through the narrow launch wrappers in dates_kernel.cuh.
//
// Reference: docs/reference/src_device_cuda_dates_kernel.cu.md
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"
#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/dates_kernel.cuh"

namespace steppe::device {

namespace {

inline constexpr int kBlock = 256;

__device__ inline std::uint8_t target_code(const std::uint8_t* __restrict__ packed,
                                           std::size_t bytes_per_record, int sample, long s) {
    const std::uint8_t* rec = packed + static_cast<std::size_t>(sample) * bytes_per_record;
    const std::size_t byte_in_rec =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos = static_cast<int>(s % core::kCodesPerByte);
    return core::genotype_code(rec[byte_in_rec], pos);
}

// Phase A: regression dot products — reference §3
__global__ void regress_dots_kernel(const double* __restrict__ src1,
                                    const double* __restrict__ src2,
                                    const double* __restrict__ valid,
                                    const std::uint8_t* __restrict__ packed,
                                    std::size_t bytes_per_record, int sample, long M,
                                    double* __restrict__ dot12, double* __restrict__ dot22) {
    const long s = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    __shared__ double sh12[kBlock];
    __shared__ double sh22[kBlock];
    double a12 = 0.0, a22 = 0.0;
    if (s < M && valid[s] != 0.0) {
        const std::uint8_t code = target_code(packed, bytes_per_record, sample, s);
        if (code != core::kMissingGenotypeCode) {
            const double w0 = static_cast<double>(code) / 2.0;
            const double w1 = src1[s];
            const double w2 = src2[s];
            const double aa = w0 - w2;
            const double bb = w1 - w2;
            a12 = aa * bb;
            a22 = bb * bb;
        }
    }
    sh12[threadIdx.x] = a12;
    sh22[threadIdx.x] = a22;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sh12[threadIdx.x] += sh12[threadIdx.x + off];
            sh22[threadIdx.x] += sh22[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(dot12, sh12[0]);
        atomicAdd(dot22, sh22[0]);
    }
}

// Phase B: residual scatter onto the fine grid — reference §4
__global__ void scatter_kernel(const double* __restrict__ src1, const double* __restrict__ src2,
                               const double* __restrict__ valid,
                               const std::uint8_t* __restrict__ packed,
                               std::size_t bytes_per_record, int sample, long M,
                               const int* __restrict__ grid_cell, double yreg,
                               double* __restrict__ z0, double* __restrict__ z1,
                               double* __restrict__ z2) {
    const long s = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (s >= M || valid[s] == 0.0) return;
    const std::uint8_t code = target_code(packed, bytes_per_record, sample, s);
    if (code == core::kMissingGenotypeCode) return;
    const double w0 = static_cast<double>(code) / 2.0;
    const double w1 = src1[s];
    const double w2 = src2[s];
    const double pred = yreg * w1 + (1.0 - yreg) * w2;
    const double r = w0 - pred;
    const double wt = w1 - w2;
    const double y = r * wt;
    const int cell = grid_cell[s];
    atomicAdd(&z0[cell], 1.0);
    atomicAdd(&z1[cell], y);
    atomicAdd(&z2[cell], y * y);
}

// FFT glue kernels — reference §5
__global__ void pack_segments_kernel(const double* __restrict__ grid,
                                     const int* __restrict__ chrom_first,
                                     const int* __restrict__ chrom_last, int n_chrom,
                                     int numqbins, int n_fft, double* __restrict__ padded) {
    const long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_fft);
    if (cell >= total) return;
    const int kc = static_cast<int>(cell / n_fft);
    const int j = static_cast<int>(cell % n_fft);
    const int slo = chrom_first[kc];
    const int shi = chrom_last[kc];
    double val = 0.0;
    if (slo >= 0 && shi >= slo) {
        const int len = shi - slo + 1;
        if (j < len) {
            const int g = slo + j;
            if (g >= 0 && g < numqbins) val = grid[g];
        }
    }
    padded[cell] = val;
}

__global__ void power_spectrum_kernel(const double2* __restrict__ freq, int n_cplx, int n_chrom,
                                      double2* __restrict__ power) {
    const long k = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    if (k >= total) return;
    const double2 a = freq[k];
    power[k] = make_double2(a.x * a.x + a.y * a.y, 0.0);
}

__global__ void cross_power_kernel(const double2* __restrict__ a, const double2* __restrict__ b,
                                   int n_cplx, int n_chrom, double2* __restrict__ out) {
    const long k = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    if (k >= total) return;
    const double2 av = a[k];
    const double2 bv = b[k];
    out[k] = make_double2(av.x * bv.x + av.y * bv.y, av.x * bv.y - av.y * bv.x);
}

__global__ void extract_lags_kernel(const double* __restrict__ inv, int n_fft, int n_chrom,
                                    int diffmax, double* __restrict__ dd) {
    const long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long dm = static_cast<long>(diffmax) + 1;
    const long total = static_cast<long>(n_chrom) * dm;
    if (cell >= total) return;
    const int kc = static_cast<int>(cell / dm);
    const int lag = static_cast<int>(cell % dm);
    const double v = (lag < n_fft) ? inv[static_cast<long>(kc) * n_fft + lag] : 0.0;
    dd[cell] += v / static_cast<double>(n_fft);
}

// Re-bin lags into the correlation statistics — reference §6
__global__ void accumulate_bins_kernel(const double* __restrict__ dd00,
                                       const double* __restrict__ dd11,
                                       const double* __restrict__ dd02,
                                       const double* __restrict__ dd20, int n_chrom, int diffmax,
                                       int n_bin, int qbin, double* __restrict__ s0,
                                       double* __restrict__ s11, double* __restrict__ s12,
                                       double* __restrict__ s22) {
    const long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long dm = static_cast<long>(diffmax) + 1;
    const long total = static_cast<long>(n_chrom) * dm;
    if (cell >= total) return;
    const int kc = static_cast<int>(cell / dm);
    const int d = static_cast<int>(cell % dm);
    if (d < 1) return;
    const double c00 = dd00[cell];
    if (c00 < 0.5) return;
    const int s = d / qbin;
    if (s < 0 || s >= n_bin) return;
    const long o = static_cast<long>(kc) * n_bin + s;
    atomicAdd(&s0[o], c00);
    atomicAdd(&s12[o], dd11[cell]);
    atomicAdd(&s11[o], dd02[cell]);
    atomicAdd(&s22[o], dd20[cell]);
}

// Target-genotype repack — reference §7
__global__ void repack_target_kernel(const std::uint8_t* __restrict__ src, std::size_t src_bpr,
                                     const long* __restrict__ kept_src, long M_kept, int n_target,
                                     std::size_t dst_bpr, std::uint8_t* __restrict__ dst) {
    const long dst_bytes = static_cast<long>(dst_bpr);
    const long total = static_cast<long>(n_target) * dst_bytes;
    for (long gid = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(gid / dst_bytes);
        const long db = gid % dst_bytes;
        const std::uint8_t* src_rec = src + static_cast<std::size_t>(i) * src_bpr;
        std::uint8_t out = 0;
        for (int dp = 0; dp < core::kCodesPerByte; ++dp) {
            const long ks = db * core::kCodesPerByte + dp;
            if (ks >= M_kept) break;
            const long s = kept_src[ks];
            const std::size_t sb =
                static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
            const int sp = static_cast<int>(s % core::kCodesPerByte);
            const std::uint8_t code = core::genotype_code(src_rec[sb], sp);
            const int shift = (core::kCodesPerByte - 1 - dp) * core::kBitsPerCode;
            out = static_cast<std::uint8_t>(out | (code << shift));
        }
        dst[static_cast<std::size_t>(i) * dst_bpr + static_cast<std::size_t>(db)] = out;
    }
}

// The exponential-decay curve fit — reference §8
__device__ inline double dev_linfit_2x2(const double* __restrict__ y, int n, double v,
                                        bool affine, double* co0, double* c) {
    double Sbb = 0.0, Sb1 = 0.0, S11 = 0.0, Sby = 0.0, Sy = 0.0;
    double bi = 1.0;
    for (int i = 0; i < n; ++i) {
        const double b = bi;
        const double yi = y[i];
        Sbb += b * b;
        Sby += b * yi;
        if (affine) { Sb1 += b; S11 += 1.0; Sy += yi; }
        bi *= v;
    }
    if (affine) {
        const double det = Sbb * S11 - Sb1 * Sb1;
        if (det == 0.0) { *co0 = nan(""); *c = nan(""); return nan(""); }
        *co0 = (Sby * S11 - Sb1 * Sy) / det;
        *c = (Sbb * Sy - Sb1 * Sby) / det;
    } else {
        *co0 = (Sbb != 0.0) ? (Sby / Sbb) : nan("");
        *c = 0.0;
    }
    double rss = 0.0;
    double bb = 1.0;
    for (int i = 0; i < n; ++i) {
        const double pred = (*co0) * bb + (*c);
        const double r = y[i] - pred;
        rss += r * r;
        bb *= v;
    }
    return rss;
}

__global__ void dates_fit_curves_kernel(const double* __restrict__ curves, int win_len,
                                        int n_curves, double step, bool affine,
                                        double* __restrict__ d_date, double* __restrict__ d_sd,
                                        int* __restrict__ d_ok) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_curves) return;
    const double* y = curves + static_cast<long>(c) * win_len;
    d_date[c] = 0.0; d_sd[c] = 0.0; d_ok[c] = 0;
    if (win_len < 3 || !(step > 0.0)) return;

    const int coarse = 4000;
    double best_v = 0.5, best_rss = INFINITY, best_co0 = 0.0, best_c = 0.0;
    for (int k = 1; k < coarse; ++k) {
        const double v = static_cast<double>(k) / static_cast<double>(coarse);
        if (!(v > 0.0) || !(v < 1.0)) continue;
        double co0 = 0.0, cc = 0.0;
        const double rss = dev_linfit_2x2(y, win_len, v, affine, &co0, &cc);
        if (isnan(rss)) continue;
        if (co0 <= 0.0) continue;
        if (rss < best_rss) { best_rss = rss; best_v = v; best_co0 = co0; best_c = cc; }
    }
    if (!isfinite(best_rss)) {
        for (int k = 1; k < coarse; ++k) {
            const double v = static_cast<double>(k) / static_cast<double>(coarse);
            if (!(v > 0.0) || !(v < 1.0)) continue;
            double co0 = 0.0, cc = 0.0;
            const double rss = dev_linfit_2x2(y, win_len, v, affine, &co0, &cc);
            if (isnan(rss)) continue;
            if (rss < best_rss) { best_rss = rss; best_v = v; best_co0 = co0; best_c = cc; }
        }
        if (!isfinite(best_rss)) return;
    }
    double lo = fmax(1e-9, best_v - 1.0 / static_cast<double>(coarse));
    double hi = fmin(1.0 - 1e-9, best_v + 1.0 / static_cast<double>(coarse));
    const int ternary_iters = 200;
    for (int it = 0; it < ternary_iters; ++it) {
        const double m1 = lo + (hi - lo) / 3.0;
        const double m2 = hi - (hi - lo) / 3.0;
        double a = 0.0, b = 0.0;
        const double r1 = dev_linfit_2x2(y, win_len, m1, affine, &a, &b);
        const double r2 = dev_linfit_2x2(y, win_len, m2, affine, &a, &b);
        const double rr1 = isnan(r1) ? INFINITY : r1;
        const double rr2 = isnan(r2) ? INFINITY : r2;
        if (rr1 < rr2) hi = m2; else lo = m1;
    }
    best_v = 0.5 * (lo + hi);
    best_rss = dev_linfit_2x2(y, win_len, best_v, affine, &best_co0, &best_c);
    if (isnan(best_rss) || !(best_v > 0.0) || !(best_v < 1.0)) return;

    const double lambda = -log(best_v) / step;
    const bool ok = isfinite(lambda) && lambda > 0.0;
    d_date[c] = lambda;
    d_sd[c] = sqrt(best_rss / static_cast<double>(win_len));
    d_ok[c] = ok ? 1 : 0;
}

inline int grid_for(long n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

}  // namespace

// Launch wrappers and shared conventions — reference §9
void launch_dates_regress_dots(const double* d_src1, const double* d_src2, const double* d_valid,
                               const std::uint8_t* d_packed, std::size_t bytes_per_record,
                               int sample, long M, double* d_dot12, double* d_dot22,
                               cudaStream_t stream) {
    regress_dots_kernel<<<grid_for(M), kBlock, 0, stream>>>(
        d_src1, d_src2, d_valid, d_packed, bytes_per_record, sample, M, d_dot12, d_dot22);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_scatter(const double* d_src1, const double* d_src2, const double* d_valid,
                          const std::uint8_t* d_packed, std::size_t bytes_per_record, int sample,
                          long M, const int* d_grid_cell, double yreg, double* d_z0, double* d_z1,
                          double* d_z2, cudaStream_t stream) {
    scatter_kernel<<<grid_for(M), kBlock, 0, stream>>>(
        d_src1, d_src2, d_valid, d_packed, bytes_per_record, sample, M, d_grid_cell, yreg,
        d_z0, d_z1, d_z2);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_pack_segments(const double* d_grid, const int* d_chrom_first,
                                const int* d_chrom_last, int n_chrom, int numqbins, int n_fft,
                                double* d_padded, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_fft);
    pack_segments_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_grid, d_chrom_first, d_chrom_last, n_chrom, numqbins, n_fft, d_padded);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_power_spectrum(const void* d_freq, int n_cplx, int n_chrom, void* d_power,
                                 cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    power_spectrum_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        static_cast<const double2*>(d_freq), n_cplx, n_chrom, static_cast<double2*>(d_power));
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_cross_power(const void* d_freq_a, const void* d_freq_b, int n_cplx, int n_chrom,
                              void* d_out, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    cross_power_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        static_cast<const double2*>(d_freq_a), static_cast<const double2*>(d_freq_b), n_cplx,
        n_chrom, static_cast<double2*>(d_out));
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_extract_lags(const double* d_inv, int n_fft, int n_chrom, int diffmax,
                               double* d_dd, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * (static_cast<long>(diffmax) + 1);
    extract_lags_kernel<<<grid_for(total), kBlock, 0, stream>>>(d_inv, n_fft, n_chrom, diffmax,
                                                                d_dd);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_accumulate_bins(const double* d_dd00, const double* d_dd11,
                                  const double* d_dd02, const double* d_dd20, int n_chrom,
                                  int diffmax, int n_bin, int qbin, double* d_s0, double* d_s11,
                                  double* d_s12, double* d_s22, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * (static_cast<long>(diffmax) + 1);
    accumulate_bins_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_dd00, d_dd11, d_dd02, d_dd20, n_chrom, diffmax, n_bin, qbin, d_s0, d_s11, d_s12, d_s22);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_repack_target(const std::uint8_t* d_src, std::size_t src_bpr,
                                const long* d_kept_src, long M_kept, int n_target,
                                std::size_t dst_bpr, std::uint8_t* d_dst, cudaStream_t stream) {
    const long total = static_cast<long>(n_target) * static_cast<long>(dst_bpr);
    if (total <= 0) return;
    repack_target_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_src, src_bpr, d_kept_src, M_kept, n_target, dst_bpr, d_dst);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_dates_fit_curves(const double* d_curves, int win_len, int n_curves, double step,
                             bool affine, double* d_date, double* d_sd, int* d_ok,
                             cudaStream_t stream) {
    if (n_curves <= 0) return;
    STEPPE_NVTX_RANGE("dates_fit_curves");
    const int grid = grid_for(static_cast<long>(n_curves));
    dates_fit_curves_kernel<<<grid, kBlock, 0, stream>>>(
        d_curves, win_len, n_curves, step, affine, d_date, d_sd, d_ok);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
