// src/device/cuda/dates_kernel.cu
//
// The DATES cuFFT autocorrelation LD engine kernels — admixture dating (the S2 divergence;
// include/steppe/dates.hpp). NEVER the f2 GEMM, NEVER a host O(M²) SNP-pair loop. The
// per-sample weight/residual/scatter onto the fine genetic-map grid, the |F|² / cross-power
// elementwise stages, and the lag-extract + re-bin into the per-(chrom, output-bin) DATES
// CORR sufficient statistics. The cuFFT forward/inverse transforms themselves are issued from
// the backend (cuda_backend.cu) via cufftExecD2Z/cufftExecZ2D; these kernels are the
// elementwise / scatter / reduce glue around them.
//
// PINNED to the DATES reference C source (dates.c:585-665 per-sample scatter; fftsubs.c
// fftauto/fftcv2 = FWD FFT -> |F|² / conj·-> INV FFT -> /n; ddcorr lag->bin). The
// autocorrelation IS the binned pairwise double-sum (Σ_g W[g]·W[g+lag]), so the SNP pairs are
// never materialized — O(G log G) per chrom, FLAT in M.
//
// CUDA TU: PRIVATE to steppe_device (architecture.md §4). The kernel bodies + <<<>>> live
// ONLY here; the backend reaches them through the narrow launch wrappers (dates_kernel.cuh).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"  // core::genotype_code / kMissingGenotypeCode / kCodesPerByte
#include "device/cuda/check.cuh"        // STEPPE_CUDA_CHECK
#include "device/cuda/dates_kernel.cuh"

namespace steppe::device {

namespace {

inline constexpr int kBlock = 256;

/// Decode admixed-sample `sample`'s genotype code at SNP s from the dense packed record.
__device__ inline std::uint8_t target_code(const std::uint8_t* __restrict__ packed,
                                           std::size_t bytes_per_record, int sample, long s) {
    const std::uint8_t* rec = packed + static_cast<std::size_t>(sample) * bytes_per_record;
    const std::size_t byte_in_rec =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos = static_cast<int>(s % core::kCodesPerByte);
    return core::genotype_code(rec[byte_in_rec], pos);
}

/// PHASE A — the regression dot products, atomic-reduced over the sample's valid SNPs.
__global__ void regress_dots_kernel(const double* __restrict__ src1,
                                    const double* __restrict__ src2,
                                    const double* __restrict__ valid,
                                    const std::uint8_t* __restrict__ packed,
                                    std::size_t bytes_per_record, int sample, long M,
                                    double* __restrict__ dot12, double* __restrict__ dot22) {
    const long s = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    // block-wide partials then one atomicAdd per block (reduce atomic contention).
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

/// PHASE B — the residual scatter onto the fine grid.
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

/// PACK chrom segments into zero-padded FFT rows. One thread per (chrom, position) cell.
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

/// |F|² power spectrum: out[k] = re²+im² + 0i. Operates on cufftDoubleComplex (re,im pairs).
__global__ void power_spectrum_kernel(const double2* __restrict__ freq, int n_cplx, int n_chrom,
                                      double2* __restrict__ power) {
    const long k = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    if (k >= total) return;
    const double2 a = freq[k];
    power[k] = make_double2(a.x * a.x + a.y * a.y, 0.0);
}

/// Cross power conj(A)·B: out = (Ar - i·Ai)(Br + i·Bi) = (ArBr+AiBi) + i(ArBi-AiBr).
__global__ void cross_power_kernel(const double2* __restrict__ a, const double2* __restrict__ b,
                                   int n_cplx, int n_chrom, double2* __restrict__ out) {
    const long k = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    if (k >= total) return;
    const double2 av = a[k];
    const double2 bv = b[k];
    out[k] = make_double2(av.x * bv.x + av.y * bv.y, av.x * bv.y - av.y * bv.x);
}

/// Extract + accumulate lags [0, diffmax] from the C2R inverse output, /n_fft, summed (+=).
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

/// Re-bin the fine-grid lag dd moments into the per-(chrom, output-bin) corr stats.
/// One thread per (chrom, lag); atomicAdd into the output bin (multiple lags map to one bin).
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
    if (d < 1) return;  // lag 0 skipped (DATES ddcorr d=1..)
    const double c00 = dd00[cell];
    if (c00 < 0.5) return;
    const int s = d / qbin;  // output bin = floor(d·dbinsize/binsize) = floor(d/qbin)
    if (s < 0 || s >= n_bin) return;
    const long o = static_cast<long>(kc) * n_bin + s;
    atomicAdd(&s0[o], c00);
    atomicAdd(&s12[o], dd11[cell]);
    atomicAdd(&s11[o], dd02[cell]);
    atomicAdd(&s22[o], dd20[cell]);
}

/// M5 — TARGET-GENOTYPE REPACK onto the kept SNP axis (dates.cpp:296-313, on device).
/// One thread per (target individual i, kept SNP ks): read the 2-bit code from the SOURCE
/// record `i` at original SNP index s = kept_src[ks], then OR it into the dest record at
/// dense bit position ks. BIT-EXACT to the host shift/OR body (dates.cpp:304-311): the read
/// uses core::genotype_code (== io::code_in_byte, same MSB-first 6/4/2/0 shift), and the
/// write reproduces the host `(kCodesPerByte-1-dp)*kBitsPerCode` shift. The dest byte is
/// written by a SINGLE thread per (i, dest byte position) — but here one thread per (i, ks)
/// would race on the shared dest byte (4 codes/byte). To stay bit-exact AND race-free, one
/// thread owns a full DEST BYTE: it packs up to kCodesPerByte consecutive kept codes (the
/// host writes dst by ascending ks, so a dest byte db holds kept indices [db*4, db*4+3]).
__global__ void repack_target_kernel(const std::uint8_t* __restrict__ src, std::size_t src_bpr,
                                     const long* __restrict__ kept_src, long M_kept, int n_target,
                                     std::size_t dst_bpr, std::uint8_t* __restrict__ dst) {
    const long dst_bytes = static_cast<long>(dst_bpr);
    const long total = static_cast<long>(n_target) * dst_bytes;
    for (long gid = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
         gid < total; gid += static_cast<long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(gid / dst_bytes);
        const long db = gid % dst_bytes;  // which dest byte this thread owns
        const std::uint8_t* src_rec = src + static_cast<std::size_t>(i) * src_bpr;
        std::uint8_t out = 0;
        // Pack the up-to-kCodesPerByte kept codes that land in dest byte db (ascending ks).
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

// --- M6 exp-fit (device fit_exp_decay; NATIVE FP64) --------------------------------------
// linfit_2x2: solve the 2×2 (or 1×1 non-affine) normal equations for (co0,c) minimizing
// Σ(y[i] - co0·v^i - c)²; return the residual sum of squares. EXACT mirror of dates.cpp's
// linfit_2x2, FP64 (the host long double -> device double; the cancellation carve-out form
// for a device with no long double). co0/c returned via pointers.
__device__ inline double dev_linfit_2x2(const double* __restrict__ y, int n, double v,
                                        bool affine, double* co0, double* c) {
    double Sbb = 0.0, Sb1 = 0.0, S11 = 0.0, Sby = 0.0, Sy = 0.0;
    double bi = 1.0;  // v^0
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

/// One thread per curve: the EXACT dates.cpp fit_exp_decay (4000-grid + ternary refine).
__global__ void dates_fit_curves_kernel(const double* __restrict__ curves, int win_len,
                                        int n_curves, double step, bool affine,
                                        double* __restrict__ d_date, double* __restrict__ d_sd,
                                        int* __restrict__ d_ok) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_curves) return;
    const double* y = curves + static_cast<long>(c) * win_len;
    // default outputs (== ExpFit default: date 0, sd 0, ok 0).
    d_date[c] = 0.0; d_sd[c] = 0.0; d_ok[c] = 0;
    if (win_len < 3 || !(step > 0.0)) return;

    const int coarse = 4000;
    double best_v = 0.5, best_rss = INFINITY, best_co0 = 0.0, best_c = 0.0;
    // coarse grid requiring a DECAYING positive exponential (co0>0).
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
        // fallback: allow any-sign amplitude (degenerate curve).
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
    // local ternary refine.
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

void launch_dates_regress_dots(const double* d_src1, const double* d_src2, const double* d_valid,
                               const std::uint8_t* d_packed, std::size_t bytes_per_record,
                               int sample, long M, double* d_dot12, double* d_dot22,
                               cudaStream_t stream) {
    regress_dots_kernel<<<grid_for(M), kBlock, 0, stream>>>(
        d_src1, d_src2, d_valid, d_packed, bytes_per_record, sample, M, d_dot12, d_dot22);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_scatter(const double* d_src1, const double* d_src2, const double* d_valid,
                          const std::uint8_t* d_packed, std::size_t bytes_per_record, int sample,
                          long M, const int* d_grid_cell, double yreg, double* d_z0, double* d_z1,
                          double* d_z2, cudaStream_t stream) {
    scatter_kernel<<<grid_for(M), kBlock, 0, stream>>>(
        d_src1, d_src2, d_valid, d_packed, bytes_per_record, sample, M, d_grid_cell, yreg,
        d_z0, d_z1, d_z2);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_pack_segments(const double* d_grid, const int* d_chrom_first,
                                const int* d_chrom_last, int n_chrom, int numqbins, int n_fft,
                                double* d_padded, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_fft);
    pack_segments_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_grid, d_chrom_first, d_chrom_last, n_chrom, numqbins, n_fft, d_padded);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_power_spectrum(const void* d_freq, int n_cplx, int n_chrom, void* d_power,
                                 cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    power_spectrum_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        static_cast<const double2*>(d_freq), n_cplx, n_chrom, static_cast<double2*>(d_power));
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_cross_power(const void* d_freq_a, const void* d_freq_b, int n_cplx, int n_chrom,
                              void* d_out, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * static_cast<long>(n_cplx);
    cross_power_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        static_cast<const double2*>(d_freq_a), static_cast<const double2*>(d_freq_b), n_cplx,
        n_chrom, static_cast<double2*>(d_out));
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_extract_lags(const double* d_inv, int n_fft, int n_chrom, int diffmax,
                               double* d_dd, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * (static_cast<long>(diffmax) + 1);
    extract_lags_kernel<<<grid_for(total), kBlock, 0, stream>>>(d_inv, n_fft, n_chrom, diffmax,
                                                                d_dd);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_accumulate_bins(const double* d_dd00, const double* d_dd11,
                                  const double* d_dd02, const double* d_dd20, int n_chrom,
                                  int diffmax, int n_bin, int qbin, double* d_s0, double* d_s11,
                                  double* d_s12, double* d_s22, cudaStream_t stream) {
    const long total = static_cast<long>(n_chrom) * (static_cast<long>(diffmax) + 1);
    accumulate_bins_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_dd00, d_dd11, d_dd02, d_dd20, n_chrom, diffmax, n_bin, qbin, d_s0, d_s11, d_s12, d_s22);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_repack_target(const std::uint8_t* d_src, std::size_t src_bpr,
                                const long* d_kept_src, long M_kept, int n_target,
                                std::size_t dst_bpr, std::uint8_t* d_dst, cudaStream_t stream) {
    const long total = static_cast<long>(n_target) * static_cast<long>(dst_bpr);
    if (total <= 0) return;
    repack_target_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_src, src_bpr, d_kept_src, M_kept, n_target, dst_bpr, d_dst);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

void launch_dates_fit_curves(const double* d_curves, int win_len, int n_curves, double step,
                             bool affine, double* d_date, double* d_sd, int* d_ok,
                             cudaStream_t stream) {
    if (n_curves <= 0) return;
    // One thread per curve (n_curves ~ n_chrom+1 ~ 23): a single small block suffices, but
    // size the grid generically so a future many-curve batch still launches correctly.
    const int grid = grid_for(static_cast<long>(n_curves));
    dates_fit_curves_kernel<<<grid, kBlock, 0, stream>>>(
        d_curves, win_len, n_curves, step, affine, d_date, d_sd, d_ok);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace steppe::device
