// src/device/cuda/dates_kernel.cuh
//
// Narrow launch-wrapper declarations for the DATES cuFFT autocorrelation LD engine
// (include/steppe/dates.hpp; the S2 divergence). Host orchestration (cuda_backend.cu) calls
// these `launch_*`; the kernel bodies + <<<>>> live only in dates_kernel.cu (architecture.md
// §7 "host code never includes kernel bodies or <<<>>>").
//
// This header names CUDA types (cudaStream_t) and so is PRIVATE to steppe_device
// (architecture.md §4) — the device-internal seam between the backend and the DATES kernels,
// not the CUDA-free public seam (include/steppe/dates.hpp).
//
// THE GPU-BOUND SHAPE. The covariance curve is the AUTOCORRELATION of the genetic-map-binned
// weighted grid (the ALDER FFT trick): cov(lag) = Σ_g z1[g]·z1[g+lag] / Σ_g z0[g]·z0[g+lag]
// = IFFT(|FFT(z1)|²) / IFFT(|FFT(z0)|²). The ~10^12 SNP-pair object is NEVER formed. The
// per-(chrom, output-bin) DATES corr uses dd00 (count autocorr), dd11 (signal autocorr),
// dd02 (count×signal²), dd20 (signal²×count) — corr = (dd11/dd00)/sqrt((dd02/dd00)·(dd20/dd00))
// at calccorr mode 1 (means not subtracted). The cross terms dd01/dd10 are the mean terms
// (unused at mode 1) and are not computed.
#ifndef STEPPE_DEVICE_CUDA_DATES_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DATES_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

/// PER-SAMPLE regression dot products (DATES dates.c:606-626): over admixed-target sample
/// `sample`'s valid SNPs (both sources valid AND the genotype non-missing), reduce
///   dot12 = Σ (w0-w2)(w1-w2),  dot22 = Σ (w1-w2)²
/// where w0 = code/2 (dosage), w1 = d_src1[s], w2 = d_src2[s]. The regression coeff is
/// y = dot12/dot22 (computed host-side from the two scalars). d_dot12/d_dot22 are single
/// doubles, pre-zeroed by the caller; the kernel atomic-adds the per-block partials.
void launch_dates_regress_dots(const double* d_src1, const double* d_src2, const double* d_valid,
                               const std::uint8_t* d_packed, std::size_t bytes_per_record,
                               int sample, long M, double* d_dot12, double* d_dot22,
                               cudaStream_t stream);

/// PER-SAMPLE residual SCATTER (DATES dates.c:627-665): over the sample's valid SNPs,
/// residual r = w0 - (y·w1 + (1-y)·w2), signal = r·(w1-w2); scatter onto the fine grid:
///   z0[cell] += 1,  z1[cell] += signal,  z2[cell] += signal²   (atomicAdd; cell = grid_cell[s]).
/// The grids d_z0/d_z1/d_z2 are [numqbins], pre-zeroed by the caller.
void launch_dates_scatter(const double* d_src1, const double* d_src2, const double* d_valid,
                          const std::uint8_t* d_packed, std::size_t bytes_per_record, int sample,
                          long M, const int* d_grid_cell, double yreg,
                          double* d_z0, double* d_z1, double* d_z2, cudaStream_t stream);

/// PACK each chrom's grid segment [slo..shi] into a zero-padded FFT input row (length n_fft).
/// Batched over n_chrom: d_padded[kc*n_fft + (0..len-1)] = d_grid[slo..]; rest zeroed.
void launch_dates_pack_segments(const double* d_grid, const int* d_chrom_first,
                                const int* d_chrom_last, int n_chrom, int numqbins,
                                int n_fft, double* d_padded, cudaStream_t stream);

/// |F|² POWER (DATES fftauto cnorm2): out[k] = (re²+im²) + 0i, batched n_chrom × n_cplx.
void launch_dates_power_spectrum(const void* d_freq, int n_cplx, int n_chrom, void* d_power,
                                 cudaStream_t stream);

/// CROSS POWER conj(A)·B (the count×signal² cross), batched n_chrom × n_cplx. out = conj(A)*B.
void launch_dates_cross_power(const void* d_freq_a, const void* d_freq_b, int n_cplx, int n_chrom,
                              void* d_out, cudaStream_t stream);

/// EXTRACT + ACCUMULATE the inverse-FFT autocorrelation at lags [0, diffmax] into the compact
/// [n_chrom × (diffmax+1)] dd array, applying the 1/n_fft scale (DATES fftauto vst(1/yn)) and
/// SUMMING across samples (+=). lag l is at inverse-FFT index l. d_inv is C2R output [n_chrom ×
/// n_fft]; d_dd is [n_chrom × (diffmax+1)].
void launch_dates_extract_lags(const double* d_inv, int n_fft, int n_chrom, int diffmax,
                               double* d_dd, cudaStream_t stream);

/// RE-BIN the per-chrom fine-grid lag dd moments into the per-(chrom, output-bin) DATES corr
/// sufficient statistics (DATES ddcorr + addcorr2, mode-1 corr). For each chrom kc, lag d in
/// [1, diffmax] with dd00[d] >= 0.5: output bin s = floor(d/qbin); s0[kc*n_bin+s] += dd00[d],
/// s12 += dd11[d], s11 += dd02[d], s22 += dd20[d] (s1/s2 left 0 — the mode-1 means).
void launch_dates_accumulate_bins(const double* d_dd00, const double* d_dd11,
                                  const double* d_dd02, const double* d_dd20,
                                  int n_chrom, int diffmax, int n_bin, int qbin,
                                  double* d_s0, double* d_s11, double* d_s12, double* d_s22,
                                  cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DATES_KERNEL_CUH
