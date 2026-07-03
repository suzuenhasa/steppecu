// src/device/cuda/dates_kernel.cuh
//
// Launch-wrapper declarations for the DATES cuFFT autocorrelation LD engine. This header names
// CUDA types (cudaStream_t) and so is private to steppe_device — the device-internal seam
// between the backend and the DATES kernels, whose bodies and <<<>>> live in the .cu file.
//
// Reference: docs/reference/src_device_cuda_dates_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_DATES_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DATES_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// Per-sample regression and residual scatter — reference §4
void launch_dates_regress_dots(const double* d_src1, const double* d_src2, const double* d_valid,
                               const std::uint8_t* d_packed, std::size_t bytes_per_record,
                               int sample, long M, double* d_dot12, double* d_dot22,
                               cudaStream_t stream);

void launch_dates_scatter(const double* d_src1, const double* d_src2, const double* d_valid,
                          const std::uint8_t* d_packed, std::size_t bytes_per_record, int sample,
                          long M, const int* d_grid_cell, double yreg,
                          double* d_z0, double* d_z1, double* d_z2, cudaStream_t stream);

// FFT autocorrelation of the grid — reference §5
void launch_dates_pack_segments(const double* d_grid, const int* d_chrom_first,
                                const int* d_chrom_last, int n_chrom, int numqbins,
                                int n_fft, double* d_padded, cudaStream_t stream);

void launch_dates_power_spectrum(const void* d_freq, int n_cplx, int n_chrom, void* d_power,
                                 cudaStream_t stream);

void launch_dates_cross_power(const void* d_freq_a, const void* d_freq_b, int n_cplx, int n_chrom,
                              void* d_out, cudaStream_t stream);

void launch_dates_extract_lags(const double* d_inv, int n_fft, int n_chrom, int diffmax,
                               double* d_dd, cudaStream_t stream);

// Re-binning lag moments into correlation statistics — reference §6
void launch_dates_accumulate_bins(const double* d_dd00, const double* d_dd11,
                                  const double* d_dd02, const double* d_dd20,
                                  int n_chrom, int diffmax, int n_bin, int qbin,
                                  double* d_s0, double* d_s11, double* d_s12, double* d_s22,
                                  cudaStream_t stream);

// Target-genotype repack — reference §7
void launch_dates_repack_target(const std::uint8_t* d_src, std::size_t src_bpr,
                                const long* d_kept_src, long M_kept, int n_target,
                                std::size_t dst_bpr, std::uint8_t* d_dst,
                                cudaStream_t stream);

// Exponential-decay curve fit — reference §8
void launch_dates_fit_curves(const double* d_curves, int win_len, int n_curves, double step,
                             bool affine, double* d_date, double* d_sd, int* d_ok,
                             cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DATES_KERNEL_CUH
