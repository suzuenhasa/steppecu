#pragma once
// src/device/cuda/fst_wc_kernel.cuh
//
// Declarations for the per-site Weir & Cockerham 1984 FST kernel launchers. The kernel
// bodies live in fst_wc_kernel.cu and share the WC variance-component arithmetic with the
// CPU oracle via core/internal/wc_fst.hpp, so the GPU and reference paths cannot diverge.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// One thread per SNP: fold the two population segments' diploid codes into the WC
// components and store num/den/fst/valid per SNP over the device-resident packed tile.
void launch_fst_wc(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                   std::size_t segA_begin, std::size_t segA_end,
                   std::size_t segB_begin, std::size_t segB_end,
                   long M,
                   double* d_num, double* d_den, double* d_fst, std::uint8_t* d_valid,
                   cudaStream_t stream);

// Mask the per-site num/den by (valid && summary_include) into contribution buffers, and
// write a per-site inclusion count, so a plain sum-reduction yields the genome-wide
// Σnum/Σden and n_valid over the summary-eligible valid sites.
void launch_fst_summary_contrib(const double* d_num, const double* d_den,
                                const std::uint8_t* d_valid, const std::uint8_t* d_include,
                                long M,
                                double* d_cnum, double* d_cden, long* d_cnt,
                                cudaStream_t stream);

}  // namespace steppe::device
