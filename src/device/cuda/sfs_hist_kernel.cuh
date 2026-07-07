#pragma once
// src/device/cuda/sfs_hist_kernel.cuh
//
// Declaration for the 2D joint-SFS histogram kernel launcher. The kernel body lives in
// sfs_hist_kernel.cu and shares the per-pop allele-count fold (wc_accumulate) and the
// complete-site / fold / index arithmetic with the CPU oracle via
// core/internal/sfs_hist.hpp, so the GPU and reference paths cannot diverge.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

// One thread per SNP: fold the two population segments' diploid codes into per-pop A1-copy
// counts, drop sites that are not complete in BOTH pops, map the pair of counts to one
// linear cell, and atomicAdd into the (extA*extB) device grid. d_n_complete counts the
// sites histogrammed. The grid is zeroed by the launcher before accumulation. Extents
// (extA/extB) are computed host-side via core::sfs_axis_extent (2N+1 unfolded, N+1 folded).
void launch_sfs_hist_2pop(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                          std::size_t segA_begin, std::size_t segA_end,
                          std::size_t segB_begin, std::size_t segB_end,
                          long NA, long NB, long M, bool folded,
                          long extA, long extB,
                          unsigned long long* d_grid, long grid_len,
                          long long* d_n_complete,
                          cudaStream_t stream);

}  // namespace steppe::device
