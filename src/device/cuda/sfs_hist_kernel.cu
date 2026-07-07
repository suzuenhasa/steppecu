// src/device/cuda/sfs_hist_kernel.cu
//
// GPU kernel for the 2D joint site-frequency spectrum (`steppe sfs`), plus its thin host
// launch wrapper. The bit-unpack (genotype_code) and the per-pop allele-count fold
// (wc_accumulate) are shared with the FST path; the complete-site test, the fold, and the
// mixed-radix cell index are shared with the CPU oracle via core/internal/sfs_hist.hpp, so
// the GPU and reference paths cannot drift on the counts, the missing exclusion, the
// complete-data restriction, the fold, or the cell placement.
//
// Access pattern mirrors the proven decode_af / fst_wc kernels: byte = s / kCodesPerByte,
// pos = s % kCodesPerByte, each individual g's code at SNP s is packed[g*bpr + byte]. The
// tile is population-contiguous, so each population is one [begin, end) individual range.
//
// The joint grid is small (e.g. 40 diploids/pop -> unfolded 81x81 = 6561 cells; folded
// 41x41), and increments (one per complete site, ~1.2M SNPs) spread across thousands of
// cells, so global atomics on unsigned long long are fine — contention is low. A per-block
// shared-memory privatized histogram is a documented optimization if profiling ever shows
// atomic pressure; not needed for v1.
#include "device/cuda/sfs_hist_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"      // genotype_code, kCodesPerByte
#include "core/internal/launch_config.hpp"  // kDecodeBlockX/Y, grid_for_x
#include "core/internal/sfs_hist.hpp"       // sfs_site_complete, sfs_axis_index, sfs_linear_index
#include "core/internal/wc_fst.hpp"         // WcPerPop, wc_accumulate
#include "device/cuda/check.cuh"

namespace steppe::device {

using core::genotype_code;
using core::kDecodeBlockX;
using core::wc_accumulate;
using core::WcPerPop;

namespace {

// One thread per SNP.
__global__ void sfs_hist_kernel(const std::uint8_t* __restrict__ packed,
                                std::size_t bytes_per_record,
                                std::size_t segA_begin, std::size_t segA_end,
                                std::size_t segB_begin, std::size_t segB_end,
                                long NA, long NB, long M, bool folded,
                                long extA, long extB,
                                unsigned long long* __restrict__ grid,
                                long long* __restrict__ n_complete) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= M) return;

    const std::size_t byte_in_record =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);

    WcPerPop A;
    for (std::size_t g = segA_begin; g < segA_end; ++g) {
        const std::uint8_t byte = packed[g * bytes_per_record + byte_in_record];
        wc_accumulate(genotype_code(byte, pos_in_byte), A);
    }
    WcPerPop B;
    for (std::size_t g = segB_begin; g < segB_end; ++g) {
        const std::uint8_t byte = packed[g * bytes_per_record + byte_in_record];
        wc_accumulate(genotype_code(byte, pos_in_byte), B);
    }

    // Complete-data restriction (§4): drop sites with any missing genotype in either pop.
    if (!core::sfs_site_complete(A.n, NA, B.n, NB)) return;

    const long idx[2] = {core::sfs_axis_index(A.ac, NA, folded),
                         core::sfs_axis_index(B.ac, NB, folded)};
    const long ext[2] = {extA, extB};
    const long lin = core::sfs_linear_index(idx, ext, 2);
    atomicAdd(&grid[lin], 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(n_complete), 1ULL);
}

}  // namespace

void launch_sfs_hist_2pop(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                          std::size_t segA_begin, std::size_t segA_end,
                          std::size_t segB_begin, std::size_t segB_end,
                          long NA, long NB, long M, bool folded,
                          long extA, long extB,
                          unsigned long long* d_grid, long grid_len,
                          long long* d_n_complete,
                          cudaStream_t stream) {
    // Zero the grid + counter first (the accumulation is atomicAdd onto a clean slate).
    if (grid_len > 0) {
        STEPPE_CUDA_CHECK(cudaMemsetAsync(
            d_grid, 0, static_cast<std::size_t>(grid_len) * sizeof(unsigned long long), stream));
    }
    STEPPE_CUDA_CHECK(cudaMemsetAsync(d_n_complete, 0, sizeof(long long), stream));
    if (M <= 0) return;

    const int block = kDecodeBlockX * core::kDecodeBlockY;  // 256 threads / block
    const int grid = core::grid_for_x(
        M, block, "sfs gridDim.x (SNP/M axis) exceeds kMaxGridX — tile the SNP axis");
    sfs_hist_kernel<<<static_cast<unsigned>(grid), static_cast<unsigned>(block), 0, stream>>>(
        d_packed, bytes_per_record, segA_begin, segA_end, segB_begin, segB_end, NA, NB, M, folded,
        extA, extB, d_grid, d_n_complete);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
