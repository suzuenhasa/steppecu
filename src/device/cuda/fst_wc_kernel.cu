// src/device/cuda/fst_wc_kernel.cu
//
// GPU kernels for the per-site Weir & Cockerham 1984 FST over one population pair, plus
// the thin host launch wrappers. The bit-unpack (genotype_code) and the WC variance-
// component math (wc_accumulate / wc_finalize) are shared with the CPU oracle via
// core/internal/wc_fst.hpp, so the GPU and reference paths cannot drift on the counts,
// the missing-genotype exclusion, the divide, or the invalid-site guard.
//
// Access pattern mirrors the proven decode_af kernel: byte = s / kCodesPerByte, pos =
// s % kCodesPerByte, and each individual g's code at SNP s is packed[g*bpr + byte]. The
// tile is population-contiguous, so each population is one [begin, end) individual range.
#include "device/cuda/fst_wc_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"      // genotype_code, kCodesPerByte
#include "core/internal/launch_config.hpp"  // kDecodeBlockX, grid_for_x
#include "core/internal/wc_fst.hpp"         // WcPerPop, WcSite, wc_accumulate, wc_finalize
#include "device/cuda/check.cuh"

namespace steppe::device {

using core::genotype_code;
using core::kDecodeBlockX;
using core::wc_accumulate;
using core::wc_finalize;
using core::WcPerPop;
using core::WcSite;

namespace {

// One thread per SNP — reference §3
__global__ void fst_wc_kernel(const std::uint8_t* __restrict__ packed,
                              std::size_t bytes_per_record,
                              std::size_t segA_begin, std::size_t segA_end,
                              std::size_t segB_begin, std::size_t segB_end,
                              long M,
                              double* __restrict__ out_num, double* __restrict__ out_den,
                              double* __restrict__ out_fst, std::uint8_t* __restrict__ out_valid) {
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

    const WcSite r = wc_finalize(A, B);
    out_num[s] = r.num;
    out_den[s] = r.den;
    out_fst[s] = r.fst;
    out_valid[s] = r.valid ? std::uint8_t{1} : std::uint8_t{0};
}

// Mask per-site num/den by (valid && include) and emit a per-site inclusion count.
__global__ void fst_summary_contrib_kernel(const double* __restrict__ num,
                                           const double* __restrict__ den,
                                           const std::uint8_t* __restrict__ valid,
                                           const std::uint8_t* __restrict__ include,
                                           long M,
                                           double* __restrict__ cnum, double* __restrict__ cden,
                                           long* __restrict__ cnt) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= M) return;
    const bool keep = (valid[s] != 0) && (include == nullptr || include[s] != 0);
    cnum[s] = keep ? num[s] : 0.0;
    cden[s] = keep ? den[s] : 0.0;
    cnt[s] = keep ? 1L : 0L;
}

}  // namespace

void launch_fst_wc(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                   std::size_t segA_begin, std::size_t segA_end,
                   std::size_t segB_begin, std::size_t segB_end,
                   long M,
                   double* d_num, double* d_den, double* d_fst, std::uint8_t* d_valid,
                   cudaStream_t stream) {
    if (M <= 0) return;
    const int block = kDecodeBlockX * core::kDecodeBlockY;  // 256 threads / block
    const int grid = core::grid_for_x(
        M, block, "fst gridDim.x (SNP/M axis) exceeds kMaxGridX — tile the SNP axis");
    fst_wc_kernel<<<static_cast<unsigned>(grid), static_cast<unsigned>(block), 0, stream>>>(
        d_packed, bytes_per_record, segA_begin, segA_end, segB_begin, segB_end, M,
        d_num, d_den, d_fst, d_valid);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_fst_summary_contrib(const double* d_num, const double* d_den,
                                const std::uint8_t* d_valid, const std::uint8_t* d_include,
                                long M,
                                double* d_cnum, double* d_cden, long* d_cnt,
                                cudaStream_t stream) {
    if (M <= 0) return;
    const int block = kDecodeBlockX * core::kDecodeBlockY;
    const int grid = core::grid_for_x(
        M, block, "fst summary gridDim.x (SNP/M axis) exceeds kMaxGridX — tile the SNP axis");
    fst_summary_contrib_kernel<<<static_cast<unsigned>(grid), static_cast<unsigned>(block), 0,
                                 stream>>>(d_num, d_den, d_valid, d_include, M, d_cnum, d_cden,
                                           d_cnt);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
