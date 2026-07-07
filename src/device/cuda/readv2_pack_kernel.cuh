// src/device/cuda/readv2_pack_kernel.cuh
//
// launch_readv2_pack: pack one streamed individual-major 2-bit genotype CHUNK into
// the resident AoS [sample x SNP-window] Readv2Word bit-matrix. Inverts
// decode_af_kernel (which collapses samples to per-pop AF); here samples stay
// distinct and the SNP axis fans into allele/valid bits. Private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_readv2_pack_kernel.cuh.md
#ifndef STEPPE_DEVICE_CUDA_READV2_PACK_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_READV2_PACK_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "device/cuda/readv2_layout.cuh"

namespace steppe::device {

// Pack a chunk covering global SNPs [snp0, snp0+snp_count) into d_words.
//   d_chunk_packed        : individual-major chunk bytes on device (this chunk only)
//   chunk_bytes_per_record: ceil(snp_count / 4) — the chunk's per-sample stride
//   snp0                  : global SNP index the chunk starts at (multiple of window_snps)
//   snp_count             : SNPs in the chunk (multiple of window_snps, except the last)
//   m0                    : total SNPs tiled (for genome-tail padding of the last window)
//   d_words / words_per_sample : the resident bit-matrix [n_samples * words_per_sample]
void launch_readv2_pack(const std::uint8_t* d_chunk_packed,
                        std::size_t chunk_bytes_per_record,
                        int n_samples,
                        long snp0,
                        long snp_count,
                        long m0,
                        int window_snps,
                        Readv2Word* d_words,
                        long words_per_sample,
                        cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_READV2_PACK_KERNEL_CUH
