// src/device/cuda/readv2_pack_kernel.cu
//
// The READv2 pack kernel: one thread per (sample, chunk-local output word) builds the
// allele/valid bit planes of that 64-SNP block from the sample's 2-bit codes and
// writes one Readv2Word into the resident bit-matrix. Consecutive threads (same
// sample, adjacent word) read adjacent record bytes and write adjacent 16-byte cells,
// so both ends are fully coalesced. This inverts decode_af_kernel.
#include "device/cuda/readv2_pack_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"
#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"

namespace steppe::device {

namespace {

// Block geometry: x walks output words (contiguous cells), y walks samples.
constexpr int kPackBlockX = 64;
constexpr int kPackBlockY = 4;

__global__ void readv2_pack_kernel(const std::uint8_t* __restrict__ chunk,
                                   std::size_t chunk_bytes_per_record,
                                   int n_samples,
                                   long snp0,
                                   long m0,
                                   int window_snps,
                                   int wpw,
                                   long chunk_out_words,
                                   Readv2Word* __restrict__ d_words,
                                   long words_per_sample) {
    const long wl = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int s = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (wl >= chunk_out_words || s >= n_samples) return;

    // chunk-local output word -> (window within chunk, word within window)
    const long g_local = wl / wpw;              // window index within this chunk
    const int wlocal = static_cast<int>(wl % wpw);  // word index within the window

    // Window-relative SNP base (NOT wl*64 — the tail padding makes that wrong, scope T2).
    const long g_global = snp0 / window_snps + g_local;            // global window index
    const long snp_base_global = g_global * static_cast<long>(window_snps) +
                                 static_cast<long>(wlocal) * kReadv2SnpsPerWord;
    const long snp_base_local = snp_base_global - snp0;            // chunk-local base

    // Real SNPs this word holds: clamp to 64, the window tail, and the genome tail.
    long real = kReadv2SnpsPerWord;
    const long window_tail = static_cast<long>(window_snps) -
                             static_cast<long>(wlocal) * kReadv2SnpsPerWord;
    if (window_tail < real) real = window_tail;
    const long genome_tail = m0 - snp_base_global;
    if (genome_tail < real) real = genome_tail;

    std::uint64_t allele = 0;
    std::uint64_t valid = 0;
    const std::uint8_t* rec = chunk + static_cast<std::size_t>(s) * chunk_bytes_per_record;
    for (long l = 0; l < real; ++l) {
        const long local_snp = snp_base_local + l;
        const std::size_t byte_idx =
            static_cast<std::size_t>(local_snp) / static_cast<std::size_t>(core::kCodesPerByte);
        const int pos = static_cast<int>(local_snp % core::kCodesPerByte);
        const std::uint8_t code = core::genotype_code(rec[byte_idx], pos);
        // Only genuine 0/2 pseudo-haploid hardcalls are valid; het(1)/missing(3) drop.
        if (code == 0u || code == 2u) {
            const std::uint64_t bit = 1ULL << l;  // LSB-first
            valid |= bit;
            if (code == 2u) allele |= bit;
        }
    }

    const long out = static_cast<long>(s) * words_per_sample +
                     g_global * static_cast<long>(wpw) + wlocal;
    d_words[out].allele = allele;
    d_words[out].valid = valid;
}

}  // namespace

void launch_readv2_pack(const std::uint8_t* d_chunk_packed,
                        std::size_t chunk_bytes_per_record,
                        int n_samples,
                        long snp0,
                        long snp_count,
                        long m0,
                        int window_snps,
                        Readv2Word* d_words,
                        long words_per_sample,
                        cudaStream_t stream) {
    if (n_samples <= 0 || snp_count <= 0 || window_snps <= 0) return;
    const int wpw = readv2_wpw(window_snps);
    const long chunk_win = (snp_count + window_snps - 1) / window_snps;  // windows in chunk
    const long chunk_out_words = chunk_win * static_cast<long>(wpw);

    const dim3 block(kPackBlockX, kPackBlockY);
    const int grid_x = core::grid_for_x(
        chunk_out_words, kPackBlockX,
        "readv2 pack gridDim.x (word axis) exceeds kMaxGridX — reduce chunk size");
    const dim3 grid(static_cast<unsigned>(grid_x),
                    static_cast<unsigned>(core::grid_for(n_samples, kPackBlockY)));
    readv2_pack_kernel<<<grid, block, 0, stream>>>(
        d_chunk_packed, chunk_bytes_per_record, n_samples, snp0, m0, window_snps, wpw,
        chunk_out_words, d_words, words_per_sample);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
