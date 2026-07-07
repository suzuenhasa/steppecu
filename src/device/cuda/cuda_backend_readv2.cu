// src/device/cuda/cuda_backend_readv2.cu
//
// Out-of-line CudaBackend bodies for the three READv2 seams: allocate+zero the
// resident [sample x SNP-window] bit-matrix, pack a streamed 2-bit genotype chunk
// into it, and run the all-pairs __popc windowed-mismatch reduction. A CUDA TU
// private to the steppe_device target.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_readv2.cu.md
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/readv2_bitmatrix_impl.cuh"
#include "device/cuda/readv2_layout.cuh"
#include "device/cuda/readv2_mismatch_kernel.cuh"
#include "device/cuda/readv2_pack_kernel.cuh"

namespace steppe::device {

// Allocate + zero the resident bit-matrix. The zero is load-bearing: any never-written
// padding word (window tails, genome tail) must stay valid=0 (scope T5/T10).
Readv2Bitmatrix CudaBackend::readv2_alloc_bitmatrix(int n_samples, int window_snps, long m0) {
    guard_device();
    STEPPE_NVTX_RANGE("readv2_alloc_bitmatrix");
    Readv2Bitmatrix bits;
    bits.n_samples = n_samples;
    bits.window_snps = window_snps;
    bits.m0 = m0;
    bits.device_id = device_id_;
    if (n_samples <= 0 || window_snps <= 0 || m0 <= 0) return bits;

    bits.wpw = readv2_wpw(window_snps);
    bits.n_win = readv2_n_win(m0, window_snps);
    bits.words_per_sample = bits.n_win * static_cast<long>(bits.wpw);

    const std::size_t total =
        static_cast<std::size_t>(n_samples) * static_cast<std::size_t>(bits.words_per_sample);
    bits.impl = std::make_unique<Readv2Bitmatrix::Impl>();
    bits.impl->words = DeviceBuffer<Readv2Word>(total);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(bits.impl->words.data(), 0,
                                      total * sizeof(Readv2Word), stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return bits;
}

// Pack one individual-major 2-bit chunk (host bytes) into the resident bit-matrix.
void CudaBackend::readv2_pack_chunk(Readv2Bitmatrix& bits, const std::uint8_t* chunk_packed,
                                    std::size_t chunk_bytes_per_record, int n_samples, long snp0,
                                    long snp_count) {
    guard_device();
    STEPPE_NVTX_RANGE("readv2_pack_chunk");
    if (!bits.impl || n_samples <= 0 || snp_count <= 0) return;

    const std::size_t chunk_bytes =
        static_cast<std::size_t>(n_samples) * chunk_bytes_per_record;
    DeviceBuffer<std::uint8_t> d_chunk(chunk_bytes);
    h2d_async(d_chunk, chunk_packed, chunk_bytes, stream_.get());
    launch_readv2_pack(d_chunk.data(), chunk_bytes_per_record, n_samples, snp0, snp_count,
                       bits.m0, bits.window_snps, bits.impl->words.data(),
                       bits.words_per_sample, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));  // d_chunk freed after use
}

// Run the all-pairs windowed-mismatch reduction; return the four per-pair arrays.
Readv2Pairs CudaBackend::readv2_mismatch(const Readv2Bitmatrix& bits, long long n_pairs,
                                         bool tiled) {
    guard_device();
    STEPPE_NVTX_RANGE("readv2_mismatch");
    Readv2Pairs out;
    if (!bits.impl || n_pairs <= 0) return out;

    const std::size_t np = static_cast<std::size_t>(n_pairs);
    DeviceBuffer<double> d_sum_p0(np), d_sum_p0_sq(np);
    DeviceBuffer<int> d_n_win_used(np);
    DeviceBuffer<long long> d_tot_comp(np);

    launch_readv2_mismatch(bits.impl->words.data(), bits.words_per_sample, bits.wpw,
                           bits.n_win, bits.n_samples, n_pairs, d_sum_p0.data(),
                           d_sum_p0_sq.data(), d_n_win_used.data(), d_tot_comp.data(), tiled,
                           stream_.get());

    out.sum_p0.assign(np, 0.0);
    out.sum_p0_sq.assign(np, 0.0);
    out.n_win_used.assign(np, 0);
    out.tot_comp.assign(np, 0);
    d2h_async(out.sum_p0.data(), d_sum_p0, np, stream_.get());
    d2h_async(out.sum_p0_sq.data(), d_sum_p0_sq, np, stream_.get());
    d2h_async(out.n_win_used.data(), d_n_win_used, np, stream_.get());
    d2h_async(out.tot_comp.data(), d_tot_comp, np, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

}  // namespace steppe::device
