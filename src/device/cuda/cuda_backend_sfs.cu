// src/device/cuda/cuda_backend_sfs.cu
//
// CudaBackend override for the 2D joint site-frequency spectrum (`steppe sfs`). It uploads
// the packed genotype tile device-resident, resolves the two population segments from the
// tile's pop_offsets, computes the per-axis category extents, launches the joint-histogram
// kernel over the device tile, and D2H's ONLY the finished (extA*extB)-cell integer grid +
// the complete-site counter — the joint-histogram accumulation is entirely on the GPU (no
// host genotype loop). A CUDA TU private to steppe_device, mirroring cuda_backend_fst.cu.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/internal/sfs_hist.hpp"   // sfs_axis_extent
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/sfs_hist_kernel.cuh"

namespace steppe::device {

SfsJoint CudaBackend::joint_sfs_2pop(const DecodeTileView& tile, int popA, int popB,
                                     bool folded) {
    guard_device();
    SfsJoint out;
    out.precision_tag = Precision::Kind::Fp64;
    out.folded = folded;

    const long M = static_cast<long>(tile.n_snp);
    const int P = tile.n_pop;
    if (M <= 0 || P <= 0 || popA < 0 || popB < 0 || popA >= P || popB >= P) {
        return out;
    }
    out.n_total = M;

    const std::size_t segA_begin = tile.pop_offsets[static_cast<std::size_t>(popA)];
    const std::size_t segA_end = tile.pop_offsets[static_cast<std::size_t>(popA) + 1];
    const std::size_t segB_begin = tile.pop_offsets[static_cast<std::size_t>(popB)];
    const std::size_t segB_end = tile.pop_offsets[static_cast<std::size_t>(popB) + 1];

    const long NA = static_cast<long>(segA_end - segA_begin);
    const long NB = static_cast<long>(segB_end - segB_begin);
    out.NA = NA;
    out.NB = NB;
    if (NA <= 0 || NB <= 0) return out;

    const long extA = core::sfs_axis_extent(NA, folded);
    const long extB = core::sfs_axis_extent(NB, folded);
    out.extA = extA;
    out.extB = extB;
    const long grid_len = extA * extB;

    // Upload the whole packed tile device-resident (individual-major, population-contiguous).
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
    if (packed_bytes > 0) {
        h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());
    }

    DeviceBuffer<unsigned long long> dGrid(static_cast<std::size_t>(grid_len));
    DeviceBuffer<long long> dNComplete(1);
    launch_sfs_hist_2pop(dPacked.data(), tile.bytes_per_record, segA_begin, segA_end, segB_begin,
                         segB_end, NA, NB, M, folded, extA, extB, dGrid.data(), grid_len,
                         dNComplete.data(), stream_.get());

    // D2H only the finished integer grid + the complete-site count.
    std::vector<unsigned long long> host_grid(static_cast<std::size_t>(grid_len), 0ULL);
    long long n_complete = 0;
    d2h_async(host_grid.data(), dGrid, static_cast<std::size_t>(grid_len), stream_.get());
    d2h_async(&n_complete, dNComplete, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    out.grid.assign(static_cast<std::size_t>(grid_len), 0);
    for (std::size_t c = 0; c < host_grid.size(); ++c) {
        out.grid[c] = static_cast<std::int64_t>(host_grid[c]);
    }
    out.n_complete = static_cast<long>(n_complete);
    out.n_dropped_incomplete = out.n_total - out.n_complete;
    return out;
}

}  // namespace steppe::device
