// src/device/cuda/cuda_backend_f2_blocks.cu
//
// CudaBackend's f2 block-tensor compute translation unit: the whole-matrix
// compute_f2 plus the resident / host-RAM / disk block-tensor family and its two
// engines. As the first out-of-line virtual override it is also the vtable
// key-function home for the CudaBackend class.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_f2_blocks.cu.md
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/internal/nvtx.hpp"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/block_sink.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/device_f2_blocks_impl.cuh"
#include "device/cuda/device_partial_impl.cuh"
#include "device/cuda/f2_batched_kernel.cuh"
#include "device/cuda/f2_block_kernel.cuh"
#include "device/f2_blocks_out.hpp"
#include "device/stream_f2_blocks.hpp"
#include "device/vram_budget.hpp"

namespace steppe::device {

namespace {

// Shared block-layout and size-bucketing helpers — reference §2
struct Bucket { int s_pad = 0; std::vector<int> blocks; };

struct BlockLayout {
    std::vector<long> block_offsets;
    std::vector<int>  block_sizes;
};

[[nodiscard]] BlockLayout compute_block_layout(const int* block_id, long M, int n_block) {
    BlockLayout out;
    out.block_offsets.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
    out.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
    const std::vector<core::BlockRange> ranges =
        core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                           M, n_block);
    for (int b = 0; b < n_block; ++b) {
        out.block_offsets[static_cast<std::size_t>(b)] = ranges[static_cast<std::size_t>(b)].begin;
        out.block_sizes[static_cast<std::size_t>(b)] =
            static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }
    return out;
}

[[nodiscard]] std::vector<Bucket> size_buckets(const std::vector<int>& block_sizes,
                                               int n_block) {
    auto ceil_bucket = [](int x) {
        int p = 1;
        while (p < x) p *= steppe::kBlockGroupPadBase;
        return p;
    };
    std::vector<Bucket> buckets;
    for (int b = 0; b < n_block; ++b) {
        const int sp = ceil_bucket(block_sizes[static_cast<std::size_t>(b)]);
        int gi = -1;
        for (std::size_t k = 0; k < buckets.size(); ++k)
            if (buckets[k].s_pad == sp) { gi = static_cast<int>(k); break; }
        if (gi < 0) { gi = static_cast<int>(buckets.size()); buckets.push_back({sp, {}}); }
        buckets[static_cast<std::size_t>(gi)].blocks.push_back(b);
    }
    std::sort(buckets.begin(), buckets.end(),
              [](const Bucket& a, const Bucket& c) { return a.s_pad < c.s_pad; });
    return buckets;
}

}  // namespace

// compute_f2 — a single f2 matrix over all SNPs — reference §3
F2Result CudaBackend::compute_f2(const core::MatView& Q,
                                  const core::MatView& V,
                                  const core::MatView& N,
                                  const Precision& precision) {
    guard_device();
    const int P = Q.P;
    const long M = Q.M;

    F2Result out;
    out.P = P;
    if (P <= 0 || M <= 0) return out;

    if (M > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "steppe::device::CudaBackend::compute_f2: SNP count M=" +
            std::to_string(M) + " exceeds INT_MAX (" +
            std::to_string(std::numeric_limits<int>::max()) +
            "); the M0 whole-matrix cublasGemmEx contraction extent k is a 32-bit "
            "int. Tile the SNP axis or route this path through cublasGemmEx_64 "
            "(cleanup B22).");
    }

    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t two_pm = 2u * pm;
    const std::size_t pp = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t two_pp = 2u * pp;

    DeviceBuffer<double> dQ_raw(pm), dV_raw(pm), dN_raw(pm);
    DeviceBuffer<double> dQ(pm), dV(pm), dS(two_pm);
    DeviceBuffer<double> dG(pp), dVpair(pp), dR(two_pp), dF2(pp);

    const std::size_t raw_bytes = pm * sizeof(double);
    pinned_in_.ensure(Q.data, raw_bytes);
    pinned_in_.ensure(V.data, raw_bytes);
    pinned_in_.ensure(N.data, raw_bytes);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, raw_bytes,
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, raw_bytes,
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, raw_bytes,
                                      cudaMemcpyHostToDevice, stream_.get()));

    launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                     dQ.data(), dV.data(), dS.data(), P, M, stream_.get());
    run_f2_gemms(blas_.get(), precision, P, M,
                 dQ.data(), dV.data(), dS.data(),
                 dG.data(), dVpair.data(), dR.data());
    launch_assemble_f2(dG.data(), dVpair.data(), dR.data(), dF2.data(), P, stream_.get());

    out.f2.resize(pp);
    out.vpair.resize(pp);
    d2h_async(out.f2.data(), dF2, pp, stream_.get());
    d2h_async(out.vpair.data(), dVpair, pp, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

// The block-tensor public entry points — reference §4
F2BlockTensor CudaBackend::compute_f2_blocks(const core::MatView& Q,
                                              const core::MatView& V,
                                              const core::MatView& N,
                                              const int* block_id,
                                              int n_block,
                                              const Precision& precision) {
    return compute_f2_blocks_device(Q, V, N, block_id, n_block, precision).to_host();
}

DeviceF2Blocks CudaBackend::compute_f2_blocks_device(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, const Precision& precision) {
    ResidentBlocks rb = run_f2_blocks_resident(Q, V, N, block_id, n_block, precision);
    DeviceF2Blocks h;
    h.P = rb.P;
    h.n_block = rb.n_block;
    h.device_id = device_id_;
    h.block_sizes = std::move(rb.block_sizes);
    if (h.n_block > 0 && h.P > 0) {
        h.impl = std::make_unique<DeviceF2Blocks::Impl>();
        h.impl->f2 = std::move(rb.f2);
        h.impl->vpair = std::move(rb.vpair);
    }
    return h;
}

DevicePartial CudaBackend::compute_f2_blocks_resident(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, int b0, const Precision& precision) {
    ResidentBlocks rb = run_f2_blocks_resident(Q, V, N, block_id, n_block, precision);
    DevicePartial h;
    h.P = rb.P;
    h.n_block_local = rb.n_block;
    h.b0 = b0;
    h.device_id = device_id_;
    h.block_sizes = std::move(rb.block_sizes);
    if (h.n_block_local > 0) {
        h.impl = std::make_unique<DevicePartial::Impl>();
        h.impl->f2 = std::move(rb.f2);
        h.impl->vpair = std::move(rb.vpair);
    }
    return h;
}

void CudaBackend::compute_f2_blocks_into(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, int b0,
    double* dst_f2, double* dst_vpair, int* block_sizes_dst,
    const Precision& precision) {
    ResidentBlocks rb = run_f2_blocks_resident(Q, V, N, block_id, n_block, precision);
    for (int b = 0; b < rb.n_block; ++b)
        block_sizes_dst[static_cast<std::size_t>(b0) + static_cast<std::size_t>(b)] =
            rb.block_sizes[static_cast<std::size_t>(b)];
    const std::size_t total = rb.f2.size();
    if (total > 0) {
        guard_device();
        const std::size_t slab_off =
            static_cast<std::size_t>(rb.P) * static_cast<std::size_t>(rb.P) *
            static_cast<std::size_t>(b0);
        const std::size_t bytes = total * sizeof(double);

        STEPPE_ASSERT(total == static_cast<std::size_t>(rb.P) *
                                   static_cast<std::size_t>(rb.P) *
                                   static_cast<std::size_t>(rb.n_block),
                      "compute_f2_blocks_into: pinned staging exceeds the "
                      "P*P*n_block partial bound");
        if (stage_f2_.size()    < total) stage_f2_    = PinnedBuffer<double>(total);
        if (stage_vpair_.size() < total) stage_vpair_ = PinnedBuffer<double>(total);

        STEPPE_CUDA_CHECK(cudaMemcpyAsync(stage_f2_.data(), rb.f2.data(), bytes,
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(stage_vpair_.data(), rb.vpair.data(), bytes,
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

        std::memcpy(dst_f2    + slab_off, stage_f2_.data(),    bytes);
        std::memcpy(dst_vpair + slab_off, stage_vpair_.data(), bytes);
    }
}

void CudaBackend::compute_f2_blocks_streamed(
    const core::MatView& Q, const core::MatView& V, const core::MatView& N,
    const int* block_id, int n_block, const Precision& precision,
    StreamTarget& target, const RedecodeSource* redecode) {
    if (target.tier == OutputTier::HostRam) {
        if (!target.host_dst)
            throw std::runtime_error("compute_f2_blocks_streamed: HostRam tier needs a "
                                     "host_dst destination");
        HostRamSink sink(*target.host_dst);
        stream_f2_blocks_impl(Q, V, N, block_id, n_block, precision, sink, redecode);
    } else if (target.tier == OutputTier::Disk) {
        if (!target.disk_dst)
            throw std::runtime_error("compute_f2_blocks_streamed: Disk tier needs a "
                                     "disk_dst descriptor");
        DiskSink sink(target.disk_path);
        stream_f2_blocks_impl(Q, V, N, block_id, n_block, precision, sink, redecode);
        sink.take_descriptor(*target.disk_dst);
    } else {
        throw std::runtime_error("compute_f2_blocks_streamed: Resident tier must not "
                                 "route through the stream seam (it bypasses it)");
    }
}

// run_f2_blocks_resident — the in-VRAM block engine — reference §5
CudaBackend::ResidentBlocks CudaBackend::run_f2_blocks_resident(const core::MatView& Q,
                                                    const core::MatView& V,
                                                    const core::MatView& N,
                                                    const int* block_id,
                                                    int n_block,
                                                    const Precision& precision) {
    guard_device();
    STEPPE_NVTX_RANGE("f2_gemm");
    const int P = Q.P;
    const long M = Q.M;

    ResidentBlocks rb;
    rb.P = P;
    rb.n_block = n_block;
    const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total = slab * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
    rb.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
    if (P <= 0 || M <= 0 || n_block <= 0) return rb;

    const BlockLayout layout = compute_block_layout(block_id, M, n_block);
    const std::vector<long>& block_offsets = layout.block_offsets;
    rb.block_sizes = layout.block_sizes;
    const std::vector<Bucket> buckets = size_buckets(rb.block_sizes, n_block);

    std::size_t free_b = 0, total_b = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t two_pm = 2u * pm;
    DeviceBuffer<double> dQ(pm), dV(pm), dS(two_pm);
    rb.f2 = DeviceBuffer<double>(total);
    rb.vpair = DeviceBuffer<double>(total);
    DeviceBuffer<long>   dOffsets(static_cast<std::size_t>(n_block));
    DeviceBuffer<int>    dSizes(static_cast<std::size_t>(n_block));

    h2d_async(dOffsets, block_offsets.data(),
              static_cast<std::size_t>(n_block), stream_.get());
    h2d_async(dSizes, rb.block_sizes.data(),
              static_cast<std::size_t>(n_block), stream_.get());

    {
        DeviceBuffer<double> dQ_raw(pm), dV_raw(pm), dN_raw(pm);
        const std::size_t raw_bytes = pm * sizeof(double);
        pinned_in_.ensure(Q.data, raw_bytes);
        pinned_in_.ensure(V.data, raw_bytes);
        pinned_in_.ensure(N.data, raw_bytes);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, raw_bytes,
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, raw_bytes,
                                          cudaMemcpyHostToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, raw_bytes,
                                          cudaMemcpyHostToDevice, stream_.get()));
        launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                         dQ.data(), dV.data(), dS.data(), P, M, stream_.get());
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }

    engage_f2_precision(blas_.get(), precision);

    std::size_t max_nb = 0;
    std::size_t max_psp_nb = 0;
    std::vector<int> bucket_max_blocks(buckets.size(), 0);
    for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
        const int s_pad = buckets[bi].s_pad;
        const int nb_total = static_cast<int>(buckets[bi].blocks.size());
        const int mb = max_blocks_per_chunk(free_b, P, n_block, s_pad, nb_total);
        bucket_max_blocks[bi] = mb;
        const std::size_t nb = static_cast<std::size_t>(mb < 0 ? 0 : mb);
        max_nb = std::max(max_nb, nb);
        max_psp_nb = std::max(
            max_psp_nb,
            static_cast<std::size_t>(P) * static_cast<std::size_t>(s_pad) * nb);
    }
    DeviceBuffer<int>    dIds(max_nb);
    DeviceBuffer<double> dQg(max_psp_nb), dVg(max_psp_nb), dSg(2u * max_psp_nb);
    const std::size_t max_pp_nb = slab * max_nb;
    DeviceBuffer<double> dGg(max_pp_nb), dVpairg(max_pp_nb), dRg(2u * max_pp_nb);

    for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
        const Bucket& bk = buckets[bi];
        const int s_pad = bk.s_pad;
        const int nb_total = static_cast<int>(bk.blocks.size());
        const int max_blocks = bucket_max_blocks[bi];

        for (int start = 0; start < nb_total; start += max_blocks) {
            const int nb = std::min(max_blocks, nb_total - start);

            h2d_async(dIds, bk.blocks.data() + start,
                      static_cast<std::size_t>(nb), stream_.get());

            launch_gather_group(dQ.data(), dV.data(), dS.data(),
                                dIds.data(), dOffsets.data(), dSizes.data(),
                                P, s_pad, nb, dQg.data(), dVg.data(), dSg.data(), stream_.get());
            run_f2_gemms_group(blas_.get(), precision, P, s_pad, nb,
                               dQg.data(), dVg.data(), dSg.data(),
                               dGg.data(), dVpairg.data(), dRg.data());
            launch_assemble_blocks_group(dGg.data(), dVpairg.data(), dRg.data(),
                                         dIds.data(), P, nb,
                                         rb.f2.data(), rb.vpair.data(), stream_.get());
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        }
    }

    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return rb;
}

// stream_f2_blocks_impl — the spill-to-RAM-or-disk engine — reference §6
void CudaBackend::stream_f2_blocks_impl(const core::MatView& Q, const core::MatView& V,
                           const core::MatView& N, const int* block_id, int n_block,
                           const Precision& precision, BlockSink& sink,
                           const RedecodeSource* redecode) {
    guard_device();
    const int P = Q.P;
    const long M = Q.M;
    const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

    std::vector<int> block_sizes(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
    if (P <= 0 || M <= 0 || n_block <= 0) {
        sink.begin(P, n_block, block_sizes);
        sink.finish();
        return;
    }
    const BlockLayout layout = compute_block_layout(block_id, M, n_block);
    const std::vector<long>& block_offsets = layout.block_offsets;
    block_sizes = layout.block_sizes;
    const std::vector<Bucket> buckets = size_buckets(block_sizes, n_block);

    sink.begin(P, n_block, block_sizes);

    std::size_t free_b = 0, total_b = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    DeviceBuffer<int>    dSizes(static_cast<std::size_t>(n_block));
    h2d_async(dSizes, block_sizes.data(),
              static_cast<std::size_t>(n_block), stream_.get());
    // In re-decode mode Q/V/N.data is null (the dense host tensor is never built), so
    // there is nothing to pin — the per-chunk tile is produced on-device instead.
    if (redecode == nullptr) {
        const std::size_t raw_bytes = pm * sizeof(double);
        pinned_in_.ensure(Q.data, raw_bytes);
        pinned_in_.ensure(V.data, raw_bytes);
        pinned_in_.ensure(N.data, raw_bytes);
    }
    engage_f2_precision(blas_.get(), precision);

    const std::size_t net_free_b =
        (free_b > kCublasWorkspaceBytes) ? (free_b - kCublasWorkspaceBytes) : 0u;
    const std::size_t envelope_b = static_cast<std::size_t>(
        kMaxVramUtilizationFraction * static_cast<double>(net_free_b));
    const std::size_t tile_budget_b = static_cast<std::size_t>(
        kStreamTileBudgetFraction * static_cast<double>(envelope_b));
    const std::size_t slab_budget_b = envelope_b - tile_budget_b;
    const std::size_t per_col_feeder_b =
        7u * static_cast<std::size_t>(P) * sizeof(double);
    const long max_tile_cols = static_cast<long>(std::max<std::size_t>(
        per_col_feeder_b > 0u ? (tile_budget_b / per_col_feeder_b) : 1u, 1u));

    auto streamed_max_blocks = [&](int s_pad, int nb_total) -> int {
        if (nb_total <= 0) return 0;
        const std::size_t p = static_cast<std::size_t>(P);
        const std::size_t sp = static_cast<std::size_t>(s_pad < 0 ? 0 : s_pad);
        const std::size_t per_block_b =
            (4u * p * sp + 8u * p * p) * sizeof(double);
        const std::size_t fit = (per_block_b > 0u) ? (slab_budget_b / per_block_b) : 0u;
        const std::size_t capped = std::min({fit, static_cast<std::size_t>(nb_total),
                                             static_cast<std::size_t>(core::kMaxGridZ)});
        return static_cast<int>(std::max<std::size_t>(capped, 1u));
    };

    auto chunk_extent = [&](const Bucket& bk, int start, int max_blocks,
                            long& s_lo, long& s_hi) -> int {
        const int nb_total = static_cast<int>(bk.blocks.size());
        const int cap = std::min(max_blocks, nb_total - start);
        int nb = 0;
        s_lo = 0;
        s_hi = 0;
        for (int kk = 0; kk < cap; ++kk) {
            const int gid = bk.blocks[static_cast<std::size_t>(start + kk)];
            const long b_lo = block_offsets[static_cast<std::size_t>(gid)];
            const long b_hi = b_lo + block_sizes[static_cast<std::size_t>(gid)];
            const long new_lo = (nb == 0) ? b_lo : std::min(s_lo, b_lo);
            const long new_hi = (nb == 0) ? b_hi : std::max(s_hi, b_hi);
            if (nb > 0 && (new_hi - new_lo) > max_tile_cols) break;
            s_lo = new_lo;
            s_hi = new_hi;
            ++nb;
        }
        return nb;
    };

    std::size_t max_nb = 0;
    std::size_t max_psp_nb = 0;
    long        max_tile = 0;
    std::vector<int> bucket_max_blocks(buckets.size(), 0);
    for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
        const int s_pad = buckets[bi].s_pad;
        const int nb_total = static_cast<int>(buckets[bi].blocks.size());
        const int mb = streamed_max_blocks(s_pad, nb_total);
        bucket_max_blocks[bi] = mb;
        for (int start = 0; start < nb_total; ) {
            long s_lo = 0, s_hi = 0;
            const int nb = chunk_extent(buckets[bi], start, mb, s_lo, s_hi);
            const std::size_t nbz = static_cast<std::size_t>(nb < 0 ? 0 : nb);
            max_nb = std::max(max_nb, nbz);
            max_psp_nb = std::max(
                max_psp_nb,
                static_cast<std::size_t>(P) * static_cast<std::size_t>(s_pad) * nbz);
            max_tile = std::max(max_tile, s_hi - s_lo);
            start += (nb > 0 ? nb : 1);
        }
    }
    DeviceBuffer<double> dQg(max_psp_nb), dVg(max_psp_nb), dSg(2u * max_psp_nb);
    const std::size_t max_pp_nb = slab * max_nb;
    DeviceBuffer<double> dGg(max_pp_nb), dVpairg(max_pp_nb), dRg(2u * max_pp_nb);

    const std::size_t max_tile_z = static_cast<std::size_t>(max_tile < 0 ? 0 : max_tile);
    const std::size_t p_tile = static_cast<std::size_t>(P) * max_tile_z;
    DeviceBuffer<double> dQ_raw(p_tile), dV_raw(p_tile), dN_raw(p_tile);
    DeviceBuffer<double> dQt(p_tile), dVt(p_tile), dSt(2u * p_tile);
    DeviceBuffer<long> dOffsetsTile(max_nb);
    DeviceBuffer<int>  dSizesTile(max_nb);
    std::vector<long>  h_offsets_tile(max_nb);
    std::vector<int>   h_sizes_tile(max_nb);

    std::vector<int> local_ids(max_nb);
    for (std::size_t k = 0; k < max_nb; ++k) local_ids[k] = static_cast<int>(k);
    DeviceBuffer<int> dIdsLocal(max_nb);
    if (max_nb > 0)
        h2d_async(dIdsLocal, local_ids.data(), max_nb, stream_.get());

    struct Ring {
        DeviceBuffer<double> f2;
        DeviceBuffer<double> vpair;
        Event reuse;
        bool used = false;
    };
    Ring ring[kStreamDeviceChunks];
    for (Ring& r : ring) {
        r.f2 = DeviceBuffer<double>(max_pp_nb);
        r.vpair = DeviceBuffer<double>(max_pp_nb);
    }

    int chunk_idx = 0;
    for (std::size_t bi = 0; bi < buckets.size(); ++bi) {
        const Bucket& bk = buckets[bi];
        const int s_pad = bk.s_pad;
        const int nb_total = static_cast<int>(bk.blocks.size());
        const int max_blocks = bucket_max_blocks[bi];

        for (int start = 0; start < nb_total; ) {
            long s_lo = 0, s_hi = 0;
            const int nb = chunk_extent(bk, start, max_blocks, s_lo, s_hi);
            const long tile = s_hi - s_lo;
            Ring& r = ring[chunk_idx % kStreamDeviceChunks];
            if (r.used)
                STEPPE_CUDA_CHECK(cudaEventSynchronize(r.reuse.get()));

            const std::size_t tile_elems =
                static_cast<std::size_t>(P) * static_cast<std::size_t>(tile);
            const std::size_t tile_bytes = tile_elems * sizeof(double);
            if (redecode == nullptr) {
                // Default path: copy the tile [s_lo, s_hi) straight out of the dense
                // host Q/V/N (identical to every non-extract streamed caller).
                const std::size_t src_off = static_cast<std::size_t>(P) *
                                            static_cast<std::size_t>(s_lo);
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data + src_off, tile_bytes,
                                                  cudaMemcpyHostToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data + src_off, tile_bytes,
                                                  cudaMemcpyHostToDevice, stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data + src_off, tile_bytes,
                                                  cudaMemcpyHostToDevice, stream_.get()));
            } else {
                // Re-decode path (extract-f2 host-RAM-wall fix): fill the SAME dQ_raw/
                // dV_raw/dN_raw tile [s_lo, s_hi) by re-decoding those kept columns from
                // the packed genotypes instead of reading a dense host tensor that was
                // never materialized. The decode is deterministic, so these bytes are
                // identical to the H2D branch and the f2 result is bit-identical.
                // decode_af_compact_filter requires a 4-aligned raw start, so we decode a
                // slightly wider window [aligned_lo, aligned_lo+n_snp) and skip the 0..3
                // kept columns that land ahead of s_lo. Reference §6a.
                const long raw_lo = redecode->kept_to_raw[static_cast<std::size_t>(s_lo)];
                const long raw_hi =
                    redecode->kept_to_raw[static_cast<std::size_t>(s_hi - 1)] + 1;
                const long aligned_lo = raw_lo & ~3L;
                const long first_kept = static_cast<long>(
                    std::lower_bound(redecode->kept_to_raw,
                                     redecode->kept_to_raw + redecode->M_kept, aligned_lo) -
                    redecode->kept_to_raw);
                const long head = s_lo - first_kept;  // 0..3 kept cols decoded before s_lo
                const long n_snp = raw_hi - aligned_lo;
                const std::size_t alo = static_cast<std::size_t>(aligned_lo);
                const std::size_t ns = static_cast<std::size_t>(n_snp);
                DecodeTileView tview = *redecode->base_view;
                tview.n_snp = ns;
                DeviceDecodeResult ddr = decode_af_compact_filter(
                    tview,
                    std::span<const char>(redecode->ref + alo, ns),
                    std::span<const char>(redecode->alt + alo, ns),
                    std::span<const int>(redecode->chrom + alo, ns),
                    std::span<const double>(redecode->genpos + alo, ns),
                    std::span<const double>(redecode->physpos + alo, ns),
                    *redecode->filter,
                    std::span<const std::size_t>(redecode->pop_individuals, redecode->n_pop),
                    core::kPloidyDiploid, redecode->maxmiss, aligned_lo);
                STEPPE_ASSERT(ddr.M_kept == s_hi - first_kept,
                              "stream_f2_blocks_impl: re-decoded kept-set does not match "
                              "Phase A (extract-f2 decode desync across passes)");
                const std::size_t head_off =
                    static_cast<std::size_t>(head) * static_cast<std::size_t>(P);
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), ddr.q_device() + head_off,
                                                  tile_bytes, cudaMemcpyDeviceToDevice,
                                                  stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), ddr.v_device() + head_off,
                                                  tile_bytes, cudaMemcpyDeviceToDevice,
                                                  stream_.get()));
                STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), ddr.n_device() + head_off,
                                                  tile_bytes, cudaMemcpyDeviceToDevice,
                                                  stream_.get()));
                // ddr's device buffers free at scope end — the async D2D must complete
                // before that (avoid use-after-free of ddr's Q/V/N).
                STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
            }
            launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                             dQt.data(), dVt.data(), dSt.data(), P, tile, stream_.get());

            for (int kk = 0; kk < nb; ++kk) {
                const int gid = bk.blocks[static_cast<std::size_t>(start + kk)];
                h_offsets_tile[static_cast<std::size_t>(kk)] =
                    block_offsets[static_cast<std::size_t>(gid)] - s_lo;
                h_sizes_tile[static_cast<std::size_t>(kk)] =
                    block_sizes[static_cast<std::size_t>(gid)];
            }
            h2d_async(dOffsetsTile, h_offsets_tile.data(),
                      static_cast<std::size_t>(nb), stream_.get());
            h2d_async(dSizesTile, h_sizes_tile.data(),
                      static_cast<std::size_t>(nb), stream_.get());
            launch_gather_group(dQt.data(), dVt.data(), dSt.data(),
                                dIdsLocal.data(), dOffsetsTile.data(), dSizesTile.data(),
                                P, s_pad, nb, dQg.data(), dVg.data(), dSg.data(), stream_.get());
            run_f2_gemms_group(blas_.get(), precision, P, s_pad, nb,
                               dQg.data(), dVg.data(), dSg.data(),
                               dGg.data(), dVpairg.data(), dRg.data());
            launch_assemble_blocks_group(dGg.data(), dVpairg.data(), dRg.data(),
                                         dIdsLocal.data(), P, nb,
                                         r.f2.data(), r.vpair.data(), stream_.get());

            for (int kk = 0; kk < nb; ++kk) {
                const int gid = bk.blocks[static_cast<std::size_t>(start + kk)];
                const std::size_t off = slab * static_cast<std::size_t>(kk);
                sink.spill_block(gid, r.f2.data() + off, r.vpair.data() + off,
                                 slab, stream_.get());
            }
            r.reuse.record(stream_);
            r.used = true;
            ++chunk_idx;
            start += (nb > 0 ? nb : 1);
        }
    }

    sink.finish();
}

}  // namespace steppe::device
