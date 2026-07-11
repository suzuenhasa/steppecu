// src/device/cuda/cuda_backend_decode.cu
//
// CudaBackend genotype-decode front-end: the out-of-line method bodies that turn
// packed genotype bytes into resident Q/V/N (decode, per-sample ploidy detection,
// SNP-major to canonical transpose, and the two device-resident compact-decode
// regimes). A CUDA TU private to steppe_device, so it inherits that library's codegen.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_decode.cu.md
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/internal/host_device.hpp"
#include "core/internal/nvtx.hpp"
#include "io/filter/snp_filter.hpp"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/decode_af_kernel.cuh"
#include "device/cuda/decode_compact_kernel.cuh"
#include "device/cuda/detect_ploidy_kernel.cuh"
#include "device/cuda/device_decode_result_impl.cuh"
#include "device/cuda/transpose_canonical_kernel.cuh"

namespace steppe::device {

// Shared decode front-end: packed genotypes to resident Q/V/N — reference §3
void CudaBackend::decode_af_resident(const DecodeTileView& tile, int P, long M,
                        DeviceBuffer<double>& dQ, DeviceBuffer<double>& dV,
                        DeviceBuffer<double>& dN, long s_lo) {
    STEPPE_ASSERT(s_lo % core::kCodesPerByte == 0,
                  "decode_af_resident: s_lo must be a multiple of kCodesPerByte (4) "
                  "so the packed 2-bit tile slice stays byte-aligned");
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t tile_width =
        (static_cast<std::size_t>(M) + cpb - 1u) / cpb;
    const std::size_t packed_bytes = tile.n_individuals * tile_width;
    const std::size_t n_off = static_cast<std::size_t>(P) + 1u;
    DeviceBuffer<std::size_t> dOffsets(n_off);

    const bool explicit_sample_ploidy = (tile.sample_ploidy != nullptr);
    const bool device_detect =
        (!explicit_sample_ploidy && tile.detect_ploidy_on_device);
    const bool have_sample_ploidy = explicit_sample_ploidy || device_detect;
    DeviceBuffer<int> dSamplePloidy(have_sample_ploidy ? tile.n_individuals : 0u);

    // Source pane: the M-column packed slice at s_lo, decoded at pitch tile_width. When the
    // canonical tile is already device-resident (GPU-native load) AND this is the whole-width
    // slice (M == n_snp, s_lo == 0), the resident tile ALREADY IS that pane at exactly this
    // pitch, so decode it in place — no copy. Otherwise stage the pitched slice: device->device
    // when resident (a sub-SNP window of a resident tile, e.g. the streamed f2 arm), host->device
    // for the legacy host tile. The staged bits are identical to the resident bits either way.
    DeviceBuffer<std::uint8_t> dPacked;
    const std::uint8_t* packed_src = nullptr;
    if (tile.packed_on_device && tile_width == tile.bytes_per_record && s_lo == 0) {
        packed_src = tile.packed;
    } else {
        dPacked = DeviceBuffer<std::uint8_t>(packed_bytes == 0 ? 1u : packed_bytes);
        STEPPE_CUDA_CHECK(cudaMemcpy2DAsync(
            dPacked.data(), tile_width,
            tile.packed + static_cast<std::size_t>(s_lo) / cpb, tile.bytes_per_record,
            tile_width, tile.n_individuals,
            tile.packed_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
            stream_.get()));
        packed_src = dPacked.data();
    }
    h2d_async(dOffsets, tile.pop_offsets, n_off, stream_.get());
    if (explicit_sample_ploidy) {
        h2d_async(dSamplePloidy, tile.sample_ploidy, tile.n_individuals,
                  stream_.get());
    } else if (device_detect) {
        launch_detect_ploidy(packed_src, tile_width,
                             tile.n_individuals, tile.n_snp, dSamplePloidy.data(),
                             stream_.get());
    }

    launch_decode_af(packed_src, tile_width, dOffsets.data(),
                     P, M, tile.ploidy,
                     have_sample_ploidy ? dSamplePloidy.data() : nullptr,
                     dQ.data(), dV.data(), dN.data(), stream_.get());
}

// Host-oracle decode with full copy-back — reference §5
DecodeResult CudaBackend::decode_af(const DecodeTileView& tile) {
    guard_device();
    STEPPE_NVTX_RANGE("decode");
    const int P = tile.n_pop;
    const long M = static_cast<long>(tile.n_snp);

    DecodeResult out;
    out.P = P;
    out.M = M;
    const std::size_t pm =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    out.q.assign(pm, 0.0);
    out.v.assign(pm, 0.0);
    out.n.assign(pm, 0.0);
    if (P <= 0 || M <= 0) return out;

    DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
    decode_af_resident(tile, P, M, dQ, dV, dN);
    d2h_async(out.q.data(), dQ, pm, stream_.get());
    d2h_async(out.v.data(), dV, pm, stream_.get());
    d2h_async(out.n.data(), dN, pm, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

// Device-resident pooled-per-SNP QC summary — reference §5b.
// Decodes the tile to resident Q/N (never copied back), reduces each SNP across the P
// populations ON THE DEVICE with the shared derive_pooled_summary_one, and D2Hs only the
// four O(M) summary planes. Host traffic drops from 3*P*M doubles (decode_af) to 4*M — a
// P-fold cut that erases the singleton-P (KING/kinship, P == N) single-core decode wall.
// Bit-for-bit identical to the decode_af + host-reduction default: same resident Q/N bits,
// same STEPPE_HD reduction (p = 0..P-1 order, FMA-safe pooled_ref_fma).
std::vector<steppe::io::filter::PerSnpSummary> CudaBackend::decode_af_pooled_summary(
    const DecodeTileView& tile, double ploidy_d, double total_indiv_d) {
    guard_device();
    STEPPE_NVTX_RANGE("decode_summary");
    const int P = tile.n_pop;
    const long M = static_cast<long>(tile.n_snp);
    std::vector<steppe::io::filter::PerSnpSummary> summary(
        static_cast<std::size_t>(M < 0 ? 0 : M));
    if (P <= 0 || M <= 0) return summary;

    const std::size_t pm =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t Mz = static_cast<std::size_t>(M);
    DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
    decode_af_resident(tile, P, M, dQ, dV, dN);

    DeviceBuffer<double> dRefAf(Mz), dMinorAf(Mz), dMissing(Mz), dAlleleCount(Mz);
    launch_pooled_summary(dQ.data(), dN.data(), P, M, ploidy_d, total_indiv_d,
                          dRefAf.data(), dMinorAf.data(), dMissing.data(),
                          dAlleleCount.data(), stream_.get());

    std::vector<double> ref_af(Mz), minor_af(Mz), missing(Mz), allele_count(Mz);
    d2h_async(ref_af.data(), dRefAf, Mz, stream_.get());
    d2h_async(minor_af.data(), dMinorAf, Mz, stream_.get());
    d2h_async(missing.data(), dMissing, Mz, stream_.get());
    d2h_async(allele_count.data(), dAlleleCount, Mz, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    for (std::size_t s = 0; s < Mz; ++s) {
        steppe::io::filter::PerSnpSummary& sm = summary[s];
        sm.pooled_ref_af = ref_af[s];
        sm.pooled_minor_af = minor_af[s];
        sm.missing_frac = missing[s];
        sm.pooled_allele_count = allele_count[s];
    }
    return summary;
}

// Standalone on-device per-sample ploidy detection — reference §6
std::vector<int> CudaBackend::detect_sample_ploidy_device(
    const DecodeTileView& tile) {
    guard_device();
    std::vector<int> out(tile.n_individuals, core::kPloidyPseudoHaploid);
    if (tile.n_individuals == 0) return out;
    DeviceBuffer<std::uint8_t> dPacked;  // staging; empty when the tile is device-resident
    const std::uint8_t* packed_dev = packed_device_ptr(tile, dPacked);
    DeviceBuffer<int> dPloidy(tile.n_individuals);
    launch_detect_ploidy(packed_dev, tile.bytes_per_record, tile.n_individuals,
                         tile.n_snp, dPloidy.data(), stream_.get());
    d2h_async(out.data(), dPloidy, tile.n_individuals, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

// SNP-major to canonical individual-major transpose — reference §7
CanonicalTile CudaBackend::transpose_to_canonical(
    const SnpMajorTileView& view) {
    guard_device();
    CanonicalTile out;
    out.n_snp = view.n_snp;
    out.n_individuals = view.n_individuals;
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    out.bytes_per_record = view.n_snp == 0 ? 0 : (view.n_snp + cpb - 1) / cpb;
    if (view.pop_offsets != nullptr && view.n_pop >= 0) {
        out.pop_offsets.assign(
            view.pop_offsets,
            view.pop_offsets + static_cast<std::size_t>(view.n_pop) + 1u);
    }
    const std::size_t out_total = view.n_individuals * out.bytes_per_record;
    out.packed.assign(out_total, std::uint8_t{0});
    if (out_total == 0) return out;

    const std::size_t src_total = view.n_snp * view.src_bytes_per_record;
    DeviceBuffer<std::uint8_t> dSrc(src_total);
    DeviceBuffer<std::size_t> dSel(view.n_individuals);
    DeviceBuffer<std::uint8_t> dOut(out_total);
    h2d_async(dSrc, view.snp_major, src_total, stream_.get());
    h2d_async(dSel, view.sel_rows, view.n_individuals, stream_.get());

    TransposeEncoding enc = TransposeEncoding::Identity;
    switch (view.encoding) {
        case TileEncoding::Identity:
        default:
            enc = TransposeEncoding::Identity;
            break;
    }
    launch_transpose_to_canonical(dSrc.data(), view.src_bytes_per_record,
                                  dSel.data(), view.n_individuals, view.n_snp,
                                  out.bytes_per_record, enc, dOut.data(),
                                  stream_.get());
    d2h_async(out.packed.data(), dOut, out_total, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

namespace {

// TransposeEncoding from the CUDA-free TileEncoding (shared by the device-resident builders).
[[nodiscard]] TransposeEncoding transpose_encoding_of(TileEncoding e) {
    switch (e) {
        case TileEncoding::Identity:
        default:
            return TransposeEncoding::Identity;
    }
}

}  // namespace

// packed_device_ptr — GPU-native-load seam. Resident tile -> use in place (no upload); host
// tile -> stage into `staging` with the SAME (packed_bytes==0?1u) guard + h2d the legacy path
// used, so the host branch is byte-for-byte the prior behavior. Reference: §5.
const std::uint8_t* CudaBackend::packed_device_ptr(const DecodeTileView& tile,
                                                   DeviceBuffer<std::uint8_t>& staging) {
    if (tile.packed_on_device) return tile.packed;
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    staging = DeviceBuffer<std::uint8_t>(packed_bytes == 0 ? 1u : packed_bytes);
    if (packed_bytes > 0) h2d_async(staging, tile.packed, packed_bytes, stream_.get());
    return staging.data();
}

// alloc_canonical_device — a ZEROED device-resident canonical tile + its descriptor. Zeroing
// guarantees byte-identity with the host path even for a would-be-unwritten byte (the streamed
// scatter writes every output byte exactly once, so this is belt-and-suspenders). Reference: §7.
DeviceGenotypeTile CudaBackend::alloc_canonical_device(
    std::size_t n_individuals, std::size_t out_bytes_per_record, std::size_t n_snp,
    std::vector<std::size_t> pop_offsets, std::vector<std::string> pop_labels) {
    guard_device();
    DeviceGenotypeTile out;
    out.bytes_per_record = out_bytes_per_record;
    out.n_snp = n_snp;
    out.n_individuals = n_individuals;
    out.pop_offsets = std::move(pop_offsets);
    out.pop_labels = std::move(pop_labels);
    const std::size_t total = n_individuals * out_bytes_per_record;
    auto buf = std::make_shared<DeviceBuffer<std::uint8_t>>(total);
    if (total > 0) {
        STEPPE_CUDA_CHECK(cudaMemsetAsync(buf->data(), 0, total, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }
    out.packed = buf->data();
    out.owner = std::move(buf);
    return out;
}

// transpose_block_into_canonical_device — transpose+gather+encode ONE SNP-major block on the
// GPU (raw block bytes -> H2D -> the SHARED transpose kernel) and scatter its canonical byte-
// columns into `dst` at output byte-column col_off. The kernel writes straight into the resident
// tile (dst_row_stride = dst.bytes_per_record, dst_col_off = col_off) — no host reassembly, no
// D2H. Byte-identical to the one-shot transpose because the kernel + inputs are the same and the
// blocks' byte-columns tile [0, out_bpr) exactly. Reference: §7.
void CudaBackend::transpose_block_into_canonical_device(
    DeviceGenotypeTile& dst, std::size_t col_off, const SnpMajorTileView& block) {
    guard_device();
    if (block.n_snp == 0 || block.n_individuals == 0) return;
    auto* buf = static_cast<DeviceBuffer<std::uint8_t>*>(dst.owner.get());
    std::uint8_t* d_out = buf->data();

    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t block_out_bpr = (block.n_snp + cpb - 1) / cpb;
    const std::size_t src_total = block.n_snp * block.src_bytes_per_record;

    DeviceBuffer<std::uint8_t> dSrc(src_total == 0 ? 1u : src_total);
    DeviceBuffer<std::size_t> dSel(block.n_individuals);
    if (src_total > 0) h2d_async(dSrc, block.snp_major, src_total, stream_.get());
    h2d_async(dSel, block.sel_rows, block.n_individuals, stream_.get());

    launch_transpose_to_canonical(dSrc.data(), block.src_bytes_per_record, dSel.data(),
                                  block.n_individuals, block.n_snp, block_out_bpr,
                                  transpose_encoding_of(block.encoding), d_out,
                                  stream_.get(), dst.bytes_per_record, col_off);
    // Sync before dSrc/dSel free (per-block staging); the raw-bytes H2D is the one legit copy.
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
}

// upload_canonical_device — H2D an already-canonical host tile (TGENO individual-major) ONCE
// into a device-resident tile: the ONE legitimate whole-matrix copy for that arm. Reference: §7.
DeviceGenotypeTile CudaBackend::upload_canonical_device(
    const std::uint8_t* host_packed, std::size_t bytes_per_record, std::size_t n_snp,
    std::size_t n_individuals, std::vector<std::size_t> pop_offsets,
    std::vector<std::string> pop_labels) {
    guard_device();
    DeviceGenotypeTile out;
    out.bytes_per_record = bytes_per_record;
    out.n_snp = n_snp;
    out.n_individuals = n_individuals;
    out.pop_offsets = std::move(pop_offsets);
    out.pop_labels = std::move(pop_labels);
    const std::size_t total = n_individuals * bytes_per_record;
    auto buf = std::make_shared<DeviceBuffer<std::uint8_t>>(total);
    if (total > 0) {
        h2d_async(*buf, host_packed, total, stream_.get());
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }
    out.packed = buf->data();
    out.owner = std::move(buf);
    return out;
}

// compact_tile_columns_device — device-resident SNP-column compaction for apply_snp_filter.
// H2D the host QC keep mask, cub::DeviceSelect::Flagged the ascending kept source-column list off a
// counting iterator (== repack_tile_columns' kept_cols), alloc a zeroed compacted resident tile,
// and gather its 2-bit packed columns in place — no host materialize / single-core gather /
// re-upload. Byte-for-byte identical to repack_tile_columns (apply_snp_filter host path).
DeviceGenotypeTile CudaBackend::compact_tile_columns_device(
    const DeviceGenotypeTile& src, const std::vector<std::uint8_t>& keep_flags) {
    guard_device();
    STEPPE_NVTX_RANGE("compact_tile");
    const long M = static_cast<long>(src.n_snp);
    const std::size_t Mz = static_cast<std::size_t>(M < 0 ? 0 : M);

    // H2D the host QC keep mask (the SAME mask repack_tile_columns consumes -> identical kept set).
    DeviceBuffer<std::uint8_t> dFlags(Mz == 0 ? 1u : Mz);
    if (Mz > 0) h2d_async(dFlags, keep_flags.data(), Mz, stream_.get());

    // cub::DeviceSelect::Flagged over the iota [0, M) by dFlags -> the ascending kept source-column
    // list dKeptCols (output column -> source column; stable select == kept_cols) + the count. Same
    // sizing-then-run idiom as regime B (:502-531), fed a counting iterator so no M-length iota is
    // materialized.
    DeviceBuffer<long> dKeptCols(Mz == 0 ? 1u : Mz);
    DeviceBuffer<int> dNumSel(1);
    {
        auto iota = thrust::make_counting_iterator<long>(0);
        std::size_t sel_bytes = 0;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, sel_bytes, iota, dFlags.data(), dKeptCols.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        DeviceBuffer<unsigned char> dSelTemp(sel_bytes == 0 ? 1 : sel_bytes);
        std::size_t sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            dSelTemp.data(), sb, iota, dFlags.data(), dKeptCols.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
    }
    int m_kept = 0;
    d2h_async(&m_kept, dNumSel, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // Allocate the ZEROED compacted device tile (newbpr = ceil(n_kept/4)); the descriptor carries
    // the unchanged individuals/pops (only the SNP axis shrinks).
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t newbpr =
        m_kept <= 0 ? 0 : (static_cast<std::size_t>(m_kept) + cpb - 1u) / cpb;
    DeviceGenotypeTile out = alloc_canonical_device(
        src.n_individuals, newbpr, static_cast<std::size_t>(m_kept < 0 ? 0 : m_kept),
        src.pop_offsets, src.pop_labels);
    if (m_kept <= 0 || src.n_individuals == 0 || newbpr == 0) return out;

    // 2-bit packed-column gather: one thread per (row, out-byte), reading src.packed in place and
    // writing each output byte exactly once (no atomics) -> byte-identical to repack_tile_columns.
    auto* buf = static_cast<DeviceBuffer<std::uint8_t>*>(out.owner.get());
    launch_compact_packed_columns(src.packed, src.bytes_per_record, dKeptCols.data(),
                                  static_cast<long>(m_kept), buf->data(), newbpr,
                                  src.n_individuals, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

// Compact-decode regime A: autosome keep-mask — reference §8
steppe::device::DeviceDecodeResult CudaBackend::decode_af_compact_autosome(
    const DecodeTileView& tile, std::span<const int> chrom,
    std::span<const double> genpos, std::span<const double> physpos,
    int chrom_min, int chrom_max) {
    guard_device();
    steppe::device::DeviceDecodeResult out;
    out.device_id = device_id_;
    const int P = tile.n_pop;
    const long M = static_cast<long>(tile.n_snp);
    out.P = P;
    if (P <= 0 || M <= 0) { out.M_kept = 0; return out; }

    const std::size_t pm =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t Mz = static_cast<std::size_t>(M);
    const bool has_physpos = physpos.size() >= Mz;

    DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
    decode_af_resident(tile, P, M, dQ, dV, dN);

    DeviceBuffer<int> dChrom(Mz);
    DeviceBuffer<double> dGenpos(Mz);
    DeviceBuffer<double> dPhyspos(has_physpos ? Mz : 0u);
    DeviceBuffer<std::uint8_t> dFlags(Mz);
    h2d_async(dChrom, chrom.data(), Mz, stream_.get());
    h2d_async(dGenpos, genpos.data(), Mz, stream_.get());
    if (has_physpos) {
        h2d_async(dPhyspos, physpos.data(), Mz, stream_.get());
    }
    launch_autosome_keep_mask(dChrom.data(), M, chrom_min, chrom_max, dFlags.data(),
                              stream_.get());

    DeviceBuffer<long> dKeepIdx(Mz);
    DeviceBuffer<int> dNumSel(1);
    {
        std::size_t scan_bytes = 0;
        STEPPE_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, scan_bytes, dFlags.data(), dKeepIdx.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        DeviceBuffer<unsigned char> dScanTemp(scan_bytes == 0 ? 1 : scan_bytes);
        std::size_t sb = scan_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            dScanTemp.data(), sb, dFlags.data(), dKeepIdx.data(),
            static_cast<std::int64_t>(M), stream_.get()));
    }

    DeviceBuffer<int> dChromKept(Mz);
    DeviceBuffer<double> dGenposKept(Mz);
    DeviceBuffer<double> dPhysposKept(has_physpos ? Mz : 0u);
    {
        std::size_t sel_bytes_i = 0, sel_bytes_d = 0;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, sel_bytes_i, dChrom.data(), dFlags.data(),
            dChromKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, sel_bytes_d, dGenpos.data(), dFlags.data(),
            dGenposKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        const std::size_t sel_bytes = std::max(sel_bytes_i, sel_bytes_d);
        DeviceBuffer<unsigned char> dSelTemp(sel_bytes == 0 ? 1 : sel_bytes);
        std::size_t sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            dSelTemp.data(), sb, dChrom.data(), dFlags.data(),
            dChromKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            dSelTemp.data(), sb, dGenpos.data(), dFlags.data(),
            dGenposKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        if (has_physpos) {
            sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
                dSelTemp.data(), sb, dPhyspos.data(), dFlags.data(),
                dPhysposKept.data(), dNumSel.data(),
                static_cast<std::int64_t>(M), stream_.get()));
        }
    }

    int m_kept = 0;
    d2h_async(&m_kept, dNumSel, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.M_kept = static_cast<long>(m_kept);
    if (m_kept <= 0) return out;

    const std::size_t pmk =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(m_kept);
    out.impl = std::make_unique<steppe::device::DeviceDecodeResult::Impl>();
    out.impl->q = DeviceBuffer<double>(pmk);
    out.impl->v = DeviceBuffer<double>(pmk);
    launch_compact_columns_gather(dQ.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                  out.impl->q.data(), stream_.get());
    launch_compact_columns_gather(dV.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                  out.impl->v.data(), stream_.get());

    out.chrom_kept.assign(static_cast<std::size_t>(m_kept), 0);
    out.genpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
    d2h_async(out.chrom_kept.data(), dChromKept,
              static_cast<std::size_t>(m_kept), stream_.get());
    d2h_async(out.genpos_kept.data(), dGenposKept,
              static_cast<std::size_t>(m_kept), stream_.get());
    if (has_physpos) {
        out.physpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
        d2h_async(out.physpos_kept.data(), dPhysposKept,
                  static_cast<std::size_t>(m_kept), stream_.get());
    }
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

// Compact-decode regime B: coverage filter — reference §8
steppe::device::DeviceDecodeResult CudaBackend::decode_af_compact_filter(
    const DecodeTileView& tile, std::span<const char> ref, std::span<const char> alt,
    std::span<const int> chrom, std::span<const double> genpos,
    std::span<const double> physpos, const FilterConfig& cfg,
    std::span<const std::size_t> pop_individuals, int ploidy, double maxmiss,
    long s_lo) {
    guard_device();
    steppe::device::DeviceDecodeResult out;
    out.device_id = device_id_;
    const int P = tile.n_pop;
    const long M = static_cast<long>(tile.n_snp);
    out.P = P;
    if (P <= 0 || M <= 0) { out.M_kept = 0; return out; }

    const std::size_t pm =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t Mz = static_cast<std::size_t>(M);
    const bool has_physpos = physpos.size() >= Mz;

    DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
    decode_af_resident(tile, P, M, dQ, dV, dN, s_lo);

    FilterConfig kernel_cfg = cfg;
    kernel_cfg.geno_max_missing = 1.0;

    double total_indiv_d = 0.0;
    if (pop_individuals.size() == static_cast<std::size_t>(P)) {
        std::size_t total = 0;
        for (int p = 0; p < P; ++p) total += pop_individuals[static_cast<std::size_t>(p)];
        total_indiv_d = static_cast<double>(total);
    } else {
        std::size_t total = 0;
        for (int p = 0; p < P; ++p)
            total += tile.pop_offsets[static_cast<std::size_t>(p) + 1] -
                     tile.pop_offsets[static_cast<std::size_t>(p)];
        total_indiv_d = static_cast<double>(total);
    }

    DeviceBuffer<char> dRef(Mz), dAlt(Mz);
    DeviceBuffer<int> dChrom(Mz);
    DeviceBuffer<double> dGenpos(Mz);
    DeviceBuffer<double> dPhyspos(has_physpos ? Mz : 0u);
    DeviceBuffer<std::uint8_t> dFlags(Mz);
    h2d_async(dRef, ref.data(), Mz, stream_.get());
    h2d_async(dAlt, alt.data(), Mz, stream_.get());
    h2d_async(dChrom, chrom.data(), Mz, stream_.get());
    h2d_async(dGenpos, genpos.data(), Mz, stream_.get());
    if (has_physpos) {
        h2d_async(dPhyspos, physpos.data(), Mz, stream_.get());
    }
    launch_regimeb_keep_mask(dQ.data(), dN.data(), P, M, dRef.data(), dAlt.data(),
                             dChrom.data(), kernel_cfg,
                             static_cast<double>(ploidy), total_indiv_d, maxmiss,
                             dFlags.data(), stream_.get());

    DeviceBuffer<long> dKeepIdx(Mz);
    DeviceBuffer<int> dNumSel(1);
    {
        std::size_t scan_bytes = 0;
        STEPPE_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, scan_bytes, dFlags.data(), dKeepIdx.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        DeviceBuffer<unsigned char> dScanTemp(scan_bytes == 0 ? 1 : scan_bytes);
        std::size_t sb = scan_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            dScanTemp.data(), sb, dFlags.data(), dKeepIdx.data(),
            static_cast<std::int64_t>(M), stream_.get()));
    }

    DeviceBuffer<int> dChromKept(Mz);
    DeviceBuffer<double> dGenposKept(Mz);
    DeviceBuffer<double> dPhysposKept(has_physpos ? Mz : 0u);
    {
        std::size_t sel_bytes_i = 0, sel_bytes_d = 0;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, sel_bytes_i, dChrom.data(), dFlags.data(),
            dChromKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, sel_bytes_d, dGenpos.data(), dFlags.data(),
            dGenposKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        const std::size_t sel_bytes = std::max(sel_bytes_i, sel_bytes_d);
        DeviceBuffer<unsigned char> dSelTemp(sel_bytes == 0 ? 1 : sel_bytes);
        std::size_t sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            dSelTemp.data(), sb, dChrom.data(), dFlags.data(),
            dChromKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
            dSelTemp.data(), sb, dGenpos.data(), dFlags.data(),
            dGenposKept.data(), dNumSel.data(),
            static_cast<std::int64_t>(M), stream_.get()));
        if (has_physpos) {
            sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
                dSelTemp.data(), sb, dPhyspos.data(), dFlags.data(),
                dPhysposKept.data(), dNumSel.data(),
                static_cast<std::int64_t>(M), stream_.get()));
        }
    }

    int m_kept = 0;
    d2h_async(&m_kept, dNumSel, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.M_kept = static_cast<long>(m_kept);
    if (m_kept <= 0) return out;

    const std::size_t pmk =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(m_kept);
    out.impl = std::make_unique<steppe::device::DeviceDecodeResult::Impl>();
    out.impl->q = DeviceBuffer<double>(pmk);
    out.impl->v = DeviceBuffer<double>(pmk);
    out.impl->n = DeviceBuffer<double>(pmk);
    launch_compact_columns_gather(dQ.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                  out.impl->q.data(), stream_.get());
    launch_compact_columns_gather(dV.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                  out.impl->v.data(), stream_.get());
    launch_compact_columns_gather(dN.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                  out.impl->n.data(), stream_.get());

    out.chrom_kept.assign(static_cast<std::size_t>(m_kept), 0);
    out.genpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
    d2h_async(out.chrom_kept.data(), dChromKept,
              static_cast<std::size_t>(m_kept), stream_.get());
    d2h_async(out.genpos_kept.data(), dGenposKept,
              static_cast<std::size_t>(m_kept), stream_.get());
    if (has_physpos) {
        out.physpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
        d2h_async(out.physpos_kept.data(), dPhysposKept,
                  static_cast<std::size_t>(m_kept), stream_.get());
    }
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

}  // namespace steppe::device
