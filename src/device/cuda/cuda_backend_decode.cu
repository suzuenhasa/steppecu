// src/device/cuda/cuda_backend_decode.cu
//
// CudaBackend — decode / transpose format-reader engine subsystem TU (cuda_backend.cu
// split T3; docs/kimiactions/05-cuda-backend-split.md §2.3 TU-C). Out-of-line homes of
// `CudaBackend::decode_af_resident` + `decode_af` + `detect_sample_ploidy_device` +
// `transpose_to_canonical` + `decode_af_compact_autosome` + `decode_af_compact_filter`
// — the genotype-decode front-end (2-bit unpack + segmented allele-freq reduction →
// resident Q/V/N), the on-device per-sample ploidy prepass (M-FR-0), the SNP-major →
// canonical individual-major transpose (M-FR-1), and the two device-resident
// compact-decode regimes (autosome keep-mask + regime-B coverage filter, each a CUB
// ExclusiveSum compacted-column index + DeviceSelect::Flagged chrom/genpos compaction +
// scan-keyed column gather). Bodies MOVED VERBATIM from cuda_backend.cu; nothing about
// codegen / math / precision / file-order changed by the split.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). Joins the SAME
// steppe_device target, so it inherits identical codegen/macros/RDC.
#include <cub/device/device_scan.cuh>    // cub::DeviceScan::ExclusiveSum — the compacted-column index
#include <cub/device/device_select.cuh>  // cub::DeviceSelect::Flagged — the kept chrom/genpos compaction

#include <cstdint>  // std::uint8_t — the packed-genotype record byte type
#include <vector>   // std::vector — DecodeResult Q/V/N + the per-sample ploidy host vector

#include "core/internal/host_device.hpp"               // STEPPE_ASSERT (the s_lo % 4 == 0 tile-slice precondition)
#include "core/internal/nvtx.hpp"                      // STEPPE_NVTX_RANGE (coarse phase-boundary marker; used by decode_af)
#include "device/cuda/cuda_backend.cuh"                // the CudaBackend class declaration (split T0)
#include "device/cuda/check.cuh"                       // STEPPE_CUDA_CHECK
#include "device/cuda/decode_af_kernel.cuh"            // launch_decode_af (2-bit unpack + segmented AF reduction → Q/V/N)
#include "device/cuda/decode_compact_kernel.cuh"       // launch_autosome_keep_mask / _regimeb_keep_mask / _compact_columns_gather (+ FilterConfig)
#include "device/cuda/detect_ploidy_kernel.cuh"        // launch_detect_ploidy (M-FR-0 on-device AT2 per-sample ploidy prepass)
#include "device/cuda/device_decode_result_impl.cuh"   // DeviceDecodeResult::Impl (the resident Q/V/N owners)
#include "device/cuda/transpose_canonical_kernel.cuh"  // launch_transpose_to_canonical + TransposeEncoding (M-FR-1)

namespace steppe::device {

void CudaBackend::decode_af_resident(const DecodeTileView& tile, int P, long M,
                        DeviceBuffer<double>& dQ, DeviceBuffer<double>& dV,
                        DeviceBuffer<double>& dN, long s_lo) {
    // SNP-TILE SLICE. This tile spans SNPs [s_lo, s_lo + M) of each individual's
    // FULL-row packed record (M == tile.n_snp columns; tile.bytes_per_record is the
    // full on-disk row stride). `tile_width` = ceil(M/4) is the COMPACTED tile-local
    // record stride — the 2-bit slice actually uploaded — so the resident set is
    // O(P × M) in the TILE M, never O(P × M_total). s_lo % 4 == 0 is REQUIRED (the
    // caller SNP-tiles in multiple-of-4 steps) so SNP local_s maps to byte local_s/4,
    // position local_s%4 — the SAME codes the full-M decode reads, hence the decode /
    // keep-mask / scan / gather kernels are UNCHANGED (they address the tile-local
    // stride). The untiled callers (s_lo=0, M=tile.n_snp) stay bit-identical: the
    // slice reads bytes [0, ceil(M/4)) per record — exactly the codes the flat upload
    // fed the kernel (record PADDING beyond ceil(M/4) was never decoded).
    STEPPE_ASSERT(s_lo % core::kCodesPerByte == 0,
                  "decode_af_resident: s_lo must be a multiple of kCodesPerByte (4) "
                  "so the packed 2-bit tile slice stays byte-aligned");
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t tile_width =
        (static_cast<std::size_t>(M) + cpb - 1u) / cpb;  // ceil(M/4): the slice stride
    const std::size_t packed_bytes = tile.n_individuals * tile_width;
    const std::size_t n_off = static_cast<std::size_t>(P) + 1u;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes);
    DeviceBuffer<std::size_t> dOffsets(n_off);

    // PER-SAMPLE ploidy (AT2 adjust_pseudohaploid). THREE cases (precedence in
    // order): (1) an EXPLICIT per-sample vector — upload it; (2) tile.sample_ploidy
    // NULL but detect_ploidy_on_device set (M-FR-0, the L2 on-device prepass) —
    // DERIVE the per-sample ploidy on the GPU from the just-uploaded dPacked
    // (launch_detect_ploidy, one thread/individual, the same packed bytes the decode
    // reads), bit-identical to the host io::detect_sample_ploidy; (3) neither — the
    // kernel falls back to the uniform scalar tile.ploidy (the legacy all-diploid
    // path), a null device pointer (no alloc/copy). A zero-length DeviceBuffer yields
    // a null .data(), so the same handle serves the no-ploidy case.
    const bool explicit_sample_ploidy = (tile.sample_ploidy != nullptr);
    const bool device_detect =
        (!explicit_sample_ploidy && tile.detect_ploidy_on_device);
    const bool have_sample_ploidy = explicit_sample_ploidy || device_detect;
    DeviceBuffer<int> dSamplePloidy(have_sample_ploidy ? tile.n_individuals : 0u);

    // STRIDED SLICE H2D: source row g starts at packed + g*bytes_per_record + s_lo/4;
    // copy `tile_width` bytes into dPacked's g-th `tile_width`-strided row. spitch is
    // the FULL on-disk row stride (bytes_per_record); dpitch is the compacted tile-local
    // stride (tile_width). (packed_bytes == n_individuals*tile_width sizes dPacked.)
    STEPPE_CUDA_CHECK(cudaMemcpy2DAsync(
        dPacked.data(), tile_width,
        tile.packed + static_cast<std::size_t>(s_lo) / cpb, tile.bytes_per_record,
        tile_width, tile.n_individuals,
        cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dOffsets.data(), tile.pop_offsets,
                                      n_off * sizeof(std::size_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (explicit_sample_ploidy) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSamplePloidy.data(), tile.sample_ploidy,
                                          tile.n_individuals * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
    } else if (device_detect) {
        // The L2 on-device prepass: scan dPacked (already in-flight on stream_; the
        // kernel is enqueued AFTER the upload, so the ordering is correct) and write
        // dSamplePloidy. Feeds launch_decode_af below in the SAME stream — no D2H.
        launch_detect_ploidy(dPacked.data(), tile_width,
                             tile.n_individuals, tile.n_snp, dSamplePloidy.data(),
                             stream_.get());
    }

    // S0 unpack + S1 segmented reduction → Q/V/N (RESIDENT).
    launch_decode_af(dPacked.data(), tile_width, dOffsets.data(),
                     P, M, tile.ploidy,
                     have_sample_ploidy ? dSamplePloidy.data() : nullptr,
                     dQ.data(), dV.data(), dN.data(), stream_.get());
    // dPacked/dOffsets/dSamplePloidy are still in-flight uploads; they free at
    // scope exit. The decode kernel and the uploads are all on stream_, so the
    // ordering is correct and the caller's later stream_ work observes dQ/dV/dN.
}

DecodeResult CudaBackend::decode_af(const DecodeTileView& tile) {
    guard_device();
    STEPPE_NVTX_RANGE("decode");  // coarse phase boundary: genotype decode -> Q/V/N
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

    // Decode → dQ/dV/dN RESIDENT (the shared front-end), then the C1 full D2H
    // across the CUDA-free seam (this is the host/oracle path; the resident
    // device-resident seam, decode_af_compact_autosome, drops this D2H).
    DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
    decode_af_resident(tile, P, M, dQ, dV, dN);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.q.data(), dQ.data(), pm * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.v.data(), dV.data(), pm * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.n.data(), dN.data(), pm * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

std::vector<int> CudaBackend::detect_sample_ploidy_device(
    const DecodeTileView& tile) {
    guard_device();
    std::vector<int> out(tile.n_individuals, core::kPloidyPseudoHaploid);
    if (tile.n_individuals == 0) return out;
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes);
    DeviceBuffer<int> dPloidy(tile.n_individuals);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPacked.data(), tile.packed,
                                      packed_bytes * sizeof(std::uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_detect_ploidy(dPacked.data(), tile.bytes_per_record, tile.n_individuals,
                         tile.n_snp, dPloidy.data(), stream_.get());
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.data(), dPloidy.data(),
                                      tile.n_individuals * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

CanonicalTile CudaBackend::transpose_to_canonical(
    const SnpMajorTileView& view) {
    guard_device();
    CanonicalTile out;
    out.n_snp = view.n_snp;
    out.n_individuals = view.n_individuals;
    // ceil(n_snp/4) — the canonical output record stride (the SAME radix the base
    // oracle / io::packed_bytes use; via core::kCodesPerByte so they cannot drift).
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    out.bytes_per_record = view.n_snp == 0 ? 0 : (view.n_snp + cpb - 1) / cpb;
    if (view.pop_offsets != nullptr && view.n_pop >= 0) {
        out.pop_offsets.assign(
            view.pop_offsets,
            view.pop_offsets + static_cast<std::size_t>(view.n_pop) + 1u);
    }
    const std::size_t out_total = view.n_individuals * out.bytes_per_record;
    out.packed.assign(out_total, std::uint8_t{0});
    if (out_total == 0) return out;  // empty tile (no individuals or no SNPs)

    // Upload the SNP-major source + the per-output-column source-row selection.
    const std::size_t src_total = view.n_snp * view.src_bytes_per_record;
    DeviceBuffer<std::uint8_t> dSrc(src_total);
    DeviceBuffer<std::size_t> dSel(view.n_individuals);
    DeviceBuffer<std::uint8_t> dOut(out_total);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSrc.data(), view.snp_major,
                                      src_total * sizeof(std::uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSel.data(), view.sel_rows,
                                      view.n_individuals * sizeof(std::size_t),
                                      cudaMemcpyHostToDevice, stream_.get()));

    // Map the CUDA-FREE seam encoding onto the device-private kernel enum (1:1).
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
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.packed.data(), dOut.data(),
                                      out_total * sizeof(std::uint8_t),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

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
    // The AT2 bp block-fallback axis (physical position), compacted in lockstep
    // with chrom/genpos ONLY when the caller supplies a full-length physpos span;
    // an empty span leaves out.physpos_kept empty (the fallback off). Same value
    // type (double) as genpos, so it needs no extra temp-size query below.
    const bool has_physpos = physpos.size() >= Mz;

    // ---- 1. Decode → dQ/dV/dN RESIDENT (dN is decoded but the regime-(A) Q/V
    // consumers ignore it; it is NOT compacted/kept). NO host D2H. ----------
    DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
    decode_af_resident(tile, P, M, dQ, dV, dN);

    // ---- 2. The per-SNP autosome keep-mask (INTEGER-EXACT) → d_flags. -------
    DeviceBuffer<int> dChrom(Mz);
    DeviceBuffer<double> dGenpos(Mz);
    DeviceBuffer<double> dPhyspos(has_physpos ? Mz : 0u);
    DeviceBuffer<std::uint8_t> dFlags(Mz);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dChrom.data(), chrom.data(), Mz * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dGenpos.data(), genpos.data(), Mz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (has_physpos) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPhyspos.data(), physpos.data(),
                                          Mz * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
    }
    launch_autosome_keep_mask(dChrom.data(), M, chrom_min, chrom_max, dFlags.data(),
                              stream_.get());

    // ---- 3. The compacted column index = EXCLUSIVE prefix sum of d_flags, and
    // the kept count M_kept = keep_idx[M-1] + flags[M-1] (CUB DeviceScan). -----
    DeviceBuffer<long> dKeepIdx(Mz);
    DeviceBuffer<int> dNumSel(1);
    {
        // ExclusiveSum over the uint8 flags into a long index buffer (the
        // compacted column position). Two-call temp-storage idiom (CUDA 13.x
        // CCCL CUB; d_temp_storage=nullptr query then the sized buffer).
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

    // ---- 4. Compact the 1-D chrom/genpos with CUB DeviceSelect::Flagged
    // (the SAME idiom run_fstat_sweep_device uses; ordering preserved). The
    // num_selected_out is M_kept — read it back small. ----------------------
    DeviceBuffer<int> dChromKept(Mz);
    DeviceBuffer<double> dGenposKept(Mz);
    DeviceBuffer<double> dPhysposKept(has_physpos ? Mz : 0u);
    {
        // CUB DeviceSelect::Flagged temp size depends on the VALUE TYPE (the
        // internal tile-state carries the data type), so query BOTH the int
        // (chrom) and double (genpos) calls and size dSelTemp to the MAX — the
        // double query is the larger; reusing an int-sized temp for the double
        // Flagged silently undersizes it (the observed all-zero genpos bug).
        // physpos is ALSO a double, so the double query already covers it (no
        // separate query needed); the same dSelTemp serves the third Flagged.
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

    // Read M_kept back (one int) — needed to size the resident Q/V buffers.
    int m_kept = 0;
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&m_kept, dNumSel.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.M_kept = static_cast<long>(m_kept);
    if (m_kept <= 0) return out;

    // ---- 5. Allocate the RESIDENT compacted Q/V [P × M_kept] and gather the
    // kept columns (scan-keyed; FILE ORDER preserved). The handle ESCAPES. ---
    const std::size_t pmk =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(m_kept);
    out.impl = std::make_unique<steppe::device::DeviceDecodeResult::Impl>();
    out.impl->q = DeviceBuffer<double>(pmk);
    out.impl->v = DeviceBuffer<double>(pmk);
    launch_compact_columns_gather(dQ.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                  out.impl->q.data(), stream_.get());
    launch_compact_columns_gather(dV.data(), P, M, dFlags.data(), dKeepIdx.data(),
                                  out.impl->v.data(), stream_.get());

    // ---- 6. The small kept chrom/genpos/physpos D2H (for the CUDA-free assign_blocks;
    // physpos only when supplied — it feeds the AT2 bp fallback). ----------------
    out.chrom_kept.assign(static_cast<std::size_t>(m_kept), 0);
    out.genpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.chrom_kept.data(), dChromKept.data(),
                                      static_cast<std::size_t>(m_kept) * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.genpos_kept.data(), dGenposKept.data(),
                                      static_cast<std::size_t>(m_kept) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    if (has_physpos) {
        out.physpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.physpos_kept.data(), dPhysposKept.data(),
                                          static_cast<std::size_t>(m_kept) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
    }
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

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
    // The AT2 bp block-fallback axis, compacted in lockstep ONLY when the caller
    // supplies a full-length physpos span (empty ⇒ out.physpos_kept left empty).
    const bool has_physpos = physpos.size() >= Mz;

    // ---- 1. Decode → dQ/dV/dN RESIDENT (the SHARED front-end; NO host D2H). The
    // regime-B keep-mask reads the UNCOMPACTED dQ/dN; the gather then compacts
    // Q/V/N from the SAME resident dQ/dV/dN — no second decode. `s_lo` slices the
    // packed upload to SNPs [s_lo, s_lo+M) so the resident set is O(P × M) in the
    // TILE M (the SNP-tile loop's peak-VRAM cure). ----------------------------
    DeviceBuffer<double> dQ(pm), dV(pm), dN(pm);
    decode_af_resident(tile, P, M, dQ, dV, dN, s_lo);

    // ---- 2. The regime-B per-SNP keep-mask (the SHARED reduction + decision +
    // the SEPARATE maxmiss) → d_flags. The sample-axis geno predicate is the
    // no-op (geno_max_missing FORCED to 1.0); the pop-axis maxmiss is separate. -
    FilterConfig kernel_cfg = cfg;
    kernel_cfg.geno_max_missing = 1.0;  // pop-coverage maxmiss is the `maxmiss` arg.

    // total_indiv = Σ_pop pop_individuals (the SNP-independent missing-frac
    // denominator). pop_individuals length == P (the caller's contract); if it is
    // short/empty, fall back to the segment sizes from the tile pop_offsets so the
    // reduction never reads OOB. (Mirrors snp_filter.cpp's total_indiv loop.)
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

    // Upload the per-SNP ref/alt allele chars + chrom (genpos rides only the
    // Flagged companion). dChrom is needed BOTH by the keep-mask (autosomes_only)
    // and by the Flagged compaction, so it is uploaded once.
    DeviceBuffer<char> dRef(Mz), dAlt(Mz);
    DeviceBuffer<int> dChrom(Mz);
    DeviceBuffer<double> dGenpos(Mz);
    DeviceBuffer<double> dPhyspos(has_physpos ? Mz : 0u);
    DeviceBuffer<std::uint8_t> dFlags(Mz);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRef.data(), ref.data(), Mz * sizeof(char),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dAlt.data(), alt.data(), Mz * sizeof(char),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dChrom.data(), chrom.data(), Mz * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dGenpos.data(), genpos.data(), Mz * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (has_physpos) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPhyspos.data(), physpos.data(),
                                          Mz * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_.get()));
    }
    launch_regimeb_keep_mask(dQ.data(), dN.data(), P, M, dRef.data(), dAlt.data(),
                             dChrom.data(), kernel_cfg,
                             static_cast<double>(ploidy), total_indiv_d, maxmiss,
                             dFlags.data(), stream_.get());

    // ---- 3. The compacted column index = EXCLUSIVE prefix sum of d_flags (the
    // IDENTICAL CUB DeviceScan idiom as decode_af_compact_autosome). -----------
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

    // ---- 4. Compact the 1-D chrom/genpos with CUB DeviceSelect::Flagged (the
    // IDENTICAL idiom + the int/double MAX-temp-size guard). physpos (also double)
    // rides the same dSelTemp — a third Flagged when supplied. -------------------
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
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&m_kept, dNumSel.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.M_kept = static_cast<long>(m_kept);
    if (m_kept <= 0) return out;

    // ---- 5. The RESIDENT compacted Q/V/N [P × M_kept], the THREE lockstep
    // scan-keyed gathers (Q, V, AND N — N is the regime-B addition; the SAME
    // dFlags + dKeepIdx, so N is compacted in EXACT lockstep with Q/V, FILE ORDER
    // preserved by the monotone exclusive scan). ------------------------------
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

    // ---- 6. The small kept chrom/genpos/physpos D2H (for the CUDA-free assign_blocks;
    // physpos only when supplied — it feeds the AT2 bp fallback). ----------------
    out.chrom_kept.assign(static_cast<std::size_t>(m_kept), 0);
    out.genpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.chrom_kept.data(), dChromKept.data(),
                                      static_cast<std::size_t>(m_kept) * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.genpos_kept.data(), dGenposKept.data(),
                                      static_cast<std::size_t>(m_kept) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    if (has_physpos) {
        out.physpos_kept.assign(static_cast<std::size_t>(m_kept), 0.0);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.physpos_kept.data(), dPhysposKept.data(),
                                          static_cast<std::size_t>(m_kept) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
    }
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

}  // namespace steppe::device
