// src/core/stats/read_canonical_tile.cpp
//
// The genotype-path format funnel: turns any supported on-disk genotype file
// into the one canonical individual-major tile every downstream stage expects.
// Every non-TGENO format is SNP-major on disk and is transposed to that shape
// on the GPU at read time.
//
// Reference: docs/reference/src_core_stats_read_canonical_tile.cpp.md
#include "core/stats/read_canonical_tile.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "core/internal/decode_af.hpp"  // kCodesPerByte
#include "device/backend.hpp"

namespace steppe::core {

// Shared SNP-major transpose path — reference §4. Declared in the header so the
// native VCF-ingest path (the sixth reader arm) reuses the one transpose seam.
io::GenotypeTile transpose_snp_major(const io::SnpMajorTile& src,
                                     ComputeBackend& backend) {
    SnpMajorTileView view;
    view.snp_major = src.snp_major.data();
    view.src_bytes_per_record = src.src_bytes_per_record;
    view.n_snp = src.n_snp;
    view.sel_rows = src.sel_rows.data();
    view.n_individuals = src.n_individuals;
    view.pop_offsets = src.pop_offsets.data();
    view.n_pop = static_cast<int>(src.n_pop());
    view.encoding = TileEncoding::Identity;
    CanonicalTile canon = backend.transpose_to_canonical(view);

    io::GenotypeTile tile;
    tile.packed = std::move(canon.packed);
    tile.bytes_per_record = canon.bytes_per_record;
    tile.n_snp = canon.n_snp;
    tile.n_individuals = canon.n_individuals;
    tile.pop_offsets = std::move(canon.pop_offsets);
    tile.pop_labels = src.pop_labels;
    return tile;
}

namespace {

constexpr std::size_t kCpb = static_cast<std::size_t>(kCodesPerByte);

// Bound one streamed SNP-block's source bytes (blk * src_bytes_per_record) to ~1 GiB — and never
// past a quarter of free VRAM, since the block is uploaded whole for the transpose. Rounded DOWN
// to a multiple of kCodesPerByte so each block's canonical output byte-columns are a clean,
// non-overlapping slice of the whole tile; a single block (blk >= n_snp) is the whole SNP axis,
// byte-identical to the pre-streaming one-shot transpose.
constexpr std::size_t kTransposeBlockBytesTarget = std::size_t{1} << 30;  // 1 GiB

std::size_t choose_transpose_block(std::size_t n_snp, std::size_t src_bpr,
                                   std::size_t free_vram_bytes) {
    if (n_snp == 0 || src_bpr == 0) return n_snp;
    std::size_t budget = kTransposeBlockBytesTarget;
    if (free_vram_bytes != 0 && free_vram_bytes / 4u < budget) budget = free_vram_bytes / 4u;
    std::size_t blk = budget / src_bpr;
    blk -= blk % kCpb;               // multiple of kCodesPerByte
    if (blk < kCpb) blk = kCpb;
    if (blk >= n_snp) return n_snp;  // single block == the whole axis (byte-identical)
    return blk;
}

// Stream a seekable binary SNP-major format (GENO / PLINK) into the canonical individual-major
// tile SNP-block by SNP-block: read ONE block's SNP-major bytes (bounded), transpose+gather it on
// the device, and scatter that block's canonical byte-columns into the whole output tile — so
// peak host/device INPUT is bounded by a block, not the whole SNP-major matrix (the 1240K-cohort
// transpose-on-read host wall). The produced tile.packed is bit-for-bit identical to the one-shot
// transpose: each output byte is written exactly once and block boundaries fall on kCodesPerByte-
// SNP lines, so the byte-column that packs SNPs [4b, 4b+4) comes from exactly one block.
io::GenotypeTile read_snp_major_streamed(io::GenoReader& reader, const io::IndPartition& part,
                                         ComputeBackend& backend, std::size_t snp_begin,
                                         std::size_t snp_end, io::GenoFormat format) {
    const std::size_t M = snp_end - snp_begin;
    const std::size_t src_bpr = reader.header().bytes_per_record;
    const std::size_t blk =
        choose_transpose_block(M, src_bpr, backend.capabilities().free_vram_bytes);

    io::GenotypeTile whole;
    std::size_t out_bpr = 0;
    bool inited = false;
    for (std::size_t sb = snp_begin; sb < snp_end; sb += blk) {
        const std::size_t bn = std::min(blk, snp_end - sb);
        io::SnpMajorTile src = (format == io::GenoFormat::Plink)
                                   ? reader.read_plink_snp_major_tile(part, sb, sb + bn)
                                   : reader.read_snp_major_tile(part, sb, sb + bn);
        const io::GenotypeTile block = transpose_snp_major(src, backend);
        if (!inited) {
            whole.n_individuals = block.n_individuals;
            whole.n_snp = M;
            whole.bytes_per_record = (M + kCpb - 1) / kCpb;
            whole.pop_offsets = block.pop_offsets;
            whole.pop_labels = block.pop_labels;
            whole.packed.assign(whole.n_individuals * whole.bytes_per_record, std::uint8_t{0});
            out_bpr = whole.bytes_per_record;
            inited = true;
        }
        // sb is a multiple of kCodesPerByte (blk is, and the front-end passes snp_begin == 0), so
        // the block's output byte-columns start exactly at col_off with no cross-block carry.
        const std::size_t col_off = (sb - snp_begin) / kCpb;
        const std::size_t bbpr = block.bytes_per_record;  // ceil(bn / kCodesPerByte)
        for (std::size_t g = 0; g < whole.n_individuals; ++g) {
            std::memcpy(whole.packed.data() + g * out_bpr + col_off,
                        block.packed.data() + g * bbpr, bbpr);
        }
    }
    return whole;  // M == 0 -> a well-formed empty tile (never reached from the front-end)
}

}  // namespace

// Format dispatch — reference §3
io::GenotypeTile read_canonical_tile(io::GenoReader& reader,
                                     const io::IndPartition& part,
                                     ComputeBackend& backend,
                                     std::size_t snp_begin,
                                     std::size_t snp_end) {
    const io::GenoHeader& hdr = reader.header();
    switch (hdr.format) {
        case io::GenoFormat::Tgeno:
            return reader.read_tile(part, snp_begin, snp_end);

        case io::GenoFormat::Geno:
            return read_snp_major_streamed(reader, part, backend, snp_begin, snp_end,
                                           io::GenoFormat::Geno);

        case io::GenoFormat::Eigenstrat:
            return transpose_snp_major(
                reader.read_eigenstrat_snp_major_tile(part, snp_begin, snp_end), backend);

        case io::GenoFormat::Plink:
            return read_snp_major_streamed(reader, part, backend, snp_begin, snp_end,
                                           io::GenoFormat::Plink);

        case io::GenoFormat::Ancestrymap:
            return transpose_snp_major(
                reader.read_ancestrymap_snp_major_tile(part, snp_begin, snp_end), backend);

        case io::GenoFormat::Unknown:
        default:
            throw std::runtime_error(
                "core::read_canonical_tile: unsupported .geno format (not TGENO, GENO, "
                "EIGENSTRAT, ANCESTRYMAP, nor PLINK) — no canonical-tile dispatch");
    }
}

}  // namespace steppe::core
