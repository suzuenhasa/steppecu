// src/core/stats/read_canonical_tile.cpp
//
// The genotype-path format funnel: turns any supported on-disk genotype file
// into the one canonical individual-major tile every downstream stage expects.
// Every non-TGENO format is SNP-major on disk and is transposed to that shape
// on the GPU at read time.
//
// Reference: docs/reference/src_core_stats_read_canonical_tile.cpp.md
#include "core/stats/read_canonical_tile.hpp"

#include <stdexcept>
#include <utility>

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
            return transpose_snp_major(
                reader.read_snp_major_tile(part, snp_begin, snp_end), backend);

        case io::GenoFormat::Eigenstrat:
            return transpose_snp_major(
                reader.read_eigenstrat_snp_major_tile(part, snp_begin, snp_end), backend);

        case io::GenoFormat::Plink:
            return transpose_snp_major(
                reader.read_plink_snp_major_tile(part, snp_begin, snp_end), backend);

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
