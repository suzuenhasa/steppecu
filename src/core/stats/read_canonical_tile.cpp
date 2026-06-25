// src/core/stats/read_canonical_tile.cpp
//
// The genotype front-end FORMAT DISPATCH (M-FR-2; docs/design/format-readers.md
// §2.4). TGENO is the existing read_tile; GENO (SNP-major PACKEDANCESTRYMAP) is the
// `io`-leaf SNP-major gather + the on-device transpose_to_canonical, assembled here
// into the canonical individual-major io::GenotypeTile every downstream consumer
// already expects. See read_canonical_tile.hpp for the layering rationale.
#include "core/stats/read_canonical_tile.hpp"

#include <stdexcept>
#include <utility>

#include "device/backend.hpp"  // ComputeBackend, SnpMajorTileView, CanonicalTile, TileEncoding

namespace steppe::core {

namespace {

// Shared SNP-MAJOR finish for the GENO and EIGENSTRAT paths: hand the `io`-leaf
// SnpMajorTile gather to the backend transpose (on the GPU; the CpuBackend host-loop
// oracle) and assemble the canonical individual-major io::GenotypeTile. BOTH formats
// produce the IDENTICAL canonical SNP-major source packing (record s = SNP s, source
// individual i at byte i/4 MSB-first) and the canonical raw-value 0/1/2/3 codes — GENO
// reads those bytes raw, EIGENSTRAT packs them from the ASCII line — so the transpose
// + assemble is byte-for-byte the same. The encoding is IDENTITY for both (the native
// code IS the canonical code). The view borrows `src`'s storage for the call.
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

    // Assemble the canonical io::GenotypeTile from the transpose output + the partition
    // labels (the transpose carries packed/pop_offsets but not labels — a backend-local
    // POD; the labels come from the `io` gather).
    io::GenotypeTile tile;
    tile.packed = std::move(canon.packed);
    tile.bytes_per_record = canon.bytes_per_record;
    tile.n_snp = canon.n_snp;
    tile.n_individuals = canon.n_individuals;
    tile.pop_offsets = std::move(canon.pop_offsets);
    tile.pop_labels = src.pop_labels;
    // sample_ploidy stays EMPTY: ploidy is detected later on the canonical tile (the
    // M-FR-0 on-device prepass) or supplied as an explicit vector.
    return tile;
}

}  // namespace

io::GenotypeTile read_canonical_tile(io::GenoReader& reader,
                                     const io::IndPartition& part,
                                     ComputeBackend& backend,
                                     std::size_t snp_begin,
                                     std::size_t snp_end) {
    const io::GenoHeader& hdr = reader.header();
    switch (hdr.format) {
        case io::GenoFormat::Tgeno:
            // The canonical individual-major path — unchanged (the existing reader).
            return reader.read_tile(part, snp_begin, snp_end);

        case io::GenoFormat::Geno:
            // SNP-major PACKEDANCESTRYMAP: the `io`-leaf raw-byte gather + the on-device
            // transpose. P0 PACKEDANCESTRYMAP uses the canonical raw-value convention +
            // MSB-first bit order, so the encoding is IDENTITY.
            return transpose_snp_major(
                reader.read_snp_major_tile(part, snp_begin, snp_end), backend);

        case io::GenoFormat::Eigenstrat:
            // EIGENSTRAT (ASCII SNP-major, M-FR-EIG): the `io`-leaf ASCII parse + pack
            // into the SAME canonical SNP-major source, then the SAME on-device transpose.
            // The char→code map is the identity on the value (0/1/2 copies, 9→missing), so
            // the encoding is IDENTITY — only the gather differs from the GENO path.
            return transpose_snp_major(
                reader.read_eigenstrat_snp_major_tile(part, snp_begin, snp_end), backend);

        case io::GenoFormat::Plink:
            // PLINK (.bed SNP-major, 2-bit LSB-first, M-FR PLINK): the `io`-leaf gather
            // reads the .bed records and NORMALIZES them into the SAME canonical SNP-major
            // source — flipping the bit order LSB-first->MSB-first AND remapping the .bed
            // code through kBedToCanon (00->2/01->3 missing/10->1/11->0 in A1-copies ==
            // canonical ref copies, ref:=A1). After that normalization the source is
            // byte-for-byte what the GENO path produces, so the encoding is IDENTITY and
            // the SAME on-device transpose runs unchanged.
            return transpose_snp_major(
                reader.read_plink_snp_major_tile(part, snp_begin, snp_end), backend);

        case io::GenoFormat::Unknown:
        default:
            // The GenoReader ctor already rejects an unrecognized format, so this is
            // defensive (a future format reaching here without a dispatch arm).
            throw std::runtime_error(
                "core::read_canonical_tile: unsupported .geno format (not TGENO, GENO, "
                "EIGENSTRAT, nor PLINK) — no canonical-tile dispatch");
    }
}

}  // namespace steppe::core
