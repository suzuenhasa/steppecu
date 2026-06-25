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

        case io::GenoFormat::Geno: {
            // SNP-major PACKEDANCESTRYMAP. (1) `io`-leaf gather of the raw SNP-major
            // records + the selected, pop-contiguous individual list (CUDA-free).
            const io::SnpMajorTile src =
                reader.read_snp_major_tile(part, snp_begin, snp_end);

            // (2) Hand the gather to the backend transpose (on the GPU; the CpuBackend
            // host-loop oracle). The view borrows the SnpMajorTile's storage for the
            // call's duration. P0 PACKEDANCESTRYMAP uses the canonical raw-value
            // convention + MSB-first bit order, so the encoding is IDENTITY.
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

            // (3) Assemble the canonical io::GenotypeTile from the transpose output +
            // the partition labels (the transpose carries packed/pop_offsets but not
            // labels — a backend-local POD; the labels come from the `io` gather).
            io::GenotypeTile tile;
            tile.packed = std::move(canon.packed);
            tile.bytes_per_record = canon.bytes_per_record;
            tile.n_snp = canon.n_snp;
            tile.n_individuals = canon.n_individuals;
            tile.pop_offsets = std::move(canon.pop_offsets);
            tile.pop_labels = src.pop_labels;
            // sample_ploidy stays EMPTY: ploidy is detected later on the canonical
            // tile (the M-FR-0 on-device prepass) or supplied as an explicit vector.
            return tile;
        }

        case io::GenoFormat::Unknown:
        default:
            // The GenoReader ctor already rejects an unrecognized magic, so this is
            // defensive (a future format reaching here without a dispatch arm).
            throw std::runtime_error(
                "core::read_canonical_tile: unsupported .geno format (neither TGENO "
                "nor GENO) — no canonical-tile dispatch");
    }
}

}  // namespace steppe::core
