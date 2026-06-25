// src/io/snp_major_tile.hpp
//
// SnpMajorTile — the plain `io`-leaf output struct for a SNP-MAJOR genotype source
// (PACKEDANCESTRYMAP / GENO; later EIGENSTRAT / PLINK). It is the SNP-major TWIN of
// io::GenotypeTile: where GenotypeTile is the canonical INDIVIDUAL-major tile the
// decode/ploidy backend consumes directly, SnpMajorTile carries the RAW SNP-major
// bytes (one record per SNP, individuals interleaved within each SNP's bytes) plus
// the SELECTED, pop-contiguous individual gather list — the input the app layer
// hands to ComputeBackend::transpose_to_canonical to produce a GenotypeTile
// (architecture.md §4 `io` LEAF; docs/design/format-readers.md §2.4, M-FR-2).
//
// WHY a SEPARATE struct (not a GenotypeTile): a SNP-major source CANNOT be packed
// into the individual-major GenotypeTile WITHOUT the on-device transpose, and the
// `io` leaf is CUDA-FREE (it may not call a kernel). So the leaf produces this raw
// SNP-major view + the selection, and the app layer (the only `io`<->`device` wiring
// point) runs the transpose. The fields here map 1:1 onto device::SnpMajorTileView
// (backend.hpp), but this is a plain `io`-leaf POD with NO core/device dependency.
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, depends on
// nothing in core/device, no CUDA.
#ifndef STEPPE_IO_SNP_MAJOR_TILE_HPP
#define STEPPE_IO_SNP_MAJOR_TILE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace steppe::io {

/// A raw SNP-MAJOR genotype tile + the selected, pop-contiguous individual gather
/// list — the `io`-leaf input to the on-device transpose-to-canonical pass.
///
/// SNP-MAJOR source packing: `snp_major` holds `n_snp` records of
/// `src_bytes_per_record` bytes each (record s = SNP s); SOURCE individual `i` of
/// SNP `s` is the 2-bit code at byte `s*src_bytes_per_record + i/4`, position `i%4`
/// (MSB-first; the same code_in_byte order, individual axis interleaved within each
/// SNP byte). `src_bytes_per_record` is the GENO rlen-floored stride
/// max(kGenoHeaderBytes, ceil(n_ind/4)) — possibly LARGER than ceil(n_ind/4); only
/// `i/4` for the SELECTED rows (each < the source's n_ind) is ever read, so a
/// small-n_ind record's PADDING bytes are never decoded as phantom individuals
/// (format-readers.md §3.4).
struct SnpMajorTile {
    /// Raw SNP-major source bytes for the SNP range this tile covers (record s = SNP
    /// s, full rlen-floored width). Length `n_snp * src_bytes_per_record`.
    std::vector<std::uint8_t> snp_major;

    /// Source SNP-record stride (the rlen-floored max(kGenoHeaderBytes,
    /// ceil(n_ind/4))). The transpose reads only `src_row/4 < src_bytes_per_record`.
    std::size_t src_bytes_per_record = 0;

    /// SNPs this tile covers (the source-record count AND the output column axis).
    std::size_t n_snp = 0;

    /// SOURCE individual ROW per OUTPUT column g, in pop-contiguous Q/V/N order: the
    /// IndPartition's selected, label-ASC-ordered set flattened pop-by-pop (output
    /// column g -> source row sel_rows[g]). Length `n_individuals`. The transpose
    /// applies this selection + reorder output-driven (the SNP-major source
    /// interleaves ALL individuals, so the gather happens at transpose time).
    std::vector<std::size_t> sel_rows;

    /// Number of gathered output individuals (the output record count).
    std::size_t n_individuals = 0;

    /// Population segment boundaries over the OUTPUT individual axis: pop `p` owns
    /// gathered individuals [pop_offsets[p], pop_offsets[p+1]). Size P+1,
    /// pop_offsets[0] == 0, pop_offsets[P] == n_individuals.
    std::vector<std::size_t> pop_offsets;

    /// Population labels in Q/V/N row order (size P), parallel to the segments —
    /// carried so the canonical GenotypeTile the app builds keeps the pop labels.
    std::vector<std::string> pop_labels;

    /// Number of populations P (== pop_labels.size() == pop_offsets.size() - 1).
    [[nodiscard]] std::size_t n_pop() const noexcept { return pop_labels.size(); }
};

}  // namespace steppe::io

#endif  // STEPPE_IO_SNP_MAJOR_TILE_HPP
