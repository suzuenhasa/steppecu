// src/io/geno_reader.hpp
//
// .geno reader: parse the packed header, detect TGENO vs PACKEDANCESTRYMAP, and
// expose a TILED raw-byte reader that gathers the selected individuals' packed
// records into a GenotypeTile (architecture.md §4 `io` LEAF, §5 S0, §11.1;
// ROADMAP M1). NO decode happens here — `io` reads/tiles packed bytes; the device
// (or CPU reference) kernel decodes (architecture.md §5 S0 row).
//
// TILE SHAPE (keep the signatures tile-shaped per §11.1, even though M1 decodes a
// small set in one shot — the SNP-tile loop is M5): the reader takes a SNP range
// [snp_begin, snp_end) and the selected population partition, and returns the
// packed-byte prefix for those SNPs across exactly the selected individuals,
// gathered into population-contiguous order. M5 will call this per SNP tile; M1
// calls it once with [0, M).
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, depends on
// nothing in core/device, no CUDA. I/O failures surface as std::runtime_error to
// the `app`/test caller (the leaf does not use core/device error types).
#ifndef STEPPE_IO_GENO_READER_HPP
#define STEPPE_IO_GENO_READER_HPP

#include <cstddef>
#include <string>

#include "io/eigenstrat_format.hpp"  // GenoHeader, GenoFormat
#include "io/genotype_tile.hpp"      // GenotypeTile
#include "io/ind_reader.hpp"         // IndPartition
#include "io/snp_major_tile.hpp"     // SnpMajorTile (the SNP-major / GENO gather output)

namespace steppe::io {

/// Owns an open packed .geno file and its parsed header; serves tiled raw-byte
/// reads. Construct from the .geno path (parses the header and validates the file
/// size against the DERIVED record stride); then call `read_tile` per SNP tile.
/// Move-only (it owns a file stream); cheap to keep open across the tile loop.
class GenoReader {
public:
    /// Open `geno_path`, read and parse its `kGenoHeaderBytes` header, and
    /// validate the on-disk size against `header_bytes + n_records *
    /// bytes_per_record`. Throws std::runtime_error on open failure, a bad/
    /// unrecognized magic, or a size inconsistent with the header.
    explicit GenoReader(const std::string& geno_path);

    /// The parsed header (format + counts + DERIVED record stride).
    [[nodiscard]] const GenoHeader& header() const noexcept { return header_; }

    /// Number of complete records actually present on disk (may be < n_records
    /// for a partial file). For TGENO this is the number of individual records
    /// available; pass it to ind_reader so selection ignores absent individuals.
    [[nodiscard]] std::size_t records_present() const noexcept { return records_present_; }

    /// Read SNPs [snp_begin, snp_end) for the selected populations into a
    /// GenotypeTile (TGENO individual-major). For each selected individual (in
    /// `part.groups` order, pop-contiguous), reads the packed byte prefix covering
    /// the SNP range and copies it into the tile; fills `pop_offsets`/`pop_labels`
    /// from the partition. The byte range read per individual is
    /// `[snp_begin/4 .. ceil(snp_end/4))`; for M1 `snp_begin == 0`, so codes align
    /// to byte boundaries and `bytes_per_record == ceil((snp_end-snp_begin)/4)`.
    ///
    /// REQUIRES snp_begin == 0 in M1 (the byte-aligned prefix case the decoder
    /// expects); a nonzero begin is reserved for the M5 tile loop and rejected
    /// here so a future caller does not silently get mis-aligned codes.
    ///
    /// The partition is VALIDATED at the boundary (fail-fast, architecture.md §2):
    /// an empty `part.groups`, any individual row >= records_present() (a stale /
    /// wrong-dataset partition), or a gathered size that would overflow size_t are
    /// REJECTED with a thrown std::runtime_error rather than producing a wrong
    /// gather or a silent heap-buffer-overflow write.
    ///
    /// EXCEPTION-TYPE CONTRACT (cleanup geno_reader 2.1): a large-but-non-wrapping
    /// tile whose backing-buffer allocation fails — `std::vector::resize` throwing
    /// std::length_error (sz > max_size()) or std::bad_alloc — is TRANSLATED into
    /// std::runtime_error, so EVERY failure of this function surfaces as the
    /// documented std::runtime_error (no raw std::bad_alloc / std::logic_error
    /// escapes to a caller written to catch(std::runtime_error&)).
    ///
    /// Throws std::runtime_error on a read error, an out-of-range SNP range, a
    /// non-TGENO file (TGENO is the canonical individual-major path; a SNP-major
    /// GENO file is read via `read_snp_major_tile` + the on-device transpose), a
    /// malformed partition (empty / row out of range / size overflow), or a failed
    /// tile allocation.
    [[nodiscard]] GenotypeTile read_tile(const IndPartition& part,
                                         std::size_t snp_begin,
                                         std::size_t snp_end);

    /// SNP-MAJOR gather (PACKEDANCESTRYMAP / GENO; format-readers.md §2.4, M-FR-2).
    /// Read SNPs [snp_begin, snp_end) of a SNP-major .geno into a raw SnpMajorTile:
    /// the SNP-major source records for that range (one per SNP, full rlen-floored
    /// width, ALL individuals interleaved) PLUS the selected, pop-contiguous
    /// individual gather list (sel_rows / pop_offsets / pop_labels) built from the
    /// partition. NO decode and NO transpose happens here — the `io` leaf is
    /// CUDA-FREE; the app layer hands this tile to ComputeBackend::transpose_to_
    /// canonical, which applies the selection + reorder + encoding on-device to
    /// produce the canonical individual-major GenotypeTile decode_af consumes.
    ///
    /// This is the axis-swapped twin of `read_tile`: it seeks WHOLE per-SNP records
    /// (header_bytes + snp*bytes_per_record) rather than per-individual records, and
    /// the selection moves into the gather LIST (the transpose applies it) rather
    /// than the seek loop (the SNP-major source interleaves all individuals within a
    /// SNP byte, so a row cannot be skipped at read time). It keeps the SAME
    /// fail-fast guards: empty partition, any selected row >= the source n_ind, and
    /// the checked-multiply tile-size overflow are all rejected with a thrown
    /// std::runtime_error (architecture.md §2 fail-fast).
    ///
    /// REQUIRES snp_begin == 0 for P0 (the byte-aligned SNP prefix; a nonzero begin
    /// is the M5 tile loop). Throws std::runtime_error on a read error, an
    /// out-of-range SNP range, a non-GENO file (TGENO uses `read_tile`), a malformed
    /// partition, or a failed allocation — the SAME documented exception contract as
    /// `read_tile`.
    [[nodiscard]] SnpMajorTile read_snp_major_tile(const IndPartition& part,
                                                   std::size_t snp_begin,
                                                   std::size_t snp_end);

    /// EIGENSTRAT (ASCII SNP-major) gather (M-FR-EIG). The text twin of
    /// `read_snp_major_tile`: parse the ASCII .geno (one line per SNP, one char per
    /// individual — 0/1/2 ref-allele copies, 9 missing) for SNPs [snp_begin,
    /// snp_end), map each char to its canonical 2-bit code (eigenstrat_char_to_code),
    /// and PACK the line into the canonical SNP-major 2-bit layout (record s = SNP s,
    /// `packed_bytes(n_ind)` bytes, source individual i at byte i/4, MSB-first) — the
    /// SAME packing read_snp_major_tile produces from raw GENO bytes. It then builds
    /// the identical selected, pop-contiguous individual gather list. The result is a
    /// SnpMajorTile the app hands to the SAME ComputeBackend::transpose_to_canonical
    /// (with TileEncoding::Identity, since the char→code map is the identity on the
    /// value) — so NOTHING downstream of the transpose changes; only the gather
    /// (ASCII parse + pack vs raw byte copy) and the geometry source differ.
    ///
    /// REQUIRES snp_begin == 0 (the byte-aligned SNP prefix; a nonzero begin is the
    /// M5 tile loop). Keeps the SAME fail-fast guards as read_snp_major_tile (empty
    /// partition, any selected row >= n_ind, the checked-multiply tile-size overflow)
    /// PLUS a malformed-line guard: a .geno line whose length != n_ind, or carrying a
    /// non-{0,1,2,9} byte, is REJECTED with the 1-based SNP line + column context
    /// (the .geno line index IS the SNP index — a desync corrupts the SNP axis).
    /// Throws std::runtime_error on a read error, an out-of-range SNP range, a
    /// non-EIGENSTRAT file (TGENO/GENO use the packed readers), a malformed partition,
    /// a malformed .geno line, or a failed allocation — the SAME documented contract.
    [[nodiscard]] SnpMajorTile read_eigenstrat_snp_major_tile(const IndPartition& part,
                                                             std::size_t snp_begin,
                                                             std::size_t snp_end);

    /// PLINK .bed (SNP-major, 2-bit LSB-first) gather (M-FR PLINK). The binary twin of
    /// read_snp_major_tile: read the .bed SNP records for SNPs [snp_begin, snp_end) — one
    /// record per SNP, ceil(n_ind/4) bytes, 4 individuals/byte LSB-first — then NORMALIZE
    /// each record into the canonical SNP-major 2-bit layout (record s = SNP s,
    /// `packed_bytes(n_ind)` bytes, source individual i at byte i/4 MSB-first) by (a)
    /// flipping the bit order LSB-first -> MSB-first AND (b) remapping the .bed code
    /// through kBedToCanon (00->2, 01->3 missing, 10->1, 11->0 in A1-copies == canonical
    /// ref copies, since ref := A1; format-readers.md §3.2). The result is the SAME
    /// canonical SNP-major source read_snp_major_tile produces from raw GENO bytes, so the
    /// app hands it to the SAME ComputeBackend::transpose_to_canonical with
    /// TileEncoding::Identity (the LUT + bit-flip happen HERE, in the io-leaf gather, so
    /// the transpose body and the encoding enum are untouched — the EIGENSTRAT precedent).
    ///
    /// The .bed magic (0x6c 0x1b 0x01) is consumed at construction (header_bytes ==
    /// kBedMagicBytes == 3); each SNP record is at `kBedMagicBytes + s*bytes_per_record`.
    /// REQUIRES snp_begin == 0 (the byte-aligned SNP prefix; a nonzero begin is the M5
    /// tile loop). Keeps the SAME fail-fast guards as read_snp_major_tile (empty partition,
    /// any selected row >= n_ind, the checked-multiply tile-size overflow, the short-read
    /// throw). Throws std::runtime_error on a read error, an out-of-range SNP range, a
    /// non-PLINK file (TGENO/GENO/EIGENSTRAT use their own readers), a malformed partition,
    /// or a failed allocation — the SAME documented contract.
    [[nodiscard]] SnpMajorTile read_plink_snp_major_tile(const IndPartition& part,
                                                        std::size_t snp_begin,
                                                        std::size_t snp_end);

    /// ANCESTRYMAP (unpacked legacy EIGENSOFT, TEXT triple) gather (M-FR-AM). The
    /// text-triple twin of read_eigenstrat_snp_major_tile: the .geno is one line per
    /// (SNP, individual) pair `<snp_id> <sample_id> <genotype>`, laid out SNP-major
    /// (each SNP's n_ind rows consecutive, in .ind order; SNPs in .snp order), so it is
    /// POSITIONALLY addressed — line L is SNP L/n_ind, individual L%n_ind. For SNPs
    /// [snp_begin, snp_end) this reads the leading tile_snps*n_ind lines SEQUENTIALLY
    /// (the .geno carries no header), maps each line's 3rd token to its canonical 2-bit
    /// code (ancestrymap_token_to_code: 0/1/2 copies, "-1"->kMissingCode), and PACKS it
    /// MSB-first into the canonical SNP-major 2-bit layout (record s = SNP s,
    /// `packed_bytes(n_ind)` bytes, source individual i at byte i/4) — the SAME packing
    /// read_snp_major_tile / read_eigenstrat_snp_major_tile produce. It then builds the
    /// identical selected, pop-contiguous individual gather list. The result is a
    /// SnpMajorTile the app hands to the SAME ComputeBackend::transpose_to_canonical with
    /// TileEncoding::Identity (the token->code map is the value identity, like EIGENSTRAT;
    /// only the parse + the "-1" missing sentinel differ) — so NOTHING downstream of the
    /// transpose changes.
    ///
    /// REQUIRES snp_begin == 0 (the byte-aligned SNP prefix; a nonzero begin is the M5
    /// tile loop). Keeps the SAME fail-fast guards as read_eigenstrat_snp_major_tile
    /// (empty partition, any selected row >= n_ind, the checked-multiply tile-size
    /// overflow) PLUS a malformed-line guard: a .geno line that does not carry exactly
    /// kAncestrymapFields whitespace tokens, or whose 3rd token is not 0/1/2/-1, is
    /// REJECTED with the 1-based line + SNP/individual context (the line index IS the
    /// SNP*n_ind+individual position — a desync corrupts the genotype matrix). Throws
    /// std::runtime_error on a read error, an out-of-range SNP range, a non-ANCESTRYMAP
    /// file (the other formats use their own readers), a malformed partition, a malformed
    /// .geno line, or a failed allocation — the SAME documented contract.
    [[nodiscard]] SnpMajorTile read_ancestrymap_snp_major_tile(const IndPartition& part,
                                                              std::size_t snp_begin,
                                                              std::size_t snp_end);

    GenoReader(const GenoReader&) = delete;
    GenoReader& operator=(const GenoReader&) = delete;
    GenoReader(GenoReader&&) noexcept = default;
    GenoReader& operator=(GenoReader&&) noexcept = default;
    ~GenoReader() = default;

private:
    /// Build the SnpMajorTile SELECTION + pop-contiguous reorder gather list shared by
    /// all four SNP-major readers (read_snp_major_tile / EIGENSTRAT / PLINK / ANCESTRYMAP):
    /// set the tile geometry (src_bytes_per_record / n_snp), fill pop_offsets/pop_labels,
    /// and push each selected source `row` into sel_rows after validating row < n_ind so
    /// the transpose never reads a padding byte as a phantom individual (format-readers.md
    /// §3.4). The readers differ ONLY in `reader_name`, which prefixes the out-of-range
    /// throw (cleanup geno_reader 7.1). Throws std::runtime_error on a row >= n_ind.
    void build_selection(const IndPartition& part,
                         std::size_t src_bpr,
                         std::size_t tile_snps,
                         const char* reader_name,
                         SnpMajorTile& tile) const;

    /// Allocate the SNP-major source buffer (tile_snps * src_bpr bytes) with the checked
    /// multiply + allocation-failure translation shared by all four SNP-major readers
    /// (cleanup geno_reader 7.1): a SILENT size_t wrap, std::bad_alloc, and
    /// std::length_error are each turned into the documented std::runtime_error
    /// (EXCEPTION-TYPE CONTRACT, cleanup geno_reader 2.1). `zero_init` selects assign(0)
    /// (the text/PLINK packers OR codes in and need a partial last byte pre-zeroed) vs a
    /// bare resize (the raw GENO copy overwrites every byte). `reader_name` prefixes the
    /// throws. Throws std::runtime_error on size overflow or a failed allocation.
    void checked_alloc_snp_major(std::size_t tile_snps,
                                 std::size_t src_bpr,
                                 const char* reader_name,
                                 bool zero_init,
                                 SnpMajorTile& tile) const;

    std::string path_;
    GenoHeader header_;
    std::size_t records_present_ = 0;
};

}  // namespace steppe::io

#endif  // STEPPE_IO_GENO_READER_HPP
