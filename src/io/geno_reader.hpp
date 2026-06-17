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
    /// non-TGENO file (the M1 decode path targets TGENO), a malformed partition
    /// (empty / row out of range / size overflow), or a failed tile allocation.
    [[nodiscard]] GenotypeTile read_tile(const IndPartition& part,
                                         std::size_t snp_begin,
                                         std::size_t snp_end);

    GenoReader(const GenoReader&) = delete;
    GenoReader& operator=(const GenoReader&) = delete;
    GenoReader(GenoReader&&) noexcept = default;
    GenoReader& operator=(GenoReader&&) noexcept = default;
    ~GenoReader() = default;

private:
    std::string path_;
    GenoHeader header_;
    std::size_t records_present_ = 0;
};

}  // namespace steppe::io

#endif  // STEPPE_IO_GENO_READER_HPP
