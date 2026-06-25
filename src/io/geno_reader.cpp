// src/io/geno_reader.cpp
//
// .geno header parse + tiled raw-byte gather (architecture.md §5 S0, §11.1;
// ROADMAP M1). Reads the TGENO header, validates the file size against the
// DERIVED record stride, and gathers the selected individuals' packed SNP-prefix
// bytes into a GenotypeTile — NO decode here (architecture.md §5 S0 row).
//
// LAYERING: `io`-leaf TU (architecture.md §4) — pure host C++20, no CUDA, no
// core/device dependency.
#include "io/geno_reader.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>

#include "io/eigenstrat_format.hpp"

namespace steppe::io {

// The record OFFSET (header_bytes + row * bytes_per_record) is computed in
// std::streamoff to keep the multiply 64-bit at AADR scale (~4e9; cleanup
// geno_reader 2.5). std::streamoff is implementation-defined signed; pin the
// ≥64-bit width the offset arithmetic assumes so a hypothetical 32-bit-offset
// platform fails to BUILD rather than silently truncating a seek target.
static_assert(sizeof(std::streamoff) >= 8,
              "GenoReader assumes a >=64-bit std::streamoff for record-offset math");

GenoReader::GenoReader(const std::string& geno_path) : path_(geno_path) {
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("io::GenoReader: cannot open .geno file: " + path_);
    }

    std::array<char, kGenoHeaderBytes> head{};
    // Hoist the loop-invariant streamsize header width once (cleanup geno_reader
    // 7.2): the read length and the gcount comparand are the same cast.
    const auto header_size = static_cast<std::streamsize>(kGenoHeaderBytes);
    in.read(head.data(), header_size);
    if (in.gcount() != header_size) {
        throw std::runtime_error("io::GenoReader: .geno shorter than the " +
                                 std::to_string(kGenoHeaderBytes) +
                                 "-byte header: " + path_);
    }

    header_ = parse_geno_header(head);
    if (header_.format == GenoFormat::Unknown) {
        throw std::runtime_error(
            "io::GenoReader: unrecognized .geno magic (expected TGENO or GENO): " + path_);
    }
    if (header_.bytes_per_record == 0 || header_.n_records == 0) {
        throw std::runtime_error("io::GenoReader: degenerate header (zero records/stride): " + path_);
    }

    // File-size validation against the DERIVED stride. The data region is
    // header_bytes + n_records * bytes_per_record; a shorter file is a partial
    // dataset — record how many complete records are present (the oracle handles
    // this with `n_records = (fsize - hdr) // bpi`).
    in.seekg(0, std::ios::end);
    const std::streamoff fsize = in.tellg();
    if (fsize < static_cast<std::streamoff>(header_.header_bytes)) {
        throw std::runtime_error("io::GenoReader: file smaller than header: " + path_);
    }
    const std::size_t data_bytes =
        static_cast<std::size_t>(fsize) - header_.header_bytes;
    records_present_ = data_bytes / header_.bytes_per_record;
    if (records_present_ == 0) {
        throw std::runtime_error("io::GenoReader: no complete records on disk: " + path_);
    }
    // records_present_ may be < header_.n_records for a partial file; that is the
    // cap the ind_reader uses. It must never EXCEED the header count.
    if (records_present_ > header_.n_records) records_present_ = header_.n_records;
}

GenotypeTile GenoReader::read_tile(const IndPartition& part,
                                   std::size_t snp_begin,
                                   std::size_t snp_end) {
    if (header_.format != GenoFormat::Tgeno) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: this is the TGENO (individual-major) path; "
            "this file is GENO (SNP-major PACKEDANCESTRYMAP) — read it via "
            "read_snp_major_tile + the on-device transpose_to_canonical.");
    }
    if (snp_begin != 0) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: M1 requires snp_begin == 0 (byte-aligned "
            "SNP prefix); nonzero begin is the M5 tile loop.");
    }
    if (snp_end <= snp_begin || snp_end > header_.n_snp) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: SNP range [" + std::to_string(snp_begin) + ", " +
            std::to_string(snp_end) + ") out of bounds (n_snp=" +
            std::to_string(header_.n_snp) + ")");
    }

    // FAIL-FAST on a degenerate partition (architecture.md §2). `read_ind` already
    // throws on an empty selection, but `read_tile` accepts an ARBITRARY
    // IndPartition, so a zero-population partition would otherwise produce a
    // silent n_pop()==0 tile the downstream decode short-circuits into an empty
    // Q/V/N (cleanup geno_reader 2.2).
    if (part.groups.empty()) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: empty partition (no selected populations) for " +
            path_);
    }

    const std::size_t tile_snps = snp_end - snp_begin;
    const std::size_t bytes_per_record = packed_bytes(tile_snps);  // ceil(tile_snps/4)

    // Count the gathered individuals (sum of selected segment sizes) and validate
    // each requested row is actually present on disk. A row >= records_present_
    // would otherwise seek into trailing junk / a concatenated file and read a
    // COMPLETE record of the WRONG individual silently (cleanup geno_reader 1.2);
    // bounding every row by records_present_ also caps n_individuals <=
    // records_present_, which defangs the size-multiply wrap below in practice.
    std::size_t n_individuals = 0;
    for (const auto& g : part.groups) {
        for (std::size_t row : g.rows) {
            if (row >= records_present_) {
                throw std::runtime_error(
                    "io::GenoReader::read_tile: individual row " + std::to_string(row) +
                    " out of range (records_present=" + std::to_string(records_present_) +
                    ") in " + path_);
            }
        }
        n_individuals += g.rows.size();
    }

    // CHECKED MULTIPLY before resize (cleanup geno_reader 1.5, the dominant item).
    // `n_individuals * bytes_per_record` is a std::size_t product; std::size_t
    // arithmetic wraps modulo 2^N (well-defined-but-SILENT, [basic.fundamental]).
    // On a hostile/stale partition the product can wrap to a SMALL value, `resize`
    // then allocates a too-small buffer, and the gather loop writes past the
    // allocation at `tile.packed.data() + out_ind * bytes_per_record` for out_ind
    // up to the true (un-wrapped) n_individuals — a silent heap-buffer-overflow
    // WRITE, not an exception. The row<records_present_ guard above bounds this in
    // practice; this is the direct, defense-in-depth fail-fast guard
    // (architecture.md §2). The idiom is the standard `a > MAX/b` overflow test
    // (bytes_per_record is provably nonzero here: tile_snps >= 1 ⇒
    // packed_bytes(tile_snps) >= 1).
    // The overflow throw and both resize() catch handlers report the SAME
    // n_individuals * bytes_per_record operands for the same path_; build that
    // shared count substring once and reuse it across all three, each site
    // supplying its own surrounding phrasing (cleanup geno_reader 7.1, 7.2).
    const std::string size_operands =
        std::to_string(n_individuals) + " individuals * " +
        std::to_string(bytes_per_record) + " bytes/record";
    if (n_individuals > std::numeric_limits<std::size_t>::max() / bytes_per_record) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: tile size overflow (" + size_operands +
            " exceeds size_t) for " + path_);
    }

    GenotypeTile tile;
    tile.bytes_per_record = bytes_per_record;
    tile.n_snp = tile_snps;
    tile.n_individuals = n_individuals;
    // EXCEPTION-TYPE CONTRACT (cleanup geno_reader 2.1). The checked-multiply
    // above rules out a SILENT size_t WRAP; but a large-but-NON-wrapping request
    // (an AADR-scale ~4 GB tile, or an over-budget gather) still makes `resize`
    // throw — and `std::vector::resize` throws `std::length_error` (a
    // std::logic_error, when sz > max_size(), [vector.capacity]/[container.alloc])
    // or `std::bad_alloc` (allocation failure), NEITHER of which derives from
    // std::runtime_error. The header (geno_reader.hpp:17-18) and ind_reader/
    // genotype_tile promise "I/O failures surface as std::runtime_error", so an
    // `app` caller written to the literal contract (catch(std::runtime_error&))
    // would miss the raw bad_alloc/length_error → unhandled → std::terminate.
    // Translate both into the documented runtime_error so the contract holds BY
    // CONSTRUCTION for ANY allocation-failure cause (architecture.md §2 fail-fast).
    try {
        tile.packed.resize(n_individuals * bytes_per_record);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: out of memory allocating tile (" +
            size_operands + ") for " + path_);
    } catch (const std::length_error&) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: tile too large for the allocator (" +
            size_operands + " exceeds vector::max_size()) for " + path_);
    }
    tile.pop_offsets.reserve(part.groups.size() + 1);
    tile.pop_labels.reserve(part.groups.size());

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("io::GenoReader::read_tile: cannot reopen .geno: " + path_);
    }

    // Gather: for each selected population (in Q/V/N row order), for each member
    // individual (ascending row), read its SNP-prefix bytes into the next tile
    // slot. Individuals land pop-contiguous, so pop_offsets segment the result.
    std::size_t out_ind = 0;
    tile.pop_offsets.push_back(0);
    // Hoist the loop-invariant streamsize record width once (cleanup geno_reader
    // 7.2): the read length and the gcount comparand are the same cast every row.
    const std::streamsize rec_bytes = static_cast<std::streamsize>(bytes_per_record);
    for (const auto& g : part.groups) {
        tile.pop_labels.push_back(g.label);
        for (std::size_t row : g.rows) {
            const std::streamoff off =
                static_cast<std::streamoff>(header_.header_bytes) +
                static_cast<std::streamoff>(row) *
                    static_cast<std::streamoff>(header_.bytes_per_record);
            in.seekg(off, std::ios::beg);
            char* dst = reinterpret_cast<char*>(tile.packed.data() + out_ind * bytes_per_record);
            in.read(dst, rec_bytes);
            if (in.gcount() != rec_bytes) {
                throw std::runtime_error(
                    "io::GenoReader::read_tile: short read for individual row " +
                    std::to_string(row) + " in " + path_);
            }
            ++out_ind;
        }
        tile.pop_offsets.push_back(out_ind);
    }
    return tile;
}

SnpMajorTile GenoReader::read_snp_major_tile(const IndPartition& part,
                                             std::size_t snp_begin,
                                             std::size_t snp_end) {
    if (header_.format != GenoFormat::Geno) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: this is the GENO (SNP-major "
            "PACKEDANCESTRYMAP) path; this file is TGENO (individual-major) — read it "
            "via read_tile.");
    }
    if (snp_begin != 0) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: P0 requires snp_begin == 0 "
            "(byte-aligned SNP prefix); nonzero begin is the M5 tile loop.");
    }
    if (snp_end <= snp_begin || snp_end > header_.n_snp) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: SNP range [" +
            std::to_string(snp_begin) + ", " + std::to_string(snp_end) +
            ") out of bounds (n_snp=" + std::to_string(header_.n_snp) + ")");
    }
    // FAIL-FAST on a degenerate partition (architecture.md §2; mirrors read_tile).
    if (part.groups.empty()) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: empty partition (no selected "
            "populations) for " + path_);
    }

    // SNP-major: one record per SNP, with the EIGENSOFT rlen-floored stride
    // (header_.bytes_per_record == max(kGenoHeaderBytes, ceil(n_ind/4))). Each
    // SNP record holds ALL individuals interleaved (4 individuals/byte); the
    // selected, reordered individual gather happens in the transpose, NOT here, so
    // this gathers the FULL per-SNP records for the [snp_begin, snp_end) prefix.
    const std::size_t tile_snps = snp_end - snp_begin;        // == snp_end for snp_begin==0
    const std::size_t src_bpr = header_.bytes_per_record;     // the rlen-floored SNP-record stride
    const std::size_t n_records_to_read =
        records_present_ < tile_snps ? records_present_ : tile_snps;
    if (n_records_to_read < tile_snps) {
        // The on-disk SNP records are fewer than the requested prefix (a partial
        // file). read_tile bounds individual rows by records_present_; the SNP-major
        // analogue is the SNP-record count, so a requested SNP past records_present_
        // would seek into trailing junk — reject rather than read garbage.
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: requested SNP prefix [0, " +
            std::to_string(tile_snps) + ") exceeds SNP records on disk (" +
            std::to_string(records_present_) + ") in " + path_);
    }

    // Build the SELECTION + pop-contiguous reorder (the gather LIST the transpose
    // applies output-driven). Each selected individual is a SOURCE ROW (its .ind /
    // genotype-column index); validate every row is a real individual of THIS file
    // (row < header_.n_ind) so the transpose never reads a padding byte as a phantom
    // individual (format-readers.md §3.4) — the SNP-major twin of read_tile's
    // row<records_present_ guard.
    SnpMajorTile tile;
    tile.src_bytes_per_record = src_bpr;
    tile.n_snp = tile_snps;
    tile.pop_offsets.reserve(part.groups.size() + 1);
    tile.pop_labels.reserve(part.groups.size());
    tile.pop_offsets.push_back(0);
    std::size_t out_ind = 0;
    for (const auto& g : part.groups) {
        tile.pop_labels.push_back(g.label);
        for (std::size_t row : g.rows) {
            if (row >= header_.n_ind) {
                throw std::runtime_error(
                    "io::GenoReader::read_snp_major_tile: individual row " +
                    std::to_string(row) + " out of range (n_ind=" +
                    std::to_string(header_.n_ind) + ") in " + path_);
            }
            tile.sel_rows.push_back(row);
            ++out_ind;
        }
        tile.pop_offsets.push_back(out_ind);
    }
    tile.n_individuals = out_ind;

    // CHECKED MULTIPLY before resize (mirrors read_tile's dominant guard): the
    // source-buffer size is tile_snps * src_bpr (std::size_t product, wraps modulo
    // 2^N — well-defined-but-SILENT). On a hostile header the product can wrap to a
    // SMALL value, resize allocates too little, and the gather writes past the
    // allocation. src_bpr is provably nonzero here (the ctor rejects a zero
    // bytes_per_record). Same idiom + same exception-type contract as read_tile.
    const std::string size_operands =
        std::to_string(tile_snps) + " snp-records * " +
        std::to_string(src_bpr) + " bytes/record";
    if (tile_snps > std::numeric_limits<std::size_t>::max() / src_bpr) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: source size overflow (" +
            size_operands + " exceeds size_t) for " + path_);
    }
    try {
        tile.snp_major.resize(tile_snps * src_bpr);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: out of memory allocating SNP-major "
            "source (" + size_operands + ") for " + path_);
    } catch (const std::length_error&) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: SNP-major source too large for the "
            "allocator (" + size_operands + " exceeds vector::max_size()) for " + path_);
    }

    // Gather the SNP-major records: for each SNP s in [0, tile_snps), seek to
    // header_bytes + s*src_bpr and read the FULL rlen-floored record (all
    // individuals). The axis-swapped twin of read_tile's per-individual seek/read.
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: cannot reopen .geno: " + path_);
    }
    const std::streamsize rec_bytes = static_cast<std::streamsize>(src_bpr);
    for (std::size_t s = 0; s < tile_snps; ++s) {
        const std::streamoff off =
            static_cast<std::streamoff>(header_.header_bytes) +
            static_cast<std::streamoff>(s) * static_cast<std::streamoff>(src_bpr);
        in.seekg(off, std::ios::beg);
        char* dst = reinterpret_cast<char*>(tile.snp_major.data() + s * src_bpr);
        in.read(dst, rec_bytes);
        if (in.gcount() != rec_bytes) {
            throw std::runtime_error(
                "io::GenoReader::read_snp_major_tile: short read for SNP record " +
                std::to_string(s) + " in " + path_);
        }
    }
    return tile;
}

}  // namespace steppe::io
