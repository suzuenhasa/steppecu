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

namespace {

// Strip a single trailing '\r' (a CRLF line ending) from `line` in place, so a
// DOS-encoded EIGENSTRAT .geno yields the SAME per-line individual count as a
// Unix-encoded one (the '\r' is NOT a genotype char and must not inflate n_ind /
// trip the char→code guard). std::getline already consumed the '\n'.
void strip_cr(std::string& line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
}

// EIGENSTRAT geometry scan (M-FR-EIG). The ASCII .geno is one line per SNP, one
// char per individual; the file is fully self-describing — n_ind is the first
// line's char count and n_snp is the line count. We ALSO validate the shape here
// (every line is the SAME length, every char is 0/1/2/9) so a non-EIGENSTRAT file
// that merely lacks the packed magic does NOT get misclassified as a degenerate
// EIGENSTRAT (it falls through to format==Unknown and the ctor fails loudly). The
// FULL scan (not just the first line) is required to get n_snp; it is a single
// sequential pass, cheap relative to the genotype compute, and matches the .ind/
// .snp readers' "read the whole sibling" cost.
//
// Returns a header with format==Eigenstrat + n_ind/n_snp/n_records/bytes_per_record
// on a well-formed EIGENSTRAT file, or format==Unknown on ANY non-EIGENSTRAT shape
// (empty file, a non-{0,1,2,9} byte, a ragged line). bytes_per_record is the
// canonical SNP-major stride packed_bytes(n_ind) (4 individuals/byte); header_bytes
// is 0 (the ASCII .geno has no header record — the data starts at byte 0).
GenoHeader parse_eigenstrat_geometry(const std::string& path) {
    GenoHeader h;  // defaults to format==Unknown
    std::ifstream in(path, std::ios::binary);
    if (!in) return h;  // cannot reopen → Unknown (the ctor already opened it once)

    std::string line;
    std::size_t n_ind = 0;
    std::size_t n_snp = 0;
    bool first = true;
    while (std::getline(in, line)) {
        strip_cr(line);
        // A trailing blank line at EOF is a common newline artifact (the .snp reader
        // tolerates the same). An INTERIOR blank line is a ragged-shape error — but it
        // is caught by the length check below (0 != n_ind), so only the EOF case needs
        // the explicit tolerate.
        if (line.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;  // trailing blank
            return GenoHeader{};  // interior blank → not a well-formed EIGENSTRAT
        }
        if (first) {
            n_ind = line.size();
            first = false;
            if (n_ind == 0) return GenoHeader{};  // empty first line → Unknown
        } else if (line.size() != n_ind) {
            // Ragged: not a SNP-major char matrix → not EIGENSTRAT (fail loudly later).
            return GenoHeader{};
        }
        // Validate every char is a legal EIGENSTRAT genotype (0/1/2/9). A single
        // foreign byte means this is NOT an EIGENSTRAT .geno (e.g. a packed file whose
        // bytes happened to dodge the magic, or a corrupt file) — classify Unknown.
        for (char c : line) {
            std::uint8_t code = 0;
            if (!eigenstrat_char_to_code(c, code)) return GenoHeader{};
        }
        ++n_snp;
    }
    if (n_snp == 0 || n_ind == 0) return GenoHeader{};  // no records → Unknown

    h.format = GenoFormat::Eigenstrat;
    h.n_ind = n_ind;
    h.n_snp = n_snp;
    h.n_records = n_snp;                       // SNP-major: one record per SNP
    h.bytes_per_record = packed_bytes(n_ind);  // canonical SNP-major stride (4 ind/byte)
    h.header_bytes = 0;                        // ASCII .geno: data starts at byte 0
    return h;
}

}  // namespace

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
    // A packed TGENO/GENO header is EXACTLY kGenoHeaderBytes — parse the magic ONLY
    // when the full header is present. A file SHORTER than the header cannot be packed,
    // but MAY be a small ASCII EIGENSTRAT .geno (a tiny fixture: few inds × few SNPs),
    // so a short read is NOT a hard error here — it falls through to the EIGENSTRAT
    // probe below (which fails loudly if the content is not 0/1/2/9 genotype lines).
    if (in.gcount() == header_size) {
        header_ = parse_geno_header(head);
    }

    if (header_.format == GenoFormat::Unknown) {
        // No packed TGENO/GENO magic (or the file is shorter than the packed header).
        // Probe for the ASCII EIGENSTRAT .geno (M-FR-EIG): one line per SNP, one char
        // per individual (0/1/2 ref-allele copies, 9 missing). Its leading bytes are
        // ASCII genotype chars + a newline, never the binary magic — so a file whose
        // every line is the SAME length of {0,1,2,9} chars is EIGENSTRAT.
        // parse_eigenstrat_geometry scans the file to derive n_ind (the first line's
        // char count) and n_snp (the line count); on a non-EIGENSTRAT shape it leaves
        // format==Unknown and we fail loudly below.
        header_ = parse_eigenstrat_geometry(path_);
    }
    if (header_.format == GenoFormat::Unknown) {
        throw std::runtime_error(
            "io::GenoReader: unrecognized .geno format (expected packed TGENO/GENO "
            "magic, or an ASCII EIGENSTRAT .geno of 0/1/2/9 genotype lines): " + path_);
    }
    // EIGENSTRAT is fully self-described by its ASCII content (geometry derived by the
    // scan above); it has NO packed data region, so the packed file-size validation
    // below (header_bytes + n_records*bytes_per_record) does not apply. Set
    // records_present_ to the SNP-record count (the SNP-major convention GENO uses, so
    // read_ind's individual-axis cap is the SNP count and no individual is dropped) and
    // return — the ASCII parse happens lazily in read_eigenstrat_snp_major_tile.
    if (header_.format == GenoFormat::Eigenstrat) {
        records_present_ = header_.n_records;  // == n_snp
        return;
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

SnpMajorTile GenoReader::read_eigenstrat_snp_major_tile(const IndPartition& part,
                                                        std::size_t snp_begin,
                                                        std::size_t snp_end) {
    if (header_.format != GenoFormat::Eigenstrat) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: this is the EIGENSTRAT "
            "(ASCII SNP-major) path; this file is " +
            std::string(header_.format == GenoFormat::Tgeno ? "TGENO (read via read_tile)"
                        : header_.format == GenoFormat::Geno ? "GENO (read via read_snp_major_tile)"
                                                             : "an unrecognized format") +
            " — wrong reader for " + path_);
    }
    if (snp_begin != 0) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: P0 requires snp_begin == 0 "
            "(byte-aligned SNP prefix); nonzero begin is the M5 tile loop.");
    }
    if (snp_end <= snp_begin || snp_end > header_.n_snp) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: SNP range [" +
            std::to_string(snp_begin) + ", " + std::to_string(snp_end) +
            ") out of bounds (n_snp=" + std::to_string(header_.n_snp) + ")");
    }
    // FAIL-FAST on a degenerate partition (architecture.md §2; mirrors read_snp_major_tile).
    if (part.groups.empty()) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: empty partition (no selected "
            "populations) for " + path_);
    }

    const std::size_t tile_snps = snp_end - snp_begin;       // == snp_end for snp_begin==0
    const std::size_t src_bpr = header_.bytes_per_record;    // canonical SNP-major stride packed_bytes(n_ind)
    if (tile_snps > records_present_) {
        // The ASCII .geno has fewer SNP lines than the requested prefix (a partial /
        // truncated file). records_present_ == n_snp here, so this is the EIGENSTRAT twin
        // of read_snp_major_tile's SNP-records-on-disk guard.
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: requested SNP prefix [0, " +
            std::to_string(tile_snps) + ") exceeds SNP lines on disk (" +
            std::to_string(records_present_) + ") in " + path_);
    }

    // Build the SELECTION + pop-contiguous reorder (the gather list the transpose applies
    // output-driven) — byte-for-byte the same construction as read_snp_major_tile, since
    // EIGENSTRAT shares the .ind/.snp and the canonical SNP-major source packing. Validate
    // every selected row is a real individual of THIS file (row < header_.n_ind == the .geno
    // line length) so the transpose never reads a phantom individual.
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
                    "io::GenoReader::read_eigenstrat_snp_major_tile: individual row " +
                    std::to_string(row) + " out of range (n_ind=" +
                    std::to_string(header_.n_ind) + ") in " + path_);
            }
            tile.sel_rows.push_back(row);
            ++out_ind;
        }
        tile.pop_offsets.push_back(out_ind);
    }
    tile.n_individuals = out_ind;

    // CHECKED MULTIPLY before resize (mirrors read_snp_major_tile's dominant guard):
    // tile_snps * src_bpr is a std::size_t product (wraps modulo 2^N, SILENT). src_bpr is
    // provably nonzero (the ctor sets bytes_per_record = packed_bytes(n_ind) with n_ind>=1).
    const std::string size_operands =
        std::to_string(tile_snps) + " snp-records * " +
        std::to_string(src_bpr) + " bytes/record";
    if (tile_snps > std::numeric_limits<std::size_t>::max() / src_bpr) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: source size overflow (" +
            size_operands + " exceeds size_t) for " + path_);
    }
    try {
        // Zero-init: a partial last SNP byte (n_ind % 4 != 0) keeps its unused high-row
        // slots at 0, AND every code slot is written by the pack loop below, so the
        // canonical SNP-major source is fully defined.
        tile.snp_major.assign(tile_snps * src_bpr, std::uint8_t{0});
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: out of memory allocating "
            "SNP-major source (" + size_operands + ") for " + path_);
    } catch (const std::length_error&) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: SNP-major source too large for "
            "the allocator (" + size_operands + " exceeds vector::max_size()) for " + path_);
    }

    // PARSE + PACK the ASCII .geno into the canonical SNP-major 2-bit layout. For each
    // SNP line s in [0, tile_snps): each char c at column i is individual i's genotype;
    // map c → its canonical 2-bit code (eigenstrat_char_to_code) and OR it MSB-first into
    // byte s*src_bpr + i/4 at position i%4 — the SAME packing read_snp_major_tile reads
    // raw from a packed GENO file, so the transpose consumes it identically (Identity
    // encoding). The .geno line index IS the SNP index, so a length/char mismatch is a
    // fail-fast format error (a desync would corrupt the SNP axis vs the .snp).
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: cannot reopen .geno: " + path_);
    }
    const std::size_t cpb = static_cast<std::size_t>(kCodesPerByte);
    std::string line;
    for (std::size_t s = 0; s < tile_snps; ++s) {
        if (!std::getline(in, line)) {
            throw std::runtime_error(
                "io::GenoReader::read_eigenstrat_snp_major_tile: unexpected EOF at SNP line " +
                std::to_string(s + 1) + " (expected " + std::to_string(tile_snps) +
                " lines) in " + path_);
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF tolerance
        if (line.size() != header_.n_ind) {
            throw std::runtime_error(
                "io::GenoReader::read_eigenstrat_snp_major_tile: SNP line " +
                std::to_string(s + 1) + " has " + std::to_string(line.size()) +
                " genotype chars, expected n_ind=" + std::to_string(header_.n_ind) +
                " (ragged .geno desyncs the SNP axis) in " + path_);
        }
        std::uint8_t* rec = tile.snp_major.data() + s * src_bpr;
        for (std::size_t i = 0; i < header_.n_ind; ++i) {
            std::uint8_t code = 0;
            if (!eigenstrat_char_to_code(line[i], code)) {
                throw std::runtime_error(
                    "io::GenoReader::read_eigenstrat_snp_major_tile: illegal genotype char '" +
                    std::string(1, line[i]) + "' at SNP line " + std::to_string(s + 1) +
                    ", individual column " + std::to_string(i + 1) +
                    " (expected 0/1/2/9) in " + path_);
            }
            // MSB-first pack into the canonical SNP-major byte (the read_snp_major_tile /
            // transpose convention): individual i at byte i/4, position i%4, shift
            // (kCodesPerByte-1 - i%4)*kBitsPerCode = 6/4/2/0.
            const std::size_t byte_in_rec = i / cpb;
            const int shift =
                (kCodesPerByte - 1 - static_cast<int>(i % cpb)) * kBitsPerCode;
            rec[byte_in_rec] = static_cast<std::uint8_t>(
                rec[byte_in_rec] | static_cast<std::uint8_t>(code << shift));
        }
    }
    return tile;
}

}  // namespace steppe::io
