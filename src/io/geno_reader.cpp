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
#include <cstdint>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include "io/eigenstrat_format.hpp"

namespace steppe::io {

GenoReader::GenoReader(const std::string& geno_path) : path_(geno_path) {
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("io::GenoReader: cannot open .geno file: " + path_);
    }

    std::array<char, kGenoHeaderBytes> head{};
    in.read(head.data(), static_cast<std::streamsize>(kGenoHeaderBytes));
    if (in.gcount() != static_cast<std::streamsize>(kGenoHeaderBytes)) {
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
            "io::GenoReader::read_tile: M1 decode targets TGENO (individual-major); "
            "this file is GENO (SNP-major PACKEDANCESTRYMAP).");
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

    const std::size_t tile_snps = snp_end - snp_begin;
    const std::size_t bytes_per_rec = packed_bytes(tile_snps);  // ceil(tile_snps/4)

    // Count the gathered individuals (sum of selected segment sizes).
    std::size_t n_ind = 0;
    for (const auto& g : part.groups) n_ind += g.rows.size();

    GenotypeTile tile;
    tile.bytes_per_record = bytes_per_rec;
    tile.n_snp = tile_snps;
    tile.n_individuals = n_ind;
    tile.packed.resize(n_ind * bytes_per_rec);
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
    for (const auto& g : part.groups) {
        tile.pop_labels.push_back(g.label);
        for (std::size_t row : g.rows) {
            const std::streamoff off =
                static_cast<std::streamoff>(header_.header_bytes) +
                static_cast<std::streamoff>(row) *
                    static_cast<std::streamoff>(header_.bytes_per_record);
            in.seekg(off, std::ios::beg);
            char* dst = reinterpret_cast<char*>(tile.packed.data() + out_ind * bytes_per_rec);
            in.read(dst, static_cast<std::streamsize>(bytes_per_rec));
            if (in.gcount() != static_cast<std::streamsize>(bytes_per_rec)) {
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

}  // namespace steppe::io
