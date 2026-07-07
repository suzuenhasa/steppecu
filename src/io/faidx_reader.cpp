// Reference: docs/reference/src_io_faidx_reader.cpp.md
// src/io/faidx_reader.cpp
//
// FaidxReader implementation — see the header. Parses the `.fai` sidecar into a
// contig->(length,offset,linebases,linewidth) map, then answers base_at() with a
// single seek + 1-byte read using the standard faidx offset arithmetic.
#include "io/faidx_reader.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace steppe::io {

namespace {

[[nodiscard]] char up(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

}  // namespace

FaidxReader::FaidxReader(const std::string& fasta_path) : fasta_path_(fasta_path) {
    const std::string fai_path = fasta_path + ".fai";
    std::ifstream fai(fai_path);
    if (!fai) {
        throw std::runtime_error("io::FaidxReader: cannot open FASTA index: " + fai_path +
                                 " (run `samtools faidx` on " + fasta_path + " first)");
    }

    std::string line;
    std::size_t line_no = 0;
    while (std::getline(fai, line)) {
        ++line_no;
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string name;
        FaiEntry e;
        if (!(ls >> name >> e.length >> e.offset >> e.linebases >> e.linewidth)) {
            throw std::runtime_error("io::FaidxReader: malformed .fai record at line " +
                                     std::to_string(line_no) + " in " + fai_path);
        }
        if (e.linebases <= 0 || e.linewidth < e.linebases) {
            throw std::runtime_error("io::FaidxReader: nonsensical LINEBASES/LINEWIDTH at line " +
                                     std::to_string(line_no) + " in " + fai_path);
        }
        index_.emplace(std::move(name), e);
    }

    fasta_.open(fasta_path, std::ios::binary);
    if (!fasta_) {
        throw std::runtime_error("io::FaidxReader: cannot open FASTA: " + fasta_path);
    }
}

const FaidxReader::FaiEntry* FaidxReader::resolve(const std::string& contig) const {
    const auto it = index_.find(contig);
    if (it != index_.end()) return &it->second;
    // Toggle a leading 'chr' prefix so a "1" query resolves a "chr1" FASTA and
    // vice versa (spec §1: inputs are unprefixed, but tolerate a prefixed FASTA).
    if (contig.rfind("chr", 0) == 0) {
        const auto alt = index_.find(contig.substr(3));
        if (alt != index_.end()) return &alt->second;
    } else {
        const auto alt = index_.find("chr" + contig);
        if (alt != index_.end()) return &alt->second;
    }
    return nullptr;
}

bool FaidxReader::has_contig(const std::string& contig) const {
    return resolve(contig) != nullptr;
}

char FaidxReader::base_at(const std::string& contig, long long pos1) const {
    const FaiEntry* e = resolve(contig);
    if (e == nullptr) {
        throw std::runtime_error("io::FaidxReader: unknown contig '" + contig + "' in " +
                                 fasta_path_);
    }
    if (pos1 < 1 || pos1 > e->length) {
        throw std::runtime_error("io::FaidxReader: position " + std::to_string(pos1) +
                                 " out of range [1," + std::to_string(e->length) + "] on contig '" +
                                 contig + "' in " + fasta_path_);
    }
    const long long z = pos1 - 1;  // 0-based
    const long long byte = e->offset + (z / e->linebases) * e->linewidth + (z % e->linebases);
    fasta_.clear();
    fasta_.seekg(byte);
    const int c = fasta_.get();
    if (c < 0) {
        throw std::runtime_error("io::FaidxReader: read failed at byte " + std::to_string(byte) +
                                 " (contig '" + contig + "' pos " + std::to_string(pos1) + ") in " +
                                 fasta_path_);
    }
    return up(static_cast<char>(c));
}

}  // namespace steppe::io
