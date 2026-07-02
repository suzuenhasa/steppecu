// src/io/snp_reader.cpp
//
// Parses an EIGENSTRAT .snp file into a SnpTable of parallel per-SNP arrays in
// file order. The row index is the SNP index, so the reader never silently
// skips a record: anything unparseable is a hard error or a well-defined
// default. Pure host C++20, no CUDA or core/device dependency.
//
// Reference: docs/reference/src_io_snp_reader.cpp.md
#include "io/snp_reader.hpp"

#include "io/eigenstrat_format.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace steppe::io {

namespace {

// Shared numeric-parse contract — reference §2
template <class T>
[[nodiscard]] bool parse_full(const std::string& tok, T& out) {
    const char* begin = tok.data();
    const char* end = tok.data() + tok.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

// Chromosome codes — reference §5
int chrom_code(const std::string& tok, std::map<std::string, int>& other_codes,
               int& next_other) {
    bool numeric = !tok.empty();
    for (char c : tok) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { numeric = false; break; }
    }
    if (numeric) {
        int value = 0;
        if (parse_full(tok, value)) {
            return value;
        }
    }
    if (tok == "X" || tok == "x") return kChromCodeX;
    if (tok == "Y" || tok == "y") return kChromCodeY;
    if (tok == "MT" || tok == "mt" || tok == "M") return kChromCodeMt;
    auto it = other_codes.find(tok);
    if (it != other_codes.end()) return it->second;
    const int code = next_other--;
    other_codes.emplace(tok, code);
    return code;
}

// Splitting a record into tokens — reference §6
[[nodiscard]] std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream ls(line);
    std::string tok;
    while (ls >> tok) tokens.push_back(tok);
    return tokens;
}

// Genetic position: strict, finite-checked — reference §3
[[nodiscard]] double parse_genpos(const std::string& tok, std::size_t line_no) {
    double value = 0.0;
    if (!parse_full(tok, value) || !std::isfinite(value)) {
        throw std::runtime_error(
            "io::read_snp: malformed genetic position \"" + tok +
            "\" at line " + std::to_string(line_no));
    }
    return value;
}

// Physical position: lenient, degrades to zero — reference §4
[[nodiscard]] double parse_physpos(const std::string& tok) {
    double value = 0.0;
    if (!parse_full(tok, value) || !std::isfinite(value)) return 0.0;
    return value;
}

}  // namespace

// Reading a record — reference §6
SnpTable read_snp(const std::string& path, std::size_t max_snps) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_snp: cannot open .snp file: " + path);
    }

    SnpTable table;
    std::map<std::string, int> other_codes;
    int next_other = kFirstOtherChromCode;
    std::string line;
    std::size_t line_no = 0;
    while (table.count < max_snps && std::getline(in, line)) {
        ++line_no;
        const std::vector<std::string> fields = split_ws(line);

        if (fields.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;
            throw std::runtime_error(
                "io::read_snp: blank line at line " + std::to_string(line_no) +
                " (interior blank lines desync the SNP axis from the .geno)");
        }

        if (fields.size() < kMinSnpFields) {
            throw std::runtime_error(
                "io::read_snp: malformed record (expected >= " +
                std::to_string(kMinSnpFields) +
                " whitespace-separated fields <id> <chrom> <genpos>, got " +
                std::to_string(fields.size()) + ") at line " +
                std::to_string(line_no));
        }

        const std::string& id = fields[0];
        const std::string& chrom_tok = fields[1];
        const double genpos = parse_genpos(fields[2], line_no);
        const double physpos =
            fields.size() > kPhysposCol ? parse_physpos(fields[kPhysposCol]) : 0.0;
        const bool has_alleles = fields.size() >= kFullSnpFields;
        const auto allele = [&](std::size_t col) {
            return has_alleles && !fields[col].empty() ? fields[col][0] : kMissingAllele;
        };
        const char ref = allele(kRefAlleleCol);
        const char alt = allele(kAltAlleleCol);

        table.id.push_back(id);
        table.chrom.push_back(chrom_code(chrom_tok, other_codes, next_other));
        table.genpos_morgans.push_back(genpos);
        table.physpos.push_back(physpos);
        table.ref.push_back(ref);
        table.alt.push_back(alt);
        ++table.count;
    }
    return table;
}

}  // namespace steppe::io
