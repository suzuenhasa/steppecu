// src/io/detail/snp_text_parse.hpp
//
// Shared SNP-text field parsers for the EIGENSTRAT .snp and PLINK .bim readers
// (byte-identical logic previously copy-pasted into both TUs). Pure host C++20.
//
// Reference: docs/reference/src_io_detail_snp_text_parse.hpp.md
#ifndef STEPPE_IO_DETAIL_SNP_TEXT_PARSE_HPP
#define STEPPE_IO_DETAIL_SNP_TEXT_PARSE_HPP

#include "io/eigenstrat_format.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace steppe::io::detail {

// Shared numeric-parse contract — reference §2
template <class T>
[[nodiscard]] inline bool parse_full(const std::string& tok, T& out) {
    const char* begin = tok.data();
    const char* end = tok.data() + tok.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

// Chromosome codes — reference §5
[[nodiscard]] inline int chrom_code(const std::string& tok,
                                    std::map<std::string, int>& other_codes,
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
[[nodiscard]] inline std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream ls(line);
    std::string tok;
    while (ls >> tok) tokens.push_back(tok);
    return tokens;
}

// Genetic position: strict, finite-checked — reference §3
[[nodiscard]] inline double parse_genpos(const std::string& tok, std::size_t line_no,
                                         const std::string& reader_name) {
    double value = 0.0;
    if (!parse_full(tok, value) || !std::isfinite(value)) {
        throw std::runtime_error(
            "io::" + reader_name + ": malformed genetic position \"" + tok +
            "\" at line " + std::to_string(line_no));
    }
    return value;
}

// Physical position: lenient, degrades to zero — reference §4
[[nodiscard]] inline double parse_physpos(const std::string& tok) {
    double value = 0.0;
    if (!parse_full(tok, value) || !std::isfinite(value)) return 0.0;
    return value;
}

}  // namespace steppe::io::detail

#endif  // STEPPE_IO_DETAIL_SNP_TEXT_PARSE_HPP
