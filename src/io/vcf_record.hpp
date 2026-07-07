// src/io/vcf_record.hpp
//
// Reference: docs/reference/src_io_vcf_record.hpp.md
//
// Small, field-exact VCF text-parse helpers, header-only. Keeps vcf_reader.cpp
// thin. Every extractor tokenizes on the correct delimiter and matches keys
// EXACTLY (INFO on ';' up to '=', FORMAT on ':') so a bare "DP=" never matches
// inside "MinDP=" and a FORMAT "DP" key never hits "AD" — the field-boundary
// robustness the oracle got for free from bcftools %INFO/%FORMAT accessors.
//
// Pure host C++20 io-leaf, standard library only.
#ifndef STEPPE_IO_VCF_RECORD_HPP
#define STEPPE_IO_VCF_RECORD_HPP

#include <cmath>     // std::isfinite — the GL/GP non-finite -> missing guard
#include <cstdlib>   // std::strtod — the GL/GP float parse (sign/decimal/scientific)
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace steppe::io::vcfdetail {

// Split a line on a single delimiter into string_views over the source buffer.
[[nodiscard]] inline std::vector<std::string_view> split(std::string_view s, char delim) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (;;) {
        const std::size_t d = s.find(delim, start);
        if (d == std::string_view::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, d - start));
        start = d + 1;
    }
    return out;
}

// Look up an INFO key (e.g. "END", "MinDP"). INFO is ';'-delimited; each token is
// KEY or KEY=VALUE. Returns the VALUE for KEY=VALUE (or "" for a bare flag) when
// KEY matches exactly up to '='; std::nullopt if absent.
[[nodiscard]] inline std::optional<std::string_view> info_field(std::string_view info,
                                                                std::string_view key) {
    if (info == ".") return std::nullopt;
    std::size_t start = 0;
    for (;;) {
        std::size_t semi = info.find(';', start);
        std::string_view tok =
            (semi == std::string_view::npos) ? info.substr(start) : info.substr(start, semi - start);
        const std::size_t eq = tok.find('=');
        const std::string_view k = (eq == std::string_view::npos) ? tok : tok.substr(0, eq);
        if (k == key) {
            return (eq == std::string_view::npos) ? std::string_view{} : tok.substr(eq + 1);
        }
        if (semi == std::string_view::npos) break;
        start = semi + 1;
    }
    return std::nullopt;
}

// Index of an exact FORMAT subfield key (":"-delimited), or -1 if absent.
[[nodiscard]] inline int format_index(std::string_view format, std::string_view key) {
    int idx = 0;
    std::size_t start = 0;
    for (;;) {
        std::size_t colon = format.find(':', start);
        std::string_view tok = (colon == std::string_view::npos)
                                   ? format.substr(start)
                                   : format.substr(start, colon - start);
        if (tok == key) return idx;
        if (colon == std::string_view::npos) break;
        start = colon + 1;
        ++idx;
    }
    return -1;
}

// The idx-th ':'-delimited subfield of a sample column, or "" if out of range.
[[nodiscard]] inline std::string_view subfield(std::string_view sample, int idx) {
    if (idx < 0) return {};
    int cur = 0;
    std::size_t start = 0;
    for (;;) {
        std::size_t colon = sample.find(':', start);
        std::string_view tok = (colon == std::string_view::npos)
                                   ? sample.substr(start)
                                   : sample.substr(start, colon - start);
        if (cur == idx) return tok;
        if (colon == std::string_view::npos) break;
        start = colon + 1;
        ++cur;
    }
    return {};
}

// Parse a non-negative integer field; returns nullopt on "." / empty / non-numeric.
[[nodiscard]] inline std::optional<long long> parse_int(std::string_view s) {
    if (s.empty() || s == ".") return std::nullopt;
    long long v = 0;
    bool any = false;
    for (char c : s) {
        if (c < '0' || c > '9') return std::nullopt;
        v = v * 10 + (c - '0');
        any = true;
    }
    return any ? std::optional<long long>{v} : std::nullopt;
}

// Parse a floating-point FORMAT value (FORMAT/GL is negative log10; FORMAT/GP may
// carry scientific notation like 1e-05) — accepting sign, decimal, and scientific
// forms, which the non-negative integer parse_int cannot. Returns nullopt on
// "." / empty / trailing garbage / a NON-FINITE value (the 'nan'/'inf'/'-inf'
// sentinels Beagle/GLIMPSE can write) so a fully-uninformative or extreme genotype
// flows to the MISSING path (present_mask=0, uninformative triplet) instead of
// propagating a NaN into the tensor (NaN sentinels are barred — critic fix #4).
[[nodiscard]] inline std::optional<double> parse_double(std::string_view s) {
    if (s.empty() || s == ".") return std::nullopt;
    const std::string buf(s);              // strtod needs a NUL-terminated buffer
    const char* b = buf.c_str();
    char* end = nullptr;
    const double v = std::strtod(b, &end);
    if (end != b + buf.size()) return std::nullopt;  // partial / trailing garbage
    if (!std::isfinite(v)) return std::nullopt;       // nan/inf -> missing path
    return v;
}

}  // namespace steppe::io::vcfdetail

#endif  // STEPPE_IO_VCF_RECORD_HPP
