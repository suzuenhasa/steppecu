// src/io/eigenstrat_format.cpp
//
// Header parse for packed EIGENSTRAT .geno files (architecture.md §5 S0; ROADMAP
// §4, M1). The format magic + the two decimal counts after it are parsed here;
// the record stride is DERIVED from those counts via the `io` format constants
// (eigenstrat_format.hpp), never hardcoded at a call site.
//
// LAYERING: `io`-leaf TU (architecture.md §4) — pure host C++20, no CUDA, no
// core/device dependency.
#include "io/eigenstrat_format.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>

namespace steppe::io {

GenoHeader parse_geno_header(const std::array<char, kGenoHeaderBytes>& head) noexcept {
    GenoHeader h;

    // The header is "<MAGIC> <n_ind> <n_snp> <hash> <hash>" NUL-padded to
    // kGenoHeaderBytes. Read up to the first NUL, then tokenize on whitespace.
    std::string text;
    text.reserve(kGenoHeaderBytes);
    for (char c : head) {
        if (c == '\0') break;
        text.push_back(c);
    }

    // First token = magic. Determine the format (TGENO individual-major vs GENO
    // SNP-major PACKEDANCESTRYMAP). A trailing-substring check keeps it robust to
    // surrounding whitespace; we match the leading token exactly.
    std::size_t pos = text.find_first_not_of(" \t");
    if (pos == std::string::npos) return h;  // empty → Unknown
    const std::size_t magic_end = text.find_first_of(" \t", pos);
    const std::string magic = text.substr(pos, magic_end - pos);
    if (magic == "TGENO") {
        h.format = GenoFormat::Tgeno;
    } else if (magic == "GENO") {
        h.format = GenoFormat::Geno;
    } else {
        return h;  // unrecognized magic → Unknown (caller fails loudly)
    }

    // Parse the first two decimal integers after the magic: n_ind, n_snp.
    std::size_t ints[2] = {0, 0};
    int got = 0;
    std::size_t i = magic_end;
    while (i < text.size() && got < 2) {
        while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
        if (i >= text.size()) break;
        std::size_t v = 0;
        bool any = false;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            v = v * 10u + static_cast<std::size_t>(text[i] - '0');
            any = true;
            ++i;
        }
        if (any) ints[got++] = v;
    }
    if (got < 2) {  // could not read both counts → leave format set but counts 0
        h.format = GenoFormat::Unknown;
        return h;
    }

    h.n_ind = ints[0];
    h.n_snp = ints[1];
    h.header_bytes = kGenoHeaderBytes;

    if (h.format == GenoFormat::Tgeno) {
        // Individual-major: one record per individual, ceil(n_snp/4) bytes.
        h.n_records = h.n_ind;
        h.bytes_per_record = packed_bytes(h.n_snp);
    } else {
        // SNP-major PACKEDANCESTRYMAP: one record per SNP, with the rlen floor
        // of kGenoHeaderBytes (EIGENSOFT: rlen = max(48, ceil(n_ind/4))).
        h.n_records = h.n_snp;
        h.bytes_per_record = std::max(kGenoHeaderBytes, packed_bytes(h.n_ind));
    }
    return h;
}

}  // namespace steppe::io
