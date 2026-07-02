// src/io/eigenstrat_format.cpp
//
// Parses the fixed-size header of a packed EIGENSTRAT .geno file (the magic plus
// the n_ind/n_snp counts) and derives the per-record stride from those counts.
// Pure host C++20 io-leaf: no CUDA, no core/device dependency.
#include "io/eigenstrat_format.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace steppe::io {

GenoHeader parse_geno_header(const std::array<char, kGenoHeaderBytes>& head) noexcept {
    GenoHeader h;

    const auto fail = [&]() -> GenoHeader {
        h.format = GenoFormat::Unknown;
        return h;
    };

    std::string text;
    text.reserve(kGenoHeaderBytes);
    for (char c : head) {
        if (c == '\0') break;
        text.push_back(c);
    }

    std::size_t pos = text.find_first_not_of(" \t");
    if (pos == std::string::npos) return fail();
    const std::size_t magic_end = text.find_first_of(" \t", pos);
    const std::string magic = text.substr(pos, magic_end - pos);
    if (magic == kMagicTgeno) {
        h.format = GenoFormat::Tgeno;
    } else if (magic == kMagicGeno) {
        h.format = GenoFormat::Geno;
    } else {
        return fail();
    }

    constexpr int kHeaderCounts = 2;
    std::size_t counts[kHeaderCounts] = {0, 0};
    int got = 0;
    std::size_t i = magic_end;
    while (i < text.size() && got < kHeaderCounts) {
        while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
        if (i >= text.size()) break;
        std::size_t v = 0;
        bool any = false;
        bool overflow = false;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            const std::size_t d = static_cast<std::size_t>(text[i] - '0');
            constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
            if (v > (kMax - d) / 10u) {
                overflow = true;
            } else {
                v = v * 10u + d;
            }
            any = true;
            ++i;
        }
        if (overflow) {
            return fail();
        }
        if (any) counts[got++] = v;
    }
    if (got < kHeaderCounts) {
        return fail();
    }

    h.n_ind = counts[0];
    h.n_snp = counts[1];

    if (h.format == GenoFormat::Tgeno) {
        h.header_bytes = kGenoHeaderBytes;
        h.n_records = h.n_ind;
        h.bytes_per_record = packed_bytes(h.n_snp);
    } else {
        h.bytes_per_record = std::max(kGenoHeaderBytes, packed_bytes(h.n_ind));
        h.header_bytes = h.bytes_per_record;
        h.n_records = h.n_snp;
    }
    return h;
}

}  // namespace steppe::io
