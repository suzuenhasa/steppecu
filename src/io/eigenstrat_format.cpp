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
#include <limits>
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
    // SNP-major PACKEDANCESTRYMAP). We skip leading whitespace, isolate the leading
    // token (up to the next space/tab via find_first_of), and compare it EXACTLY (==)
    // against the known magics — so surrounding whitespace is tolerated while a magic
    // that merely contains "GENO" as a substring is NOT accepted.
    std::size_t pos = text.find_first_not_of(" \t");
    if (pos == std::string::npos) return h;  // empty → Unknown
    const std::size_t magic_end = text.find_first_of(" \t", pos);
    const std::string magic = text.substr(pos, magic_end - pos);
    if (magic == kMagicTgeno) {
        h.format = GenoFormat::Tgeno;
    } else if (magic == kMagicGeno) {
        h.format = GenoFormat::Geno;
    } else {
        return h;  // unrecognized magic → Unknown (caller fails loudly)
    }

    // Parse the first two decimal integers after the magic: n_ind, n_snp. The
    // count `2` is single-homed here so the array dimension, the parse-loop bound,
    // and the underflow guard cannot drift — a mismatch between the array size and
    // the loop bound would be an out-of-bounds write (DRY; NAMING-STYLE-STANDARD
    // §2.5 single-source; group-5 5.3).
    constexpr int kHeaderCounts = 2;  // # decimal counts after the magic (n_ind, n_snp)
    std::size_t ints[kHeaderCounts] = {0, 0};
    int got = 0;
    std::size_t i = magic_end;
    while (i < text.size() && got < kHeaderCounts) {
        while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
        if (i >= text.size()) break;
        std::size_t v = 0;
        bool any = false;
        bool overflow = false;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            // OVERFLOW-GUARD the decimal accumulation (cleanup eigenstrat_format
            // C2). `v = v*10 + d` on a std::size_t WRAPS modulo 2^N (well-defined
            // unsigned modular arithmetic, [basic.fundamental] — but SILENT), so a
            // malformed/adversarial header like "TGENO 9999...9 ..." (a digit run
            // far longer than any 48-byte file could legitimately hold) would
            // otherwise yield a wrapped-but-plausible n_ind/n_snp that flows into
            // packed_bytes() and geno_reader's size validation as a wrong-but-
            // plausible stride, NOT the documented Unknown. Detect the wrap up
            // front (v > (MAX - d)/10 ⇒ v*10 + d would exceed SIZE_MAX) and route
            // the whole header to Unknown so the caller fails loudly (fail-fast,
            // architecture.md §2; STEPPE_ERR_IO_FORMAT, §10).
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
        if (overflow) {  // an unrepresentable count is a malformed header → Unknown
            h.format = GenoFormat::Unknown;
            return h;
        }
        if (any) ints[got++] = v;
    }
    if (got < kHeaderCounts) {  // could not read both counts → malformed header, route to Unknown
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
