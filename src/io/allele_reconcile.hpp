// src/io/allele_reconcile.hpp
//
// allele_reconcile — the shared strand/allele resolver hoisted out of
// vcf_reader.cpp so the native VCF reader AND the consumer-raw (23andMe /
// AncestryDNA / MyHeritage) reader resolve an observed nucleotide against a panel
// A1/A2 pair through ONE implementation (same strand first, then complement),
// exactly as the Stage-0 oracle (oracle.py reconcile(), spec §5c). De-duped from
// the two readers; behaviour is byte-identical to the original vcf_reader copy.
//
// Pure host C++20 io-leaf, standard library only (no zlib, no CUDA).
#ifndef STEPPE_IO_ALLELE_RECONCILE_HPP
#define STEPPE_IO_ALLELE_RECONCILE_HPP

#include <cctype>

namespace steppe::io {

[[nodiscard]] inline char up(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

[[nodiscard]] inline char complement(char b) {
    switch (up(b)) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default: return 'N';
    }
}

// reconcile(base, A1, A2) -> which in {+1=A1, 0=A2, -1=none}, flip flag.
// Same strand first, then complement (oracle.py reconcile(), spec §5c).
struct Recon {
    int which = -1;  // +1 == A1, 0 == A2, -1 == neither
    bool flip = false;
};

[[nodiscard]] inline Recon reconcile(char base, char a1, char a2) {
    const char b = up(base);
    if (b == a1) return {1, false};
    if (b == a2) return {0, false};
    const char cb = complement(b);
    if (cb == a1) return {1, true};
    if (cb == a2) return {0, true};
    return {-1, false};
}

}  // namespace steppe::io

#endif  // STEPPE_IO_ALLELE_RECONCILE_HPP
