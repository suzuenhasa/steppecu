// src/io/vcf_panel_decode.hpp
//
// The pure decode rules shared by the CPU and GPU phased-VCF panel readers — the
// single source of truth for the field-exact contract both paths must reproduce
// BIT-IDENTICALLY:
//   * strip_chr / chrom_to_int   — the CHROM token normalisation + .snp int code
//   * is_snp_allele              — the biallelic-SNP allele filter (`-v snps`)
//   * hap_code                   — the load-bearing {0,2,3} haplotype code map
//   * diploid_dosage_code        — the phase-agnostic {0,1,2,3} hardcall dosage map
//   * ChromMap / read_genetic_map / interp_morgans — the genetic-map join
//
// hap_code and diploid_dosage_code are ALSO re-implemented on-device (in the GPU
// GT-parse kernels) to match these bytes; keeping the host definitions here
// documents the exact rules the kernels mirror. Pure host C++20 io-leaf (standard
// library only, no CUDA).
#ifndef STEPPE_IO_VCF_PANEL_DECODE_HPP
#define STEPPE_IO_VCF_PANEL_DECODE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "io/eigenstrat_format.hpp"
#include "io/vcf_record.hpp"

namespace steppe::io {

namespace vpd = vcfdetail;

// Strip a leading 'chr'/'CHR' from a CHROM token (phase-3 uses bare "22", but be
// permissive). Returns a view over the original buffer.
[[nodiscard]] inline std::string_view strip_chr(std::string_view s) {
    if (s.size() > 3 && (s[0] == 'c' || s[0] == 'C') && (s[1] == 'h' || s[1] == 'H') &&
        (s[2] == 'r' || s[2] == 'R')) {
        s.remove_prefix(3);
    }
    return s;
}

// Map a 'chr'-stripped CHROM token to the .snp integer convention (numeric, or
// X/Y/MT -> 23/24/90, else -1). Only used for SnpTable.chrom; region filtering
// compares the string form.
[[nodiscard]] inline int chrom_to_int(std::string_view stripped) {
    if (stripped == "X" || stripped == "x") return kChromCodeX;
    if (stripped == "Y" || stripped == "y") return kChromCodeY;
    if (stripped == "MT" || stripped == "mt" || stripped == "M" || stripped == "m")
        return kChromCodeMt;
    const auto v = vpd::parse_int(stripped);
    return v ? static_cast<int>(*v) : kFirstOtherChromCode;
}

// A single-base SNP allele: exactly one A/C/G/T/N nucleotide (rejects indels,
// '*' spanning deletions, '<SYMBOLIC>', and '.'). Mirrors bcftools `-v snps`.
[[nodiscard]] inline bool is_snp_allele(std::string_view a) {
    if (a.size() != 1) return false;
    switch (a[0]) {
        case 'A': case 'C': case 'G': case 'T': case 'N':
        case 'a': case 'c': case 'g': case 't': case 'n':
            return true;
        default:
            return false;
    }
}

// The per-haplotype allele-token -> 2-bit code map (the load-bearing {0,2,3}
// contract): '0'->0, '1'->2, everything else (".", multi-digit, empty) -> 3
// (missing). NEVER emits code 1. The GPU kernel re-implements this exactly.
[[nodiscard]] inline std::uint8_t hap_code(std::string_view allele) {
    if (allele.size() == 1) {
        if (allele[0] == '0') return 0u;
        if (allele[0] == '1') return 2u;
    }
    return kMissingCode;  // 3
}

// The phase-agnostic diploid-dosage code for a biallelic GT's two allele tokens
// (the HARDCALL path's per-sample rule): both '0' -> 0; exactly one '1' (het, either
// order, phased OR unphased '0|1'=='0/1'=='1|0') -> 1; both '1' -> 2; any missing '.'
// / multi-digit / malformed allele -> 3. Mirrors the bcftools oracle GT ->
// {0/0:0, het:1, 1/1:2, else:3}. Used by BOTH the CPU and GPU hardcall readers (the
// GPU kernel re-implements it byte-exact). Unlike hap_code, an unphased het is a
// LEGITIMATE dosage-1 call here — nothing is dropped for lacking a phase.
[[nodiscard]] inline std::uint8_t diploid_dosage_code(std::string_view a0,
                                                      std::string_view a1) {
    const auto bit = [](std::string_view a) -> int {
        if (a.size() == 1) {
            if (a[0] == '0') return 0;
            if (a[0] == '1') return 1;
        }
        return -1;  // '.', a multiallelic index, or malformed
    };
    const int b0 = bit(a0);
    const int b1 = bit(a1);
    if (b0 < 0 || b1 < 0) return kMissingCode;  // 3
    return static_cast<std::uint8_t>(b0 + b1);  // 0 / 1 / 2  (het = 0 + 1 = 1)
}

// A per-chromosome genetic map: (bp, cM) sorted ascending by bp, for linear
// interpolation to Morgans.
struct ChromMap {
    std::vector<long long> bp;
    std::vector<double> cm;
};

// Parse a plink-format .map (whitespace: chrom id cM bp) into per-chrom (bp,cM)
// tables. Non-conforming / header lines (a bp or cM token that will not parse)
// are skipped. Kept sorted by bp per chromosome.
[[nodiscard]] inline std::unordered_map<int, ChromMap> read_genetic_map(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_vcf_panel: cannot open --map genetic map: " + path);
    }
    std::unordered_map<int, ChromMap> maps;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string chrom_tok, id_tok, cm_tok, bp_tok;
        if (!(ls >> chrom_tok >> id_tok >> cm_tok >> bp_tok)) continue;  // need 4 columns
        char* endp = nullptr;
        const double cm = std::strtod(cm_tok.c_str(), &endp);
        if (endp != cm_tok.c_str() + cm_tok.size()) continue;  // header / non-numeric cM
        const auto bp = vpd::parse_int(bp_tok);
        if (!bp) continue;
        const int chrom = chrom_to_int(strip_chr(chrom_tok));
        ChromMap& m = maps[chrom];
        m.bp.push_back(*bp);
        m.cm.push_back(cm);
    }
    for (auto& [chrom, m] : maps) {
        (void)chrom;
        std::vector<std::size_t> order(m.bp.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) { return m.bp[a] < m.bp[b]; });
        ChromMap sorted;
        sorted.bp.reserve(m.bp.size());
        sorted.cm.reserve(m.cm.size());
        for (std::size_t i : order) {
            sorted.bp.push_back(m.bp[i]);
            sorted.cm.push_back(m.cm[i]);
        }
        m = std::move(sorted);
    }
    return maps;
}

// Linear-interpolate cM at bp within a per-chrom map, clamping to the endpoints
// (keeps genpos monotonic non-decreasing, which paint requires). Returns Morgans.
[[nodiscard]] inline double interp_morgans(const ChromMap& m, long long bp) {
    if (m.bp.empty()) return 0.0;
    if (bp <= m.bp.front()) return m.cm.front() / 100.0;
    if (bp >= m.bp.back()) return m.cm.back() / 100.0;
    // upper_bound: first entry with bp_i > bp
    const auto it = std::upper_bound(m.bp.begin(), m.bp.end(), bp);
    const std::size_t hi = static_cast<std::size_t>(it - m.bp.begin());
    const std::size_t lo = hi - 1;
    const long long b0 = m.bp[lo];
    const long long b1 = m.bp[hi];
    const double c0 = m.cm[lo];
    const double c1 = m.cm[hi];
    const double frac = (b1 == b0) ? 0.0
                                   : static_cast<double>(bp - b0) / static_cast<double>(b1 - b0);
    return (c0 + (c1 - c0) * frac) / 100.0;
}

}  // namespace steppe::io

#endif  // STEPPE_IO_VCF_PANEL_DECODE_HPP
