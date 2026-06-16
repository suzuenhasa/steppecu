// src/io/snp_reader.cpp
//
// .snp parse (architecture.md §5 S0; ROADMAP M1). Reads chromosome, genetic
// position (Morgans, as-read), and ref/alt alleles per SNP in file order, capped
// to the first `max_snps`. The genetic position feeds M3's shared
// block_partition_rule unchanged; this reader does not assign blocks itself.
//
// LAYERING: `io`-leaf TU (architecture.md §4) — pure host C++20, no CUDA, no
// core/device dependency.
#include "io/snp_reader.hpp"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace steppe::io {

namespace {

// EIGENSTRAT chromosome-label conventions for the common non-numeric codes, so
// adjacent-equality (all the block rule consumes) is well-defined. Numeric codes
// pass through as their integer value; X/Y/MT take the EIGENSOFT codes; any other
// non-numeric label gets a stable, distinct negative sentinel.
int chrom_code(const std::string& tok, std::map<std::string, int>& other_codes,
               int& next_other) {
    bool numeric = !tok.empty();
    for (char c : tok) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { numeric = false; break; }
    }
    if (numeric) {
        return std::stoi(tok);
    }
    if (tok == "X" || tok == "x") return 23;
    if (tok == "Y" || tok == "y") return 24;
    if (tok == "MT" || tok == "mt" || tok == "M") return 90;
    auto it = other_codes.find(tok);
    if (it != other_codes.end()) return it->second;
    const int code = next_other--;  // distinct negative sentinel per new label
    other_codes.emplace(tok, code);
    return code;
}

}  // namespace

SnpTable read_snp(const std::string& path, std::size_t max_snps) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_snp: cannot open .snp file: " + path);
    }

    SnpTable t;
    std::map<std::string, int> other_codes;
    int next_other = -1;
    std::string line;
    while (t.count < max_snps && std::getline(in, line)) {
        std::istringstream ls(line);
        std::string id, chrom_tok;
        double genpos = 0.0;
        std::string physpos;
        std::string ref_tok, alt_tok;
        if (!(ls >> id >> chrom_tok >> genpos >> physpos >> ref_tok >> alt_tok)) {
            // Tolerate a record without explicit alleles (rare); require at least
            // id, chrom, genpos. Missing alleles default to 'N'.
            std::istringstream ls2(line);
            if (!(ls2 >> id >> chrom_tok >> genpos)) continue;  // blank/short → skip
            ref_tok = "N";
            alt_tok = "N";
        }
        t.id.push_back(id);
        t.chrom.push_back(chrom_code(chrom_tok, other_codes, next_other));
        t.genpos_morgans.push_back(genpos);
        t.ref.push_back(ref_tok.empty() ? 'N' : ref_tok[0]);
        t.alt.push_back(alt_tok.empty() ? 'N' : alt_tok[0]);
        ++t.count;
    }
    return t;
}

}  // namespace steppe::io
