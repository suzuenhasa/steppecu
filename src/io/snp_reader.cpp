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

#include "io/eigenstrat_format.hpp"  // kChromCodeX / kChromCodeY / kChromCodeMt

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

// EIGENSTRAT chromosome-label conventions for the common non-numeric codes, so
// adjacent-equality (all the block rule consumes) is well-defined. Numeric codes
// pass through as their integer value; X/Y/MT take the EIGENSOFT codes named in
// eigenstrat_format.hpp (kChromCodeX/Y/Mt — single-homed there because the M2
// autosomes-only filter drops exactly these, cleanup F12/B16); any other
// non-numeric label (or an all-digit label too large for int) gets a stable,
// distinct negative sentinel.
//
// The numeric parse uses std::from_chars, NOT std::stoi: [charconv.from.chars] is
// locale-free, allocation-free, and — load-bearing for read_snp's documented
// `runtime_error`-only contract — reports failure through an out-parameter rather
// than THROWING. std::stoi throws std::out_of_range (a logic_error, NOT a
// runtime_error) on an all-digit token that exceeds INT_MAX, e.g. "99999999999",
// which would escape read_snp uncaught and uncontextualized — violating the §10
// fail-fast "io malformed-input carries context" contract and the header's
// "Throws std::runtime_error on …" promise (cleanup snp_reader F2/B15;
// https://en.cppreference.com/cpp/string/basic_string/stol). With from_chars an
// overflowing all-digit token (errc::result_out_of_range) instead falls through to
// the negative-sentinel path: only adjacent-equality between SNPs matters to the
// block rule (block_partition_rule.hpp), so a stable distinct sentinel for a
// pathological code is correct and never throws.
int chrom_code(const std::string& tok, std::map<std::string, int>& other_codes,
               int& next_other) {
    bool numeric = !tok.empty();
    for (char c : tok) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { numeric = false; break; }
    }
    if (numeric) {
        int value = 0;
        const char* begin = tok.data();
        const char* end = tok.data() + tok.size();
        const auto [ptr, ec] = std::from_chars(begin, end, value);
        // The all-digit check above guarantees a non-empty, sign-free, fully-
        // consumed run, so the only possible failure is errc::result_out_of_range
        // (token > INT_MAX). On success return the integer; on overflow fall
        // through to the sentinel path below (never throw — B15).
        if (ec == std::errc{} && ptr == end) {
            return value;
        }
    }
    // X/Y/MT take the EIGENSOFT codes single-homed in eigenstrat_format.hpp
    // (the autosomes-only filter drops exactly these — config.hpp kAutosomeChromMax).
    if (tok == "X" || tok == "x") return kChromCodeX;
    if (tok == "Y" || tok == "y") return kChromCodeY;
    if (tok == "MT" || tok == "mt" || tok == "M") return kChromCodeMt;
    auto it = other_codes.find(tok);
    if (it != other_codes.end()) return it->second;
    const int code = next_other--;  // distinct negative sentinel per new label
    other_codes.emplace(tok, code);
    return code;
}

// Split a .snp line into whitespace-separated tokens (the oracle uses line.split(),
// any-whitespace; operator>> skips any whitespace run identically — see the
// "Considered & rejected" note in docs/cleanup/io-snp_reader.md). The token COUNT
// then drives the column decision deterministically (B14), replacing the old
// extraction-failure fall-through that silently `continue`d on a short line and
// could desync the SNP axis from the .geno.
[[nodiscard]] std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream ls(line);
    std::string tok;
    while (ls >> tok) tokens.push_back(tok);
    return tokens;
}

// Parse the genetic-position token (Morgans, as-read) with std::from_chars: it is
// locale-free, allocation-free, and throws nothing (architecture.md §12 wants a
// correctly-rounded decimal→double parse matching the oracle/AT2; libstdc++'s
// from_chars float backend is correctly-rounded). The whole token must be consumed
// (ptr == end), so trailing garbage like "0.5x" and overflow (errc::result_out_of_
// range, e.g. "1e400") are rejected.
//
// NON-FINITE guard is EXPLICIT, not delegated to from_chars: [charconv.from.chars]
// does not forbid the C99 "inf"/"nan" forms, and libstdc++ (GCC 13) DOES accept
// them (verified on the box: from_chars("nan"/"inf") returns errc{} with the whole
// token consumed). A non-finite genpos must never reach core::block_of, where
// static_cast<int>(std::floor(NaN_or_Inf / bs)) is undefined behavior that
// silently corrupts the parity-critical block partition. So after a successful
// parse we reject !std::isfinite(value) outright (this also catches the overflowed
// HUGE_VAL on implementations that map overflow to ±inf rather than out_of_range).
[[nodiscard]] double parse_genpos(const std::string& tok, std::size_t line_no) {
    double value = 0.0;
    const char* begin = tok.data();
    const char* end = tok.data() + tok.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end || !std::isfinite(value)) {
        throw std::runtime_error(
            "io::read_snp: malformed genetic position \"" + tok +
            "\" at line " + std::to_string(line_no));
    }
    return value;
}

}  // namespace

SnpTable read_snp(const std::string& path, std::size_t max_snps) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_snp: cannot open .snp file: " + path);
    }

    SnpTable table;
    std::map<std::string, int> other_codes;
    int next_other = kFirstOtherChromCode;  // distinct negative codes, decrementing
    std::string line;
    std::size_t line_no = 0;  // 1-based, counts EVERY physical line for diagnostics
    while (table.count < max_snps && std::getline(in, line)) {
        ++line_no;
        const std::vector<std::string> fields = split_ws(line);

        // A truly empty/whitespace-only line carries no SNP record. The .snp row
        // index IS the SNP index (positional), so we tolerate a blank line ONLY at
        // EOF (a common trailing newline); a blank line followed by more records
        // would desync the SNP axis from the .geno, so it is a format error.
        if (fields.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;  // trailing blank
            throw std::runtime_error(
                "io::read_snp: blank line at line " + std::to_string(line_no) +
                " (interior blank lines desync the SNP axis from the .geno)");
        }

        // Token-count column decision (B14): EIGENSTRAT .snp is
        //   <id> <chrom> <genpos> [<physpos> <ref> <alt>]
        // so a well-formed record has >= 3 fields. A record with all 6 columns
        // carries explicit alleles (cols 5,6); a >=3-but-<6 record legitimately
        // omits the alleles (they default to 'N'). Fewer than 3 fields is a
        // malformed record — fail-fast with the line number rather than the old
        // silent `continue` (which dropped the row and shifted every later SNP's
        // metadata relative to its genotype).
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
        const double genpos = parse_genpos(fields[2], line_no);  // throws if non-finite/garbage
        // Alleles present only when the full 6-column record is given (cols 5,6);
        // otherwise default to the EIGENSTRAT "missing/unknown base" 'N'.
        const bool has_alleles = fields.size() >= kFullSnpFields;
        const char ref =
            has_alleles && !fields[kRefAlleleCol].empty() ? fields[kRefAlleleCol][0] : kMissingAllele;
        const char alt =
            has_alleles && !fields[kAltAlleleCol].empty() ? fields[kAltAlleleCol][0] : kMissingAllele;

        table.id.push_back(id);
        table.chrom.push_back(chrom_code(chrom_tok, other_codes, next_other));
        table.genpos_morgans.push_back(genpos);
        table.ref.push_back(ref);
        table.alt.push_back(alt);
        ++table.count;
    }
    return table;
}

}  // namespace steppe::io
