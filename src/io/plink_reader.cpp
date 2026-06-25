// src/io/plink_reader.cpp
//
// PLINK .bim / .fam parsers (M-FR PLINK; docs/design/format-readers.md §3.2, §3.3).
// The PLINK twin of snp_reader / ind_reader: read_bim produces the SAME io::SnpTable
// and read_fam the SAME io::IndPartition, so everything downstream of the parse
// (filter / block partition / decode / selection) is byte-for-byte unchanged. The
// ONLY differences are the COLUMN ORDER (.bim swaps chrom/id vs .snp) and the
// population label COLUMN (.fam egroup = col 6 / phenotype, per AT2 mcio.c:1180-1205, vs
// .ind pop = col 3), plus the canonical polarity decision ref := A1 (the .bed counts A1
// copies; format-readers.md §3.2).
//
// The chromosome-label convention (numeric pass-through; X/Y/MT -> the EIGENSOFT
// codes single-homed in eigenstrat_format.hpp; other -> a stable negative sentinel)
// and the genetic-position parse (locale-free from_chars + non-finite guard) are the
// SAME conventions read_snp uses, replicated here against the SAME single-homed format
// constants so the .bim block partition matches the .snp one EXACTLY without modifying
// the golden-gated snp_reader.cpp.
//
// LAYERING: `io`-leaf TU (architecture.md §4) — pure host C++20, no CUDA, no
// core/device dependency.
#include "io/plink_reader.hpp"

#include "io/eigenstrat_format.hpp"  // kChromCodeX/Y/Mt, kFirstOtherChromCode

#include <algorithm>
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
#include <unordered_set>
#include <vector>

namespace steppe::io {

namespace {

// .bim / .snp share the EIGENSTRAT chromosome-label convention. These helpers mirror
// snp_reader.cpp's anon-namespace helpers EXACTLY (same from_chars-based, throw-free
// numeric parse; same X/Y/MT -> single-homed code mapping; same negative sentinel for
// other labels) so the .bim-derived block partition is identical to the .snp one. They
// are replicated rather than shared because snp_reader's helpers are file-local to its
// golden-gated TU; the FORMAT CONVENTION (the codes) is single-homed in
// eigenstrat_format.hpp, which both consume.

// Parse the WHOLE token as a T via std::from_chars (locale-free, throw-free); true iff
// the parse succeeded AND consumed the entire token (snp_reader's parse_full twin).
template <class T>
[[nodiscard]] bool parse_full(const std::string& tok, T& out) {
    const char* begin = tok.data();
    const char* end = tok.data() + tok.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

// EIGENSTRAT chromosome-label -> integer code (snp_reader's chrom_code twin): numeric
// labels pass through as their value; X/Y/MT take the single-homed EIGENSOFT codes; any
// other label (or an all-digit label too large for int) gets a stable distinct negative
// sentinel. Only adjacent-equality matters to the block rule, so a sentinel is correct.
int chrom_code(const std::string& tok, std::map<std::string, int>& other_codes,
               int& next_other) {
    bool numeric = !tok.empty();
    for (char c : tok) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { numeric = false; break; }
    }
    if (numeric) {
        int value = 0;
        if (parse_full(tok, value)) return value;  // success; overflow falls through
    }
    if (tok == "X" || tok == "x") return kChromCodeX;
    if (tok == "Y" || tok == "y") return kChromCodeY;
    if (tok == "MT" || tok == "mt" || tok == "M") return kChromCodeMt;
    auto it = other_codes.find(tok);
    if (it != other_codes.end()) return it->second;
    const int code = next_other--;  // distinct negative sentinel per new label
    other_codes.emplace(tok, code);
    return code;
}

// Whitespace tokenize (operator>> skips any whitespace run; matches the .snp/.ind readers).
[[nodiscard]] std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream ls(line);
    std::string tok;
    while (ls >> tok) tokens.push_back(tok);
    return tokens;
}

// Parse the genetic-position token (Morgans, as-read) with from_chars + an explicit
// non-finite guard (snp_reader's parse_genpos twin): trailing garbage / overflow /
// NaN / Inf are rejected so the value can never reach core::block_of as UB.
[[nodiscard]] double parse_genpos(const std::string& tok, std::size_t line_no) {
    double value = 0.0;
    if (!parse_full(tok, value) || !std::isfinite(value)) {
        throw std::runtime_error(
            "io::read_bim: malformed genetic position \"" + tok +
            "\" at line " + std::to_string(line_no));
    }
    return value;
}

// Column layout of a PLINK .bim record (6 cols): chrom id genpos physpos A1 A2.
constexpr std::size_t kBimFields = 6;
constexpr std::size_t kBimChromCol = 0;
constexpr std::size_t kBimIdCol = 1;
constexpr std::size_t kBimGenposCol = 2;
constexpr std::size_t kBimA1Col = 4;  // A1 -> canonical ref (the .bed counts A1 copies)
constexpr std::size_t kBimA2Col = 5;  // A2 -> canonical alt
// The PLINK "missing/unknown base" the .bim writes for an absent allele ('0'); mapped to
// the EIGENSTRAT missing-allele 'N' so the downstream class/transversion filter (which
// speaks the .snp 'N' convention) treats a PLINK no-call allele identically.
constexpr char kPlinkMissingAllele = '0';
constexpr char kCanonMissingAllele = 'N';

// Column layout of a PLINK .fam record (6 cols): FID IID pat mat sex pheno. The
// population (egroup) label is column 6 (the PHENOTYPE, 0-based index 5), per AT2's own
// PLINK reader (admixtools mcio.c:1180-1205: col6 "1"->Control, "2"->Case, "9"/"-9"/"0"
// -> ignore, otherwise the literal string IS the population egroup). convertf PACKEDPED
// with `outputgroup: YES` writes the EIGENSTRAT population label into THIS column
// (mcio.c:3940/3997). The FID (col 1) is a numeric counter convertf makes up
// (mcio.c:3917 `i+1`), NOT the population — so it is NOT used for grouping. This matches
// AT2 byte-for-byte: a PLINK file AT2 reads groups by col6, and so does steppe.
constexpr std::size_t kFamFields = 6;
constexpr std::size_t kFamGroupCol = 5;  // the 6th column (phenotype) — the egroup/population
// AT2 case/control + ignore phenotype sentinels (mcio.c:1180-1205). A col6 of "1"/"2" is
// the case/control label (Control/Case); "9"/"-9"/"0" marks the individual ignored (it is
// dropped from the partition exactly as AT2 drops it); any OTHER string is the population.
constexpr const char* kFamControlPheno = "1";
constexpr const char* kFamCasePheno = "2";

// One .bim allele (col): the single-char allele, mapping the PLINK '0' no-call to 'N'.
[[nodiscard]] char bim_allele(const std::vector<std::string>& fields, std::size_t col) {
    if (fields[col].empty()) return kCanonMissingAllele;
    const char a = fields[col][0];
    return a == kPlinkMissingAllele ? kCanonMissingAllele : a;
}

}  // namespace

SnpTable read_bim(const std::string& path, std::size_t max_snps) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_bim: cannot open .bim file: " + path);
    }

    SnpTable table;
    std::map<std::string, int> other_codes;
    int next_other = kFirstOtherChromCode;  // distinct negative codes, decrementing
    std::string line;
    std::size_t line_no = 0;  // 1-based, counts EVERY physical line for diagnostics
    while (table.count < max_snps && std::getline(in, line)) {
        ++line_no;
        const std::vector<std::string> fields = split_ws(line);

        // Tolerate a blank line ONLY at EOF (a common trailing newline); an interior
        // blank desyncs the SNP axis from the .bed (the .bim row index IS the SNP index).
        if (fields.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;  // trailing blank
            throw std::runtime_error(
                "io::read_bim: blank line at line " + std::to_string(line_no) +
                " (interior blank lines desync the SNP axis from the .bed)");
        }

        // A well-formed .bim record has EXACTLY 6 fields. PLINK always writes all 6
        // (chrom id genpos physpos A1 A2); fewer is a malformed record. Fail-fast with
        // the line number rather than silently dropping a row (which would shift every
        // later SNP's metadata relative to its .bed genotype).
        if (fields.size() < kBimFields) {
            throw std::runtime_error(
                "io::read_bim: malformed record (expected " + std::to_string(kBimFields) +
                " whitespace-separated fields <chrom> <id> <genpos> <physpos> <A1> <A2>, got " +
                std::to_string(fields.size()) + ") at line " + std::to_string(line_no));
        }

        const std::string& chrom_tok = fields[kBimChromCol];
        const std::string& id = fields[kBimIdCol];
        const double genpos = parse_genpos(fields[kBimGenposCol], line_no);  // throws on non-finite/garbage

        // CANONICAL POLARITY (format-readers.md §3.2): ref := A1, alt := A2. The .bed
        // 2-bit code counts A1 copies, so defining canonical ref as A1 makes the decode's
        // ref-copy count == the A1-copy count with NO per-SNP flip. A standalone PLINK
        // file is self-describing via the .bim; there is no external .snp ref to reconcile.
        const char ref = bim_allele(fields, kBimA1Col);  // A1 -> ref
        const char alt = bim_allele(fields, kBimA2Col);  // A2 -> alt

        table.id.push_back(id);
        table.chrom.push_back(chrom_code(chrom_tok, other_codes, next_other));
        table.genpos_morgans.push_back(genpos);
        table.ref.push_back(ref);
        table.alt.push_back(alt);
        ++table.count;
    }
    return table;
}

IndPartition read_fam(const std::string& path,
                      const PopSelection& sel,
                      std::size_t n_records_present) {
    const auto throw_io_error = [&](const std::string& msg) {
        throw std::runtime_error("io::read_fam: " + msg + path);
    };

    std::ifstream in(path);
    if (!in) {
        throw_io_error("cannot open .fam file: ");
    }

    // A population label, its first-appearance order (the AutoTopK tie-break), and its
    // member rows — the SAME RawGroup ind_reader uses, replicated file-local here.
    struct RawGroup {
        std::string label;
        std::size_t first_seen = 0;
        std::vector<std::size_t> rows;  // ascending individual-record indices
    };

    std::map<std::string, std::size_t> index_of;  // egroup label -> slot in `groups`
    std::vector<RawGroup> groups;
    // The .bed individual-record index. EVERY .fam line is a .bed individual record (the
    // universal PLINK invariant: .fam row i == .bed individual i, 1:1), so `row` increments
    // for every well-formed line — INCLUDING an "ignored" (col6 in {9,-9,0}) individual,
    // which is simply omitted from every population GROUP (so it is never selected) while
    // still consuming its .bed row index, keeping the remaining individuals aligned with
    // their .bed records.
    std::size_t row = 0;
    std::string line;
    while (std::getline(in, line)) {
        const std::vector<std::string> fields = split_ws(line);
        // A line with fewer than 6 fields is blank/short — skip without consuming a row
        // index (mirrors ind_reader's short-line skip; a real .fam always has 6 columns).
        if (fields.size() < kFamFields) continue;
        if (row >= n_records_present) {
            ++row;
            continue;  // beyond the present .bed records (partial-file cap)
        }
        // The population (egroup) is column 6 (the phenotype), per AT2 (mcio.c:1180-1205).
        // "9"/"-9"/"0" -> the individual is IGNORED (excluded from every group, never
        // selected) but still a .bed record; "1"/"2" -> the Control/Case label; any other
        // string -> the population. This matches AT2's PLINK grouping byte-for-byte.
        const std::string& pheno = fields[kFamGroupCol];
        const bool ignored = (pheno == "9" || pheno == "-9" || pheno == "0");
        if (!ignored) {
            std::string pop = pheno;
            if (pheno == kFamControlPheno) pop = "Control";
            else if (pheno == kFamCasePheno) pop = "Case";
            auto it = index_of.find(pop);
            if (it == index_of.end()) {
                index_of.emplace(pop, groups.size());
                groups.push_back(RawGroup{pop, groups.size(), {row}});
            } else {
                groups[it->second].rows.push_back(row);
            }
        }
        ++row;
    }

    IndPartition part;
    part.n_individuals_total = row;  // total .fam rows seen (the individual axis)

    if (groups.empty()) {
        throw_io_error("no individuals parsed from ");
    }

    // ---- Selection (byte-for-byte read_ind's three modes) ------------------------
    std::vector<const RawGroup*> selected;
    const auto filter_into = [&](auto pred) {
        for (const auto& g : groups) {
            if (pred(g)) selected.push_back(&g);
        }
    };

    switch (sel.mode) {
        case PopSelection::Mode::Explicit: {
            std::unordered_set<std::string> want(sel.labels.begin(), sel.labels.end());
            filter_into([&](const RawGroup& g) { return want.count(g.label) != 0; });
            break;
        }
        case PopSelection::Mode::AutoTopK: {
            std::vector<const RawGroup*> by_count;
            by_count.reserve(groups.size());
            for (const auto& g : groups) by_count.push_back(&g);
            std::stable_sort(by_count.begin(), by_count.end(),
                             [](const RawGroup* a, const RawGroup* b) {
                                 if (a->rows.size() != b->rows.size())
                                     return a->rows.size() > b->rows.size();
                                 return a->first_seen < b->first_seen;
                             });
            const std::size_t k = std::min(sel.k, by_count.size());
            selected.assign(by_count.begin(), by_count.begin() + static_cast<std::ptrdiff_t>(k));
            break;
        }
        case PopSelection::Mode::MinN: {
            filter_into([&](const RawGroup& g) { return g.rows.size() >= sel.min_n; });
            break;
        }
    }

    if (selected.empty()) {
        throw_io_error("population selection is empty for ");
    }

    // Final Q/V/N row order: sort the selected set ASCENDING by label (read_ind's
    // `sel = sorted(sel)`; std::string < is lexicographic, matching Python sorted()).
    std::sort(selected.begin(), selected.end(),
              [](const RawGroup* a, const RawGroup* b) { return a->label < b->label; });

    part.groups.reserve(selected.size());
    for (const RawGroup* g : selected) {
        part.groups.push_back(PopGroup{g->label, g->rows});  // rows already ascending
    }
    return part;
}

}  // namespace steppe::io
