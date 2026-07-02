// src/io/plink_reader.cpp
//
// PLINK .bim / .fam parsers. read_bim yields the same io::SnpTable as snp_reader
// and read_fam the same io::IndPartition as ind_reader, so everything downstream
// is byte-for-byte unchanged; the only differences are the .bim column order, the
// canonical polarity (ref := A1, since the .bed counts A1 copies), and the .fam
// col-6 phenotype -> egroup mapping.
#include "io/plink_reader.hpp"

#include "io/eigenstrat_format.hpp"

#include <algorithm>
#include <array>
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

template <class T>
[[nodiscard]] bool parse_full(const std::string& tok, T& out) {
    const char* begin = tok.data();
    const char* end = tok.data() + tok.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

int chrom_code(const std::string& tok, std::map<std::string, int>& other_codes,
               int& next_other) {
    bool numeric = !tok.empty();
    for (char c : tok) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { numeric = false; break; }
    }
    if (numeric) {
        int value = 0;
        if (parse_full(tok, value)) return value;
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

[[nodiscard]] std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream ls(line);
    std::string tok;
    while (ls >> tok) tokens.push_back(tok);
    return tokens;
}

[[nodiscard]] double parse_genpos(const std::string& tok, std::size_t line_no) {
    double value = 0.0;
    if (!parse_full(tok, value) || !std::isfinite(value)) {
        throw std::runtime_error(
            "io::read_bim: malformed genetic position \"" + tok +
            "\" at line " + std::to_string(line_no));
    }
    return value;
}

[[nodiscard]] double parse_physpos(const std::string& tok) {
    double value = 0.0;
    if (!parse_full(tok, value) || !std::isfinite(value)) return 0.0;
    return value;
}

constexpr std::size_t kBimFields = 6;
constexpr std::size_t kBimChromCol = 0;
constexpr std::size_t kBimIdCol = 1;
constexpr std::size_t kBimGenposCol = 2;
constexpr std::size_t kBimPhysposCol = 3;
constexpr std::size_t kBimA1Col = 4;
constexpr std::size_t kBimA2Col = 5;
constexpr char kPlinkMissingAllele = '0';
constexpr char kCanonMissingAllele = 'N';

constexpr std::size_t kFamFields = 6;
constexpr std::size_t kFamGroupCol = 5;
constexpr const char* kFamControlPheno = "1";
constexpr const char* kFamCasePheno = "2";
constexpr const char* kControlLabel = "Control";
constexpr const char* kCaseLabel = "Case";
constexpr std::array<const char*, 3> kFamIgnorePhenos = {"9", "-9", "0"};

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
    int next_other = kFirstOtherChromCode;
    std::string line;
    std::size_t line_no = 0;
    while (table.count < max_snps && std::getline(in, line)) {
        ++line_no;
        const std::vector<std::string> fields = split_ws(line);

        if (fields.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;
            throw std::runtime_error(
                "io::read_bim: blank line at line " + std::to_string(line_no) +
                " (interior blank lines desync the SNP axis from the .bed)");
        }

        if (fields.size() < kBimFields) {
            throw std::runtime_error(
                "io::read_bim: malformed record (expected " + std::to_string(kBimFields) +
                " whitespace-separated fields <chrom> <id> <genpos> <physpos> <A1> <A2>, got " +
                std::to_string(fields.size()) + ") at line " + std::to_string(line_no));
        }

        const std::string& chrom_tok = fields[kBimChromCol];
        const std::string& id = fields[kBimIdCol];
        const double genpos = parse_genpos(fields[kBimGenposCol], line_no);
        const double physpos = parse_physpos(fields[kBimPhysposCol]);

        const char ref = bim_allele(fields, kBimA1Col);
        const char alt = bim_allele(fields, kBimA2Col);

        table.id.push_back(id);
        table.chrom.push_back(chrom_code(chrom_tok, other_codes, next_other));
        table.genpos_morgans.push_back(genpos);
        table.physpos.push_back(physpos);
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

    struct RawGroup {
        std::string label;
        std::size_t first_seen = 0;
        std::vector<std::size_t> rows;
    };

    std::map<std::string, std::size_t> index_of;
    std::vector<RawGroup> groups;
    std::size_t row = 0;
    std::string line;
    while (std::getline(in, line)) {
        const std::vector<std::string> fields = split_ws(line);
        if (fields.size() < kFamFields) continue;
        if (row >= n_records_present) {
            ++row;
            continue;
        }
        const std::string& pheno = fields[kFamGroupCol];
        const bool ignored = std::any_of(
            kFamIgnorePhenos.begin(), kFamIgnorePhenos.end(),
            [&](const char* sentinel) { return pheno == sentinel; });
        if (!ignored) {
            std::string pop = pheno;
            if (pheno == kFamControlPheno) pop = kControlLabel;
            else if (pheno == kFamCasePheno) pop = kCaseLabel;
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
    part.n_individuals_total = row;

    if (groups.empty()) {
        throw_io_error("no individuals parsed from ");
    }

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

    std::sort(selected.begin(), selected.end(),
              [](const RawGroup* a, const RawGroup* b) { return a->label < b->label; });

    part.groups.reserve(selected.size());
    for (const RawGroup* g : selected) {
        part.groups.push_back(PopGroup{g->label, g->rows});
    }
    return part;
}

}  // namespace steppe::io
