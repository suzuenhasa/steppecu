// src/io/plink_reader.cpp
//
// PLINK .bim / .fam parsers. read_bim yields the same io::SnpTable as snp_reader
// and read_fam the same io::IndPartition as ind_reader, so everything downstream
// is byte-for-byte unchanged; the only differences are the .bim column order, the
// canonical polarity (ref := A1, since the .bed counts A1 copies), and the .fam
// col-6 phenotype -> egroup mapping.
#include "io/plink_reader.hpp"

#include "io/detail/pop_select.hpp"
#include "io/detail/snp_text_parse.hpp"
#include "io/eigenstrat_format.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace steppe::io {

namespace {

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
        const std::vector<std::string> fields = detail::split_ws(line);

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
        const double genpos = detail::parse_genpos(fields[kBimGenposCol], line_no, "read_bim");
        const double physpos = detail::parse_physpos(fields[kBimPhysposCol]);

        const char ref = bim_allele(fields, kBimA1Col);
        const char alt = bim_allele(fields, kBimA2Col);

        table.id.push_back(id);
        table.chrom.push_back(detail::chrom_code(chrom_tok, other_codes, next_other));
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

    std::map<std::string, std::size_t> index_of;
    std::vector<detail::RawGroup> groups;
    std::size_t row = 0;
    std::string line;
    while (std::getline(in, line)) {
        const std::vector<std::string> fields = detail::split_ws(line);
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
                groups.push_back(detail::RawGroup{pop, groups.size(), {row}});
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

    part.groups = detail::select_populations(groups, sel, throw_io_error);
    return part;
}

}  // namespace steppe::io
