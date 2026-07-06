// src/io/snp_reader.cpp
//
// Parses an EIGENSTRAT .snp file into a SnpTable of parallel per-SNP arrays in
// file order. The row index is the SNP index, so the reader never silently
// skips a record: anything unparseable is a hard error or a well-defined
// default. Pure host C++20, no CUDA or core/device dependency.
//
// Reference: docs/reference/src_io_snp_reader.cpp.md
#include "io/snp_reader.hpp"

#include "io/detail/snp_text_parse.hpp"
#include "io/eigenstrat_format.hpp"

#include <cstddef>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace steppe::io {

// Reading a record — reference §6
SnpTable read_snp(const std::string& path, std::size_t max_snps) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_snp: cannot open .snp file: " + path);
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
                "io::read_snp: blank line at line " + std::to_string(line_no) +
                " (interior blank lines desync the SNP axis from the .geno)");
        }

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
        const double genpos = detail::parse_genpos(fields[2], line_no, "read_snp");
        const double physpos =
            fields.size() > kPhysposCol ? detail::parse_physpos(fields[kPhysposCol]) : 0.0;
        const bool has_alleles = fields.size() >= kFullSnpFields;
        const auto allele = [&](std::size_t col) {
            return has_alleles && !fields[col].empty() ? fields[col][0] : kMissingAllele;
        };
        const char ref = allele(kRefAlleleCol);
        const char alt = allele(kAltAlleleCol);

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

}  // namespace steppe::io
