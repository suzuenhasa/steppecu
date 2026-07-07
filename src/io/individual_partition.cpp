// src/io/individual_partition.cpp
//
// Parses an EIGENSTRAT .ind (or PLINK .fam) into a per-individual SINGLETON partition
// labelled by Genetic ID — the modeling step READv2 needs so each sample is its own
// sweep index rather than being collapsed into a pop. Host-pure C++20, no CUDA.
//
// Reference: docs/reference/src_io_individual_partition.cpp.md
#include "io/individual_partition.hpp"

#include <cstddef>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace steppe::io {

IndPartition read_individual_partition(
    GenoFormat format, const std::string& path,
    const std::optional<std::vector<std::string>>& samples, std::size_t n_records_present) {
    const auto fail = [&](const std::string& msg) {
        throw std::runtime_error("io::read_individual_partition: " + msg);
    };

    std::ifstream in(path);
    if (!in) fail("cannot open individual file: " + path);

    // Optional restriction set + a tracker of which requested IDs we have seen.
    const bool restrict = samples.has_value();
    std::unordered_set<std::string> want;
    if (restrict) {
        for (const std::string& s : *samples) want.insert(s);
    }

    IndPartition part;
    std::unordered_map<std::string, std::size_t> seen;  // Genetic ID -> retained slot
    std::set<std::string> matched;                       // requested IDs actually found

    std::size_t row = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (row >= n_records_present) break;
        std::istringstream ls(line);
        std::vector<std::string> tok;
        std::string t;
        while (ls >> t) tok.push_back(t);
        if (tok.empty()) continue;

        // Genetic ID: PLINK .fam uses the within-family IID (column 2); the EIGENSTRAT
        // .ind family uses column 1.
        const std::string& gid =
            (format == GenoFormat::Plink && tok.size() >= 2) ? tok[1] : tok[0];
        const std::size_t this_row = row++;

        if (restrict && want.find(gid) == want.end()) continue;
        if (restrict) matched.insert(gid);

        if (seen.find(gid) != seen.end()) {
            fail("duplicate Genetic ID '" + gid +
                 "' among selected samples — the name->index resolver would be ambiguous; "
                 "restrict --samples to unique IDs (or de-duplicate the .ind)");
        }
        seen.emplace(gid, part.groups.size());
        part.groups.push_back(PopGroup{gid, {this_row}});
    }
    part.n_individuals_total = row;

    if (restrict && matched.size() != want.size()) {
        for (const std::string& req : *samples) {
            if (matched.find(req) == matched.end()) {
                fail("requested --samples ID '" + req + "' not found in " + path);
            }
        }
    }
    if (part.groups.empty()) fail("no individuals selected from " + path);
    return part;
}

}  // namespace steppe::io
