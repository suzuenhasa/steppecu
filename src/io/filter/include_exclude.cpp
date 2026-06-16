// src/io/filter/include_exclude.cpp
//
// Include/exclude + external prune.in resolution (architecture.md §1, §5 S-1;
// ROADMAP M2). The prune.in is READ, never computed. Host-pure `io`-leaf TU.
#include "io/filter/include_exclude.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace steppe::io::filter {

void read_snp_id_list(const std::string& path, std::vector<std::string>& out) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(
            "io::filter::read_snp_id_list: cannot open SNP-id list (prune.in): " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string id;
        if (ls >> id) {  // first whitespace token is the id; skip blank lines
            out.push_back(id);
        }
    }
}

SnpMembership::SnpMembership(const FilterConfig& cfg) {
    // Keep-set = include_snp_ids ∪ prune.in ids. The prune.in is READ here (never
    // computed); both sources union into one keep-set so snp_filter sees a single
    // "this SNP is wanted" test, not two.
    for (const std::string& id : cfg.include_snp_ids) {
        keep_set_.insert(id);
    }
    if (!cfg.prune_in_path.empty()) {
        std::vector<std::string> ids;
        read_snp_id_list(cfg.prune_in_path, ids);  // throws on open failure
        for (std::string& id : ids) {
            keep_set_.insert(std::move(id));
        }
    }

    // Drop-set = exclude_snp_ids (overrides the keep-set per the `--exclude`
    // / .missnp convention).
    for (const std::string& id : cfg.exclude_snp_ids) {
        drop_set_.insert(id);
    }
}

bool SnpMembership::passes(const std::string& snp_id) const noexcept {
    // Exclude wins: any id in the drop-set fails regardless of the keep-set.
    if (!drop_set_.empty() && drop_set_.find(snp_id) != drop_set_.end()) {
        return false;
    }
    // Include constraint only when the keep-set is non-empty: then the id must be
    // present. An empty keep-set imposes no include constraint (no-op).
    if (!keep_set_.empty() && keep_set_.find(snp_id) == keep_set_.end()) {
        return false;
    }
    return true;
}

}  // namespace steppe::io::filter
