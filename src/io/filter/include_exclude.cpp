// src/io/filter/include_exclude.cpp
//
// Resolves the user's include/exclude SNP-id lists and an optional external
// prune.in file into one per-SNP keep/drop test. Host-pure; the prune.in is
// read, never computed.
//
// Reference: docs/reference/src_io_filter_include_exclude.cpp.md
#include "io/filter/include_exclude.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace steppe::io::filter {

// Read a prune.in SNP-id list — reference §2
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
        if (ls >> id) {
            out.push_back(id);
        }
    }
    if (in.bad() || (in.fail() && !in.eof())) {
        throw std::runtime_error(
            "io::filter::read_snp_id_list: read failed on SNP-id list "
            "(not a regular file?): " + path);
    }
}

// Resolve the keep-set and drop-set — reference §3
SnpMembership::SnpMembership(const FilterConfig& cfg) {
    for (const std::string& id : cfg.include_snp_ids) {
        keep_set_.insert(id);
    }
    if (!cfg.prune_in_path.empty()) {
        std::vector<std::string> ids;
        read_snp_id_list(cfg.prune_in_path, ids);
        for (std::string& id : ids) {
            keep_set_.insert(std::move(id));
        }
    }

    for (const std::string& id : cfg.exclude_snp_ids) {
        drop_set_.insert(id);
    }
}

// Membership test — reference §4
bool SnpMembership::passes(const std::string& snp_id) const noexcept {
    if (!drop_set_.empty() && drop_set_.find(snp_id) != drop_set_.end()) {
        return false;
    }
    if (!keep_set_.empty() && keep_set_.find(snp_id) == keep_set_.end()) {
        return false;
    }
    return true;
}

}  // namespace steppe::io::filter
