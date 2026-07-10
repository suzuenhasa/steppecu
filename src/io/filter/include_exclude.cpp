// src/io/filter/include_exclude.cpp
//
// Resolves the user's include/exclude SNP-id lists and an optional external
// prune.in file into one per-SNP keep/drop test. Host-pure; the prune.in is
// read, never computed.
//
// Reference: docs/reference/src_io_filter_include_exclude.cpp.md
#include "io/filter/include_exclude.hpp"

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace steppe::io::filter {

namespace {
// Trim ASCII whitespace from both ends (host-pure; no locale).
[[nodiscard]] std::string trim_ws(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
    return s.substr(b, e - b);
}
}  // namespace

// Parse "WIN:STEP:R2" — reference §5
bool parse_ld_prune_spec(const std::string& spec_in, FilterConfig& cfg, std::string& err) {
    const std::string spec = trim_ws(spec_in);
    const std::size_t c1 = spec.find(':');
    const std::size_t c2 = (c1 == std::string::npos) ? std::string::npos : spec.find(':', c1 + 1);
    if (c1 == std::string::npos || c2 == std::string::npos) {
        err = "--ld-prune must be WIN:STEP:R2 (e.g. 200:50:0.5)";
        return false;
    }
    const std::string win_s = trim_ws(spec.substr(0, c1));
    const std::string step_s = trim_ws(spec.substr(c1 + 1, c2 - c1 - 1));
    const std::string r2_s = trim_ws(spec.substr(c2 + 1));
    int win = 0, step = 0;
    double r2 = 0.0;
    try {
        std::size_t pw = 0, ps = 0, pr = 0;
        win = std::stoi(win_s, &pw);
        step = std::stoi(step_s, &ps);
        r2 = std::stod(r2_s, &pr);
        if (pw != win_s.size() || ps != step_s.size() || pr != r2_s.size()) {
            err = "--ld-prune WIN:STEP:R2 has non-numeric fields (got '" + spec + "')";
            return false;
        }
    } catch (const std::exception&) {
        err = "--ld-prune WIN:STEP:R2 has non-numeric fields (got '" + spec + "')";
        return false;
    }
    if (win < 2) { err = "--ld-prune window (WIN) must be >= 2 variants"; return false; }
    if (step < 1) { err = "--ld-prune step (STEP) must be >= 1 variant"; return false; }
    if (r2 <= 0.0 || r2 > 1.0) { err = "--ld-prune r^2 must lie in (0, 1]"; return false; }
    cfg.ld_prune_window = win;
    cfg.ld_prune_step = step;
    cfg.ld_prune_r2 = r2;
    return true;
}

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
