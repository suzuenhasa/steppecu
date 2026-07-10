// src/io/filter/include_exclude.hpp
//
// Resolve user include/exclude SNP id sets (and an external prune.in, if
// supplied) into a per-SNP membership predicate. Host-only io-leaf header: C++20,
// no CUDA, no core/device dependency; file reads surface as std::runtime_error.
//
// Reference: docs/reference/src_io_filter_include_exclude.hpp.md
#ifndef STEPPE_IO_FILTER_INCLUDE_EXCLUDE_HPP
#define STEPPE_IO_FILTER_INCLUDE_EXCLUDE_HPP

#include <string>
#include <unordered_set>
#include <vector>

#include "steppe/config.hpp"

namespace steppe::io::filter {

// Resolved SNP-id membership — reference §3
class SnpMembership {
public:
    explicit SnpMembership(const FilterConfig& cfg);

    [[nodiscard]] bool passes(const std::string& snp_id) const noexcept;

    [[nodiscard]] bool is_noop() const noexcept {
        return keep_set_.empty() && drop_set_.empty();
    }

    [[nodiscard]] std::size_t keep_count() const noexcept { return keep_set_.size(); }
    [[nodiscard]] std::size_t drop_count() const noexcept { return drop_set_.size(); }

private:
    std::unordered_set<std::string> keep_set_;
    std::unordered_set<std::string> drop_set_;
};

// Read a prune.in-style SNP-id list file — reference §4
void read_snp_id_list(const std::string& path, std::vector<std::string>& out);

// Parse a "--ld-prune WIN:STEP:R2" spec into a FilterConfig's LD-prune fields. WIN and STEP are
// variant counts (WIN >= 2, STEP >= 1), R2 the r^2 threshold in (0, 1]. Returns true on success
// (fields written into cfg); on a malformed spec returns false and fills `err`. Shared by the
// RunConfig (pca/fst/kinship) and AdmixtureArgs CLI paths so the two cannot drift.
[[nodiscard]] bool parse_ld_prune_spec(const std::string& spec, FilterConfig& cfg, std::string& err);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_INCLUDE_EXCLUDE_HPP
