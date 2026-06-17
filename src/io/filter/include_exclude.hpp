// src/io/filter/include_exclude.hpp
//
// Resolve user include/exclude SNP id sets (and an external prune.in, if supplied)
// into a per-SNP membership predicate (architecture.md §1 "we accept only an
// externally supplied prune.in", "merge is a plan"; §5 S-1; ROADMAP M2). The
// prune.in is READ, NEVER computed — steppe does not compute LD itself
// (architecture.md §1).
//
// LAYERING: an `io`-leaf header (architecture.md §4) — host C++20, no CUDA, no
// core/device dependency. File reads surface as std::runtime_error to the caller.
//
// THE MEMBERSHIP RULE (composed in one place so snp_filter does not re-derive it):
//   keep-set  = include_snp_ids ∪ prune.in ids   (if EITHER is non-empty, the SNP
//               must be in the union to pass; if BOTH empty, no include constraint)
//   drop-set  = exclude_snp_ids                  (any id here fails, overriding the
//               keep-set — exclude wins, the `--exclude` / .missnp convention)
// A SNP passes membership iff: (keep-set empty OR id ∈ keep-set) AND id ∉ drop-set.
#ifndef STEPPE_IO_FILTER_INCLUDE_EXCLUDE_HPP
#define STEPPE_IO_FILTER_INCLUDE_EXCLUDE_HPP

#include <string>
#include <unordered_set>
#include <vector>

#include "steppe/config.hpp"  // steppe::FilterConfig (include/exclude + prune.in path)

namespace steppe::io::filter {

/// Resolved SNP-id membership: a keep-set (the include ∪ prune.in union; EMPTY ⇒
/// no include constraint) and a drop-set (exclude; EMPTY ⇒ no exclude). Built once
/// from a FilterConfig, then queried per SNP id by `passes`. Hash sets so the
/// per-SNP query is O(1) — the .snp can be ~584k ids.
class SnpMembership {
public:
    /// Build the membership from the include/exclude id lists and (if non-empty)
    /// the prune.in path in `cfg`. The prune.in file is READ (one SNP id per
    /// whitespace-delimited line; blank lines skipped), its ids unioned into the
    /// keep-set. Throws std::runtime_error if `cfg.prune_in_path` is set but the
    /// file cannot be opened OR cannot be read (e.g. it is a directory, which opens
    /// but read-fails — fail-fast rather than a silently-empty keep-set; see
    /// read_snp_id_list). When include_snp_ids, exclude_snp_ids, and
    /// prune_in_path are all empty/unset, the membership is a NO-OP (`passes`
    /// returns true for every id) — the parity-path default.
    explicit SnpMembership(const FilterConfig& cfg);

    /// Whether `snp_id` passes membership: (keep-set empty OR id ∈ keep-set) AND
    /// id ∉ drop-set. No-op (always true) when both sets are empty.
    [[nodiscard]] bool passes(const std::string& snp_id) const noexcept;

    /// True iff this membership imposes NO constraint (both sets empty) — i.e. it
    /// is the no-op default. Lets snp_filter skip the per-SNP query entirely on the
    /// parity path.
    [[nodiscard]] bool is_noop() const noexcept {
        return keep_set_.empty() && drop_set_.empty();
    }

    /// Sizes, for diagnostics / tests (the resolved keep-set and drop-set counts).
    [[nodiscard]] std::size_t keep_count() const noexcept { return keep_set_.size(); }
    [[nodiscard]] std::size_t drop_count() const noexcept { return drop_set_.size(); }

private:
    std::unordered_set<std::string> keep_set_;  ///< include ∪ prune.in; empty ⇒ no constraint
    std::unordered_set<std::string> drop_set_;  ///< exclude; overrides keep-set
};

/// Read a prune.in-style SNP-id list file: one id per whitespace-delimited token,
/// blank lines skipped, into `out` (appended). Exposed for tests / reuse. Throws
/// std::runtime_error if the file cannot be opened, OR if it opens but read-fails
/// (a directory / FIFO / other non-regular node opens on POSIX but sets badbit on
/// the first read — fail-fast instead of silently returning an empty list). (The
/// first whitespace token of each line is taken as the id, tolerating trailing
/// columns.)
void read_snp_id_list(const std::string& path, std::vector<std::string>& out);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_INCLUDE_EXCLUDE_HPP
