// src/app/pop_resolver.hpp
//
// The pop-NAME -> P-axis INDEX resolver (cli-bindings.md §1, §4.2). The compute seam
// is INDEX-ONLY (QpAdmModel references pops by index into the f2 P axis); names are an
// app concern resolved against <dir>/pops.txt. This is the single home for that map +
// the fail-fast on an unknown label (Status::InvalidConfig with the offending name,
// cli-bindings.md §4.2: "An unknown pop name is Status::InvalidConfig fail-fast with
// the offending label"). PLAIN C++20, app-only, NO CUDA header.
//
// It replaces the hardcoded name->index map the reference fit test carries
// (tests/reference/test_qpadm_parity.cu) — the gap cli-bindings.md §1 flags as the
// single biggest design decision the access layer exists to close.
#ifndef STEPPE_APP_POP_RESOLVER_HPP
#define STEPPE_APP_POP_RESOLVER_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include "steppe/error.hpp"  // steppe::Status

namespace steppe::app {

/// Result of a name->index resolution: ok + the resolved index on success, or a fault
/// Status + the reason (naming the offending label) on failure.
struct ResolveResult {
    bool ok = false;
    Status status = Status::Ok;
    std::string error;  // empty on success; "unknown population 'X' ..." on failure.
    int index = -1;     // valid only when ok.
};

/// Resolve a list of labels (target/left/right) to indices in one shot.
struct ResolveListResult {
    bool ok = false;
    Status status = Status::Ok;
    std::string error;
    std::vector<int> indices;  // valid only when ok; same order as the input labels.
};

/// Maps pop labels to their P-axis index. Built once from pops.txt (index order);
/// resolve() is then O(1) per name. A duplicate label in pops.txt is a fault at
/// construction (the map would be ambiguous) — surfaced via valid()/error().
class PopResolver {
public:
    /// Build the label->index map from the P labels in index order (the pops.txt
    /// content). A duplicate label makes the resolver INVALID (valid()==false) with a
    /// reason — the app fail-fasts before any resolve.
    explicit PopResolver(const std::vector<std::string>& labels_in_index_order);

    /// True iff the map was built without a duplicate-label conflict.
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    /// The construction error reason (empty iff valid()).
    [[nodiscard]] const std::string& error() const noexcept { return error_; }

    /// The number of populations (== P, the pops.txt line count).
    [[nodiscard]] int size() const noexcept { return static_cast<int>(labels_.size()); }
    /// The label at a P-axis index (for emitting names back onto the result rows).
    [[nodiscard]] const std::string& label_at(int index) const { return labels_.at(static_cast<std::size_t>(index)); }

    /// Resolve ONE label to its P-axis index. Unknown label ⇒ ok=false,
    /// Status::InvalidConfig, the reason naming the label + the dir hint.
    [[nodiscard]] ResolveResult resolve(const std::string& label) const;

    /// Resolve a LIST of labels in order; the first unknown short-circuits to a fault
    /// naming it (so the user sees the exact offending name, cli-bindings.md §4.2).
    [[nodiscard]] ResolveListResult resolve_all(const std::vector<std::string>& labels) const;

private:
    std::vector<std::string> labels_;                  // index -> label (pops.txt order)
    std::unordered_map<std::string, int> by_name_;     // label -> index
    bool valid_ = true;
    std::string error_;
};

}  // namespace steppe::app

#endif  // STEPPE_APP_POP_RESOLVER_HPP
