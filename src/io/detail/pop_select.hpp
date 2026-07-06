// src/io/detail/pop_select.hpp
//
// Shared population-selection tail for the EIGENSTRAT .ind and PLINK .fam readers:
// the Explicit / AutoTopK / MinN filter over the parsed raw groups (byte-identical
// logic previously copy-pasted into both TUs). Pure host C++20.
#ifndef STEPPE_IO_DETAIL_POP_SELECT_HPP
#define STEPPE_IO_DETAIL_POP_SELECT_HPP

#include "io/ind_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace steppe::io::detail {

// One parsed-but-unselected population, in first-seen order.
struct RawGroup {
    std::string label;
    std::size_t first_seen = 0;
    std::vector<std::size_t> rows;
};

// Apply sel (Explicit / AutoTopK / MinN) to the raw groups and return the chosen
// PopGroups sorted by label. throw_io_error(msg) reports the reader-specific
// empty-selection error (its "io::read_ind:"/"io::read_fam:" prefix + path).
template <class ThrowFn>
[[nodiscard]] inline std::vector<PopGroup> select_populations(
    const std::vector<RawGroup>& groups,
    const PopSelection& sel,
    const ThrowFn& throw_io_error) {
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

    std::vector<PopGroup> result;
    result.reserve(selected.size());
    for (const RawGroup* g : selected) {
        result.push_back(PopGroup{g->label, g->rows});
    }
    return result;
}

}  // namespace steppe::io::detail

#endif  // STEPPE_IO_DETAIL_POP_SELECT_HPP
