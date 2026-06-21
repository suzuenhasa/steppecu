// src/io/ind_reader.cpp
//
// .ind parse + population selection (architecture.md §5 S−2/S0; ROADMAP M1).
// Reproduces the on-box oracle build_tgeno_matrix.py selection semantics exactly
// so the GPU decoder yields the SAME 50-pop set as the derived_acc validation
// matrices.
//
// LAYERING: `io`-leaf TU (architecture.md §4) — pure host C++20, no CUDA, no
// core/device dependency.
#include "io/ind_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace steppe::io {

namespace {

/// A population label, its first-appearance order, and its member rows. The
/// first-appearance index is the tie-break for auto-top K (Python Counter
/// .most_common breaks equal counts by insertion order == first .ind appearance).
struct RawGroup {
    std::string label;
    std::size_t first_seen = 0;          // tie-break key for auto-top
    std::vector<std::size_t> rows;       // ascending individual-record indices
};

}  // namespace

IndPartition read_ind(const std::string& path,
                      const PopSelection& sel,
                      std::size_t n_records_present) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_ind: cannot open .ind file: " + path);
    }

    // Walk the .ind in row order, grouping individual-record indices by the
    // column-3 population label. Insertion order into `order` is first-appearance
    // order (the auto-top tie-break). Cap the individual axis at
    // n_records_present (the oracle's pops_all[:n_records]).
    std::map<std::string, std::size_t> index_of;  // label → slot in `groups`
    std::vector<RawGroup> groups;
    std::size_t row = 0;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream line_stream(line);
        std::string id, sex, pop;
        if (!(line_stream >> id >> sex >> pop)) continue;  // skip blank/short lines
        if (row >= n_records_present) {
            ++row;
            continue;  // beyond the present genotype records (partial-file cap)
        }
        auto it = index_of.find(pop);
        if (it == index_of.end()) {
            index_of.emplace(pop, groups.size());
            groups.push_back(RawGroup{pop, groups.size(), {row}});
        } else {
            groups[it->second].rows.push_back(row);
        }
        ++row;
    }

    IndPartition part;
    part.n_individuals_total = row;  // total .ind rows seen (the individual axis)

    if (groups.empty()) {
        throw std::runtime_error("io::read_ind: no individuals parsed from " + path);
    }

    // ---- Selection -----------------------------------------------------------
    std::vector<const RawGroup*> selected;

    switch (sel.mode) {
        case PopSelection::Mode::Explicit: {
            // Keep exactly the requested labels that are present.
            std::unordered_set<std::string> want(sel.labels.begin(), sel.labels.end());
            for (const auto& g : groups) {
                if (want.count(g.label)) selected.push_back(&g);
            }
            break;
        }
        case PopSelection::Mode::AutoTopK: {
            // The k largest by count; ties broken by first-appearance order (the
            // oracle's Counter.most_common). Rank by (count desc, first_seen asc),
            // take the first k.
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
            for (const auto& g : groups) {
                if (g.rows.size() >= sel.min_n) selected.push_back(&g);
            }
            break;
        }
    }

    if (selected.empty()) {
        throw std::runtime_error("io::read_ind: population selection is empty for " + path);
    }

    // Final Q/V/N row order: sort the selected set ASCENDING by label (the
    // oracle's `sel = sorted(sel)`). std::string < is byte/lexicographic order,
    // matching Python's sorted() on str.
    std::sort(selected.begin(), selected.end(),
              [](const RawGroup* a, const RawGroup* b) { return a->label < b->label; });

    part.groups.reserve(selected.size());
    for (const RawGroup* g : selected) {
        part.groups.push_back(PopGroup{g->label, g->rows});  // rows already ascending
    }
    return part;
}

}  // namespace steppe::io
