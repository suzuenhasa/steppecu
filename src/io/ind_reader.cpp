// src/io/ind_reader.cpp
//
// Parses an EIGENSTRAT .ind file and selects populations (explicit list,
// auto-top-K, or min-N) into the ascending row groups the GPU decoder consumes.
// Host-pure C++20 — no CUDA, no core/device dependency.
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

struct RawGroup {
    std::string label;
    std::size_t first_seen = 0;
    std::vector<std::size_t> rows;
};

}  // namespace

IndPartition read_ind(const std::string& path,
                      const PopSelection& sel,
                      std::size_t n_records_present) {
    const auto throw_io_error = [&](const std::string& msg) {
        throw std::runtime_error("io::read_ind: " + msg + path);
    };

    std::ifstream in(path);
    if (!in) {
        throw_io_error("cannot open .ind file: ");
    }

    std::map<std::string, std::size_t> index_of;
    std::vector<RawGroup> groups;
    std::size_t row = 0;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream line_stream(line);
        std::string id, sex, pop;
        if (!(line_stream >> id >> sex >> pop)) continue;
        if (row >= n_records_present) {
            continue;
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
    part.n_individuals_total = row;

    if (groups.empty()) {
        throw_io_error("no individuals parsed from ");
    }

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

    part.groups.reserve(selected.size());
    for (const RawGroup* g : selected) {
        part.groups.push_back(PopGroup{g->label, g->rows});
    }
    return part;
}

}  // namespace steppe::io
