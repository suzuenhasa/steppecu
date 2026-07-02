// src/app/pop_resolver.cpp
//
// Resolves population names to their P-axis index. Host-only C++, no CUDA.
#include "app/pop_resolver.hpp"

#include <utility>

namespace steppe::app {

PopResolver::PopResolver(const std::vector<std::string>& labels_in_index_order)
    : labels_(labels_in_index_order) {
    by_name_.reserve(labels_.size());
    for (int i = 0; i < static_cast<int>(labels_.size()); ++i) {
        auto [it, inserted] = by_name_.emplace(labels_[static_cast<std::size_t>(i)], i);
        if (!inserted) {
            valid_ = false;
            error_ = "pops.txt has a duplicate population label '" +
                     labels_[static_cast<std::size_t>(i)] +
                     "' (lines " + std::to_string(it->second) + " and " +
                     std::to_string(i) + ", 0-based) — the name<->index map is ambiguous";
            return;
        }
    }
}

ResolveResult PopResolver::resolve(const std::string& label) const {
    ResolveResult r;
    const auto it = by_name_.find(label);
    if (it == by_name_.end()) {
        r.ok = false;
        r.status = Status::InvalidConfig;
        r.error = "unknown population '" + label +
                  "' (not in pops.txt; check spelling against the f2-dir's pops.txt)";
        return r;
    }
    r.ok = true;
    r.status = Status::Ok;
    r.index = it->second;
    return r;
}

ResolveListResult PopResolver::resolve_all(const std::vector<std::string>& labels) const {
    ResolveListResult r;
    r.indices.reserve(labels.size());
    for (const std::string& name : labels) {
        const ResolveResult one = resolve(name);
        if (!one.ok) {
            r.ok = false;
            r.status = one.status;
            r.error = one.error;
            r.indices.clear();
            return r;
        }
        r.indices.push_back(one.index);
    }
    r.ok = true;
    r.status = Status::Ok;
    return r;
}

}  // namespace steppe::app
