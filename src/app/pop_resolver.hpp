// src/app/pop_resolver.hpp
//
// Resolves population names to their P-axis indices: names are an app-level
// concern read from <dir>/pops.txt, while the compute seam is index-only. Plain
// C++20 with no CUDA, so the CLI and app layer can include it without the GPU stack.
//
// Reference: docs/reference/src_app_pop_resolver.hpp.md
#ifndef STEPPE_APP_POP_RESOLVER_HPP
#define STEPPE_APP_POP_RESOLVER_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include "steppe/error.hpp"

namespace steppe::app {

// One-name resolution result — reference §3
struct ResolveResult {
    bool ok = false;
    Status status = Status::Ok;
    std::string error;
    int index = -1;
};

// List resolution result — reference §3
struct ResolveListResult {
    bool ok = false;
    Status status = Status::Ok;
    std::string error;
    std::vector<int> indices;
};

// The label->index resolver — reference §4
class PopResolver {
public:
    explicit PopResolver(const std::vector<std::string>& labels_in_index_order);

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] const std::string& error() const noexcept { return error_; }

    [[nodiscard]] int size() const noexcept { return static_cast<int>(labels_.size()); }
    [[nodiscard]] const std::string& label_at(int index) const { return labels_.at(static_cast<std::size_t>(index)); }

    [[nodiscard]] ResolveResult resolve(const std::string& label) const;

    [[nodiscard]] ResolveListResult resolve_all(const std::vector<std::string>& labels) const;

private:
    std::vector<std::string> labels_;
    std::unordered_map<std::string, int> by_name_;
    bool valid_ = true;
    std::string error_;
};

}  // namespace steppe::app

#endif  // STEPPE_APP_POP_RESOLVER_HPP
