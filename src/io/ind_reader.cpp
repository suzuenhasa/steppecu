// src/io/ind_reader.cpp
//
// Parses an EIGENSTRAT .ind file and selects populations (explicit list,
// auto-top-K, or min-N) into the ascending row groups the GPU decoder consumes.
// Host-pure C++20 — no CUDA, no core/device dependency.
#include "io/ind_reader.hpp"

#include "io/detail/pop_select.hpp"

#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace steppe::io {

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
    std::vector<detail::RawGroup> groups;
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
            groups.push_back(detail::RawGroup{pop, groups.size(), {row}});
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

    part.groups = detail::select_populations(groups, sel, throw_io_error);
    return part;
}

}  // namespace steppe::io
