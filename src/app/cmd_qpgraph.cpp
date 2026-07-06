// src/app/cmd_qpgraph.cpp
//
// The `steppe qpgraph` (single-graph fit) and `steppe qpgraph-search` (topology
// search) commands. Plain C++20 with no CUDA header: the GPU is reached only
// through the CUDA-free seams.
//
// Reference: docs/reference/src_app_cmd_qpgraph.cpp.md
#include "app/cmd_qpgraph.hpp"

#include <cctype>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_io.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/error.hpp"
#include "steppe/qpgraph.hpp"
#include "steppe/qpgraph_search.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Edge-list file parser — reference §2
[[nodiscard]] bool read_edge_list(const std::string& path,
                                  std::vector<steppe::QpGraphEdge>& edges, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open --graph file: " + path; return false; }
    const auto strip = [](std::string s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
        return s;
    };
    std::string line;
    int lineno = 0;
    bool first_data = true;
    while (std::getline(f, line)) {
        ++lineno;
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream ss(line);
        std::string a, b, extra;
        if (!(ss >> a)) continue;
        if (a.size() && a[0] == '#') continue;
        if (!(ss >> b)) { err = "malformed edge at line " + std::to_string(lineno) + " (need 2 columns)"; return false; }
        a = strip(a); b = strip(b);
        if (first_data) {
            std::string al = a, bl = b;
            for (char& c : al) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            for (char& c : bl) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            first_data = false;
            if ((al == "from" && bl == "to") || (al == "parent" && bl == "child")) continue;
        }
        edges.push_back({a, b});
    }
    if (edges.empty()) { err = "--graph file has no edges: " + path; return false; }
    return true;
}

// Single-graph result emitter — reference §6
void emit(std::ostream& os, OutputFormat fmt, const steppe::QpGraphResult& r) {
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    const auto stat = [&](steppe::Status s) -> const char* {
        switch (s) {
            case steppe::Status::Ok: return "ok";
            case steppe::Status::NonSpdCovariance: return "nonspd";
            case steppe::Status::RankDeficient: return "rankdeficient";
            case steppe::Status::InvalidConfig: return "invalid_graph";
            default: return "error";
        }
    };
    if (fmt == OutputFormat::Json) {
        os << "{\n";
        os << "  \"score\": " << r.score << ",\n";
        os << "  \"restart_spread\": " << r.restart_spread << ",\n";
        os << "  \"worst_residual_z\": " << r.worst_residual_z << ",\n";
        os << "  \"worst_pair\": [" << json_quote(r.worst_pop2) << ", " << json_quote(r.worst_pop3)
           << "],\n";
        os << "  \"status\": \"" << stat(r.status) << "\",\n";
        os << "  \"admix\": [\n";
        for (std::size_t j = 0; j < r.weight.size(); ++j) {
            os << "    {\"from\": " << json_quote(r.admix_from[j]) << ", \"to\": "
               << json_quote(r.admix_to[j])
               << ", \"weight\": " << r.weight[j]
               << ", \"low\": " << (j < r.weight_lo.size() ? r.weight_lo[j] : r.weight[j])
               << ", \"high\": " << (j < r.weight_hi.size() ? r.weight_hi[j] : r.weight[j]) << "}"
               << (j + 1 < r.weight.size() ? "," : "") << "\n";
        }
        os << "  ],\n  \"edges\": [\n";
        for (std::size_t e = 0; e < r.edge_length.size(); ++e) {
            os << "    {\"from\": " << json_quote(r.edge_from[e]) << ", \"to\": "
               << json_quote(r.edge_to[e])
               << ", \"length\": " << r.edge_length[e] << "}"
               << (e + 1 < r.edge_length.size() ? "," : "") << "\n";
        }
        os << "  ]\n}\n";
        return;
    }
    os << "# section: edges\n";
    os << "from" << sep << "to" << sep << "type" << sep << "weight\n";
    for (std::size_t j = 0; j < r.weight.size(); ++j)
        os << csv_field(r.admix_from[j], sep) << sep << csv_field(r.admix_to[j], sep) << sep
           << csv_field("admix", sep) << sep << r.weight[j] << "\n";
    for (std::size_t e = 0; e < r.edge_length.size(); ++e)
        os << csv_field(r.edge_from[e], sep) << sep << csv_field(r.edge_to[e], sep) << sep
           << csv_field("edge", sep) << sep << r.edge_length[e] << "\n";
    os << "# section: summary\n";
    os << "score" << sep << "restart_spread" << sep << "worst_z" << sep << "worst_pop2" << sep
       << "worst_pop3" << sep << "status\n";
    os << r.score << sep << r.restart_spread << sep << r.worst_residual_z << sep
       << csv_field(r.worst_pop2, sep) << sep << csv_field(r.worst_pop3, sep) << sep
       << stat(r.status) << "\n";
}

// Shared device-fit dispatch — reference §5
template <typename Result, typename RunFit>
[[nodiscard]] std::optional<int> dispatch_device_fit(const cfg::RunConfig& config,
                                                     const char* prefix, const F2DirResult& dir,
                                                     Result& result, RunFit&& run_fit) {
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, prefix)) return cfg::kExitRuntimeError;
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_fit(dev_f2, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe %s: device error: %s\n", prefix, e.what());
        return exit_code_for_caught(e);
    }
    return std::nullopt;
}

}  // namespace

// Single-graph command flow — reference §3
int run_qpgraph_command(const cfg::RunConfig& config) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpgraph: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.graph_file().empty()) {
        std::fprintf(stderr, "steppe qpgraph: --graph (the edge-list file) is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpgraph: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }
    std::vector<steppe::QpGraphEdge> edges;
    std::string err;
    if (!read_edge_list(config.graph_file(), edges, err)) {
        std::fprintf(stderr, "steppe qpgraph: %s\n", err.c_str());
        return cfg::kExitInvalidConfig;
    }

    steppe::QpGraphOptions opts;
    opts.fudge = config.qpadm_options().fudge;
    opts.diag_f3 = config.qpgraph_diag_f3();
    opts.numstart = config.qpgraph_numstart();
    opts.constrained = config.qpgraph_constrained();

    steppe::QpGraphResult result;
    if (const auto rc = dispatch_device_fit(
            config, "qpgraph", dir, result,
            [&](const device::DeviceF2Blocks& dev_f2, device::Resources& resources) {
                return run_qpgraph(dev_f2, edges, dir.dir.pop_labels, opts, resources);
            })) {
        return *rc;
    }

    if (result.status == steppe::Status::InvalidConfig) {
        std::fprintf(stderr,
                     "steppe qpgraph: the graph could not be fit (a leaf is not an f2 "
                     "population, or the topology is unrooted/cyclic/invalid)\n");
        return cfg::kExitInvalidConfig;
    }

    if (const auto rc = emit_to_destination(
            config, "qpgraph",
            [&](std::ostream& os, OutputFormat fmt) { emit(os, fmt, result); })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

namespace {

// Topology-search result emitter — reference §7
void emit_search(std::ostream& os, OutputFormat fmt, const steppe::QpGraphSearchResult& r) {
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    const auto edges_str = [&](const std::vector<steppe::QpGraphEdge>& e) {
        std::string s;
        for (std::size_t i = 0; i < e.size(); ++i) {
            s += e[i].from; s += '>'; s += e[i].to;
            if (i + 1 < e.size()) s += ';';
        }
        return s;
    };
    if (fmt == OutputFormat::Json) {
        os << "{\n";
        os << "  \"n_trees\": " << r.n_trees << ",\n";
        os << "  \"n_admix1\": " << r.n_admix1 << ",\n";
        os << "  \"n_candidates\": " << r.n_candidates << ",\n";
        os << "  \"best_score\": " << r.best.score << ",\n";
        os << "  \"second_best_score\": " << r.second_best_score << ",\n";
        os << "  \"best_nadmix\": " << r.best.nadmix << ",\n";
        os << "  \"best_hash\": " << json_quote(std::to_string(r.best.hash)) << ",\n";
        os << "  \"best_edges\": " << json_quote(edges_str(r.best.edges)) << ",\n";
        os << "  \"heuristic_recovered\": " << (r.heuristic_recovered ? "true" : "false") << ",\n";
        os << "  \"fit_all_wall_ms\": " << r.fit_all_wall_ms << ",\n";
        os << "  \"topologies_per_s\": " << r.topologies_per_s << "\n}\n";
        return;
    }
    os << "# section: search\n";
    os << "n_trees" << sep << "n_admix1" << sep << "n_candidates" << sep << "best_score" << sep
       << "second_best_score" << sep << "best_nadmix" << sep << "heuristic_recovered" << sep
       << "fit_all_wall_ms" << sep << "topologies_per_s\n";
    os << r.n_trees << sep << r.n_admix1 << sep << r.n_candidates << sep << r.best.score << sep
       << r.second_best_score << sep << r.best.nadmix << sep << (r.heuristic_recovered ? 1 : 0)
       << sep << r.fit_all_wall_ms << sep << r.topologies_per_s << "\n";
    os << "# section: best_edges\n" << csv_field(edges_str(r.best.edges), sep) << "\n";
}

}  // namespace

// Topology-search command flow — reference §3
int run_qpgraph_search_command(const cfg::RunConfig& config) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpgraph-search: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.pops().size() < 3) {
        std::fprintf(stderr,
                     "steppe qpgraph-search: --pops needs >= 3 population labels (the bounded "
                     "leaf set the search enumerates topologies over)\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpgraph-search: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    steppe::QpGraphSearchOptions opts;
    opts.pops = config.pops();
    opts.max_nadmix = config.qpgraph_max_nadmix();
    opts.fit.fudge = config.qpadm_options().fudge;
    opts.fit.diag_f3 = config.qpgraph_diag_f3();
    opts.fit.numstart = config.qpgraph_numstart();
    opts.fit.constrained = config.qpgraph_constrained();

    steppe::QpGraphSearchResult result;
    if (const auto rc = dispatch_device_fit(
            config, "qpgraph-search", dir, result,
            [&](const device::DeviceF2Blocks& dev_f2, device::Resources& resources) {
                return run_qpgraph_search(dev_f2, dir.dir.pop_labels, opts, resources);
            })) {
        return *rc;
    }
    if (result.status == steppe::Status::InvalidConfig) {
        std::fprintf(stderr,
                     "steppe qpgraph-search: invalid pop-set (a pop is not an f2 population, or "
                     "< 3 leaves)\n");
        return cfg::kExitInvalidConfig;
    }

    if (const auto rc = emit_to_destination(
            config, "qpgraph-search",
            [&](std::ostream& os, OutputFormat fmt) { emit_search(os, fmt, result); })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
