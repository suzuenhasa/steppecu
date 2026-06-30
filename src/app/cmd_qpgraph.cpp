// src/app/cmd_qpgraph.cpp — the `steppe qpgraph` command (single-graph fit).
//
// Mirrors cmd_qpadm.cpp: read the f2_blocks dir -> build_resources(DeviceConfig) ->
// upload_f2_blocks_to_device -> run_qpgraph(DeviceF2Blocks, edges, leaf_names, opts) ->
// emit the result. The graph is read from --graph (an admixtools-format 2-column edge
// list). The leaf_names map is the f2 dir's pops.txt order (the P-axis). PLAIN C++20, no
// CUDA header (the §4 layering): the GPU is reached only via the CUDA-free seams.
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

#include "app/cmd_emit.hpp"             // emit_to_destination (shared open->write->flush->verify)
#include "app/exit_code_for_caught.hpp" // exit_code_for_caught (5 -> 3 on a real device OOM, B2)
#include "app/f2_dir_io.hpp"
#include "app/result_emit.hpp"          // OutputFormat / parse_output_format
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"  // CUDA-FREE
#include "device/resources.hpp"         // CUDA-FREE
#include "steppe/error.hpp"
#include "steppe/qpgraph.hpp"           // run_qpgraph + QpGraphEdge/Result/Options
#include "steppe/qpgraph_search.hpp"    // run_qpgraph_search (the topology SEARCH v1)

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

/// Parse an admixtools-format edge-list file: each non-blank, non-comment line has TWO
/// whitespace- OR comma-separated tokens (parent child). A leading header row "from,to"
/// (case-insensitive) and #-comments are skipped. Surrounding quotes are stripped (the R
/// write.csv format quotes the labels). Returns false (with `err`) on a malformed line.
[[nodiscard]] bool read_edge_list(const std::string& path,
                                  std::vector<steppe::QpGraphEdge>& edges, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open --graph file: " + path; return false; }
    const auto strip = [](std::string s) {
        // trim whitespace
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
        // strip surrounding double quotes (R write.csv)
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
        return s;
    };
    std::string line;
    int lineno = 0;
    bool first_data = true;
    while (std::getline(f, line)) {
        ++lineno;
        // admixtools' R write.csv emits comma-separated edge lists; normalize commas to
        // whitespace so the istringstream split below handles CSV and space/tab edges alike.
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream ss(line);
        std::string a, b, extra;
        if (!(ss >> a)) continue;       // blank line
        if (a.size() && a[0] == '#') continue;  // comment
        if (!(ss >> b)) { err = "malformed edge at line " + std::to_string(lineno) + " (need 2 columns)"; return false; }
        a = strip(a); b = strip(b);
        // skip a header row.
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

/// Emit the qpGraph fit result (CSV/TSV or JSON). CSV/TSV: an `edges` section
/// (from,to,type,weight) for every edge (admix rows carry the mixture weight; drift rows
/// the fitted length) + a `summary` section (score,restart_spread,worst_z,status). JSON: a
/// single object. Self-contained (no shared format primitive needed).
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
    // CSV/TSV.
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

/// Shared device-fit dispatch for both qpgraph commands (§2.11 cross-ref: factored from the
/// run_qpgraph_command / run_qpgraph_search_command bodies, which were byte-identical save the
/// command-name prefix and the run call). Builds the device resources, guards the no-GPU case,
/// uploads the f2 blocks RESIDENT (device 0), then invokes `run_fit(dev_f2, resources)` inside
/// the one try/catch. `prefix` is the "steppe <prefix>:" diagnostic tag. On success returns
/// std::nullopt and writes the fit into `result`; on a no-GPU / device fault returns the
/// exit code the caller must propagate.
template <typename Result, typename RunFit>
[[nodiscard]] std::optional<int> dispatch_device_fit(const cfg::RunConfig& config,
                                                     const char* prefix, const F2DirResult& dir,
                                                     Result& result, RunFit&& run_fit) {
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe %s: no CUDA device available (steppe is a GPU "
                         "product; a CUDA-capable GPU is required)\n",
                         prefix);
            return cfg::kExitRuntimeError;
        }
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

int run_qpgraph_command(const cfg::RunConfig& config) {
    // ---- 1. f2_blocks dir + the graph file --------------------------------------
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

    // ---- 2. options (defaults == the AT2 golden's) ------------------------------
    steppe::QpGraphOptions opts;
    opts.fudge = config.qpadm_options().fudge;      // --fudge (shared QpAdmOptions::fudge == AT2 diag)
    opts.diag_f3 = config.qpgraph_diag_f3();
    opts.numstart = config.qpgraph_numstart();
    opts.constrained = config.qpgraph_constrained();

    // ---- 3. build_resources -> upload f2 RESIDENT -> run_qpgraph (GPU path) ------
    steppe::QpGraphResult result;
    if (const auto rc = dispatch_device_fit(
            config, "qpgraph", dir, result,
            [&](const device::DeviceF2Blocks& dev_f2, device::Resources& resources) {
                return run_qpgraph(dev_f2, edges, dir.dir.pop_labels, opts, resources);
            })) {
        return *rc;
    }

    if (result.status == steppe::Status::InvalidConfig) {
        // A graph parse / structural domain outcome (e.g. a leaf not in the f2 set, an
        // unrooted / cyclic graph). Surface as InvalidConfig (a clear fault, not a silent row).
        std::fprintf(stderr,
                     "steppe qpgraph: the graph could not be fit (a leaf is not an f2 "
                     "population, or the topology is unrooted/cyclic/invalid)\n");
        return cfg::kExitInvalidConfig;
    }

    // ---- 4. Emit ---------------------------------------------------------------
    if (const auto rc = emit_to_destination(
            config, "qpgraph",
            [&](std::ostream& os, OutputFormat fmt) { emit(os, fmt, result); })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

namespace {

/// Emit the topology-search result (CSV/TSV or JSON): the exhaustive-coverage count, the
/// global-best (nadmix / score / edges), the second-best gap, the heuristic recovery, the
/// wall-clock + topologies/s, and the best graph's fitted edges/weights.
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
