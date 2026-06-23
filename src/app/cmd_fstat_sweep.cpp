// src/app/cmd_fstat_sweep.cpp
//
// The `steppe f4-sweep` / `steppe f3-sweep` commands (GPU-only all-combinations f-stat sweep).
// Read the f2_blocks dir -> resolve an OPTIONAL --pops subset to P-axis indices (empty ⇒ the
// whole dir) -> build_resources -> upload f2 RESIDENT -> run_f4_sweep / run_f3_sweep (the
// on-device enumerate+compute+filter+compact pipeline) -> emit ONLY the survivors. The full
// C(P,k) table is NEVER materialized (a multi-TB dump): the device filter + CUB compaction keep
// only the survivors, and only they cross the seam.
//
// FILTER: --min-z Z (keep |z|>=Z; default 3.0) OR --top-k K (keep the K largest |z|). CAP:
// --sure lifts the maxcomb cap (a sweep over more than kFstatMaxComb items refuses without it).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering gate). main() owns stdout/stderr. A
// capped/empty sweep is a clean exit (record-and-continue); only device/IO faults return nonzero.
#include "app/cmd_fstat_sweep.hpp"

#include <array>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"  // OutputFormat, parse_output_format
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/error.hpp"             // steppe::Status, status_string
#include "steppe/fstat_sweep.hpp"       // run_f4_sweep / run_f3_sweep, SweepRequest/Result

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

/// Emit the survivor table (pop1..popK, est, se, z, p). CSV/TSV share the delimiter; JSON is an
/// array of row objects. Survivors only (the full N is never here). `k` = 4 (f4) / 3 (f3).
void emit_sweep(std::ostream& os, OutputFormat fmt, const SweepResult& r,
                const PopResolver& resolver, int k) {
    const char* status = (r.status == Status::Ok) ? "ok" : "error";
    if (fmt == OutputFormat::Json) {
        os << "{\"status\":\"" << status << "\",\"enumerated\":" << r.enumerated
           << ",\"survivors\":" << r.survivors << ",\"capped\":" << (r.capped ? "true" : "false")
           << ",\"rows\":[";
        for (std::size_t i = 0; i < r.survivors; ++i) {
            if (i) os << ",";
            os << "{";
            for (int c = 0; c < k; ++c)
                os << "\"pop" << (c + 1) << "\":\""
                   << resolver.label_at(r.keys[i][static_cast<std::size_t>(c)]) << "\",";
            os << "\"est\":" << r.est[i] << ",\"se\":" << r.se[i] << ",\"z\":" << r.z[i]
               << ",\"p\":" << r.p[i] << "}";
        }
        os << "]}\n";
        return;
    }
    const char d = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    for (int c = 0; c < k; ++c) os << "pop" << (c + 1) << d;
    os << "est" << d << "se" << d << "z" << d << "p\n";
    for (std::size_t i = 0; i < r.survivors; ++i) {
        for (int c = 0; c < k; ++c)
            os << resolver.label_at(r.keys[i][static_cast<std::size_t>(c)]) << d;
        os << r.est[i] << d << r.se[i] << d << r.z[i] << d << r.p[i] << "\n";
    }
}

}  // namespace

/// Shared body for the sweep — reused by the dedicated `f4-sweep`/`f3-sweep` subcommands AND
/// by the standalone `f4`/`f3`/`qpdstat` commands in --all-quartets/--all-triples mode. `k`
/// selects the arity (4 ⇒ run_f4_sweep, 3 ⇒ run_f3_sweep); `prog` is the stderr program name.
int run_fstat_sweep(const cfg::RunConfig& config, int k, const char* cmd) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe %s: --f2-dir is required\n", cmd);
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe %s: %s\n", cmd, dir.error.c_str());
        return cfg::kExitIoError;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe %s: %s\n", cmd, resolver.error().c_str());
        return cfg::kExitIoError;
    }

    // OPTIONAL --pops subset: resolve each name to a P-axis index. Empty ⇒ sweep the whole dir.
    std::vector<int> subset;
    for (const std::string& name : config.pops()) {
        const ResolveResult rr = resolver.resolve(name);
        if (!rr.ok) {
            std::fprintf(stderr, "steppe %s: %s\n", cmd, rr.error.c_str());
            return cfg::kExitInvalidConfig;
        }
        subset.push_back(rr.index);
    }

    // Build the sweep request from the frozen config (--min-z / --top-k / --sure).
    //
    // TopK now means a DEVICE-BOUNDED reservoir (the GPU keeps the K most-significant |z| in a
    // fixed CAP=O(K) buffer with a rising tau threshold), so HOST RAM is O(K) regardless of how
    // many billions are computed. A bare --all-quartets/--all-triples therefore DEFAULTS to
    // TopK=kFstatDefaultSweepTopK (1e6) so a full C(P,4) sweep cannot OOM the host. --top-k
    // overrides K; --min-z stays the tau FLOOR (a device pre-filter) and COEXISTS with the K cap.
    SweepRequest req;
    req.pop_subset = subset;
    req.sure = config.sweep_sure();
    req.min_z = config.sweep_min_z();  // the device tau FLOOR (default 3.0; --min-z raises it).
    if (config.sweep_top_k() > 0) {
        req.filter = SweepFilter::TopK;
        req.top_k = static_cast<std::size_t>(config.sweep_top_k());
    } else if (config.sweep_all_combinations()) {
        // Bare sweep on a standalone-stat command: default to the bounded 1M top-K cap so the
        // host cannot OOM on a signal-rich full sweep (the |z| filter alone is NOT a real bound).
        req.filter = SweepFilter::TopK;
        req.top_k = steppe::kFstatDefaultSweepTopK;
    } else {
        // The dedicated f4-sweep/f3-sweep subcommands keep the explicit MinZ default.
        req.filter = SweepFilter::MinZ;
    }

    SweepResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe %s: no CUDA device available (steppe is a GPU product; a "
                         "CUDA-capable GPU is required)\n", cmd);
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = (k == 4) ? run_f4_sweep(dev_f2, req, resources)
                          : run_f3_sweep(dev_f2, req, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe %s: device error: %s\n", cmd, e.what());
        return cfg::kExitRuntimeError;
    }

    // A CAPPED sweep is a hard, actionable refusal (not a domain row): tell the user and fail.
    if (result.capped) {
        std::fprintf(stderr,
                     "steppe %s: refusing to enumerate %zu combinations (> the maxcomb cap). "
                     "Pass --sure to override, or restrict --pops.\n",
                     cmd, result.enumerated);
        return cfg::kExitInvalidConfig;
    }

    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe %s: unknown --format '%s' (csv|tsv|json)\n",
                     cmd, config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    // Destination precedence: --shard-dir (a CSV file under the dir) > --out FILE > stdout.
    // The survivor set (post |z| / top-k filter) is the SMALL output; the full C(P,k) table
    // is never materialized (the device keeps it). --shard-dir is the sweep-scale destination
    // (a stable on-disk path the survivors are written to, vs piping a big stdout stream).
    if (!config.shard_dir().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(config.shard_dir(), ec);
        if (ec) {
            std::fprintf(stderr, "steppe %s: cannot create --shard-dir '%s': %s\n",
                         cmd, config.shard_dir().c_str(), ec.message().c_str());
            return cfg::kExitIoError;
        }
        const char* ext = (fmt == OutputFormat::Json) ? "json"
                          : (fmt == OutputFormat::Tsv) ? "tsv" : "csv";
        const std::filesystem::path shard_path =
            std::filesystem::path(config.shard_dir()) /
            (std::string("survivors.") + ext);
        std::ofstream out(shard_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe %s: cannot open shard file: %s\n",
                         cmd, shard_path.string().c_str());
            return cfg::kExitIoError;
        }
        emit_sweep(out, fmt, result, resolver, k);
        out.flush();
        std::fprintf(stderr, "steppe %s: enumerated %zu, %zu survivors -> %s\n",
                     cmd, result.enumerated, result.survivors, shard_path.string().c_str());
        return cfg::exit_code_for(result.status);
    }

    if (config.out_file().empty()) {
        emit_sweep(std::cout, fmt, result, resolver, k);
    } else {
        std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe %s: cannot open --out file: %s\n",
                         cmd, config.out_file().c_str());
            return cfg::kExitIoError;
        }
        emit_sweep(out, fmt, result, resolver, k);
    }

    // Report the sweep summary to stderr (observability; not on the data stream).
    std::fprintf(stderr, "steppe %s: enumerated %zu, %zu survivors\n",
                 cmd, result.enumerated, result.survivors);
    return cfg::exit_code_for(result.status);
}

int run_f4_sweep_command(const cfg::RunConfig& config) {
    return run_fstat_sweep(config, 4, "f4-sweep");
}
int run_f3_sweep_command(const cfg::RunConfig& config) {
    return run_fstat_sweep(config, 3, "f3-sweep");
}

}  // namespace steppe::app
