// src/app/cmd_ingest.cpp
//
// The `steppe ingest` handler: read the target-site table, run the native
// VcfReader over the .vcf.gz, write the per-site report TSV, and (only when
// --emit-tile is given) route the SnpMajorTile through the shared device
// transpose to write the raw canonical 2-bit tile bytes. The report path is
// pure host (no GPU) — the Stage-1 block-correctness gate never needs a device.
#include "app/cmd_ingest.hpp"

#include <cctype>
#include <cstdio>
#include <exception>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/exit_code_for_caught.hpp"
#include "core/config/exit_code.hpp"
#include "core/stats/read_canonical_tile.hpp"
#include "device/resources.hpp"
#include "io/genotype_tile.hpp"
#include "io/target_sites.hpp"
#include "io/vcf_reader.hpp"
#include "steppe/config.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

[[nodiscard]] const char* call_str(io::VcfCall c) {
    switch (c) {
        case io::VcfCall::Homref: return "homref";
        case io::VcfCall::Het: return "het";
        case io::VcfCall::Homalt: return "homalt";
        case io::VcfCall::Missing: return "missing";
        case io::VcfCall::Dropped: return "dropped";
    }
    return "dropped";
}

// Write the per-site report in the Stage-0 oracle schema (spec §9).
[[nodiscard]] bool write_report(const std::string& path, const io::VcfIngestResult& r,
                                std::string& err) {
    std::ofstream o(path, std::ios::trunc);
    if (!o) { err = "cannot open --report file: " + path; return false; }
    o << "rsID\tchrom\tpos37\tpos38\tA1\tA2\tcall\tdosage\tsource\tflip\tdrop_reason\n";
    for (const io::VcfSiteCall& c : r.calls) {
        o << c.rsid << '\t' << c.chrom << '\t' << c.pos37 << '\t' << c.pos38 << '\t' << c.a1
          << '\t' << c.a2 << '\t' << call_str(c.call) << '\t';
        if (c.dosage < 0) o << "NA"; else o << c.dosage;
        o << '\t' << c.source << '\t' << c.flip << '\t' << c.drop_reason << '\n';
    }
    return static_cast<bool>(o);
}

// Parse a "--device" ordinal list into a DeviceConfig (empty/"auto" -> default).
[[nodiscard]] bool parse_device(const std::string& raw, steppe::DeviceConfig& dc, std::string& err) {
    std::string s;
    for (char c : raw) if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty() || s == "auto") return true;
    std::vector<int> ords;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) {
            try { ords.push_back(std::stoi(tok)); }
            catch (...) { err = "--device ordinal '" + tok + "' is not an integer"; return false; }
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    dc.devices = std::move(ords);
    return true;
}

}  // namespace

int run_ingest(const IngestArgs& args) {
    if (args.vcf.empty()) {
        std::fprintf(stderr, "steppe ingest: --vcf PATH is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.targets.empty()) {
        std::fprintf(stderr,
                     "steppe ingest: --targets PATH is required "
                     "(the GRCh38 target-site table rsID chrom [pos37] pos38 A1 A2 ref38)\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.report.empty() && args.emit_tile.empty()) {
        std::fprintf(stderr, "steppe ingest: nothing to do — pass --report and/or --emit-tile\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.min_dp < 0 || args.min_gq < 0) {
        std::fprintf(stderr, "steppe ingest: --min-dp/--min-gq must be >= 0\n");
        return cfg::kExitInvalidConfig;
    }

    io::VcfIngestResult result;
    try {
        const io::TargetSites targets = io::read_target_sites(args.targets);
        io::VcfReader::Options opts;
        opts.min_dp = args.min_dp;
        opts.min_gq = args.min_gq;
        io::VcfReader reader(args.vcf, targets, args.sample, opts);
        result = reader.genotype();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe ingest: %s\n", e.what());
        return cfg::kExitIoError;
    }

    if (!args.report.empty()) {
        std::string err;
        if (!write_report(args.report, result, err)) {
            std::fprintf(stderr, "steppe ingest: %s\n", err.c_str());
            return cfg::kExitIoError;
        }
    }

    // The canonical-tile path needs a device for the shared transpose.
    if (!args.emit_tile.empty()) {
        try {
            steppe::DeviceConfig dc;
            std::string derr;
            if (!parse_device(args.device, dc, derr)) {
                std::fprintf(stderr, "steppe ingest: %s\n", derr.c_str());
                return cfg::kExitInvalidConfig;
            }
            device::Resources resources = device::build_resources(dc);
            if (!require_first_gpu(resources, "ingest")) return cfg::kExitRuntimeError;
            ComputeBackend& backend = *resources.gpus.front().backend;
            const io::GenotypeTile canon = core::transpose_snp_major(result.tile, backend);
            std::ofstream tf(args.emit_tile, std::ios::binary | std::ios::trunc);
            if (!tf) {
                std::fprintf(stderr, "steppe ingest: cannot open --emit-tile file: %s\n",
                             args.emit_tile.c_str());
                return cfg::kExitIoError;
            }
            tf.write(reinterpret_cast<const char*>(canon.packed.data()),
                     static_cast<std::streamsize>(canon.packed.size()));
            if (!tf) {
                std::fprintf(stderr, "steppe ingest: failed writing --emit-tile file\n");
                return cfg::kExitIoError;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe ingest: device/transpose error: %s\n", e.what());
            return exit_code_for_caught(e);
        }
    }

    const io::VcfCounts& c = result.counts;
    std::fprintf(stderr,
                 "steppe ingest: sample=%s sites=%zu | called_variant=%lld called_refblock=%lld "
                 "(homref=%lld het=%lld homalt=%lld) | missing=%lld dropped=%lld | records=%lld\n",
                 result.sample_id.c_str(), result.calls.size(), c.called_variant, c.called_refblock,
                 c.homref, c.het, c.homalt, c.missing_total, c.dropped_total, c.records_seen);
    return cfg::kExitOk;
}

}  // namespace steppe::app
