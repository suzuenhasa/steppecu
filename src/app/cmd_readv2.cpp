// src/app/cmd_readv2.cpp
//
// The `steppe readv2` command: pseudo-haploid windowed-mismatch kinship. Resolves each
// selected Genetic ID to its own singleton sweep index (the PER-INDIVIDUAL modeling
// step, NOT the pop-collapse), fail-fast validates the inputs (samples resolve, N>=2,
// C(N,2) under the cap unless --sure, diploid/het rejected inside run_readv2), then runs
// the GPU sweep and streams the frozen-schema rows to a shard writer or a single output.
// App-only C++; the GPU is reached only through the run_readv2 seam.
#include "app/cmd_readv2.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/pop_resolver.hpp"
#include "app/readv2_emit.hpp"
#include "app/readv2_shard_writer.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/resources.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/individual_partition.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/readv2.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Overflow-safe C(N,2) (mirrors the sweep's choose_saturating for k=2).
[[nodiscard]] unsigned long long choose2_saturating(long long n) {
    if (n < 2) return 0ULL;
    const unsigned long long a = static_cast<unsigned long long>(n);
    if (a > (~0ULL) / (a - 1)) return ~0ULL;
    return a * (a - 1) / 2ULL;
}

// Read a --samples file of Genetic IDs (one per line; blanks/whitespace ignored).
[[nodiscard]] bool read_samples_file(const std::string& path, std::vector<std::string>& out,
                                     std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open --samples file: " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tok;
        if (ls >> tok) out.push_back(tok);
    }
    if (out.empty()) { err = "--samples file is empty: " + path; return false; }
    return true;
}

}  // namespace

int run_readv2_command(const cfg::RunConfig& config) {
    if (config.qpdstat_prefix().empty()) {
        std::fprintf(stderr, "steppe readv2: --prefix PREFIX.{geno,snp,ind} is required\n");
        return cfg::kExitInvalidConfig;
    }
    const io::GenotypeTriple triple = io::resolve_genotype_triple(config.qpdstat_prefix());

    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe readv2: unknown --format '%s' (csv|tsv|json)\n",
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    // Optional --samples restriction (a file of Genetic IDs).
    std::optional<std::vector<std::string>> samples;
    if (!config.samples_file().empty()) {
        std::vector<std::string> ids;
        std::string serr;
        if (!read_samples_file(config.samples_file(), ids, serr)) {
            std::fprintf(stderr, "steppe readv2: %s\n", serr.c_str());
            return cfg::kExitInvalidConfig;
        }
        samples = std::move(ids);
    }

    // PER-INDIVIDUAL resolution: each Genetic ID is its own singleton sweep index.
    io::IndPartition part;
    std::vector<std::string> pop_labels;
    try {
        io::GenoReader reader(triple.geno);
        const io::GenoFormat geno_fmt = reader.header().format;
        const std::size_t n_present = reader.records_present();
        part = io::read_individual_partition(geno_fmt, triple.ind, samples, n_present);
        pop_labels.reserve(part.groups.size());
        for (const io::PopGroup& g : part.groups) pop_labels.push_back(g.label);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe readv2: %s\n", e.what());
        return cfg::kExitInvalidConfig;
    }

    const int N = static_cast<int>(pop_labels.size());
    if (N < 2) {
        std::fprintf(stderr, "steppe readv2: need >= 2 individuals to form a pair (got %d)\n", N);
        return cfg::kExitInvalidConfig;
    }

    // Enumeration cap (mirrors the f-stat sweep --sure gate).
    const unsigned long long n_pairs = choose2_saturating(N);
    if (n_pairs > kFstatMaxComb && !config.sweep_sure()) {
        std::fprintf(stderr,
                     "steppe readv2: refusing to enumerate %llu pairs (> the maxcomb cap). "
                     "Pass --sure to override, or restrict --samples.\n", n_pairs);
        return cfg::kExitInvalidConfig;
    }

    const PopResolver resolver(pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe readv2: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    Readv2Options opts;
    opts.window_snps = config.window_snps();
    opts.norm = (config.norm_mode() == "mean") ? Readv2Norm::Mean : Readv2Norm::Median;
    opts.min_overlap = config.min_overlap();
    opts.autosomes_only = config.filter().autosomes_only;  // READv2 default on; --no-auto-only off

    Readv2Manifest info;
    info.n_individuals = static_cast<std::size_t>(N);
    info.n_pairs = static_cast<std::size_t>(n_pairs);
    info.norm = config.norm_mode();
    info.window_snps = opts.window_snps;

    // The sink: resolve i/j -> Genetic-ID labels, canonicalize sampleA < sampleB
    // lexicographically (matching the concord validator's pair_key), format, append.
    auto make_sink = [&](Readv2ShardWriter& writer) {
        return [&writer, &resolver](const Readv2PairRow& row) {
            const std::string& li = resolver.label_at(row.i);
            const std::string& lj = resolver.label_at(row.j);
            const bool i_first = (li <= lj);
            Readv2OutRow out;
            out.sampleA = i_first ? li : lj;
            out.sampleB = i_first ? lj : li;
            out.n_windows = row.n_windows;
            out.n_overlap = row.n_overlap;
            out.p0_mean = row.p0_mean;
            out.p0_norm = row.p0_norm;
            out.degree = row.degree;
            out.z = row.z;
            const int sampleA_index = i_first ? row.i : row.j;
            writer.append(out, sampleA_index);
        };
    };

    Readv2Result result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "readv2")) return cfg::kExitRuntimeError;

        if (!config.shard_dir().empty()) {
            std::error_code ec;
            std::filesystem::create_directories(config.shard_dir(), ec);
            if (ec) {
                std::fprintf(stderr, "steppe readv2: cannot create --shard-dir '%s': %s\n",
                             config.shard_dir().c_str(), ec.message().c_str());
                return cfg::kExitIoError;
            }
            // Shard stride: keep every pair of a given first-sample together; ~a few
            // hundred first-samples per shard is a reasonable default.
            const long shard_stride = 256;
            Readv2ShardWriter writer(config.shard_dir(), fmt, shard_stride, info);
            auto sink = make_sink(writer);
            result = run_readv2(triple.geno, triple.snp, triple.ind, part, opts, sink, resources);
            writer.set_background(result.background);  // for the manifest sidecar
            if (const auto rc = writer.finish("readv2")) return *rc;
        } else if (config.out_file().empty()) {
            Readv2ShardWriter writer(std::cout, fmt, info);
            auto sink = make_sink(writer);
            result = run_readv2(triple.geno, triple.snp, triple.ind, part, opts, sink, resources);
            if (const auto rc = writer.finish("readv2")) return *rc;
        } else {
            std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
            if (!out) {
                std::fprintf(stderr, "steppe readv2: cannot open --out file: %s\n",
                             config.out_file().c_str());
                return cfg::kExitIoError;
            }
            Readv2ShardWriter writer(out, fmt, info);
            auto sink = make_sink(writer);
            result = run_readv2(triple.geno, triple.snp, triple.ind, part, opts, sink, resources);
            if (const auto rc = writer.finish("readv2")) return *rc;
        }
    } catch (const std::invalid_argument& e) {
        // Fail-fast INPUT reject (e.g. a diploid/het sample) -> invalid config.
        std::fprintf(stderr, "steppe readv2: %s\n", e.what());
        return cfg::kExitInvalidConfig;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe readv2: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    std::fprintf(stderr, "steppe readv2: %zu individuals, %zu pairs, %zu emitted, background=%.8g\n",
                 result.n_individuals, result.n_pairs, result.n_emitted, result.background);
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
