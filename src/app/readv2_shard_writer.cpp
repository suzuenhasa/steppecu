// src/app/readv2_shard_writer.cpp — the streaming per-pair READv2 row writer.
// Reference: docs/reference/src_app_readv2_shard_writer.cpp.md
#include "app/readv2_shard_writer.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <utility>

#include "core/config/exit_code.hpp"

namespace steppe::app {

namespace {

const char* ext_for(OutputFormat fmt) {
    return (fmt == OutputFormat::Json) ? "json" : (fmt == OutputFormat::Tsv) ? "tsv" : "csv";
}

void write_prologue(std::ostream& os, OutputFormat fmt, const Readv2Manifest& info) {
    if (fmt == OutputFormat::Json) {
        // NOTE: `background` is the all-pairs median, known only AFTER every pair is
        // reduced — but rows stream out during that same pass, so it cannot appear in
        // the streamed prologue without buffering all C(N,2) rows. It is reported in the
        // shard manifest and on stderr instead; the frozen ROW schema is unaffected.
        os << "{\"n_individuals\":" << info.n_individuals << ",\"n_pairs\":" << info.n_pairs
           << ",\"norm\":" << json_quote(info.norm) << ",\"window_snps\":" << info.window_snps
           << ",\"status\":\"ok\",\"rows\":[";
    } else {
        readv2_emit_header(os, fmt);
    }
}

}  // namespace

Readv2ShardWriter::Readv2ShardWriter(std::ostream& os, OutputFormat fmt,
                                     const Readv2Manifest& info)
    : fmt_(fmt), info_(info), stream_(&os) {
    write_prologue(os, fmt_, info_);
}

Readv2ShardWriter::Readv2ShardWriter(std::string dir, OutputFormat fmt, long shard_stride,
                                     const Readv2Manifest& info)
    : fmt_(fmt), info_(info), sharded_(true), dir_(std::move(dir)),
      shard_stride_(shard_stride > 0 ? shard_stride : 1) {}

void Readv2ShardWriter::write_row_to(std::ostream& os, bool& first, const Readv2OutRow& row) {
    if (fmt_ == OutputFormat::Json) {
        if (!first) os << ",";
        readv2_emit_json_row(os, row);
    } else {
        readv2_emit_row(os, fmt_, row);
    }
    first = false;
}

void Readv2ShardWriter::append(const Readv2OutRow& row, int sampleA_index) {
    if (!sharded_) {
        write_row_to(*stream_, single_first_, row);
        return;
    }
    const int shard = static_cast<int>(static_cast<long>(sampleA_index) / shard_stride_);
    auto it = shards_.find(shard);
    if (it == shards_.end()) {
        char name[64];
        std::snprintf(name, sizeof(name), "readv2.%05d.%s", shard, ext_for(fmt_));
        const std::filesystem::path p = std::filesystem::path(dir_) / name;
        auto os = std::make_unique<std::ofstream>(p, std::ios::binary | std::ios::trunc);
        if (!*os) {
            worst_error_ = "cannot open shard file: " + p.string();
            return;
        }
        write_prologue(*os, fmt_, info_);
        it = shards_.emplace(shard, Shard{std::move(os), true}).first;
    }
    write_row_to(*it->second.os, it->second.first, row);
}

std::optional<int> Readv2ShardWriter::finish(const char* cmd) {
    namespace cfg = steppe::config;
    if (!sharded_) {
        if (fmt_ == OutputFormat::Json) *stream_ << "]}\n";
        stream_->flush();
        if (!stream_->good()) {
            std::fprintf(stderr, "steppe %s: write failed (short write / closed pipe)\n", cmd);
            return cfg::kExitIoError;
        }
        return std::nullopt;
    }

    if (!worst_error_.empty()) {
        std::fprintf(stderr, "steppe %s: %s\n", cmd, worst_error_.c_str());
        return cfg::kExitIoError;
    }
    for (auto& [idx, shard] : shards_) {
        (void)idx;
        if (fmt_ == OutputFormat::Json) *shard.os << "]}\n";
        shard.os->flush();
        if (!shard.os->good()) {
            std::fprintf(stderr, "steppe %s: shard write failed\n", cmd);
            return cfg::kExitIoError;
        }
    }

    // Manifest sidecar.
    const std::filesystem::path mpath =
        std::filesystem::path(dir_) / "readv2.manifest.json";
    std::ofstream m(mpath, std::ios::binary | std::ios::trunc);
    if (!m) {
        std::fprintf(stderr, "steppe %s: cannot write manifest: %s\n", cmd, mpath.string().c_str());
        return cfg::kExitIoError;
    }
    m << "{\"n_individuals\":" << info_.n_individuals << ",\"n_pairs\":" << info_.n_pairs
      << ",\"n_shards\":" << shards_.size() << ",\"shard_stride\":" << shard_stride_
      << ",\"window_snps\":" << info_.window_snps << ",\"norm\":" << json_quote(info_.norm)
      << ",\"background\":" << json_double(info_.background)
      << ",\"schema\":\"sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z\"}\n";
    m.flush();
    if (!m.good()) {
        std::fprintf(stderr, "steppe %s: manifest write failed\n", cmd);
        return cfg::kExitIoError;
    }
    return std::nullopt;
}

}  // namespace steppe::app
