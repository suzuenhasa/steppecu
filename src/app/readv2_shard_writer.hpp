// src/app/readv2_shard_writer.hpp
//
// The incremental per-pair row writer for READv2. Unlike the f-stat sweep's single
// host-materialized survivor table (bounded by a top-K filter), READv2 emits ONE row
// per unordered pair with no filter — C(N,2) rows, which do not fit in host memory at
// scale. So rows stream to disk the instant they are produced, never buffered into a
// std::vector<Row>. Two modes: a single stream (--out FILE / stdout, header + rows,
// JSON streamed too) and a shard directory (rows partitioned into readv2.NNNNN.<ext>
// files by the sampleA index + a manifest sidecar). App-layer, CUDA-free.
#ifndef STEPPE_APP_READV2_SHARD_WRITER_HPP
#define STEPPE_APP_READV2_SHARD_WRITER_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include "app/readv2_emit.hpp"
#include "app/result_emit.hpp"

namespace steppe::app {

// Run-level facts recorded in the JSON prologue / shard manifest.
struct Readv2Manifest {
    std::size_t n_individuals = 0;
    std::size_t n_pairs = 0;
    double background = 0.0;
    std::string norm = "median";
    int window_snps = 0;
};

class Readv2ShardWriter {
public:
    // Single-stream mode: writes the header (csv/tsv) or the JSON prologue to `os` now.
    Readv2ShardWriter(std::ostream& os, OutputFormat fmt, const Readv2Manifest& info);

    // Shard-directory mode: partitions rows into readv2.NNNNN.<ext> by sampleA index /
    // shard_stride; each shard file is self-contained; a readv2.manifest.json sidecar is
    // written at finish(). `dir` must already exist.
    Readv2ShardWriter(std::string dir, OutputFormat fmt, long shard_stride,
                      const Readv2Manifest& info);

    // Append one row (sampleA_index = the index of the sample that became sampleA).
    void append(const Readv2OutRow& row, int sampleA_index);

    // Record the all-pairs background for the shard manifest (known only after the run).
    void set_background(double background) { info_.background = background; }

    // Close JSON arrays / write the manifest, then verify the write. Returns an exit
    // code on I/O failure, or nullopt on success.
    [[nodiscard]] std::optional<int> finish(const char* cmd);

private:
    void write_row_to(std::ostream& os, bool& first, const Readv2OutRow& row);

    OutputFormat fmt_;
    Readv2Manifest info_;

    // Single-stream mode.
    std::ostream* stream_ = nullptr;
    bool single_first_ = true;

    // Shard mode.
    bool sharded_ = false;
    std::string dir_;
    long shard_stride_ = 1;
    struct Shard {
        std::unique_ptr<std::ostream> os;
        bool first = true;
    };
    std::map<int, Shard> shards_;
    std::string worst_error_;
};

}  // namespace steppe::app

#endif  // STEPPE_APP_READV2_SHARD_WRITER_HPP
