// Reference: docs/reference/src_app_readv2_emit.hpp.md
// src/app/readv2_emit.hpp
//
// The single-row READv2 serializers (csv/tsv/json) matching the frozen Phase-0 schema:
//   sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z
// Reuses the result_emit primitives (csv_field / json_quote / fmt_double / json_double)
// so numbers format identically to every other command. App-layer, CUDA-free.
#ifndef STEPPE_APP_READV2_EMIT_HPP
#define STEPPE_APP_READV2_EMIT_HPP

#include <cstdint>
#include <ostream>
#include <string>

#include "app/result_emit.hpp"

namespace steppe::app {

// One output row, labels already resolved + canonicalized (sampleA < sampleB).
struct Readv2OutRow {
    std::string sampleA;
    std::string sampleB;
    long n_windows = 0;
    std::int64_t n_overlap = 0;
    double p0_mean = 0.0;
    double p0_norm = 0.0;
    std::string degree;
    double z = 0.0;  // NaN -> NA (csv/tsv) / null (json)
};

// The csv/tsv header line (exact frozen column order), written once.
void readv2_emit_header(std::ostream& os, OutputFormat fmt);

// One csv/tsv row.
void readv2_emit_row(std::ostream& os, OutputFormat fmt, const Readv2OutRow& row);

// One JSON row object {...} (the streaming writer wraps these in the rows array).
void readv2_emit_json_row(std::ostream& os, const Readv2OutRow& row);

}  // namespace steppe::app

#endif  // STEPPE_APP_READV2_EMIT_HPP
