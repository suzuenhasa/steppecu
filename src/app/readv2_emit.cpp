// src/app/readv2_emit.cpp — the frozen-schema READv2 row serializers (csv/tsv/json).
#include "app/readv2_emit.hpp"

#include <ostream>
#include <string>

namespace steppe::app {

void readv2_emit_header(std::ostream& os, OutputFormat fmt) {
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "sampleA" << sep << "sampleB" << sep << "n_windows" << sep << "n_overlap_sites"
       << sep << "P0_mean" << sep << "P0_norm" << sep << "degree" << sep << "z" << "\n";
}

void readv2_emit_row(std::ostream& os, OutputFormat fmt, const Readv2OutRow& row) {
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << csv_field(row.sampleA, sep) << sep << csv_field(row.sampleB, sep) << sep
       << row.n_windows << sep << row.n_overlap << sep << fmt_double(row.p0_mean) << sep
       << fmt_double(row.p0_norm) << sep << csv_field(row.degree, sep) << sep
       << fmt_double(row.z) << "\n";
}

void readv2_emit_json_row(std::ostream& os, const Readv2OutRow& row) {
    os << "{\"sampleA\":" << json_quote(row.sampleA) << ",\"sampleB\":" << json_quote(row.sampleB)
       << ",\"n_windows\":" << row.n_windows << ",\"n_overlap_sites\":" << row.n_overlap
       << ",\"P0_mean\":" << json_double(row.p0_mean) << ",\"P0_norm\":" << json_double(row.p0_norm)
       << ",\"degree\":" << json_quote(row.degree) << ",\"z\":" << json_double(row.z) << "}";
}

}  // namespace steppe::app
