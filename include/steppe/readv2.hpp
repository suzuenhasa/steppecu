// include/steppe/readv2.hpp
//
// Public, CUDA-free entry point for READv2: pseudo-haploid windowed-mismatch kinship.
// For every unordered sample pair it computes P0_mean (the mean over non-overlapping
// SNP-count windows of the per-window allele-mismatch proportion), normalizes by the
// all-pairs MEDIAN background to P0_norm, classifies a relatedness degree, and reports
// a z confidence — the frozen Phase-0 schema:
//   sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z
// The GPU is reached only through the ComputeBackend seam (like run_dates / run_f4_sweep).
// Rows are handed to a streaming sink one at a time — C(N,2) rows never all materialize.
#ifndef STEPPE_READV2_HPP
#define STEPPE_READV2_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

// CUDA-free device seam + io singleton partition, forward-declared (no include so the
// public API header stays a leaf; the driver .cpp includes the full definitions).
namespace device {
struct Resources;
}  // namespace device
namespace io {
struct IndPartition;
}  // namespace io


// All-pairs background normalizer (MEDIAN default; MEAN opt-in via --norm mean).
enum class Readv2Norm { Median, Mean };

// Run knobs (validated by the CLI before this is built).
struct Readv2Options {
    int window_snps = 1000;                // non-overlapping SNP-count window (READv2 default)
    Readv2Norm norm = Readv2Norm::Median;  // background = median (default) | mean of P0_mean
    double min_overlap = 0.0;              // drop a pair with < this fraction of M0 comparable
    bool autosomes_only = true;            // restrict the SNP axis to chr 1-22 (READv2 convention)
    bool tiled = false;                    // use the shared-mem tiled mismatch kernel
};

// One emitted per-pair row. i/j are per-individual sweep indices (the sink maps them to
// Genetic-ID labels and canonicalizes sampleA < sampleB); degree is one of the four
// frozen tokens; z is NaN when n_windows < 2.
struct Readv2PairRow {
    int i = 0;
    int j = 0;
    long n_windows = 0;
    std::int64_t n_overlap = 0;
    double p0_mean = 0.0;
    double p0_norm = 0.0;
    double z = 0.0;
    const char* degree = "";
};

// The streaming sink: called once per surviving pair, in ascending pair-rank order.
using Readv2RowSink = std::function<void(const Readv2PairRow&)>;

// Run summary (the rows themselves went to the sink).
struct Readv2Result {
    std::size_t n_individuals = 0;
    std::size_t n_pairs = 0;
    std::size_t n_emitted = 0;
    double background = 0.0;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// Run READv2 over a genotype triple restricted to `individuals` (each a singleton pop
// = one sample). Streams SNP-tile chunks into a resident device bit-matrix, runs the
// all-pairs __popc mismatch sweep, forms the median background, and emits each pair to
// `sink`. Throws std::runtime_error on a fail-fast input reject (e.g. a diploid/het
// sample: READv2 v1 requires pseudo-haploid hardcalls).
[[nodiscard]] Readv2Result run_readv2(const std::string& geno, const std::string& snp,
                                      const std::string& ind,
                                      const io::IndPartition& individuals,
                                      const Readv2Options& opts, const Readv2RowSink& sink,
                                      device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_READV2_HPP
