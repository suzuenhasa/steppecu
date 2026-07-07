// src/core/stats/read_canonical_tile.hpp
//
// Declares read_canonical_tile, the one place the genotype-path tools turn an open
// genotype reader into the canonical individual-major GenotypeTile, dispatching on the
// on-disk format (TGENO passes through; SNP-major formats transpose on read). Sits at
// the CUDA-free seam between the io layer and the compute backend.
//
// Reference: docs/reference/src_core_stats_read_canonical_tile.hpp.md
#ifndef STEPPE_CORE_STATS_READ_CANONICAL_TILE_HPP
#define STEPPE_CORE_STATS_READ_CANONICAL_TILE_HPP

#include <cstddef>

#include "io/geno_reader.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_major_tile.hpp"

namespace steppe {

class ComputeBackend;

namespace core {

// read_canonical_tile — reference §4
[[nodiscard]] io::GenotypeTile read_canonical_tile(io::GenoReader& reader,
                                                   const io::IndPartition& part,
                                                   ComputeBackend& backend,
                                                   std::size_t snp_begin,
                                                   std::size_t snp_end);

// transpose_snp_major — the shared SNP-major -> canonical individual-major
// transpose seam (Identity encoding). Promoted from an anonymous-namespace
// helper so both the five-arm packed-format funnel and the native VCF ingest
// path (the sixth arm) share the one on-device transpose. No behavior change.
[[nodiscard]] io::GenotypeTile transpose_snp_major(const io::SnpMajorTile& src,
                                                   ComputeBackend& backend);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_READ_CANONICAL_TILE_HPP
