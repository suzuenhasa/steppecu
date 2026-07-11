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

#include "device/backend.hpp"  // DeviceGenotypeTile
#include "io/geno_reader.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_major_tile.hpp"

namespace steppe {

class ComputeBackend;

namespace core {

// device_load_enabled — the GPU-native load selector (STEPPE_GPU_LOAD). Default ON; "0"/"off"/
// "false"/"no" forces the legacy host round-trip path (the byte-exact invariance gate diffs the
// two). Shared by the shared front-end and the kinship reader so both honor the same switch.
[[nodiscard]] bool device_load_enabled();

// read_canonical_tile — reference §4
[[nodiscard]] io::GenotypeTile read_canonical_tile(io::GenoReader& reader,
                                                   const io::IndPartition& part,
                                                   ComputeBackend& backend,
                                                   std::size_t snp_begin,
                                                   std::size_t snp_end);

// read_canonical_tile_device — the GPU-native twin of read_canonical_tile. Builds the SAME
// canonical individual-major bytes DEVICE-RESIDENT (no host materialization, no per-tool
// re-upload): TGENO H2Ds its already-canonical read once; the SNP-major formats (GENO/PLINK/
// EIGENSTRAT/ANCESTRYMAP) transpose+gather on the GPU straight into the resident tile (the
// streamed arms scatter each block by byte-column). Requires a CUDA backend (device_count>0);
// the selector (STEPPE_GPU_LOAD / filter-inactive) chooses this vs the host read. The produced
// tile is byte-identical to read_canonical_tile's packed bytes (same transpose kernel).
[[nodiscard]] DeviceGenotypeTile read_canonical_tile_device(io::GenoReader& reader,
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
