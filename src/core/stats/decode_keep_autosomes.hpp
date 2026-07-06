// src/core/stats/decode_keep_autosomes.hpp
//
// The two shared decode helpers the genotype-path tools lean on after the front-end:
// make_decode_tile_view fills a DecodeTileView from a canonical tile + a sample-ploidy
// vector, and decode_and_keep_autosomes runs the resident-or-host decode and keeps the
// autosomal SNPs into Q/V plus the position vectors. Sits at the CUDA-free seam.
//
// Lifetime: the returned DecodeTileView aliases the caller's sample_ploidy vector, so the
// caller must keep that vector alive for the view's whole lifetime.
//
// Reference: docs/reference/src_core_stats_decode_keep_autosomes.hpp.md
#ifndef STEPPE_CORE_STATS_DECODE_KEEP_AUTOSOMES_HPP
#define STEPPE_CORE_STATS_DECODE_KEEP_AUTOSOMES_HPP

#include <vector>

#include "device/backend.hpp"
#include "device/device_decode_result.hpp"
#include "io/genotype_tile.hpp"
#include "io/snp_reader.hpp"

namespace steppe {
namespace core {

// DecodeTileView wiring from a canonical tile + sample-ploidy vector.
[[nodiscard]] DecodeTileView make_decode_tile_view(const io::GenotypeTile& tile,
                                                   const std::vector<int>& sample_ploidy, int P);

// Resident-or-host decode + autosome keep result.
struct DecodeKeepResult {
    bool resident = false;
    device::DeviceDecodeResult ddr;
    std::vector<double> Qk;
    std::vector<double> Vk;
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    std::vector<double> physpos_kept;
};

// Decode the tile (device-resident when a GPU is present, host otherwise) and keep the
// autosomal SNPs into the returned result.
[[nodiscard]] DecodeKeepResult decode_and_keep_autosomes(ComputeBackend& be,
                                                         const io::GenotypeTile& tile,
                                                         const io::SnpTable& snptab, int P, long M);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_DECODE_KEEP_AUTOSOMES_HPP
