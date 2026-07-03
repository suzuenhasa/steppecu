// src/device/stream_f2_blocks.hpp
//
// The CUDA-free StreamTarget descriptor: the seam the CUDA-free orchestrator uses to
// tell the CUDA backend where a streamed f2_blocks result should be spilled (host RAM
// or disk). It names the destination without naming any CUDA sink type.
//
// Reference: docs/reference/src_device_stream_f2_blocks.hpp.md
#ifndef STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP
#define STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP

#include <cstddef>
#include <string>

#include "device/tier_select.hpp"
#include "device/f2_blocks_out.hpp"
#include "steppe/fstats.hpp"

namespace steppe {

// Forward declaration only — the full definition lives in device/backend.hpp. A
// pointer is all RedecodeSource needs, which keeps this header free of the decode
// interface (and of any io:: dependency).
struct DecodeTileView;

}  // namespace steppe

namespace steppe::device {

// StreamTarget request descriptor — reference §4
struct StreamTarget {
    OutputTier tier = OutputTier::HostRam;
    F2BlockTensor* host_dst = nullptr;
    std::string disk_path;
    DiskF2Blocks* disk_dst = nullptr;
};

// RedecodeSource: the optional per-chunk input override for stream_f2_blocks_impl.
// When present, the streamed engine re-decodes the packed genotypes for each chunk
// tile [s_lo, s_hi) in KEPT-column space instead of copying that tile from a dense
// host Q/V/N (which extract-f2 never materializes at high P). The decode is
// deterministic, so the engine sees byte-identical dQ_raw/dV_raw/dN_raw and the f2
// bytes stay bit-identical to the resident path. Host pointers only (CUDA-free).
// Reference §6a
struct RedecodeSource {
    const DecodeTileView* base_view = nullptr;
    const char* ref = nullptr;
    const char* alt = nullptr;
    const int* chrom = nullptr;
    const double* genpos = nullptr;
    const double* physpos = nullptr;
    const FilterConfig* filter = nullptr;
    const std::size_t* pop_individuals = nullptr;
    std::size_t n_pop = 0;
    double maxmiss = 1.0;
    // kept column -> raw .snp row index; strictly increasing, length M_kept.
    const long* kept_to_raw = nullptr;
    long M_kept = 0;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_STREAM_F2_BLOCKS_HPP
