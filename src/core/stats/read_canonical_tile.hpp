// src/core/stats/read_canonical_tile.hpp
//
// read_canonical_tile — the genotype front-end FORMAT DISPATCH (M-FR-2; docs/design/
// format-readers.md §2.4). The single place the four genotype-path tools
// (extract_f2 / qpDstat-B / qpfstats / DATES) turn an open io::GenoReader + an
// IndPartition into the canonical individual-major io::GenotypeTile, regardless of
// the on-disk axis:
//
//   * TGENO (individual-major)  -> reader.read_tile (UNCHANGED — the existing path).
//   * GENO  (SNP-major PA)       -> reader.read_snp_major_tile (the `io`-leaf gather)
//                                   + ComputeBackend::transpose_to_canonical (on the
//                                   GPU; the CpuBackend host-loop oracle) -> the
//                                   canonical individual-major tile.
//
// After this call the tile is the SAME canonical packing every downstream consumer
// (detect_ploidy / decode_af / the regime-B filter / f2 / D / DATES / qpfstats)
// already expects, so NOTHING downstream changes — only the read-time layout
// transposes (the transpose-on-read thesis, format-readers.md §2.1).
//
// LAYERING (architecture.md §4): this is the `app`/orchestration wiring point where
// the CUDA-FREE `io` leaf meets the CUDA-FREE ComputeBackend seam — NOT a CUDA TU
// and NOT inside the `io` leaf (the leaf may not call a kernel). It is compiled into
// steppe_core (which links steppe::io + steppe::device) and is reused by both
// steppe_core's genotype tools AND steppe_extract's run_extract_f2 (which links
// steppe::core). No CUDA header appears here; the transpose is reached through the
// CUDA-free backend virtual.
#ifndef STEPPE_CORE_STATS_READ_CANONICAL_TILE_HPP
#define STEPPE_CORE_STATS_READ_CANONICAL_TILE_HPP

#include <cstddef>

#include "io/geno_reader.hpp"     // GenoReader, GenotypeTile, SnpMajorTile
#include "io/ind_reader.hpp"      // IndPartition

namespace steppe {

class ComputeBackend;  // device/backend.hpp (CUDA-free seam); fwd-declared to keep this lean.

namespace core {

/// Read SNPs [snp_begin, snp_end) for the selected populations into the canonical
/// individual-major io::GenotypeTile, dispatching on the on-disk format:
///   * GenoFormat::Tgeno -> reader.read_tile(part, snp_begin, snp_end) (unchanged).
///   * GenoFormat::Geno  -> reader.read_snp_major_tile(...) + backend.transpose_to_
///       canonical(...) -> a GenotypeTile with the SAME packing/pop_offsets/labels.
///
/// `backend` is consulted ONLY on the GENO path (the on-device transpose); the TGENO
/// path never touches it. The returned tile's `sample_ploidy` is EMPTY (ploidy is
/// detected later, on the canonical tile — the M-FR-0 on-device prepass or the
/// caller's explicit vector); the dispatch does not touch ploidy.
///
/// Throws std::runtime_error on any reader/transpose failure (the same exception
/// contract read_tile / read_snp_major_tile carry) or on an unknown format.
[[nodiscard]] io::GenotypeTile read_canonical_tile(io::GenoReader& reader,
                                                   const io::IndPartition& part,
                                                   ComputeBackend& backend,
                                                   std::size_t snp_begin,
                                                   std::size_t snp_end);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_READ_CANONICAL_TILE_HPP
