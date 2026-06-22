// src/io/ploidy_detect.hpp
//
// AT2 pseudo-haploid per-sample PLOIDY auto-detection (adjust_pseudohaploid=TRUE) —
// the io-leaf decode-front-end step that classifies each gathered sample as diploid
// (ploidy 2) or pseudo-haploid (ploidy 1) so the downstream decode uses AT2's
// per-sample N convention (architecture.md §5 S0/S1; the f2-estimator pseudo-haploid
// fix, docs/research/f2-estimator-at2.md).
//
// THE RULE (verified against admixtools 2.0.10 src/cpp_readgeno.cpp cpp_*_ploidy):
// AT2 initializes every sample to ploidy 1, scans the FIRST ntest (= 1000) SNPs, and
// bumps a sample to ploidy 2 the moment it sees a HETEROZYGOUS call (genotype code 1)
// — a haploid genome can never be heterozygous, so a het ⇒ diploid; no het in the
// window ⇒ pseudo-haploid. An all-missing / all-homozygous prefix stays ploidy 1.
// This is PER SAMPLE, so MIXED-PLOIDY populations (a diploid and a pseudo-haploid
// sample in the same pop — real for aDNA: Turkey_N, Serbia, Yamnaya, Karitiana) are
// classified correctly (the per-pop ploidy tag the old code carried could not).
//
// SCANS THE TILE, NOT THE DISK: a GenotypeTile already holds each gathered sample's
// packed SNP-prefix bytes, so detection reads the SAME bytes the decode will, with no
// extra I/O — it scans the first min(kPloidyDetectSnps, tile.n_snp) SNPs of each
// sample. (When the tile covers fewer than kPloidyDetectSnps SNPs the window is the
// whole record; AT2's ntest likewise caps at the available SNP count.)
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, no core/device
// dependency, no CUDA. Uses only the io format primitives (code_in_byte, kHetCode).
#ifndef STEPPE_IO_PLOIDY_DETECT_HPP
#define STEPPE_IO_PLOIDY_DETECT_HPP

#include <vector>

#include "io/genotype_tile.hpp"

namespace steppe::io {

/// Detect the per-sample ploidy (AT2 adjust_pseudohaploid) for every gathered
/// individual in `tile`, returning a vector of length `tile.n_individuals` (parallel
/// to the gathered sample axis): 2 (diploid) if the sample has a het call in its first
/// min(kPloidyDetectSnps, tile.n_snp) SNPs, else 1 (pseudo-haploid). Does NOT mutate
/// the tile. Pure function of the tile bytes; safe to call once per tile.
[[nodiscard]] std::vector<int> detect_sample_ploidy(const GenotypeTile& tile);

}  // namespace steppe::io

#endif  // STEPPE_IO_PLOIDY_DETECT_HPP
