// src/io/genotype_tile.hpp
//
// GenotypeTile — the plain output struct the `io` decode front-end produces and
// the device/CPU decode backend consumes (architecture.md §4 `io` LEAF emits
// plain data structs, §5 S0, §11.1; ROADMAP §2 Q/V/N contract upstream, M1).
//
// This is the seam between the host-pure `io` reader and the (CUDA-free)
// ComputeBackend::decode_af. It carries PACKED genotype bytes plus the layout
// descriptor needed to decode them, and the population partition (segment offsets
// into the sample axis) — NO decode happens in `io` (architecture.md §5 S0: `io`
// reads/tiles; the device kernel decodes). The struct is plain std::vector /
// POD, so it crosses the layer boundary with no `io`→core/device dependency and
// no CUDA type (architecture.md §4).
//
// TGENO INDIVIDUAL-MAJOR PACKING (the real AADR layout): the tile holds, for each
// SELECTED individual (gathered pop-contiguous), the first `bytes_per_record`
// packed bytes of that individual's record — i.e. the SNP prefix covering SNPs
// [0, n_snp). The individuals are laid out segment-by-segment: pop 0's
// individuals, then pop 1's, ... in the Q/V/N row order. `pop_offsets` gives the
// half-open individual-index range of each population's segment, so a decoder can
// reduce over exactly the individuals of one population.
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, depends on
// nothing in core/device, no CUDA.
#ifndef STEPPE_IO_GENOTYPE_TILE_HPP
#define STEPPE_IO_GENOTYPE_TILE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace steppe::io {

/// A decoded-ready genotype tile in TGENO individual-major packing: packed bytes
/// for the selected individuals (gathered into population-contiguous order),
/// the per-individual record stride, the SNP count this tile covers, and the
/// population partition over the individual (sample) axis.
///
/// `packed` length is `n_individuals * bytes_per_record`. Individual `g`
/// (0-based over the gathered set) occupies `packed[g*bytes_per_record ..
/// (g+1)*bytes_per_record)`; SNP `s` of that individual is the 2-bit code at byte
/// `s/4`, position `s%4` (MSB-first; eigenstrat_format::code_in_byte). The first
/// `pop_offsets[p+1] - pop_offsets[p]` individuals after index `pop_offsets[p]`
/// belong to population `p` (in Q/V/N row order). `pop_labels[p]` names that
/// population. P == pop_labels.size() == pop_offsets.size() - 1.
struct GenotypeTile {
    /// Packed genotype bytes for the gathered individuals, pop-contiguous.
    std::vector<std::uint8_t> packed;

    /// Bytes per individual record in `packed` (== ceil(n_snp/4) for this tile's
    /// SNP count). DERIVED from the format constants, never hardcoded.
    std::size_t bytes_per_record = 0;

    /// Number of SNPs this tile covers (the SNP axis length M). SNP codes
    /// 0..n_snp-1 are valid within each record.
    std::size_t n_snp = 0;

    /// Number of gathered individuals across all selected populations (the sum of
    /// the segment sizes). == packed.size() / bytes_per_record.
    std::size_t n_individuals = 0;

    /// Population segment boundaries over the individual axis: population `p`
    /// owns gathered individuals [pop_offsets[p], pop_offsets[p+1]). Size P+1,
    /// pop_offsets[0] == 0, pop_offsets[P] == n_individuals.
    std::vector<std::size_t> pop_offsets;

    /// Population labels in Q/V/N row order (size P), parallel to the segments.
    std::vector<std::string> pop_labels;

    /// PER-SAMPLE PLOIDY (length n_individuals, parallel to the GATHERED sample axis
    /// — sample_ploidy[g] is the ploidy of gathered individual g): 2 diploid, 1
    /// pseudo-haploid. AT2 adjust_pseudohaploid auto-detection (a het call in the
    /// leading SNPs ⇒ diploid, else pseudo-haploid), filled by detect_sample_ploidy.
    /// EMPTY ⇒ "not detected": the caller treats every sample as the uniform fallback
    /// ploidy (the legacy all-diploid path). When non-empty its size == n_individuals.
    std::vector<int> sample_ploidy;

    /// Number of populations P (the leading dimension of the decoded Q/V/N).
    [[nodiscard]] std::size_t n_pop() const noexcept { return pop_labels.size(); }
};

}  // namespace steppe::io

#endif  // STEPPE_IO_GENOTYPE_TILE_HPP
