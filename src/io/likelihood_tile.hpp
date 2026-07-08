// src/io/likelihood_tile.hpp
//
// LikelihoodTile — the host POD hand-off the GL reader produces and the device
// upload consumes: a NORMALIZED LINEAR genotype-likelihood tensor
// [n_site x n_sample x 3] (FP64) plus a parallel present-mask and the site/sample
// metadata a consumer needs to join back to the panel. The GL analogue of
// io::GenotypeTile (which is packed 2-bit hard calls); this carries three doubles
// per (site, sample) instead — a genotype likelihood is a different data shape.
//
// LAYOUT: SITE-MAJOR. l[(site*n_sample + sample)*3 + g], where g = copies of panel
// A1 (the SAME axis as the 2-bit hard-call dosage, so an argmax over g is directly
// comparable to the hard-call call). Site-major gives coalesced per-site tiles for
// the PCAngsd covariance SYRK (SNP-tile-outer), and a fixed n_sample*3 stride for
// the per-sample ancIBD forward-backward.
//
// Host-only std::vector/POD, no CUDA.
#ifndef STEPPE_IO_LIKELIHOOD_TILE_HPP
#define STEPPE_IO_LIKELIHOOD_TILE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace steppe::io {

// Which FORMAT field produced the triplet. Recorded in the tile + the artifact
// header so a consumer never conflates a POSTERIOR (GP) with a LIKELIHOOD (PL/GL).
enum class GlField : std::uint8_t { PL = 0, GL = 1, GP = 2 };

// Minimal per-site metadata, parallel to the target-site table (panel order), so a
// tensor row is joinable back to the panel A1/A2.
struct LikelihoodSite {
    std::string rsid;
    int chrom = 0;
    long long pos37 = 0;
    long long pos38 = 0;
    char a1 = 'N';
    char a2 = 'N';
};

struct LikelihoodTile {
    std::vector<double> l;              // n_site*n_sample*3, SITE-MAJOR, g = copies-of-A1
    std::vector<std::uint8_t> present;  // n_site*n_sample (1 = observed, 0 = missing)
    std::size_t n_site = 0;
    std::size_t n_sample = 0;
    GlField field = GlField::PL;
    std::vector<std::string> sample_ids;   // n_sample
    std::vector<LikelihoodSite> sites;     // n_site, panel order

    // ancIBD path only (the --ancibd-native read): the phased haplotype hardcall bit
    // per (site, sample, haplotype), VCF-NATIVE {0 = REF/ancestral, 1 = ALT/derived,
    // 0xFF = missing/unphased}, SITE-MAJOR. Empty unless the reader was asked to keep
    // phase (the PCAngsd GP path leaves it empty). When populated, `l` is stored in
    // VCF-NATIVE (RR, RA, AA) order too (NOT the A1-copy swap) so g0/g1 + these bits
    // feed LoadH5Multi2.get_haplo_prob directly. Reference: ancibd-face-spec §0b, §4.1.
    std::vector<std::uint8_t> phased_gt;   // 2*n_site*n_sample (haplotype-minor)
    bool native_order = false;             // true -> `l` is VCF-native, phased_gt filled

    // Flat index of the g-triplet base for (site, sample) in `l`.
    [[nodiscard]] std::size_t base(std::size_t site, std::size_t sample) const noexcept {
        return (site * n_sample + sample) * 3;
    }
    [[nodiscard]] std::size_t mask_index(std::size_t site, std::size_t sample) const noexcept {
        return site * n_sample + sample;
    }
    // Flat index of the first (of two) phased allele bit for (site, sample).
    [[nodiscard]] std::size_t phase_base(std::size_t site, std::size_t sample) const noexcept {
        return (site * n_sample + sample) * 2;
    }
};

// One raw (pre-normalization, pre-polarity-swap, VCF-NATIVE-order) triplet row for
// the --emit-pl-raw debug dump — self-keyed by chrom:pos38:rsid + sample so the
// bit-exact gate can join it to a `bcftools query -f '[%PL ]'` dump on the SAME
// positions (critic fix #2) rather than diffing two differently-ordered site sets.
// The three values are the exact subfield tokens as they appeared in the file
// (no reformatting), so the compare is truly bit-exact.
struct RawGlRow {
    std::string rsid;
    int chrom = 0;
    long long pos38 = 0;
    std::string sample_id;
    std::string v0, v1, v2;  // VCF-native (RR, RA, AA) tokens, verbatim
};

}  // namespace steppe::io

#endif  // STEPPE_IO_LIKELIHOOD_TILE_HPP
