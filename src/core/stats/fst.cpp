// src/core/stats/fst.cpp
//
// run_fst: the standalone pairwise per-SNP Weir & Cockerham 1984 FST computed straight
// from genotype files. It reuses the shared genotype decode front-end (read the triple to
// the canonical population-contiguous tile) and then branches into its own per-site WC
// variance-component kernel — a distinct per-site reduction, NOT the f2/AF decode path
// (the AF grid drops the allele count n and the observed het h that WC needs). Native FP64.
//
// Diploid is forced (matching plink2 on a shared .bed): each 2-bit code is a diploid
// dosage 0/1/2. Missing genotypes are excluded per-site per-sample by the WC accumulator.
#include "steppe/fst.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/internal/decode_af.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"
#include "io/genotype_tile.hpp"
#include "io/snp_reader.hpp"

namespace steppe {

FstResult run_fst(const std::string& geno, const std::string& snp, const std::string& ind,
                  const std::string& popA, const std::string& popB,
                  device::Resources& resources) {
    FstResult res;
    res.precision_tag = Precision::Kind::Fp64;
    res.popA = popA;
    res.popB = popB;
    res.fst_ratio = std::nan("");
    res.fst_mean = std::nan("");

    if (popA == popB) {
        res.status = Status::InvalidConfig;
        return res;
    }

    ComputeBackend& be = device::primary_backend(resources);

    // Read the triple to the canonical tile, keeping ONLY the two requested populations
    // (population-contiguous, so each pop is one [begin, end) individual range).
    const std::vector<std::string> want{popA, popB};
    const core::GenotypeFrontEnd fe = core::read_genotype_front_end(
        geno, snp, ind, std::span<const std::string>(want), be);
    const io::GenotypeTile& tile = fe.tile;
    const io::SnpTable& snptab = fe.snptab;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P < 2 || M <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Resolve the two population labels to tile indices (order-independent).
    int idxA = -1, idxB = -1;
    for (std::size_t p = 0; p < tile.pop_labels.size(); ++p) {
        if (tile.pop_labels[p] == popA) idxA = static_cast<int>(p);
        if (tile.pop_labels[p] == popB) idxB = static_cast<int>(p);
    }
    if (idxA < 0 || idxB < 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Force diploid: the WC accumulator reads each code as a diploid dosage; the ploidy
    // vector only sizes the view (the kernel ignores it), so a diploid fill is correct.
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view = core::make_decode_tile_view(tile, sample_ploidy, P);

    // Autosome mask for the genome-wide summary (plink2 method=wc summarizes autosomes).
    const std::size_t Mz = static_cast<std::size_t>(M);
    std::vector<std::uint8_t> summary_include(Mz, 0);
    for (std::size_t s = 0; s < Mz && s < snptab.chrom.size(); ++s) {
        const int chr = snptab.chrom[s];
        summary_include[s] =
            (chr >= kAutosomeChromMin && chr <= kAutosomeChromMax) ? std::uint8_t{1}
                                                                   : std::uint8_t{0};
    }

    const FstPerSite fps = be.fst_wc_per_site(
        view, idxA, idxB, std::span<const std::uint8_t>(summary_include));

    res.num = fps.num;
    res.den = fps.den;
    res.fst = fps.fst;
    res.valid = fps.valid;
    res.n_valid = fps.n_valid;
    res.precision_tag = fps.precision_tag;

    // Carry the per-SNP coordinates (kept order == tile SNP order).
    res.chrom.assign(Mz, 0);
    res.physpos.assign(Mz, 0.0);
    res.snp_id.assign(Mz, std::string());
    res.a1.assign(Mz, '?');
    res.a2.assign(Mz, '?');
    for (std::size_t s = 0; s < Mz; ++s) {
        if (s < snptab.chrom.size()) res.chrom[s] = snptab.chrom[s];
        if (s < snptab.physpos.size()) res.physpos[s] = snptab.physpos[s];
        if (s < snptab.id.size()) res.snp_id[s] = snptab.id[s];
        if (s < snptab.ref.size()) res.a1[s] = snptab.ref[s];
        if (s < snptab.alt.size()) res.a2[s] = snptab.alt[s];
    }

    // Genome-wide summary: the ratio-of-averages (plink2 .fst.summary) and the unweighted
    // mean of the per-SNP FST over the same autosomal valid sites.
    res.fst_ratio = (fps.sum_den != 0.0) ? (fps.sum_num / fps.sum_den) : std::nan("");
    if (fps.n_valid > 0) {
        long double acc = 0.0L;
        for (std::size_t s = 0; s < Mz; ++s) {
            if (res.valid[s] && summary_include[s]) acc += static_cast<long double>(res.fst[s]);
        }
        res.fst_mean = static_cast<double>(acc / static_cast<long double>(fps.n_valid));
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
