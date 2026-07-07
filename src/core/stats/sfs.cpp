// src/core/stats/sfs.cpp
//
// run_sfs: the standalone 2D joint site-frequency spectrum computed straight from genotype
// files. It reuses the shared genotype decode front-end (read the triple to the canonical
// population-contiguous tile) and then runs a GPU joint-histogram over the per-pop A1-copy
// counts — the SAME per-pop allele-count fold the FST path uses, fed into a joint grid
// instead of the WC variance algebra. A pure integer-count stat.
//
// Diploid is forced (matching the FST path + the scikit-allel oracle): each 2-bit code is
// a diploid dosage 0/1/2; missing genotypes make a site incomplete and it is dropped
// (complete-data restriction, §4). NO autosome filter — the histogram covers all kept
// SNPs, so the SNP set is exactly what the gate bridge feeds scikit-allel (the shared
// 2-pop subset defines the site set).
#include "steppe/sfs.hpp"

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

namespace steppe {

SfsResult run_sfs(const std::string& geno, const std::string& snp, const std::string& ind,
                  const std::string& popA, const std::string& popB, bool folded,
                  device::Resources& resources) {
    SfsResult res;
    res.precision_tag = Precision::Kind::Fp64;
    res.popA = popA;
    res.popB = popB;
    res.folded = folded;

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

    // Force diploid: the joint-histogram fold reads each code as a diploid dosage; the
    // ploidy vector only sizes the view (the kernel ignores it), so a diploid fill is right.
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view = core::make_decode_tile_view(tile, sample_ploidy, P);

    const SfsJoint sj = be.joint_sfs_2pop(view, idxA, idxB, folded);

    res.grid = sj.grid;
    res.extA = sj.extA;
    res.extB = sj.extB;
    res.NA = sj.NA;
    res.NB = sj.NB;
    res.n_total = sj.n_total;
    res.n_complete = sj.n_complete;
    res.n_dropped_incomplete = sj.n_dropped_incomplete;
    res.folded = sj.folded;
    res.precision_tag = sj.precision_tag;
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
