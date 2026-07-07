// src/core/stats/pca.cpp
//
// run_pca: the standalone genotype-PCA driver. It reuses the shared genotype decode
// front-end (read the triple to the canonical population-contiguous tile), then runs the
// Patterson-2006 standardize -> SYRK covariance -> cuSOLVER eigen -> top-K projection on the
// device (be.pca_covariance_eig). Per-sample identity (Genetic IDs) is resolved in EXACT
// tile individual order so the emitted coordinate rows join to the scikit-allel/sklearn
// oracle by IID.
//
// Diploid is forced (matching the plink2 .raw dosage the oracle bridge feeds scikit-allel):
// each 2-bit code is a diploid dosage 0/1/2, per-SNP freq p is over the non-missing samples,
// missing genotypes are mean-imputed to 0 after centering, and monomorphic/all-missing SNPs
// are excluded. Precision: the covariance SYRK follows the passed precision (emulated-FP64
// default, matmul-heavy); the eigen solve is native FP64 inside the backend.
#include "steppe/pca.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/internal/decode_af.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"
#include "io/genotype_source.hpp"
#include "io/genotype_tile.hpp"
#include "io/individual_partition.hpp"
#include "io/ind_reader.hpp"

namespace steppe {

PcaResult run_pca(const std::string& geno, const std::string& snp, const std::string& ind,
                  std::span<const std::string> pops, int k, const Precision& precision,
                  device::Resources& resources) {
    PcaResult res;
    res.precision_tag = Precision::Kind::Fp64;

    if (k < 1) {
        res.status = Status::InvalidConfig;
        return res;
    }

    ComputeBackend& be = device::primary_backend(resources);

    // Population selection: an explicit set (color groups) or ALL pops (MinN>=1).
    io::PopSelection sel;
    if (!pops.empty()) {
        sel.mode = io::PopSelection::Mode::Explicit;
        sel.labels.assign(pops.begin(), pops.end());
    } else {
        sel.mode = io::PopSelection::Mode::MinN;
        sel.min_n = 1;
    }

    const core::GenotypeFrontEnd fe = core::read_genotype_front_end(geno, snp, ind, sel, be);
    const io::GenotypeTile& tile = fe.tile;

    const int P = static_cast<int>(tile.n_pop());
    const long N = static_cast<long>(tile.n_individuals);
    const long M = static_cast<long>(tile.n_snp);
    if (P < 1 || N <= 0 || M <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Per-record Genetic IDs (a singleton partition labels each .ind/.fam row by its ID in
    // record order), then permute into EXACT tile individual order: the tile is
    // population-contiguous, so tile individual order == concatenating fe.part.groups' rows.
    const io::IndPartition ids =
        io::read_individual_partition(fe.fmt, ind, std::nullopt, SIZE_MAX);
    std::vector<std::string> id_by_record(ids.n_individuals_total);
    for (const io::PopGroup& g : ids.groups) {
        if (!g.rows.empty() && g.rows[0] < id_by_record.size()) {
            id_by_record[g.rows[0]] = g.label;
        }
    }

    res.sample_id.reserve(static_cast<std::size_t>(N));
    res.sample_pop.reserve(static_cast<std::size_t>(N));
    for (const io::PopGroup& grp : fe.part.groups) {
        bool seen = false;
        for (const std::string& pl : res.pop_labels)
            if (pl == grp.label) { seen = true; break; }
        if (!seen) res.pop_labels.push_back(grp.label);
        for (std::size_t row : grp.rows) {
            res.sample_id.push_back(row < id_by_record.size() ? id_by_record[row] : std::string());
            res.sample_pop.push_back(grp.label);
        }
    }

    // Force diploid: the standardize kernel reads each code as a diploid dosage; the ploidy
    // vector only sizes the view (the PCA kernels ignore it), so a diploid fill is correct.
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view = core::make_decode_tile_view(tile, sample_ploidy, P);

    const PcaEig eig = be.pca_covariance_eig(view, k, precision);
    if (eig.status != Status::Ok) {
        res.status = eig.status;
        return res;
    }

    // k must not exceed the usable rank (min(N, n_snp_used)); fail-fast on a degenerate ask.
    if (static_cast<long>(k) > N || static_cast<long>(k) > eig.n_snp_used) {
        res.status = Status::InvalidConfig;
        return res;
    }

    res.coords = eig.coords;
    res.N = eig.N;
    res.K = eig.K;
    res.eigenvalues = eig.eigenvalues;
    res.var_explained = eig.var_explained;
    res.n_snp_total = M;
    res.n_snp_used = eig.n_snp_used;
    res.n_snp_monomorphic = eig.n_snp_monomorphic;
    res.precision_tag = eig.precision_tag;
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
