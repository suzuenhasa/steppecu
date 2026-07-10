// src/core/admixture/admixture.cpp
//
// run_admixture: the host-pure ADMIXTURE Q/F driver (`steppe admixture`). It reads the
// genotype triple through the shared front-end into the canonical individual-major tile,
// resolves per-sample identity in tile order, builds the mode's fixed F (supervised: the
// labeled populations' per-SNP allele-2 frequencies via decode_af; projection: the caller's
// F.tsv), then hands the per-individual dosage decode + block-EM to the backend
// (be.admixture_fit — GPU when present, else the CpuBackend reference oracle) and maps the
// device AdmixtureFit into the public AdmixtureResult. The EM loop lives in native C++
// (inside the backend); this driver is orchestration only, mirroring core/stats/pca.cpp.
#include "steppe/admixture.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/internal/decode_af.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "device/backend.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/individual_partition.hpp"

namespace steppe {

AdmixtureResult run_admixture(const std::string& geno, const std::string& snp,
                              const std::string& ind, const std::vector<std::string>& pops,
                              const AdmixtureParams& params, const AdmixtureInputs& inputs,
                              ComputeBackend& be) {
    AdmixtureResult res;
    res.mode = params.mode;
    res.precision_tag = params.precision.kind;

    // Population selection: an explicit set, or ALL pops (MinN>=1).
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
    res.n_snp_total = M;
    res.n_indiv = N;

    // Per-record Genetic IDs in tile (population-contiguous) order.
    const io::IndPartition ids =
        io::read_individual_partition(fe.fmt, ind, std::nullopt, SIZE_MAX);
    std::vector<std::string> id_by_record(ids.n_individuals_total);
    for (const io::PopGroup& g : ids.groups)
        if (!g.rows.empty() && g.rows[0] < id_by_record.size()) id_by_record[g.rows[0]] = g.label;
    res.sample_id.reserve(static_cast<std::size_t>(N));
    res.sample_pop.reserve(static_cast<std::size_t>(N));
    for (const io::PopGroup& grp : fe.part.groups)
        for (std::size_t row : grp.rows) {
            res.sample_id.push_back(row < id_by_record.size() ? id_by_record[row] : std::string());
            res.sample_pop.push_back(grp.label);
        }

    // Force diploid (the decode reads each 2-bit code as a diploid dosage 0/1/2).
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view = core::make_decode_tile_view(tile, sample_ploidy, P);

    // Resolve K + the fixed-F table per mode.
    int K = params.K;
    std::vector<double> fixedF;  // row-major M x K
    const double* fixedF_ptr = nullptr;

    if (params.mode == AdmixtureMode::Supervised) {
        // F column k = the labeled population's per-SNP allele-2 frequency (decode_af over the
        // tile's pops). This is the fixed-P convex Q solve (critic must-fix #1: gate vs
        // ADMIXTURE's fixed-P/projection oracle, NOT `-supervised`).
        if (inputs.supervised_labels.empty()) {
            res.status = Status::InvalidConfig;
            return res;
        }
        K = static_cast<int>(inputs.supervised_labels.size());
        std::vector<int> pop_of_label(inputs.supervised_labels.size(), -1);
        for (std::size_t k = 0; k < inputs.supervised_labels.size(); ++k)
            for (std::size_t p = 0; p < tile.pop_labels.size(); ++p)
                if (tile.pop_labels[p] == inputs.supervised_labels[k])
                    pop_of_label[k] = static_cast<int>(p);
        for (int pk : pop_of_label)
            if (pk < 0) { res.status = Status::InvalidConfig; return res; }  // label not selected

        const DecodeResult dec = be.decode_af(view);  // q[p + P*s] = pop p allele-2 freq
        fixedF.assign(static_cast<std::size_t>(M) * static_cast<std::size_t>(K), 0.0);
        for (long s = 0; s < M; ++s)
            for (int k = 0; k < K; ++k)
                fixedF[static_cast<std::size_t>(s) * static_cast<std::size_t>(K) +
                       static_cast<std::size_t>(k)] =
                    dec.q[static_cast<std::size_t>(pop_of_label[static_cast<std::size_t>(k)]) +
                          static_cast<std::size_t>(P) * static_cast<std::size_t>(s)];
        fixedF_ptr = fixedF.data();
    } else if (params.mode == AdmixtureMode::Projection) {
        // F is fixed entirely from the caller's reference table. The SNP set must match the
        // retained SNPs (scope Open-Decision-3): hard-fail on a row-count mismatch, and — when
        // per-row SNP ids are provided — on any id that does not match the retained .snp order.
        if (inputs.fixed_F.empty() || inputs.fixed_F_K <= 0) {
            res.status = Status::InvalidConfig;
            return res;
        }
        if (inputs.fixed_F_M != M) {
            res.status = Status::InvalidConfig;  // SNP-count mismatch: the F panel != target panel
            return res;
        }
        if (!inputs.fixed_F_snps.empty()) {
            if (static_cast<long>(inputs.fixed_F_snps.size()) != M) {
                res.status = Status::InvalidConfig;
                return res;
            }
            for (long s = 0; s < M; ++s)
                if (inputs.fixed_F_snps[static_cast<std::size_t>(s)] != fe.snptab.id[static_cast<std::size_t>(s)]) {
                    res.status = Status::InvalidConfig;  // SNP-set mismatch -> would misproject
                    return res;
                }
        }
        K = inputs.fixed_F_K;
        fixedF_ptr = inputs.fixed_F.data();
    }

    if (K < 1 || static_cast<long>(K) >= N) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const AdmixtureFit fit = be.admixture_fit(
        view, K, fixedF_ptr, fixedF_ptr ? M : 0, params.seed, params.seeds, params.max_iter,
        params.tol, static_cast<int>(params.init), params.precision);
    if (fit.status != Status::Ok) {
        res.status = fit.status;
        return res;
    }

    res.Q = fit.Q;
    res.F = fit.F;
    res.N = fit.N;
    res.M = fit.M;
    res.K = fit.K;
    res.seed_loglik = fit.seed_loglik;
    res.seed_iters = fit.seed_iters;
    res.seed_converged = fit.seed_converged;
    res.best_loglik = fit.best_loglik;
    res.best_seed = fit.best_seed;
    res.iters_run = fit.iters_run;
    res.converged = fit.converged;
    res.precision_tag = fit.precision_tag;
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
