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
#include <unordered_set>
#include <vector>

#include "core/internal/decode_af.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/stats/apply_snp_filter.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"
#include "io/bgen_reader.hpp"
#include "io/filter/snp_filter.hpp"
#include "io/genotype_source.hpp"
#include "io/genotype_tile.hpp"
#include "io/individual_partition.hpp"
#include "io/ind_reader.hpp"

namespace steppe {

PcaResult run_pca(const std::string& geno, const std::string& snp, const std::string& ind,
                  std::span<const std::string> pops, int k, const Precision& precision,
                  device::Resources& resources, std::span<const std::string> project_pops,
                  std::span<const std::string> project_samples, PcaProjectMode project_mode,
                  const FilterConfig& filter, PcaSolver pca_solver) {
    PcaResult res;
    res.precision_tag = Precision::Kind::Fp64;

    if (k < 1) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const bool projecting = !project_pops.empty() || !project_samples.empty();
    const std::unordered_set<std::string> proj_pop_set(project_pops.begin(), project_pops.end());
    std::unordered_set<std::string> proj_id_set(project_samples.begin(), project_samples.end());

    // A label cannot be BOTH a selected reference pop and a projected pop.
    if (projecting && !pops.empty()) {
        for (const std::string& pl : pops)
            if (proj_pop_set.count(pl)) { res.status = Status::InvalidConfig; return res; }
    }

    ComputeBackend& be = device::primary_backend(resources);

    // Population selection: an explicit set (color groups) or ALL pops (MinN>=1). A projected
    // pop that is not otherwise selected is auto-added to the decode set so `--project-pops X`
    // alone (default all-pops) still decodes X.
    io::PopSelection sel;
    if (!pops.empty()) {
        sel.mode = io::PopSelection::Mode::Explicit;
        sel.labels.assign(pops.begin(), pops.end());
        for (const std::string& pp : project_pops) {
            bool present = false;
            for (const std::string& l : sel.labels)
                if (l == pp) { present = true; break; }
            if (!present) sel.labels.push_back(pp);
        }
    } else {
        sel.mode = io::PopSelection::Mode::MinN;
        sel.min_n = 1;
    }

    // GPU-native device-resident load; a filtered run compacts the resident tile on-device via
    // apply_snp_filter (no host round-trip). STEPPE_HOST_FILTER=1 forces the legacy host repack
    // oracle. Byte-identical either way.
    const bool allow_device =
        !(io::filter::filter_is_active(filter) && core::host_filter_forced());
    core::GenotypeFrontEnd fe =
        core::read_genotype_front_end(geno, snp, ind, sel, be, allow_device);
    // Per-SNP QC filter: subset the SNP axis in place BEFORE the decode view, so the covariance
    // (hence the eigenvectors, up to per-component sign) is bit-exact vs an externally pre-subset
    // triple. Throws on a same-ascertainment refusal / an all-filtered set.
    const core::SnpFilterOutcome flt =
        core::apply_snp_filter(fe.tile, fe.dev_tile, fe.snptab, filter, be);
    res.kept_snp_ids = flt.kept_ids;
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

    // Walk the tile in individual order (population-contiguous == concatenated group rows),
    // recording per-sample id/pop and — when projecting — the reference/target split. Tile row
    // index r increments exactly as we push, so is_projected[r] joins res.sample_id[r].
    res.sample_id.reserve(static_cast<std::size_t>(N));
    res.sample_pop.reserve(static_cast<std::size_t>(N));
    res.is_projected.assign(static_cast<std::size_t>(N), 0);
    std::vector<int> ref_rows, tgt_rows;
    int r = 0;
    for (const io::PopGroup& grp : fe.part.groups) {
        bool seen = false;
        for (const std::string& pl : res.pop_labels)
            if (pl == grp.label) { seen = true; break; }
        if (!seen) res.pop_labels.push_back(grp.label);
        const bool pop_projected = proj_pop_set.count(grp.label) != 0;
        for (std::size_t row : grp.rows) {
            const std::string id = row < id_by_record.size() ? id_by_record[row] : std::string();
            res.sample_id.push_back(id);
            res.sample_pop.push_back(grp.label);
            bool projected = pop_projected;
            if (!projected) {
                const auto it = proj_id_set.find(id);
                if (it != proj_id_set.end()) { projected = true; proj_id_set.erase(it); }
            } else {
                proj_id_set.erase(id);
            }
            if (projecting) {
                res.is_projected[static_cast<std::size_t>(r)] = projected ? 1 : 0;
                (projected ? tgt_rows : ref_rows).push_back(r);
            }
            ++r;
        }
    }

    // Any --project-samples IID that never matched a decoded record is a config error.
    if (projecting && !proj_id_set.empty()) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Force diploid: the standardize kernel reads each code as a diploid dosage; the ploidy
    // vector only sizes the view (the PCA kernels ignore it), so a diploid fill is correct.
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view =
        fe.dev_tile.valid()
            ? core::make_decode_tile_view(fe.dev_tile, sample_ploidy, P)
            : core::make_decode_tile_view(tile, sample_ploidy, P);

    if (!projecting) {
        // No targets marked: the plain covariance-eig path. The solver selector chooses the
        // exact dense N x N Gram or the matrix-free randomized path (Auto resolves in the
        // backend from N + free VRAM); projection always uses the exact reference basis.
        const PcaEig eig =
            be.pca_covariance_eig(view, k, static_cast<int>(pca_solver), precision);
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
        res.n_ref = eig.N;
        res.eigenvalues = eig.eigenvalues;
        res.var_explained = eig.var_explained;
        res.n_snp_total = M;
        res.n_snp_used = eig.n_snp_used;
        res.n_snp_monomorphic = eig.n_snp_monomorphic;
        res.precision_tag = eig.precision_tag;
        res.status = Status::Ok;
        return res;
    }

    // Projection: reference must be non-empty, targets non-empty, and K <= N_ref.
    const long N_ref = static_cast<long>(ref_rows.size());
    const long N_tgt = static_cast<long>(tgt_rows.size());
    if (N_ref <= 0 || N_tgt <= 0 || static_cast<long>(k) > N_ref) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const PcaProject pj = be.pca_project_lsq(view, k, std::span<const int>(ref_rows),
                                             std::span<const int>(tgt_rows),
                                             static_cast<int>(project_mode), precision);
    if (pj.status != Status::Ok) {
        res.status = pj.status;
        return res;
    }
    if (static_cast<long>(k) > pj.n_snp_used) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const int K = pj.K;
    res.N = static_cast<int>(N);
    res.K = K;
    res.n_ref = pj.N_ref;
    res.eigenvalues = pj.eigenvalues;
    res.var_explained = pj.var_explained;
    res.n_snp_total = M;
    res.n_snp_used = pj.n_snp_used;
    res.n_snp_monomorphic = pj.n_snp_monomorphic;
    res.precision_tag = pj.precision_tag;

    // Stitch the reference (U*S) and target (lsq) coordinate blocks back into tile row order:
    // ref_rows[i] -> pj.coords_ref row i, tgt_rows[j] -> pj.coords_tgt row j.
    res.coords.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(K), 0.0);
    for (std::size_t i = 0; i < ref_rows.size(); ++i) {
        const std::size_t dst = static_cast<std::size_t>(ref_rows[i]) * static_cast<std::size_t>(K);
        const std::size_t src = i * static_cast<std::size_t>(K);
        for (int kk = 0; kk < K; ++kk) res.coords[dst + kk] = pj.coords_ref[src + kk];
    }
    for (std::size_t j = 0; j < tgt_rows.size(); ++j) {
        const std::size_t dst = static_cast<std::size_t>(tgt_rows[j]) * static_cast<std::size_t>(K);
        const std::size_t src = j * static_cast<std::size_t>(K);
        for (int kk = 0; kk < K; ++kk) res.coords[dst + kk] = pj.coords_tgt[src + kk];
    }
    res.status = Status::Ok;
    return res;
}

PcaResult run_pca_bgen(const std::string& bgen_path, int k, const Precision& precision,
                       device::Resources& resources) {
    PcaResult res;
    res.precision_tag = Precision::Kind::Fp64;
    if (k < 1) {
        res.status = Status::InvalidConfig;
        return res;
    }

    ComputeBackend& be = device::primary_backend(resources);

    // Read the BGEN into the real-valued dosage tile (throws on I/O / out-of-scope shape).
    const io::DosageTile tile = io::read_bgen_dosages(bgen_path);
    const long N = static_cast<long>(tile.n_individuals);
    const long M = static_cast<long>(tile.n_snp);
    if (N <= 0 || M <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Per-sample identity from the BGEN sample IDs; a single "ALL" color group for the MVP
    // (a --pops sidecar mapping IDs to populations is a documented follow-on).
    res.sample_id = tile.sample_ids;
    res.sample_pop.assign(static_cast<std::size_t>(N), "ALL");
    res.pop_labels = {"ALL"};
    res.is_projected.assign(static_cast<std::size_t>(N), 0);

    DosageTileView view;
    view.dosage = tile.dosage.data();
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;

    const PcaEig eig = be.pca_covariance_eig_dosage(view, k, precision);
    if (eig.status != Status::Ok) {
        res.status = eig.status;
        return res;
    }
    if (static_cast<long>(k) > N || static_cast<long>(k) > eig.n_snp_used) {
        res.status = Status::InvalidConfig;
        return res;
    }

    res.coords = eig.coords;
    res.N = eig.N;
    res.K = eig.K;
    res.n_ref = eig.N;
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
