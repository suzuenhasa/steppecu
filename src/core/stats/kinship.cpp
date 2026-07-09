// src/core/stats/kinship.cpp — run_kinship_*, the KING-robust kinship drivers.
//
// Host-pure and CUDA-free: it reaches the GPU only through the ComputeBackend seam
// (primary_backend), exactly like readv2.cpp / fst.cpp. It resolves each selected Genetic ID
// to its own singleton index (the readv2 per-individual partition), reads the whole SNP axis
// to the canonical diploid tile ONCE, builds the FST-style full interspersed autosome mask,
// runs the GPU pair sweep (all-pairs C(N,2) or an explicit pair list), and expands the five
// per-pair integer counts into phi + the KING degree band on the host.
//
// Precision: INTEGER counting on device + a SINGLE native-FP64 ratio (phi) per pair on the
// host — never emulated-FP64 (gated vs plink2 --make-king-table, NOT ADMIXTOOLS2).
#include "steppe/kinship.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/internal/king_kinship.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/read_canonical_tile.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/genotype_tile.hpp"
#include "io/individual_partition.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"

namespace steppe {

namespace {

// Host mirror of readv2_unrank_pair: flat rank r = C(j,2)+i (i<j) -> (i, j).
void unrank_pair_host(long long r, int& i, int& j) {
    const double rd = static_cast<double>(r);
    long long jj = static_cast<long long>((1.0 + std::sqrt(1.0 + 8.0 * rd)) * 0.5);
    while (jj * (jj - 1) / 2 > r) --jj;
    while ((jj + 1) * jj / 2 <= r) ++jj;
    j = static_cast<int>(jj);
    i = static_cast<int>(r - jj * (jj - 1) / 2);
}

// Read the individual partition + build the canonical diploid tile once. Throws on IO errors.
struct KinshipTile {
    io::GenotypeTile tile;
    std::vector<std::string> labels;  // Genetic ID per singleton index
    std::vector<std::uint8_t> summary_include;
    long autosomal = 0;
};

[[nodiscard]] KinshipTile read_kinship_tile(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, ComputeBackend& be) {
    KinshipTile kt;

    io::GenoReader reader(geno);
    const io::GenoFormat fmt = reader.header().format;
    const std::size_t n_present = reader.records_present();
    const io::IndPartition part =
        io::read_individual_partition(fmt, ind, samples, n_present);
    kt.labels.reserve(part.groups.size());
    for (const io::PopGroup& g : part.groups) kt.labels.push_back(g.label);

    const io::SnpTable snptab = io::read_snp_table(fmt, snp, SIZE_MAX);
    const long M0 = static_cast<long>(std::min(reader.header().n_snp, snptab.count));
    if (M0 <= 0) return kt;  // caller treats an empty tile as InvalidConfig

    // Read the whole SNP axis as ONE tile (the readers only support a byte-aligned prefix from
    // 0; the 1240K panel per individual is small and bounded).
    kt.tile = core::read_canonical_tile(reader, part, be, 0, static_cast<std::size_t>(M0));

    // FST-style FULL interspersed autosome mask (indexed by global SNP), NOT readv2's
    // contiguous-prefix restriction — so chr23/24 exclusion is clean even when interspersed.
    const std::size_t Mz = static_cast<std::size_t>(M0);
    kt.summary_include.assign(Mz, 0);
    long auto_n = 0;
    for (std::size_t s = 0; s < Mz && s < snptab.chrom.size(); ++s) {
        const int chr = snptab.chrom[s];
        if (chr >= kAutosomeChromMin && chr <= kAutosomeChromMax) {
            kt.summary_include[s] = 1;
            ++auto_n;
        }
    }
    kt.autosomal = auto_n;
    return kt;
}

// Expand the device count vectors into the KinshipResult, applying the --min-kinship filter
// and canonicalizing id1 <= id2. `pair_ij(r, i, j)` yields the singleton indices for row r.
template <typename PairIndex>
void finalize(const KingMatrix& mat, const std::vector<std::string>& labels,
              double min_kinship, PairIndex&& pair_ij, KinshipResult& res) {
    const std::size_t np = mat.nsnp.size();
    const bool keep_all = (min_kinship == -std::numeric_limits<double>::infinity());
    res.emitted = 0;
    for (std::size_t r = 0; r < np; ++r) {
        int i = 0, j = 0;
        pair_ij(r, i, j);
        core::KingCounts c;
        c.nsnp = mat.nsnp[r];
        c.hethet = mat.hethet[r];
        c.ibs0 = mat.ibs0[r];
        c.het_i = mat.het_i[r];
        c.het_j = mat.het_j[r];
        const double phi = core::king_phi(c);
        const bool keep = keep_all ? true : (phi >= min_kinship);
        if (!keep) continue;

        const std::string& li = labels[static_cast<std::size_t>(i)];
        const std::string& lj = labels[static_cast<std::size_t>(j)];
        const bool i_first = (li <= lj);
        res.id1.push_back(i_first ? li : lj);
        res.id2.push_back(i_first ? lj : li);
        res.nsnp.push_back(c.nsnp);
        res.hethet.push_back(c.hethet);
        res.ibs0.push_back(c.ibs0);
        res.phi.push_back(phi);
        res.degree.emplace_back(core::king_degree_label(core::degree_from_phi(phi)));
        ++res.emitted;
    }
}

}  // namespace

KinshipResult run_kinship_all_pairs(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, double min_kinship, bool sure,
    device::Resources& resources) {
    KinshipResult res;
    res.precision_tag = Precision::Kind::Fp64;

    ComputeBackend& be = device::primary_backend(resources);
    const KinshipTile kt = read_kinship_tile(geno, snp, ind, samples, be);
    const int N = static_cast<int>(kt.labels.size());
    res.N = N;
    res.autosomal_snps = kt.autosomal;
    if (N < 2 || kt.tile.n_snp == 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const std::vector<int> sample_ploidy(kt.tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view = core::make_decode_tile_view(kt.tile, sample_ploidy, N);

    const KingMatrix mat = be.king_robust_all_pairs(
        view, std::span<const std::uint8_t>(kt.summary_include), std::span<const int>(),
        std::span<const int>(), sure);
    res.enumerated = mat.enumerated;
    res.capped = mat.capped;
    res.precision_tag = mat.precision_tag;
    res.status = mat.status;
    if (mat.status != Status::Ok) return res;

    finalize(mat, kt.labels, min_kinship,
             [](std::size_t r, int& i, int& j) {
                 unrank_pair_host(static_cast<long long>(r), i, j);
             },
             res);
    res.status = Status::Ok;
    return res;
}

KinshipResult run_kinship_pairs(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples,
    const std::vector<std::pair<std::string, std::string>>& pairs, double min_kinship,
    device::Resources& resources) {
    KinshipResult res;
    res.precision_tag = Precision::Kind::Fp64;

    ComputeBackend& be = device::primary_backend(resources);
    const KinshipTile kt = read_kinship_tile(geno, snp, ind, samples, be);
    const int N = static_cast<int>(kt.labels.size());
    res.N = N;
    res.autosomal_snps = kt.autosomal;
    if (N < 1 || kt.tile.n_snp == 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Resolve each Genetic ID to its singleton index (fail-fast on an unknown / self pair).
    std::unordered_map<std::string, int> id_to_idx;
    id_to_idx.reserve(kt.labels.size() * 2);
    for (int g = 0; g < N; ++g) id_to_idx.emplace(kt.labels[static_cast<std::size_t>(g)], g);

    std::vector<int> pi, pj;
    pi.reserve(pairs.size());
    pj.reserve(pairs.size());
    for (const auto& [a, b] : pairs) {
        const auto ia = id_to_idx.find(a);
        const auto ib = id_to_idx.find(b);
        if (ia == id_to_idx.end() || ib == id_to_idx.end()) {
            throw std::invalid_argument(
                "steppe kinship: --pairs Genetic ID '" + a + "'/'" + b +
                "' not in the selected individual set");
        }
        if (ia->second == ib->second) {
            throw std::invalid_argument(
                "steppe kinship: --pairs names a self-pair ('" + a + "' with itself)");
        }
        pi.push_back(ia->second);
        pj.push_back(ib->second);
    }
    if (pi.empty()) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const std::vector<int> sample_ploidy(kt.tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view = core::make_decode_tile_view(kt.tile, sample_ploidy, N);

    const KingMatrix mat = be.king_robust_all_pairs(
        view, std::span<const std::uint8_t>(kt.summary_include), std::span<const int>(pi),
        std::span<const int>(pj), false);
    res.enumerated = mat.enumerated;
    res.capped = mat.capped;
    res.precision_tag = mat.precision_tag;
    res.status = mat.status;
    if (mat.status != Status::Ok) return res;

    finalize(mat, kt.labels, min_kinship,
             [&pi, &pj](std::size_t r, int& i, int& j) {
                 i = pi[r];
                 j = pj[r];
             },
             res);
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
