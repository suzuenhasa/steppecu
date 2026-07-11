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
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/internal/king_kinship.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/stats/apply_snp_filter.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/read_canonical_tile.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"
#include "io/filter/snp_filter.hpp"
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
    DeviceGenotypeTile dev_tile;      // GPU-native load: valid when the tile is device-resident
    std::vector<std::string> labels;  // Genetic ID per singleton index
    std::vector<std::uint8_t> summary_include;
    long autosomal = 0;
    std::vector<std::string> kept_snp_ids;  // retained ids (--emit-kept-snps); empty if no filter
};

[[nodiscard]] KinshipTile read_kinship_tile(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, ComputeBackend& be,
    const FilterConfig& filter) {
    KinshipTile kt;

    io::GenoReader reader(geno);
    const io::GenoFormat fmt = reader.header().format;
    const std::size_t n_present = reader.records_present();
    const io::IndPartition part =
        io::read_individual_partition(fmt, ind, samples, n_present);
    kt.labels.reserve(part.groups.size());
    for (const io::PopGroup& g : part.groups) kt.labels.push_back(g.label);

    io::SnpTable snptab = io::read_snp_table(fmt, snp, SIZE_MAX);
    const long M0 = static_cast<long>(std::min(reader.header().n_snp, snptab.count));
    if (M0 <= 0) return kt;  // caller treats an empty tile as InvalidConfig

    // Read the whole SNP axis as ONE tile (the readers only support a byte-aligned prefix from
    // 0; the 1240K panel per individual is small and bounded). GPU-native device-resident load
    // when no SNP filter is active (an active filter subsets the HOST tile in place); the host
    // `tile` then carries only the descriptor (empty packed). Byte-identical decode either way.
    const bool allow_device = !io::filter::filter_is_active(filter);
    if (allow_device && core::device_load_enabled() && be.capabilities().device_count > 0) {
        kt.dev_tile = core::read_canonical_tile_device(reader, part, be, 0,
                                                       static_cast<std::size_t>(M0));
        if (kt.dev_tile.valid()) {
            kt.tile.packed.clear();
            kt.tile.bytes_per_record = kt.dev_tile.bytes_per_record;
            kt.tile.n_snp = kt.dev_tile.n_snp;
            kt.tile.n_individuals = kt.dev_tile.n_individuals;
            kt.tile.pop_offsets = kt.dev_tile.pop_offsets;
            kt.tile.pop_labels = kt.dev_tile.pop_labels;
        }
    }
    if (!kt.dev_tile.valid()) {
        kt.tile = core::read_canonical_tile(reader, part, be, 0, static_cast<std::size_t>(M0));
    }

    // Per-SNP QC filter: subset the SNP axis (individuals untouched) in lockstep with the SnpTable
    // BEFORE the autosome mask + KING sweep, so a filtered run's integer counts (nsnp/hethet/ibs0)
    // and phi are bit-exact vs an externally pre-subset triple. Throws on a same-ascertainment
    // refusal / an all-filtered set.
    const core::SnpFilterOutcome flt = core::apply_snp_filter(kt.tile, snptab, filter, be);
    kt.kept_snp_ids = flt.kept_ids;

    // FST-style FULL interspersed autosome mask (indexed by global SNP), NOT readv2's
    // contiguous-prefix restriction — so chr23/24 exclusion is clean even when interspersed.
    // Rebuilt from the SUBSET SnpTable so the mask layers cleanly UNDER the filter.
    const std::size_t Mz = kt.tile.n_snp;
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

// Expand the STREAMED survivor SoA into the KinshipResult. The device already applied the emit
// predicate (phi >= min_kinship / phi > cutoff) and the phi it carries is the SHARED king_phi over
// the same integer counts the dense path folds -> this produces BYTE-IDENTICAL rows to `finalize`
// on the same above-threshold set (same id1<=id2 canonicalization, same degree band, same order:
// cub::DeviceSelect is stable and blocks run in ascending rank order).
void finalize_streamed(const KingStreamResult& sr,
                       const std::vector<std::string>& labels, KinshipResult& res) {
    const std::size_t n = sr.i.size();
    res.emitted = 0;
    for (std::size_t r = 0; r < n; ++r) {
        const int i = sr.i[r];
        const int j = sr.j[r];
        const double phi = sr.phi[r];
        const std::string& li = labels[static_cast<std::size_t>(i)];
        const std::string& lj = labels[static_cast<std::size_t>(j)];
        const bool i_first = (li <= lj);
        res.id1.push_back(i_first ? li : lj);
        res.id2.push_back(i_first ? lj : li);
        res.nsnp.push_back(sr.nsnp[r]);
        res.hethet.push_back(sr.hethet[r]);
        res.ibs0.push_back(sr.ibs0[r]);
        res.phi.push_back(phi);
        res.degree.emplace_back(core::king_degree_label(core::degree_from_phi(phi)));
        ++res.emitted;
    }
}

// plink2's --king-cutoff edge nudge (plink2_matrix_calc.cc KingCutoffBatchTable:
// `king_cutoff *= 1.0 + kSmallEpsilon`, kSmallEpsilon = k2m44 = 2^-44). An edge is kept iff
// phi > cutoff * (1 + 2^-44). Reproducing it makes steppe's edge set identical to plink2's even at
// the exact boundary, and keeps the .kin0 steppe writes consistent with plink2 --king-cutoff-table.
constexpr double kPlink2KinshipEpsilon = 0x1p-44;

// Greedy relatedness prune reproducing plink2's KinshipPruneDestructive
// (plink2_matrix_calc.cc) EXACTLY, so steppe's retained set matches `plink2 --king-cutoff` given
// the same edge set and sample-index order. The rule, each step while any edge remains:
//   (a) if any vertex has degree == 1, remove the PARTNER of the LOWEST-index such vertex;
//   (b) otherwise remove the LOWEST-index vertex of MAXIMUM degree (strict-greater scan -> the
//       first/lowest index wins a degree tie).
// Removing a vertex detaches it from every still-present neighbour (decrementing their degrees).
// `edge_i`/`edge_j` are the above-cutoff pairs (singleton indices); returns a per-sample removed
// flag (true = pruned to the .out set). Sample index == the .ind singleton order (== a matching
// plink2 roster order), which fixes the tie-break identically on both sides.
[[nodiscard]] std::vector<char> king_greedy_prune(int N, const std::vector<int>& edge_i,
                                                  const std::vector<int>& edge_j) {
    std::vector<std::unordered_set<int>> adj(static_cast<std::size_t>(N));
    const std::size_t ne = edge_i.size();
    for (std::size_t e = 0; e < ne; ++e) {
        const int a = edge_i[e];
        const int b = edge_j[e];
        if (a == b) continue;
        adj[static_cast<std::size_t>(a)].insert(b);
        adj[static_cast<std::size_t>(b)].insert(a);
    }
    std::vector<int> degree(static_cast<std::size_t>(N), 0);
    std::vector<char> present(static_cast<std::size_t>(N), 0);  // in the nonzero-degree graph
    std::vector<int> related;                                   // ascending-index nz vertices
    for (int v = 0; v < N; ++v) {
        const int d = static_cast<int>(adj[static_cast<std::size_t>(v)].size());
        degree[static_cast<std::size_t>(v)] = d;
        if (d > 0) {
            present[static_cast<std::size_t>(v)] = 1;
            related.push_back(v);
        }
    }
    std::vector<char> removed(static_cast<std::size_t>(N), 0);

    while (true) {
        int prune = -1;
        // (a) lowest-index present vertex with degree exactly 1 -> remove its single partner.
        int d1 = -1;
        for (const int v : related) {
            if (present[static_cast<std::size_t>(v)] && degree[static_cast<std::size_t>(v)] == 1) {
                d1 = v;
                break;
            }
        }
        if (d1 >= 0) {
            prune = *adj[static_cast<std::size_t>(d1)].begin();  // its one remaining neighbour
        } else {
            // (b) lowest-index present vertex of maximum degree (strict-greater -> first wins ties).
            int best = -1;
            int best_deg = 0;
            for (const int v : related) {
                if (!present[static_cast<std::size_t>(v)]) continue;
                const int d = degree[static_cast<std::size_t>(v)];
                if (d < 1) continue;
                if (best < 0 || d > best_deg) {
                    best = v;
                    best_deg = d;
                }
            }
            if (best < 0) break;  // no edges remain
            prune = best;
        }
        // Detach prune from every still-present neighbour, then remove it.
        const std::vector<int> nbrs(adj[static_cast<std::size_t>(prune)].begin(),
                                    adj[static_cast<std::size_t>(prune)].end());
        for (const int y : nbrs) {
            if (!present[static_cast<std::size_t>(y)]) continue;
            adj[static_cast<std::size_t>(y)].erase(prune);
            const int nd = static_cast<int>(adj[static_cast<std::size_t>(y)].size());
            degree[static_cast<std::size_t>(y)] = nd;
            if (nd == 0) present[static_cast<std::size_t>(y)] = 0;
        }
        adj[static_cast<std::size_t>(prune)].clear();
        degree[static_cast<std::size_t>(prune)] = 0;
        present[static_cast<std::size_t>(prune)] = 0;
        removed[static_cast<std::size_t>(prune)] = 1;
    }
    return removed;
}

}  // namespace

KinshipResult run_kinship_all_pairs(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, double min_kinship, bool sure,
    device::Resources& resources, const FilterConfig& filter) {
    KinshipResult res;
    res.precision_tag = Precision::Kind::Fp64;

    ComputeBackend& be = device::primary_backend(resources);
    const KinshipTile kt = read_kinship_tile(geno, snp, ind, samples, be, filter);
    const int N = static_cast<int>(kt.labels.size());
    res.N = N;
    res.autosomal_snps = kt.autosomal;
    res.kept_snp_ids = kt.kept_snp_ids;
    if (N < 2 || kt.tile.n_snp == 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const std::vector<int> sample_ploidy(kt.tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view =
        kt.dev_tile.valid() ? core::make_decode_tile_view(kt.dev_tile, sample_ploidy, N)
                            : core::make_decode_tile_view(kt.tile, sample_ploidy, N);

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
    device::Resources& resources, const FilterConfig& filter) {
    KinshipResult res;
    res.precision_tag = Precision::Kind::Fp64;

    ComputeBackend& be = device::primary_backend(resources);
    const KinshipTile kt = read_kinship_tile(geno, snp, ind, samples, be, filter);
    const int N = static_cast<int>(kt.labels.size());
    res.N = N;
    res.autosomal_snps = kt.autosomal;
    res.kept_snp_ids = kt.kept_snp_ids;
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
    const DecodeTileView view =
        kt.dev_tile.valid() ? core::make_decode_tile_view(kt.dev_tile, sample_ploidy, N)
                            : core::make_decode_tile_view(kt.tile, sample_ploidy, N);

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

KinshipResult run_kinship_streamed(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, double min_kinship,
    device::Resources& resources, const FilterConfig& filter) {
    KinshipResult res;
    res.precision_tag = Precision::Kind::Fp64;

    ComputeBackend& be = device::primary_backend(resources);
    const KinshipTile kt = read_kinship_tile(geno, snp, ind, samples, be, filter);
    const int N = static_cast<int>(kt.labels.size());
    res.N = N;
    res.autosomal_snps = kt.autosomal;
    res.kept_snp_ids = kt.kept_snp_ids;
    if (N < 2 || kt.tile.n_snp == 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const std::vector<int> sample_ploidy(kt.tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view =
        kt.dev_tile.valid() ? core::make_decode_tile_view(kt.dev_tile, sample_ploidy, N)
                            : core::make_decode_tile_view(kt.tile, sample_ploidy, N);

    // Streamed/compacted all-pairs: emit phi >= min_kinship ON-DEVICE (strict_greater=false), so
    // the 5*C(N,2) accumulator (and its maxcomb cap) is never allocated. Bit-identical to the dense
    // path filtered to phi >= min_kinship.
    const KingStreamResult sr = be.king_robust_filtered(
        view, std::span<const std::uint8_t>(kt.summary_include), min_kinship, /*strict=*/false);
    res.enumerated = sr.enumerated;
    res.precision_tag = sr.precision_tag;
    res.status = sr.status;
    if (sr.status != Status::Ok) return res;

    finalize_streamed(sr, kt.labels, res);
    res.status = Status::Ok;
    return res;
}

KinshipCutoffResult run_kinship_cutoff(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, double cutoff,
    device::Resources& resources, const FilterConfig& filter) {
    KinshipCutoffResult res;
    res.precision_tag = Precision::Kind::Fp64;
    res.cutoff = cutoff;

    ComputeBackend& be = device::primary_backend(resources);
    const KinshipTile kt = read_kinship_tile(geno, snp, ind, samples, be, filter);
    const int N = static_cast<int>(kt.labels.size());
    res.N = N;
    res.autosomal_snps = kt.autosomal;
    res.kept_snp_ids = kt.kept_snp_ids;
    if (N < 2 || kt.tile.n_snp == 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const std::vector<int> sample_ploidy(kt.tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view =
        kt.dev_tile.valid() ? core::make_decode_tile_view(kt.dev_tile, sample_ploidy, N)
                            : core::make_decode_tile_view(kt.tile, sample_ploidy, N);

    // The above-cutoff relatedness graph: phi > cutoff*(1 + 2^-44) exactly reproduces plink2's
    // nudged edge rule, so this survivor set == plink2 --king-cutoff's edge set (and a .kin0 built
    // from it is consistent under plink2 --king-cutoff-table with the same raw cutoff).
    const double edge_threshold = cutoff * (1.0 + kPlink2KinshipEpsilon);
    const KingStreamResult sr = be.king_robust_filtered(
        view, std::span<const std::uint8_t>(kt.summary_include), edge_threshold, /*strict=*/true);
    res.enumerated = sr.enumerated;
    res.precision_tag = sr.precision_tag;
    res.status = sr.status;
    if (sr.status != Status::Ok) return res;

    // Surface the edges (for the sparse .kin0) in survivor order.
    const std::size_t ne = sr.i.size();
    res.n_edges = ne;
    res.edge_id1.reserve(ne);
    res.edge_id2.reserve(ne);
    res.edge_nsnp.reserve(ne);
    res.edge_hethet.reserve(ne);
    res.edge_ibs0.reserve(ne);
    res.edge_phi.reserve(ne);
    for (std::size_t e = 0; e < ne; ++e) {
        res.edge_id1.push_back(kt.labels[static_cast<std::size_t>(sr.i[e])]);
        res.edge_id2.push_back(kt.labels[static_cast<std::size_t>(sr.j[e])]);
        res.edge_nsnp.push_back(sr.nsnp[e]);
        res.edge_hethet.push_back(sr.hethet[e]);
        res.edge_ibs0.push_back(sr.ibs0[e]);
        res.edge_phi.push_back(sr.phi[e]);
    }

    // Greedy prune (plink2 KinshipPruneDestructive) -> retained / removed, in singleton-index order.
    const std::vector<char> removed = king_greedy_prune(N, sr.i, sr.j);
    for (int v = 0; v < N; ++v) {
        if (removed[static_cast<std::size_t>(v)]) {
            res.removed.push_back(kt.labels[static_cast<std::size_t>(v)]);
        } else {
            res.retained.push_back(kt.labels[static_cast<std::size_t>(v)]);
        }
    }
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
