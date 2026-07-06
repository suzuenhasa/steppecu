// src/core/stats/qpfstats.cpp — run_qpfstats, the joint f2/f3/f4 smoother (genotype path).
//
// Reuses the D-statistic genotype front-end and numerator engine unchanged, then jointly
// fits every f2/f3/f4 against their linear identities in one fused, device-resident backend
// call and scatters out the smoothed per-block f2 tensor the downstream tools consume.
//
// Reference: docs/reference/src_core_stats_qpfstats.cpp.md

#include "steppe/qpfstats.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/internal/index_cast.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe {

using core::idx;

namespace {

// Named constants — reference §2
inline constexpr int kPloidyDiploid = 2;
inline constexpr std::size_t kPrimaryGpu = 0;
inline constexpr double kRidge = 1e-5;

// The population-pair index — reference §3
[[nodiscard]] inline int pair_index(int i, int j, int npop) {
    if (i > j) std::swap(i, j);
    return i * npop - (i * (i + 1)) / 2 + (j - i - 1);
}

// The f2/f3/f4 population combination — reference §4
struct PopComb { int p1, p2, p3, p4; };

// Population combinations + regression design matrix — reference §4-5
void build_popcomb_and_design(int npop, std::vector<PopComb>& combs,
                              std::vector<double>& x, int& npopcomb, int& npairs) {
    combs.clear();
    npairs = npop * (npop - 1) / 2;

    for (int i = 0; i < npop; ++i)
        for (int j = i + 1; j < npop; ++j)
            combs.push_back({i, j, i, j});

    {
        std::vector<std::array<int, 3>> tri;
        for (int i = 0; i < npop; ++i)
            for (int j = i + 1; j < npop; ++j)
                for (int k = j + 1; k < npop; ++k)
                    tri.push_back({i, j, k});
        const int rot[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
        for (int r = 0; r < 3; ++r) {
            for (const std::array<int, 3>& t : tri) {
                const int P1 = t[rot[r][0]], P2 = t[rot[r][1]], P3 = t[rot[r][2]];
                combs.push_back({P1, P2, P1, P3});
            }
        }
    }

    {
        std::vector<std::array<int, 4>> quad;
        for (int i = 0; i < npop; ++i)
            for (int j = i + 1; j < npop; ++j)
                for (int k = j + 1; k < npop; ++k)
                    for (int l = k + 1; l < npop; ++l)
                        quad.push_back({i, j, k, l});
        const int rot[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
        for (const std::array<int, 4>& q : quad)
            for (int r = 0; r < 3; ++r)
                combs.push_back({q[rot[r][0]], q[rot[r][1]], q[rot[r][2]], q[rot[r][3]]});
    }

    if (combs.size() > idx(INT_MAX))
        throw std::runtime_error("qpfstats: npopcomb overflows int (pop set too large)");
    npopcomb = static_cast<int>(combs.size());

    x.assign(idx(npopcomb) * idx(npairs), 0.0);
    const auto set = [&](int c, int i, int j, double v) {
        if (i == j) return;
        const int p = pair_index(i, j, npop);
        x[idx(c) + idx(npopcomb) *
                                            idx(p)] = v;
    };
    for (int c = 0; c < npopcomb; ++c) {
        const PopComb& pc = combs[idx(c)];
        set(c, pc.p1, pc.p4, 1.0);
        set(c, pc.p2, pc.p3, 1.0);
        set(c, pc.p1, pc.p3, -1.0);
        set(c, pc.p2, pc.p4, -1.0);
        if (pc.p1 == pc.p3 && pc.p2 == pc.p4) {
            for (int p = 0; p < npairs; ++p)
                x[idx(c) + idx(npopcomb) *
                                                    idx(p)] *= 2.0;
        }
    }
    for (double& v : x) v *= 0.5;
}

}  // namespace

// run_qpfstats: the pipeline — reference §6
QpfstatsResult run_qpfstats(const std::string& geno, const std::string& snp,
                            const std::string& ind, std::span<const std::string> pops,
                            double blgsize_morgans, const Precision& precision,
                            device::Resources& resources) {
    QpfstatsResult res;
    res.precision_tag = precision.kind;

    std::vector<std::string> sp(pops.begin(), pops.end());
    std::sort(sp.begin(), sp.end());
    sp.erase(std::unique(sp.begin(), sp.end()), sp.end());
    const int npop = static_cast<int>(sp.size());
    res.pop_labels = sp;
    if (npop < 4) {
        res.status = Status::InvalidConfig;
        return res;
    }

    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
    const core::GenotypeFrontEnd fe = core::read_genotype_front_end(geno, snp, ind, sp, be);
    const io::SnpTable& snptab = fe.snptab;
    const io::GenotypeTile& tile = fe.tile;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P != npop) {
        res.status = Status::InvalidConfig;
        return res;
    }
    if (P <= 0 || M <= 0) { res.status = Status::Ok; return res; }

    std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);
    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.data();
    view.ploidy = kPloidyDiploid;

    const bool resident = (be.capabilities().device_count > 0);
    steppe::device::DeviceDecodeResult ddr;
    std::vector<double> Qk, Vk;
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    std::vector<double> physpos_kept;
    if (resident) {
        ddr = be.decode_af_compact_autosome(
            view, std::span<const int>(snptab.chrom.data(), idx(M)),
            std::span<const double>(snptab.genpos_morgans.data(), idx(M)),
            std::span<const double>(snptab.physpos.data(), idx(M)),
            kAutosomeChromMin, kAutosomeChromMax);
        chrom_kept = ddr.chrom_kept;
        genpos_kept = ddr.genpos_kept;
        physpos_kept = ddr.physpos_kept;
    } else {
        const DecodeResult dec = be.decode_af(view);
        Qk.reserve(idx(P) * idx(M));
        Vk.reserve(idx(P) * idx(M));
        chrom_kept.reserve(idx(M));
        genpos_kept.reserve(idx(M));
        physpos_kept.reserve(idx(M));
        for (long s = 0; s < M; ++s) {
            const int chr = snptab.chrom[idx(s)];
            if (chr < kAutosomeChromMin || chr > kAutosomeChromMax) continue;
            const std::size_t src = idx(P) * idx(s);
            for (int p = 0; p < P; ++p) {
                Qk.push_back(dec.q[src + idx(p)]);
                Vk.push_back(dec.v[src + idx(p)]);
            }
            chrom_kept.push_back(chr);
            genpos_kept.push_back(snptab.genpos_morgans[idx(s)]);
            physpos_kept.push_back(snptab.physpos[idx(s)]);
        }
    }
    const long M_kept = static_cast<long>(chrom_kept.size());
    if (M_kept <= 0) { res.status = Status::Ok; return res; }

    const core::BlockPartition partition = core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans, std::span<const double>(physpos_kept));
    const int n_block = partition.n_block;
    if (n_block <= 0) { res.status = Status::Ok; return res; }

    const std::vector<core::BlockRange> ranges = core::block_ranges(
        std::span<const int>(partition.block_id), M_kept, n_block);
    std::vector<int> block_lengths(idx(n_block), 0);
    for (int b = 0; b < n_block; ++b)
        block_lengths[idx(b)] =
            static_cast<int>(ranges[idx(b)].size());

    std::vector<PopComb> combs;
    std::vector<double> x;
    int npopcomb = 0, npairs = 0;
    build_popcomb_and_design(npop, combs, x, npopcomb, npairs);

    std::vector<int> flat;
    flat.reserve(idx(npopcomb) * 4);
    for (const PopComb& pc : combs) {
        flat.push_back(pc.p1); flat.push_back(pc.p2); flat.push_back(pc.p3); flat.push_back(pc.p4);
    }

    const QpfstatsSmooth sm = resident
        ? be.qpfstats_blocks_smooth(
              ddr, partition.block_id.data(), n_block,
              std::span<const int>(flat), std::span<const double>(x), npopcomb, npairs,
              std::span<const int>(block_lengths), kRidge, precision)
        : be.qpfstats_blocks_smooth(
              Qk.data(), Vk.data(), P, M_kept, partition.block_id.data(), n_block,
              std::span<const int>(flat), std::span<const double>(x), npopcomb, npairs,
              std::span<const int>(block_lengths), kRidge, precision);
    if (sm.status != Status::Ok) { res.status = sm.status; return res; }

    F2BlockTensor& T = res.f2;
    T.P = npop;
    T.n_block = n_block;
    const std::size_t slab = idx(npop) * idx(npop);
    T.f2.assign(slab * idx(n_block), 0.0);
    T.vpair.assign(slab * idx(n_block), 0.0);
    T.block_sizes.assign(idx(n_block), 0);
    const std::size_t np = idx(npairs);
    for (int b = 0; b < n_block; ++b) {
        T.block_sizes[idx(b)] =
            block_lengths[idx(b)];
        const std::size_t boff = slab * idx(b);
        for (int i = 0; i < npop; ++i) {
            for (int j = i + 1; j < npop; ++j) {
                const int p = pair_index(i, j, npop);
                const double v = sm.b[idx(p) +
                                      np * idx(b)] +
                                 sm.recenter_shift[idx(p)];
                T.f2[boff + idx(i) +
                     idx(npop) * idx(j)] = v;
                T.f2[boff + idx(j) +
                     idx(npop) * idx(i)] = v;
            }
        }
    }

    for (int b = 0; b < n_block; ++b) {
        const double bs = static_cast<double>(T.block_sizes[idx(b)]);
        const std::size_t boff = slab * idx(b);
        for (int i = 0; i < npop; ++i)
            for (int j = 0; j < npop; ++j)
                if (i != j)
                    T.vpair[boff + idx(i) +
                            idx(npop) * idx(j)] = bs;
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
