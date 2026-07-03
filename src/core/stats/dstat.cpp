// src/core/stats/dstat.cpp
//
// run_dstat: the genotype-path normalized-D statistic (qpDstat, the ABBA-BABA
// test) computed straight from genotype files. It reuses the f4-ratio decode
// front-end and the shared on-device ratio block-jackknife; only the per-SNP D
// reduction is its own.
//
// Reference: docs/reference/src_core_stats_dstat.cpp.md

#include "steppe/dstat.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
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

namespace {

// Named constants — reference §4
inline constexpr int kPloidyDiploid = 2;
inline constexpr std::size_t kPrimaryGpu = 0;

// Degenerate-outcome NaN fill — reference §9
void fill_nan(DstatResult& res, int row_count) {
    const auto count = static_cast<std::size_t>(row_count);
    res.est.assign(count, std::nan(""));
    res.se.assign(count, std::nan(""));
    res.z.assign(count, std::nan(""));
    res.p.assign(count, std::nan(""));
    res.status = Status::Ok;
}

}  // namespace

// Public entry: run_dstat — reference §6
DstatResult run_dstat(const std::string& geno, const std::string& snp, const std::string& ind,
                      std::span<const std::string> pop_union,
                      std::span<const std::array<int, 4>> quadruples, double blgsize_morgans,
                      device::Resources& resources) {
    DstatResult res;
    res.precision_tag = Precision::Kind::Fp64;

    const int N = static_cast<int>(quadruples.size());
    if (N <= 0) { res.status = Status::Ok; return res; }

    res.p1.reserve(static_cast<std::size_t>(N));
    res.p2.reserve(static_cast<std::size_t>(N));
    res.p3.reserve(static_cast<std::size_t>(N));
    res.p4.reserve(static_cast<std::size_t>(N));
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(N) * 4);
    for (const std::array<int, 4>& q : quadruples) {
        res.p1.push_back(q[0]); res.p2.push_back(q[1]);
        res.p3.push_back(q[2]); res.p4.push_back(q[3]);
        flat.push_back(q[0]); flat.push_back(q[1]); flat.push_back(q[2]); flat.push_back(q[3]);
    }

    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
    const core::GenotypeFrontEnd fe =
        core::read_genotype_front_end(geno, snp, ind, pop_union, be);
    const io::SnpTable& snptab = fe.snptab;
    const io::GenotypeTile& tile = fe.tile;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) {
        fill_nan(res, N);
        return res;
    }

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
            view, std::span<const int>(snptab.chrom.data(), static_cast<std::size_t>(M)),
            std::span<const double>(snptab.genpos_morgans.data(), static_cast<std::size_t>(M)),
            std::span<const double>(snptab.physpos.data(), static_cast<std::size_t>(M)),
            kAutosomeChromMin, kAutosomeChromMax);
        chrom_kept = ddr.chrom_kept;
        genpos_kept = ddr.genpos_kept;
        physpos_kept = ddr.physpos_kept;
    } else {
        const DecodeResult dec = be.decode_af(view);
        Qk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M));
        Vk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M));
        chrom_kept.reserve(static_cast<std::size_t>(M));
        genpos_kept.reserve(static_cast<std::size_t>(M));
        physpos_kept.reserve(static_cast<std::size_t>(M));
        for (long s = 0; s < M; ++s) {
            const int chr = snptab.chrom[static_cast<std::size_t>(s)];
            if (chr < kAutosomeChromMin || chr > kAutosomeChromMax) continue;
            const std::size_t src = static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
            for (int p = 0; p < P; ++p) {
                Qk.push_back(dec.q[src + static_cast<std::size_t>(p)]);
                Vk.push_back(dec.v[src + static_cast<std::size_t>(p)]);
            }
            chrom_kept.push_back(chr);
            genpos_kept.push_back(snptab.genpos_morgans[static_cast<std::size_t>(s)]);
            physpos_kept.push_back(snptab.physpos[static_cast<std::size_t>(s)]);
        }
    }
    const long M_kept = static_cast<long>(chrom_kept.size());
    if (M_kept <= 0) {
        fill_nan(res, N);
        return res;
    }

    const core::BlockPartition partition = core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans, std::span<const double>(physpos_kept));
    const int n_block = partition.n_block;
    if (n_block <= 0) {
        fill_nan(res, N);
        return res;
    }

    const RatioBlockJackknife jk =
        resident
            ? be.dstat_blocks_jackknife(ddr, partition.block_id.data(), n_block,
                                        std::span<const int>(flat))
            : be.dstat_blocks_jackknife(Qk.data(), Vk.data(), P, M_kept,
                                        partition.block_id.data(), n_block,
                                        std::span<const int>(flat));
    res.est = jk.est;
    res.se = jk.se;
    res.z = jk.z;
    res.p = jk.p;
    res.status = jk.status;
    return res;
}

}  // namespace steppe
