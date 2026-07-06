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
#include "core/internal/decode_af.hpp"
#include "core/internal/index_cast.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
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

// Degenerate-outcome NaN fill — reference §9
void fill_nan(DstatResult& res, int row_count) {
    const auto count = idx(row_count);
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

    res.p1.reserve(idx(N));
    res.p2.reserve(idx(N));
    res.p3.reserve(idx(N));
    res.p4.reserve(idx(N));
    std::vector<int> flat;
    flat.reserve(idx(N) * 4);
    for (const std::array<int, 4>& q : quadruples) {
        res.p1.push_back(q[0]); res.p2.push_back(q[1]);
        res.p3.push_back(q[2]); res.p4.push_back(q[3]);
        flat.push_back(q[0]); flat.push_back(q[1]); flat.push_back(q[2]); flat.push_back(q[3]);
    }

    ComputeBackend& be = device::primary_backend(resources);
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

    const core::DecodeKeepResult dk = core::decode_and_keep_autosomes(be, tile, snptab, P, M);
    const bool resident = dk.resident;
    const device::DeviceDecodeResult& ddr = dk.ddr;
    const std::vector<double>& Qk = dk.Qk;
    const std::vector<double>& Vk = dk.Vk;
    const std::vector<int>& chrom_kept = dk.chrom_kept;
    const std::vector<double>& genpos_kept = dk.genpos_kept;
    const std::vector<double>& physpos_kept = dk.physpos_kept;
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
