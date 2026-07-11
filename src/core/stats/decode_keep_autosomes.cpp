// src/core/stats/decode_keep_autosomes.cpp — the two shared genotype-path decode helpers.
//
// make_decode_tile_view is the byte-identical DecodeTileView wiring block; decode_and_keep_autosomes
// is the resident-or-host decode + autosome-keep block shared verbatim by dstat and qpfstats.
//
// Reference: docs/reference/src_core_stats_decode_keep_autosomes.hpp.md

#include "core/stats/decode_keep_autosomes.hpp"

#include <cstddef>
#include <span>

#include "core/internal/decode_af.hpp"
#include "core/internal/index_cast.hpp"
#include "steppe/config.hpp"

namespace steppe {
namespace core {

DecodeTileView make_decode_tile_view(const io::GenotypeTile& tile,
                                     const std::vector<int>& sample_ploidy, int P) {
    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.data();
    view.ploidy = core::kPloidyDiploid;
    return view;
}

DecodeTileView make_decode_tile_view(const DeviceGenotypeTile& tile,
                                     const std::vector<int>& sample_ploidy, int P) {
    DecodeTileView view;
    view.packed = tile.packed;              // DEVICE pointer to the resident canonical tile
    view.packed_on_device = true;
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.data();
    view.ploidy = core::kPloidyDiploid;
    return view;
}

DecodeKeepResult decode_and_keep_autosomes(ComputeBackend& be, const io::GenotypeTile& tile,
                                           const io::SnpTable& snptab, int P, long M) {
    std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view = make_decode_tile_view(tile, sample_ploidy, P);

    DecodeKeepResult out;
    out.resident = (be.capabilities().device_count > 0);
    if (out.resident) {
        out.ddr = be.decode_af_compact_autosome(
            view, std::span<const int>(snptab.chrom.data(), idx(M)),
            std::span<const double>(snptab.genpos_morgans.data(), idx(M)),
            std::span<const double>(snptab.physpos.data(), idx(M)),
            kAutosomeChromMin, kAutosomeChromMax);
        out.chrom_kept = out.ddr.chrom_kept;
        out.genpos_kept = out.ddr.genpos_kept;
        out.physpos_kept = out.ddr.physpos_kept;
    } else {
        const DecodeResult dec = be.decode_af(view);
        out.Qk.reserve(idx(P) * idx(M));
        out.Vk.reserve(idx(P) * idx(M));
        out.chrom_kept.reserve(idx(M));
        out.genpos_kept.reserve(idx(M));
        out.physpos_kept.reserve(idx(M));
        for (long s = 0; s < M; ++s) {
            const int chr = snptab.chrom[idx(s)];
            if (chr < kAutosomeChromMin || chr > kAutosomeChromMax) continue;
            const std::size_t src = idx(P) * idx(s);
            for (int p = 0; p < P; ++p) {
                out.Qk.push_back(dec.q[src + idx(p)]);
                out.Vk.push_back(dec.v[src + idx(p)]);
            }
            out.chrom_kept.push_back(chr);
            out.genpos_kept.push_back(snptab.genpos_morgans[idx(s)]);
            out.physpos_kept.push_back(snptab.physpos[idx(s)]);
        }
    }
    return out;
}

}  // namespace core
}  // namespace steppe
