// src/core/stats/genotype_front_end.cpp
//
// The shared genotype decode front-end: open the reader, read the individual
// partition and SNP table, compute M0, and read the canonical tile. Stops at the
// {tile, snptab, part, fmt, M0} handoff; each tool's per-sample decode stays in the caller.
#include "core/stats/genotype_front_end.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "core/stats/read_canonical_tile.hpp"
#include "device/backend.hpp"

namespace steppe::core {

GenotypeFrontEnd read_genotype_front_end(const std::string& geno, const std::string& snp,
                                         const std::string& ind, const io::PopSelection& sel,
                                         ComputeBackend& backend, bool allow_device) {
    GenotypeFrontEnd fe;
    io::GenoReader reader(geno);
    fe.fmt = reader.header().format;
    const std::size_t n_present = reader.records_present();
    fe.part = io::read_ind_partition(fe.fmt, ind, sel, n_present);
    fe.snptab = io::read_snp_table(fe.fmt, snp, SIZE_MAX);
    fe.M0 = std::min(reader.header().n_snp, fe.snptab.count);

    // GPU-native device-resident load when the caller allows it, a CUDA backend is present, the
    // selector is on, and there is a non-empty SNP axis. The canonical bytes are built ONCE on
    // device; the host `tile` keeps only the descriptor (empty packed), so the whole-matrix host
    // materialization + every tool's re-upload are eliminated. A migrated caller passes
    // allow_device=true even with an active SNP filter — the filter then compacts the resident tile
    // on-device (apply_snp_filter -> compact_tile_columns_device), no host round-trip.
    if (allow_device && fe.M0 > 0 && device_load_enabled() &&
        backend.capabilities().device_count > 0) {
        fe.dev_tile = read_canonical_tile_device(reader, fe.part, backend, 0, fe.M0);
        if (fe.dev_tile.valid()) {
            fe.tile.packed.clear();
            fe.tile.bytes_per_record = fe.dev_tile.bytes_per_record;
            fe.tile.n_snp = fe.dev_tile.n_snp;
            fe.tile.n_individuals = fe.dev_tile.n_individuals;
            fe.tile.pop_offsets = fe.dev_tile.pop_offsets;
            fe.tile.pop_labels = fe.dev_tile.pop_labels;
            return fe;
        }
    }

    fe.tile = read_canonical_tile(reader, fe.part, backend, 0, fe.M0);
    return fe;
}

GenotypeFrontEnd read_genotype_front_end(const std::string& geno, const std::string& snp,
                                         const std::string& ind,
                                         std::span<const std::string> pop_labels,
                                         ComputeBackend& backend, bool allow_device) {
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::Explicit;
    sel.labels.assign(pop_labels.begin(), pop_labels.end());
    return read_genotype_front_end(geno, snp, ind, sel, backend, allow_device);
}

}  // namespace steppe::core
