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

namespace steppe::core {

GenotypeFrontEnd read_genotype_front_end(const std::string& geno, const std::string& snp,
                                         const std::string& ind, const io::PopSelection& sel,
                                         ComputeBackend& backend) {
    GenotypeFrontEnd fe;
    io::GenoReader reader(geno);
    fe.fmt = reader.header().format;
    const std::size_t n_present = reader.records_present();
    fe.part = io::read_ind_partition(fe.fmt, ind, sel, n_present);
    fe.snptab = io::read_snp_table(fe.fmt, snp, SIZE_MAX);
    fe.M0 = std::min(reader.header().n_snp, fe.snptab.count);
    fe.tile = read_canonical_tile(reader, fe.part, backend, 0, fe.M0);
    return fe;
}

GenotypeFrontEnd read_genotype_front_end(const std::string& geno, const std::string& snp,
                                         const std::string& ind,
                                         std::span<const std::string> pop_labels,
                                         ComputeBackend& backend) {
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::Explicit;
    sel.labels.assign(pop_labels.begin(), pop_labels.end());
    return read_genotype_front_end(geno, snp, ind, sel, backend);
}

}  // namespace steppe::core
