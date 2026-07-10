// src/core/stats/read_vcf_panel_front_end.cpp
//
// Build the canonical GenotypeFrontEnd directly from a phased VCF: stream the file
// once through the io leaf io::read_vcf_panel to get {SnpMajorTile, SnpTable}, then
// run the shared on-device transpose to produce the individual-major GenotypeTile
// paint consumes. No .snp/.ind sidecars, no GenoReader random-access window.
#include "core/stats/read_vcf_panel_front_end.hpp"

#include <utility>

#include "core/stats/read_canonical_tile.hpp"
#include "io/eigenstrat_format.hpp"

namespace steppe::core {

GenotypeFrontEnd read_vcf_panel_front_end(const std::string& vcf,
                                          const io::VcfPanelOptions& opts,
                                          ComputeBackend& backend) {
    // Forward-only streaming decode (throws on I/O failure / phase loss over the guard).
    io::VcfPanelResult panel = io::read_vcf_panel(vcf, opts);

    GenotypeFrontEnd fe;
    fe.snptab = std::move(panel.snptab);

    // The ONE shared SNP-major -> canonical individual-major transpose seam (Identity
    // encoding), the same call the five packed-format arms use.
    fe.tile = transpose_snp_major(panel.tile, backend);

    fe.M0 = fe.tile.n_snp;

    // Inline VCF has no on-disk GenoReader format arm (this is a dedicated producer,
    // not the 6th read_canonical_tile switch case).
    fe.fmt = io::GenoFormat::Unknown;

    // A single "PANEL" partition over every haplotype column, mirroring the tile's one
    // PANEL population. paint reads per-column labels off the tile's pop_offsets /
    // pop_labels (per_individual_labels), not off part; part is filled for struct
    // completeness and any future consumer of the front-end.
    io::PopGroup g;
    g.label = "PANEL";
    g.rows.resize(fe.tile.n_individuals);
    for (std::size_t i = 0; i < fe.tile.n_individuals; ++i) g.rows[i] = i;
    fe.part.groups.clear();
    fe.part.groups.push_back(std::move(g));
    fe.part.n_individuals_total = fe.tile.n_individuals;

    return fe;
}

}  // namespace steppe::core
