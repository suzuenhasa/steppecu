// src/core/stats/genotype_front_end.cpp
//
// The shared genotype DECODE FRONT-END (C1) — the exact lift of the front-end the four
// genotype-path tools (extract_f2 / qpDstat-B / qpfstats / DATES) each copy-pasted: open the
// io::GenoReader, read the IndPartition + SnpTable, compute M0, and read the canonical tile
// via read_canonical_tile (the M-FR-2 format dispatch). It stops at the {tile, snptab, part,
// fmt, M0} handoff — the per-tool decode (forced diploid + autosome-keep for D/qpfstats/DATES,
// per-sample ploidy + the regime-B filter for extract) stays in the caller. The produced
// tile/snptab/part bytes are identical to the inline copies (same reader, same SIZE_MAX, same
// M0, same read_canonical_tile call), so no golden moves. See genotype_front_end.hpp for the
// layering rationale.
#include "core/stats/genotype_front_end.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>  // SIZE_MAX

#include "core/stats/read_canonical_tile.hpp"  // M-FR-2 TGENO/GENO/EIGENSTRAT/PLINK/ANCESTRYMAP dispatch

namespace steppe::core {

GenotypeFrontEnd read_genotype_front_end(const std::string& geno, const std::string& snp,
                                         const std::string& ind, const io::PopSelection& sel,
                                         ComputeBackend& backend) {
    GenotypeFrontEnd fe;
    io::GenoReader reader(geno);
    fe.fmt = reader.header().format;  // .snp|.bim, .ind|.fam parser (M-FR PLINK)
    const std::size_t n_present = reader.records_present();
    fe.part = io::read_ind_partition(fe.fmt, ind, sel, n_present);
    fe.snptab = io::read_snp_table(fe.fmt, snp, SIZE_MAX);
    fe.M0 = std::min(reader.header().n_snp, fe.snptab.count);
    // M-FR-2 FORMAT DISPATCH: TGENO -> read_tile (unchanged); GENO/EIGENSTRAT/PLINK/
    // ANCESTRYMAP -> the io-leaf SNP-major gather + the on-device transpose_to_canonical.
    // `tile` is the canonical individual-major packing every downstream decode expects.
    fe.tile = read_canonical_tile(reader, fe.part, backend, 0, fe.M0);
    return fe;
}

GenotypeFrontEnd read_genotype_front_end(const std::string& geno, const std::string& snp,
                                         const std::string& ind,
                                         std::span<const std::string> pop_labels,
                                         ComputeBackend& backend) {
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::Explicit;
    sel.labels.assign(pop_labels.begin(), pop_labels.end());  // the AT2 indvec (only these pops).
    return read_genotype_front_end(geno, snp, ind, sel, backend);
}

}  // namespace steppe::core
