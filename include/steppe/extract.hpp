// include/steppe/extract.hpp
//
// PUBLIC, CUDA-FREE genotype->f2 EXTRACT entry point — M(py-2). This is the library-level
// counterpart of the `steppe extract-f2` CLI command: the decode->filter->assign_blocks->
// compute_f2_blocks_multigpu_tiered->to_host chain, lifted VERBATIM out of
// src/app/cmd_extract_f2.cpp (the math is byte-identical — the goldens are untouched), with
// the CLI's std::fprintf(stderr)/kExit* exit codes REPLACED by exceptions + a value result.
//
// WHY a library entry (architecture.md §4, §10; cli-bindings.md §1.3): the CLI command
// `run_extract_f2_command` prints to stdout/stderr and returns process exit codes, so it is
// NOT bindable — the binding contract is EXCEPTION-based and returns Python objects, never an
// exit code, and the library never prints. This entry mirrors the run_dstat (dstat.hpp)
// precedent: a public, CUDA-FREE function that reads the genotype TRIPLE directly, takes a
// PopSelection + a FilterConfig + a Precision + a device::Resources&, THROWS std::exception on
// a fault, and RETURNS a value type. run_extract_f2_command is now a thin CLI wrapper over it
// (no behavior change). The binding (bindings/module.cpp run_extract_f2_py) reaches the GPU
// ONLY through this seam + device/resources.hpp, staying CUDA-free (the §4 layering proof).
//
// THE PARITY PINS carried verbatim from cmd_extract_f2.cpp (so a bare extract reproduces the
// AT2 golden): (1) the P axis is read_ind(Explicit{pops}) sorted ASC by label = pops.txt
// order; (2) per-SAMPLE ploidy auto-detect (AT2 adjust_pseudohaploid) unless forced; (3) the
// AT2 maxmiss is the POPULATION-axis coverage test (NOT the sample-axis predicate), applied as
// a separate per-SNP test while the sample-axis geno_max_missing is forced to the no-op; (4)
// autosomes_only is the FilterConfig flag (the CLI/AT2 extract_f2 default is auto_only=TRUE).
#ifndef STEPPE_EXTRACT_HPP
#define STEPPE_EXTRACT_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"   // steppe::Precision, FilterConfig
#include "steppe/error.hpp"    // steppe::Status
#include "steppe/fstats.hpp"   // steppe::F2BlockTensor

namespace steppe {

namespace device {
struct Resources;  // CUDA-free fwd-decl (real decl: src/device/resources.hpp)
}  // namespace device

namespace io {
struct PopSelection;  // CUDA-free fwd-decl (real decl: src/io/ind_reader.hpp)
}  // namespace io

/// Ploidy convention for the decode (mirrors config::PloidyMode; named here so the public
/// extract surface does not pull the CLI's RunConfig). Auto = AT2 adjust_pseudohaploid
/// per-sample detection (the extract-f2 default); PseudoHaploid/Diploid force a uniform
/// per-sample vector.
enum class ExtractPloidy { Auto, PseudoHaploid, Diploid };

/// Which TIER the tiered f2 compute placed the result in (a CUDA-free mirror of
/// device::OutputTier — named here so the PUBLIC extract surface does not pull the internal
/// device/tier_select.hpp header). Resident = device-resident (the win at small P); HostRam/
/// Disk = the M5 SNP-tile input-streaming tiers. PARITY-NEUTRAL observability only (the tier
/// changes WHERE a slab lands, never its bits — architecture.md §12); echoed by the CLI
/// summary + meta so a `--tier`/STEPPE_FORCE_TIER override is visibly honored.
enum class ExtractTier { Resident, HostRam, Disk };

/// The result of run_extract_f2: the host F2BlockTensor (REAL f2 + vpair, FP64 in every
/// precision mode), the P pop labels in P-axis index order (the name<->index map the
/// INDEX-ONLY engine lacks — = pops.txt order), and the provenance the CLI/meta.json
/// records. A by-value result (the binding wraps it in an F2Handle or hands it to the
/// dir writer). On a fault (no device, bad pop name, missing file, every SNP filtered)
/// run_extract_f2 THROWS std::exception — never a partial result (architecture.md §10:
/// a FAULT raises; a domain outcome would ride on a field, but extract has no per-row
/// domain outcome).
struct F2ExtractResult {
    F2BlockTensor f2;                       ///< the host f2 tensor (REAL f2 + REAL vpair).
    std::vector<std::string> pop_labels;    ///< P labels in P-axis index order (= pops.txt).
    long n_snp_total = 0;                   ///< SNPs READ from the .snp (pre-filter).
    long n_snp_kept = 0;                    ///< SNPs kept after the filters.
    std::size_t n_pseudo_haploid = 0;       ///< samples decoded pseudo-haploid (observability).
    std::size_t n_diploid = 0;              ///< samples decoded diploid (observability).
    Precision::Kind precision_tag = Precision::Kind::EmulatedFp64;  ///< the ENGAGED precision.
    ExtractTier tier = ExtractTier::Resident;  ///< the tier the tiered compute used (observability).
    Status status = Status::Ok;             ///< Ok on success (a fault throws, never a status).
};

/// Genotype-path f2 EXTRACT over the genotype triple (.geno/.snp/.ind) — the library entry.
/// `geno`/`snp`/`ind` are the EIGENSTRAT/TGENO triple paths. `pops` selects the populations
/// (Explicit{labels} for the named-subset case the binding uses; AutoTopK/MinN supported too
/// — IDENTICAL to the CLI). The P axis is that selection's partition, SORTED ASC by label.
/// `filter` is the on-the-fly QC config (the AT2 maxmiss = filter.geno_max_missing population-
/// axis coverage; autosomes_only = filter.autosomes_only; MAF / drop-mono / transversions as
/// the FilterConfig flags). `precision` governs the f2 GEMMs (default EmulatedFp64 40-bit).
/// `blgsize_morgans` is the jackknife block size in MORGANS (AT2 default 0.05). `ploidy` is
/// Auto by default (AT2 adjust_pseudohaploid per-sample). Routes the decode + the tiered f2
/// compute through resources.gpus[0] (GPU-first, device-resident; multi-GPU PARKED). THROWS
/// std::runtime_error / std::invalid_argument on any fault (no device, an unknown Explicit
/// pop name, a missing/unreadable file, an empty selection, or every SNP filtered out).
[[nodiscard]] F2ExtractResult run_extract_f2(const std::string& geno,
                                             const std::string& snp,
                                             const std::string& ind,
                                             const io::PopSelection& pops,
                                             const FilterConfig& filter,
                                             const Precision& precision,
                                             double blgsize_morgans,
                                             ExtractPloidy ploidy,
                                             device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_EXTRACT_HPP
