// include/steppe/extract.hpp
//
// The public, CUDA-free library entry point that turns a genotype triple into
// f2 statistics — the bindable counterpart of the `steppe extract-f2` command.
// It takes plain arguments, returns a value, and throws on failure (never
// prints), so both the CLI and the Python bindings run the same computation.
//
// Reference: docs/reference/include_steppe_extract.hpp.md
#ifndef STEPPE_EXTRACT_HPP
#define STEPPE_EXTRACT_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

namespace io {
struct PopSelection;
}  // namespace io

// ExtractPloidy — reference §3
enum class ExtractPloidy { Auto, PseudoHaploid, Diploid };

// ExtractTier — reference §4
enum class ExtractTier { Resident, HostRam, Disk };

// F2ExtractResult — reference §5
struct F2ExtractResult {
    F2BlockTensor f2;
    std::vector<std::string> pop_labels;
    long n_snp_total = 0;
    long n_snp_kept = 0;
    std::size_t n_pseudo_haploid = 0;
    std::size_t n_diploid = 0;
    Precision::Kind precision_tag = Precision::Kind::EmulatedFp64;
    ExtractTier tier = ExtractTier::Resident;
    Status status = Status::Ok;
};

// run_extract_f2 — reference §6
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
