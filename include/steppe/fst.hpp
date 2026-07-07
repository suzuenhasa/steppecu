// include/steppe/fst.hpp
//
// Public, CUDA-free entry point for the standalone per-SNP Weir & Cockerham 1984 FST
// (`steppe fst`). Reads a .geno/.snp/.ind triple through the shared genotype decode
// front-end, then runs a per-site WC variance-component kernel over the device-resident
// genotype tile for one population pair — an engine-independent standalone stat (no
// f2 cache, no Li-Stephens). Gated against plink2 `--fst method=wc`, NOT ADMIXTOOLS2, so
// the emulated-FP64-matmul parity policy does not bind: FST uses native FP64.
//
// v1 scope: pairwise (2-pop) per-SNP WC FST. Multi-pop / all-pairs is a documented
// follow-up.
#ifndef STEPPE_FST_HPP
#define STEPPE_FST_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

// FstResult — the per-SNP FST table plus the genome-wide summary.
struct FstResult {
    // Per-SNP (kept order), length M:
    std::vector<double> num;                 // WC numerator a (== plink2 FST_NUMER)
    std::vector<double> den;                 // a + b + c       (== plink2 FST_DENOM)
    std::vector<double> fst;                 // a/(a+b+c)       (== plink2 WC_FST; NaN if invalid)
    std::vector<std::uint8_t> valid;         // 1 = usable site, 0 = invalid (NaN fst)
    std::vector<int> chrom;
    std::vector<double> physpos;
    std::vector<std::string> snp_id;
    std::vector<char> a1;                    // .snp ref allele
    std::vector<char> a2;                    // .snp alt allele

    std::string popA;
    std::string popB;

    // Genome-wide summary over the autosomal valid sites:
    double fst_ratio = 0.0;                  // sum_num / sum_den (== plink2 .fst.summary WC_FST)
    double fst_mean = 0.0;                   // unweighted mean of per-SNP fst (secondary)
    long n_valid = 0;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_fst — the pairwise per-SNP WC FST driver. popA/popB are .ind population labels.
[[nodiscard]] FstResult run_fst(const std::string& geno,
                                const std::string& snp,
                                const std::string& ind,
                                const std::string& popA,
                                const std::string& popB,
                                device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_FST_HPP
