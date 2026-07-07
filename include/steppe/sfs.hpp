// include/steppe/sfs.hpp
//
// Public, CUDA-free entry point for the 2D joint site-frequency spectrum (`steppe sfs`).
// Reads a .geno/.snp/.ind triple through the shared genotype decode front-end, then runs a
// GPU joint-histogram over the device-resident genotype tile for one population pair — an
// engine-independent standalone stat (no f2 cache, no Li-Stephens, no likelihoods). It is
// a pure INTEGER-count stat gated BIT-EXACT (cell-by-cell) against scikit-allel
// joint_sfs / joint_sfs_folded, NOT ADMIXTOOLS2, so no FP parity policy binds.
//
// v1 scope: 2D (two-pop) joint SFS, folded (per-pop minor) or unfolded (A1-copy count),
// restricted to sites with COMPLETE data in both pops (§4). A 3D (three-pop) extension and
// a hypergeometric down-projection are documented follow-ups.
#ifndef STEPPE_SFS_HPP
#define STEPPE_SFS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

// SfsResult — the 2D joint SFS matrix (row-major, extA x extB) plus provenance.
struct SfsResult {
    std::vector<std::int64_t> grid;   // row-major extA*extB; cell (i,j) at grid[i*extB + j]
    long extA = 0, extB = 0;          // category extents (2N+1 unfolded, N+1 folded)
    long NA = 0, NB = 0;              // individuals per pop (chromosome count = 2N)
    long n_total = 0;                 // SNPs in the tile
    long n_complete = 0;              // sites histogrammed (complete in both pops)
    long n_dropped_incomplete = 0;    // n_total - n_complete
    bool folded = false;
    std::string popA, popB;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_sfs — the 2D joint SFS driver. popA/popB are .ind/.fam population labels; folded
// selects the per-pop minor fold (polarity-free) vs the unfolded A1-copy count.
[[nodiscard]] SfsResult run_sfs(const std::string& geno,
                                const std::string& snp,
                                const std::string& ind,
                                const std::string& popA,
                                const std::string& popB,
                                bool folded,
                                device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_SFS_HPP
