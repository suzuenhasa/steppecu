// src/device/readv2_bitmatrix.hpp
//
// Move-only, CUDA-free opaque handle to the resident READv2 [sample x SNP-window]
// bit-matrix left in VRAM across the all-pairs sweep. PIMPL: it names no CUDA type —
// the DeviceBuffer<Readv2Word> owner lives in a forward-declared Impl (mirrors
// DeviceDecodeResult). The core driver streams packed genotype chunks into it via the
// backend, then runs the mismatch sweep against it; the host never touches packed bits.
#ifndef STEPPE_DEVICE_READV2_BITMATRIX_HPP
#define STEPPE_DEVICE_READV2_BITMATRIX_HPP

#include <cstdint>
#include <memory>
#include <vector>

namespace steppe::device {

// The four per-pair reductions the mismatch sweep returns (host-resident; one entry
// per unordered pair r in [0, C(N,2)), r = C(j,2)+i for the pair i<j). The reduction/
// emit layer forms P0_mean = sum_p0 / n_win_used and the window-jackknife SE from
// sum_p0_sq; n_win_used -> n_windows, tot_comp -> n_overlap_sites.
struct Readv2Pairs {
    std::vector<double> sum_p0;
    std::vector<double> sum_p0_sq;
    std::vector<int> n_win_used;
    std::vector<std::int64_t> tot_comp;
};

// Move-only resident bit-matrix handle.
class Readv2Bitmatrix {
public:
    Readv2Bitmatrix();
    ~Readv2Bitmatrix();
    Readv2Bitmatrix(Readv2Bitmatrix&&) noexcept;
    Readv2Bitmatrix& operator=(Readv2Bitmatrix&&) noexcept;
    Readv2Bitmatrix(const Readv2Bitmatrix&) = delete;
    Readv2Bitmatrix& operator=(const Readv2Bitmatrix&) = delete;

    // Window geometry (host scalars).
    int n_samples = 0;
    int window_snps = 0;
    long m0 = 0;
    int wpw = 0;                 // words per window = ceil(window_snps / 64)
    long n_win = 0;              // windows tiling the SNP axis = ceil(m0 / window_snps)
    long words_per_sample = 0;   // W = n_win * wpw
    int device_id = -1;

    [[nodiscard]] bool empty() const noexcept {
        return n_samples <= 0 || words_per_sample <= 0;
    }

    // Opaque CUDA payload (PIMPL).
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_READV2_BITMATRIX_HPP
