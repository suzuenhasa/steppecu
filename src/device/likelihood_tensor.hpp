// src/device/likelihood_tensor.hpp
//
// Move-only, CUDA-free opaque handle to the resident genotype-likelihood tensor
// [n_site x n_sample x 3] (FP64) left in VRAM for the downstream GL consumers
// (PCAngsd covariance / ancIBD forward-backward). PIMPL: it names no CUDA type —
// the DeviceBuffer<double> payload + DeviceBuffer<uint8_t> present-mask live in a
// forward-declared Impl (mirrors Readv2Bitmatrix / DeviceDecodeResult). This is a
// GPU product: the deliverable is the on-device tensor, not a host array.
//
// LAYOUT (site-major, documented on the host io::LikelihoodTile it is uploaded
// from): l[(site*n_sample + sample)*3 + g], g = copies of panel A1.
#ifndef STEPPE_DEVICE_LIKELIHOOD_TENSOR_HPP
#define STEPPE_DEVICE_LIKELIHOOD_TENSOR_HPP

#include <memory>

namespace steppe::device {

// Move-only resident likelihood-tensor handle.
class LikelihoodTensor {
public:
    LikelihoodTensor();
    ~LikelihoodTensor();
    LikelihoodTensor(LikelihoodTensor&&) noexcept;
    LikelihoodTensor& operator=(LikelihoodTensor&&) noexcept;
    LikelihoodTensor(const LikelihoodTensor&) = delete;
    LikelihoodTensor& operator=(const LikelihoodTensor&) = delete;

    // Geometry (host scalars).
    long n_site = 0;
    int n_sample = 0;
    int device_id = -1;

    [[nodiscard]] long n_elem() const noexcept {
        return n_site * static_cast<long>(n_sample) * 3;
    }
    [[nodiscard]] bool empty() const noexcept { return n_site <= 0 || n_sample <= 0; }

    // Opaque CUDA payload (PIMPL).
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_LIKELIHOOD_TENSOR_HPP
