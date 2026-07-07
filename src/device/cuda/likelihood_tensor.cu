// src/device/cuda/likelihood_tensor.cu — CUDA-side out-of-line special members of
// LikelihoodTensor so unique_ptr<Impl> sees a complete Impl. The Impl (the
// DeviceBuffer<double> payload + present-mask owner) lives in
// likelihood_tensor_impl.cuh. Private to steppe_device. Mirrors
// readv2_bitmatrix.cu.
#include <memory>

#include "device/cuda/likelihood_tensor_impl.cuh"

namespace steppe::device {

LikelihoodTensor::LikelihoodTensor() = default;
LikelihoodTensor::~LikelihoodTensor() = default;
LikelihoodTensor::LikelihoodTensor(LikelihoodTensor&&) noexcept = default;
LikelihoodTensor& LikelihoodTensor::operator=(LikelihoodTensor&&) noexcept = default;

}  // namespace steppe::device
