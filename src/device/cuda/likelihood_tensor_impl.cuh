// src/device/cuda/likelihood_tensor_impl.cuh
//
// The CUDA-side definition of LikelihoodTensor::Impl — the DeviceBuffer<double>
// payload + DeviceBuffer<uint8_t> present-mask owner. Shared by
// likelihood_tensor.cu (the special members) and cuda_backend_likelihood.cu (which
// uploads/reduces it), mirroring readv2_bitmatrix_impl.cuh. Private to
// steppe_device.
#ifndef STEPPE_DEVICE_CUDA_LIKELIHOOD_TENSOR_IMPL_CUH
#define STEPPE_DEVICE_CUDA_LIKELIHOOD_TENSOR_IMPL_CUH

#include <cstdint>

#include "device/cuda/device_buffer.cuh"
#include "device/likelihood_tensor.hpp"

namespace steppe::device {

struct LikelihoodTensor::Impl {
    DeviceBuffer<double> l;              // [n_site * n_sample * 3], site-major
    DeviceBuffer<std::uint8_t> present;  // [n_site * n_sample]
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_LIKELIHOOD_TENSOR_IMPL_CUH
