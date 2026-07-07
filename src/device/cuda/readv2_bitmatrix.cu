// src/device/cuda/readv2_bitmatrix.cu — CUDA-side out-of-line special members of
// Readv2Bitmatrix so unique_ptr<Impl> sees a complete Impl. The Impl (the
// DeviceBuffer<Readv2Word> owner) lives in readv2_bitmatrix_impl.cuh. Private to
// steppe_device.
// Reference: docs/reference/src_device_cuda_readv2_bitmatrix.cu.md
#include "device/cuda/readv2_bitmatrix_impl.cuh"

#include <memory>

namespace steppe::device {

Readv2Bitmatrix::Readv2Bitmatrix() = default;
Readv2Bitmatrix::~Readv2Bitmatrix() = default;
Readv2Bitmatrix::Readv2Bitmatrix(Readv2Bitmatrix&&) noexcept = default;
Readv2Bitmatrix& Readv2Bitmatrix::operator=(Readv2Bitmatrix&&) noexcept = default;

}  // namespace steppe::device
