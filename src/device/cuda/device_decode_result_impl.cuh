// src/device/cuda/device_decode_result_impl.cuh — the CUDA side of
// DeviceDecodeResult: the Impl holding the resident DeviceBuffer<double> q/v owners.
// PRIVATE to steppe_device. Shared by cuda/device_decode_result.cu (special members +
// q_device/v_device) and cuda_backend.cu (builds the result device-resident).
#ifndef STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH

#include "device/device_decode_result.hpp"
#include "device/cuda/device_buffer.cuh"

namespace steppe::device {

struct DeviceDecodeResult::Impl {
    DeviceBuffer<double> q;  ///< [P × M_kept] resident on device_id (column-major).
    DeviceBuffer<double> v;  ///< [P × M_kept] resident on device_id (column-major).
    // N is populated ONLY by the regime-B filtered path (decode_af_compact_filter);
    // the regime-A autosome path leaves it empty (its consumers read only Q/V). The
    // extract_f2 f2-GEMM needs the compacted N too (the SEPARATE regime-B staging the
    // device_decode_result.hpp:14-18 doc names).
    DeviceBuffer<double> n;  ///< [P × M_kept] resident (regime-B only; empty for regime-A).
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_CUDA_DEVICE_DECODE_RESULT_IMPL_CUH
