// src/device/cuda/device_decode_result.cu — CUDA-side implementation of
// DeviceDecodeResult: out-of-line special members (so unique_ptr<Impl> sees a
// complete Impl), the device-pointer accessors, and the compacted-Q/V/N D2H copy.
// Private to steppe_device (a CUDA TU, architecture.md §4).
#include "device/cuda/device_decode_result_impl.cuh"

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "device/cuda/check.cuh"

namespace steppe::device {

DeviceDecodeResult::DeviceDecodeResult() = default;
DeviceDecodeResult::~DeviceDecodeResult() = default;
DeviceDecodeResult::DeviceDecodeResult(DeviceDecodeResult&&) noexcept = default;
DeviceDecodeResult& DeviceDecodeResult::operator=(DeviceDecodeResult&&) noexcept = default;

const double* DeviceDecodeResult::q_device() const noexcept {
    return impl ? impl->q.data() : nullptr;
}
const double* DeviceDecodeResult::v_device() const noexcept {
    return impl ? impl->v.data() : nullptr;
}
const double* DeviceDecodeResult::n_device() const noexcept {
    return impl ? impl->n.data() : nullptr;
}

void DeviceDecodeResult::to_host_qvn(std::vector<double>& q_host,
                                     std::vector<double>& v_host,
                                     std::vector<double>& n_host) const {
    if (!impl || M_kept <= 0 || P <= 0) {
        q_host.clear();
        v_host.clear();
        n_host.clear();
        return;
    }
    const std::size_t pmk =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(M_kept);
    q_host.assign(pmk, 0.0);
    v_host.assign(pmk, 0.0);
    n_host.assign(pmk, 0.0);
    if (impl->n.data() == nullptr) {
        throw std::invalid_argument(
            "DeviceDecodeResult::to_host_qvn: N buffer is empty — this is a "
            "regime-A (autosome-only) result, but to_host_qvn requires a "
            "regime-B (filtered extract_f2) result with a resident compacted N "
            "(see device_decode_result.hpp: 'Requires n_device() non-null')");
    }
    STEPPE_CUDA_CHECK(cudaMemcpy(q_host.data(), impl->q.data(), pmk * sizeof(double),
                                 cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(v_host.data(), impl->v.data(), pmk * sizeof(double),
                                 cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(n_host.data(), impl->n.data(), pmk * sizeof(double),
                                 cudaMemcpyDeviceToHost));
}

}  // namespace steppe::device
