// src/device/cuda/device_decode_result.cu — the CUDA side of DeviceDecodeResult.
// Out-of-line special members (so unique_ptr<Impl> sees a complete Impl at
// instantiation) + the device-pointer accessors. PRIVATE to steppe_device (a CUDA
// TU, architecture.md §4). Mirrors device_f2_blocks.cu's defaulted-special-members
// pattern: the DeviceBuffer<double> q/v free device-agnostically in the dtor
// (cudaFree carries the pointer's device).
#include "device/cuda/device_decode_result_impl.cuh"

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "device/cuda/check.cuh"  // STEPPE_CUDA_CHECK

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
    // Synchronous D2H of the resident COMPACTED Q/V/N. The buffers are resident on
    // `device_id`; the producing backend already ran on that device and synchronized
    // its stream before returning the handle, so a default-stream copy here observes
    // the gathered values. Only the SMALL compacted arrays cross (the regime-B cure:
    // the host per-SNP filter loop + the full-tile D2H are gone).
    // Fail-fast on the documented regime-B precondition (header
    // device_decode_result.hpp:77: "Requires n_device() non-null"). A regime-A
    // result has `impl` non-null but `impl->n` empty (n.data() == nullptr) while
    // pmk > 0, so the N copy below would be cudaMemcpy(dst, nullptr, pmk*8, D2H) —
    // STEPPE_CUDA_CHECK would then surface an opaque CUDA-runtime throw (a null device
    // source) rather than the contract violated here. The empty() short-circuit at the top
    // tests only P/M_kept, not N residency, so it cannot catch this misuse. This is
    // a REAL runtime guard (not STEPPE_ASSERT, which compiles out under NDEBUG).
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
