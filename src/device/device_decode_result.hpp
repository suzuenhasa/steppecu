// src/device/device_decode_result.hpp
//
// Move-only, CUDA-free opaque handle to the autosome-compacted decode output
// Q/V/N [P × M_kept] left resident in VRAM. PIMPL: it names no CUDA type — the
// DeviceBuffer<double> owners live in a forward-declared Impl.
//
// Reference: docs/reference/src_device_device_decode_result.hpp.md
#ifndef STEPPE_DEVICE_DEVICE_DECODE_RESULT_HPP
#define STEPPE_DEVICE_DEVICE_DECODE_RESULT_HPP

#include <cstddef>
#include <memory>
#include <vector>

namespace steppe::device {

// Move-only, CUDA-free resident-decode handle — reference §4
class DeviceDecodeResult {
public:
    DeviceDecodeResult();
    ~DeviceDecodeResult();
    DeviceDecodeResult(DeviceDecodeResult&&) noexcept;
    DeviceDecodeResult& operator=(DeviceDecodeResult&&) noexcept;
    DeviceDecodeResult(const DeviceDecodeResult&) = delete;
    DeviceDecodeResult& operator=(const DeviceDecodeResult&) = delete;

    // Shape (host scalars) — reference §5
    int P = 0;
    long M_kept = 0;
    int device_id = -1;

    // Kept-axis metadata (file-ordered) — reference §6
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    std::vector<double> physpos_kept;

    // Resident Q/V/N device pointers (N regime-B only) — reference §7
    [[nodiscard]] const double* q_device() const noexcept;
    [[nodiscard]] const double* v_device() const noexcept;

    [[nodiscard]] const double* n_device() const noexcept;

    // Regime-B host read-back — reference §8
    void to_host_qvn(std::vector<double>& q_host, std::vector<double>& v_host,
                     std::vector<double>& n_host) const;

    // empty() predicate — reference §9
    [[nodiscard]] bool empty() const noexcept { return M_kept <= 0 || P <= 0; }

    // Opaque CUDA payload (PIMPL) — reference §9
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_DEVICE_DECODE_RESULT_HPP
