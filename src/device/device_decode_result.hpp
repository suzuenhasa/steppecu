// src/device/device_decode_result.hpp — CUDA-FREE opaque handle to the AUTOSOME-
// COMPACTED decode result Q/V [P × M_kept] left RESIDENT in VRAM (the device-
// resident decode seam; host-compute audit C1/C2/M3/M4 cure). Mirrors
// device_f2_blocks.hpp's CUDA-free-decl PIMPL pattern: names NO CUDA type; the
// DeviceBuffer<double> owners live in the Impl in cuda/device_decode_result_impl.cuh.
//
// WHAT THIS HANDLE CARRIES. The producer (CudaBackend::decode_af_compact_autosome)
// decodes the tile to Q/V/N RESIDENT (no host D2H — the C1 ~1.1GB Q/V/N copy is
// GONE), runs the on-device per-SNP autosome keep-mask kernel (the C2 host filter
// loop is GONE), and CUB/scan-gather compacts Q/V onto the kept axis (the M3/M4 host
// lockstep subset + H2D re-upload are GONE). The resident COMPACTED Q/V escape in
// this handle; only the small kept chrom/genpos cross to host (for the CUDA-free
// assign_blocks, which is tiny and unchanged → identical block_id parity).
//
// DELIBERATELY NO N: the regime-(A) consumers (qpfstats / dstat) read ONLY Q/V (the
// host-filter audit consumer list). The extract_f2 regime-(B) path (which needs N
// and the FP-sensitive MAF/maxmiss filter) is staged SEPARATELY and is not served by
// this autosome-only handle.
#ifndef STEPPE_DEVICE_DEVICE_DECODE_RESULT_HPP
#define STEPPE_DEVICE_DEVICE_DECODE_RESULT_HPP

#include <cstddef>
#include <memory>
#include <vector>

namespace steppe::device {

/// Move-only, OPAQUE owner of the AUTOSOME-COMPACTED Q/V [P × M_kept] RESIDENT in
/// VRAM on `device_id` — the output of decode_af_compact_autosome. CUDA-FREE: the
/// DeviceBuffer<double> q/v owners live in `Impl` (cuda/device_decode_result_impl.cuh);
/// the shape fields + the small kept chrom/genpos are plain host data so the
/// CUDA-free orchestrator (steppe::core) can hold the handle, run the (CUDA-free)
/// assign_blocks over the kept axis, and forward the resident Q/V into the reduce.
class DeviceDecodeResult {
public:
    DeviceDecodeResult();                                        // empty / moved-from
    ~DeviceDecodeResult();                                       // frees the resident Q/V
    DeviceDecodeResult(DeviceDecodeResult&&) noexcept;           // move-only
    DeviceDecodeResult& operator=(DeviceDecodeResult&&) noexcept;
    DeviceDecodeResult(const DeviceDecodeResult&) = delete;
    DeviceDecodeResult& operator=(const DeviceDecodeResult&) = delete;

    // ---- Shape (plain host scalars; CUDA-free) ----
    int P = 0;            ///< population count (leading dim of Q/V).
    long M_kept = 0;      ///< compacted (autosome-kept) SNP count (column count).
    int device_id = -1;   ///< the CUDA ordinal the q/v buffers are resident on.

    // ---- The compacted kept axis (small host metadata; FILE-ORDERED) ----
    // The kept SNPs' chrom / genetic position, in FILE ORDER (the CUB-Flagged
    // ordering guarantee), parallel to the resident Q/V columns. The CUDA-free
    // assign_blocks reads these to produce block_id — bit-identical to the host
    // autosome loop's chrom_kept/genpos_kept.
    std::vector<int> chrom_kept;            ///< length M_kept.
    std::vector<double> genpos_kept;        ///< length M_kept (Morgans).

    /// Borrowed device pointers to the resident compacted Q / V (column-major
    /// [P × M_kept], element (pop i, SNP s) at i + P·s). null when impl is null.
    /// Defined in cuda/device_decode_result.cu (dereferences Impl).
    [[nodiscard]] const double* q_device() const noexcept;
    [[nodiscard]] const double* v_device() const noexcept;

    /// True for a degenerate/empty result (no resident buffers).
    [[nodiscard]] bool empty() const noexcept { return M_kept <= 0 || P <= 0; }

    // ---- Opaque CUDA payload (the DeviceBuffer<double> q/v owners) ----
    struct Impl;                  // defined in cuda/device_decode_result_impl.cuh
    std::unique_ptr<Impl> impl;   // null => no resident buffers.
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_DEVICE_DECODE_RESULT_HPP
