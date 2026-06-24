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
// N IS CARRIED BY REGIME (B): the regime-(A) consumers (qpfstats / dstat) read ONLY
// Q/V, so decode_af_compact_autosome leaves N empty. The extract_f2 regime-(B) path
// (decode_af_compact_filter — the FP-sensitive MAF/maxmiss/class filter) compacts
// Q/V/N in LOCKSTEP onto the kept axis (the f2-GEMM needs N), so it populates the
// optional N buffer too. n_device() is null on a regime-(A) result, non-null on a
// regime-(B) result.
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

    /// Borrowed device pointer to the resident compacted N ([P × M_kept], i + P·s).
    /// NON-NULL only for a regime-(B) (filtered extract_f2) result; null for a
    /// regime-(A) (autosome-only) result, which does not compact N. Defined in
    /// cuda/device_decode_result.cu.
    [[nodiscard]] const double* n_device() const noexcept;

    /// REGIME-B read-back: synchronously D2H the resident COMPACTED Q/V/N
    /// [P × M_kept] into the supplied host vectors (resized to P·M_kept). The safe
    /// first regime-B landing (host-compute audit consumer option A): the host
    /// per-SNP filter loop + the FULL-tile D2H are GONE (the keep-set + compaction
    /// ran ON-DEVICE); only the SMALL compacted Q/V/N cross to host to feed the
    /// existing compute_f2_blocks_multigpu_tiered (which takes host MatViews) —
    /// analogous to the regime-A chrom/genpos D2H, just the bigger compacted arrays.
    /// Requires n_device() non-null (a regime-B result); a no-op on an empty result.
    /// Defined in cuda/device_decode_result.cu.
    void to_host_qvn(std::vector<double>& q_host, std::vector<double>& v_host,
                     std::vector<double>& n_host) const;

    /// True for a degenerate/empty result (no resident buffers).
    [[nodiscard]] bool empty() const noexcept { return M_kept <= 0 || P <= 0; }

    // ---- Opaque CUDA payload (the DeviceBuffer<double> q/v owners) ----
    struct Impl;                  // defined in cuda/device_decode_result_impl.cuh
    std::unique_ptr<Impl> impl;   // null => no resident buffers.
};

}  // namespace steppe::device
#endif  // STEPPE_DEVICE_DEVICE_DECODE_RESULT_HPP
