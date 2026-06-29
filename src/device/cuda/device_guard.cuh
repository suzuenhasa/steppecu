// src/device/cuda/device_guard.cuh
//
// DeviceGuard — THE single scoped CUDA current-device restore RAII helper
// (architecture.md §2 DRY/RAII, §7 idioms; NAMING-STYLE-STANDARD §2.12 RAII template,
// §3.3 "replace the duplicated device-restore guard with one shared DeviceGuard"). It
// captures a device ordinal at construction and, at scope exit, restores it as the
// current CUDA device — even on a throwing / early-return path.
//
// This replaces the `struct DeviceGuard { int dev; ~DeviceGuard(){...} }` that was
// copy-pasted across the device-resident f2 pipeline (device_f2_blocks.cu to_host /
// upload_f2_blocks_to_device, p2p_combine.cu's two combine entries, f2_blocks_out.cu's
// tiered read-back): each bound a buffer's owning device for a D2H / H2D / peer copy
// and then had to hand the caller's device back. One home ⇒ no drift.
//
// TEARDOWN -> WARN (standard §2.12: "a destructor never throws; a nonzero teardown
// status routes to STEPPE_LOG_WARN"). The dtor cannot throw during unwinding, so the
// restore goes through the NON-throwing STEPPE_CUDA_WARN (it logs one diagnostic line
// in debug and YIELDS the cudaError_t — cudaSetDevice can surface a prior async launch
// error; CUDA 13.x Device Management: `__host__ cudaError_t cudaSetDevice(int device)`,
// DOC-VERIFIED) rather than the throwing STEPPE_CUDA_CHECK. The [[nodiscard]] status is
// (void)-discarded for the -Werror build; the happy path is byte-identical to a bare
// restore. This HARMONIZES the two former plain `(void)cudaSetDevice` sites to the WARN
// form, so a failed restore there no longer vanishes silently.
//
// MOVE-ONLY (standard §2.12 RAII template): copy is deleted; the move parks the
// moved-from at kNoDevice so it owns nothing and its dtor is a no-op (no double
// restore). At every current call site the guard is a never-moved named local, so
// move-only is behavior-preserving there and simply makes it a reusable RAII owner.
// PRIVATE to steppe_device (a CUDA header, architecture.md §4 layering).
#ifndef STEPPE_DEVICE_CUDA_DEVICE_GUARD_CUH
#define STEPPE_DEVICE_CUDA_DEVICE_GUARD_CUH

#include <cuda_runtime.h>   // cudaSetDevice (returns cudaError_t)

#include <utility>          // std::exchange (moved-from -> kNoDevice)

#include "device/cuda/check.cuh"  // STEPPE_CUDA_WARN -> STEPPE_LOG_WARN (non-throwing teardown)

namespace steppe::device {

/// Scoped restore of the CUDA current device (the SINGLE shared device-restore RAII
/// helper; standard §3.3). Construct with the ordinal to restore at scope exit;
/// binding the WORKING device is the caller's job (an explicit
/// STEPPE_CUDA_CHECK(cudaSetDevice(...)) after construction), so the guard owns ONLY
/// the restore. Move-only; the dtor never throws (teardown routes to STEPPE_CUDA_WARN).
class DeviceGuard {
public:
    /// Capture `dev` (typically the caller's current device from cudaGetDevice) to
    /// restore at scope exit. `explicit` so a bare int never implicitly becomes a guard.
    explicit DeviceGuard(int dev) noexcept : dev_(dev) {}

    /// Restore the captured device unless moved-from. Never throws (standard §2.12):
    /// the nonzero status routes through the non-throwing STEPPE_CUDA_WARN.
    ~DeviceGuard() { restore(); }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    /// Move: adopt other's ordinal and park other at kNoDevice (owns nothing) so only
    /// ONE guard ever restores. noexcept (standard §2.12). kNoDevice is referenced in
    /// the function body (complete-class scope), not the mem-initializer.
    DeviceGuard(DeviceGuard&& other) noexcept : dev_(other.dev_) {
        other.dev_ = kNoDevice;
    }
    DeviceGuard& operator=(DeviceGuard&& other) noexcept {
        if (this != &other) {
            restore();  // give back our own captured device before adopting other's
            dev_ = std::exchange(other.dev_, kNoDevice);
        }
        return *this;
    }

private:
    /// Sentinel for a moved-from (owns-nothing) guard: a negative ordinal is never a
    /// valid CUDA device, so restore() skips the cudaSetDevice.
    static constexpr int kNoDevice = -1;

    /// The ONE restore site. Non-throwing (standard §2.12): cudaSetDevice's
    /// [[nodiscard]] cudaError_t is surfaced via STEPPE_CUDA_WARN and (void)-discarded.
    void restore() noexcept {
        if (dev_ != kNoDevice) (void)STEPPE_CUDA_WARN(cudaSetDevice(dev_));
    }

    int dev_;  ///< Device ordinal to restore at scope exit; kNoDevice once moved-from.
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DEVICE_GUARD_CUH
