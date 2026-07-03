// src/device/cuda/cuda_backend.cu
//
// CudaBackend — the GPU implementation of the CUDA-free ComputeBackend interface,
// and the one place a host caller meets the GPU f2 path. Kept in a single
// translation unit because a C++ class cannot be split across TUs; the heavy
// compute methods live in companion files, this one holds construction, the
// capability probe, the factory helpers, and the device-fault translator.
//
// Reference: docs/reference/src_device_cuda_cuda_backend.cu.md
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/internal/pchisq.hpp"
#include "core/internal/nvtx.hpp"
#include "core/internal/qpfstats_jackknife.hpp"
#include "core/internal/small_linalg.hpp"
#include "core/qpadm/qpadm_bounds.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "device/resources.hpp"
#include "device/device_partial.hpp"
#include "device/cuda/device_partial_impl.cuh"
#include "device/device_f2_blocks.hpp"
#include "device/cuda/device_f2_blocks_impl.cuh"
#include "device/device_decode_result.hpp"
#include "device/cuda/device_decode_result_impl.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/decode_af_kernel.cuh"
#include "device/cuda/detect_ploidy_kernel.cuh"
#include "device/cuda/transpose_canonical_kernel.cuh"
#include "device/cuda/decode_compact_kernel.cuh"
#include "device/cuda/dstat_kernel.cuh"
#include "device/cuda/dates_kernel.cuh"
#include "device/cuda/qpfstats_kernel.cuh"
#include "device/cuda/qpfstats_jackknife_kernel.cuh"
#include "device/cuda/ratio_block_jackknife_kernel.cuh"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/f2_block_kernel.cuh"
#include "device/cuda/f2_batched_kernel.cuh"
#include "device/cuda/block_sink.cuh"
#include "device/stream_f2_blocks.hpp"
#include "device/f2_blocks_out.hpp"
#include "device/cuda/handles.hpp"
#include "device/cuda/qpadm_fit_kernels.cuh"
#include "device/cuda/qpgraph_fit_kernels.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "device/cuda/stream.hpp"
#include "device/vram_budget.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"
#include "device/cuda/cuda_backend.cuh"

namespace steppe::device {

// Construction: binding the workspace and stream — reference §4
CudaBackend::CudaBackend(int device_id)
    : device_id_(set_and_return_device(device_id)) {
    blas_.set_workspace(workspace_.data(), workspace_.bytes());
    blas_.set_stream(stream_.get());
    solver_.set_stream(stream_.get());
}

// capabilities(): probing the bound GPU — reference §5
BackendCapabilities CudaBackend::capabilities() const {
    BackendCapabilities caps;

    int count = 0;
    STEPPE_CUDA_CHECK(cudaGetDeviceCount(&count));
    caps.device_count = count;

    int entry_device = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&entry_device));
    STEPPE_CUDA_CHECK(cudaSetDevice(device_id_));

    cudaDeviceProp prop{};
    STEPPE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    caps.compute_major = prop.major;
    caps.compute_minor = prop.minor;

    std::size_t free_b = 0, total_b = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    caps.free_vram_bytes = free_b;
    caps.total_vram_bytes = total_b;

    bool can_peer = false;
    for (int peer = 0; peer < count; ++peer) {
        if (peer == device_id_) continue;
        int access = 0;
        const cudaError_t s =
            STEPPE_CUDA_WARN(cudaDeviceCanAccessPeer(&access, device_id_, peer));
        if (s == cudaSuccess && access != 0) {
            can_peer = true;
            break;
        }
    }
    caps.can_access_peer = can_peer;

    const Precision emu_probe{Precision::Kind::EmulatedFp64,
                              steppe::kDefaultMantissaBits};
    caps.emulated_fp64_honorable = emulation_honorable(emu_probe);

    STEPPE_CUDA_CHECK(cudaSetDevice(entry_device));
    return caps;
}

void CudaBackend::set_solve_precision(const Precision& precision) {
    solve_precision_ = precision;
}

std::size_t CudaBackend::batched_dispatch_count() const {
    return batched_dispatch_count_;
}

// Factory functions — reference §8
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(int device_id) {
    return std::make_unique<CudaBackend>(device_id);
}

[[nodiscard]] int visible_device_count() {
    int count = 0;
    STEPPE_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

// Translating device faults to exit codes — reference §9
[[nodiscard]] std::optional<Status> device_fault_status(
    const std::exception& e) noexcept {
    if (const auto* ce = dynamic_cast<const CudaError*>(&e)) {
        return ce->status() == cudaErrorMemoryAllocation
                   ? std::optional<Status>(Status::DeviceOom)
                   : std::nullopt;
    }
    if (const auto* be = dynamic_cast<const CublasError*>(&e)) {
        return be->status() == CUBLAS_STATUS_ALLOC_FAILED
                   ? std::optional<Status>(Status::DeviceOom)
                   : std::nullopt;
    }
    if (const auto* se = dynamic_cast<const CusolverError*>(&e)) {
        return se->status() == CUSOLVER_STATUS_ALLOC_FAILED
                   ? std::optional<Status>(Status::DeviceOom)
                   : std::nullopt;
    }
    return std::nullopt;
}

}  // namespace steppe::device
