// src/device/cuda/cuda_backend.cu
//
// CudaBackend — the GPU implementation of ComputeBackend (architecture.md §4, §8;
// ROADMAP §2, M0). Implements `compute_f2` via the f2 3-GEMM reformulation:
// upload the Q/V/N contract, run the fused feeder + three GEMMs + fused
// numerator/divide (f2_block_kernel.cu), and copy f2 + Vpair back across the
// CUDA-free ComputeBackend seam (architecture.md §4).
//
// STANDARDS (ROADMAP §1, §5):
//   * RAII for ALL device memory and handles — DeviceBuffer<T> + CublasHandle, no
//     raw cudaMalloc / cublasCreate here (architecture.md §2, §7). This TU is NOT
//     on the allocation allowlist.
//   * Precision is typed config: forwarded unchanged into run_f2_gemms, which
//     engages FIXED-slice Ozaki / native FP64 (architecture.md §12; ROADMAP §0).
//   * The numerator/divide stays native FP64 (in the assemble kernel).
//   * The formula lives ONCE in core/internal/f2_estimator.hpp, shared with the
//     CPU oracle, so CPU and GPU cannot diverge (architecture.md §13).
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It is the only
// place a host caller meets the GPU f2 path; `core` reaches it solely through the
// CUDA-free ComputeBackend interface in device/backend.hpp.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "device/backend.hpp"               // ComputeBackend, F2Result, MatView
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK, CUBLAS_CHECK
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (RAII)
#include "device/cuda/f2_block_kernel.cuh"  // launch_f2_feeder, run_f2_gemms, launch_assemble_f2
#include "device/cuda/handles.hpp"          // CublasHandle (RAII)
#include "steppe/config.hpp"                // Precision

namespace steppe::device {

namespace {

/// cuBLAS workspace size for the f2 GEMMs. An explicit workspace is REQUIRED for
/// run-to-run reproducibility of emulated FP64 (architecture.md §12; spike
/// f2_emu_spike.cu:765 used the same 64 MiB). Ample for these reduce-to-[P×P]
/// GEMMs; promoted out of the spike's bare literal into a named constant.
inline constexpr std::size_t kCublasWorkspaceBytes = 64u * 1024u * 1024u;

}  // namespace

/// GPU compute backend. The 3-GEMM f2 reformulation; one CublasHandle created
/// once (architecture.md §7) and reused, with its workspace set for emulated-FP64
/// determinism. Move-only via the ComputeBackend base (architecture.md §8).
class CudaBackend final : public ComputeBackend {
public:
    CudaBackend() {
        // RAII handle, created once; bind the workspace for emulated-FP64
        // bit-stability on the single statistic stream (architecture.md §12).
        CUBLAS_CHECK(cublasSetWorkspace(blas_.get(), workspace_.data(),
                                        workspace_.bytes()));
    }

    [[nodiscard]] F2Result compute_f2(const core::MatView& Q,
                                      const core::MatView& V,
                                      const core::MatView& N,
                                      const Precision& precision) override {
        const int P = Q.P;
        const long M = Q.M;
        const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        const std::size_t two_pm = 2u * pm;
        const std::size_t pp = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
        const std::size_t two_pp = 2u * pp;

        // ---- Device allocations (RAII; freed on scope exit) ------------------
        // Inputs (raw Q/V/N from the contract) + feeder outputs (masked Q, V,
        // stacked S=[Qsq;Hc]) + GEMM outputs (G, Vpair, R) + final f2. Only
        // [P×M], [2P×M], [P×P], [2P×P] buffers — never the [SNP×pop×pop]
        // intermediate (architecture.md §5 S2, §11.1).
        DeviceBuffer<double> dQ_raw(pm), dV_raw(pm), dN_raw(pm);
        DeviceBuffer<double> dQ(pm), dV(pm), dS(two_pm);
        DeviceBuffer<double> dG(pp), dVpair(pp), dR(two_pp), dF2(pp);

        // ---- Upload the Q/V/N contract (column-major [P × M]) ----------------
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ_raw.data(), Q.data, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV_raw.data(), V.data, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dN_raw.data(), N.data, pm * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_));

        // ---- Fused feeder -> 3 GEMMs -> fused numerator/divide ---------------
        launch_f2_feeder(dQ_raw.data(), dV_raw.data(), dN_raw.data(),
                         dQ.data(), dV.data(), dS.data(), P, M, stream_);
        run_f2_gemms(blas_.get(), precision, P, M,
                     dQ.data(), dV.data(), dS.data(),
                     dG.data(), dVpair.data(), dR.data(), stream_);
        launch_assemble_f2(dG.data(), dVpair.data(), dR.data(), dF2.data(), P, stream_);

        // ---- Copy results back across the CUDA-free seam --------------------
        F2Result out;
        out.P = P;
        out.f2.resize(pp);
        out.vpair.resize(pp);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.f2.data(), dF2.data(), pp * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.vpair.data(), dVpair.data(),
                                          pp * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return out;
    }

private:
    // Single statistic stream for bit-stability (architecture.md §12). The
    // default stream suffices at M0 (one f2 call at a time); a dedicated RAII
    // Stream lands with the streaming pipeline (architecture.md §11.1).
    cudaStream_t stream_ = nullptr;
    CublasHandle blas_{stream_};
    DeviceBuffer<std::byte> workspace_{kCublasWorkspaceBytes};
};

/// Factory for the GPU backend (architecture.md §9 — backend chosen at build()).
/// Returns the abstract interface so `core` / `Resources` never name the concrete
/// type or touch a CUDA header (architecture.md §4, §8).
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend() {
    return std::make_unique<CudaBackend>();
}

}  // namespace steppe::device
