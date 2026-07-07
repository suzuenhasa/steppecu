// src/device/cuda/cuda_backend_likelihood.cu
//
// Out-of-line CudaBackend bodies for the two GL-tensor seams: upload the host
// [n_site x n_sample x 3] FP64 likelihood tile (+ present mask) into a resident
// DeviceBuffer, and a device reduction (grid-stride block sum + atomicAdd) whose
// result — compared to the host sum — proves the tensor is genuinely on-device.
// A CUDA TU private to the steppe_device target. Mirrors cuda_backend_readv2.cu.
#include <cstddef>
#include <cstdint>

#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/likelihood_tensor_impl.cuh"

namespace steppe::device {

namespace {

// Grid-stride block reduction of `l[0..n)` into a single device double via
// atomicAdd. Reads the resident payload through its device pointer — so a nonzero
// result IS the residency proof (a kernel consumed the device buffer).
__global__ void likelihood_sum_kernel(const double* __restrict__ l, long n, double* __restrict__ acc) {
    extern __shared__ double sdata[];
    const unsigned tid = threadIdx.x;
    double local = 0.0;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + tid; i < n;
         i += static_cast<long>(gridDim.x) * blockDim.x) {
        local += l[i];
    }
    sdata[tid] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(acc, sdata[0]);
}

}  // namespace

LikelihoodTensor CudaBackend::upload_likelihood_tensor(const double* host_l,
                                                       const std::uint8_t* host_present,
                                                       long n_site, int n_sample) {
    guard_device();
    STEPPE_NVTX_RANGE("upload_likelihood_tensor");
    LikelihoodTensor t;
    t.n_site = n_site;
    t.n_sample = n_sample;
    t.device_id = device_id_;
    if (n_site <= 0 || n_sample <= 0) return t;

    const std::size_t cells = static_cast<std::size_t>(n_site) * static_cast<std::size_t>(n_sample);
    const std::size_t n_l = cells * 3;
    t.impl = std::make_unique<LikelihoodTensor::Impl>();
    t.impl->l = DeviceBuffer<double>(n_l);
    t.impl->present = DeviceBuffer<std::uint8_t>(cells);
    h2d_async(t.impl->l, host_l, n_l, stream_.get());
    h2d_async(t.impl->present, host_present, cells, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));  // host buffers reusable after return
    return t;
}

double CudaBackend::likelihood_tensor_checksum(const LikelihoodTensor& t) {
    guard_device();
    STEPPE_NVTX_RANGE("likelihood_tensor_checksum");
    if (!t.impl || t.n_elem() <= 0) return 0.0;
    const long n = t.n_elem();

    DeviceBuffer<double> d_acc(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(d_acc.data(), 0, sizeof(double), stream_.get()));
    constexpr int kBlock = 256;
    long blocks_l = (n + kBlock - 1) / kBlock;
    if (blocks_l > 65535) blocks_l = 65535;  // grid-stride mops up the remainder
    const int blocks = static_cast<int>(blocks_l);
    likelihood_sum_kernel<<<blocks, kBlock, kBlock * sizeof(double), stream_.get()>>>(
        t.impl->l.data(), n, d_acc.data());
    STEPPE_CUDA_CHECK(cudaGetLastError());

    double host_sum = 0.0;
    d2h_async(&host_sum, d_acc, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return host_sum;
}

}  // namespace steppe::device
