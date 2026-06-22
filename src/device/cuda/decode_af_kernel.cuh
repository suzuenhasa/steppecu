// src/device/cuda/decode_af_kernel.cuh
//
// Narrow launch-wrapper declaration for the GPU genotype decode → allele-freq
// reduction (architecture.md §5 S0 Format decode + S1 Allele-freq reduction, §7
// "host code never includes kernel bodies or <<<>>>"; ROADMAP M1). Host
// orchestration (cuda_backend.cu) calls this `void launch_decode_af(...)`; the
// kernel body and `<<<>>>` live only in decode_af_kernel.cu.
//
// This header names a CUDA type (cudaStream_t) and so is PRIVATE to steppe_device
// (architecture.md §4) — the device-internal seam between the backend and the
// decode kernel TU, not the CUDA-free public ComputeBackend seam (backend.hpp).
#ifndef STEPPE_DEVICE_CUDA_DECODE_AF_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DECODE_AF_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

/// Decode a packed TGENO tile (individual-major, pop-contiguous) into the Q/V/N
/// contract on the GPU (architecture.md §5 S0/S1, §11.3 bandwidth-bound). One
/// thread owns one (population p, SNP s) output entry: it segmented-reduces over
/// the individuals of population p — unpacking SNP s from each record's byte
/// (2-bit raw-value mapping via the SHARED core primitive) — into integer AC
/// (ref-allele copies) / AN (non-missing individuals), then a single FP64 divide
/// yields Q = AC/(ploidy·AN), N = ploidy·AN, V = (AN>0). Coalesced on the SNP
/// axis (adjacent threads read adjacent record bytes). All native FP64 / integer.
///
///   d_packed       [n_individuals × bytes_per_record] bytes, pop-contiguous
///   d_pop_offsets  [P+1] segment boundaries over the individual axis
///   d_Q,d_V,d_N    [P × M] column-major outputs (element (i,s) at i + P·s)
///
/// `M` is the SNP count (column axis). `d_sample_ploidy` is the PER-SAMPLE ploidy
/// vector (device, length n_individuals, parallel to the gathered sample axis; 2
/// diploid / 1 pseudo-haploid, AT2-auto-detected upstream). When non-null the kernel
/// folds it via the AT2 adjust_pseudohaploid accumulation (AC += code/(3-ploidy),
/// N += ploidy); when NULL it falls back to the uniform scalar `ploidy` for every
/// sample (the legacy all-diploid path, bit-identical).
void launch_decode_af(const std::uint8_t* d_packed,
                      std::size_t bytes_per_record,
                      const std::size_t* d_pop_offsets,
                      int P, long M, int ploidy,
                      const int* d_sample_ploidy,
                      double* d_Q, double* d_V, double* d_N,
                      cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DECODE_AF_KERNEL_CUH
