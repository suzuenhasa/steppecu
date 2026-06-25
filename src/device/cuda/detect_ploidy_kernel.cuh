// src/device/cuda/detect_ploidy_kernel.cuh
//
// Narrow launch-wrapper declaration for the GPU AT2 pseudo-haploid PER-SAMPLE
// PLOIDY prepass (M-FR-0, the L2 host-compute fix; the on-device port of
// io::detect_sample_ploidy). One thread owns one gathered individual: it scans the
// first min(kPloidyDetectSnps, n_snp) SNPs of that individual's packed record for a
// HETEROZYGOUS call (core::genotype_code == kHeterozygousGenotypeCode) and writes 2
// (diploid) on the first het, else 1 (pseudo-haploid). A literal bit-parity port of
// the host loop (src/io/ploidy_detect.cpp) over the SAME shared core decode
// primitives (core/internal/decode_af.hpp), so the device vector is bit-identical to
// the host detector by construction (integer/bit ops only; no FP, no precision lane).
//
// This header names a CUDA type (cudaStream_t) and so is PRIVATE to steppe_device
// (architecture.md §4) — the device-internal seam between the backend and the
// prepass kernel TU, not the CUDA-free public ComputeBackend seam (backend.hpp). The
// kernel body and `<<<>>>` live only in detect_ploidy_kernel.cu.
#ifndef STEPPE_DEVICE_CUDA_DETECT_PLOIDY_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DETECT_PLOIDY_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

/// AT2 pseudo-haploid per-sample ploidy detection on the GPU (the L2 on-device
/// prepass). One thread per gathered individual g in [0, n_individuals): scan the
/// first `min(kPloidyDetectSnps, n_snp)` SNPs of g's packed record (the SAME
/// individual-major byte layout decode_af reads — SNP s at byte s/4, position s%4,
/// MSB-first via core::genotype_code) and write d_ploidy[g] = 2 (diploid) on the
/// FIRST het call (code == kHeterozygousGenotypeCode), else 1 (pseudo-haploid).
/// Integer/bit ops only — bit-identical to the host io::detect_sample_ploidy loop.
///
///   d_packed          [n_individuals × bytes_per_record] bytes, pop-contiguous
///   bytes_per_record  stride between individual records
///   n_individuals     number of gathered individuals (the prepass axis)
///   n_snp             SNPs the tile covers (caps the detection window)
///   d_ploidy          [n_individuals] output (2 diploid / 1 pseudo-haploid)
void launch_detect_ploidy(const std::uint8_t* d_packed,
                          std::size_t bytes_per_record,
                          std::size_t n_individuals, std::size_t n_snp,
                          int* d_ploidy, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DETECT_PLOIDY_KERNEL_CUH
