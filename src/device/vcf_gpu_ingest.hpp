// src/device/vcf_gpu_ingest.hpp
//
// The GPU-native phased-VCF ingest entry — the CUDA-free seam the app calls to run
// the nvcomp GPU-DEFLATE + GPU GT-parse + device-resident 2-bit pack pipeline in
// place of the CPU io::read_vcf_panel. It returns the SAME io::VcfPanelResult
// (SNP-major tile + inline SnpTable + pass counts), so the CLI's --emit-tile /
// --emit-hap-codes / --emit-snp emitters are byte-for-byte agnostic to which path
// produced the panel. This is the bit-exact invariance-gate contract: the GPU tile
// must equal the CPU reader's tile (and the bcftools oracle) on the same input.
//
// Declared CUDA-free (no CUDA types in the signature) and implemented in the
// steppe_device CUDA TU cuda/vcf_gpu_ingest.cu. When the build is configured
// WITHOUT nvcomp, vcf_gpu_available() returns false and read_vcf_panel_gpu throws
// a clear std::runtime_error — the symbols always exist, so the app links
// regardless of nvcomp availability.
#ifndef STEPPE_DEVICE_VCF_GPU_INGEST_HPP
#define STEPPE_DEVICE_VCF_GPU_INGEST_HPP

#include <string>

#include "io/vcf_panel_reader.hpp"

namespace steppe::device {

// True iff steppe_device was compiled with nvcomp (STEPPE_HAVE_NVCOMP). When
// false, read_vcf_panel_gpu throws rather than silently falling back to the CPU.
[[nodiscard]] bool vcf_gpu_available();

// Stream a phased .vcf.gz into a haplotype panel entirely on the GPU: nvcomp
// batched DEFLATE decompresses the BGZF blocks device-side, a CUDA kernel parses
// the phased GT matrix, and the 2-bit SNP-major tile is packed device-resident
// before a single D2H. `device_id` selects the CUDA device (default 0). Throws
// std::runtime_error on I/O failure, a malformed header, phase loss above
// opts.unphased_max, a non-BGZF input, or (when built without nvcomp) unavailability.
[[nodiscard]] io::VcfPanelResult read_vcf_panel_gpu(const std::string& vcf_path,
                                                    const io::VcfPanelOptions& opts,
                                                    int device_id = 0);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_VCF_GPU_INGEST_HPP
