// src/device/cuda/transpose_canonical_kernel.cuh
//
// Narrow launch-wrapper declaration for the M-FR-1 SNP-major -> canonical
// individual-major TRANSPOSE+GATHER+ENCODING primitive (the format-reader engine,
// docs/design/format-readers.md §2.4/M-FR-1). A SNP-major source (one record per
// SNP, individuals interleaved within each SNP's bytes — the PACKEDANCESTRYMAP /
// GENO axis) is gathered for a SELECTED, REORDERED set of individuals and packed
// into the canonical individual-major tile decode_af / detect_ploidy consume (one
// record per individual, SNP s at byte s/4 position s%4, MSB-first).
//
// ONE THREAD PER OUTPUT BYTE (g, byte b in [0, out_bytes_per_record)). Each thread
// assembles the 4 codes of output SNPs {4b, 4b+1, 4b+2, 4b+3} for gathered
// individual g into ONE output byte — so no two threads ever write the same byte
// (race-free WITHOUT atomics or a partial-bit OR). For each of the up-to-4 SNPs s:
//   src_row = d_sel_rows[g]                       (selection + pop-contiguous reorder)
//   code    = code @ snp-major byte s*src_bpr + src_row/4, position src_row%4
//   canon   = ENCODING(code)                      (P0 PACKEDANCESTRYMAP = IDENTITY)
// then pack canon into the output byte at position s%4 (MSB-first). SNPs s >= n_snp
// (the partial last byte) contribute 0 bits — exactly the host packer's behavior.
//
// rlen-floor (GENO max(48, ceil(n_ind/4))): individuals are bounded by the EXPLICIT
// src_rows (each < n_ind), so a small-n_ind record's PADDING bytes beyond
// ceil(n_ind/4) are never read as phantom individuals (format-readers.md §3.4).
//
// This header names a CUDA type (cudaStream_t) and so is PRIVATE to steppe_device
// (architecture.md §4) — the device-internal seam between the backend and the
// transpose kernel TU, NOT the CUDA-free public ComputeBackend seam (backend.hpp).
// The kernel body and `<<<>>>` live only in transpose_canonical_kernel.cu.
#ifndef STEPPE_DEVICE_CUDA_TRANSPOSE_CANONICAL_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_TRANSPOSE_CANONICAL_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace steppe::device {

/// Encoding map applied to each source 2-bit code before it is packed into the
/// canonical tile (format-readers.md §3.2). P0 PACKEDANCESTRYMAP/GENO shares the
/// canonical raw-value convention (0/1/2 ref-allele copies, 3 missing) and the
/// MSB-first bit order, so its map is IDENTITY. Later formats (EIGENSTRAT ASCII,
/// PLINK) plug a different enumerator WITHOUT touching the transpose body.
enum class TransposeEncoding : int {
    Identity = 0,  ///< P0 PACKEDANCESTRYMAP/GENO — native code IS the canonical code.
};

/// SNP-major -> canonical individual-major transpose + GATHER + ENCODING on the GPU.
/// One thread per output byte; writes the canonical individual-major `d_out`.
///
///   d_snp_major     [n_snp_records x src_bytes_per_record] SNP-major source bytes
///                   (record s = SNP s; individual i at byte i/4, position i%4).
///   src_bytes_per_record  stride between SNP records in the source (the GENO
///                   max(48, ceil(n_ind/4)) rlen-floored stride — may exceed
///                   ceil(n_ind/4); the kernel reads only src_row/4 < that).
///   d_sel_rows      [n_individuals] source individual ROW per output column g, in
///                   pop-contiguous Q/V/N order (the IndPartition selection+reorder).
///   n_individuals   number of gathered output individuals (the output record count).
///   n_snp           SNPs the tile covers (the output column axis; caps the codes).
///   out_bytes_per_record  ceil(n_snp/4) — the canonical output record stride.
///   encoding        the native-code -> canonical-code map (P0 = Identity).
///   d_out           [n_individuals x out_bytes_per_record] canonical tile (output).
void launch_transpose_to_canonical(const std::uint8_t* d_snp_major,
                                   std::size_t src_bytes_per_record,
                                   const std::size_t* d_sel_rows,
                                   std::size_t n_individuals, std::size_t n_snp,
                                   std::size_t out_bytes_per_record,
                                   TransposeEncoding encoding,
                                   std::uint8_t* d_out, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_TRANSPOSE_CANONICAL_KERNEL_CUH
