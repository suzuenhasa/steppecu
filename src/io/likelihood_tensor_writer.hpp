// src/io/likelihood_tensor_writer.hpp
//
// Writes the self-describing, seekable, little-endian binary GL-tensor artifact
// (magic "STPGL1") from a host io::LikelihoodTile — so ancIBD / PCAngsd can later
// consume the normalized [n_site x n_sample x 3] likelihood tensor. Mirrors the
// existing binary-writer style (fixed header + section offset table + payload).
//
// Pure host C++20 io-leaf; failures surface as std::runtime_error.
#ifndef STEPPE_IO_LIKELIHOOD_TENSOR_WRITER_HPP
#define STEPPE_IO_LIKELIHOOD_TENSOR_WRITER_HPP

#include <string>

#include "io/likelihood_tile.hpp"

namespace steppe::io {

// Layout of the STPGL1 artifact (byte-exact):
//   [0]   char magic[8]      = "STPGL1\0\0"
//   [8]   u32  version       = 1
//   [12]  u8   layout        = 0 (site-major)
//   [13]  u8   field         = GlField (0=PL,1=GL,2=GP)
//   [14]  u8   dtype         = 0 (FP64)
//   [15]  u8   reserved      = 0
//   [16]  u64  n_site
//   [24]  u64  n_sample
//   [32]  u64  off_samples   (byte offset to the sample-id table)
//   [40]  u64  off_sites     (byte offset to the site table)
//   [48]  u64  off_payload   (byte offset to the FP64 payload)
//   [56]  u64  off_present   (byte offset to the u8 present mask)
//   = 64-byte header
//   samples: for each of n_sample -> u32 len + bytes
//   sites:   for each of n_site   -> u32 rsid_len + bytes, i32 chrom, i64 pos37,
//                                    i64 pos38, char a1, char a2
//   payload: n_site*n_sample*3 doubles (site-major, g = copies-of-A1)
//   present: n_site*n_sample u8
void write_likelihood_tensor(const std::string& path, const LikelihoodTile& tile);

}  // namespace steppe::io

#endif  // STEPPE_IO_LIKELIHOOD_TENSOR_WRITER_HPP
