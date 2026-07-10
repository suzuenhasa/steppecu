// src/io/bgzf_index.hpp
//
// Header-only BGZF block index — the ZERO-decompression walk that turns a
// slurped .vcf.gz into the list of its independent gzip-member blocks. Shared by
// the CPU phased-VCF reader (block-parallel zlib inflate) and the GPU phased-VCF
// reader (nvcomp batched DEFLATE): both need the exact same (coff, clen, xlen)
// per block, so the walk lives in ONE pure-host header rather than being copied.
//
// A BGZF .vcf.gz is a stream of INDEPENDENT gzip members (blocks), each <=64 KB
// uncompressed and carrying a `BC` extra subfield = BSIZE (the block's total
// compressed length minus 1). The GPU path additionally needs xlen (the gzip
// FEXTRA length) so it can slice the RAW deflate payload out of each block:
//   deflate payload = [coff + 12 + xlen, coff + clen - 8)   (skip gzip hdr+extra,
//                                                             drop the 8-byte trailer)
//   ISIZE (uncompressed size) = uint32 LE at [coff + clen - 4]
//
// Pure host C++20 io-leaf, standard library only — no CUDA, no zlib. Callers that
// need the actual bytes own the decompression (zlib on CPU, nvcomp on GPU).
#ifndef STEPPE_IO_BGZF_INDEX_HPP
#define STEPPE_IO_BGZF_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace steppe::io {

// One BGZF block: its byte offset in the compressed file, its compressed length
// (BSIZE+1), and xlen (the gzip FEXTRA field length, needed to locate the raw
// deflate payload for the GPU path).
struct BgzfBlock {
    std::size_t coff = 0;
    std::size_t clen = 0;
    std::size_t xlen = 0;
};

// Header-only BGZF walk (ZERO decompression). Reads each block's 12-byte gzip
// header + XLEN + extra field, locates the BC subfield (SI1='B',SI2='C',SLEN=2),
// reads BSIZE (uint16 LE), records (coff, BSIZE+1, xlen), and advances. Returns
// false (caller falls back to serial / a non-BGZF path) on ANY structural
// anomaly: not gzip, not deflate, no FEXTRA, no BC subfield (plain gzip), a
// truncated/over-running block, or trailing garbage. On success `out` holds every
// block including the 28-byte EOF marker, contiguously covering [0, size).
[[nodiscard]] inline bool scan_bgzf(const std::uint8_t* d, std::size_t size,
                                    std::vector<BgzfBlock>& out) {
    out.clear();
    std::size_t off = 0;
    while (off + 18 <= size) {
        if (d[off] != 0x1f || d[off + 1] != 0x8b) return false;   // gzip magic
        if (d[off + 2] != 8) return false;                        // CM = deflate
        if ((d[off + 3] & 0x04) == 0) return false;               // FLG.FEXTRA
        const std::size_t xlen =
            static_cast<std::size_t>(d[off + 10]) | (static_cast<std::size_t>(d[off + 11]) << 8);
        const std::size_t ex = off + 12;
        const std::size_t ex_end = ex + xlen;
        if (ex_end > size) return false;
        long bsize = -1;
        std::size_t p = ex;
        while (p + 4 <= ex_end) {
            const std::uint8_t si1 = d[p];
            const std::uint8_t si2 = d[p + 1];
            const std::size_t slen =
                static_cast<std::size_t>(d[p + 2]) | (static_cast<std::size_t>(d[p + 3]) << 8);
            if (si1 == 'B' && si2 == 'C' && slen == 2) {
                if (p + 6 > ex_end) return false;
                bsize = static_cast<long>(d[p + 4]) | (static_cast<long>(d[p + 5]) << 8);
                break;
            }
            p += 4 + slen;
        }
        if (bsize < 0) return false;                              // no BC -> plain gzip
        const std::size_t clen = static_cast<std::size_t>(bsize) + 1;
        if (off + clen > size) return false;                      // over-run
        out.push_back(BgzfBlock{off, clen, xlen});
        off += clen;
    }
    if (off != size) return false;                               // trailing garbage
    return !out.empty();
}

}  // namespace steppe::io

#endif  // STEPPE_IO_BGZF_INDEX_HPP
