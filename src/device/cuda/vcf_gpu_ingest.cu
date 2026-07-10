// src/device/cuda/vcf_gpu_ingest.cu
//
// GPU-native phased-VCF ingest: nvcomp batched DEFLATE decompresses the BGZF
// blocks device-side, a CUDA kernel parses the phased GT matrix, and the 2-bit
// SNP-major tile is packed device-resident. The panel it returns is BYTE-IDENTICAL
// to the CPU io::read_vcf_panel (the invariance gate) — the same {0,2,3} code map,
// the same biallelic/region/dup/`norm -d all` decisions, the same SNP-major pack.
//
// Pipeline (per block-batch, streamed so full-genome text never has to be resident
// and the region break stops early exactly like the CPU reader):
//   1. HOST  scan_bgzf -> (coff,clen,xlen) per block (zero decompression); derive
//            each block's raw-deflate payload slice + ISIZE (uncompressed size,
//            the block trailer's last 4 bytes) with ZERO device work.
//   2. GPU   nvcompBatchedDeflateDecompressAsync inflates the batch's blocks into
//            per-chunk (4-byte-aligned) device slots; a compaction kernel gathers
//            them into ONE contiguous device text buffer (carry-prefixed so a VCF
//            line straddling the previous batch is whole here).
//   3. HOST  parse the batch text (D2H) line-by-line, mirroring the CPU serial
//            decision loop EXACTLY (bounded metadata tokenize; region/dup/biallelic
//            filters; SnpTable + map-join) -> the emitted records' (line offset,
//            GT FORMAT index). The trailing partial line carries to the next batch.
//   4. GPU   kernel A builds each emitted record's column-start index from the
//            device text; kernel B (one thread per (record, tile byte)) decodes the
//            phased GT of its 2 samples and packs a full 2-bit byte device-resident.
//   5. D2H   append the batch's packed tile rows to the panel's snp_major bytes.
//
// The GT decode (dev_hap_code + the phased/unphased split) re-implements
// io::hap_code and the CPU per-sample rule BYTE-FOR-BYTE. A CUDA TU private to
// steppe_device; the CUDA-free entry is device/vcf_gpu_ingest.hpp.
#include "device/vcf_gpu_ingest.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace steppe::device {

bool vcf_gpu_available() {
#ifdef STEPPE_HAVE_NVCOMP
    return true;
#else
    return false;
#endif
}

}  // namespace steppe::device

#ifndef STEPPE_HAVE_NVCOMP

namespace steppe::device {

io::VcfPanelResult read_vcf_panel_gpu(const std::string&, const io::VcfPanelOptions&, int) {
    throw std::runtime_error(
        "steppe was built without nvcomp; the GPU phased-VCF path is unavailable "
        "(configure steppe_device with nvcomp to enable STEPPE_VCF_GPU / --gpu)");
}

}  // namespace steppe::device

#else  // STEPPE_HAVE_NVCOMP ===================================================

#include <cuda_runtime.h>
#include <nvcomp/deflate.h>

#include "device/cuda/check.cuh"
#include "device/cuda/device_buffer.cuh"
#include "io/bgzf_index.hpp"
#include "io/eigenstrat_format.hpp"
#include "io/vcf_panel_decode.hpp"
#include "io/vcf_record.hpp"

namespace steppe::device {

namespace {

namespace vd = io::vcfdetail;

// ---------------------------------------------------------------------------
// Device-side GT decode — the byte-for-byte mirror of io::hap_code + the CPU
// per-sample phased/unphased rule (vcf_panel_reader serial loop).
// ---------------------------------------------------------------------------

// io::hap_code on-device: single-char allele '0'->0, '1'->2, else 3 (missing).
__device__ inline std::uint8_t dev_hap_code(const char* a, int len) {
    if (len == 1) {
        if (a[0] == '0') return 0u;
        if (a[0] == '1') return 2u;
    }
    return 3u;  // io::kMissingCode
}

// vd::subfield on-device: the fi-th ':'-delimited token of [field,field+flen).
// fi<0 or out-of-range -> empty (len 0), matching vd::subfield.
__device__ inline void dev_subfield(const char* field, int flen, int fi,
                                    const char** out, int* outlen) {
    if (fi < 0) { *out = field; *outlen = 0; return; }
    int cur = 0;
    int start = 0;
    for (int i = 0; i <= flen; ++i) {
        if (i == flen || field[i] == ':') {
            if (cur == fi) { *out = field + start; *outlen = i - start; return; }
            if (i == flen) break;
            start = i + 1;
            ++cur;
        }
    }
    *out = field + flen;
    *outlen = 0;
}

// Decode one sample's GT field into two 2-bit haplotype codes + the per-sample
// unphased/half/missing sub-counts, exactly as the CPU serial loop does.
__device__ inline void decode_gt(const char* field, int flen, int fi, std::uint8_t* h0o,
                                 std::uint8_t* h1o, int* unph, int* half, int* miss) {
    const char* gtv;
    int gtlen;
    dev_subfield(field, flen, fi, &gtv, &gtlen);

    std::uint8_t h0 = 3u, h1 = 3u;
    int u = 0, hf = 0;

    bool phased = false;
    for (int i = 0; i < gtlen; ++i) {
        if (gtv[i] == '|') { phased = true; break; }
    }
    const char sep = phased ? '|' : '/';

    int sp = -1;
    for (int i = 0; i < gtlen; ++i) {
        if (gtv[i] == sep) { sp = i; break; }
    }
    if (sp >= 0) {
        int sp2 = -1;
        for (int i = sp + 1; i < gtlen; ++i) {
            if (gtv[i] == sep) { sp2 = i; break; }
        }
        if (sp2 < 0) {  // exactly two tokens (ga.size()==2)
            const char* a0 = gtv;
            const int l0 = sp;
            const char* a1 = gtv + sp + 1;
            const int l1 = gtlen - sp - 1;
            bool eq = (l0 == l1);
            for (int i = 0; eq && i < l0; ++i) eq = (a0[i] == a1[i]);
            const bool unphased_het = (!phased) && !eq;
            if (unphased_het) {
                u = 1;  // both haps stay missing
            } else {
                h0 = dev_hap_code(a0, l0);
                h1 = dev_hap_code(a1, l1);
                const bool q0 = (h0 == 3u);
                const bool q1 = (h1 == 3u);
                if (q0 != q1) hf = 1;
            }
        }
        // sp2>=0 -> >2 tokens -> ga.size()!=2 -> both missing (no sub-counts)
    }
    // sp<0 -> 1 token -> ga.size()!=2 -> both missing

    *h0o = h0;
    *h1o = h1;
    *unph = u;
    *half = hf;
    *miss = (h0 == 3u ? 1 : 0) + (h1 == 3u ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Gather each decompressed chunk (padded/aligned slot in `out`) into the
// contiguous text buffer at `text_off[c]`. One CUDA block per chunk.
__global__ void compact_kernel(const std::uint8_t* __restrict__ out,
                               const std::size_t* __restrict__ out_off,
                               const std::size_t* __restrict__ text_off,
                               const std::size_t* __restrict__ isize, long n_chunks,
                               std::uint8_t* __restrict__ text) {
    const long c = blockIdx.x;
    if (c >= n_chunks) return;
    const std::uint8_t* src = out + out_off[c];
    std::uint8_t* dst = text + text_off[c];
    const std::size_t n = isize[c];
    for (std::size_t j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[j];
}

// For each emitted record, index the start offset (line-local) of every column
// into colstart[i*(n_col+1) + j]; colstart[n_col] is the separator that ends the
// last needed column (a tab if more columns follow, else the newline / buffer end),
// so column j spans [colstart[j], colstart[j+1]-1) uniformly.
__global__ void colscan_kernel(const char* __restrict__ text,
                               const std::uint64_t* __restrict__ rec_off, long n_emit,
                               std::size_t text_size, int n_col,
                               std::uint32_t* __restrict__ colstart) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_emit) return;
    const std::size_t base = rec_off[i];
    std::uint32_t* cs = colstart + static_cast<std::size_t>(i) * (n_col + 1);
    cs[0] = 0u;
    int col = 0;
    std::size_t j = base;
    for (; j < text_size; ++j) {
        const char c = text[j];
        if (c == '\n') { cs[n_col] = static_cast<std::uint32_t>(j + 1 - base); return; }
        if (c == '\t') {
            ++col;
            if (col < n_col) {
                cs[col] = static_cast<std::uint32_t>(j + 1 - base);
            } else if (col == n_col) {
                cs[n_col] = static_cast<std::uint32_t>(j + 1 - base);
                return;
            }
        }
    }
    // Ran to the buffer end without a newline (final line, no trailing '\n'):
    // treat end-of-buffer as the last column's terminator.
    cs[n_col] = static_cast<std::uint32_t>(text_size - base) + 1u;
}

// One thread per (emitted record, output tile byte). A byte holds 4 haplotype
// codes = 2 samples; the thread decodes both samples' GT and writes the packed
// byte (no atomics on the tile — one writer per byte). Per-sample sub-counts are
// atomically summed into the 3 global counters.
__global__ void gtpack_kernel(const char* __restrict__ text,
                              const std::uint64_t* __restrict__ rec_off,
                              const std::uint32_t* __restrict__ colstart,
                              const int* __restrict__ gt_fi, long n_emit, int n_sample,
                              int n_col, long src_bpr, std::uint8_t* __restrict__ tile,
                              unsigned long long* __restrict__ counters) {
    const long total = n_emit * src_bpr;
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const long i = tid / src_bpr;   // record
    const long b = tid % src_bpr;   // tile byte within the record

    const std::size_t base = rec_off[i];
    const std::uint32_t* cs = colstart + static_cast<std::size_t>(i) * (n_col + 1);
    const int fi = gt_fi[i];

    std::uint8_t out = 0u;
    unsigned long long unph = 0, half = 0, miss = 0;
    // This byte covers haplotypes [4b, 4b+4) => samples s0=2b, s1=2b+1.
    for (int hh = 0; hh < 2; ++hh) {
        const int s = static_cast<int>(2 * b) + hh;
        if (s >= n_sample) continue;  // padding slots stay 0 (matches the zero-init pack)
        const int colA = 9 + s;
        const std::uint32_t fs = cs[colA];
        const std::uint32_t fe = cs[colA + 1] - 1u;  // drop the separator
        const char* field = text + base + fs;
        const int flen = static_cast<int>(fe - fs);
        std::uint8_t h0, h1;
        int su, sh, sm;
        decode_gt(field, flen, fi, &h0, &h1, &su, &sh, &sm);
        unph += su;
        half += sh;
        miss += sm;
        const int slot0 = (4 * static_cast<int>(b) + 2 * hh) & 3;  // = 2*hh
        out = static_cast<std::uint8_t>(out | (h0 << ((3 - slot0) * 2)));
        out = static_cast<std::uint8_t>(out | (h1 << ((3 - (slot0 + 1)) * 2)));
    }
    tile[static_cast<std::size_t>(i) * src_bpr + b] = out;
    if (unph) atomicAdd(&counters[0], unph);
    if (half) atomicAdd(&counters[1], half);
    if (miss) atomicAdd(&counters[2], miss);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

[[nodiscard]] std::vector<std::uint8_t> slurp_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("read_vcf_panel_gpu: cannot open VCF file: " + path);
    const std::streamoff n = in.tellg();
    if (n < 0) throw std::runtime_error("read_vcf_panel_gpu: cannot size VCF file: " + path);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(n));
    in.seekg(0);
    if (n > 0 && !in.read(reinterpret_cast<char*>(buf.data()), n)) {
        throw std::runtime_error("read_vcf_panel_gpu: short read on VCF file: " + path);
    }
    return buf;
}

[[nodiscard]] std::size_t align_up(std::size_t x, std::size_t a) {
    return (x + a - 1) / a * a;
}

// One decompressed block-batch resident on the device as one contiguous text
// buffer (carry-prefixed). Carries the raw device text + its host mirror.
struct BatchText {
    DeviceBuffer<std::uint8_t> d_text;   // carry ++ decompressed blocks, contiguous
    std::string host;                    // D2H mirror (record offsets index into this)
    std::size_t size = 0;
};

// Decompress blocks [b0,b1) of `filebuf` (BGZF payload slices from `blocks`) with
// nvcomp, prefixing `carry` bytes, into ONE contiguous device text buffer + host
// mirror. Alignment: input/output chunks are padded to kAlign (>= nvcomp's 4-byte
// requirement).
[[nodiscard]] BatchText decompress_batch(const std::vector<std::uint8_t>& filebuf,
                                         const std::vector<io::BgzfBlock>& blocks,
                                         std::size_t b0, std::size_t b1,
                                         const std::string& carry, cudaStream_t stream) {
    constexpr std::size_t kAlign = 16;

    std::vector<const void*> h_comp_ptrs;      // filled after d_comp base is known
    std::vector<std::size_t> h_comp_off, h_comp_len, h_isize, h_out_off, h_text_off;
    std::vector<std::uint8_t> comp_packed;     // aligned host staging of deflate payloads
    const std::size_t carry_len = carry.size();
    std::size_t out_cursor = 0;
    std::size_t text_cursor = carry_len;       // text offsets start past the carry prefix

    for (std::size_t bi = b0; bi < b1; ++bi) {
        const io::BgzfBlock& blk = blocks[bi];
        const std::size_t hdr = 12 + blk.xlen;
        if (blk.clen < hdr + 8) continue;                       // malformed/too small; skip
        const std::size_t payload_len = blk.clen - hdr - 8;
        const std::size_t isize = static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 4]) |
                                  (static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 3]) << 8) |
                                  (static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 2]) << 16) |
                                  (static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 1]) << 24);
        if (isize == 0) continue;                               // BGZF EOF marker / empty block

        const std::size_t co = align_up(comp_packed.size(), kAlign);
        comp_packed.resize(co + payload_len);
        std::memcpy(comp_packed.data() + co, filebuf.data() + blk.coff + hdr, payload_len);
        h_comp_off.push_back(co);
        h_comp_len.push_back(payload_len);
        h_isize.push_back(isize);
        const std::size_t oo = align_up(out_cursor, kAlign);
        h_out_off.push_back(oo);
        out_cursor = oo + isize;
        h_text_off.push_back(text_cursor);
        text_cursor += isize;
    }

    const std::size_t n_chunks = h_isize.size();
    BatchText bt;
    bt.size = text_cursor;                                      // carry + sum(isize)
    if (n_chunks == 0) {                                        // nothing decompressible
        bt.host = carry;
        bt.d_text = DeviceBuffer<std::uint8_t>(bt.size == 0 ? 1 : bt.size);
        if (carry_len) h2d_async(bt.d_text, carry.data(), carry_len, stream);
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
        return bt;
    }

    DeviceBuffer<std::uint8_t> d_comp(comp_packed.size());
    h2d_async(d_comp, comp_packed.data(), comp_packed.size(), stream);
    DeviceBuffer<std::uint8_t> d_out(out_cursor == 0 ? 1 : out_cursor);
    bt.d_text = DeviceBuffer<std::uint8_t>(bt.size == 0 ? 1 : bt.size);
    if (carry_len) h2d_async(bt.d_text, carry.data(), carry_len, stream);

    h_comp_ptrs.resize(n_chunks);
    std::vector<void*> h_out_ptrs(n_chunks);
    for (std::size_t i = 0; i < n_chunks; ++i) {
        h_comp_ptrs[i] = d_comp.data() + h_comp_off[i];
        h_out_ptrs[i] = d_out.data() + h_out_off[i];
    }

    DeviceBuffer<const void*> d_comp_ptrs(n_chunks);
    DeviceBuffer<void*> d_out_ptrs(n_chunks);
    DeviceBuffer<std::size_t> d_comp_bytes(n_chunks);
    DeviceBuffer<std::size_t> d_isize(n_chunks);
    DeviceBuffer<std::size_t> d_out_off(n_chunks);
    DeviceBuffer<std::size_t> d_text_off(n_chunks);
    DeviceBuffer<nvcompStatus_t> d_status(n_chunks);
    h2d_async(d_comp_ptrs, h_comp_ptrs.data(), n_chunks, stream);
    h2d_async(d_out_ptrs, h_out_ptrs.data(), n_chunks, stream);
    h2d_async(d_comp_bytes, h_comp_len.data(), n_chunks, stream);
    h2d_async(d_isize, h_isize.data(), n_chunks, stream);
    h2d_async(d_out_off, h_out_off.data(), n_chunks, stream);
    h2d_async(d_text_off, h_text_off.data(), n_chunks, stream);

    std::size_t temp_bytes = 0;
    const nvcompStatus_t ts = nvcompBatchedDeflateDecompressGetTempSizeAsync(
        n_chunks, 65536, nvcompBatchedDeflateDecompressDefaultOpts, &temp_bytes, text_cursor);
    if (ts != nvcompSuccess) {
        throw std::runtime_error("read_vcf_panel_gpu: nvcompBatchedDeflateDecompressGetTempSize failed");
    }
    DeviceBuffer<std::uint8_t> d_temp(temp_bytes == 0 ? 1 : temp_bytes);

    // Actual-decompressed-sizes array: REQUIRED by the hardware-decompress backend,
    // which the DEFAULT backend may select on Blackwell — so always provide it.
    DeviceBuffer<std::size_t> d_actual(n_chunks);
    const nvcompStatus_t ds = nvcompBatchedDeflateDecompressAsync(
        d_comp_ptrs.data(), d_comp_bytes.data(), d_isize.data(), d_actual.data(), n_chunks,
        d_temp.data(), temp_bytes, d_out_ptrs.data(), nvcompBatchedDeflateDecompressDefaultOpts,
        d_status.data(), stream);
    if (ds != nvcompSuccess) {
        throw std::runtime_error(
            "read_vcf_panel_gpu: nvcompBatchedDeflateDecompressAsync launch failed (status " +
            std::to_string(static_cast<int>(ds)) + ")");
    }

    // Verify every chunk decompressed cleanly.
    std::vector<nvcompStatus_t> h_status(n_chunks);
    d2h_async(h_status.data(), d_status, n_chunks, stream);
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < n_chunks; ++i) {
        if (h_status[i] != nvcompSuccess) {
            throw std::runtime_error("read_vcf_panel_gpu: nvcomp reported a DEFLATE decode error");
        }
    }

    // Gather the padded per-chunk outputs into the contiguous text buffer.
    compact_kernel<<<static_cast<unsigned>(n_chunks), 256, 0, stream>>>(
        d_out.data(), d_out_off.data(), d_text_off.data(), d_isize.data(),
        static_cast<long>(n_chunks), bt.d_text.data());
    STEPPE_CUDA_CHECK_KERNEL();

    bt.host.resize(bt.size);
    if (bt.size) d2h_async(bt.host.data(), bt.d_text, bt.size, stream);
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
    return bt;
}

}  // namespace

io::VcfPanelResult read_vcf_panel_gpu(const std::string& vcf_path,
                                      const io::VcfPanelOptions& opts, int device_id) {
    STEPPE_CUDA_CHECK(cudaSetDevice(device_id));
    cudaStream_t stream = nullptr;
    STEPPE_CUDA_CHECK(cudaStreamCreate(&stream));

    io::VcfPanelResult out;
    unsigned long long h_counters[3] = {0, 0, 0};  // unphased_het, half_missing, missing
    try {
        // --- host: slurp + BGZF block index (zero decompression) --------------
        std::vector<std::uint8_t> filebuf = slurp_file(vcf_path);
        std::vector<io::BgzfBlock> blocks;
        if (!io::scan_bgzf(filebuf.data(), filebuf.size(), blocks) || blocks.empty()) {
            throw std::runtime_error(
                "read_vcf_panel_gpu: input is not a BGZF .vcf.gz (block index scan failed); the GPU "
                "path needs BGZF blocks — use the CPU reader for plain gzip");
        }

        // --- optional genetic map (shared io helper) --------------------------
        std::unordered_map<int, io::ChromMap> maps;
        if (!opts.map_path.empty()) maps = io::read_genetic_map(opts.map_path);

        // --- streaming state --------------------------------------------------
        constexpr std::size_t kBatchTextCap = 256ull << 20;  // ~256 MB decompressed / batch
        std::size_t n_sample = 0;
        std::size_t n_hap = 0;
        std::size_t src_bpr = 0;
        int n_col = 0;
        int max_col = 8;
        bool header_done = false;
        bool region_done = false;

        io::SnpTable& snptab = out.snptab;
        std::vector<std::uint8_t>& snp_major = out.tile.snp_major;
        io::VcfPanelCounts counts;
        long long last_pos = -1;
        std::string last_chrom;  // stripped-CHROM string compare (matches CPU)

        std::string carry;  // trailing partial line carried across batches
        std::size_t bi = 0;
        const std::size_t n_blocks = blocks.size();

        while (bi < n_blocks && !region_done) {
            // Grow the batch until its decompressed size reaches the cap (ISIZE is
            // known host-side from each block trailer — no decompression).
            std::size_t b0 = bi;
            std::size_t acc = 0;
            while (bi < n_blocks && acc < kBatchTextCap) {
                const io::BgzfBlock& blk = blocks[bi];
                const std::size_t isize =
                    static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 4]) |
                    (static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 3]) << 8) |
                    (static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 2]) << 16) |
                    (static_cast<std::size_t>(filebuf[blk.coff + blk.clen - 1]) << 24);
                acc += isize;
                ++bi;
            }
            const bool last_batch = (bi >= n_blocks);

            BatchText bt = decompress_batch(filebuf, blocks, b0, bi, carry, stream);
            const std::string& text = bt.host;
            const std::size_t tsz = text.size();

            // Parse only up to the last complete line; the tail becomes the carry.
            std::size_t last_nl = text.rfind('\n');
            std::size_t parse_end;
            if (last_nl == std::string::npos) {
                // No newline in the whole batch: either the header/line spans it
                // (carry the lot) or, on the final batch, the file lacks a trailing
                // newline (treat the buffer end as the line end).
                if (last_batch && tsz > 0) {
                    parse_end = tsz;
                    carry.clear();
                } else {
                    carry = text;
                    continue;
                }
            } else {
                parse_end = last_nl + 1;              // include the last '\n'
                carry = text.substr(last_nl + 1);     // trailing partial -> next batch
            }
            if (last_batch) carry.clear();

            // --- per-batch host decision pass + emit list -----------------------
            std::vector<std::uint64_t> emit_off;  // record line offset in bt.d_text
            std::vector<int> emit_gt_fi;

            std::size_t p = 0;
            while (p < parse_end) {
                std::size_t nl = text.find('\n', p);
                std::size_t end = (nl == std::string::npos) ? parse_end : nl;
                const std::size_t line_off = p;
                std::string_view line(text.data() + p, end - p);
                p = (nl == std::string::npos) ? parse_end : nl + 1;
                if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

                if (line.rfind("##", 0) == 0) continue;
                if (!line.empty() && line[0] == '#') {
                    // #CHROM header: resolve the sample columns (once).
                    const std::vector<std::string_view> h = vd::split(line, '\t');
                    if (h.size() < 10) {
                        throw std::runtime_error(
                            "read_vcf_panel_gpu: #CHROM header has no sample column in " + vcf_path);
                    }
                    n_sample = h.size() - 9;
                    n_hap = n_sample * 2;
                    src_bpr = io::packed_bytes(n_hap);
                    n_col = 9 + static_cast<int>(n_sample);
                    max_col = static_cast<int>(h.size()) - 1;  // last sample column index
                    out.sample_ids.clear();
                    out.sample_ids.reserve(n_sample);
                    for (std::size_t c = 9; c < h.size(); ++c) out.sample_ids.emplace_back(h[c]);
                    header_done = true;
                    continue;
                }
                if (!header_done) continue;               // data before #CHROM (skip)
                if (line.empty()) continue;

                const std::vector<std::string_view> f = vd::split(line, '\t');
                if (static_cast<int>(f.size()) <= max_col) continue;  // truncated record
                ++counts.records_seen;

                const std::string_view chrom_sv = io::strip_chr(f[0]);
                const auto pos_opt = vd::parse_int(f[1]);
                if (!pos_opt) continue;
                const long long pos = *pos_opt;

                if (opts.region.active) {
                    if (chrom_sv == opts.region.chrom) {
                        if (pos > opts.region.end) { region_done = true; break; }
                        if (pos < opts.region.start) { ++counts.skipped_out_of_region; continue; }
                    } else {
                        ++counts.skipped_out_of_region;
                        continue;
                    }
                }

                const std::string_view ref = f[3];
                const std::string_view alt = f[4];
                if (alt.find(',') != std::string_view::npos) {
                    ++counts.skipped_multiallelic;
                    continue;
                }
                if (!io::is_snp_allele(ref) || !io::is_snp_allele(alt)) {
                    ++counts.skipped_non_snp;
                    continue;
                }
                if (pos == last_pos && chrom_sv == last_chrom) {
                    ++counts.skipped_dup_pos;  // norm -d all
                    continue;
                }

                const int gt_fi = vd::format_index(f[8], "GT");
                const int chrom_int = io::chrom_to_int(chrom_sv);
                double genpos = 0.0;
                if (!maps.empty()) {
                    const auto it = maps.find(chrom_int);
                    if (it != maps.end()) genpos = io::interp_morgans(it->second, pos);
                }
                snptab.id.emplace_back(f[2]);
                snptab.chrom.push_back(chrom_int);
                snptab.genpos_morgans.push_back(genpos);
                snptab.physpos.push_back(static_cast<double>(pos));
                snptab.ref.push_back(ref[0]);
                snptab.alt.push_back(alt[0]);
                emit_off.push_back(static_cast<std::uint64_t>(line_off));
                emit_gt_fi.push_back(gt_fi);
                ++counts.emitted_sites;
                last_pos = pos;
                last_chrom.assign(chrom_sv);
            }

            // --- GPU: column index + GT decode/pack for this batch's records ----
            const std::size_t n_emit = emit_off.size();
            if (n_emit > 0) {
                DeviceBuffer<std::uint64_t> d_rec_off(n_emit);
                DeviceBuffer<int> d_gt_fi(n_emit);
                h2d_async(d_rec_off, emit_off.data(), n_emit, stream);
                h2d_async(d_gt_fi, emit_gt_fi.data(), n_emit, stream);

                const std::size_t cs_count = n_emit * (static_cast<std::size_t>(n_col) + 1);
                DeviceBuffer<std::uint32_t> d_colstart(cs_count);
                const unsigned cs_grid = static_cast<unsigned>((n_emit + 255) / 256);
                colscan_kernel<<<cs_grid, 256, 0, stream>>>(
                    reinterpret_cast<const char*>(bt.d_text.data()), d_rec_off.data(),
                    static_cast<long>(n_emit), tsz, n_col, d_colstart.data());
                STEPPE_CUDA_CHECK_KERNEL();

                DeviceBuffer<std::uint8_t> d_tile(n_emit * src_bpr);
                DeviceBuffer<unsigned long long> d_counters(3);
                STEPPE_CUDA_CHECK(
                    cudaMemsetAsync(d_counters.data(), 0, 3 * sizeof(unsigned long long), stream));
                const long total = static_cast<long>(n_emit) * static_cast<long>(src_bpr);
                const unsigned gt_grid = static_cast<unsigned>((total + 255) / 256);
                gtpack_kernel<<<gt_grid, 256, 0, stream>>>(
                    reinterpret_cast<const char*>(bt.d_text.data()), d_rec_off.data(),
                    d_colstart.data(), d_gt_fi.data(), static_cast<long>(n_emit),
                    static_cast<int>(n_sample), n_col, static_cast<long>(src_bpr), d_tile.data(),
                    d_counters.data());
                STEPPE_CUDA_CHECK_KERNEL();

                // Append the batch's packed rows to the panel (device-resident tile
                // -> one D2H per batch, in emit order == file order).
                const std::size_t off = snp_major.size();
                snp_major.resize(off + n_emit * src_bpr);
                d2h_async(snp_major.data() + off, d_tile, n_emit * src_bpr, stream);
                unsigned long long batch_counters[3] = {0, 0, 0};
                d2h_async(batch_counters, d_counters, 3, stream);
                STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
                h_counters[0] += batch_counters[0];
                h_counters[1] += batch_counters[1];
                h_counters[2] += batch_counters[2];
            }
        }

        if (!header_done) {
            throw std::runtime_error("read_vcf_panel_gpu: no #CHROM header found in " + vcf_path);
        }

        // --- finalize the tile (identical shape to the CPU reader) ------------
        counts.unphased_het_dropped = static_cast<long long>(h_counters[0]);
        counts.half_missing_haps = static_cast<long long>(h_counters[1]);
        counts.missing_haps = static_cast<long long>(h_counters[2]);
        snptab.count = static_cast<std::size_t>(counts.emitted_sites);
        counts.diploid_calls = counts.emitted_sites * static_cast<long long>(n_sample);

        out.n_sample = n_sample;
        out.tile.src_bytes_per_record = src_bpr;
        out.tile.n_snp = static_cast<std::size_t>(counts.emitted_sites);
        out.tile.n_individuals = n_hap;
        out.tile.sel_rows.resize(n_hap);
        for (std::size_t i = 0; i < n_hap; ++i) out.tile.sel_rows[i] = i;
        out.tile.pop_offsets = {0, n_hap};
        out.tile.pop_labels = {std::string("PANEL")};

        if (counts.diploid_calls > 0) {
            const double frac = static_cast<double>(counts.unphased_het_dropped) /
                                static_cast<double>(counts.diploid_calls);
            if (frac > opts.unphased_max) {
                cudaStreamDestroy(stream);
                std::string msg =
                    "read_vcf_panel_gpu: unphased-het fraction exceeds --unphased-max (" +
                    std::to_string(counts.unphased_het_dropped) + " of " +
                    std::to_string(counts.diploid_calls) + " diploid GT calls)";
                throw std::runtime_error(msg);
            }
        }
        out.counts = counts;
    } catch (...) {
        if (stream) cudaStreamDestroy(stream);
        throw;
    }

    cudaStreamDestroy(stream);
    return out;
}

}  // namespace steppe::device

#endif  // STEPPE_HAVE_NVCOMP
