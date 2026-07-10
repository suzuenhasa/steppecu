// src/io/vcf_panel_reader.cpp
//
// The multi-sample phased-VCF -> canonical haplotype-panel reader (Phase-1 core).
// Streams the .vcf.gz once, decodes every kept biallelic-SNP record's phased GT
// into two haploid columns per sample, and packs them SNP-major into an
// io::SnpMajorTile with codes {0,2,3}. See vcf_panel_reader.hpp for the contract.
//
// Reuse (SHAPE, cited): the tab/format/pipe tokenizers vcfdetail::split /
// format_index / subfield / parse_int (io/vcf_record.hpp); the multi-sample
// #CHROM all-columns scan (vcf_reader.cpp:608-643); the phased bit-split
// (vcf_reader.cpp:836-852), here hoisted to a top-level UNCONDITIONAL per-sample
// loop, made per-hap independent, and remapped to the panel's 2-bit code (0/2/3)
// instead of the GL path's 0xFF sentinel; the 2-bit packers pack_code_into_byte /
// code_in_byte / packed_bytes (io/eigenstrat_format.hpp); the SNP-major fresh-pack
// layout (geno_reader.cpp read_eigenstrat_snp_major_tile: src_bpr=packed_bytes(all
// source rows), byte_in_rec = i/4, pack_code_into_byte(byte, i, code)).
#include "io/vcf_panel_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <zlib.h>

#include "io/bgzf_index.hpp"
#include "io/eigenstrat_format.hpp"
#include "io/gzip_line_reader.hpp"
#include "io/vcf_panel_decode.hpp"
#include "io/vcf_record.hpp"

namespace steppe::io {

namespace vd = vcfdetail;

namespace {

// The pure decode rules (strip_chr, chrom_to_int, is_snp_allele, hap_code,
// ChromMap, read_genetic_map, interp_morgans) now live in the shared io header
// io/vcf_panel_decode.hpp so the GPU reader reproduces them byte-for-byte from one
// source of truth. They are steppe::io names, visible unqualified here.

// ============================================================================
// Phase-2: block-parallel BGZF decompression + parallel GT parse.
//
// A BGZF .vcf.gz is a stream of INDEPENDENT gzip members (blocks), each <=64 KB
// uncompressed and carrying a `BC` extra subfield = BSIZE (the block's total
// compressed length minus 1). Boundaries are discoverable by a header-only walk
// with ZERO decompression, so decompression is embarrassingly parallel: a cheap
// scanner emits (coff,clen) block ranges; a worker pool inflates disjoint
// block-runs independently; the <=1 line straddling each run seam is stitched by
// the driver; and a final SEQUENTIAL merge replays the exact serial per-record
// decision loop (region filter + `norm -d all` dup-collapse + the counters) over
// the workers' pre-decoded records, so the panel is BIT-IDENTICAL to the
// single-thread path. The heavy work (inflate + tokenize + phased-GT pack) runs
// in parallel; the merge is a cheap ordered concatenation with two comparisons
// and a memcpy per record.
//
// STEPPE_VCF_THREADS mirrors STEPPE_ROH_THREADS: unset -> hardware_concurrency,
// ==1 forces the serial reference path (the invariance gate), >1 caps the pool.
// A non-BGZF (plain-gzip) input has no block index -> falls back to serial.
// ============================================================================

// STEPPE_VCF_THREADS cap: a positive integer pins the worker count (=1 forces the
// serial reference path); unset/invalid -> hardware_concurrency (0 -> 1).
[[nodiscard]] std::size_t vcf_thread_cap() {
    const char* e = std::getenv("STEPPE_VCF_THREADS");
    if (e != nullptr && *e != '\0') {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && v >= 1) return static_cast<std::size_t>(v);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return (hw == 0) ? std::size_t{1} : static_cast<std::size_t>(hw);
}

// BgzfBlock + scan_bgzf (the zero-decompression block-index walk) now live in the
// shared io header io/bgzf_index.hpp, extended with xlen for the GPU reader's raw
// deflate-payload slice. Same struct + walk; the CPU path just ignores xlen.

// Inflate a byte range [begin,end) of the in-memory compressed buffer that spans
// whole BGZF blocks, into `out`. Uses the same concatenated-member loop the serial
// GzipLineReader runs (inflateInit2(15+32), inflateReset per Z_STREAM_END), so a
// worker inflating a run of blocks reuses the proven decode with no cross-block
// state. Throws std::runtime_error on a zlib error.
void inflate_range(const std::uint8_t* d, std::size_t begin, std::size_t end,
                   std::string& out) {
    z_stream strm;
    std::memset(&strm, 0, sizeof(strm));
    if (inflateInit2(&strm, 15 + 32) != Z_OK) {
        throw std::runtime_error("io::read_vcf_panel: inflateInit2 failed (parallel inflate)");
    }
    constexpr std::size_t kOut = 1u << 20;
    constexpr std::size_t kInCap = 256u << 20;  // cap avail_in to fit uInt safely
    std::vector<char> buf(kOut);
    std::size_t pos = begin;
    for (;;) {
        if (strm.avail_in == 0) {
            if (pos >= end) break;
            const std::size_t chunk = std::min(end - pos, kInCap);
            strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(d + pos));
            strm.avail_in = static_cast<uInt>(chunk);
            pos += chunk;
        }
        strm.next_out = reinterpret_cast<Bytef*>(buf.data());
        strm.avail_out = static_cast<uInt>(buf.size());
        const int ret = inflate(&strm, Z_NO_FLUSH);
        const std::size_t produced = buf.size() - strm.avail_out;
        if (produced > 0) out.append(buf.data(), produced);
        if (ret == Z_STREAM_END) {
            inflateReset(&strm);
            continue;
        }
        if (ret == Z_OK || ret == Z_BUF_ERROR) continue;
        inflateEnd(&strm);
        throw std::runtime_error("io::read_vcf_panel: zlib inflate error (code " +
                                 std::to_string(ret) + ") in parallel decode");
    }
    inflateEnd(&strm);
}

// A record's region classification when opts.region is active (matches the serial
// region block byte-for-byte via the same stripped-CHROM string compare).
enum RegionClass : std::uint8_t { RC_IN = 0, RC_BEFORE = 1, RC_WRONG_CHROM = 2, RC_PAST_END = 3 };

// One "seen" data record (every line that reaches the serial `++records_seen`),
// carrying just enough for the merge to replay the exact serial decision order
// WITHOUT re-tokenizing. Candidate records (biallelic SNP, in-region) additionally
// carry a packed GT fragment + SnpTable entry appended IN ORDER to the worker's
// accumulator; the merge consumes those by a per-source cursor.
struct WMeta {
    long long pos = 0;
    int chrom_int = 0;
    bool pos_valid = false;
    std::uint8_t region_class = RC_IN;   // meaningful only when region active
    bool is_multiallelic = false;
    bool is_non_snp = false;
    bool is_candidate = false;           // has a packed fragment + snptab entry
};

// A worker's (or a stitched seam line's) decode output: the ordered `metas`, plus
// the candidate fragments (packed GT bytes, one SnpTable row each, and the per-
// record unphased/half-missing/missing sub-counts) in candidate order.
struct WorkerAccum {
    std::vector<WMeta> metas;
    std::vector<std::uint8_t> packed;    // src_bpr bytes per candidate, contiguous
    SnpTable snptab;                     // one row per candidate
    std::vector<std::uint32_t> cand_unphased;
    std::vector<std::uint32_t> cand_half;
    std::vector<std::uint32_t> cand_missing;
};

// The decompressed-text edges of a run needed to stitch the <=1 straddling line at
// each seam: the leading partial (bytes before the run's first '\n' — belongs to
// the PREVIOUS seam; empty for run 0 which starts at a true line boundary) and the
// trailing partial (bytes after the run's last '\n' — belongs to the NEXT seam).
struct RunEdges {
    std::string prefix;
    std::string suffix;
    bool has_newline = false;
};

// Immutable per-decode context shared read-only across all workers.
struct DecodeCtx {
    const std::vector<int>* sample_cols = nullptr;
    std::size_t n_sample = 0;
    std::size_t src_bpr = 0;
    int max_col = 8;
    bool region_active = false;
    std::string region_chrom;            // compared as-is (matches serial)
    long long region_start = 0;
    long long region_end = 0;
    const std::unordered_map<int, ChromMap>* maps = nullptr;
};

// Decode ONE VCF text line into `acc`, mirroring the serial per-record body
// (vcf_panel_reader serial loop) exactly. '#'/empty/truncated lines produce no
// meta (serial `continue`s before ++records_seen). Every other line pushes one
// WMeta; a biallelic in-region SNP additionally packs its phased GT + SnpTable row.
// Region/dup/break decisions are NOT made here — they are the merge's job — so
// this is purely per-record and safe to run in parallel.
void decode_record(std::string_view line, const DecodeCtx& ctx, WorkerAccum& acc) {
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);  // CRLF tolerance
    if (line.empty() || line[0] == '#') return;
    const std::vector<std::string_view> f = vd::split(line, '\t');
    if (static_cast<int>(f.size()) <= ctx.max_col) return;           // truncated record

    WMeta m;
    const std::string_view chrom_sv = strip_chr(f[0]);
    m.chrom_int = chrom_to_int(chrom_sv);
    const auto pos_opt = vd::parse_int(f[1]);
    m.pos_valid = pos_opt.has_value();
    if (!pos_opt) {
        acc.metas.push_back(m);
        return;
    }
    const long long pos = *pos_opt;
    m.pos = pos;

    if (ctx.region_active) {
        if (chrom_sv == ctx.region_chrom) {
            if (pos > ctx.region_end) m.region_class = RC_PAST_END;
            else if (pos < ctx.region_start) m.region_class = RC_BEFORE;
            else m.region_class = RC_IN;
        } else {
            m.region_class = RC_WRONG_CHROM;
        }
    }

    const std::string_view ref = f[3];
    const std::string_view alt = f[4];
    m.is_multiallelic = (alt.find(',') != std::string_view::npos);
    m.is_non_snp = !m.is_multiallelic && (!is_snp_allele(ref) || !is_snp_allele(alt));

    const bool biallelic_snp = !m.is_multiallelic && !m.is_non_snp;
    const bool in_region = !ctx.region_active || (m.region_class == RC_IN);
    if (biallelic_snp && in_region) {
        m.is_candidate = true;
        const int gt_fi = vd::format_index(f[8], "GT");
        const std::size_t rec_off = acc.packed.size();
        acc.packed.resize(rec_off + ctx.src_bpr, std::uint8_t{0});
        std::uint8_t* rec = acc.packed.data() + rec_off;
        std::uint32_t unphased = 0, half = 0, missing = 0;
        const std::size_t n_sample = ctx.n_sample;
        for (std::size_t k = 0; k < n_sample; ++k) {
            const std::string_view sample = f[static_cast<std::size_t>((*ctx.sample_cols)[k])];
            const std::string_view gtv =
                (gt_fi >= 0) ? vd::subfield(sample, gt_fi) : std::string_view{};
            std::uint8_t h0 = kMissingCode;
            std::uint8_t h1 = kMissingCode;
            const bool phased = gtv.find('|') != std::string_view::npos;
            const char sep = phased ? '|' : '/';
            const std::vector<std::string_view> ga = vd::split(gtv, sep);
            if (ga.size() == 2) {
                const bool unphased_het = !phased && ga[0] != ga[1];
                if (unphased_het) {
                    ++unphased;
                } else {
                    h0 = hap_code(ga[0]);
                    h1 = hap_code(ga[1]);
                    const bool q0 = (h0 == kMissingCode);
                    const bool q1 = (h1 == kMissingCode);
                    if (q0 != q1) ++half;
                }
            }
            if (h0 == kMissingCode) ++missing;
            if (h1 == kMissingCode) ++missing;
            const std::size_t i0 = 2u * k;
            const std::size_t i1 = i0 + 1u;
            rec[i0 / static_cast<std::size_t>(kCodesPerByte)] = pack_code_into_byte(
                rec[i0 / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(i0), h0);
            rec[i1 / static_cast<std::size_t>(kCodesPerByte)] = pack_code_into_byte(
                rec[i1 / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(i1), h1);
        }

        double genpos = 0.0;
        if (ctx.maps != nullptr && !ctx.maps->empty()) {
            const auto mit = ctx.maps->find(m.chrom_int);
            if (mit != ctx.maps->end()) genpos = interp_morgans(mit->second, pos);
        }
        acc.snptab.id.emplace_back(f[2]);
        acc.snptab.chrom.push_back(m.chrom_int);
        acc.snptab.genpos_morgans.push_back(genpos);
        acc.snptab.physpos.push_back(static_cast<double>(pos));
        acc.snptab.ref.push_back(ref[0]);
        acc.snptab.alt.push_back(alt[0]);
        acc.cand_unphased.push_back(unphased);
        acc.cand_half.push_back(half);
        acc.cand_missing.push_back(missing);
    }

    acc.metas.push_back(m);
}

// Decode one block-run's decompressed text into `acc`, recording the seam edges.
// Worker 0's run starts at a true line boundary (no leading partial); every other
// run's leading bytes before its first '\n' are a partial owned by the previous
// seam. Only lines FULLY inside the run (terminating '\n' at or before the run's
// last '\n') are decoded here; the leading/trailing partials are stitched by the
// driver. Matches the serial stream's line split exactly.
void decode_run(bool is_first, const std::string& text, const DecodeCtx& ctx,
                WorkerAccum& acc, RunEdges& edge) {
    if (text.empty()) {
        edge.has_newline = false;
        return;
    }
    const std::size_t first_nl = text.find('\n');
    if (first_nl == std::string::npos) {
        // No newline in the whole run: the entire run is one partial line (only
        // possible if a single VCF line exceeds a whole run — not for these panels,
        // but handled: it carries into the seam stitch).
        edge.has_newline = false;
        edge.prefix = text;   // whole run is a leading partial for the prev seam
        return;
    }
    const std::size_t last_nl = text.rfind('\n');
    edge.has_newline = true;
    edge.prefix = is_first ? std::string() : text.substr(0, first_nl);
    edge.suffix = text.substr(last_nl + 1);
    std::size_t cur = is_first ? 0 : (first_nl + 1);
    while (cur <= last_nl) {
        const std::size_t nl = text.find('\n', cur);
        // nl is guaranteed valid and <= last_nl because cur <= last_nl.
        decode_record(std::string_view(text.data() + cur, nl - cur), ctx, acc);
        cur = nl + 1;
    }
}

// Lift of parallel_blocks_roh (cuda_backend_roh.cu): one std::thread per run, each
// worker's throw captured as an exception_ptr and rethrown AFTER join (an uncaught
// throw crossing a std::thread boundary calls std::terminate). Runs inline for T<=1.
template <typename Fn>
void parallel_block_runs(std::size_t T, Fn&& fn) {
    if (T <= 1) {
        if (T == 1) fn(std::size_t{0});
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(T);
    std::vector<std::exception_ptr> errs(T);
    for (std::size_t r = 0; r < T; ++r) {
        threads.emplace_back([&fn, &errs, r]() {
            try {
                fn(r);
            } catch (...) {
                errs[r] = std::current_exception();
            }
        });
    }
    for (std::thread& th : threads) th.join();
    for (const std::exception_ptr& e : errs)
        if (e) std::rethrow_exception(e);
}

// Read the whole compressed file into memory (the parallel scanner + workers index
// it directly). Throws on open/read failure.
[[nodiscard]] std::vector<std::uint8_t> slurp_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("io::read_vcf_panel: cannot open VCF file: " + path);
    const std::streamoff n = in.tellg();
    if (n < 0) throw std::runtime_error("io::read_vcf_panel: cannot size VCF file: " + path);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(n));
    in.seekg(0);
    if (n > 0 && !in.read(reinterpret_cast<char*>(buf.data()), n)) {
        throw std::runtime_error("io::read_vcf_panel: short read on VCF file: " + path);
    }
    return buf;
}

}  // namespace

// The frozen single-thread reference path (Phase-1). STEPPE_VCF_THREADS=1 selects
// this; the parallel path is gated bit-identical against it. UNCHANGED behaviour.
static VcfPanelResult read_vcf_panel_serial(const std::string& vcf_path,
                                            const VcfPanelOptions& opts) {
    GzipLineReader reader(vcf_path);

    // --- optional genetic map ------------------------------------------------
    std::unordered_map<int, ChromMap> maps;
    if (!opts.map_path.empty()) maps = read_genetic_map(opts.map_path);

    // --- header: resolve EVERY sample column (>=9) --- reuse vcf_reader.cpp:608-643
    std::vector<int> sample_cols;
    std::vector<std::string> sample_ids;
    std::string line;
    bool saw_chrom_header = false;
    while (reader.next_line(line)) {
        if (line.rfind("##", 0) == 0) continue;
        if (!line.empty() && line[0] == '#') {
            const std::vector<std::string_view> h = vd::split(line, '\t');
            if (h.size() < 10) {
                throw std::runtime_error(
                    "io::read_vcf_panel: #CHROM header has no sample column in " + vcf_path);
            }
            for (std::size_t c = 9; c < h.size(); ++c) {
                sample_cols.push_back(static_cast<int>(c));
                sample_ids.emplace_back(h[c]);
            }
            saw_chrom_header = true;
            break;
        }
    }
    if (!saw_chrom_header) {
        throw std::runtime_error("io::read_vcf_panel: no #CHROM header found in " + vcf_path);
    }

    VcfPanelResult out;
    out.sample_ids = std::move(sample_ids);
    out.n_sample = out.sample_ids.size();
    const std::size_t n_sample = out.n_sample;
    const std::size_t n_hap = n_sample * 2;                     // two haploid columns / sample
    const std::size_t src_bpr = packed_bytes(n_hap);            // SNP-major record width
    const int max_col = sample_cols.empty() ? 8 : sample_cols.back();

    VcfPanelCounts counts;
    std::vector<std::uint8_t>& packed = out.tile.snp_major;     // append src_bpr bytes / kept SNP
    SnpTable& snptab = out.snptab;

    long long last_emitted_pos = -1;
    std::string last_emitted_chrom;

    // --- single streaming pass: decode+pack each kept record inline ----------
    while (reader.next_line(line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string_view> f = vd::split(line, '\t');
        if (static_cast<int>(f.size()) <= max_col) continue;   // truncated record
        ++counts.records_seen;

        const std::string_view chrom_sv = strip_chr(f[0]);
        const auto pos_opt = vd::parse_int(f[1]);
        if (!pos_opt) continue;
        const long long pos = *pos_opt;

        // In-stream POS-range filter (no tabix; stop once past the range end on the
        // matching chromosome — VCF is position-sorted within a chromosome).
        if (opts.region.active) {
            if (chrom_sv == opts.region.chrom) {
                if (pos > opts.region.end) break;               // past the range: done
                if (pos < opts.region.start) { ++counts.skipped_out_of_region; continue; }
            } else {
                ++counts.skipped_out_of_region;
                continue;
            }
        }

        const std::string_view ref = f[3];
        const std::string_view alt = f[4];
        // Biallelic filter, matching `-m2 -M2 -v snps`.
        if (alt.find(',') != std::string_view::npos) { ++counts.skipped_multiallelic; continue; }
        if (!is_snp_allele(ref) || !is_snp_allele(alt)) { ++counts.skipped_non_snp; continue; }

        // `norm -d all`: collapse a duplicate POS (keep the first-seen). VCF is
        // position-sorted so a duplicate is adjacent to the last emitted site.
        if (pos == last_emitted_pos && chrom_sv == last_emitted_chrom) {
            ++counts.skipped_dup_pos;
            continue;
        }

        // Locate GT once per record (reuse vcfdetail::format_index).
        const int gt_fi = vd::format_index(f[8], "GT");

        // Append one zero-initialized SNP-major record and fill its haplotype codes.
        const std::size_t rec_off = packed.size();
        packed.resize(rec_off + src_bpr, std::uint8_t{0});
        std::uint8_t* rec = packed.data() + rec_off;

        for (std::size_t k = 0; k < n_sample; ++k) {
            const std::string_view sample = f[static_cast<std::size_t>(sample_cols[k])];
            const std::string_view gtv = (gt_fi >= 0) ? vd::subfield(sample, gt_fi)
                                                      : std::string_view{};

            std::uint8_t h0 = kMissingCode;  // 3
            std::uint8_t h1 = kMissingCode;
            // Phased bit-split (reuse vcf_reader.cpp:836-852 SHAPE): split on '|'
            // (phased) or '/' (unphased), hap1 = allele before, hap2 = after.
            const bool phased = gtv.find('|') != std::string_view::npos;
            const char sep = phased ? '|' : '/';
            const std::vector<std::string_view> ga = vd::split(gtv, sep);
            if (ga.size() == 2) {
                const bool unphased_het = !phased && ga[0] != ga[1];
                if (unphased_het) {
                    // No defined hap order -> both haplotypes missing. Paint's own
                    // n_diploid gate does NOT catch this; the reader counts it.
                    ++counts.unphased_het_dropped;
                    // h0, h1 stay kMissingCode
                } else {
                    h0 = hap_code(ga[0]);
                    h1 = hap_code(ga[1]);
                    const bool m0 = (h0 == kMissingCode);
                    const bool m1 = (h1 == kMissingCode);
                    if (m0 != m1) ++counts.half_missing_haps;   // one hap present, one '.'
                }
            }
            // else: malformed / empty GT -> both haplotypes missing (code 3).

            if (h0 == kMissingCode) ++counts.missing_haps;
            if (h1 == kMissingCode) ++counts.missing_haps;

            // Pack the two haplotype columns 2k and 2k+1 (SNP-major fresh-pack shape,
            // geno_reader.cpp:561-573): byte_in_rec = i/4, pack_code_into_byte(byte, i, code).
            const std::size_t i0 = 2u * k;
            const std::size_t i1 = i0 + 1u;
            rec[i0 / static_cast<std::size_t>(kCodesPerByte)] = pack_code_into_byte(
                rec[i0 / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(i0), h0);
            rec[i1 / static_cast<std::size_t>(kCodesPerByte)] = pack_code_into_byte(
                rec[i1 / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(i1), h1);
        }

        // SNP metadata (inline .snp): id, chrom, genpos (Morgans), physpos, ref/alt.
        const int chrom_int = chrom_to_int(chrom_sv);
        double genpos = 0.0;
        if (!maps.empty()) {
            const auto mit = maps.find(chrom_int);
            if (mit != maps.end()) genpos = interp_morgans(mit->second, pos);
        }
        snptab.id.emplace_back(f[2]);
        snptab.chrom.push_back(chrom_int);
        snptab.genpos_morgans.push_back(genpos);
        snptab.physpos.push_back(static_cast<double>(pos));
        snptab.ref.push_back(ref[0]);
        snptab.alt.push_back(alt[0]);

        ++counts.emitted_sites;
        last_emitted_pos = pos;
        last_emitted_chrom.assign(chrom_sv);
    }

    // --- finalize the tile ---------------------------------------------------
    snptab.count = static_cast<std::size_t>(counts.emitted_sites);
    counts.diploid_calls = counts.emitted_sites * static_cast<long long>(n_sample);

    out.tile.src_bytes_per_record = src_bpr;
    out.tile.n_snp = static_cast<std::size_t>(counts.emitted_sites);
    out.tile.n_individuals = n_hap;
    out.tile.sel_rows.resize(n_hap);
    for (std::size_t i = 0; i < n_hap; ++i) out.tile.sel_rows[i] = i;   // identity: all haps
    out.tile.pop_offsets = {0, n_hap};                                  // one "PANEL" pop
    out.tile.pop_labels = {std::string("PANEL")};

    // --- unphased-drop guard: fail loud above the threshold ------------------
    if (counts.diploid_calls > 0) {
        const double frac =
            static_cast<double>(counts.unphased_het_dropped) /
            static_cast<double>(counts.diploid_calls);
        if (frac > opts.unphased_max) {
            std::ostringstream oss;
            oss << "io::read_vcf_panel: unphased-het fraction " << frac << " ("
                << counts.unphased_het_dropped << " of " << counts.diploid_calls
                << " diploid GT calls) exceeds --unphased-max " << opts.unphased_max
                << " — the panel is not sufficiently phased for haplotype painting (unphased "
                   "GTs were dropped to missing and would silently pass paint's n_diploid gate)";
            throw std::runtime_error(oss.str());
        }
    }

    out.counts = counts;
    return out;
}

VcfPanelResult read_vcf_panel(const std::string& vcf_path, const VcfPanelOptions& opts) {
    const std::size_t T_req = vcf_thread_cap();
    if (T_req <= 1) return read_vcf_panel_serial(vcf_path, opts);  // gate reference path

    // --- Stage 1: slurp + header-only BGZF block scan (zero decompression) ----
    std::vector<std::uint8_t> filebuf = slurp_file(vcf_path);
    std::vector<BgzfBlock> blocks;
    if (!scan_bgzf(filebuf.data(), filebuf.size(), blocks) || blocks.size() < 2) {
        // Plain gzip / not BGZF / trivially small -> the serial path is still correct.
        return read_vcf_panel_serial(vcf_path, opts);
    }
    const std::size_t n_blocks = blocks.size();
    const std::size_t T = std::min(T_req, n_blocks);
    if (T <= 1) return read_vcf_panel_serial(vcf_path, opts);

    // --- optional genetic map (read once, shared read-only across workers) ----
    std::unordered_map<int, ChromMap> maps;
    if (!opts.map_path.empty()) maps = read_genetic_map(opts.map_path);

    // --- header pre-parse: resolve EVERY sample column (matches serial 167-190) -
    std::vector<int> sample_cols;
    std::vector<std::string> sample_ids;
    {
        GzipLineReader reader(vcf_path);
        std::string line;
        bool saw_chrom_header = false;
        while (reader.next_line(line)) {
            if (line.rfind("##", 0) == 0) continue;
            if (!line.empty() && line[0] == '#') {
                const std::vector<std::string_view> h = vd::split(line, '\t');
                if (h.size() < 10) {
                    throw std::runtime_error(
                        "io::read_vcf_panel: #CHROM header has no sample column in " + vcf_path);
                }
                for (std::size_t c = 9; c < h.size(); ++c) {
                    sample_cols.push_back(static_cast<int>(c));
                    sample_ids.emplace_back(h[c]);
                }
                saw_chrom_header = true;
                break;
            }
        }
        if (!saw_chrom_header) {
            throw std::runtime_error("io::read_vcf_panel: no #CHROM header found in " + vcf_path);
        }
    }

    const std::size_t n_sample = sample_ids.size();
    const std::size_t n_hap = n_sample * 2;
    const std::size_t src_bpr = packed_bytes(n_hap);

    DecodeCtx ctx;
    ctx.sample_cols = &sample_cols;
    ctx.n_sample = n_sample;
    ctx.src_bpr = src_bpr;
    ctx.max_col = sample_cols.empty() ? 8 : sample_cols.back();
    ctx.region_active = opts.region.active;
    ctx.region_chrom = opts.region.chrom;
    ctx.region_start = opts.region.start;
    ctx.region_end = opts.region.end;
    ctx.maps = &maps;

    // --- Stage 2+3: parallel inflate + per-run GT decode (disjoint block runs) -
    std::vector<WorkerAccum> accums(T);
    std::vector<RunEdges> edges(T);
    parallel_block_runs(T, [&](std::size_t r) {
        const std::size_t b0 = r * n_blocks / T;          // balanced: each run >=1 block
        const std::size_t b1 = (r + 1) * n_blocks / T;
        const std::size_t coff = blocks[b0].coff;
        const std::size_t cend = blocks[b1 - 1].coff + blocks[b1 - 1].clen;
        std::string text;
        inflate_range(filebuf.data(), coff, cend, text);
        decode_run(r == 0, text, ctx, accums[r], edges[r]);
    });

    // --- boundary-line stitch: the <=1 line straddling each run seam -----------
    // Straddle after run r = suffix_r + (any no-newline runs' whole text) + prefix
    // of the next run that has a newline. For these panels every run has newlines,
    // so this is simply suffix_r + prefix_{r+1}; the loop covers the general case.
    std::vector<WorkerAccum> straddles(T > 0 ? T - 1 : 0);
    for (std::size_t r = 0; r + 1 < T; ++r) {
        std::string seam = edges[r].suffix;
        std::size_t j = r + 1;
        while (j < T && !edges[j].has_newline) {
            seam += edges[j].prefix;   // no-newline run: its whole text is a partial
            ++j;
        }
        if (j < T) seam += edges[j].prefix;
        decode_record(std::string_view(seam), ctx, straddles[r]);
    }
    // The file's final line (bytes after the last run's last '\n'); usually empty
    // since VCF lines are newline-terminated (serial flushes it at EOF otherwise).
    WorkerAccum final_line;
    decode_record(std::string_view(edges[T - 1].suffix), ctx, final_line);

    // --- Stage 4: SEQUENTIAL merge — replay the exact serial decision loop -----
    // Sources in global stream order: run0, seam0, run1, seam1, ..., run_{T-1},
    // final_line. Each candidate meta maps 1:1 to a packed fragment consumed by a
    // per-source cursor; region-break, dup-collapse, and all counters are applied
    // here so the emitted panel is byte-identical to the serial fill.
    std::vector<const WorkerAccum*> sources;
    sources.reserve(2 * T);
    for (std::size_t r = 0; r < T; ++r) {
        sources.push_back(&accums[r]);
        if (r + 1 < T) sources.push_back(&straddles[r]);
    }
    sources.push_back(&final_line);

    VcfPanelResult out;
    out.sample_ids = std::move(sample_ids);
    out.n_sample = n_sample;
    SnpTable& snptab = out.snptab;
    std::vector<std::uint8_t>& packed = out.tile.snp_major;

    std::size_t total_cand = 0;
    for (const WorkerAccum* s : sources) total_cand += s->snptab.chrom.size();
    packed.reserve(total_cand * src_bpr);
    snptab.id.reserve(total_cand);
    snptab.chrom.reserve(total_cand);
    snptab.genpos_morgans.reserve(total_cand);
    snptab.physpos.reserve(total_cand);
    snptab.ref.reserve(total_cand);
    snptab.alt.reserve(total_cand);

    VcfPanelCounts counts;
    long long last_pos = -1;
    int last_chrom = 0;
    bool have_last = false;
    bool broke = false;

    for (const WorkerAccum* src : sources) {
        if (broke) break;
        std::size_t cur = 0;  // per-source candidate cursor
        for (const WMeta& m : src->metas) {
            ++counts.records_seen;
            if (!m.pos_valid) continue;
            if (ctx.region_active) {
                if (m.region_class == RC_PAST_END) {
                    broke = true;
                    break;
                }
                if (m.region_class == RC_BEFORE || m.region_class == RC_WRONG_CHROM) {
                    ++counts.skipped_out_of_region;
                    continue;
                }
                // RC_IN falls through.
            }
            if (m.is_multiallelic) {
                ++counts.skipped_multiallelic;
                continue;
            }
            if (m.is_non_snp) {
                ++counts.skipped_non_snp;
                continue;
            }
            // Biallelic in-region SNP == the cur-th candidate of this source.
            if (have_last && m.pos == last_pos && m.chrom_int == last_chrom) {
                ++counts.skipped_dup_pos;   // norm -d all: collapse duplicate POS
                ++cur;
                continue;
            }
            const std::uint8_t* fr = src->packed.data() + cur * src_bpr;
            packed.insert(packed.end(), fr, fr + src_bpr);
            snptab.id.push_back(src->snptab.id[cur]);
            snptab.chrom.push_back(src->snptab.chrom[cur]);
            snptab.genpos_morgans.push_back(src->snptab.genpos_morgans[cur]);
            snptab.physpos.push_back(src->snptab.physpos[cur]);
            snptab.ref.push_back(src->snptab.ref[cur]);
            snptab.alt.push_back(src->snptab.alt[cur]);
            counts.unphased_het_dropped += src->cand_unphased[cur];
            counts.half_missing_haps += src->cand_half[cur];
            counts.missing_haps += src->cand_missing[cur];
            ++counts.emitted_sites;
            last_pos = m.pos;
            last_chrom = m.chrom_int;
            have_last = true;
            ++cur;
        }
    }

    // --- finalize the tile (identical to the serial path) ---------------------
    snptab.count = static_cast<std::size_t>(counts.emitted_sites);
    counts.diploid_calls = counts.emitted_sites * static_cast<long long>(n_sample);

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
            std::ostringstream oss;
            oss << "io::read_vcf_panel: unphased-het fraction " << frac << " ("
                << counts.unphased_het_dropped << " of " << counts.diploid_calls
                << " diploid GT calls) exceeds --unphased-max " << opts.unphased_max
                << " — the panel is not sufficiently phased for haplotype painting (unphased "
                   "GTs were dropped to missing and would silently pass paint's n_diploid gate)";
            throw std::runtime_error(oss.str());
        }
    }

    out.counts = counts;
    return out;
}

void dump_hap_codes(const SnpMajorTile& tile, const SnpTable& snptab, std::ostream& os) {
    const std::size_t n_snp = tile.n_snp;
    const std::size_t n_ind = tile.n_individuals;
    const std::size_t bpr = tile.src_bytes_per_record;
    for (std::size_t s = 0; s < n_snp; ++s) {
        const std::uint8_t* rec = tile.snp_major.data() + s * bpr;
        // CHROM<TAB>POS prefix (matches the bcftools oracle's %CHROM %POS).
        const int chrom = (s < snptab.chrom.size()) ? snptab.chrom[s] : kFirstOtherChromCode;
        const long long pos =
            (s < snptab.physpos.size()) ? static_cast<long long>(snptab.physpos[s]) : 0;
        os << chrom << '\t' << pos;
        for (std::size_t g = 0; g < n_ind; ++g) {
            const std::size_t src_row = tile.sel_rows.empty() ? g : tile.sel_rows[g];
            const std::uint8_t code = code_in_byte(
                rec[src_row / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(src_row));
            os << '\t' << static_cast<int>(code);
        }
        os << '\n';
    }
}

}  // namespace steppe::io
