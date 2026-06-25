// tests/reference/test_transpose_canonical.cu
//
// M-FR-1 GATE — the SNP-major -> canonical individual-major TRANSPOSE+GATHER+ENCODING
// primitive (docs/design/format-readers.md §2.4, M-FR-1; the format-reader engine).
// A DUAL gate, NO new statistical golden, NO AT2 rerun:
//
//   (A) SYNTHETIC bit-primitive unit. A hand-built small SNP-major tile + a KNOWN
//       expected individual-major output; asserts GPU == CPU == expected EXACTLY
//       (memcmp == 0). It covers the edge cases REAL AADR cannot exercise:
//         * the GENO rlen-FLOOR (max(48, ceil(n_ind/4))) PADDING — a 3-individual
//           source has bytes_per_record = 48, so 45 padding bytes per SNP record
//           that must NEVER be read as phantom individuals (format-readers.md §3.4);
//         * SELECTION + pop-contiguous REORDER — output column g reads a non-identity
//           source row (the IndPartition gather), in a re-ordered pop layout;
//         * the PARTIAL LAST BYTE — n_snp not a multiple of 4 ⇒ the last output byte
//           carries < 4 codes, the tail bits must be 0 (the host packer's behavior).
//       This is a unit test (a KNOWN expected tile, like the synthetic-TGENO
//       test_geno_reader.cpp), NOT a reported statistic.
//
//   (B) REAL-AADR same-data-two-ways. The first K SNPs × ALL individuals of the
//       convertf-PACKEDANCESTRYMAP /workspace/data/aadr/converted_pa/v66_HO_pa.geno
//       (magic GENO, SNP-major) is gathered (identity selection) and TRANSPOSED, then
//       asserted BYTE-IDENTICAL (memcmp == 0) to the first-K-SNP slice of the raw v66
//       TGENO /workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.geno (individual-major,
//       via the existing read_tile). convertf is a lossless transcode, so PA and TGENO
//       are the SAME genotypes on two axes — a correct transpose MUST reproduce the
//       TGENO tile bit-for-bit. SKIP exit-0 if the AADR root is absent.
//
// INTEGER/BIT-ONLY: the transpose is a 2-bit unpack + remap + MSB-first re-pack — no
// float, no tolerance, no oracle file. Bit-exactness is by construction (the
// emulated-FP64 policy is matmul-only and does NOT apply; format-readers.md §2.2).
//
// Exits NONZERO on any failure (CTest gates on the exit code). Self-checking main()
// — NOT a GoogleTest TU (mirrors test_decode_equivalence.cu).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "device/backend.hpp"          // ComputeBackend, SnpMajorTileView, CanonicalTile, TileEncoding
#include "device/backend_factory.hpp"  // steppe::device::make_cpu_backend / make_cuda_backend

#include "io/eigenstrat_format.hpp"    // GenoFormat, packed_bytes, code_in_byte, kCodesPerByte
#include "io/geno_reader.hpp"          // GenoReader, read_tile (the TGENO oracle path)
#include "io/genotype_tile.hpp"        // GenotypeTile
#include "io/ind_reader.hpp"           // IndPartition, PopGroup

using steppe::CanonicalTile;
using steppe::ComputeBackend;
using steppe::SnpMajorTileView;
using steppe::TileEncoding;

namespace {

constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kTgenoBase = "v66.p1_HO.aadr.patch.PUB";  // raw TGENO (individual-major)
constexpr const char* kPaBase = "v66_HO_pa";                    // convertf PA (SNP-major GENO)

// Tier-1 SNP prefix: a few thousand SNPs over ALL 27594 inds — enough to span many
// output bytes / a partial-last-byte while keeping the gather I/O small.
constexpr std::size_t kTier1Snps = 4099;  // not a multiple of 4 (exercise the partial last byte)

// MSB-first set: pack 2-bit `code` into position `pos` (0..3) of `*byte`.
void set_code(std::uint8_t& byte, int pos, std::uint8_t code) {
    const int shift = (steppe::io::kCodesPerByte - 1 - pos) * steppe::io::kBitsPerCode;  // 6,4,2,0
    byte = static_cast<std::uint8_t>(byte | static_cast<std::uint8_t>((code & 0x3u) << shift));
}

// ---------------------------------------------------------------------------
// (A) SYNTHETIC bit-primitive unit.
// ---------------------------------------------------------------------------
// A tiny SNP-major source with KNOWN codes, then a transpose with a non-identity
// selection/reorder, and a hand-built expected individual-major tile. Returns the
// number of failures.
int run_synthetic(ComputeBackend& cpu, ComputeBackend& gpu) {
    int failures = 0;

    // SOURCE geometry: 3 individuals, 7 SNPs (partial last byte: 7 = 4+3). GENO's
    // rlen floor makes bytes_per_record = max(48, ceil(3/4)=1) = 48 — so the source
    // SNP record is 48 bytes wide with 45 PADDING bytes after the single real byte.
    constexpr std::size_t kNindSrc = 3;
    constexpr std::size_t kNsnp = 7;
    const std::size_t src_bpr = std::max<std::size_t>(48, steppe::io::packed_bytes(kNindSrc));
    if (src_bpr != 48) {  // pin the rlen-floor assumption (the whole point of this case)
        std::fprintf(stderr, "[synthetic] FAIL: expected rlen-floored src_bpr=48, got %zu\n", src_bpr);
        return 1;
    }

    // KNOWN per-(individual i, SNP s) codes. i in 0..2 (source rows), s in 0..6.
    // Distinct values incl. the missing code 3 and the het code 1.
    //   ind 0: 0 1 2 3 | 0 1 2
    //   ind 1: 3 2 1 0 | 1 2 3
    //   ind 2: 1 0 3 2 | 2 3 0
    const std::uint8_t codes[kNindSrc][kNsnp] = {
        {0, 1, 2, 3, 0, 1, 2},
        {3, 2, 1, 0, 1, 2, 3},
        {1, 0, 3, 2, 2, 3, 0},
    };

    // Build the SNP-major source: record s = SNP s, individual i at byte i/4 pos i%4.
    // The PADDING bytes (i/4 >= ceil(3/4)=1 i.e. bytes 1..47) stay a NON-ZERO sentinel
    // (0xFF) so a kernel that wrongly walked bytes_per_record*4 individuals would read
    // garbage and the compare would catch it.
    std::vector<std::uint8_t> snp_major(kNsnp * src_bpr, 0xFF);
    for (std::size_t s = 0; s < kNsnp; ++s) {
        // zero the REAL byte (byte 0) before OR-ing codes; leave bytes 1..47 = 0xFF padding.
        snp_major[s * src_bpr + 0] = 0;
        for (std::size_t i = 0; i < kNindSrc; ++i) {
            const std::size_t byte_in_snp = i / static_cast<std::size_t>(steppe::io::kCodesPerByte);
            const int pos = static_cast<int>(i % static_cast<std::size_t>(steppe::io::kCodesPerByte));
            set_code(snp_major[s * src_bpr + byte_in_snp], pos, codes[i][s]);
        }
    }

    // SELECTION + pop-contiguous REORDER. Two populations in Q/V/N order:
    //   pop 0 = {source row 2, source row 0}   (reorder: NOT ascending source order)
    //   pop 1 = {source row 1}
    // output column g -> source row:  g0->2, g1->0, g2->1
    const std::vector<std::size_t> sel_rows = {2, 0, 1};
    const std::vector<std::size_t> pop_offsets = {0, 2, 3};  // P=2 segments
    const std::size_t n_individuals = sel_rows.size();

    SnpMajorTileView view;
    view.snp_major = snp_major.data();
    view.src_bytes_per_record = src_bpr;
    view.n_snp = kNsnp;
    view.sel_rows = sel_rows.data();
    view.n_individuals = n_individuals;
    view.pop_offsets = pop_offsets.data();
    view.n_pop = 2;
    view.encoding = TileEncoding::Identity;

    // EXPECTED canonical individual-major tile (hand-built INDEPENDENTLY of the
    // primitive). out_bpr = ceil(7/4) = 2. Output record g packs SNPs 0..6 of source
    // row sel_rows[g], MSB-first; SNP 7 (the 8th slot of byte 1) does not exist ⇒ 0.
    const std::size_t out_bpr = steppe::io::packed_bytes(kNsnp);  // = 2
    std::vector<std::uint8_t> expected(n_individuals * out_bpr, 0);
    for (std::size_t g = 0; g < n_individuals; ++g) {
        const std::size_t src = sel_rows[g];
        for (std::size_t s = 0; s < kNsnp; ++s) {
            const std::size_t byte_in_rec = s / static_cast<std::size_t>(steppe::io::kCodesPerByte);
            const int pos = static_cast<int>(s % static_cast<std::size_t>(steppe::io::kCodesPerByte));
            set_code(expected[g * out_bpr + byte_in_rec], pos, codes[src][s]);
        }
    }

    // Run BOTH backends.
    const CanonicalTile t_cpu = cpu.transpose_to_canonical(view);
    const CanonicalTile t_gpu = gpu.transpose_to_canonical(view);

    // Shape checks.
    const bool shape_cpu = (t_cpu.bytes_per_record == out_bpr &&
                            t_cpu.n_individuals == n_individuals &&
                            t_cpu.n_snp == kNsnp &&
                            t_cpu.packed.size() == expected.size() &&
                            t_cpu.pop_offsets == pop_offsets);
    const bool shape_gpu = (t_gpu.bytes_per_record == out_bpr &&
                            t_gpu.n_individuals == n_individuals &&
                            t_gpu.n_snp == kNsnp &&
                            t_gpu.packed.size() == expected.size() &&
                            t_gpu.pop_offsets == pop_offsets);
    if (!shape_cpu) { ++failures; std::fprintf(stderr, "[synthetic] CPU shape MISMATCH\n"); }
    if (!shape_gpu) { ++failures; std::fprintf(stderr, "[synthetic] GPU shape MISMATCH\n"); }

    // Bit-exact: CPU == expected, GPU == expected, GPU == CPU.
    const bool cpu_eq = shape_cpu && (std::memcmp(t_cpu.packed.data(), expected.data(),
                                                  expected.size()) == 0);
    const bool gpu_eq = shape_gpu && (std::memcmp(t_gpu.packed.data(), expected.data(),
                                                  expected.size()) == 0);
    const bool gc_eq = (t_cpu.packed.size() == t_gpu.packed.size()) &&
                       (std::memcmp(t_cpu.packed.data(), t_gpu.packed.data(),
                                    t_cpu.packed.size()) == 0);
    if (!cpu_eq) ++failures;
    if (!gpu_eq) ++failures;
    if (!gc_eq) ++failures;
    std::fprintf(stderr,
                 "[synthetic A] rlen-floor(48)+reorder+partial-byte: CPU==expected %s, "
                 "GPU==expected %s, GPU==CPU %s\n",
                 cpu_eq ? "PASS" : "FAIL", gpu_eq ? "PASS" : "FAIL", gc_eq ? "PASS" : "FAIL");

    return failures;
}

// ---------------------------------------------------------------------------
// (B) REAL-AADR same-data-two-ways. PA (SNP-major GENO) transpose == TGENO slice.
// Returns the number of failures; SKIPs (0 failures) if the data root is absent.
// ---------------------------------------------------------------------------
int run_real_aadr(const std::string& root, ComputeBackend& cpu, ComputeBackend& gpu) {
    const std::string pa_geno = root + "/converted_pa/" + kPaBase + ".geno";
    const std::string tgeno_geno = root + "/raw/" + kTgenoBase + ".geno";
    const std::string tgeno_ind = root + "/raw/" + kTgenoBase + ".ind";

    // Open both readers; SKIP (exit 0) if either file is absent (CI portability —
    // the converted_pa/raw triples live only on box5090).
    std::unique_ptr<steppe::io::GenoReader> pa_reader, tgeno_reader;
    try {
        pa_reader = std::make_unique<steppe::io::GenoReader>(pa_geno);
        tgeno_reader = std::make_unique<steppe::io::GenoReader>(tgeno_geno);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[real B] SKIP (AADR data absent: %s)\n", e.what());
        return 0;
    }
    const auto& pa_hdr = pa_reader->header();
    const auto& tg_hdr = tgeno_reader->header();
    if (pa_hdr.format != steppe::io::GenoFormat::Geno ||
        tg_hdr.format != steppe::io::GenoFormat::Tgeno) {
        std::fprintf(stderr, "[real B] FAIL: expected PA=GENO + raw=TGENO magics (got %d / %d)\n",
                     static_cast<int>(pa_hdr.format), static_cast<int>(tg_hdr.format));
        return 1;
    }
    // Apples-to-apples: same n_ind / n_snp (the same dataset two ways).
    if (pa_hdr.n_ind != tg_hdr.n_ind || pa_hdr.n_snp != tg_hdr.n_snp) {
        std::fprintf(stderr, "[real B] FAIL: PA (%zu ind, %zu snp) != TGENO (%zu ind, %zu snp)\n",
                     pa_hdr.n_ind, pa_hdr.n_snp, tg_hdr.n_ind, tg_hdr.n_snp);
        return 1;
    }
    const std::size_t n_ind = tg_hdr.n_ind;
    const std::size_t K = std::min<std::size_t>(kTier1Snps, tg_hdr.n_snp);
    std::fprintf(stderr, "[real B] PA=GENO + raw=TGENO, n_ind=%zu, first K=%zu SNPs (all inds)\n",
                 n_ind, K);

    // ---- The TGENO oracle tile: first-K-SNP × ALL inds, IDENTITY selection -------
    // One pop containing every individual in file (row) order — the canonical tile the
    // TGENO read_tile already produces (the trusted individual-major reference).
    steppe::io::PopGroup all_pop;
    all_pop.label = "ALL";
    all_pop.rows.resize(n_ind);
    for (std::size_t i = 0; i < n_ind; ++i) all_pop.rows[i] = i;
    steppe::io::IndPartition part;
    part.groups = {all_pop};
    part.n_individuals_total = n_ind;

    steppe::io::GenotypeTile t_tgeno;
    try {
        t_tgeno = tgeno_reader->read_tile(part, 0, K);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[real B] FAIL: TGENO read_tile threw: %s\n", e.what());
        return 1;
    }

    // ---- Gather the PA SNP-major source: first-K SNP records (all inds) ----------
    // PA record s = SNP s at header_bytes + s*pa_bpr; the gather is the SNP-major
    // analogue of read_tile's per-record seek/read loop (format-readers.md §2.3 step 2).
    const std::size_t pa_bpr = pa_hdr.bytes_per_record;  // max(48, ceil(n_ind/4)) = 6899 here
    std::vector<std::uint8_t> pa_src(K * pa_bpr);
    {
        FILE* f = std::fopen(pa_geno.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "[real B] FAIL: cannot reopen PA .geno\n"); return 1; }
        bool short_read = false;
        for (std::size_t s = 0; s < K && !short_read; ++s) {
            const long off = static_cast<long>(pa_hdr.header_bytes) +
                             static_cast<long>(s) * static_cast<long>(pa_bpr);
            if (std::fseek(f, off, SEEK_SET) != 0) { short_read = true; break; }
            const std::size_t got = std::fread(pa_src.data() + s * pa_bpr, 1, pa_bpr, f);
            if (got != pa_bpr) short_read = true;
        }
        std::fclose(f);
        if (short_read) { std::fprintf(stderr, "[real B] FAIL: short read on PA source\n"); return 1; }
    }

    // IDENTITY selection: output column g = source row g (every individual, file order),
    // ONE pop spanning all inds — exactly the TGENO tile's partition.
    std::vector<std::size_t> sel_rows(n_ind);
    for (std::size_t i = 0; i < n_ind; ++i) sel_rows[i] = i;
    const std::vector<std::size_t> pop_offsets = {0, n_ind};

    SnpMajorTileView view;
    view.snp_major = pa_src.data();
    view.src_bytes_per_record = pa_bpr;
    view.n_snp = K;
    view.sel_rows = sel_rows.data();
    view.n_individuals = n_ind;
    view.pop_offsets = pop_offsets.data();
    view.n_pop = 1;
    view.encoding = TileEncoding::Identity;

    const CanonicalTile t_pa_cpu = cpu.transpose_to_canonical(view);
    const CanonicalTile t_pa_gpu = gpu.transpose_to_canonical(view);

    int failures = 0;
    // The transposed PA tile must be byte-identical to the TGENO tile (same data,
    // same canonical packing). Both backends.
    const bool size_ok = (t_pa_gpu.packed.size() == t_tgeno.packed.size()) &&
                         (t_pa_cpu.packed.size() == t_tgeno.packed.size()) &&
                         (t_tgeno.bytes_per_record == steppe::io::packed_bytes(K));
    if (!size_ok) {
        ++failures;
        std::fprintf(stderr, "[real B] FAIL: tile-size mismatch (PA gpu=%zu cpu=%zu, TGENO=%zu)\n",
                     t_pa_gpu.packed.size(), t_pa_cpu.packed.size(), t_tgeno.packed.size());
    } else {
        // PARTIAL-LAST-BYTE TAIL CONVENTION (the ONLY legitimate divergence between
        // the two tile paths, root-caused exhaustively: with K%4==0 the two tiles are
        // byte-identical; with K%4!=0 100% of the diff is confined to the final byte's
        // UNUSED code slots, 0 elsewhere). The transpose ZEROES output SNP slots >= K
        // (s >= n_snp ⇒ 0 bits — the host packer's all-zero tail), but the TGENO
        // read_tile copies the last byte out of a CONTIGUOUS individual-major record,
        // so its unused slots carry the PHYSICAL next-SNP genotype (bits past column K
        // that are never consumed). Both representations are correct; they only differ
        // on bits BEYOND the K-th SNP. To compare the meaningful payload (every full
        // byte + the in-range codes of the partial byte) we mask the TGENO tail to the
        // transpose's zero-tail convention before the memcmp: keep only the top
        // 2*(K%4) MSB-first bits of each individual's last byte, zero the rest. With K
        // a multiple of 4 this mask is a no-op (no partial byte). The K here (kTier1Snps
        // = 4099, NOT a multiple of 4) deliberately exercises this partial last byte.
        const std::size_t obpr = t_tgeno.bytes_per_record;
        const std::size_t tail_codes = K % static_cast<std::size_t>(steppe::io::kCodesPerByte);
        if (tail_codes != 0 && obpr > 0) {
            // top `tail_codes` codes kept ⇒ high 2*tail_codes bits set.
            const int keep_bits = static_cast<int>(tail_codes) * steppe::io::kBitsPerCode;
            const std::uint8_t mask =
                static_cast<std::uint8_t>(0xFFu << (8 - keep_bits));  // MSB-first keep
            for (std::size_t g = 0; g < n_ind; ++g) {
                t_tgeno.packed[g * obpr + (obpr - 1)] &= mask;
            }
        }
        const bool gpu_eq = (std::memcmp(t_pa_gpu.packed.data(), t_tgeno.packed.data(),
                                         t_tgeno.packed.size()) == 0);
        const bool cpu_eq = (std::memcmp(t_pa_cpu.packed.data(), t_tgeno.packed.data(),
                                         t_tgeno.packed.size()) == 0);
        const bool gc_eq = (std::memcmp(t_pa_gpu.packed.data(), t_pa_cpu.packed.data(),
                                        t_pa_gpu.packed.size()) == 0);
        if (!gpu_eq) ++failures;
        if (!cpu_eq) ++failures;
        if (!gc_eq) ++failures;
        std::fprintf(stderr,
                     "[real B] PA-transpose == TGENO-slice (%zu bytes, tail-masked %zu codes): "
                     "GPU %s, CPU %s, GPU==CPU %s\n",
                     t_tgeno.packed.size(), tail_codes, gpu_eq ? "PASS" : "FAIL",
                     cpu_eq ? "PASS" : "FAIL", gc_eq ? "PASS" : "FAIL");
    }
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : kDefaultDataRoot;

    auto cpu = steppe::device::make_cpu_backend();
    auto gpu = steppe::device::make_cuda_backend();

    int failures = 0;
    failures += run_synthetic(*cpu, *gpu);
    failures += run_real_aadr(root, *cpu, *gpu);

    std::fprintf(stderr, "\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
