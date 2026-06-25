// tests/unit/test_geno_reader.cpp
//
// Host-only unit test of io::GenoReader (architecture.md §13 "Unit tests";
// ROADMAP M1, §5). Pure C++ TU, NO GPU, NO real AADR data: it writes tiny
// SYNTHETIC TGENO .geno files to a temp path and exercises the reader's
// happy-path gather plus its FAIL-FAST boundary checks — control flow / boundary
// logic, not a precision claim (ROADMAP §0: the no-synthetic rule is for
// precision claims; a hand-packed 4-individual TGENO record is a layout fixture).
// The numerical-accuracy gate stays the real-AADR equivalence test
// (tests/reference/test_decode_equivalence.cu).
//
// PRIMARY PURPOSE (cleanup geno_reader B17 verdict gate): assert that a
// HOSTILE / OVERSIZED partition THROWS rather than silently corrupting the heap.
// Before the B17 fix, `read_tile` computed `n_ind * bytes_per_rec` as an
// unguarded std::size_t product (can WRAP) and never checked `row <
// records_present_`, so a stale/hostile IndPartition produced a too-small buffer
// and the gather wrote PAST the allocation (silent heap-buffer-overflow). The fix
// (a) rejects an empty partition, (b) validates every row < records_present_, and
// (c) adds a checked-multiply guard before resize. This test pins all three plus
// a normal partition that decodes correctly.
//
// Dual harness (identical to tests/unit/test_filters.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise a self-checking main()
// returning non-zero on the first failure — all CTest needs to gate. No CUDA.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "io/eigenstrat_format.hpp"  // kGenoHeaderBytes, packed_bytes
#include "io/geno_reader.hpp"        // GenoReader
#include "io/ind_reader.hpp"         // IndPartition, PopGroup

namespace {

using steppe::io::GenoReader;
using steppe::io::GenotypeTile;
using steppe::io::IndPartition;
using steppe::io::PopGroup;
using steppe::io::kGenoHeaderBytes;
using steppe::io::packed_bytes;

// ---- synthetic-TGENO writer --------------------------------------------------
// Write a TGENO .geno: a kGenoHeaderBytes-byte "TGENO <n_ind> <n_snp> ..." header
// (NUL-padded to the fixed width), followed by `records_on_disk` records each of
// packed_bytes(n_snp) bytes. Each record's bytes are filled with a deterministic,
// per-individual marker so the gather can be checked exactly: individual `i`'s
// byte `b` is (uint8_t)(i * 31 + b + 1) (nonzero, distinct per (i,b) for the
// small fixtures used here). `records_on_disk` lets a test write a SHORT file
// (fewer records than the header claims) to drive records_present_ < n_records.
[[nodiscard]] std::vector<std::uint8_t> record_bytes(std::size_t ind,
                                                     std::size_t bytes_per_rec) {
    std::vector<std::uint8_t> rec(bytes_per_rec);
    for (std::size_t b = 0; b < bytes_per_rec; ++b) {
        rec[b] = static_cast<std::uint8_t>((ind * 31u + b + 1u) & 0xFFu);
    }
    return rec;
}

void write_tgeno(const std::filesystem::path& path, std::size_t n_ind,
                 std::size_t n_snp, std::size_t records_on_disk) {
    std::array<char, kGenoHeaderBytes> head{};  // zero-initialized = NUL padding
    const std::string text =
        "TGENO " + std::to_string(n_ind) + " " + std::to_string(n_snp) + " hash1 hash2";
    // text fits the 48-byte header for the small fixtures used here.
    for (std::size_t i = 0; i < text.size() && i < kGenoHeaderBytes; ++i) {
        head[i] = text[i];
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(head.data(), static_cast<std::streamsize>(kGenoHeaderBytes));
    const std::size_t bpr = packed_bytes(n_snp);
    for (std::size_t i = 0; i < records_on_disk; ++i) {
        const std::vector<std::uint8_t> rec = record_bytes(i, bpr);
        out.write(reinterpret_cast<const char*>(rec.data()),
                  static_cast<std::streamsize>(bpr));
    }
    out.close();
}

// A unique temp path per fixture so parallel ctest invocations do not collide.
[[nodiscard]] std::filesystem::path temp_geno(const char* tag) {
    static int counter = 0;
    auto p = std::filesystem::temp_directory_path();
    p /= ("steppe_geno_reader_test_" + std::string(tag) + "_" +
          std::to_string(++counter) + ".geno");
    return p;
}

struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const char* tag) : path(temp_geno(tag)) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

[[nodiscard]] PopGroup group(std::string label, std::vector<std::size_t> rows) {
    PopGroup g;
    g.label = std::move(label);
    g.rows = std::move(rows);
    return g;
}

[[nodiscard]] bool threw_runtime_error(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;  // wrong exception type fails the contract
    }
    return false;  // no throw fails the test
}

// ---- (1) NORMAL partition decodes correctly ----------------------------------
// 6 individuals × 10 SNPs (packed_bytes(10)==3 bytes/record). Select two
// populations out of order to confirm pop-contiguous gather + pop_offsets/labels.
[[nodiscard]] bool test_normal_decode() {
    const std::size_t n_ind = 6, n_snp = 10;
    const std::size_t bpr = packed_bytes(n_snp);  // 3
    TempFile f("normal");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/n_ind);

    GenoReader reader(f.path.string());
    if (reader.records_present() != n_ind) return false;
    if (reader.header().bytes_per_record != bpr) return false;

    IndPartition part;
    part.groups.push_back(group("popB", {4, 5}));  // 2 individuals (rows 4,5)
    part.groups.push_back(group("popA", {0, 2}));  // 2 individuals (rows 0,2)
    part.n_individuals_total = n_ind;

    GenotypeTile tile = reader.read_tile(part, 0, n_snp);

    if (tile.n_individuals != 4) return false;
    if (tile.n_snp != n_snp) return false;
    if (tile.bytes_per_record != bpr) return false;
    if (tile.packed.size() != 4 * bpr) return false;
    if (tile.n_pop() != 2) return false;
    if (tile.pop_labels.size() != 2 || tile.pop_labels[0] != "popB" ||
        tile.pop_labels[1] != "popA") {
        return false;
    }
    // pop_offsets: {0, 2, 4} (2 individuals each, in groups order).
    if (tile.pop_offsets.size() != 3 || tile.pop_offsets[0] != 0 ||
        tile.pop_offsets[1] != 2 || tile.pop_offsets[2] != 4) {
        return false;
    }

    // Gathered order is pop-contiguous in groups order: rows 4,5,0,2.
    const std::size_t gathered_rows[] = {4, 5, 0, 2};
    for (std::size_t g = 0; g < 4; ++g) {
        const std::vector<std::uint8_t> expect = record_bytes(gathered_rows[g], bpr);
        for (std::size_t b = 0; b < bpr; ++b) {
            if (tile.packed[g * bpr + b] != expect[b]) return false;
        }
    }
    return true;
}

// ---- (2) HOSTILE / OVERSIZED partition THROWS (the B17 heap-corruption gate) --
// A partition with a row >= records_present_ is exactly the stale/wrong-dataset
// partition that, pre-fix, seeked into trailing junk and (with enough such rows)
// wrapped the size multiply ⇒ too-small buffer ⇒ gather writes PAST it. The fix
// rejects it at the boundary with a thrown std::runtime_error BEFORE any resize
// or write — so no heap corruption can occur. We assert the THROW.
[[nodiscard]] bool test_hostile_oversized_partition_throws() {
    const std::size_t n_ind = 4, n_snp = 8;
    TempFile f("hostile");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/n_ind);

    GenoReader reader(f.path.string());
    if (reader.records_present() != n_ind) return false;

    // Row n_ind is one past the last present record; a far-out row models a
    // partition built against a stale/larger dataset.
    IndPartition bad1;
    bad1.groups.push_back(group("pop", {0, n_ind}));  // n_ind == 4 is out of range
    if (!threw_runtime_error([&] { (void)reader.read_tile(bad1, 0, n_snp); })) return false;

    IndPartition bad2;
    // A hugely oversized row — the kind that, combined with duplicates, drove the
    // size-multiply wrap. The row-bound guard rejects it before any allocation.
    bad2.groups.push_back(group("pop", {static_cast<std::size_t>(1) << 40}));
    if (!threw_runtime_error([&] { (void)reader.read_tile(bad2, 0, n_snp); })) return false;

    // SANITY: after the throws, a VALID partition on the SAME reader still decodes
    // (the reader was not left in a broken state, and nothing was corrupted).
    IndPartition good;
    good.groups.push_back(group("pop", {0, 1, 2, 3}));
    GenotypeTile tile = reader.read_tile(good, 0, n_snp);
    if (tile.n_individuals != 4) return false;
    const std::size_t bpr = packed_bytes(n_snp);
    for (std::size_t g = 0; g < 4; ++g) {
        const std::vector<std::uint8_t> expect = record_bytes(g, bpr);
        for (std::size_t b = 0; b < bpr; ++b) {
            if (tile.packed[g * bpr + b] != expect[b]) return false;
        }
    }
    return true;
}

// ---- (3) EMPTY partition THROWS ----------------------------------------------
// read_ind throws on an empty selection, but read_tile accepts an arbitrary
// partition; a zero-population partition must fail-fast, not yield a 0-pop tile.
[[nodiscard]] bool test_empty_partition_throws() {
    const std::size_t n_ind = 3, n_snp = 5;
    TempFile f("empty");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/n_ind);
    GenoReader reader(f.path.string());

    IndPartition empty;  // groups is empty
    return threw_runtime_error([&] { (void)reader.read_tile(empty, 0, n_snp); });
}

// ---- (4) SNP-range and partial-file edge cases (boundary correctness) ---------
[[nodiscard]] bool test_range_and_partial_edges() {
    const std::size_t n_ind = 5, n_snp = 12;
    // Header claims 5 records; write only 3 on disk -> records_present_ == 3.
    TempFile f("partial");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/3);
    GenoReader reader(f.path.string());
    if (reader.records_present() != 3) return false;  // partial-file cap (clamped down)

    IndPartition part;
    part.groups.push_back(group("pop", {0, 1, 2}));  // all in range

    // snp_begin != 0 rejected (M1 byte-aligned prefix only).
    if (!threw_runtime_error([&] { (void)reader.read_tile(part, 1, n_snp); })) return false;
    // snp_end > n_snp rejected.
    if (!threw_runtime_error([&] { (void)reader.read_tile(part, 0, n_snp + 1); })) return false;
    // snp_end == 0 (degenerate range) rejected.
    if (!threw_runtime_error([&] { (void)reader.read_tile(part, 0, 0); })) return false;

    // A row that is valid against the HEADER (row 3 < n_ind==5) but NOT present on
    // disk (>= records_present_==3) must throw — the silent wrong-individual /
    // heap-overflow path the B17 fix closes.
    IndPartition stale;
    stale.groups.push_back(group("pop", {0, 3}));  // row 3 >= records_present_(3)
    if (!threw_runtime_error([&] { (void)reader.read_tile(stale, 0, n_snp); })) return false;

    // A partial SNP prefix decodes (read fewer SNPs than the record holds).
    GenotypeTile tile = reader.read_tile(part, 0, 4);  // 1 byte/record prefix
    if (tile.bytes_per_record != packed_bytes(4)) return false;
    if (tile.n_individuals != 3) return false;
    if (tile.packed.size() != 3 * packed_bytes(4)) return false;
    return true;
}

// ---- (5) Constructor fail-fast on a degenerate / bad file --------------------
[[nodiscard]] bool test_ctor_failures() {
    // Missing file.
    if (!threw_runtime_error([] { GenoReader r("/nonexistent/steppe/no_such.geno"); })) {
        return false;
    }
    // File shorter than the 48-byte header.
    {
        TempFile f("shorthdr");
        std::ofstream out(f.path, std::ios::binary | std::ios::trunc);
        const char junk[8] = {'T', 'G', 'E', 'N', 'O', ' ', '1', ' '};
        out.write(junk, sizeof(junk));
        out.close();
        if (!threw_runtime_error([&] { GenoReader r(f.path.string()); })) return false;
    }
    // Valid header but no complete records on disk (header-only, 48 bytes exactly).
    {
        TempFile f("hdronly");
        write_tgeno(f.path, /*n_ind=*/4, /*n_snp=*/8, /*records_on_disk=*/0);
        if (!threw_runtime_error([&] { GenoReader r(f.path.string()); })) return false;
    }
    return true;
}

// ---- (6) OVERSIZED tile allocation surfaces as std::runtime_error (B18 2.1) ---
// The checked-multiply guard rules out a SILENT size_t WRAP, but a large-but-
// NON-wrapping tile (n_ind * bytes_per_record below SIZE_MAX yet far beyond any
// machine's RAM) still makes `tile.packed.resize` throw — and the stdlib throws
// std::length_error (a logic_error, when sz > vector::max_size()) or
// std::bad_alloc, NEITHER of which derives from std::runtime_error. The header
// contract (geno_reader.hpp:17-18) promises "I/O failures surface as
// std::runtime_error", so the B18 fix translates BOTH into std::runtime_error.
// This pins that contract: a multi-terabyte tile request (every row VALID and
// below records_present_, the product NON-wrapping) must throw a
// std::runtime_error, not leak a raw std::bad_alloc / std::length_error.
//
// Construction: a 1 MB/record file (n_snp = 4,000,000 -> bytes_per_record =
// packed_bytes = 1,000,000) with 2 records on disk (~2 MB temp file). A partition
// of 8,000,000 DUPLICATE valid rows (row 0, < records_present_ == 2) gives
// n_ind * bytes_per_record = 8e6 * 1e6 = 8e12 bytes (~8 TB): non-wrapping (8e12 <
// SIZE_MAX / 1e6 -> wrap guard passes it through) and far above any test box's RAM
// (the box has 251 GB) -> the resize allocation fails -> translated runtime_error.
[[nodiscard]] bool test_oversized_tile_alloc_throws_runtime_error() {
    const std::size_t n_ind_hdr = 2, n_snp = 4'000'000;
    const std::size_t bpr = packed_bytes(n_snp);  // 1,000,000 bytes/record
    if (bpr != 1'000'000) return false;            // pin the fixture geometry
    TempFile f("oversize");
    write_tgeno(f.path, n_ind_hdr, n_snp, /*records_on_disk=*/n_ind_hdr);

    GenoReader reader(f.path.string());
    if (reader.records_present() != n_ind_hdr) return false;  // 2 records present

    // 8,000,000 copies of the VALID row 0 (< records_present_): the row-bound and
    // checked-multiply guards both PASS, so control reaches `resize(8e12)`.
    constexpr std::size_t kDupRows = 8'000'000;
    IndPartition big;
    big.groups.push_back(group("pop", std::vector<std::size_t>(kDupRows, 0)));

    // The ~8 TB allocation must fail and surface as a std::runtime_error (the B18
    // exception-type contract), NOT a raw std::bad_alloc / std::length_error.
    return threw_runtime_error([&] { (void)reader.read_tile(big, 0, n_snp); });
}

// ---- (7) Malformed-header decimal digits are handled (B18 eigenstrat C2) ------
// `parse_geno_header` accumulates the two counts as `v = v*10 + digit` over a
// std::size_t, which WRAPS modulo 2^64 (well-defined-but-SILENT). A malformed /
// adversarial header whose count is a digit run far longer than any real 48-byte
// file could hold would otherwise yield a wrapped-but-plausible n_ind/n_snp that
// flows into packed_bytes()/the size validation as a wrong-but-plausible stride.
// The B18 fix detects the wrap (v > (SIZE_MAX - d)/10) and routes the whole header
// to GenoFormat::Unknown. This pins: (a) a normal TGENO header parses both counts;
// (b) an overflowing count -> Unknown; (c) the GenoReader ctor on an overflowing-
// count header throws std::runtime_error ("unrecognized magic" path, since
// Unknown), not a silent wrong stride.
[[nodiscard]] bool test_malformed_header_digits_handled() {
    using steppe::io::GenoFormat;
    using steppe::io::GenoHeader;
    using steppe::io::parse_geno_header;

    auto make_head = [](const std::string& text) {
        std::array<char, kGenoHeaderBytes> head{};  // NUL-padded
        for (std::size_t i = 0; i < text.size() && i < kGenoHeaderBytes; ++i) {
            head[i] = text[i];
        }
        return head;
    };

    // (a) a well-formed header parses both counts and the derived stride.
    {
        const GenoHeader h = parse_geno_header(make_head("TGENO 6 10 h1 h2"));
        if (h.format != GenoFormat::Tgeno) return false;
        if (h.n_ind != 6 || h.n_snp != 10) return false;
        if (h.bytes_per_record != packed_bytes(10)) return false;
    }

    // (b) an n_ind digit run that overflows size_t -> Unknown (NOT a wrapped count).
    {
        // 25 nines: 9.99...e24, far above SIZE_MAX (1.8e19). Fits the 48-byte header.
        const std::string overflow_nine(25, '9');
        const GenoHeader h =
            parse_geno_header(make_head("TGENO " + overflow_nine + " 10 h1 h2"));
        if (h.format != GenoFormat::Unknown) return false;
    }

    // (b') an n_snp overflow (second count) is caught too -> Unknown.
    {
        const std::string overflow_nine(30, '9');
        const GenoHeader h =
            parse_geno_header(make_head("TGENO 6 " + overflow_nine + " h1 h2"));
        if (h.format != GenoFormat::Unknown) return false;
    }

    // (c) a header whose count overflows surfaces through the GenoReader ctor as a
    // std::runtime_error (Unknown -> the "unrecognized magic" reject), not a
    // silently-wrapped stride that would mis-validate the file size.
    {
        const std::string overflow_nine(25, '9');
        TempFile f("ovflhdr");
        std::array<char, kGenoHeaderBytes> head =
            make_head("TGENO " + overflow_nine + " 10 h1 h2");
        std::ofstream out(f.path, std::ios::binary | std::ios::trunc);
        out.write(head.data(), static_cast<std::streamsize>(kGenoHeaderBytes));
        // a few data bytes so the file is not rejected merely for being short.
        const char pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        out.write(pad, sizeof(pad));
        out.close();
        if (!threw_runtime_error([&] { GenoReader r(f.path.string()); })) return false;
    }

    return true;
}

// GENO (SNP-major PACKEDANCESTRYMAP) HEADER GEOMETRY (the M-FR-1 correctness pin;
// format-readers.md §3.4 + the M-FR-1 GENO-header fix). The whole format-reader
// transpose rides on the GENO header geometry being parsed RIGHT: GENO writes its
// header into ONE FULL rlen-width record, so `header_bytes == bytes_per_record ==
// max(kGenoHeaderBytes, ceil(n_ind/4))` — NOT a fixed 48. A 48-byte assumption
// mis-seeks every GENO record by (rlen - 48). This pins both axes against TGENO:
//   * SMALL n_ind  -> the rlen floor fires: header_bytes == bytes_per_record == 48.
//   * LARGE n_ind  -> header_bytes == bytes_per_record == ceil(n_ind/4) (> 48),
//     matching the empirically-verified v66 6899 (n_ind=27594 -> ceil/4 = 6899).
//   * n_records == n_snp (the SNP-major axis), vs TGENO's n_records == n_ind.
// TGENO's header stays the fixed 48-byte record (its geometry is unaffected).
[[nodiscard]] bool test_geno_header_geometry() {
    using steppe::io::GenoFormat;
    using steppe::io::GenoHeader;
    using steppe::io::parse_geno_header;

    auto make_head = [](const std::string& text) {
        std::array<char, kGenoHeaderBytes> head{};  // NUL-padded
        for (std::size_t i = 0; i < text.size() && i < kGenoHeaderBytes; ++i) head[i] = text[i];
        return head;
    };

    // (a) SMALL n_ind: the rlen floor fires (ceil(3/4)=1 < 48 -> 48). header_bytes
    // == bytes_per_record == 48; n_records == n_snp.
    {
        const GenoHeader h = parse_geno_header(make_head("GENO 3 7 h1 h2"));
        if (h.format != GenoFormat::Geno) return false;
        if (h.n_ind != 3 || h.n_snp != 7) return false;
        const std::size_t expect_bpr = std::max<std::size_t>(kGenoHeaderBytes, packed_bytes(3));
        if (expect_bpr != kGenoHeaderBytes) return false;  // the floor must fire here
        if (h.bytes_per_record != expect_bpr) return false;
        if (h.header_bytes != expect_bpr) return false;     // header == one full rlen record
        if (h.n_records != h.n_snp) return false;            // SNP-major axis
    }

    // (b) LARGE n_ind (the v66 geometry): ceil(27594/4) = 6899 > 48, so the rlen is
    // 6899 and the header is ONE 6899-byte record — the empirically-verified value.
    {
        const GenoHeader h = parse_geno_header(make_head("GENO 27594 584131 h1 h2"));
        if (h.format != GenoFormat::Geno) return false;
        if (h.n_ind != 27594 || h.n_snp != 584131) return false;
        if (packed_bytes(27594) != 6899) return false;       // pin the derived rlen
        if (h.bytes_per_record != 6899) return false;
        if (h.header_bytes != 6899) return false;            // NOT kGenoHeaderBytes(48)
        if (h.header_bytes == kGenoHeaderBytes) return false;  // the exact bug guarded against
        if (h.n_records != h.n_snp) return false;
    }

    // (c) TGENO is UNAFFECTED: the header stays the fixed 48-byte record; the data
    // record stride is ceil(n_snp/4); n_records == n_ind (individual-major axis).
    {
        const GenoHeader h = parse_geno_header(make_head("TGENO 27594 584131 h1 h2"));
        if (h.format != GenoFormat::Tgeno) return false;
        if (h.header_bytes != kGenoHeaderBytes) return false;       // fixed 48
        if (h.bytes_per_record != packed_bytes(584131)) return false;
        if (h.n_records != h.n_ind) return false;                   // individual-major axis
    }
    return true;
}

// ---- EIGENSTRAT (ASCII SNP-major) reader (M-FR-EIG) --------------------------
// Hand-built tiny ASCII .geno: one line per SNP, one char per individual
// (0/1/2 ref-allele copies, 9 missing). NO packed magic — the ctor detects
// EIGENSTRAT from its leading 0/1/2/9 content and derives n_ind (first line length)
// / n_snp (line count). This is the §8.5 CI-portable fixture (a few inds × a few
// SNPs, like the synthetic TGENO above) — it pins the char->2-bit map, the MSB-first
// SNP-major pack, and the geometry WITHOUT real AADR or a GPU (the real-AADR
// bit-exact gate is tests/reference/test_eigenstrat_decode_equivalence.cu).
[[nodiscard]] bool write_eigenstrat(const std::filesystem::path& p,
                                    const std::vector<std::string>& lines) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    for (const std::string& l : lines) out << l << '\n';
    return static_cast<bool>(out);
}

[[nodiscard]] bool test_eigenstrat_reader() {
    using steppe::io::GenoFormat;
    using steppe::io::SnpMajorTile;
    using steppe::io::packed_bytes;

    // 5 individuals × 3 SNPs. Columns are individuals (0..4); rows are SNPs.
    //   SNP0: 0 1 2 9 0   SNP1: 2 2 9 1 0   SNP2: 1 0 0 2 9
    const std::vector<std::string> lines = {"01290", "22910", "10029"};
    const std::size_t n_ind = 5, n_snp = 3;
    const std::size_t bpr = packed_bytes(n_ind);  // ceil(5/4) == 2

    TempFile f("eigenstrat");
    if (!write_eigenstrat(f.path, lines)) return false;

    GenoReader reader(f.path.string());
    // (a) Geometry: detected as EIGENSTRAT, n_ind/n_snp from the ASCII shape, the
    // canonical SNP-major stride packed_bytes(n_ind), header_bytes 0 (ASCII has no
    // header), records_present == n_snp (the SNP-major convention).
    if (reader.header().format != GenoFormat::Eigenstrat) return false;
    if (reader.header().n_ind != n_ind || reader.header().n_snp != n_snp) return false;
    if (reader.header().bytes_per_record != bpr) return false;
    if (reader.header().header_bytes != 0) return false;
    if (reader.records_present() != n_snp) return false;

    // (b) The canonical SNP-major pack: select all 5 individuals (one pop) and assert
    // the packed bytes match the HAND-COMPUTED MSB-first layout. The char->code map is
    // the identity on the value (0/1/2) with '9'->3, and individual i sits at byte i/4,
    // position i%4, shift (3 - i%4)*2 = 6/4/2/0. For SNP s, byte 0 holds inds 0..3 and
    // byte 1 holds ind 4 (high 2 bits, the rest zero).
    IndPartition part;
    part.groups.push_back(group("pop", {0, 1, 2, 3, 4}));
    part.n_individuals_total = n_ind;
    const SnpMajorTile tile = reader.read_eigenstrat_snp_major_tile(part, 0, n_snp);
    if (tile.n_snp != n_snp || tile.n_individuals != n_ind) return false;
    if (tile.src_bytes_per_record != bpr) return false;
    if (tile.snp_major.size() != n_snp * bpr) return false;

    // Hand-pack each SNP line into the expected 2 bytes (MSB-first, codes 0/1/2/3).
    auto code_of = [](char c) -> std::uint8_t {
        return c == '9' ? std::uint8_t{3} : static_cast<std::uint8_t>(c - '0');
    };
    for (std::size_t s = 0; s < n_snp; ++s) {
        std::uint8_t b0 = 0, b1 = 0;
        for (std::size_t i = 0; i < n_ind; ++i) {
            const std::uint8_t code = code_of(lines[s][i]);
            const int shift = (3 - static_cast<int>(i % 4)) * 2;
            if (i < 4) b0 = static_cast<std::uint8_t>(b0 | (code << shift));
            else       b1 = static_cast<std::uint8_t>(b1 | (code << shift));
        }
        if (tile.snp_major[s * bpr + 0] != b0) return false;
        if (tile.snp_major[s * bpr + 1] != b1) return false;
    }

    // (c) FAIL-FAST: an illegal genotype char ('3' is not 0/1/2/9) makes the ENTIRE
    // file NOT EIGENSTRAT (the geometry scan rejects it), so the ctor throws — never a
    // silent mis-decode.
    {
        TempFile bad("eigenstrat_badchar");
        if (!write_eigenstrat(bad.path, {"01290", "22310", "10029"})) return false;  // '3' at SNP1
        if (!threw_runtime_error([&] { GenoReader r(bad.path.string()); })) return false;
    }
    // (d) FAIL-FAST: a RAGGED line (different individual count) is rejected at ctor
    // (a ragged char matrix is not a SNP-major EIGENSTRAT — would desync the SNP axis).
    {
        TempFile bad("eigenstrat_ragged");
        if (!write_eigenstrat(bad.path, {"01290", "229", "10029"})) return false;  // short line 2
        if (!threw_runtime_error([&] { GenoReader r(bad.path.string()); })) return false;
    }
    // (e) FAIL-FAST: calling the EIGENSTRAT gather with an out-of-range individual row
    // (>= n_ind) throws (the phantom-individual guard).
    {
        IndPartition bad_part;
        bad_part.groups.push_back(group("pop", {0, 5}));  // row 5 >= n_ind(5)
        bad_part.n_individuals_total = n_ind;
        if (!threw_runtime_error(
                [&] { (void)reader.read_eigenstrat_snp_major_tile(bad_part, 0, n_snp); }))
            return false;
    }
    return true;
}

struct Case { const char* name; bool (*fn)(); };

constexpr Case kCases[] = {
    {"normal partition decodes correctly", test_normal_decode},
    {"hostile/oversized partition THROWS (B17 heap-corruption gate)",
     test_hostile_oversized_partition_throws},
    {"empty partition THROWS", test_empty_partition_throws},
    {"SNP-range / partial-file edges", test_range_and_partial_edges},
    {"constructor fail-fast (missing/short/degenerate)", test_ctor_failures},
    {"oversized tile alloc -> std::runtime_error (B18 exception-type contract)",
     test_oversized_tile_alloc_throws_runtime_error},
    {"malformed header decimal digits handled (B18 overflow guard)",
     test_malformed_header_digits_handled},
    {"GENO header geometry (rlen floor + header==rlen record, M-FR-1 fix)",
     test_geno_header_geometry},
    {"EIGENSTRAT reader (ASCII detect + char->2-bit MSB-first pack + fail-fast, M-FR-EIG)",
     test_eigenstrat_reader},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(GenoReader, AllCases) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_geno_reader: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_geno_reader: all %zu cases PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif
