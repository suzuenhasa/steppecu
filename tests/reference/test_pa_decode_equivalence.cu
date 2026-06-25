// tests/reference/test_pa_decode_equivalence.cu
//
// M-FR-2 TIER-1 GATE — the FULL PA reader end-to-end == the TGENO reader, BIT-EXACT
// (docs/design/format-readers.md §5.1, M-FR-2). This is the keystone of the format-
// reader work: it drives the PRODUCTION dispatch (core::read_canonical_tile) on a
// GENO (SNP-major PACKEDANCESTRYMAP) prefix and asserts it reproduces the raw-TGENO
// (individual-major) reader BYTE-FOR-BYTE — same canonical tile, same decoded Q/V/N,
// same ploidy. It is the format-reader analogue of test_decode_equivalence.cu's
// GPU==CPU max|Δ|==0, with NO new statistical golden and NO AT2 rerun.
//
// WHY IT IS A BIT-EXACT INTEGER COMPARE (no tolerance, no oracle file): every AT2
// golden was generated from the convertf-PA v66_HO_pa PACKEDANCESTRYMAP fixture, and
// steppe reads the raw v66 TGENO (v66.p1_HO.aadr.patch.PUB) — SAME inds, SAME SNPs,
// SAME .ind/.snp. convertf is a LOSSLESS transcode, so the two files are the SAME
// genotypes on two axes. A correct PA reader MUST therefore produce the byte-identical
// canonical tile and the bit-identical Q/V/N. The transpose/gather/encoding are pure
// integer/bit ops (the emulated-FP64 policy is matmul-only; format-readers.md §2.2),
// so equality is by construction.
//
// WHAT IT DOES (data root as argv[1], default /workspace/data/aadr):
//   1. Open the PA reader (converted_pa/v66_HO_pa.geno, magic GENO) + the raw TGENO
//      reader (raw/v66.p1_HO.aadr.patch.PUB.geno, magic TGENO). SKIP exit-0 if absent.
//   2. Reproduce the SAME selection both ways: auto-top-K pops over the SHARED .ind,
//      first-K SNPs in file order. (PA and TGENO share the .ind/.snp, so read_ind on
//      either yields the same partition.)
//   3. PATH A (oracle): tgeno_reader -> read_canonical_tile (dispatches to read_tile)
//      -> T_tgeno. PATH B (the new full reader): pa_reader -> read_canonical_tile
//      (dispatches to read_snp_major_tile + the on-device transpose) -> T_pa.
//   4. ASSERT the canonical tiles are byte-identical (memcmp==0) after masking the
//      partial-last-byte tail (the only legitimate divergence — see the note inline),
//      and pop_offsets identical.
//   5. ASSERT detect_sample_ploidy(T_pa) == detect_sample_ploidy(T_tgeno) (memcmp==0).
//   6. ASSERT decode_af(T_pa) == decode_af(T_tgeno) for Q/V/N, memcmp==0, on BOTH the
//      GPU backend and the CPU oracle. The decode reads ONLY SNPs [0, n_snp), so the
//      tail bits never enter Q/V/N — the decoded-equivalence is unconditional (even
//      without the tail mask), which is the strongest end-to-end statement.
//
// Exits NONZERO on any failure (CTest gates on the exit code). Self-checking main()
// — NOT a GoogleTest TU (mirrors test_decode_equivalence.cu / test_transpose_canonical.cu).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/stats/read_canonical_tile.hpp"  // the M-FR-2 production format dispatch
#include "device/backend.hpp"                  // ComputeBackend, DecodeTileView, DecodeResult
#include "device/backend_factory.hpp"          // make_cpu_backend / make_cuda_backend

#include "io/eigenstrat_format.hpp"            // GenoFormat, packed_bytes, kCodesPerByte, kBitsPerCode
#include "io/geno_reader.hpp"                  // GenoReader, read_tile / read_snp_major_tile
#include "io/genotype_tile.hpp"                // GenotypeTile
#include "io/ind_reader.hpp"                   // read_ind, PopSelection, IndPartition
#include "io/ploidy_detect.hpp"               // detect_sample_ploidy (AT2 adjust_pseudohaploid)

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::core::read_canonical_tile;

namespace {

constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kTgenoBase = "v66.p1_HO.aadr.patch.PUB";  // raw TGENO (individual-major)
constexpr const char* kPaBase = "v66_HO_pa";                    // convertf PA (SNP-major GENO)

// Selection: a real auto-top-K pop set + a SNP prefix not a multiple of 4 (so the
// partial-last-byte tail is exercised). K pops over many thousands of SNPs spans many
// output bytes while keeping the PA SNP-major gather I/O bounded.
constexpr std::size_t kAutoTopK = 50;
constexpr std::size_t kSnps = 4099;  // not a multiple of 4

// Decode both tiles to Q/V/N via `be` (forced auto-detect ploidy on the tile's own
// sample_ploidy vector, set by the caller). Returns the DecodeResult.
DecodeResult decode_tile(ComputeBackend& be, const steppe::io::GenotypeTile& tile,
                         const std::vector<int>& ploidy) {
    DecodeTileView v;
    v.packed = tile.packed.data();
    v.bytes_per_record = tile.bytes_per_record;
    v.n_snp = tile.n_snp;
    v.n_individuals = tile.n_individuals;
    v.pop_offsets = tile.pop_offsets.data();
    v.n_pop = static_cast<int>(tile.n_pop());
    v.sample_ploidy = ploidy.data();
    v.ploidy = 2;
    return be.decode_af(v);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : kDefaultDataRoot;
    const std::string pa_geno = root + "/converted_pa/" + kPaBase + ".geno";
    const std::string tg_geno = root + "/raw/" + kTgenoBase + ".geno";
    const std::string tg_ind = root + "/raw/" + kTgenoBase + ".ind";
    const std::string tg_snp = root + "/raw/" + kTgenoBase + ".snp";

    // ---- (0) Open both readers; SKIP exit-0 if either triple is absent (CI portability:
    // the converted_pa/raw files live only on box5090) --------------------------------
    std::unique_ptr<steppe::io::GenoReader> pa_reader, tg_reader;
    try {
        pa_reader = std::make_unique<steppe::io::GenoReader>(pa_geno);
        tg_reader = std::make_unique<steppe::io::GenoReader>(tg_geno);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[pa-decode-eq] SKIP (AADR data absent: %s)\n", e.what());
        return EXIT_SUCCESS;
    }
    const auto& pa_hdr = pa_reader->header();
    const auto& tg_hdr = tg_reader->header();
    if (pa_hdr.format != steppe::io::GenoFormat::Geno ||
        tg_hdr.format != steppe::io::GenoFormat::Tgeno) {
        std::fprintf(stderr, "[pa-decode-eq] FAIL: expected PA=GENO + raw=TGENO (got %d / %d)\n",
                     static_cast<int>(pa_hdr.format), static_cast<int>(tg_hdr.format));
        return EXIT_FAILURE;
    }
    if (pa_hdr.n_ind != tg_hdr.n_ind || pa_hdr.n_snp != tg_hdr.n_snp) {
        std::fprintf(stderr, "[pa-decode-eq] FAIL: PA(%zu ind,%zu snp) != TGENO(%zu ind,%zu snp)\n",
                     pa_hdr.n_ind, pa_hdr.n_snp, tg_hdr.n_ind, tg_hdr.n_snp);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "[pa-decode-eq] PA=GENO(bpr=%zu,hdr=%zu) raw=TGENO(bpr=%zu,hdr=%zu) "
                 "n_ind=%zu n_snp=%zu\n",
                 pa_hdr.bytes_per_record, pa_hdr.header_bytes,
                 tg_hdr.bytes_per_record, tg_hdr.header_bytes, tg_hdr.n_ind, tg_hdr.n_snp);

    // ---- (1) Reproduce the SAME selection both ways (shared .ind) --------------------
    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::AutoTopK;
    sel.k = kAutoTopK;
    steppe::io::IndPartition pa_part, tg_part;
    try {
        // The PA file's n_records is its n_snp (SNP-major); cap the .ind by the
        // INDIVIDUAL count for both (read_ind caps the individual axis). Use n_ind so
        // the partition is the same on both paths (the .ind is shared).
        pa_part = steppe::io::read_ind(tg_ind, sel, pa_hdr.n_ind);
        tg_part = steppe::io::read_ind(tg_ind, sel, tg_reader->records_present());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[pa-decode-eq] FAIL: read_ind threw: %s\n", e.what());
        return EXIT_FAILURE;
    }
    const std::size_t K = std::min<std::size_t>(kSnps, tg_hdr.n_snp);
    std::fprintf(stderr, "[pa-decode-eq] auto-top %zu -> P=%zu pops, K=%zu SNPs\n",
                 kAutoTopK, tg_part.groups.size(), K);

    auto cpu = steppe::device::make_cpu_backend();
    auto gpu = steppe::device::make_cuda_backend();

    int failures = 0;

    // ---- (2) Build BOTH canonical tiles through the PRODUCTION dispatch --------------
    // PATH A (oracle): TGENO -> read_canonical_tile -> read_tile.
    // PATH B (new reader): GENO  -> read_canonical_tile -> read_snp_major_tile + the
    //                       on-device transpose (run on the GPU backend).
    steppe::io::GenotypeTile T_tgeno, T_pa;
    try {
        T_tgeno = read_canonical_tile(*tg_reader, tg_part, *gpu, 0, K);
        T_pa = read_canonical_tile(*pa_reader, pa_part, *gpu, 0, K);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[pa-decode-eq] FAIL: read_canonical_tile threw: %s\n", e.what());
        return EXIT_FAILURE;
    }

    const bool shape_ok =
        (T_pa.n_individuals == T_tgeno.n_individuals) &&
        (T_pa.n_snp == T_tgeno.n_snp) &&
        (T_pa.bytes_per_record == T_tgeno.bytes_per_record) &&
        (T_pa.packed.size() == T_tgeno.packed.size()) &&
        (T_pa.pop_offsets == T_tgeno.pop_offsets) &&
        (T_pa.pop_labels == T_tgeno.pop_labels);
    if (!shape_ok) {
        ++failures;
        std::fprintf(stderr, "[pa-decode-eq] FAIL: tile SHAPE/partition mismatch "
                     "(PA ind=%zu snp=%zu bpr=%zu pop=%zu | TGENO ind=%zu snp=%zu bpr=%zu pop=%zu)\n",
                     T_pa.n_individuals, T_pa.n_snp, T_pa.bytes_per_record, T_pa.n_pop(),
                     T_tgeno.n_individuals, T_tgeno.n_snp, T_tgeno.bytes_per_record, T_tgeno.n_pop());
    }

    // ---- (3) Canonical tile BYTE-EXACT (tail-masked) ---------------------------------
    // The ONLY legitimate divergence between the transpose tile and the read_tile tile
    // is the PARTIAL-LAST-BYTE tail: the transpose ZEROES output SNP slots >= K, but the
    // TGENO read_tile copies the last byte out of a CONTIGUOUS individual-major record,
    // so its unused slots carry the PHYSICAL next-SNP genotype (bits past column K that
    // decode never reads). Both are correct; mask the TGENO tail to the transpose's
    // zero-tail convention before the memcmp (no-op when K%4==0). K=4099 (%4 != 0)
    // deliberately exercises this.
    if (shape_ok) {
        std::vector<std::uint8_t> tg_masked = T_tgeno.packed;
        const std::size_t obpr = T_tgeno.bytes_per_record;
        const std::size_t tail_codes = K % static_cast<std::size_t>(steppe::io::kCodesPerByte);
        if (tail_codes != 0 && obpr > 0) {
            const int keep_bits = static_cast<int>(tail_codes) * steppe::io::kBitsPerCode;
            const std::uint8_t mask = static_cast<std::uint8_t>(0xFFu << (8 - keep_bits));
            for (std::size_t g = 0; g < T_tgeno.n_individuals; ++g)
                tg_masked[g * obpr + (obpr - 1)] &= mask;
        }
        const bool tile_eq = (std::memcmp(T_pa.packed.data(), tg_masked.data(),
                                          tg_masked.size()) == 0);
        if (!tile_eq) ++failures;
        std::fprintf(stderr, "[pa-decode-eq] canonical tile (%zu bytes, tail-masked %zu codes): %s\n",
                     tg_masked.size(), tail_codes, tile_eq ? "BIT-EXACT (PASS)" : "MISMATCH (FAIL)");
    }

    // ---- (4) Ploidy bit-exact (AT2 adjust_pseudohaploid on each canonical tile) ------
    const std::vector<int> ploidy_tgeno = steppe::io::detect_sample_ploidy(T_tgeno);
    const std::vector<int> ploidy_pa = steppe::io::detect_sample_ploidy(T_pa);
    const bool ploidy_eq = (ploidy_pa.size() == ploidy_tgeno.size()) &&
                           (std::memcmp(ploidy_pa.data(), ploidy_tgeno.data(),
                                        ploidy_tgeno.size() * sizeof(int)) == 0);
    if (!ploidy_eq) ++failures;
    std::size_t n_ph = 0, n_dip = 0;
    for (int pl : ploidy_tgeno) (pl == 1 ? n_ph : n_dip)++;
    std::fprintf(stderr, "[pa-decode-eq] sample_ploidy (%zu samples: %zu ph + %zu dip): %s\n",
                 ploidy_tgeno.size(), n_ph, n_dip, ploidy_eq ? "BIT-EXACT (PASS)" : "MISMATCH (FAIL)");

    // ---- (5) DECODED Q/V/N bit-exact on BOTH backends (the keystone) -----------------
    // decode reads ONLY SNPs [0, K), so the partial-tail bits never enter Q/V/N — this
    // is the UNCONDITIONAL end-to-end equivalence (memcmp==0, no tail mask needed).
    if (shape_ok && ploidy_eq) {
        for (ComputeBackend* be : {static_cast<ComputeBackend*>(gpu.get()),
                                   static_cast<ComputeBackend*>(cpu.get())}) {
            const DecodeResult d_tgeno = decode_tile(*be, T_tgeno, ploidy_tgeno);
            const DecodeResult d_pa = decode_tile(*be, T_pa, ploidy_pa);
            const bool q_eq = (d_pa.q.size() == d_tgeno.q.size()) &&
                              (std::memcmp(d_pa.q.data(), d_tgeno.q.data(),
                                           d_tgeno.q.size() * sizeof(double)) == 0);
            const bool v_eq = (d_pa.v.size() == d_tgeno.v.size()) &&
                              (std::memcmp(d_pa.v.data(), d_tgeno.v.data(),
                                           d_tgeno.v.size() * sizeof(double)) == 0);
            const bool n_eq = (d_pa.n.size() == d_tgeno.n.size()) &&
                              (std::memcmp(d_pa.n.data(), d_tgeno.n.data(),
                                           d_tgeno.n.size() * sizeof(double)) == 0);
            const bool all_eq = q_eq && v_eq && n_eq;
            if (!all_eq) ++failures;
            std::fprintf(stderr, "[pa-decode-eq] decode_af PA==TGENO (%s): Q %s V %s N %s\n",
                         be == gpu.get() ? "GPU" : "CPU",
                         q_eq ? "ok" : "FAIL", v_eq ? "ok" : "FAIL", n_eq ? "ok" : "FAIL");
        }
    }

    std::fprintf(stderr, "\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
