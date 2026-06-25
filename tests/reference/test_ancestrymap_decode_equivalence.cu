// tests/reference/test_ancestrymap_decode_equivalence.cu
//
// M-FR-AM TIER-1 GATE — the ANCESTRYMAP (unpacked legacy EIGENSOFT, TEXT triple)
// reader == the PA (PACKEDANCESTRYMAP / GENO SNP-major) reader, BIT-EXACT
// (docs/design/format-readers.md, the Tier-3-by-reduction-to-PA gate; the
// EIGENSTRAT/PLINK precedent). This is the keystone of the ANCESTRYMAP work: it
// drives the PRODUCTION dispatch (core::read_canonical_tile) on an ANCESTRYMAP .geno
// AND on the convertf-PA .geno of the SAME subset, and asserts it reproduces the PA
// decode BYTE-FOR-BYTE — same canonical tile, same auto-detected ploidy, same decoded
// Q/V/N. Since PA == TGENO == the AT2 goldens (proven bit-exact by
// test_pa_decode_equivalence, M-FR-2), this transitively pins ANCESTRYMAP to the
// goldens with NO new statistical golden + NO AT2 rerun.
//
// WHY IT IS A BIT-EXACT INTEGER COMPARE (no tolerance, no oracle file): the two
// fixtures are convertf outputs from the SAME PA source with the SAME poplist (the 9
// fit0 pops), so they carry the SAME individuals (identical .ind), the SAME SNPs
// (identical .snp), and the SAME genotypes on two encodings — packed-SNP-major vs the
// ANCESTRYMAP text triple `<snp_id> <sample_id> <genotype>`. convertf is a lossless
// transcode, so a correct ANCESTRYMAP reader MUST produce the byte-identical canonical
// tile and the bit-identical Q/V/N. The transpose/gather/encoding are pure integer/bit
// ops (the emulated-FP64 policy is matmul-only), so equality is by construction.
//
// THE FIXTURE (built on the box from the convertf-PA v66_HO_pa, M-FR-AM; a build
// fixture, not a steppe statistic — allowed): two convertf subsets of the SAME 9-pop
// poplist live under <root>/fixtures_eig/:
//   v66_fit9_pa.{geno,snp,ind}  — outputformat PACKEDANCESTRYMAP (GENO, SNP-major).
//   v66_fit9_am.{geno,snp,ind}  — outputformat ANCESTRYMAP (unpacked text triple).
// Their .ind/.snp are byte-identical (the SAME convertf pop-subset), so read_ind on
// either yields the SAME partition. SKIP exit-0 if the fixture dir is absent (it
// lives only on box5090).
//
// WHAT IT DOES (data root as argv[1], default /workspace/data/aadr):
//   1. Open the ANCESTRYMAP reader (v66_fit9_am.geno) + the PA reader (v66_fit9_pa.geno).
//      ASSERT formats Ancestrymap / Geno and matching n_ind/n_snp. SKIP exit-0 if absent.
//   2. read_ind on the SHARED .ind (Explicit = all 9 fit0 pops) -> the same partition.
//   3. PATH A (oracle): pa_reader -> read_canonical_tile (read_snp_major_tile + transpose).
//      PATH B (new reader): am_reader -> read_canonical_tile (read_ancestrymap_snp_major_tile
//      + the SAME transpose).
//   4. ASSERT the canonical tiles are byte-identical (memcmp==0) — NO tail mask is
//      needed: BOTH formats are SNP-major and zero the partial-last-byte tail in the
//      transpose, so the tiles agree unconditionally (a K not a multiple of 4 is used).
//   5. ASSERT detect_sample_ploidy(T_am) == detect_sample_ploidy(T_pa) (memcmp==0).
//   6. ASSERT decode_af(T_am) == decode_af(T_pa) for Q/V/N (memcmp==0) on BOTH the GPU
//      backend and the CPU oracle.
//
// Exits NONZERO on any failure (CTest gates on the exit code). Self-checking main()
// — NOT a GoogleTest TU (mirrors test_eigenstrat_decode_equivalence.cu).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/stats/read_canonical_tile.hpp"  // the production format dispatch
#include "device/backend.hpp"                  // ComputeBackend, DecodeTileView, DecodeResult
#include "device/backend_factory.hpp"          // make_cpu_backend / make_cuda_backend

#include "io/eigenstrat_format.hpp"            // GenoFormat
#include "io/geno_reader.hpp"                  // GenoReader, read_ancestrymap_snp_major_tile
#include "io/genotype_tile.hpp"                // GenotypeTile
#include "io/ind_reader.hpp"                   // read_ind, PopSelection, IndPartition
#include "io/ploidy_detect.hpp"               // detect_sample_ploidy (AT2 adjust_pseudohaploid)

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::core::read_canonical_tile;

namespace {

constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kPaBase = "v66_fit9_pa";  // convertf PA subset (SNP-major GENO)
constexpr const char* kAmBase = "v66_fit9_am";  // convertf ANCESTRYMAP subset (text triple)

// A SNP prefix not a multiple of 4 (exercises the partial-last-byte tail, which BOTH
// SNP-major paths zero in the transpose). All 9 pops -> 430 inds spanning many bytes.
// ANCESTRYMAP's .geno has n_ind lines per SNP, so K SNPs == K*430 text lines read
// sequentially — bounded and fast for K in the low thousands.
constexpr std::size_t kSnps = 4099;  // not a multiple of 4

// The 9 fit0 pops (the convertf poplist); Explicit selection over the shared .ind.
const std::vector<std::string> kFit9Pops = {
    "Czechia_EBA_CordedWare", "England_BellBeaker", "Han", "Iran_GanjDareh_N",
    "Israel_Natufian", "Karitiana", "Mbuti", "Papuan", "Turkey_N"};

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
    const std::string dir = root + "/fixtures_eig/";
    const std::string pa_geno = dir + kPaBase + ".geno";
    const std::string am_geno = dir + kAmBase + ".geno";
    const std::string shared_ind = dir + kAmBase + ".ind";  // == PA .ind (convertf same subset)

    // ---- (0) Open both readers; SKIP exit-0 if the fixture dir is absent --------------
    std::unique_ptr<steppe::io::GenoReader> pa_reader, am_reader;
    try {
        pa_reader = std::make_unique<steppe::io::GenoReader>(pa_geno);
        am_reader = std::make_unique<steppe::io::GenoReader>(am_geno);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[am-decode-eq] SKIP (ANCESTRYMAP fixture absent: %s)\n", e.what());
        return EXIT_SUCCESS;
    }
    const auto& pa_hdr = pa_reader->header();
    const auto& am_hdr = am_reader->header();
    if (am_hdr.format != steppe::io::GenoFormat::Ancestrymap ||
        pa_hdr.format != steppe::io::GenoFormat::Geno) {
        std::fprintf(stderr, "[am-decode-eq] FAIL: expected AM=Ancestrymap + PA=GENO (got %d / %d)\n",
                     static_cast<int>(am_hdr.format), static_cast<int>(pa_hdr.format));
        return EXIT_FAILURE;
    }
    if (am_hdr.n_ind != pa_hdr.n_ind || am_hdr.n_snp != pa_hdr.n_snp) {
        std::fprintf(stderr, "[am-decode-eq] FAIL: AM(%zu ind,%zu snp) != PA(%zu ind,%zu snp)\n",
                     am_hdr.n_ind, am_hdr.n_snp, pa_hdr.n_ind, pa_hdr.n_snp);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "[am-decode-eq] AM=Ancestrymap(bpr=%zu,hdr=%zu) PA=GENO(bpr=%zu,hdr=%zu) "
                 "n_ind=%zu n_snp=%zu\n",
                 am_hdr.bytes_per_record, am_hdr.header_bytes,
                 pa_hdr.bytes_per_record, pa_hdr.header_bytes, am_hdr.n_ind, am_hdr.n_snp);

    // ---- (1) Reproduce the SAME selection both ways (shared .ind) --------------------
    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::Explicit;
    sel.labels = kFit9Pops;
    steppe::io::IndPartition pa_part, am_part;
    try {
        // Both readers report records_present == n_snp (SNP-major), >> n_ind, so the
        // read_ind individual cap is the SNP count and no individual is dropped.
        pa_part = steppe::io::read_ind(shared_ind, sel, pa_reader->records_present());
        am_part = steppe::io::read_ind(shared_ind, sel, am_reader->records_present());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[am-decode-eq] FAIL: read_ind threw: %s\n", e.what());
        return EXIT_FAILURE;
    }
    const std::size_t K = std::min<std::size_t>(kSnps, am_hdr.n_snp);
    std::fprintf(stderr, "[am-decode-eq] explicit 9 pops -> P=%zu pops, K=%zu SNPs\n",
                 am_part.groups.size(), K);

    auto cpu = steppe::device::make_cpu_backend();
    auto gpu = steppe::device::make_cuda_backend();

    int failures = 0;

    // ---- (2) Build BOTH canonical tiles through the PRODUCTION dispatch --------------
    // PATH A (oracle): GENO -> read_canonical_tile -> read_snp_major_tile + transpose.
    // PATH B (new reader): ANCESTRYMAP -> read_canonical_tile -> read_ancestrymap_snp_major_tile
    //                       + the SAME transpose (run on the GPU backend).
    steppe::io::GenotypeTile T_pa, T_am;
    try {
        T_pa = read_canonical_tile(*pa_reader, pa_part, *gpu, 0, K);
        T_am = read_canonical_tile(*am_reader, am_part, *gpu, 0, K);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[am-decode-eq] FAIL: read_canonical_tile threw: %s\n", e.what());
        return EXIT_FAILURE;
    }

    const bool shape_ok =
        (T_am.n_individuals == T_pa.n_individuals) &&
        (T_am.n_snp == T_pa.n_snp) &&
        (T_am.bytes_per_record == T_pa.bytes_per_record) &&
        (T_am.packed.size() == T_pa.packed.size()) &&
        (T_am.pop_offsets == T_pa.pop_offsets) &&
        (T_am.pop_labels == T_pa.pop_labels);
    if (!shape_ok) {
        ++failures;
        std::fprintf(stderr, "[am-decode-eq] FAIL: tile SHAPE/partition mismatch "
                     "(AM ind=%zu snp=%zu bpr=%zu pop=%zu | PA ind=%zu snp=%zu bpr=%zu pop=%zu)\n",
                     T_am.n_individuals, T_am.n_snp, T_am.bytes_per_record, T_am.n_pop(),
                     T_pa.n_individuals, T_pa.n_snp, T_pa.bytes_per_record, T_pa.n_pop());
    }

    // ---- (3) Canonical tile BYTE-EXACT (UNCONDITIONAL — no tail mask) -----------------
    // BOTH formats are SNP-major: the transpose ZEROES output SNP slots >= K identically
    // for each, so the canonical tiles agree byte-for-byte even for the partial last byte
    // (K=4099, %4 != 0, deliberately exercises it).
    if (shape_ok) {
        const bool tile_eq = (std::memcmp(T_am.packed.data(), T_pa.packed.data(),
                                          T_pa.packed.size()) == 0);
        if (!tile_eq) ++failures;
        std::fprintf(stderr, "[am-decode-eq] canonical tile (%zu bytes, no tail mask): %s\n",
                     T_pa.packed.size(), tile_eq ? "BIT-EXACT (PASS)" : "MISMATCH (FAIL)");
    }

    // ---- (4) Ploidy bit-exact (AT2 adjust_pseudohaploid on each canonical tile) -------
    const std::vector<int> ploidy_pa = steppe::io::detect_sample_ploidy(T_pa);
    const std::vector<int> ploidy_am = steppe::io::detect_sample_ploidy(T_am);
    const bool ploidy_eq = (ploidy_am.size() == ploidy_pa.size()) &&
                           (std::memcmp(ploidy_am.data(), ploidy_pa.data(),
                                        ploidy_pa.size() * sizeof(int)) == 0);
    if (!ploidy_eq) ++failures;
    std::size_t n_ph = 0, n_dip = 0;
    for (int pl : ploidy_pa) (pl == 1 ? n_ph : n_dip)++;
    std::fprintf(stderr, "[am-decode-eq] sample_ploidy (%zu samples: %zu ph + %zu dip): %s\n",
                 ploidy_pa.size(), n_ph, n_dip, ploidy_eq ? "BIT-EXACT (PASS)" : "MISMATCH (FAIL)");

    // ---- (5) DECODED Q/V/N bit-exact on BOTH backends (the keystone) -----------------
    if (shape_ok && ploidy_eq) {
        for (ComputeBackend* be : {static_cast<ComputeBackend*>(gpu.get()),
                                   static_cast<ComputeBackend*>(cpu.get())}) {
            const DecodeResult d_pa = decode_tile(*be, T_pa, ploidy_pa);
            const DecodeResult d_am = decode_tile(*be, T_am, ploidy_am);
            const bool q_eq = (d_am.q.size() == d_pa.q.size()) &&
                              (std::memcmp(d_am.q.data(), d_pa.q.data(),
                                           d_pa.q.size() * sizeof(double)) == 0);
            const bool v_eq = (d_am.v.size() == d_pa.v.size()) &&
                              (std::memcmp(d_am.v.data(), d_pa.v.data(),
                                           d_pa.v.size() * sizeof(double)) == 0);
            const bool n_eq = (d_am.n.size() == d_pa.n.size()) &&
                              (std::memcmp(d_am.n.data(), d_pa.n.data(),
                                           d_pa.n.size() * sizeof(double)) == 0);
            const bool all_eq = q_eq && v_eq && n_eq;
            if (!all_eq) ++failures;
            std::fprintf(stderr, "[am-decode-eq] decode_af AM==PA (%s): Q %s V %s N %s\n",
                         be == gpu.get() ? "GPU" : "CPU",
                         q_eq ? "ok" : "FAIL", v_eq ? "ok" : "FAIL", n_eq ? "ok" : "FAIL");
        }
    }

    std::fprintf(stderr, "\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
