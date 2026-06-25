// tests/reference/test_plink_decode_equivalence.cu
//
// M-FR PLINK TIER-1 GATE — the PLINK (.bed/.bim/.fam, SNP-major 2-bit LSB-first)
// reader == the PA (PACKEDANCESTRYMAP / GENO SNP-major) reader, BIT-EXACT
// (docs/design/format-readers.md §5.3, the Tier-3-by-reduction-to-PA gate; §8.4 the
// highest-risk plug-in pinned by this bit-exact compare). This is the keystone of the
// PLINK work: it drives the PRODUCTION dispatch (core::read_canonical_tile) on a PLINK
// .bed AND on the convertf-PA .geno of the SAME 9-pop subset, and asserts it reproduces
// the PA decode BYTE-FOR-BYTE — same canonical tile, same auto-detected ploidy, same
// decoded Q/V/N. Since PA == TGENO == the AT2 goldens (proven bit-exact by
// test_pa_decode_equivalence, M-FR-2), this transitively pins PLINK to the goldens with
// NO new statistical golden + NO AT2 rerun.
//
// WHY IT IS A BIT-EXACT INTEGER COMPARE (no tolerance, no oracle file): the PLINK and PA
// fixtures are convertf outputs from the SAME PA source with the SAME poplist (the 9 fit0
// pops), so they carry the SAME individuals, the SAME SNPs, and the SAME genotypes on two
// encodings — packed-SNP-major (PA) vs PLINK 2-bit LSB-first .bed. convertf writes the
// PA/EIGENSTRAT ref allele as PLINK A1 for EVERY SNP (verified: .bim A1/A2 == .snp ref/alt
// for all 584131 SNPs), so the PLINK canonical-ref (:= A1) == the PA ref. A correct PLINK
// reader — the LUT (00->2/01->3 missing/10->1/11->0 in A1-copies) + the LSB->MSB bit-order
// flip + ref:=A1 polarity — MUST therefore produce the byte-identical canonical tile and
// the bit-identical Q/V/N. The transpose/gather/encoding are pure integer/bit ops (the
// emulated-FP64 policy is matmul-only; format-readers.md §6), so equality is by construction.
// A WRONG LUT, bit order, or polarity is caught immediately by the byte-exact compare.
//
// THE FIXTURE (built on the box from the convertf-PA v66_HO_pa via convertf PACKEDPED;
// a build fixture, not a steppe statistic — allowed): under <root>/fixtures_eig/:
//   v66_fit9_pa.{geno,snp,ind}  — outputformat PACKEDANCESTRYMAP (GENO, SNP-major).
//   v66_fit9_ped.{bed,bim,fam}  — outputformat PACKEDPED (PLINK 2-bit LSB-first).
// Both are the SAME 9-pop convertf subset, so the .bim SNP order == the .snp SNP order and
// the .fam individual order == the .ind individual order. SKIP exit-0 if the fixture dir is
// absent (it lives only on box5090).
//
// WHAT IT DOES (data root as argv[1], default /workspace/data/aadr):
//   1. Open the PLINK reader (v66_fit9_ped.bed) + the PA reader (v66_fit9_pa.geno).
//      ASSERT formats Plink / Geno and matching n_ind/n_snp. SKIP exit-0 if absent.
//   2. read_fam on the PLINK .fam (Explicit 9 pops) + read_ind on the PA .ind (same) ->
//      the SAME partition (same individual order, same pop sort).
//   3. PATH A (oracle): pa_reader -> read_canonical_tile (read_snp_major_tile + transpose).
//      PATH B (new reader): ped_reader -> read_canonical_tile (read_plink_snp_major_tile
//      + the SAME transpose).
//   4. ASSERT the canonical tiles are byte-identical (memcmp==0) — NO tail mask is needed:
//      BOTH formats are SNP-major and zero the partial-last-byte tail in the transpose, so
//      the tiles agree unconditionally (a K not a multiple of 4 is used).
//   5. ASSERT detect_sample_ploidy(T_ped) == detect_sample_ploidy(T_pa) (memcmp==0).
//   6. ASSERT decode_af(T_ped) == decode_af(T_pa) for Q/V/N (memcmp==0) on BOTH the GPU
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
#include "io/geno_reader.hpp"                  // GenoReader, read_plink_snp_major_tile
#include "io/genotype_tile.hpp"                // GenotypeTile
#include "io/ind_reader.hpp"                   // read_ind, PopSelection, IndPartition
#include "io/plink_reader.hpp"                 // read_fam (.fam -> IndPartition)
#include "io/ploidy_detect.hpp"               // detect_sample_ploidy (AT2 adjust_pseudohaploid)

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::core::read_canonical_tile;

namespace {

constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kPaBase = "v66_fit9_pa";    // convertf PA subset (SNP-major GENO)
constexpr const char* kPedBase = "v66_fit9_ped";  // convertf PACKEDPED subset (PLINK .bed)

// A SNP prefix not a multiple of 4 (exercises the partial-last-byte tail, which BOTH
// SNP-major paths zero in the transpose). All 9 pops -> 430 inds spanning many bytes.
constexpr std::size_t kSnps = 4099;  // not a multiple of 4

// The 9 fit0 pops (the convertf poplist); Explicit selection over the shared subset.
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
    const std::string pa_ind = dir + kPaBase + ".ind";
    const std::string ped_bed = dir + kPedBase + ".bed";
    const std::string ped_fam = dir + kPedBase + ".fam";

    // ---- (0) Open both readers; SKIP exit-0 if the fixture dir is absent --------------
    std::unique_ptr<steppe::io::GenoReader> pa_reader, ped_reader;
    try {
        pa_reader = std::make_unique<steppe::io::GenoReader>(pa_geno);
        ped_reader = std::make_unique<steppe::io::GenoReader>(ped_bed);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[plink-decode-eq] SKIP (PLINK fixture absent: %s)\n", e.what());
        return EXIT_SUCCESS;
    }
    const auto& pa_hdr = pa_reader->header();
    const auto& ped_hdr = ped_reader->header();
    if (ped_hdr.format != steppe::io::GenoFormat::Plink ||
        pa_hdr.format != steppe::io::GenoFormat::Geno) {
        std::fprintf(stderr, "[plink-decode-eq] FAIL: expected PED=Plink + PA=GENO (got %d / %d)\n",
                     static_cast<int>(ped_hdr.format), static_cast<int>(pa_hdr.format));
        return EXIT_FAILURE;
    }
    if (ped_hdr.n_ind != pa_hdr.n_ind || ped_hdr.n_snp != pa_hdr.n_snp) {
        std::fprintf(stderr, "[plink-decode-eq] FAIL: PED(%zu ind,%zu snp) != PA(%zu ind,%zu snp)\n",
                     ped_hdr.n_ind, ped_hdr.n_snp, pa_hdr.n_ind, pa_hdr.n_snp);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "[plink-decode-eq] PED=Plink(bpr=%zu,hdr=%zu) PA=GENO(bpr=%zu,hdr=%zu) "
                 "n_ind=%zu n_snp=%zu\n",
                 ped_hdr.bytes_per_record, ped_hdr.header_bytes,
                 pa_hdr.bytes_per_record, pa_hdr.header_bytes, ped_hdr.n_ind, ped_hdr.n_snp);

    // ---- (1) Reproduce the SAME selection both ways (PLINK .fam vs PA .ind) -----------
    // The .fam pop label is the FID (col 1); the .ind pop label is col 3. read_fam and
    // read_ind reproduce the SAME selection semantics, and the convertf subset wrote the
    // individuals in the SAME order to both, so the partitions are row-for-row identical.
    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::Explicit;
    sel.labels = kFit9Pops;
    steppe::io::IndPartition pa_part, ped_part;
    try {
        pa_part = steppe::io::read_ind(pa_ind, sel, pa_reader->records_present());
        ped_part = steppe::io::read_fam(ped_fam, sel, ped_reader->records_present());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[plink-decode-eq] FAIL: read_ind/read_fam threw: %s\n", e.what());
        return EXIT_FAILURE;
    }
    // The two partitions MUST be row-for-row identical (same pop sort, same member rows),
    // else the bit-exact tile compare below would be comparing different individuals.
    bool part_ok = (pa_part.groups.size() == ped_part.groups.size());
    for (std::size_t p = 0; part_ok && p < pa_part.groups.size(); ++p) {
        part_ok = (pa_part.groups[p].label == ped_part.groups[p].label) &&
                  (pa_part.groups[p].rows == ped_part.groups[p].rows);
    }
    if (!part_ok) {
        std::fprintf(stderr, "[plink-decode-eq] FAIL: read_fam partition != read_ind partition "
                     "(the .fam FID grouping must match the .ind pop grouping row-for-row)\n");
        return EXIT_FAILURE;
    }
    const std::size_t K = std::min<std::size_t>(kSnps, ped_hdr.n_snp);
    std::fprintf(stderr, "[plink-decode-eq] explicit 9 pops -> P=%zu pops, K=%zu SNPs\n",
                 ped_part.groups.size(), K);

    auto cpu = steppe::device::make_cpu_backend();
    auto gpu = steppe::device::make_cuda_backend();

    int failures = 0;

    // ---- (2) Build BOTH canonical tiles through the PRODUCTION dispatch --------------
    // PATH A (oracle): GENO -> read_canonical_tile -> read_snp_major_tile + transpose.
    // PATH B (new reader): PLINK -> read_canonical_tile -> read_plink_snp_major_tile
    //                       + the SAME transpose (run on the GPU backend).
    steppe::io::GenotypeTile T_pa, T_ped;
    try {
        T_pa = read_canonical_tile(*pa_reader, pa_part, *gpu, 0, K);
        T_ped = read_canonical_tile(*ped_reader, ped_part, *gpu, 0, K);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[plink-decode-eq] FAIL: read_canonical_tile threw: %s\n", e.what());
        return EXIT_FAILURE;
    }

    const bool shape_ok =
        (T_ped.n_individuals == T_pa.n_individuals) &&
        (T_ped.n_snp == T_pa.n_snp) &&
        (T_ped.bytes_per_record == T_pa.bytes_per_record) &&
        (T_ped.packed.size() == T_pa.packed.size()) &&
        (T_ped.pop_offsets == T_pa.pop_offsets) &&
        (T_ped.pop_labels == T_pa.pop_labels);
    if (!shape_ok) {
        ++failures;
        std::fprintf(stderr, "[plink-decode-eq] FAIL: tile SHAPE/partition mismatch "
                     "(PED ind=%zu snp=%zu bpr=%zu pop=%zu | PA ind=%zu snp=%zu bpr=%zu pop=%zu)\n",
                     T_ped.n_individuals, T_ped.n_snp, T_ped.bytes_per_record, T_ped.n_pop(),
                     T_pa.n_individuals, T_pa.n_snp, T_pa.bytes_per_record, T_pa.n_pop());
    }

    // ---- (3) Canonical tile BYTE-EXACT (UNCONDITIONAL — no tail mask) -----------------
    // BOTH formats are SNP-major: the transpose ZEROES output SNP slots >= K identically
    // for each, so the canonical tiles agree byte-for-byte even for the partial last byte
    // (K=4099, %4 != 0). A WRONG PLINK LUT / bit order / polarity diverges here.
    if (shape_ok) {
        const bool tile_eq = (std::memcmp(T_ped.packed.data(), T_pa.packed.data(),
                                          T_pa.packed.size()) == 0);
        if (!tile_eq) ++failures;
        std::fprintf(stderr, "[plink-decode-eq] canonical tile (%zu bytes, no tail mask): %s\n",
                     T_pa.packed.size(), tile_eq ? "BIT-EXACT (PASS)" : "MISMATCH (FAIL)");
    }

    // ---- (4) Ploidy bit-exact (AT2 adjust_pseudohaploid on each canonical tile) -------
    const std::vector<int> ploidy_pa = steppe::io::detect_sample_ploidy(T_pa);
    const std::vector<int> ploidy_ped = steppe::io::detect_sample_ploidy(T_ped);
    const bool ploidy_eq = (ploidy_ped.size() == ploidy_pa.size()) &&
                           (std::memcmp(ploidy_ped.data(), ploidy_pa.data(),
                                        ploidy_pa.size() * sizeof(int)) == 0);
    if (!ploidy_eq) ++failures;
    std::size_t n_ph = 0, n_dip = 0;
    for (int pl : ploidy_pa) (pl == 1 ? n_ph : n_dip)++;
    std::fprintf(stderr, "[plink-decode-eq] sample_ploidy (%zu samples: %zu ph + %zu dip): %s\n",
                 ploidy_pa.size(), n_ph, n_dip, ploidy_eq ? "BIT-EXACT (PASS)" : "MISMATCH (FAIL)");

    // ---- (5) DECODED Q/V/N bit-exact on BOTH backends (the keystone) -----------------
    if (shape_ok && ploidy_eq) {
        for (ComputeBackend* be : {static_cast<ComputeBackend*>(gpu.get()),
                                   static_cast<ComputeBackend*>(cpu.get())}) {
            const DecodeResult d_pa = decode_tile(*be, T_pa, ploidy_pa);
            const DecodeResult d_ped = decode_tile(*be, T_ped, ploidy_ped);
            const bool q_eq = (d_ped.q.size() == d_pa.q.size()) &&
                              (std::memcmp(d_ped.q.data(), d_pa.q.data(),
                                           d_pa.q.size() * sizeof(double)) == 0);
            const bool v_eq = (d_ped.v.size() == d_pa.v.size()) &&
                              (std::memcmp(d_ped.v.data(), d_pa.v.data(),
                                           d_pa.v.size() * sizeof(double)) == 0);
            const bool n_eq = (d_ped.n.size() == d_pa.n.size()) &&
                              (std::memcmp(d_ped.n.data(), d_pa.n.data(),
                                           d_pa.n.size() * sizeof(double)) == 0);
            const bool all_eq = q_eq && v_eq && n_eq;
            if (!all_eq) ++failures;
            std::fprintf(stderr, "[plink-decode-eq] decode_af PED==PA (%s): Q %s V %s N %s\n",
                         be == gpu.get() ? "GPU" : "CPU",
                         q_eq ? "ok" : "FAIL", v_eq ? "ok" : "FAIL", n_eq ? "ok" : "FAIL");
        }
    }

    std::fprintf(stderr, "\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
