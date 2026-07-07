// tests/unit/test_vcf_gl_parse.cpp
//
// Host-only unit test of VcfReader::genotype_likelihoods (the GL/PL/GP reader): the
// tile it returns is HOST-resident (l/present vectors), so this pins the single
// silent-corruption risk — the panel-A1/A2 POLARITY SWAP of the triplet — with NO
// GPU. Authors a tiny target table + plain .vcf whose every branch has a known
// answer (swap case REF==A1, no-swap case ALT==A1, multiallelic skip, non-panel
// drop, rsID mismap, missing/ref-block, palindrome, GP passthrough), runs the
// reader, and asserts the emitted triplet ordering (g = copies-of-A1) + present
// mask. Links steppe::io (mirrors test_vcf_build_detect). Reports NO science.
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "io/target_sites.hpp"
#include "io/vcf_reader.hpp"

namespace {

int g_failures = 0;

void approx(const char* what, double got, double want, double tol = 1e-7) {
    if (std::abs(got - want) > tol) {
        std::printf("  [FAIL] %-40s got=%.10g want=%.10g\n", what, got, want);
        ++g_failures;
    }
}

void check(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

std::size_t argmax3(const std::array<double, 3>& v) {
    std::size_t m = 0;
    if (v[1] > v[m]) m = 1;
    if (v[2] > v[m]) m = 2;
    return m;
}

}  // namespace

int main() {
    std::printf("=== VcfReader::genotype_likelihoods host-only polarity/parse gate ===\n");
    namespace io = steppe::io;

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_gl_parse_" + std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // ---- target table (rsID chrom pos37 pos38 A1 A2 ref38) -------------------
    const std::filesystem::path tpath = tmp / "targets.tsv";
    {
        std::ofstream t(tpath, std::ios::trunc);
        t << "rsID\tchrom\tpos37\tpos38\tA1\tA2\tref38\n";
        t << "rs_alt\t1\t100\t100\tG\tA\tA\n";     // ALT==A1(G): NO swap
        t << "rs_ref\t1\t200\t200\tA\tG\tA\n";     // REF==A1(A): SWAP RR<->AA
        t << "rs_het\t1\t300\t300\tA\tG\tA\n";     // confident het
        t << "rs_multi\t1\t400\t400\tA\tG\tA\n";   // multiallelic -> skipped
        t << "rs_np\t1\t500\t500\tA\tG\tA\n";      // REF/ALT not panel alleles -> non_panel
        t << "rs_mis\t1\t600\t600\tA\tG\tA\n";     // record rsID != panel -> mismap drop
        t << "rs_none\t1\t700\t700\tA\tG\tA\n";    // no record -> missing
        t << "rs_pal\t1\t800\t800\tA\tT\tA\n";     // palindrome -> missing
        t << "rs_gp\t1\t900\t900\tG\tA\tA\n";      // ALT==A1(G), for the GP-field run
    }

    // ---- plain .vcf with PL triplets -----------------------------------------
    // A confident RR call (0 ALT copies) => PL (0, big, big) => VCF-native lin
    // ~ (1,0,0). Then:
    //   ALT==A1 (rs_alt): g==j, no swap -> g=0 (0 copies of A1) is the peak.
    //   REF==A1 (rs_ref): g==2-j, swap  -> g=2 (2 copies of A1) is the peak.
    const std::filesystem::path vpath = tmp / "gl.vcf";
    {
        std::ofstream v(vpath, std::ios::trunc);
        v << "##fileformat=VCFv4.2\n";
        v << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n";
        v << "1\t100\trs_alt\tA\tG\t.\tPASS\t.\tGT:PL\t0/0:0,60,120\n";     // conf RR
        v << "1\t200\trs_ref\tA\tG\t.\tPASS\t.\tGT:PL\t0/0:0,60,120\n";     // conf RR
        v << "1\t300\trs_het\tA\tG\t.\tPASS\t.\tGT:PL\t0/1:60,0,60\n";      // conf het
        v << "1\t400\trs_multi\tA\tG,C\t.\tPASS\t.\tGT:PL\t0/1:20,0,20,30,30,40\n";
        // rs_np: REF=A(->A1) and ALT=T(comp->A1) both resolve to A1 -> non_panel.
        v << "1\t500\trs_np\tA\tT\t.\tPASS\t.\tGT:PL\t0/1:20,0,20\n";
        v << "1\t600\trs_OTHER\tA\tG\t.\tPASS\t.\tGT:PL\t0/1:20,0,20\n";    // id mismap
        v << "1\t800\trs_pal\tA\tT\t.\tPASS\t.\tGT:PL\t0/1:20,0,20\n";      // palindrome
    }

    io::TargetSites targets = io::read_target_sites(tpath.string());
    io::VcfReader::Options opts;
    io::VcfReader reader(vpath.string(), targets, "", opts);
    const io::VcfReader::LikelihoodResult r = reader.genotype_likelihoods(io::GlField::PL);
    const io::LikelihoodTile& tile = r.tile;

    check("n_sample == 1", tile.n_sample == 1);
    check("n_site == 9", tile.n_site == 9);

    // Map rsID -> site index for readable assertions.
    auto trip = [&](const std::string& rs) -> std::array<double, 3> {
        for (std::size_t i = 0; i < tile.sites.size(); ++i) {
            if (tile.sites[i].rsid == rs) {
                const std::size_t b = tile.base(i, 0);
                return {tile.l[b], tile.l[b + 1], tile.l[b + 2]};
            }
        }
        return {-1, -1, -1};
    };
    auto pres = [&](const std::string& rs) -> int {
        for (std::size_t i = 0; i < tile.sites.size(); ++i)
            if (tile.sites[i].rsid == rs) return tile.present[tile.mask_index(i, 0)];
        return -1;
    };

    // NO-swap (ALT==A1): confident RR -> peak at g=0.
    {
        const auto L = trip("rs_alt");
        check("rs_alt present", pres("rs_alt") == 1);
        check("rs_alt argmax g==0 (no swap)", argmax3(L) == 0);
        approx("rs_alt L[0]~1", L[0], 1.0, 1e-5);
    }
    // SWAP (REF==A1): SAME VCF PL, but peak moves to g=2 (the swap is real).
    {
        const auto L = trip("rs_ref");
        check("rs_ref present", pres("rs_ref") == 1);
        check("rs_ref argmax g==2 (swap RR<->AA)", argmax3(L) == 2);
        approx("rs_ref L[2]~1", L[2], 1.0, 1e-5);
    }
    // Het is polarity-invariant: peak at g=1 either way.
    {
        const auto L = trip("rs_het");
        check("rs_het present", pres("rs_het") == 1);
        check("rs_het argmax g==1", argmax3(L) == 1);
    }
    // Multiallelic / non-panel / mismap / no-record / palindrome -> missing (present=0),
    // uninformative triplet.
    for (const char* rs : {"rs_multi", "rs_np", "rs_mis", "rs_none", "rs_pal"}) {
        check((std::string(rs) + " present==0").c_str(), pres(rs) == 0);
        const auto L = trip(rs);
        approx((std::string(rs) + " uninformative[0]").c_str(), L[0], 1.0 / 3.0);
    }
    check("gl_multiallelic_skipped==1", r.counts.gl_multiallelic_skipped == 1);
    check("gl_non_panel>=1", r.counts.gl_non_panel >= 1);
    check("gl_rsid_mismap==1", r.counts.gl_rsid_mismap == 1);

    // ---- GP field run: posteriors passed through (ALT==A1 -> no swap) ---------
    {
        const std::filesystem::path vgp = tmp / "gp.vcf";
        std::ofstream v(vgp, std::ios::trunc);
        v << "##fileformat=VCFv4.2\n";
        v << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n";
        v << "1\t900\trs_gp\tA\tG\t.\tPASS\t.\tGT:GP\t0/0:0.9,0.08,0.02\n";
        v.close();
        io::VcfReader rgp(vgp.string(), targets, "", opts);
        const io::VcfReader::LikelihoodResult g = rgp.genotype_likelihoods(io::GlField::GP);
        // ALT==A1 => no swap => g-order == VCF-native GP order.
        std::size_t idx = 0;
        for (std::size_t i = 0; i < g.tile.sites.size(); ++i)
            if (g.tile.sites[i].rsid == "rs_gp") idx = i;
        const std::size_t b = g.tile.base(idx, 0);
        check("GP present", g.tile.present[g.tile.mask_index(idx, 0)] == 1);
        check("GP field tag", g.tile.field == io::GlField::GP);
        approx("GP[0]", g.tile.l[b + 0], 0.9);
        approx("GP[1]", g.tile.l[b + 1], 0.08);
        approx("GP[2]", g.tile.l[b + 2], 0.02);
    }

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (GL polarity swap + parse branches reproduce the truth)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
