// tests/unit/beagle_reader_test.cpp
//
// Host unit gate for the beagle-GL reader (`steppe pcangsd` ingest arm). Authors a
// tiny plain-text beagle fixture (the non-gzip path of GzipLineReader), reads it,
// and asserts: n_site/n_sample, the REVERSED-triplet placement into the g =
// copies-of-A1 tile axis (l[base+0]=P(A2A2), l[base+2]=P(A1A1)), site metadata
// (rsid/chrom/pos/alleles from a chrom_pos marker), the present flag on a degenerate
// all-zero triplet, and that a column-count mismatch throws. This is the
// beagle-decode-correct gate (the parsed triplets vs a hand read). Device-free.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "io/beagle_reader.hpp"

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}
bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
}  // namespace

int main() {
    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_beagle_" + std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    // 3 sites x 2 individuals, plain text. Ind0's row-2 triplet is all-zero (degenerate).
    const std::filesystem::path bpath = tmp / "toy.beagle";
    {
        std::ofstream b(bpath, std::ios::trunc);
        b << "marker\tallele1\tallele2\tInd0\tInd0\tInd0\tInd1\tInd1\tInd1\n";
        b << "1_100\tA\tG\t0.8\t0.15\t0.05\t0.1\t0.2\t0.7\n";
        b << "1_200\tC\tT\t0.33\t0.34\t0.33\t0.9\t0.05\t0.05\n";
        b << "1_300\tA\tC\t0\t0\t0\t0.5\t0.4\t0.1\n";
    }

    steppe::io::BeagleReadResult res = steppe::io::read_beagle_gl(bpath.string());
    const steppe::io::LikelihoodTile& t = res.tile;

    check(res.n_site == 3, "n_site == 3");
    check(res.n_sample == 2, "n_sample == 2");
    check(t.n_site == 3 && t.n_sample == 2, "tile geometry");
    check(t.sample_ids.size() == 2 && t.sample_ids[0] == "Ind0" && t.sample_ids[1] == "Ind1",
          "sample ids");

    // Site metadata from the chrom_pos marker.
    check(t.sites.size() == 3, "sites size");
    check(t.sites[0].rsid == "1_100" && t.sites[0].chrom == 1 && t.sites[0].pos38 == 100,
          "site0 marker parse");
    check(t.sites[0].a1 == 'A' && t.sites[0].a2 == 'G', "site0 alleles");

    // Reversed placement: beagle (P(A1A1),P(het),P(A2A2)) -> l[base]=P(A2A2..0 copies A1),
    // l[base+2]=P(A1A1..2 copies A1). site0/Ind0 sums to 1 so normalize is identity.
    {
        const std::size_t b = t.base(0, 0);
        check(close(t.l[b + 0], 0.05, 1e-12), "site0 Ind0 l[0]=P(A2A2)");
        check(close(t.l[b + 1], 0.15, 1e-12), "site0 Ind0 l[1]=het");
        check(close(t.l[b + 2], 0.80, 1e-12), "site0 Ind0 l[2]=P(A1A1)");
        check(t.present[t.mask_index(0, 0)] == 1, "site0 Ind0 present");
    }
    // site0/Ind1 (0.1,0.2,0.7) -> reversed (0.7,0.2,0.1).
    {
        const std::size_t b = t.base(0, 1);
        check(close(t.l[b + 0], 0.7, 1e-12) && close(t.l[b + 1], 0.2, 1e-12) &&
                  close(t.l[b + 2], 0.1, 1e-12),
              "site0 Ind1 reversed triplet");
    }
    // Degenerate site2/Ind0 -> uninformative (1/3,1/3,1/3), present=0.
    {
        const std::size_t b = t.base(2, 0);
        check(close(t.l[b + 0], 1.0 / 3.0, 1e-12) && close(t.l[b + 1], 1.0 / 3.0, 1e-12) &&
                  close(t.l[b + 2], 1.0 / 3.0, 1e-12),
              "site2 Ind0 uninformative");
        check(t.present[t.mask_index(2, 0)] == 0, "site2 Ind0 present=0");
        check(t.present[t.mask_index(2, 1)] == 1, "site2 Ind1 present=1");
    }

    // Column-count mismatch must throw.
    const std::filesystem::path badp = tmp / "bad.beagle";
    {
        std::ofstream b(badp, std::ios::trunc);
        b << "marker\tallele1\tallele2\tInd0\tInd0\tInd0\n";
        b << "1_100\tA\tG\t0.8\t0.15\n";  // too few fields
    }
    bool threw = false;
    try {
        (void)steppe::io::read_beagle_gl(badp.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "column-count mismatch throws");

    // Bad header (cols-3 not a multiple of 3) must throw.
    const std::filesystem::path badh = tmp / "badh.beagle";
    {
        std::ofstream b(badh, std::ios::trunc);
        b << "marker\tallele1\tallele2\tInd0\tInd0\n";  // 5 cols -> (5-3)%3 != 0
        b << "1_100\tA\tG\t0.8\t0.15\n";
    }
    threw = false;
    try {
        (void)steppe::io::read_beagle_gl(badh.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "bad header column count throws");

    std::filesystem::remove_all(tmp, ec);
    if (g_fail == 0) {
        std::printf("RESULT: PASS (beagle reader parse + reversed triplet + present + throws)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d check(s) failed)\n", g_fail);
    return 1;
}
