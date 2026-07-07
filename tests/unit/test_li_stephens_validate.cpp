// tests/unit/test_li_stephens_validate.cpp
//
// Host-only unit test (no CUDA, no GPU, no files) of the `steppe paint` up-front
// validator (core::validate_paint_request) and the Li-Stephens host model helpers
// (core::li_stephens.*): pins each of the five §3 contracts (phased/haploid input,
// a present + monotonic cM map, self-copy / leave-one-out coherence, the numeric
// knobs, and the O(N·K·M) cost guard) plus the recombination / emission / copying-
// prior helpers. Self-checking main(); CTest gates on the exit code.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/stats/li_stephens.hpp"
#include "core/stats/li_stephens_validate.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace {

int g_fail = 0;

void ok(bool cond, const char* what) {
    if (!cond) {
        std::printf("  [FAIL] %s\n", what);
        ++g_fail;
    } else {
        std::printf("  [ ok ] %s\n", what);
    }
}

using steppe::Status;
using steppe::core::PaintRequest;
using steppe::io::SnpTable;

// A well-formed 3-SNP, single-chromosome, strictly increasing cM map.
SnpTable good_map() {
    SnpTable s;
    s.id = {"rs1", "rs2", "rs3"};
    s.chrom = {1, 1, 1};
    s.genpos_morgans = {0.01, 0.02, 0.03};
    s.physpos = {1000.0, 2000.0, 3000.0};
    s.ref = {'A', 'A', 'A'};
    s.alt = {'C', 'C', 'C'};
    s.count = 3;
    return s;
}

// A well-formed baseline request against a tiny separate donor panel.
PaintRequest good_req() {
    PaintRequest r;
    r.Ne = 20000.0;
    r.theta_auto = true;
    r.self_copy = false;
    r.recip_batch = 4;
    r.allow_bp_fallback = false;
    r.n_recipients = 2;
    r.n_donors = 4;
    r.n_diploid_samples = 0;
    r.donors_superset_recipients = false;
    r.sure = false;
    return r;
}

Status run(const PaintRequest& r, const SnpTable& s, std::string& err) {
    return steppe::core::validate_paint_request(r, s, err);
}

void test_validator() {
    std::printf("== validator ==\n");
    std::string err;

    // 0. The baseline passes.
    ok(run(good_req(), good_map(), err) == Status::Ok, "baseline request validates");

    // 1. Phased/haploid: any diploid sample is rejected.
    {
        PaintRequest r = good_req();
        r.n_diploid_samples = 1;
        ok(run(r, good_map(), err) == Status::InvalidConfig, "diploid sample rejected");
        ok(err.find("PRE-PHASED") != std::string::npos, "  message says phase first");
    }

    // 2a. Absent cM map (all-zero genpos) is a hard error without bp-fallback.
    {
        SnpTable s = good_map();
        s.genpos_morgans = {0.0, 0.0, 0.0};
        ok(run(good_req(), s, err) == Status::InvalidConfig, "absent map rejected by default");
        // ... but opting into the bp-fallback (with physpos present) is accepted.
        PaintRequest r = good_req();
        r.allow_bp_fallback = true;
        ok(run(r, s, err) == Status::Ok, "absent map + --bp-fallback accepted");
    }

    // 2b. Non-monotonic map WITHIN a chromosome is rejected.
    {
        SnpTable s = good_map();
        s.genpos_morgans = {0.03, 0.02, 0.04};  // decrease at index 1
        ok(run(good_req(), s, err) == Status::InvalidConfig, "non-monotonic map rejected");
        ok(err.find("non-monotonic") != std::string::npos, "  message says non-monotonic");
    }

    // 2c. A decrease ACROSS a chromosome boundary is fine (the map resets).
    {
        SnpTable s = good_map();
        s.chrom = {1, 2, 2};
        s.genpos_morgans = {0.05, 0.01, 0.02};  // 0.05 -> 0.01 is a chrom reset
        ok(run(good_req(), s, err) == Status::Ok, "chrom-boundary map reset accepted");
    }

    // 3. Self-copy coherence: --self-copy on with donors superset recipients is rejected.
    {
        PaintRequest r = good_req();
        r.donors_superset_recipients = true;
        r.self_copy = true;
        ok(run(r, good_map(), err) == Status::InvalidConfig, "self-copy on panel-vs-self rejected");
        // ... leave-one-out (self-copy off) on the same geometry is accepted.
        r.self_copy = false;
        ok(run(r, good_map(), err) == Status::Ok, "leave-one-out on panel-vs-self accepted");
    }

    // 4. Numeric knobs.
    {
        PaintRequest r = good_req();
        r.Ne = 0.0;
        ok(run(r, good_map(), err) == Status::InvalidConfig, "Ne<=0 rejected");
        r = good_req();
        r.theta_auto = false;
        r.theta = 1.5;
        ok(run(r, good_map(), err) == Status::InvalidConfig, "theta>1 rejected");
        r = good_req();
        r.recip_batch = 0;
        ok(run(r, good_map(), err) == Status::InvalidConfig, "recip-batch<1 rejected");
        r = good_req();
        r.n_recipients = 0;
        ok(run(r, good_map(), err) == Status::InvalidConfig, "zero recipients rejected");
        r = good_req();
        r.n_donors = 0;
        ok(run(r, good_map(), err) == Status::InvalidConfig, "zero donors rejected");
    }

    // 5. Cost guard: an over-cap N*K*M refuses without --sure and proceeds with it.
    {
        SnpTable s = good_map();
        PaintRequest r = good_req();
        r.n_recipients = 1000000;
        r.n_donors = 1000000;  // 1e6 * 1e6 * 3 = 3e18 >> kLsMaxWorkStates
        r.recip_batch = 1;     // keep the per-wave footprint small so only the work cap trips
        ok(run(r, s, err) == Status::InvalidConfig, "over-cap work refused");
        ok(err.find("--sure") != std::string::npos, "  message names --sure");
        r.sure = true;
        ok(run(r, s, err) == Status::Ok, "over-cap work + --sure proceeds");
    }
    // 5b. Per-wave forward-table footprint cap (independent of --sure).
    {
        SnpTable s = good_map();
        PaintRequest r = good_req();
        // recip_batch * K * 8 bytes over the cap.
        const long K = 1000000;
        r.n_donors = K;
        r.n_recipients = 10000;
        r.recip_batch = static_cast<long>(steppe::kLsMaxAlphaFootprintBytes / (K * sizeof(double))) + 100;
        r.sure = true;  // lift the work cap so the footprint cap is what trips
        ok(run(r, s, err) == Status::InvalidConfig, "over-cap forward-table footprint rejected");
        ok(err.find("recip-batch") != std::string::npos, "  message says lower recip-batch");
    }
}

void test_helpers() {
    std::printf("== model helpers ==\n");
    using namespace steppe::core;

    // haploid_allele_from_code: 0->0, 2->1, missing/het->missing.
    ok(haploid_allele_from_code(0) == 0, "code 0 -> allele 0");
    ok(haploid_allele_from_code(2) == 1, "code 2 -> allele 1");
    ok(haploid_allele_from_code(3) == kLsMissingAllele, "missing code -> missing allele");
    ok(haploid_allele_from_code(1) == kLsMissingAllele, "het code -> missing allele");

    // watterson_emission_rate: in (0, 0.5), 0 for K<2, decreasing in K.
    ok(watterson_emission_rate(1) == 0.0, "watterson K<2 -> 0");
    const double m10 = watterson_emission_rate(10);
    const double m100 = watterson_emission_rate(100);
    ok(m10 > 0.0 && m10 < 0.5, "watterson K=10 in (0,0.5)");
    ok(m100 > 0.0 && m100 < m10, "watterson decreases with K");

    // ls_recomb_prob: in [0,1], 0 at delta 0, increasing in delta, ->1 for large delta.
    ok(ls_recomb_prob(0.0, 20000.0, 10) == 0.0, "recomb delta 0 -> 0");
    const double r_small = ls_recomb_prob(1e-5, 20000.0, 10);
    const double r_big = ls_recomb_prob(1.0, 20000.0, 10);
    ok(r_small > 0.0 && r_small < 1.0, "recomb small gap in (0,1)");
    ok(r_big > r_small && r_big <= 1.0, "recomb increases with gap, capped at 1");

    // build_recomb_probs: rho[0]=1, chrom boundary=1, same-chrom in [0,1].
    {
        std::vector<int> chrom = {1, 1, 2, 2};
        // small same-chrom gaps so rho stays in (0,1) (a 1 cM gap would saturate to 1).
        std::vector<double> gp = {0.0, 1e-5, 0.0, 2e-5};
        std::vector<double> rho = build_recomb_probs(chrom, gp, 20000.0, 4);
        ok(rho.size() == 4, "recomb-probs length M");
        ok(rho[0] == 1.0, "rho[0] == 1 (no predecessor)");
        ok(rho[2] == 1.0, "rho at chrom boundary == 1 (unlinked)");
        ok(rho[1] > 0.0 && rho[1] < 1.0, "rho within chrom in (0,1)");
    }

    // build_uniform_pi: uniform sums to 1; leave-one-out zeroes self and renormalizes.
    {
        std::vector<double> pi = build_uniform_pi(4, -1);
        double sum = 0.0;
        for (double p : pi) sum += p;
        ok(std::fabs(sum - 1.0) < 1e-12 && std::fabs(pi[0] - 0.25) < 1e-12, "uniform pi sums to 1");
        std::vector<double> loo = build_uniform_pi(4, 2);
        double s2 = 0.0;
        for (double p : loo) s2 += p;
        ok(loo[2] == 0.0, "leave-one-out zeroes the self donor");
        ok(std::fabs(s2 - 1.0) < 1e-12 && std::fabs(loo[0] - (1.0 / 3.0)) < 1e-12,
           "leave-one-out renormalizes over K-1");
    }
}

}  // namespace

int main() {
    std::printf("=== Li-Stephens paint validator + helpers unit test ===\n");
    test_validator();
    test_helpers();
    if (g_fail == 0) {
        std::printf("\nRESULT: PASS\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s))\n", g_fail);
    return 1;
}
