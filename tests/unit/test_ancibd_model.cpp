// tests/unit/test_ancibd_model.cpp
//
// Host unit gate for the ancIBD model math (`steppe ibd`): the emission ratio, the
// 3x3 transition expm, the 5-state scaled forward-backward, and the segment caller /
// per-pair summary. The forward-backward p_ibd is checked against a fixed 8-SNP
// reference computed by an INDEPENDENT numpy transcription of ancIBD's compiled
// fwd_bkwd_scaled (+ transition.py + emission.py); this is the same known-posterior
// example the design used to validate the port (agreement to ~3e-15). Device-free,
// self-checking; CTest gates on the exit code.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/stats/ancibd_fb.hpp"
#include "core/stats/ancibd_model.hpp"
#include "core/stats/ancibd_segments.hpp"

using namespace steppe::core;

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
    AncibdParams pr;  // defaults = the locked ancIBD parameters

    // --- 1) emission ratio: hand-checked share() ---------------------------------
    // hAa=0.875, hBa=0.85, p=0.30 -> 0.875*0.85/0.70 + 0.125*0.15/0.30 = 1.0625 + 0.0625.
    check(close(ancibd_share(0.875, 0.85, 0.30, pr.p_min), 1.125, 1e-12), "share hand value");
    // e[0] is always the non-IBD baseline 1.
    {
        double e[5];
        ancibd_emission5(0.875, 0.875, 0.85, 0.85, 0.30, pr.p_min, e);
        check(close(e[0], 1.0, 0.0), "e[0]==1");
        check(close(e[1], 1.125, 1e-12), "e[1] share(0,2)");
    }

    // --- 2) hts_p (LoadH5Multi2.get_haplo_prob) ----------------------------------
    // GP=[0.3,0.4,0.3], GT 0|1 -> hapA(anc)=g00+g01=0.7, hapB=g00=0.3 (critic example).
    {
        double hA, hB;
        ancibd_haplo_prob(0.3, 0.4, 0u, 1u, pr.min_error, hA, hB);
        check(close(hA, 0.7, 1e-12), "hts_p hapA GT 0|1");
        check(close(hB, 0.3, 1e-12), "hts_p hapB GT 0|1");
        // homozygous 0|0 splits the het mass 50/50.
        ancibd_haplo_prob(0.3, 0.4, 0u, 0u, pr.min_error, hA, hB);
        check(close(hA, 0.5, 1e-12) && close(hB, 0.5, 1e-12), "hts_p hom 0|0 split");
    }

    // --- 3) transition: row sums of expm(Q*dg) are ~1 (stochastic before the /3
    //        collapse-normalization; after it the [0..1][2] entries shrink) ---------
    {
        // Single-gap transition; the pre-normalization expm rows must sum to 1.
        const std::array<double, 9> Q = ancibd_rate_matrix(pr);
        std::array<double, 9> A;
        for (int i = 0; i < 9; ++i) A[i] = Q[i] * 0.01;  // dg = 0.01 Morgan
        const std::array<double, 9> E = ancibd_expm3(A);
        for (int r = 0; r < 3; ++r) {
            const double rs = E[r * 3 + 0] + E[r * 3 + 1] + E[r * 3 + 2];
            check(close(rs, 1.0, 1e-10), "expm row stochastic");
        }
    }

    // --- 4) the 5-state FB against the fixed reference (numpy fwd_bkwd_scaled) -----
    const int M = 8;
    std::vector<double> rmap = {0.00, 0.01, 0.03, 0.06, 0.10, 0.13, 0.15, 0.20};
    std::vector<double> p = {0.30, 0.50, 0.20, 0.40, 0.60, 0.10, 0.35, 0.45};
    double A_gp0[] = {0.8, 0.1, 0.85, 0.05, 0.9, 0.7, 0.2, 0.6};
    double A_gp1[] = {0.15, 0.85, 0.10, 0.90, 0.08, 0.25, 0.75, 0.35};
    int A_gt0[] = {0, 0, 0, 1, 0, 0, 0, 0}, A_gt1[] = {0, 1, 0, 0, 0, 0, 1, 0};
    double B_gp0[] = {0.75, 0.12, 0.8, 0.06, 0.88, 0.65, 0.18, 0.55};
    double B_gp1[] = {0.20, 0.80, 0.15, 0.88, 0.10, 0.30, 0.78, 0.40};
    int B_gt0[] = {0, 0, 0, 1, 0, 0, 0, 0}, B_gt1[] = {0, 1, 0, 0, 0, 0, 1, 0};

    std::vector<double> hAa(M), hAb(M), hBa(M), hBb(M);
    for (int l = 0; l < M; ++l) {
        ancibd_haplo_prob(A_gp0[l], A_gp1[l], (unsigned char)A_gt0[l], (unsigned char)A_gt1[l],
                          pr.min_error, hAa[l], hAb[l]);
        ancibd_haplo_prob(B_gp0[l], B_gp1[l], (unsigned char)B_gt0[l], (unsigned char)B_gt1[l],
                          pr.min_error, hBa[l], hBb[l]);
    }
    const std::vector<double> T = ancibd_build_transition(rmap, pr);
    std::vector<double> pibd(M);
    ancibd_fb_pair(hAa.data(), hAb.data(), hBa.data(), hBb.data(), p.data(), T.data(), M, pr,
                   pibd.data());
    // Reference p_ibd from the independent numpy oracle (design validation).
    const double ref[8] = {0.000621953315413371, 0.0158580190591122,  0.0452026604697426,
                           0.08626547209863,      0.143046230118375,   0.137330378983968,
                           0.132615996728812,     0.125895913030586};
    for (int l = 0; l < M; ++l)
        check(close(pibd[l], ref[l], 1e-9), ("FB p_ibd matches ancIBD oracle @ " + std::to_string(l)).c_str());

    // --- 5) segment calling: threshold -> drop <8cM -> merge <0.75cM gap ---------
    {
        std::vector<double> rm = {0.00, 0.02, 0.04, 0.06, 0.085, 0.10, 0.30, 0.31, 0.40, 0.50, 0.60, 0.70};
        std::vector<long long> bpv = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200};
        std::vector<double> pi = {0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.2, 0.999, 0.999, 0.999, 0.999, 0.999};
        auto segs = ancibd_call_segments(rm, bpv, pi, 20, "A", "B", pr);
        // run [0..5] = 10 cM (kept); short run [6] excluded (posterior 0.2); run [7..11] = 39 cM.
        check(segs.size() == 2, "two segments called");
        if (segs.size() == 2) {
            check(close(segs[0].lengthM, 0.10, 1e-9), "seg0 length 10cM");
            check(segs[0].endBP == 600, "seg0 endBP");
            check(close(segs[1].lengthM, 0.39, 1e-9), "seg1 length 39cM");
        }
        // A sub-8cM run is dropped entirely.
        std::vector<double> pi2 = {0.999, 0.999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // ~2cM run
        auto segs2 = ancibd_call_segments(rm, bpv, pi2, 20, "A", "B", pr);
        check(segs2.empty(), "sub-8cM run dropped");
    }

    // --- 6) summary: density + length filter, per-cutoff sum/n --------------------
    {
        // One dense 30 cM segment (length 6600 SNPs -> 220 SNP/cM, above the 220 floor
        // requires strictly >, so use 6601) and one 5 cM (below the >6cM filter).
        std::vector<IbdSegment> segs;
        IbdSegment a;
        a.iid1 = "A"; a.iid2 = "B"; a.ch = 1; a.lengthM = 0.30; a.length = 6700; segs.push_back(a);
        IbdSegment b = a; b.lengthM = 0.05; b.length = 20; segs.push_back(b);  // 5cM, filtered
        auto sm = ancibd_summarize(segs);
        check(sm.size() == 1, "one summary pair");
        if (sm.size() == 1) {
            check(close(sm[0].sum_cm[0], 30.0, 1e-9), "sum_IBD>8 == 30cM");
            check(sm[0].n_seg[0] == 1, "n_IBD>8 == 1");
            check(sm[0].n_seg[3] == 1, "n_IBD>20 == 1");  // 30cM > 20
            check(close(sm[0].max_IBD, 30.0, 1e-9), "max_IBD == 30cM");
        }
    }

    if (g_fail == 0) std::printf("test_ancibd_model: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
