// tests/unit/test_roh_model.cpp
//
// Host unit gate for the hapROH model math (`steppe roh`): the haploid emission, the
// 3x3 ROH transition expm, and the (K+1)-state pooled scaled forward-backward. The FB
// p_roh is checked against an INDEPENDENT naive full-matrix (3-state) scaled forward-
// backward built directly from the transition tensor T — a different code path from the
// pooled O(K)-per-column recursion in roh_fb_target — so the collapse + FB + emission
// layout are all validated at once. Device-free, self-checking; CTest gates on the exit
// code.
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/stats/roh_fb.hpp"
#include "core/stats/roh_model.hpp"

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

// Independent naive full-matrix (K+1=3)-state scaled FB directly from the transition
// tensor T (row-major t[l*9 + r*3 + c]) and per-site emission vectors e[l][0..2].
// Returns p_roh[l] = 1 - gamma_0(l). Column 0 = prior only (no emission).
std::vector<double> naive_fb3(long M, const std::vector<double>& T,
                              const std::vector<std::array<double, 3>>& e, double in_val) {
    auto TT = [&](long l, int r, int c) { return T[static_cast<std::size_t>(l * 9 + r * 3 + c)]; };
    std::vector<std::array<double, 3>> alpha(static_cast<std::size_t>(M));
    alpha[0] = {1.0 - 2.0 * in_val, in_val, in_val};
    for (long l = 1; l < M; ++l) {
        std::array<double, 3> a{};
        double s = 0.0;
        for (int j = 0; j < 3; ++j) {
            double pre = 0.0;
            for (int i = 0; i < 3; ++i) pre += TT(l, i, j) * alpha[static_cast<std::size_t>(l - 1)][i];
            a[j] = e[static_cast<std::size_t>(l)][j] * pre;
            s += a[j];
        }
        const double inv = (s > 0.0) ? 1.0 / s : 0.0;
        for (int j = 0; j < 3; ++j) a[j] *= inv;
        alpha[static_cast<std::size_t>(l)] = a;
    }
    std::vector<double> p_roh(static_cast<std::size_t>(M), 0.0);
    std::array<double, 3> beta{1.0, 1.0, 1.0};
    for (long l = M - 1; l >= 0; --l) {
        const auto& al = alpha[static_cast<std::size_t>(l)];
        double denom = 0.0;
        for (int s = 0; s < 3; ++s) denom += al[s] * beta[s];
        const double g0 = (denom > 0.0) ? al[0] * beta[0] / denom : 0.0;
        p_roh[static_cast<std::size_t>(l)] = 1.0 - g0;
        if (l == 0) break;
        std::array<double, 3> nb{};
        double sn = 0.0;
        for (int i = 0; i < 3; ++i) {
            double v = 0.0;
            for (int j = 0; j < 3; ++j) v += TT(l, i, j) * e[static_cast<std::size_t>(l)][j] * beta[j];
            nb[i] = v;
            sn += v;
        }
        const double inv = (sn > 0.0) ? 1.0 / sn : 0.0;
        for (int i = 0; i < 3; ++i) beta[i] = nb[i] * inv;
    }
    return p_roh;
}
}  // namespace

int main() {
    RohParams pr;  // defaults = the locked hapROH parameters

    // --- 1) emission: hand-checked haploid values --------------------------------
    // e_rate=0.01, p=0.3. e0(ob=1) = 0.3*0.99 + 0.7*0.01 = 0.297 + 0.007 = 0.304.
    check(close(roh_emission0(1u, 0.3, 0.01), 0.304, 1e-12), "e0 ob=1");
    // e0(ob=0) = 0.7*0.99 + 0.3*0.01 = 0.693 + 0.003 = 0.696.
    check(close(roh_emission0(0u, 0.3, 0.01), 0.696, 1e-12), "e0 ob=0");
    check(close(roh_emission0(255u, 0.3, 0.01), 1.0, 0.0), "e0 missing target -> 1");
    // copy: match -> 1-e = 0.99, mismatch -> e = 0.01, missing -> 1.
    check(close(roh_emission_copy(1u, 1u, 0.01), 0.99, 1e-12), "copy match");
    check(close(roh_emission_copy(0u, 1u, 0.01), 0.01, 1e-12), "copy mismatch");
    check(close(roh_emission_copy(255u, 1u, 0.01), 1.0, 0.0), "copy missing ref -> 1");

    // --- 2) transition: expm rows are stochastic before the /(K-1) collapse -------
    {
        const int K = 6;
        const std::array<double, 9> Q = roh_rate_matrix(pr, K);
        std::array<double, 9> A;
        for (int i = 0; i < 9; ++i) A[i] = Q[i] * 0.01;  // dg = 0.01 Morgan
        const std::array<double, 9> E = ancibd_expm3(A);
        for (int r = 0; r < 3; ++r) {
            const double rs = E[r * 3 + 0] + E[r * 3 + 1] + E[r * 3 + 2];
            check(close(rs, 1.0, 1e-10), "expm row stochastic");
        }
    }

    // --- 3) the (K+1)-state FB (K=2) vs the independent naive full-matrix oracle ---
    const int K = 2;
    const long M = 6;
    std::vector<double> rmap = {0.00, 0.01, 0.03, 0.06, 0.10, 0.15};
    std::vector<double> p = {0.30, 0.50, 0.20, 0.40, 0.60, 0.35};
    // Target matches ref-hap 0 everywhere, mismatches ref-hap 1 at some sites.
    std::vector<std::uint8_t> ob = {0, 1, 0, 1, 0, 1};
    std::vector<std::uint8_t> h0 = {0, 1, 0, 1, 0, 1};  // ref hap 0 (identical to ob)
    std::vector<std::uint8_t> h1 = {1, 1, 1, 0, 0, 0};  // ref hap 1
    std::vector<std::uint8_t> refhaps(static_cast<std::size_t>(K * M));
    for (long l = 0; l < M; ++l) {
        refhaps[static_cast<std::size_t>(0 * M + l)] = h0[static_cast<std::size_t>(l)];
        refhaps[static_cast<std::size_t>(1 * M + l)] = h1[static_cast<std::size_t>(l)];
    }
    const std::vector<double> T = roh_build_transition(rmap, pr, K);

    std::vector<double> p_roh(static_cast<std::size_t>(M));
    roh_fb_target(ob.data(), refhaps.data(), p.data(), T.data(), K, M, pr, p_roh.data());

    // Build the naive-oracle emission vectors and run the full-matrix FB.
    std::vector<std::array<double, 3>> e(static_cast<std::size_t>(M));
    for (long l = 0; l < M; ++l) {
        const std::size_t ls = static_cast<std::size_t>(l);
        e[ls][0] = roh_emission0(ob[ls], p[ls], pr.e_rate);
        e[ls][1] = roh_emission_copy(h0[ls], ob[ls], pr.e_rate);
        e[ls][2] = roh_emission_copy(h1[ls], ob[ls], pr.e_rate);
    }
    const std::vector<double> ref = naive_fb3(M, T, e, pr.in_val);
    for (long l = 0; l < M; ++l)
        check(close(p_roh[static_cast<std::size_t>(l)], ref[static_cast<std::size_t>(l)], 1e-10),
              ("FB p_roh matches full-matrix oracle @ " + std::to_string(l)).c_str());

    // --- 4) biology sanity: a target identical to one ref-hap over a long dense run
    //        must show high ROH posterior in the interior --------------------------
    {
        const int Kb = 3;
        const long Mb = 40;
        std::vector<double> rm(static_cast<std::size_t>(Mb));
        std::vector<double> pb(static_cast<std::size_t>(Mb), 0.5);
        std::vector<std::uint8_t> obb(static_cast<std::size_t>(Mb));
        std::vector<std::uint8_t> rh(static_cast<std::size_t>(Kb * Mb));
        for (long l = 0; l < Mb; ++l) {
            rm[static_cast<std::size_t>(l)] = 0.001 * static_cast<double>(l);  // dense 0.1 cM spacing
            const std::uint8_t a = static_cast<std::uint8_t>(l % 2);
            obb[static_cast<std::size_t>(l)] = a;
            rh[static_cast<std::size_t>(0 * Mb + l)] = a;                                   // hap 0 = target
            rh[static_cast<std::size_t>(1 * Mb + l)] = static_cast<std::uint8_t>(1 - a);    // hap 1 opposite
            rh[static_cast<std::size_t>(2 * Mb + l)] = static_cast<std::uint8_t>((l / 3) % 2);
        }
        const std::vector<double> Tb = roh_build_transition(rm, pr, Kb);
        std::vector<double> pr_out(static_cast<std::size_t>(Mb));
        roh_fb_target(obb.data(), rh.data(), pb.data(), Tb.data(), Kb, Mb, pr, pr_out.data());
        check(pr_out[static_cast<std::size_t>(Mb / 2)] > 0.9, "interior ROH posterior high on a copy run");
        for (long l = 0; l < Mb; ++l) {
            const double v = pr_out[static_cast<std::size_t>(l)];
            check(v >= -1e-9 && v <= 1.0 + 1e-9, "posterior in [0,1]");
        }
    }

    if (g_fail == 0) std::printf("test_roh_model: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
