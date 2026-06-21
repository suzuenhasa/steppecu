// src/core/qpadm/nested_models.cpp — S7 SE driver (design §1.2 S7).

#include "core/qpadm/nested_models.hpp"

#include <cmath>
#include <cstddef>

namespace steppe::core::qpadm {

namespace {

/// Sample covariance of an (nrows × ncols) row-major data matrix `w`
/// (w[i*ncols + c]), matching R's cov(): divides by (nrows - 1), columns are the
/// variables. Returns the ncols×ncols covariance (column-major), of which only the
/// diagonal is needed for SE.
[[nodiscard]] std::vector<double> sample_cov_diag(const std::vector<double>& w,
                                                  int nrows, int ncols) {
    // Hoist the column-stride widening once (§3.3: keep the widening, fold the boilerplate);
    // the (i,c)->w[i*ncols+c] row-major access then lives in one accessor used by both loops
    // (§4 one concept = one spelling — a divergent index edit can no longer miss a copy).
    const std::size_t nc = static_cast<std::size_t>(ncols);
    const auto row_major_at = [&w, nc](int i, int c) -> double {
        return w[static_cast<std::size_t>(i) * nc + static_cast<std::size_t>(c)];
    };

    std::vector<double> mean(nc, 0.0);
    for (int i = 0; i < nrows; ++i)
        for (int c = 0; c < ncols; ++c)
            mean[static_cast<std::size_t>(c)] += row_major_at(i, c);
    for (int c = 0; c < ncols; ++c) mean[static_cast<std::size_t>(c)] /= static_cast<double>(nrows);

    std::vector<double> diag(nc, 0.0);
    for (int c = 0; c < ncols; ++c) {
        const double mc = mean[static_cast<std::size_t>(c)];  // loop-invariant w.r.t. inner i
        // long double accumulator: extra precision for the sum-of-squares to match AT2 cov()
        // (§3.3 never narrow accumulators — keep long double, the rest is double-by-design).
        long double acc = 0.0L;
        for (int i = 0; i < nrows; ++i) {
            const double d = row_major_at(i, c) - mc;
            acc += static_cast<long double>(d) * static_cast<long double>(d);
        }
        diag[static_cast<std::size_t>(c)] = static_cast<double>(acc / static_cast<long double>(nrows - 1));
    }
    return diag;
}

}  // namespace

SeResult se_from_loo(ComputeBackend& be, const F4Blocks& x, const JackknifeCov& cov,
                     int r, const QpAdmOptions& opts, const std::vector<double>& weight,
                     const Precision& precision) {
    const int nl = x.nl;
    const int nb = x.n_block;

    SeResult out;
    out.se.assign(static_cast<std::size_t>(nl), 0.0);
    out.z.assign(static_cast<std::size_t>(nl), 0.0);
    // Block jackknife needs >=2 delete-1 replicates to form a sample covariance.
    constexpr int kMinJackknifeBlocks = 2;
    if (nb < kMinJackknifeBlocks || nl <= 0) return out;

    // wmat: nb × nl row-major (the AT2 replicate matrix). The per-block re-fits run
    // through the BATCHED-capable backend seam (gls_weights_loo_batched): the CUDA
    // backend runs all nb on-device in one batched launch; the CpuBackend overrides
    // it with the oracle host loop (the FROZEN CONTRACT §2e). se_from_loo is now
    // backend-agnostic — it no longer hard-codes the host loop.
    std::vector<double> wmat = be.gls_weights_loo_batched(x, cov, r, opts, precision);

    // AT2 (!boot): wmat <- wmat * (numreps-1)/sqrt(numreps); se = sqrt(diag(cov(wmat))).
    const double scale = static_cast<double>(nb - 1) / std::sqrt(static_cast<double>(nb));
    for (double& v : wmat) v *= scale;
    const std::vector<double> diag = sample_cov_diag(wmat, nb, nl);
    for (int i = 0; i < nl; ++i) {
        const double se = std::sqrt(diag[static_cast<std::size_t>(i)]);
        out.se[static_cast<std::size_t>(i)] = se;
        out.z[static_cast<std::size_t>(i)] =
            (se > 0.0) ? weight[static_cast<std::size_t>(i)] / se : 0.0;
    }
    return out;
}

}  // namespace steppe::core::qpadm
