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
    std::vector<double> mean(static_cast<std::size_t>(ncols), 0.0);
    for (int i = 0; i < nrows; ++i)
        for (int c = 0; c < ncols; ++c)
            mean[static_cast<std::size_t>(c)] +=
                w[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncols) +
                  static_cast<std::size_t>(c)];
    for (int c = 0; c < ncols; ++c) mean[static_cast<std::size_t>(c)] /= static_cast<double>(nrows);

    std::vector<double> diag(static_cast<std::size_t>(ncols), 0.0);
    for (int c = 0; c < ncols; ++c) {
        long double acc = 0.0L;
        for (int i = 0; i < nrows; ++i) {
            const double d = w[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncols) +
                               static_cast<std::size_t>(c)] - mean[static_cast<std::size_t>(c)];
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
    const int nr = x.nr;
    const int nb = x.n_block;
    const int m = nl * nr;
    const std::size_t M = static_cast<std::size_t>(m);

    SeResult out;
    out.se.assign(static_cast<std::size_t>(nl), 0.0);
    out.z.assign(static_cast<std::size_t>(nl), 0.0);
    if (nb < 2 || nl <= 0) return out;

    // wmat: nb × nl row-major (the AT2 replicate matrix).
    std::vector<double> wmat(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nl), 0.0);
    for (int b = 0; b < nb; ++b) {
        // One-block F4Blocks whose x_total is the LOO replicate slice loo[:,:,b];
        // gls_weights reads x_total → xmat, so this re-fits the b-th replicate.
        F4Blocks rep;
        rep.nl = nl;
        rep.nr = nr;
        rep.n_block = 1;
        rep.x_total.assign(M, 0.0);
        for (int k = 0; k < m; ++k)
            rep.x_total[static_cast<std::size_t>(k)] =
                x.x_loo[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)];
        const GlsWeights gw = be.gls_weights(rep, cov, r, opts, precision);
        for (int i = 0; i < nl; ++i)
            wmat[static_cast<std::size_t>(b) * static_cast<std::size_t>(nl) +
                 static_cast<std::size_t>(i)] =
                (i < static_cast<int>(gw.w.size())) ? gw.w[static_cast<std::size_t>(i)] : 0.0;
    }

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
