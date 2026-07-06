// src/core/qpadm/nested_models.cpp
//
// S7 standard-error driver for qpAdm: se_from_loo delegates the block-jackknife
// SE reduction to a single backend call and derives z = weight/se on the host.

#include "core/qpadm/nested_models.hpp"

#include <cstddef>

#include "core/internal/index_cast.hpp"

namespace steppe::core::qpadm {

SeResult se_from_loo(ComputeBackend& be, const F4Blocks& x, const JackknifeCov& cov,
                     int r, const QpAdmOptions& opts, const std::vector<double>& weight,
                     const Precision& precision) {
    const int nl = x.nl;
    const int nb = x.n_block;

    SeResult out;
    out.se.assign(idx(nl), 0.0);
    out.z.assign(idx(nl), 0.0);
    constexpr int kMinJackknifeBlocks = 2;
    if (nb < kMinJackknifeBlocks || nl <= 0) return out;

    const std::vector<double> se = be.se_from_wmat(x, cov, r, opts, precision);
    for (int i = 0; i < nl; ++i) {
        const double sei = (idx(i) < se.size())
                               ? se[idx(i)]
                               : 0.0;
        out.se[idx(i)] = sei;
        const double wi = (idx(i) < weight.size())
                              ? weight[idx(i)]
                              : 0.0;
        out.z[idx(i)] = (sei > 0.0) ? wi / sei : 0.0;
    }
    return out;
}

}  // namespace steppe::core::qpadm
