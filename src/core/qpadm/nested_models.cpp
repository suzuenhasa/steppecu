// src/core/qpadm/nested_models.cpp — S7 SE driver (design §1.2 S7).

#include "core/qpadm/nested_models.hpp"

#include <cstddef>

namespace steppe::core::qpadm {

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

    // S7 SE reduction, end-to-end on the backend (M7 — the LAST host-compute move).
    // se_from_wmat subsumes the per-block LOO re-fits (gls_weights_loo_batched) + the
    // AT2 (nb-1)/sqrt(nb) wmat scale + the sample-covariance-diagonal variance reduction
    // into ONE backend call, returning the nl-length SCALED se. The CUDA backend keeps
    // the resident dWmat and runs the EXISTING on-device SE kernel (native FP64; no
    // dWmat D2H, no host long-double reduction). The CpuBackend overrides with the
    // long-double sample_cov_diag ORACLE (UNCHANGED math). se_from_loo no longer applies
    // the host scale or the host variance reduction on ANY path — it is fully backend-
    // agnostic. z = weight/se is host-derived here (weight unchanged).
    const std::vector<double> se = be.se_from_wmat(x, cov, r, opts, precision);
    for (int i = 0; i < nl; ++i) {
        const double sei = (static_cast<std::size_t>(i) < se.size())
                               ? se[static_cast<std::size_t>(i)]
                               : 0.0;
        out.se[static_cast<std::size_t>(i)] = sei;
        const double wi = (static_cast<std::size_t>(i) < weight.size())
                              ? weight[static_cast<std::size_t>(i)]
                              : 0.0;
        out.z[static_cast<std::size_t>(i)] = (sei > 0.0) ? wi / sei : 0.0;
    }
    return out;
}

}  // namespace steppe::core::qpadm
