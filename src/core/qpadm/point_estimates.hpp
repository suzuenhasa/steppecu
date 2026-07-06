// src/core/qpadm/point_estimates.hpp
//
// fill_point_estimates() — the shared f3/f4 point-estimate read-out. run_f3 and
// run_f4 both turn a per-stat total estimate plus jackknife-diagonal variance
// into (est, se, z, p) with the exact same loop; this collapses that
// byte-identical body onto one helper. Host-pure.
#ifndef STEPPE_CORE_QPADM_POINT_ESTIMATES_HPP
#define STEPPE_CORE_QPADM_POINT_ESTIMATES_HPP

#include <cmath>
#include <cstddef>
#include <span>

#include "core/internal/index_cast.hpp"
#include "steppe/f4.hpp"

namespace steppe::core::qpadm {

// Fill res.{est,se,z,p} from the total estimate x_total and jackknife variance
// var over N stats. Preserves the f3/f4 write order verbatim (est, se, z, then
// f4_two_sided_p(z)). Templated on the result type (F3Result vs F4Result), which
// differ only by their p1..pK index columns.
template <class Result>
inline void fill_point_estimates(Result& res, std::span<const double> x_total,
                                 std::span<const double> var, int N) {
    res.est.assign(idx(N), 0.0);
    res.se.assign(idx(N), 0.0);
    res.z.assign(idx(N), 0.0);
    res.p.assign(idx(N), 0.0);
    for (int k = 0; k < N; ++k) {
        const std::size_t ks = idx(k);
        const double est = x_total[ks];
        const double var_k = var[ks];
        const double se = (var_k > 0.0) ? std::sqrt(var_k) : std::nan("");
        const double z = est / se;
        res.est[ks] = est;
        res.se[ks] = se;
        res.z[ks] = z;
        res.p[ks] = f4_two_sided_p(z);
    }
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_POINT_ESTIMATES_HPP
