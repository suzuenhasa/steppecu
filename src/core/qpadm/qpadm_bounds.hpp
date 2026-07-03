// src/core/qpadm/qpadm_bounds.hpp
//
// The single home of the qpAdm small-path fit envelope: the size limits the
// fast per-thread fit kernels are compiled for, plus two shared formulas and one
// status code. Host gate, device backend, and kernel array bounds all read these
// same constants, so the envelope cannot drift apart.
//
// Reference: docs/reference/src_core_qpadm_qpadm_bounds.hpp.md
#ifndef STEPPE_CORE_QPADM_QPADM_BOUNDS_HPP
#define STEPPE_CORE_QPADM_QPADM_BOUNDS_HPP

namespace steppe::core::qpadm {

// Small-path envelope constants — reference §2
inline constexpr int kQpMaxNl = 5;
inline constexpr int kQpMaxNr = 10;
inline constexpr int kQpMaxR = 4;
inline constexpr int kQpMaxM = kQpMaxNl * kQpMaxNr;
inline constexpr int kQpMaxT = (kQpMaxNl > kQpMaxNr ? kQpMaxNl : kQpMaxNr) * kQpMaxR;

// Small-path routing predicate — reference §3
constexpr bool model_fits_small_path(int nl, int nr, int r) {
    return nl <= kQpMaxNl && nr <= kQpMaxNr && r <= kQpMaxR;
}

// Chi-square degrees of freedom — reference §4
constexpr int qpadm_dof(int nl, int nr, int r) {
    return (nl - r) * (nr - r);
}

// Rank-deficient status code — reference §5
inline constexpr int kQpStatusRankDeficient = 6;

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPADM_BOUNDS_HPP
