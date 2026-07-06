// src/core/internal/qvn_assert.hpp
//
// assert_qvn_consistent() — the shared Q/V/N shape precondition. The three debug
// asserts (P agree, M agree, non-negative shape) that every f2 entry point runs
// on its inputs, factored into one place. Host-pure, compiles out under NDEBUG.
#ifndef STEPPE_CORE_INTERNAL_QVN_ASSERT_HPP
#define STEPPE_CORE_INTERNAL_QVN_ASSERT_HPP

#include "core/internal/host_device.hpp"
#include "core/internal/views.hpp"

namespace steppe::core {

// Q/V/N precondition: same P, same M, non-negative shape.
inline void assert_qvn_consistent([[maybe_unused]] const MatView& Q,
                                  [[maybe_unused]] const MatView& V,
                                  [[maybe_unused]] const MatView& N) {
    STEPPE_ASSERT(Q.P == V.P && V.P == N.P,
                  "assert_qvn_consistent: Q/V/N disagree on P (population count)");
    STEPPE_ASSERT(Q.M == V.M && V.M == N.M,
                  "assert_qvn_consistent: Q/V/N disagree on M (SNP count)");
    STEPPE_ASSERT(Q.P >= 0 && Q.M >= 0,
                  "assert_qvn_consistent: negative P or M (uninitialized MatView)");
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_QVN_ASSERT_HPP
