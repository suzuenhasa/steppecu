// src/app/precision_label.hpp
//
// One shared host-app helper: precision_label maps a resolved Precision to the
// short human tag (emu/tf32/fp64) recorded in an f2_blocks meta.json. Host-only
// and CUDA-free by contract — it includes only the CUDA-free steppe/config.hpp.
//
// Reference: docs/reference/src_app_precision_label.hpp.md
#ifndef STEPPE_APP_PRECISION_LABEL_HPP
#define STEPPE_APP_PRECISION_LABEL_HPP

#include "steppe/config.hpp"

namespace steppe::app {

// precision label mapping — reference §2
[[nodiscard]] inline const char* precision_label(Precision::Kind k) {
    switch (k) {
        case Precision::Kind::EmulatedFp64: return "emu";
        case Precision::Kind::Tf32:         return "tf32";
        case Precision::Kind::Fp64:         return "fp64";
    }
    return "fp64";
}

[[nodiscard]] inline const char* precision_label(const Precision& p) {
    return precision_label(p.kind);
}

}  // namespace steppe::app

#endif  // STEPPE_APP_PRECISION_LABEL_HPP
