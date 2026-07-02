// src/app/precision_label.hpp
//
// Shared host-app helper mapping a resolved Precision to its human tag
// (emu/tf32/fp64) recorded in an f2_blocks meta.json. Single-homed here so
// cmd_extract_f2.cpp and cmd_qpfstats.cpp share one copy.
//
// Distinct from result_emit.cpp's precision_str(Precision::Kind): that keys off a
// bare Kind for the result `precision` column, this takes the whole resolved
// Precision. Same vocabulary, different signatures, so they stay separate.
//
// CUDA-free by contract: reaches only the CUDA-free steppe/config.hpp for Precision.
#ifndef STEPPE_APP_PRECISION_LABEL_HPP
#define STEPPE_APP_PRECISION_LABEL_HPP

#include "steppe/config.hpp"  // steppe::Precision

namespace steppe::app {

/// Human label for the resolved precision. A switch (not a ternary) so a new
/// Precision::Kind is a compile-visible addition here; the trailing `return "fp64"`
/// keeps native FP64 the fallback label.
[[nodiscard]] inline const char* precision_label(const Precision& p) {
    switch (p.kind) {
        case Precision::Kind::EmulatedFp64: return "emu";
        case Precision::Kind::Tf32:         return "tf32";
        case Precision::Kind::Fp64:         return "fp64";
    }
    return "fp64";
}

}  // namespace steppe::app

#endif  // STEPPE_APP_PRECISION_LABEL_HPP
