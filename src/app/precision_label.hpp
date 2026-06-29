// src/app/precision_label.hpp
//
// One shared host-app helper: precision_label(const Precision&) -> the human
// precision tag (emu/tf32/fp64) recorded in an f2_blocks meta.json (cli-bindings.md
// §4.3 — the ENGAGED tag). Hoisted out of cmd_extract_f2.cpp / cmd_qpfstats.cpp,
// which each carried a byte-identical copy (the qpfstats comment even read "mirrors
// cmd_extract_f2"); single-homing it removes the duplication (ROADMAP §6 group 7).
//
// DISTINCT from result_emit.cpp's precision_str(Precision::Kind): that one keys off a
// bare Kind for the result `precision` column, this one takes the whole Precision the
// f2-dir writer path resolves. The emu/tf32/fp64 vocabulary matches; the signatures do
// not, so the two stay separate helpers.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches
// only the CUDA-FREE steppe/config.hpp for the Precision type. THIS HEADER IS CUDA-FREE
// BY CONTRACT.
#ifndef STEPPE_APP_PRECISION_LABEL_HPP
#define STEPPE_APP_PRECISION_LABEL_HPP

#include "steppe/config.hpp"  // steppe::Precision

namespace steppe::app {

/// A human label for the resolved precision (recorded in an f2_blocks meta.json;
/// cli-bindings.md §4.3 — the ENGAGED tag). A SWITCH (not a ternary) so a new
/// Precision::Kind is a compile-visible add here; the trailing `return "fp64"` keeps
/// native FP64 the fallback label (Precision::Kind is defined in steppe/config.hpp).
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
