// src/core/config/exit_code.hpp
//
// Status -> process exit-code map (the CLI's record-and-continue contract;
// cli-bindings.md §1 constraint 3, §4.4 "Status handling", architecture.md §10).
//
// THE LAW (cli-bindings.md §1.3 / §4.4): a per-MODEL DOMAIN OUTCOME
// (RankDeficient / NonSpdCovariance / ChisqUndefined) is an ordinary statistical
// result of fitting a degenerate model in a (possibly huge) search — it is emitted as
// a `status` COLUMN on that row and the run CONTINUES; a completed run exits 0 even
// when some rows carry a domain outcome. Only FAULTS — InvalidConfig (bad config),
// DeviceOom (resource), and the file/format I/O errors the app raises — exit NONZERO.
//
// This is the single source of that mapping so the app cannot drift to a second
// convention. CUDA-FREE, std-free except <cstdlib> for the canonical EXIT_* values:
// it names only the public steppe::Status enum (steppe/error.hpp). The app calls it
// from main()'s catch/return paths; the library itself never calls std::exit (§10 —
// the app owns process control).
#ifndef STEPPE_CORE_CONFIG_EXIT_CODE_HPP
#define STEPPE_CORE_CONFIG_EXIT_CODE_HPP

#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

#include "steppe/error.hpp"  // steppe::Status

namespace steppe::config {

/// Canonical CLI exit codes. EXIT_SUCCESS (0) for a completed run (even with per-row
/// domain outcomes); distinct nonzero codes for the fault categories so a calling
/// script can branch (cli-bindings.md §4.4; architecture.md §10 fault taxonomy).
/// The values are small, stable, and != EXIT_SUCCESS for every fault.
enum CliExitCode : int {
    kExitOk            = EXIT_SUCCESS,  ///< 0 — run completed (domain outcomes are rows, not faults)
    kExitInvalidConfig = 2,            ///< bad config: failed ConfigBuilder::build() validation
    kExitDeviceOom     = 3,            ///< device allocation / VRAM-budget fault (§11.2)
    kExitIoError       = 4,            ///< file / format I/O fault (missing f2-dir, bad pops.txt, ...)
    kExitRuntimeError  = 5,            ///< a CUDA-runtime / unexpected fault (the app's catch-all)
};

/// Map a top-level run Status to the process exit code (cli-bindings.md §1.3 / §4.4).
///
///   Ok                                  -> kExitOk (0)
///   RankDeficient / NonSpdCovariance /
///   ChisqUndefined  (DOMAIN OUTCOMES)   -> kExitOk (0)  [record-and-continue: a
///                                          domain outcome is a per-model ROW, never a
///                                          process-level failure — critical so a
///                                          rotation over thousands does not abort on
///                                          one degenerate model]
///   InvalidConfig                       -> kExitInvalidConfig (2)
///   DeviceOom                           -> kExitDeviceOom (3)
///
/// NB: a domain Status reaching THIS function only ever happens if a caller routes a
/// per-model outcome to the top level by mistake; the map still returns 0 for it,
/// enforcing the record-and-continue contract structurally rather than by convention.
[[nodiscard]] constexpr int exit_code_for(Status status) noexcept {
    switch (status) {
        case Status::Ok:
        // The three DOMAIN OUTCOMES are recoverable per-model results, NOT faults:
        // exit 0 (the run completed; the outcome is a row's `status` column).
        case Status::RankDeficient:
        case Status::NonSpdCovariance:
        case Status::ChisqUndefined:
            return kExitOk;
        case Status::InvalidConfig:
            return kExitInvalidConfig;
        case Status::DeviceOom:
            return kExitDeviceOom;
    }
    // Unreachable for the closed enum; a defensive nonzero keeps an added-but-unhandled
    // future status from masquerading as success.
    return kExitRuntimeError;
}

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_EXIT_CODE_HPP
