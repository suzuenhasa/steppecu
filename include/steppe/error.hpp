// include/steppe/error.hpp
//
// Minimal public error/status taxonomy for steppe — the trimmed §10 set.
//
// This is the foundational status enum the M0 contract files build against. The
// full ABI surface (architecture.md §10, §16) is a richer C enum
// `steppe_status_t` plus an internal `Error` view; this header carries the small
// strongly-typed C++ subset the early layers need so they can return outcomes
// without dragging in the whole ABI. CUDA-free; standard-library-free.
//
// The three DOMAIN-OUTCOME values (RankDeficient, NonSpdCovariance) are
// *expected* results of fitting some models in a large search — surfaced as
// ordinary statuses, never exceptions or aborts (architecture.md §10). Faults
// (e.g. InvalidConfig at build time) are fail-fast.
#ifndef STEPPE_ERROR_HPP
#define STEPPE_ERROR_HPP

namespace steppe {

/// Trimmed status taxonomy (architecture.md §10).
enum class Status {
    /// Success.
    Ok,

    /// Device allocation or VRAM-budget check failed (resource; maybe
    /// recoverable with a smaller chunk/budget). architecture.md §11.2.
    DeviceOom,

    /// Domain outcome (recoverable): the rank test / GLS hit a rank-deficient
    /// design matrix `X`; the model is unidentifiable. A statistical result of
    /// fitting a degenerate model, not a bug.
    RankDeficient,

    /// Domain outcome (recoverable): the covariance `Q` is not SPD; Cholesky
    /// failed. A degenerate/collinear model, not a bug.
    NonSpdCovariance,

    /// Configuration failed `ConfigBuilder::build()` validation (bad arch list,
    /// conflicting flags, over-budget VRAM, unhonorable precision). Fail-fast.
    InvalidConfig
};

}  // namespace steppe

#endif  // STEPPE_ERROR_HPP
