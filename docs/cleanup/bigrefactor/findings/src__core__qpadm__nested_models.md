# Review findings — src__core__qpadm__nested_models

Files: /home/suzunik/steppe/src/core/qpadm/nested_models.cpp, /home/suzunik/steppe/src/core/qpadm/nested_models.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.1][LOW] nested_models.hpp:10-12 — Stale doc comment: header still describes the impl as "an n_block host loop over gls_weights (correctness first) ... the batched device S7 is M(fit-3)", but se_from_loo now calls the batched seam gls_weights_loo_batched (nested_models.cpp:57, documented at .cpp:52-56). The described host-loop code no longer exists. Suggested: update the header comment to reflect the batched-seam implementation (not dead code, stale narrative).

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] nested_models.cpp:50 — Bare `2` in the early-out guard `if (nb < 2 || nl <= 0)`. It encodes the jackknife minimum (need ≥2 delete-1 replicates to form a covariance), but the literal is unnamed and the rationale lives only in the reader's head. Suggested: name it (e.g. `kMinJackknifeBlocks = 2`) or add an inline comment "// jackknife needs >=2 blocks". (The `(nb-1)/sqrt(nb)` scale at :60 and the Bessel `nrows-1` divisor at :34 are documented, parity-load-bearing statistical formulas — NOT flagged.)

## Group 6 — Naming

- [6.1][LOW] nested_models.cpp:16-17 — The `sample_cov_diag` parameter `w` is a single-letter, opaque name for the (nrows × ncols) row-major data matrix; it is a function parameter, not a tight-loop counter, so the cryptic form is not self-documenting. The doc comment (:12) does explain it, but the name alone reads as a placeholder. Suggested: rename to `data`/`mat`/`rows` (e.g. `data_rowmajor`). (Loop counters `i`,`c` and the per-element diff `d` at :30 are legitimate tight-loop locals — NOT flagged. `nl`,`nb`,`nr`,`wmat`,`se`,`z`,`be`,`cov` are AT2/codebase-standard domain abbreviations and are documented — NOT flagged.)

## Group 7 — Duplication

- [7.2][LOW] nested_models.cpp:21-23, :30-31 — The row-major flattened index `w[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncols) + static_cast<std::size_t>(c)]` is written verbatim in both the mean-accumulation loop (:21-23) and the variance loop (:30-31). The duplicated expression is identical and verbose; a divergent edit to one (e.g. fixing an indexing bug) would silently miss the other. Suggested: factor a tiny `[&](int i,int c){ return w[size_t(i)*size_t(ncols)+size_t(c)]; }` accessor (or a `row_major_at` helper) and call it in both loops.
- [7.3][LOW] nested_models.cpp:18-34 — Pervasive repeated `static_cast<std::size_t>(...)` casts on the same operands (`c`, `i`, `ncols`, `nrows`) appear ~12 times across `sample_cov_diag`; the index math is buried in cast boilerplate. The widening is correct/needed at scale (NOT flagged numerically), but the repetition is pure hygiene. Suggested: hoist `const std::size_t nc = static_cast<std::size_t>(ncols);` once and/or use the accessor helper from 7.2 so the casts live in one place.
- [7.2][LOW] nested_models.cpp:30-31 — `mean[static_cast<std::size_t>(c)]` is loop-invariant w.r.t. the inner `i` loop but is re-indexed (cast + vector subscript) on every one of `nrows` iterations. Minor, but it is a repeated loop-invariant subexpression. Suggested: hoist `const double mc = mean[static_cast<std::size_t>(c)];` above the inner loop.

## Group 8 — Comments

- [8.2][LOW] nested_models.hpp:30-35 — Stale docstring on `se_from_loo`: it states the weights are re-solved "via `be.gls_weights`, reusing `cov.Qinv`", but the implementation now calls `be.gls_weights_loo_batched` (nested_models.cpp:57). The named virtual no longer matches the code, so a reader looking up the contract is pointed at the wrong seam. Suggested: update the docstring to name `gls_weights_loo_batched` (the batched-capable seam). (Distinct from the Group-3 [3.1] flag at .hpp:10-12, which is the separate "M(fit-1) host loop" narrative.)
- [8.2][LOW] nested_models.hpp:10-12 — Stale header narrative: "In M(fit-1) this is an n_block host loop over gls_weights ... the batched device S7 is M(fit-3)." The code no longer hard-codes a host loop — it dispatches through the batched seam (nested_models.cpp:57, documented at .cpp:52-56). The comment describes a past milestone's behavior the code no longer has. (Already raised under Group 3 [3.1]; restated here as the squarely-Comments duplicate — fix once.) Suggested: rewrite the header to describe the current backend-agnostic batched-seam dispatch.
- [8.3][LOW] nested_models.cpp:28,32,34 — Missing rationale for the `long double` variance accumulator: `acc` is `long double` while the rest of the routine (and the project) is `double`-by-design. The widened accumulation is a deliberate precision choice (reduce catastrophic-cancellation error in the sum-of-squares for parity), but no comment says so; a future reader could "simplify" it back to `double`. Suggested: add a one-line note, e.g. "// long double accumulator: extra precision for the sum-of-squares to match AT2 cov()".

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.

