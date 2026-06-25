# include__steppe__config
Files: /home/suzunik/steppe/include/steppe/config.hpp
Subsystem: public-api

## Findings
No issues found (groups checked: G2-G10).

This header is the project's named-constant single-source-of-truth (the explicit
anti-magic-number home, ROADMAP §4) and is deliberately CUDA-free, so G2/G11-G22
are N/A. Verified clean against the remaining groups:
- G3: all three includes are used (`<cstddef>`→`std::size_t`, `<string>`→`std::string`,
  `<vector>`→`std::vector`); every constant documents a real consumer; no dead/unused symbols.
- G4: byte-count constants use `std::size_t`; `kFitBudgetFreeVramFallbackBytes`/`kFitBudgetHeadroomBytes`
  (lines 130-131) widen with `static_cast<std::size_t>(4) << 30` BEFORE the shift (no int-overflow);
  `kFstatMaxComb` is `unsigned long long` (line 54). No byte-vs-element or index-width bugs — this is a
  config header, not an indexing path. The FP64 `double` literals are intentional/parity-load-bearing.
- G5: the ambiguous-32 case (`kGesvdjMaxDim = 32`, line 112) is the exact pattern G5 targets, but it is
  already named AND explicitly disambiguated in-comment ("the cuSOLVER routine-selection threshold, NOT a
  warp size", lines 109-110); no bare/duplicated literals remain.
- G6: consistent `k`-prefixed `constexpr` naming throughout; struct fields use clear lower_snake.
- G8: comments carry rationale (measured values, §-cross-refs, single-source notes), none stale; no orphan
  TODO/FIXME/HACK.
- G9: every constant is `inline constexpr`; struct fields are all named (no positional-boolean APIs); the
  two `static_assert`s (lines 222-225) guard the tunable-fraction invariants.
- G10: all struct members are default-initialized at declaration.
