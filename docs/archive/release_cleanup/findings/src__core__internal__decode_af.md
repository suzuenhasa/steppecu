# src__core__internal__decode_af
Files: /home/suzunik/steppe/src/core/internal/decode_af.hpp
Subsystem: core-stats

## Findings

### G5
- [G5.G5][LOW] decode_af.hpp:117 — bare `0x3u` 2-bit mask in `genotype_code` is not derived from the named `kBitsPerCode` packing radix that the surrounding shift (`6,4,2,0`) IS derived from. If `kBitsPerCode` ever changed the shift would update but the mask would silently drift (the documented invariant `8 = kCodesPerByte·kBitsPerCode` is exactly what makes a 2-bit field). Suggested: derive the mask from `kBitsPerCode` (e.g. a named `kCodeMask = (1u << kBitsPerCode) - 1`) so mask and shift share one source.
- [G5.G5][LOW] decode_af.hpp:175 — bare literal `3.0` in `accumulate_genotype_ploidy` (`code / (3.0 - ploidy)`). It is the AT2 `3.0 - ploidy(i)` constant (= max-ploidy+1 / "haploidization divisor base"), heavily explained in the comment but unnamed in code, and it appears only here (the analogous comment text at line 32/151 restates it). Low drift risk since single-use, but it is an unnamed numeric tied to a hard semantic. Suggested: a named `constexpr double kPloidyDivisorBase = 3.0` (or document inline that it is intentionally bare) to match the file's own single-home-the-constant discipline.

### G7
- [G7.G7][LOW] decode_af.hpp:214-251 — `finalize_af` and `finalize_af_counts` share the same `AfResult r; if (cond) { r.n = …; r.q = ac/n; r.v = 1.0; }` skeleton, differing only in the guard (`an>0 && ploidy>0` vs `n>0`) and how `N` is formed (`ploidy*an` vs `n`). They are deliberately documented as analogues for two distinct accumulation paths and the bodies are tiny, so extraction would likely obscure more than it dedups. Flagged only for completeness; leaving as-is is defensible. Suggested: none required; if consolidated, a private helper taking precomputed `(ac_double, n_double, valid)` would unify the divide.
