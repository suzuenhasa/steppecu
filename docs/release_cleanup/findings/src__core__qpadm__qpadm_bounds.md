# src__core__qpadm__qpadm_bounds
Files: /home/suzunik/steppe/src/core/qpadm/qpadm_bounds.hpp
Subsystem: core-qpadm

## Findings

### G5
- [G5.src__core__qpadm__qpadm_bounds][LOW] qpadm_bounds.hpp:48,50 — The `// 50` and `// 40` trailing comments manually restate the result of the adjacent `constexpr` derivations (`kQpMaxNl*kQpMaxNr`, `max(nl,nr)*kQpMaxR`). If the base bounds `kQpMaxNl/kQpMaxNr/kQpMaxR` change, these literal comments silently go stale while the computed values stay correct — a low-risk drift since the value itself is compiler-derived. Suggested: drop the literal comment (or phrase it as "currently 50" / "= nl*nr") so it cannot read as an authoritative second source.
