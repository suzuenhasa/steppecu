# src__io__filter__mind_prepass
Files: /home/suzunik/steppe/src/io/filter/mind_prepass.cpp, /home/suzunik/steppe/src/io/filter/mind_prepass.hpp
Subsystem: io

## Findings

### G3
- [G3.dead-code][LOW] mind_prepass.cpp:58 — `constexpr auto kPerByte` is declared inside the inner per-SNP loop body, so its scope is re-entered every iteration. It is a compile-time constant (no runtime cost), but its placement inside the hot inner loop is gratuitous; it is a loop-invariant the rest of the file (e.g. line 47's `n_snp_d` hoist) deliberately lifts out. Suggested: hoist the `kPerByte` declaration to function scope (or just above the `ind` loop) alongside `n_snp_d`, matching the file's own hoisting convention.
