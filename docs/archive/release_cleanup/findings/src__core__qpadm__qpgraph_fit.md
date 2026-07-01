# src__core__qpadm__qpgraph_fit
Files: /home/suzunik/steppe/src/core/qpadm/qpgraph_fit.cpp
Subsystem: core-qpadm

## Findings

### G3 dead/unused
- [G3.src__core__qpadm__qpgraph_fit][LOW] qpgraph_fit.cpp:20 — `#include <algorithm>` is unused. No `std::` algorithm (max/min/sort/find/transform/etc.) appears in the file; the worst-residual argmax (lines 152-159) is hand-rolled with explicit comparison rather than `std::max_element`/`std::abs`. Suggested: drop the `<algorithm>` include (or, if intent was to use it, switch the argmax loop to `std::max_element` with a custom comparator).

### G7 duplication
- [G7.src__core__qpadm__qpgraph_fit][LOW] qpgraph_fit.cpp:101 — `const int npair = X.nl * X.nr;` re-derives a quantity already carried as `m.npair` (set in qpgraph_model.cpp:98 as `npop*(npop-1)/2`, and the `flat` triple array on lines 91-98 is built with exactly `m.npair` entries, so `X.nl*X.nr == m.npair` by construction). The two are an invariant pair that can silently drift if the seam contract changes. This reads as defensive validation of the `assemble_f3_triples` output, which is reasonable, but the relationship is undocumented. Suggested: either add a short comment that `npair` must equal `m.npair` (or assert it in debug), or reuse `m.npair` directly and validate `X.nl*X.nr` against it explicitly.

### G10 initialization
- [G10.src__core__qpadm__qpgraph_fit][LOW] qpgraph_fit.cpp:152 — `double worst = 0.0;` is used both as the running argmax accumulator and as the reported `res.worst_residual_z` default when no finite residual is found (worst_k stays -1, lines 160-161 leave `worst` = 0.0). Initializing the "max |z|" tracker to 0.0 also means a genuine residual of exactly 0 can never become the recorded worst (`std::fabs(z) > std::fabs(worst)` with worst==0 is false), so a degenerate all-zero-residual basis reports worst_k=-1 and no labels. This is benign for the real data envelope but the dual role of the 0.0 sentinel is non-obvious. Suggested: initialize with a sentinel intent comment, or track `best_absz` separately from the signed `worst` value.
