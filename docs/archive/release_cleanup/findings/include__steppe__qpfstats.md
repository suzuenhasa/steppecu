# include__steppe__qpfstats
Files: /home/suzunik/steppe/include/steppe/qpfstats.hpp, /home/suzunik/steppe/src/core/stats/qpfstats.cpp
Subsystem: core-stats

## Findings

### G3 dead/unused
- [G3.dead][LOW] src/core/stats/qpfstats.cpp:48 — `#include <stdexcept>` is unused: there is no `throw`/`runtime_error`/`logic_error` in the file (the io-fault exceptions originate inside the io readers, not here). The header comment "An io fault PROPAGATES as an exception" describes downstream behavior, not a throw in this TU. Suggested: drop `#include <stdexcept>`.
- [G3.dead][LOW] src/core/stats/qpfstats.cpp:45 — `#include <cmath>` is unused: no `std::sqrt/abs/isnan/isfinite/pow/fabs` (or any `<cmath>` symbol) appears in the file. NaN handling lives in the device seam. Suggested: drop `#include <cmath>`.

### G4 type & numeric
- [G4.narrow][LOW] src/core/stats/qpfstats.cpp:164 — `npopcomb = static_cast<int>(combs.size())` narrows a `size_t` to `int`. For the documented qpfstats envelope (small pop sets, n=9 → 666 combos) this is safe, but `npopcomb` grows ~C(npop,4)*3 and silently overflows `int` for a large pop set; the overflowed `npopcomb` then feeds the `size_t` design-matrix sizing at line 173 (`npopcomb * npairs`) where the multiply happens AFTER the int has already wrapped. Suggested: assert `combs.size()` fits `int`, or carry `npopcomb` as `std::size_t` end-to-end.

### G5 hardcoded values / duplicated constants
- [G5.dup][MED] src/core/stats/qpfstats.cpp:292 — the CpuBackend (oracle) autosome filter hardcodes `if (chr < 1 || chr > 22) continue;` as raw literals, while the resident path at line 281 passes the named `kAutosomeChromMin, kAutosomeChromMax` (config.hpp = 1/22). Same bound expressed two ways = drift risk: changing the autosome range in config.hpp would silently diverge the oracle from the production path, breaking the "BOTH produce the IDENTICAL kept SET" invariant asserted in the comment at line 271. Suggested: use `chr < kAutosomeChromMin || chr > kAutosomeChromMax` on the oracle path too.

### G8 comments
- [G8.rationale][LOW] src/core/stats/qpfstats.cpp:104-113 — the `build_popcomb_and_design` doc comment contains a visibly unfinished derivation ("Pure-f2 (A,B,A,B): pair(A,B) gets (1+1)*2/2 = 2 ... wait it collapses to ... = 2/2·... ).") with a literal "wait" and a trailing ellipsis. The actual, correct semantics (assignment-not-accumulation, the `×2` then `/2`) are restated cleanly at lines 167-172, so lines 104-113 read as stream-of-consciousness scratch left in a header comment. Suggested: delete the "...wait it collapses..." aside and keep the precise 167-172 explanation.
