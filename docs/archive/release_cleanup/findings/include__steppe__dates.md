# include__steppe__dates
Files: /home/suzunik/steppe/include/steppe/dates.hpp, /home/suzunik/steppe/src/core/stats/dates.cpp
Subsystem: core-stats

## Findings

### G3 dead/unused
- [G3.dead][MED] src/core/stats/dates.cpp:29 — `#include "core/domain/block_partition_rule.hpp"` brings in no symbol used in this TU (no `BlockPartition`/`partition_rule`/`PartitionRule` reference); chrom segmentation here is the hand-rolled `touch_chrom` lambda (lines 221-247). Unused include. Suggested: drop the include.
- [G3.dead][LOW] src/core/stats/dates.cpp:24 — `#include <limits>` unused; no `std::numeric_limits` anywhere (NaN comes from `std::nan` in `<cmath>`). Suggested: remove `<limits>`.
- [G3.dead][LOW] src/core/stats/dates.cpp:25 — `#include <stdexcept>` unused; no `throw`/`runtime_error`/exception type appears in this TU (io faults are thrown inside the io layer, not here). Suggested: remove `<stdexcept>`.
- [G3.dead][LOW] src/core/stats/dates.cpp:36 — `#include "io/genotype_source.hpp"` provides `GenotypeSource`, which is never referenced directly (the path used is `GenoReader` -> `read_canonical_tile`). Likely only transitively relevant. Suggested: confirm and drop if not needed for IWYU.

### G5 hardcoded values / magic numbers
- [G5.magic][MED] src/core/stats/dates.cpp:279 — unnamed epsilon `1.0e-20` in the `corr_from` denominator `std::sqrt(v11 * v22 + 1.0e-20)`. Load-bearing divide-by-zero guard with no named rationale. Suggested: hoist to a named `constexpr` (e.g. `kCorrDenomFloor`) with a one-line rationale.
- [G5.magic][LOW] src/core/stats/dates.cpp:70 — weightjack zero-weight filter threshold `1e-6` is an unnamed buried tunable (the comment names it as "weightjack's jwt<1e-6 filter" but the constant itself is inline). Suggested: name as `constexpr long double kMinJackWeight`.
- [G5.magic][LOW] src/core/stats/dates.cpp:184 — autosome range `chr < 1 || chr > 22` uses bare `1`/`22`. The `22` autosome-max convention is unnamed. Suggested: name `kMaxAutosome = 22` (with the existing `kInterChromGapMorgans` group).

### G7 duplication
- [G7.dup][LOW] src/core/stats/dates.cpp:307-308, 320 — the bin-center distance formula `(static_cast<double>(s) + 1.0) * opts.binsize_morgans` is repeated (curve emit at 307-308 with `*kCentimorgansPerMorgan`, fit window at 320 in Morgans). Same "center = (k+1)*binsize" convention duplicated. Suggested: a small `bin_center_morgans(s)` helper to keep the +1 offset convention single-sourced.

### G8 comments
- [G8.stale][MED] src/core/stats/dates.cpp:29 — trailing comment `// (reused elsewhere; chrom segmentation here)` on the `block_partition_rule.hpp` include is misleading: nothing from that header is used here and the chrom segmentation is done by the local `touch_chrom` lambda, not a partition rule. Stale/misleading rationale. Suggested: remove with the include.

### G4 type & numeric
- [G4.numeric][LOW] src/core/stats/dates.cpp:252-254 — `n_bin`/`diffmax` are `static_cast<int>(std::lround(...))` of user-controlled `maxdis_morgans/binsize_morgans` (and `qbin * maxdis/binsize`). At the AADR/DATES default scale these are ~1000 / ~10000 (safe), but a pathological `opts` (tiny binsize, large maxdis) could overflow `int` before the multiply at 253-254. Not a scale bug on real data; defaults are validated at 116. Suggested: widen to `long` or bound-check the derived grid sizes if arbitrary `opts` is ever exposed.
