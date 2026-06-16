# Review: core-fstats-f2_from_blocks (adversarial second pass — final)

Unit under review:
- `/home/suzunik/steppe/src/core/fstats/f2_from_blocks.hpp` (46 lines)
- `/home/suzunik/steppe/src/core/fstats/f2_from_blocks.cpp` (55 lines)

Context read in full this pass (line-by-line, not skimmed):
`src/device/backend.hpp` (the interface this unit drives — the precondition docstrings at :130–133 and
:154–169, and the exact `compute_f2_blocks` signature `(Q,V,N, const int* block_id, int n_block, const
Precision&)` at :164–169), `src/core/internal/views.hpp` (`MatView`, `P:int`/`M:long`, the no-bounds-check
`element`), `src/core/domain/block_partition_rule.hpp` + `.cpp` (`BlockPartition`, the
`n_block == max(block_id)+1` / `block_id.size()==M` / dense-non-decreasing invariant established by
`assign_blocks`, and the empty-input → `{} , 0` contract at .cpp:25–27),
`include/steppe/fstats.hpp` (`F2BlockTensor`, the public M4 handle), `include/steppe/config.hpp`
(`Precision`, `kBlockGroupPadBase`, `kDefaultBlockSizeCm`, `kCdivBlock`, `kCentimorgansPerMorgan`,
`kDefaultMantissaBits`, `kCublasWorkspaceBytes` — all confirmed present as named constants),
`include/steppe/error.hpp` (the fail-fast status surface; **no `STEPPE_EXPECTS`/`STEPPE_ASSERT` macro
exists anywhere in the tree — grep-confirmed, see below**), `src/core/internal/f2_estimator.hpp`
(`het_correction`/`f2_term`/`finalize_f2`/`assemble_f2_numerator` — where the real numerics live),
`src/core/CMakeLists.txt` (the layering targets), `src/device/cpu/cpu_backend.cpp` +
`src/device/cuda/cuda_backend.cu` (both `compute_f2_blocks` implementations — to see exactly what the
forwarded `block_id`/`n_block` feed and where they are dereferenced), `tests/unit/test_f2.cpp` +
`tests/CMakeLists.txt` (the host-only CUDA-free unit-test idiom that already exists in the tree),
`tests/reference/test_f2_blocks_equivalence.cu` (the only caller in the tree, incl. the hand-built `one`
partition at :251), `docs/architecture.md` §2/§4/§11.4/§12/§13, `docs/ROADMAP.md` §6 (the
"compute-sanitizer memcheck+racecheck clean" DoD gate at :131), `docs/TODO.md` capability-tier table
(:125–137, the M4.5 P2P combine + cross-cutting capability-tag rows).

External / standard-library behavior verified this pass (not asserted from memory):
- **`std::vector::data()` empty-vector behavior** — confirmed against the normative C++ working draft
  ([eel.is/c++draft/vector.data]): the return is specified only so that `[data(), data() + size())` is a
  valid range; `data() == addressof(front())` is guaranteed **only for a non-empty vector**, so a **null**
  return for an empty vector is standard-conforming. (Fetched and quoted this pass.) This is load-bearing
  for understanding F-1's null-pointer mechanism — see the F-2 correction in Considered & rejected.

Grep results that anchor the findings:
- `grep -rn "STEPPE_EXPECTS\|STEPPE_ASSERT\|<cassert>\|assert(" src/ include/` → **zero hits**. There is
  no precondition-checking infrastructure anywhere in steppe today; the unit is consistent with the
  current (absent) house style, which is itself the gap F-1/F-3 name.
- `grep -rln "throw\|abort\|terminate\|fprintf(stderr\|invalid_argument\|logic_error" src/core/` → **zero
  hits**. `core` currently performs no fail-fast of any kind.
- Host-only CUDA-free unit tests already exist and link first-party targets with **no GPU**:
  `test_f2_estimator` (links `steppe::core_internal` only), `test_block_partition`, `test_filters`
  (tests/CMakeLists.txt) — so the testability fix F-12 has direct, working precedent.

## Role & layering

This unit is the **composition root** of the f2 stage: the `core`-owned host orchestration that drives the
f2 computation *through* the `ComputeBackend` dependency-injection seam (architecture.md §2, §4, §5 S2,
§8). It embodies one of the repo's central design claims — the same orchestration runs unchanged against
`CpuBackend` (oracle) and `CudaBackend` (3-GEMM / grouped batched), which is what makes the pipeline
GPU-free-testable. The two free functions are deliberately thin: `compute_f2_block` (M0, single block) and
`compute_f2_blocks` (M4, per-block tensor). Both are pure forwarders.

Layering is **clean and correct**, the most important property for a `core` unit: the `.cpp` directly
includes only `device/backend.hpp` (the CUDA-free seam), `core/internal/views.hpp`, `steppe/config.hpp`,
and its own header (which additionally pulls `block_partition_rule.hpp` + `fstats.hpp`). No CUDA header, no
`<cuda_runtime.h>`, no cuBLAS — exactly the §4 dependency-direction rule. CMake confirms it (`src/core/
CMakeLists.txt:47–59`): `steppe_core` links `steppe::device PRIVATE` with `CUDA_RESOLVE_DEVICE_SYMBOLS
OFF`, so `core` *physically cannot* leak a CUDA dependency. `[[nodiscard]]` is on both functions.

Much of this unit is genuinely near-perfect, and I say so explicitly in the relevant categories. But a thin
forwarder is not automatically a *correct* forwarder, and there is one real, shipped robustness gap: the
`BlockPartition` is forwarded with **no** cross-field validation, so an inconsistent partition (`n_block`
disagreeing with the contents of `block_id`) drives an out-of-bounds write inside *both* backends that no
test catches. Around that sit a cluster of fail-fast / testability / doc-rot / include-hygiene gaps that
keep it short of the 9.5–10 bar.

## Score: 8/10 — clean layering and a correct DRY dispatch, but it forwards a `BlockPartition` whose two halves it never cross-checks (driving a real OOB write in the backend on an inconsistent partition — including a standard-conforming null `block_id.data()`), validates none of the documented Q/V/N preconditions at the one host point that sees all three views, ships a stale "M4 is a later milestone" file header for code that is right there, has a direct-use-vs-include IWYU inconsistency, and is untested in isolation.

Net change vs the prior second-pass draft: **+1 finding added** (F-17, the IWYU direct-use-but-not-
directly-included inconsistency, against the real §791/§793 IWYU gate); **−1 rejected** (the prior pass's
**F-2 was a false positive as framed**: its scenario "non-empty Q with empty partition → null-`data()`
deref" cannot happen, because an empty partition has `n_block==0` and both backends early-out on
`n_block<=0` *before* dereferencing `block_id`; the genuine null-`data()` UB only arises when `n_block>0`
while `block_id` is empty/short, which is exactly the F-1 inconsistency. The null-pointer mechanism is
preserved, correctly, *inside* F-1; the spurious independent-HIGH is moved to Considered & rejected). Also
corrected: the prior pass's own rewritten F-1 already fixed the *original* auditor's "discards `n_block`"
error — that correction stands and is re-confirmed. Several "not a defect" audit entries (old F-6, F-10,
F-16) compressed into a single completeness note to avoid padding. Severities otherwise held.

## Findings

### (1) Correctness & bugs

**F-1 [HIGH] — `compute_f2_blocks` forwards both `partition.n_block` and `partition.block_id` but never
cross-checks that they are mutually consistent, so an inconsistent `BlockPartition` drives an
out-of-bounds write inside *both* backends.**
Location: `.cpp:50–51` (`return backend.compute_f2_blocks(Q, V, N, partition.block_id.data(),
partition.n_block, precision);`).
Verified end-to-end this pass. The `BlockPartition` invariant (block_partition_rule.hpp:80–89; established
by `assign_blocks`, .cpp:51,56) is `n_block == max(block_id)+1`, `block_id.size() == M`, ids
dense/non-decreasing. **Both** backends *trust* the forwarded `n_block` to size their per-block arrays —
CPU `begin`/`end`/`block_sizes` are `n_block`-long (cpu_backend.cpp:212–213,202) and CUDA
`block_offsets`/`dOffsets`/`dSizes` are `n_block`-long (cuda_backend.cu:136,183–184) — then index those
arrays with `b = block_id[s]` while scanning all `M` columns (`begin[b]`/`end[b]`/`out.block_sizes[b]`,
cpu:220–222; `block_offsets[b]`/`out.block_sizes[b]`, cuda:143–144). If a caller hands a `BlockPartition`
whose `n_block` is **smaller** than `max(block_id)+1` — e.g. a partition assembled by hand (the test
builds `one` by hand at test:251; a future merged-dataset path may recompute `n_block` independently), or
one mutated after `assign_blocks` — the backend writes `begin[b]`/`block_offsets[b]`/`out.block_sizes[b]`
with `b >= n_block` → heap corruption / UB on the CPU path and an OOB host-vector write feeding a device
copy on the CUDA path.
**Sub-case — null/short `block_id.data()`** (the corrected F-2): if `block_id` is empty (so `data()` may
legally return `nullptr` — verified against the C++ draft) *and* `n_block > 0` (the inconsistency), the
`n_block<=0` early-out (cpu:207, cuda:130) is **skipped** and the scan dereferences a null/short pointer
at `block_id[s]`. This is a real null-deref, but it is a *consequence of the same missing invariant
check*, not an independent bug — folding it here.
Note both backends already cheaply *recompute* `block_sizes` from `block_id` (they don't trust a
partition's sizes), so they *could* equally derive `n_block` and make this moot — but **today neither the
orchestration nor the backends validate it**, so the bug is live on both paths.
Why it matters: architecture.md §2 fail-fast (faults surface "with file/line context, not as silent
corruption", §2:67) and §5/§8 single-source-of-truth — the orchestration *owns the block policy*, so it
owns the invariant *before* the data crosses the CUDA-free seam into code that cannot re-derive it. It is
also a §6 definition-of-done concern: "compute-sanitizer (memcheck) clean" (ROADMAP §6:131) holds only
because no test feeds an inconsistent partition — see F-14.
Fix: in `compute_f2_blocks`, before dispatch, enforce the invariant. The codebase has **no** assertion
macro yet (grep-confirmed); introducing `<cassert>` (CUDA-free) is the minimal tool, *or* this is the
forcing function to add the long-absent `STEPPE_EXPECTS` (architecture.md §2 fail-fast names file/line
context — a `std::source_location`-carrying check would match the `STEPPE_CUDA_CHECK` house style at
backend §453). At minimum check `partition.block_id.size() == static_cast<size_t>(Q.M)`, and when
`block_id` is non-empty: `n_block > 0`, `n_block <= Q.M`, and (debug, O(M)) `*max_element(block_id) <
n_block` with ids non-decreasing. Cheapest structural option: pass a `std::span<const int>` (F-9) so the
length travels with the pointer and the backend can cross-check.
Severity: high. Effort: S. Before M4.5: **yes** (guards the M4 deliverable's input; the gating
prerequisite for the F-14 sanitizer test).

**F-2 [MED] — neither function checks that Q, V, N share the same P and M, though the backend documents
this as a precondition it assumes.**
Location: `.cpp:31–39` and `.cpp:41–52`; the precondition is stated at backend.hpp:130–131 ("Preconditions:
Q, V, N share the same P and M and refer to the same SNP block") and re-implied at :154–157, but **enforced
nowhere**. The orchestration forwards three independent `MatView`s. If e.g. `V.M < Q.M`, the CPU oracle's
`V.element(i, s)` (cpu_backend.cpp:248,127) reads past `V`'s storage with no bound (views.hpp:62–68 does
*no* bounds check — "hot path"); the CUDA path copies `pm = P·(Q.M)` doubles out of a too-short `V.data`
(cuda:78,196–201) → OOB read. The orchestration is the natural single home for this guard: the backends
are "written once against the interface, never branching" (backend.hpp:12–13) and should not each
re-litigate input validity.
Why it matters: a documented precondition with no fail-fast guard at the one host point that sees all three
views violates §2 (fail-fast over silent corruption) and §13 (the seam should reject malformed contracts
loudly — both backends are diffed *through* it).
Fix: one shared precondition check that `Q.P==V.P && V.P==N.P && Q.M==V.M && V.M==N.M`. Co-locate with F-3
in a `validate_qvn` helper (F-8).
Severity: med. Effort: S. Before M4.5: **yes.**

### (2) Edge cases & failure modes

**F-3 [MED] — no sign-validation of `P`/`M` at the seam (overflow itself is *not* a real risk; sign is).**
`MatView::P` is `int`, `M` is `long` (views.hpp:57,60). Both backends compute `slab = (size_t)P*(size_t)P`
and `total = slab * (size_t)(n_block<0?0:n_block)` (cpu:203–204, cuda:125–126). They defend `n_block<0`
but **not** `P<0`: a negative `P` cast to `size_t` becomes ~1.8e19 → `assign(total, …)` either throws
`std::length_error` or attempts a multi-exabyte allocation; on the CUDA path `pm = (size_t)P*(size_t)M`
with `P<0` produces a colossal `cudaMalloc`. The orchestration passes `P` through untouched.
The integer-width story is otherwise **good and I verified it**: `M` is deliberately `long`
(views.hpp:49–51) to avoid 32-bit SNP overflow, and `MatView::element` promotes `i` and `P` to `long`
before the multiply (views.hpp:66–67), so for realistic P (≤ a few thousand) and `M`≈584k there is no
overflow of *legitimate* values. The realistic risk is a **negative/garbage P** from an uninitialized
view, not overflow of real data.
Why it matters: §2 fail-fast; a guard at the composition root is cheaper than in two backends.
Fix: fold `P >= 0 && M >= 0` into the F-2 `validate_qvn` helper.
Severity: med. Effort: S. Before M4.5: no (covered for free once F-2 lands).

**F-4 [LOW] — the empty/degenerate path is *correct by accident*, not by contract, and is untested through
this unit.** For `compute_f2_block`, an empty block (`Q.M==0`) is handled entirely inside the backend (the
`M<=0` early-out); the orchestration does nothing, which is fine *given consistent inputs*. For
`compute_f2_blocks`, an empty partition (`n_block==0`) is also handled by the backend early-out. But there
is no test exercising *this unit* with `M==0` or `n_block==0` — the equivalence test always uses real AADR
(`M`≈100k, test:198,214). The empty path through this unit is unverified. See F-12.
Severity: low. Effort: S. Before M4.5: no.

### (3) Numerical / precision vs §12

**N/A as a source of defects — verified, not assumed.**
This unit performs **no arithmetic**: it does not touch Q/V/N values, does not combine across blocks, does
not finalize f2, picks no accumulation order. All cancellation-prone math (the per-SNP difference +
long-double pairwise sum), the native-FP64-vs-Ozaki choice, the fixed-order host-side combine, and
determinism live downstream — confirmed in `cpu_backend.cpp:157–161` (long-double `pairwise_sum` →
`finalize_f2`) and `f2_estimator.hpp` (`assemble_f2_numerator` held native FP64). The orchestration
forwards `Precision` **verbatim**, which is exactly right (§12: the knob is an operation mode for the
matmul GEMMs, not a storage type; the numerator stays native FP64 regardless — `fstats.hpp:17–19`
re-states this). By forwarding the **same** `precision` object to both `compute_f2` and
`compute_f2_blocks`, it preserves the property the equivalence test relies on (single-block
`compute_f2_blocks` ≈ M0 `compute_f2`, test:248–268). §12 is **not violated**; nothing to fix.

### (4) CUDA idioms / RAII / streams / launch config vs §7

**N/A — by design, correctly so.** This is a CUDA-free `core` unit; it owns no device memory, launches no
kernel, touches no stream/event/graph, picks no launch geometry. All of §7 lives behind the seam in
`steppe_device`. The unit's *entire CUDA contribution* is to **not** include a CUDA header — which it
achieves and CMake enforces (`steppe::device PRIVATE`). The one §7-adjacent point: it passes `MatView` (a
span-like non-owning view) and a raw `(const int*, int)` pair rather than a `std::span<const int>` across
the seam, justified in the docstring as keeping the seam "free of any std-container ABI dependency"
(.cpp:48–49). Defensible for vtable ABI stability — but see F-9 (a `std::span` is *also* ABI-stable /
CUDA-free and would carry the length). No defect in this category.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**Clean — verified.** There is not a single numeric literal in either file. Block sizes, precision bits,
launch geometry, the pad base all live in `config.hpp` as named constants (`kDefaultBlockSizeCm`,
`kBlockGroupPadBase`, `kCdivBlock`, `kCentimorgansPerMorgan`, `kDefaultMantissaBits` — confirmed present,
config.hpp:44,51,82,94,98). The unit consumes `Precision` and `BlockPartition` as typed objects. This is
exactly the ROADMAP §4 "no literal may survive except true mathematical constants" target. Nothing to fix.

### (6) Decomposition / single-responsibility / function size vs §2

**Mostly good; small notes.**
- Both functions are single-responsibility (one dispatch each), well under any size threshold.
  Policy-in-`core` / implementation-in-backend separation is textbook §2.
- **F-7 [LOW] — `compute_f2_block` is a 1-line pure forwarder with no value-add today.** It takes
  `(backend, Q, V, N, precision)` and returns `backend.compute_f2(Q, V, N, precision)` verbatim — identical
  argument list and return type; a caller could call `backend.compute_f2(...)` directly with zero loss.
  This is *acceptable* as a deliberate composition-root seam **iff** it is where the precondition validation
  (F-2/F-3) and any future cross-cutting concern (capability tagging F-13, logging, metrics) will live. As
  written it earns its existence only as a naming/architectural anchor. The fix is not to delete it but to
  give it the responsibility its docstring claims ("the orchestration owns the policy") via the F-2/F-3
  guard. Resolves when F-2/F-3 land.
  Severity: low. Effort: trivial. Before M4.5: no.
- **F-8 [LOW] — the two functions share the same Q/V/N/P/M validation need** (F-2, F-3). Put it in one
  `static` helper (`validate_qvn(Q, V, N)`) in the `.cpp` so the M0 and M4 entry points cannot diverge —
  §8 DRY single-home applied locally. `compute_f2_blocks` additionally needs the partition-consistency
  guard (F-1), which can be a second small `validate_partition(partition, Q.M)` helper.
  Severity: low. Effort: S. Before M4.5: no.

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comments

**F-9 [LOW] — the `(const int*, int)` block-id pair is weaker-typed than the CUDA-free, ABI-stable
`std::span<const int>`.** The header justifies the raw pointer as keeping the seam "free of any
std-container ABI dependency" (.cpp:48–49), but `std::span<const int>` is *not* a container — it is a
trivially-copyable `{pointer, size}` view, is CUDA-free, and is **already used elsewhere in `core`**
(`assign_blocks` takes `std::span<const int> chrom`, block_partition_rule.hpp:125). A span would (a) make
the length explicit at the seam so the backend could cross-check `M` (structurally mitigating F-1 without a
separate length param) and (b) remove the null-`data()` foot-gun (a span over an empty vector still carries
`size()==0` explicitly). The counter-argument — ABI stability of the *virtual* vtable, the one place
layout churn is expensive — has real weight, and the team may want a C-pointer ABI there deliberately. So
this is a *judgment call*, LOW. At minimum the docstring should acknowledge `std::span` and state *why* the
raw pointer is preferred (vtable ABI), rather than implying spans are containers. Effort to switch is M
(touches the virtual signature in backend.hpp:167 and both backend definitions).
Severity: low. Effort: M. Before M4.5: no.

**F-11 [LOW] — stale file-header comment (doc-rot) + in-body paraphrase of the docstrings.** The `.cpp`
file header still describes the per-block tensor as a "later milestone" — .cpp:16–19: *"The per-block →
[P × P × n_block] f2_blocks tensor assembly … and the SNP→block partition are later milestones (ROADMAP
M3/M4); this is the host-side composition root…"* — yet `compute_f2_blocks` (the M4 deliverable) is
**implemented right there at lines 41–52**. A reader trusting the header would believe M4 isn't here. That
is a concrete doc-rot bug. Separately, the in-body comments (.cpp:33–37 and 44–49) largely re-paraphrase
the per-function header docstrings, so the contract has two homes to keep in sync.
Why it matters: §5 cross-cutting standards / §2 (comments must track code; stale "later milestone" notes
mislead). Fix: rewrite the "M0 scope" paragraph to say both M0 (`compute_f2_block`) and M4
(`compute_f2_blocks`) entry points now live here, and trim the body comments to a one-liner each, letting
the header docstrings carry the detail.
Severity: low. Effort: S. Before M4.5: no.

**Completeness audit (not defects, recorded so they are not re-raised):**
- *Neither function is `noexcept`, and that is correct.* They return `std::vector`-bearing structs by value
  and call a virtual that allocates, so they can throw `std::bad_alloc`; `noexcept` would force
  `std::terminate` on OOM, defeating the `DeviceOom` recoverable status (error.hpp:25–27). Correctly
  omitted — and consistent with the rest of `core`, whose `noexcept` members are all the leaf scalar math
  (`element`, `block_of`, `het_correction`, …), never allocating orchestration.
- *Both functions are `[[nodiscard]]`* (header :26, :38); the forward consumes the backend's own
  `[[nodiscard]]` virtual result as the return value, so discard-protection propagates correctly. The unit
  adds no discard-protection of its own beyond propagation — fine.
- *`Precision`/`MatView` pass by `const&`*, matching the backend signatures (backend.hpp:138,169) — no
  copy, correct const-correctness.

### (8) Performance

**N/A — verified.** Each function is a single virtual call plus a by-value return of a backend-constructed
struct. `MatView`/`Precision` pass by `const&` (no copy). The only cost is one indirect call (negligible vs
the GEMMs behind it) and NRVO/move of the returned `F2Result` / `F2BlockTensor` (the vectors move out, not
copy). Returning by value is correct and modern. No change. The proposed F-1/F-2/F-3 guards add at most an
O(M) debug-only scan (`max(block_id)`, non-decreasing check); gate it behind `assert`/`NDEBUG` so release
builds pay nothing on the hot path.

### (9) Layering / API / ABI vs §4

**Excellent — the central thing this unit exists to get right, verified line by line.**
- `.cpp` directly includes only `backend.hpp` (CUDA-free seam), `views.hpp`, `config.hpp`, own header. No
  CUDA, no cuBLAS, no `io`. Confirmed §4 dependency direction.
- CMake (`src/core/CMakeLists.txt:47–59`) links `steppe::device PRIVATE` and sets
  `CUDA_RESOLVE_DEVICE_SYMBOLS OFF`, so `core` physically cannot leak a CUDA dependency and the M4
  equivalence test (which links both `steppe_core` and `steppe_device`) won't double-resolve RDC symbols.
  Correct, and the CMake comment (:51–58) documents exactly why.
- Return types (`F2Result`, `F2BlockTensor`) are plain-FP64 host structs that cross the CUDA-free seam
  (backend.hpp:39–52, fstats.hpp:40–71). No CUDA type escapes.
- **F-17 [LOW] — IWYU inconsistency: the `.cpp` uses `BlockPartition` and `F2BlockTensor` directly but
  includes neither header directly, while it *does* re-include the 3 it already gets transitively.** The
  `.cpp` re-includes `device/backend.hpp`, `core/internal/views.hpp`, `steppe/config.hpp` (all also pulled
  by its own header) — that part is *correct* IWYU ("include what you use," don't lean on transitive). But
  `compute_f2_blocks` directly names `BlockPartition` (parameter type at .cpp:42, and member access
  `partition.block_id`/`partition.n_block` at :50–51) and returns `F2BlockTensor` (.cpp:41), yet the `.cpp`
  includes **neither** `core/domain/block_partition_rule.hpp` nor `steppe/fstats.hpp` directly — it relies
  on transitive inclusion via its own header for exactly the two types it uses most concretely. That is the
  inverse of the rule it follows for the other three includes. IWYU is a real gate (architecture.md §791
  "IWYU starts non-blocking, promoted once clean, scoped to first-party targets"; §793 pre-commit), so this
  will flag when IWYU is promoted. Fix: add `#include "core/domain/block_partition_rule.hpp"` and
  `#include "steppe/fstats.hpp"` to the `.cpp` (or, defensibly, drop the 3 redundant re-includes and rely on
  the header uniformly — but the former matches the file's existing "include what you use" stance).
  Severity: low. Effort: trivial. Before M4.5: no.
- **API-surface asymmetry (LOW, informational):** `F2Result` (the M0 result) is declared in
  `device/backend.hpp` (an internal header) while `F2BlockTensor` (the M4 result) is in the *public*
  `include/steppe/fstats.hpp`. `compute_f2_block` returns the internal type; `compute_f2_blocks` returns the
  public one. This is a deliberate milestone artifact (fstats.hpp:1–8 calls itself "the M4 deliverable"),
  not a layering violation, but it is a mild inconsistency worth a one-line comment (why is the M0
  `F2Result` not also a public handle?).
- **F-15 [LOW] — `int n_block` (passed) vs `block_id.size()` (a `size_t` ≤ M, where M is `long`)
  width/range asymmetry at the seam.** `BlockPartition::n_block` is `int` (block_partition_rule.hpp:88,
  justified: block counts are O(1e3)) and is forwarded as `int` (backend.hpp:168) — consistent, no
  truncation *for legitimate data*. But the seam now mixes a `long M` (views) with an `int n_block`
  (backend) and `int`-element `block_id`, and the orchestration is the place that could assert the
  relationship `n_block <= M`. It does not. Same unvalidated-contract family as F-1; recorded separately
  because it is specifically a *type-width* mismatch at the API boundary. Subsumed by the F-1 guard if that
  guard also checks `n_block <= Q.M`.
  Severity: low. Effort: S. Before M4.5: no (folds into F-1).
No fix required for the core layering claim — it is the unit's strongest property.

### (10) Testability vs §13

**F-12 [MED] — this unit is only ever tested transitively, never in isolation; the fail-fast paths
(F-1..F-3) have zero coverage, and a GPU-free test of it is trivially possible because the precedent
already exists in-tree.** The sole caller is `tests/reference/test_f2_blocks_equivalence.cu`
(grep-confirmed: the only references to `compute_f2_block(s)` outside the unit, the backends, and the
headers are in that test). It exercises the *happy path* with real AADR against both backends and needs a
real GPU (`make_cuda_backend()`, test:219). There is:
- no test that this unit/the backend **rejects** an inconsistent `n_block`/`block_id` (F-1),
- no test of the empty-partition / `M==0` path through *this function* (F-4),
- no test with deliberately malformed Q/V/N (F-2, F-3),
- no `core`-only (CUDA-free) unit test that links *just* `steppe_core` + a tiny stub `ComputeBackend` to
  prove the dispatch forwards `block_id.data()`, `n_block`, and `precision` **verbatim**.
The whole point of the seam (§8, §13: "the entire pipeline is exercisable GPU-free") is undermined if the
only test of this unit needs a real GPU and real data. **And the pattern is already established:**
`tests/unit/test_f2.cpp` (links `steppe::core_internal` only, CUDA-free, dual gtest/self-checking-main
harness), `test_block_partition.cpp`, `test_filters.cpp` all run with no GPU (tests/CMakeLists.txt). A
header-only recording-mock `ComputeBackend` that captures its arguments — linked against `steppe_core`
with no GPU and no `steppe_device` — would let you assert verbatim forwarding *and* be the natural home for
the F-1..F-3 rejection tests. (Note: `make_cpu_backend()` lives in `steppe_device`, so a real-backend test
would still drag the device target's link; the *mock* is the genuinely GPU-free path, which is what the
seam was designed to enable.)
Why it matters: §13 (the pipeline must be exercisable GPU-free; the seam must be testable without a
device), §2 (testability is first-class). The unit is *designed* to be the easiest thing in the tree to
unit-test and currently isn't.
Fix: add a `tests/unit/test_f2_from_blocks.cpp` host-only target (mirroring `test_f2.cpp`'s dual harness)
linking `steppe_core` against a recording-mock `ComputeBackend` (~few dozen lines): (a) verbatim argument
forwarding for both functions, (b) the new F-1/F-2 precondition rejections fire, (c) the `M==0`/
empty-partition path.
Severity: med. Effort: M. Before M4.5: no (cheap insurance once F-1..F-3 add the guards it tests).

**F-14 [MED] — the F-1 OOB write means the §6 "compute-sanitizer (memcheck) clean" definition-of-done is
met only *by luck of the test inputs*, with no negative test to catch a regression.** ROADMAP §6:131 makes
"compute-sanitizer (memcheck+racecheck) clean" a milestone gate. The current equivalence test only ever
passes a *consistent* `assign_blocks` partition (and the hand-built `one` partition, which happens to be
consistent), so memcheck passes — but it passes because no input ever exercises the F-1 OOB path, not
because the path is safe. If a future caller (merged-dataset `n_block`, a mutated partition) trips it,
memcheck won't have been the guard because no test feeds it. Distinct from F-12: even *with* the F-1 guard
added, you need a memcheck-run negative test (an inconsistent partition the guard must reject *before* the
backend writes) to keep the §6 gate honest.
Why it matters: §6 definition-of-done; §2 fail-fast.
Fix: once F-1's guard lands, add a memcheck-gated test asserting the guard fires on an inconsistent
partition (and, in a build with the guard compiled out, that compute-sanitizer flags the write — proving
the guard's necessity). Rides on F-1 + F-12.
Severity: med. Effort: S. Before M4.5: no (with F-12).

### (11) Capability tiers (PRO-6000-capable vs budget-5090)

**F-13 [LOW–MED] — this is the natural, currently-empty home for the cross-cutting capability tag and the
M4.5 P2P device-combine *policy*.** TODO.md's capability-tier table (the "Keeping it GPU-dominant"
section, :125–137) makes two demands that touch *this* layer, and both are grounded in the docs:
1. *"cross-cutting — a capability probe + capability-tagged results (every run records which path it took +
   why it degraded), slotting into `DeviceConfig`/`Resources`."* (TODO.md:137.) This unit is the per-stage
   composition root that sees the `Precision` and the backend; it is the obvious place to thread/emit a
   capability tag so the f2 stage's degradation is recorded in the run's provenance. Today the unit takes no
   `Resources`/`DeviceConfig` and forwards nothing of the sort. Per TODO's parity rule (:133) this is
   *parity-neutral* (observability only): adding a tag must not change a single emitted number, and the
   AT2-golden + native-FP64 oracle gate both boxes.
2. *"M4.5 — add the optional `canAccessPeer`-gated P2P device-combine (host-staged stays the baseline)."*
   (TODO.md:137; the combine row at :128.) Capable path = GPU0 pulls peer partials via `cudaMemcpyPeer`
   (byte-exact DMA), sums fixed `g=0..G-1` on-device; budget fallback = host-staged fixed-order combine,
   with the logged reason *"P2P combine unavailable (no peer access) → host-staged fixed-order combine."*
   Crucially, architecture.md §11.4:711 establishes the combine is a **policy** decision (the parity
   reduction is *host-side, fixed device order*; the P2P path is an opt-in `cudaDeviceCanAccessPeer`-gated
   fast-path that is bit-identical) — and policy belongs in `core` orchestration, not the backend. As
   written, `compute_f2_blocks` has no multi-GPU surface (single `ComputeBackend&`), so the combine policy
   has nowhere to live yet.
What to add (when M4.5 lands, not before): thread a `const Resources&` (or a small capability-probe result)
into the orchestration; emit one explicitly-tagged, parity-neutral log line per f2-stage run matching
TODO's exact tagged-degradation strings; host the `canAccessPeer`-gated combine policy here (baseline =
host-staged fixed-order).
Why it matters: TODO.md:125–137 capability-tier table; architecture.md §11.4 (multi-GPU fixed-order
combine), §9 (injected resources). This unit is the designated home of the f2-stage policy.
Severity: low–med (*future* work, not a current defect; raised because the mandate asks where THIS unit
touches the tiers). Effort: M. Before M4.5: this *is* the M4.5 work — **yes, as part of M4.5**, not before.

## Considered & rejected

- **Prior pass's F-2 as an independent HIGH ("non-empty Q with empty partition → null-`data()` deref").**
  **Rejected as a false positive *as framed*.** An empty partition has `n_block == 0`
  (block_partition_rule.cpp:25–27), and *both* backends early-out on `if (P<=0 || M<=0 || n_block<=0)
  return out;` (cpu:207, cuda:130) **before** any `block_id` dereference. So "empty partition" never
  reaches the deref; the null-`data()` UB requires `n_block > 0` *while* `block_id` is empty/short — which
  is precisely the F-1 inconsistency. The genuine null-pointer mechanism is preserved as a sub-case *inside*
  F-1 (where it belongs), and the spurious independent HIGH is removed. (The `nullptr`-on-empty `data()`
  claim itself is correct and verified against the C++ draft; only the *trigger scenario* was wrong.)
- **The original auditor's F-1 title "`compute_f2_blocks` discards `partition.n_block`."** Rejected as
  factually wrong (the prior pass already corrected this): `n_block` *is* forwarded at .cpp:51 as the 5th
  argument. Re-confirmed this pass against backend.hpp:168.
- **"`compute_f2_block` should be removed as dead M0 cruft now that M4 exists."** Rejected: the equivalence
  test still calls it (test:253) as the single-block consistency anchor (n_block=1 must equal M0), and
  architecture.md §5 S2 keeps the single-block entry point as the M0 contract. The fix (F-7) is to give it
  validation, not delete it.
- **"Forward `Precision` by value — it's tiny (an enum + an int)."** Rejected: `const&` matches the backend
  signature (backend.hpp:138,169); a copy gains nothing and diverging from the interface's parameter style
  is a readability regression.
- **"Mark the functions `noexcept`."** Rejected: they allocate via the virtual and must surface
  `std::bad_alloc`/`DeviceOom`, not `std::terminate` (error.hpp:25–27).
- **"The unit should itself loop over blocks / do the per-block batching."** Rejected: that is explicitly
  the *backend's* job (cuda_backend.cu:113–284 grouped strided-batched; cpu_backend.cpp:190–276 per-block
  oracle). `core` owning the block *policy* while the backend owns the *implementation* is precisely
  §2/§5/§8 — correctly realized.
- **"The redundant re-includes in the `.cpp` are a defect (3 headers already come via the own header)."**
  Rejected: re-including headers you directly use is *correct* IWYU, not a smell. The genuine inconsistency
  is the *opposite* direction — two directly-used types that are NOT directly included (F-17).
- **"`std::vector` return is a perf problem; return into a caller buffer."** Rejected: NRVO+move makes it
  cheap; the vectors are backend-allocated regardless; a caller-buffer API would leak allocation policy
  across the seam; cost is dwarfed by the GEMMs (Performance, N/A).
- **"The orchestration should validate `Precision::mantissa_bits`."** Rejected: that is
  `ConfigBuilder::build()`'s fail-fast job (error.hpp:38–40 `InvalidConfig`, §9), not the per-call
  dispatch's. Forwarding the already-validated typed config is correct.
- **"`block_of`/`assign_blocks` rounding for negative positions could mis-bin and the orchestration should
  guard."** Rejected: that is the domain rule's contract (block_partition_rule.cpp:46, negative bins handled
  deliberately) and out of this unit's scope; this unit only consumes the already-computed `BlockPartition`.
- **"`int n_block` could overflow / truncate from a `long` block count."** Rejected as a real bug: block
  counts are O(1e3) on a whole genome (block_partition_rule.hpp:88), and `assign_blocks` produces `int` ids
  natively (block_partition_rule.cpp:56); there is no `long`→`int` narrowing in the path. The residual
  *type-asymmetry* concern is kept as the LOW F-15.
- **"Pass `Q.M` to the backend so it can derive block ranges without scanning."** Rejected: the backend
  already has `Q.M` (cpu:197 / cuda:120) and scans `block_id[0..M)` itself; passing M redundantly would not
  help and the scan is required regardless. (The useful change is passing the *block_id length* / a span —
  F-9 — not M.)

## What it takes to reach 10/10

1. **(F-1, F-2, F-3, F-15) Add a fail-fast precondition guard at the seam** in one shared `validate_qvn(Q,
   V, N)` + a `validate_partition(partition, Q.M)` helper in the `.cpp`: check `Q.P==V.P==N.P`,
   `Q.M==V.M==N.M`, `P>=0`, `M>=0`, `partition.block_id.size()==static_cast<size_t>(Q.M)`, and (debug, O(M))
   `n_block>0` when non-empty, `*max_element(block_id) < n_block`, `n_block <= Q.M`, ids non-decreasing.
   There is **no `STEPPE_EXPECTS` in the tree yet** (grep-confirmed) — either introduce `<cassert>`
   (CUDA-free, NDEBUG-gated so release pays nothing) or make this the forcing function to add the
   file/line-carrying `STEPPE_EXPECTS` the §2 fail-fast principle implies. This closes the live
   inconsistent-`n_block`/`block_id` OOB write (incl. the null-`data()` sub-case) and makes the
   "orchestration owns the policy" docstring true. *(highest priority — a real, shipped robustness bug.)*
2. **(F-12, F-14) Add a CUDA-free unit test** `tests/unit/test_f2_from_blocks.cpp` (mirror `test_f2.cpp`'s
   dual gtest/self-checking harness) linking `steppe_core` against a recording-mock `ComputeBackend`: assert
   verbatim argument forwarding for both functions; assert the new precondition rejections fire; cover the
   `M==0`/empty-partition path; and add a memcheck-gated negative test proving the guard protects the
   backend OOB write. This is the §13 GPU-free-testability the seam was designed for and the §6
   sanitizer-gate honesty the unit currently lacks. The pattern is already in-tree (`test_f2_estimator`,
   `test_block_partition`).
3. **(F-11) De-rot the comments**: fix the stale "M4 is a later milestone" file header (M4 is implemented
   here now) and trim the in-body paraphrase so the contract has one home.
4. **(F-17) Fix the IWYU inconsistency**: include `core/domain/block_partition_rule.hpp` and
   `steppe/fstats.hpp` directly (the two types the `.cpp` uses but does not include), matching the
   "include what you use" stance it already takes for the other three.
5. **(F-9) Resolve the `const int*` vs `std::span<const int>` decision explicitly**: either switch the seam
   to `std::span<const int>` (type-safe, carries length → structurally mitigates F-1, still ABI-stable +
   CUDA-free, already used by `assign_blocks`) or document *why* the raw pointer is kept (vtable ABI)
   instead of implying spans are containers.
6. **(F-13) When M4.5 lands**, thread a `Resources`/capability surface through the orchestration, emit one
   explicitly-tagged, parity-neutral capability log line per f2-stage run (matching TODO's exact degradation
   strings, :128), and host the `canAccessPeer`-gated combine *policy* here (baseline = host-staged
   fixed-order, architecture.md §11.4). Not before M4.5, but this unit is its designated home.

Items 1–2 are the gate from 8 → ~9.5; 3–5 finish the polish to 10; 6 is M4.5-scoped and keeps it at 10 as
the capability story arrives.

## Good patterns to keep

- **Layering discipline is exemplary** — CUDA-free includes only, `steppe::device PRIVATE`,
  `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` with a CMake comment explaining the double-resolve hazard it avoids. The
  single most important property of a `core` unit, fully correct (§4).
- **Pure dependency-injection dispatch** — the same orchestration runs unchanged against `CpuBackend` and
  `CudaBackend`; no GPU-vs-CPU branching (§8). The unit *is* the proof the seam works.
- **Policy/implementation split** — `core` owns the `BlockPartition` and `Precision` policy; the backend
  owns the batching/GEMM implementation. Textbook §2/§5/§8.
- **Zero magic numbers** — everything is a typed config object or a named `config.hpp` constant (ROADMAP §4
  target met).
- **`[[nodiscard]]` on both functions; `const&` views; by-value FP64 result structs that cross the
  CUDA-free seam.** Correct const-correctness and ABI hygiene; `noexcept` correctly *omitted* and the
  forward consumes the backend's own `[[nodiscard]]` virtuals so discard-protection propagates.
- **Re-including directly-used headers (3 of them)** rather than leaning on transitive inclusion is correct
  IWYU practice — keep it; just extend it to the two types currently relying on transitive (F-17).
- **The header rationale is genuinely high quality** — it cites exact architecture sections and explains the
  raw-pointer ABI choice. Keep that standard; just keep it *in sync* (F-11) and have it acknowledge
  `std::span` (F-9).
