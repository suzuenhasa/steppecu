# Review findings — src__device__cuda__device_partial

Files: /home/suzunik/steppe/src/device/cuda/device_partial.cu, /home/suzunik/steppe/src/device/cuda/device_partial_impl.cuh, /home/suzunik/steppe/src/device/device_partial.hpp

## Group 4 — Type & numeric

- [4.2][LOW] src/device/device_partial.hpp:39,40 — The shape contract this handle carries (`int P`, `int n_block_local`) describes a `[P*P*n_block_local]` resident buffer (impl comment, device_partial_impl.cuh:17-18). No index arithmetic or allocation happens in this unit, so there is no overflow here; but the `int` shape fields are the *width source* for the consumers (cuda_backend.cu/p2p_combine.cu) that do compute `P*P*n_block_local`, where the product exceeds 2^31 at scale (P≈2500, n_block≈757). Suggested: leave as-is for this unit (correctly delegates to DeviceBuffer); ensure the consuming TUs widen to size_t/int64_t before the `P*P*n_block_local` product — flag belongs to those units, not this one.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/device_partial.hpp:8 — `#include <cstddef>` is unused: the header declares only `int`/`bool` scalars, `std::vector<int>` (`<vector>`), and `std::unique_ptr<Impl>` (`<memory>`); no `std::size_t`/`ptrdiff_t`/`offsetof`/`std::byte` etc. appears (grep over the file is empty). Harmless, but it's a stray include. Suggested: drop `#include <cstddef>` (or, if kept deliberately as a self-sufficiency safety net, leave it — pure hygiene, no correctness impact).

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/device/device_partial.hpp:42 — `int device_id = -1` is an unnamed sentinel literal for "no device / unset CUDA ordinal". The intent is clear from the member name + comment ("the physical CUDA ordinal"), and `-1` is the conventional invalid-ordinal value, so this is not a correctness risk; it is only a hygiene note that the same sentinel recurs across the CUDA seam. Suggested: optionally hoist a shared `constexpr int kInvalidDeviceId = -1` (or reuse an existing project constant) so the empty/moved-from sentinel is named in one place rather than spelled `-1` inline — pure hygiene, no behavior change.

## Group 6 — Naming

No Group 6 issues found. The unit's names are domain-canonical or self-documenting: `P` (device_partial.hpp:39) is the project/AT2-canonical population-count symbol, not a cryptic single letter; `f2`/`vpair` (device_partial_impl.cuh:17-18) are the established f2-statistic / variance-pair domain terms matching the header's "f2/Vpair" comment; `n_block_local`, `b0`, `device_id`, `block_sizes`, `impl`, `empty()` are accurate and each documented inline. No misleading names (6.2), the `P` (uppercase domain symbol) + snake_case scalars mix is an intentional domain convention rather than an in-file inconsistency (6.3), and no nonstandard abbreviations beyond documented domain terms (6.4). device_partial.cu defines only out-of-line special members with no local identifiers.

## Group 7 — Duplication

No Group 7 issues found. The unit is declaration-only and intentionally minimal: device_partial.cu:10-13 is the four-line out-of-line pimpl special-member idiom (`= default` ctor/dtor/move-ctor/move-assign), which exists *because* the CUDA-free header cannot default them with an incomplete `Impl` — it is the minimum boilerplate the pimpl pattern requires, not collapsible duplication (a macro would only obscure it) (7.4). The two `DeviceBuffer<double>` members `f2`/`vpair` (device_partial_impl.cuh:17-18) are distinct domain tensors (the f2 statistic and its variance pair), a legitimate two-field aggregate, not a copy-pasted block differing by a constant (7.1). There are no loops, expressions, or computed values anywhere in the unit, so there is nothing loop-invariant to hoist (7.2) and no repeated `sizeof`/cast to fold (7.3) — the `[P*P*n_block_local]` shape appears only in twin comments, not as live arithmetic.

## Group 8 — Comments

- [8.2][LOW] src/device/device_partial.hpp:16-18 — Stale comment: the doc comment says "the DeviceBuffer<double> f2/vpair owners + the resident device pointers live in `Impl`, defined only in cuda/device_partial.cu", but `struct DevicePartial::Impl` is actually defined in cuda/device_partial_impl.cuh:16-19; device_partial.cu only includes that .cuh and defines the out-of-line special members. The sibling reference at device_partial.hpp:53 already correctly points to `cuda/device_partial_impl.cuh`, so this comment is internally inconsistent and predates the .cuh split. Suggested: change "defined only in cuda/device_partial.cu" to "defined in cuda/device_partial_impl.cuh" to match line 53 and the actual location.

## Group 9 — Constants & configuration

No Group 9 issues found. This is a declaration-only unit with no tunable knobs and no executable logic. 9.1: the public shape fields (device_partial.hpp:39-46 `P`, `n_block_local`, `b0`, `device_id`, `block_sizes`) are intentionally mutable — DevicePartial is a move-only data-carrier handle populated incrementally after construction by the producing worker and read by the combine, so they cannot be `const`/`constexpr`; their sentinel defaults are in-class initializers (the `-1`/`0` literals are already covered as a Group 5 hygiene note). 9.2: no configuration knobs are buried in logic anywhere in the three files — there is no logic, only the pimpl special-member declarations (device_partial.cu:10-13) and the two-field Impl aggregate (device_partial_impl.cuh:16-19). 9.3: no positional-boolean call sites exist; device_partial.cu defines only `= default` special members and `empty()` (device_partial.hpp:50) takes no arguments.

## Group 10 — Initialization

No Group 10 issues found. 10.1: there are no local variables anywhere in the unit — device_partial.cu:10-13 is only `= default` special members, and device_partial_impl.cuh / device_partial.hpp are declaration-only, so there is no "declared far from first use" or "uninitialized-then-assigned" pattern to flag. 10.2: every scalar data member carries an explicit in-class initializer (device_partial.hpp:39-42: `int P = 0`, `int n_block_local = 0`, `int b0 = 0`, `int device_id = -1`), so none of them rely on implicit zero-init; the class-type members (`std::vector<int> block_sizes` at device_partial.hpp:46, `std::unique_ptr<Impl> impl` at device_partial.hpp:54, and the two `DeviceBuffer<double>` f2/vpair at device_partial_impl.cuh:17-18) are default-constructed to their well-defined empty/null states by their own default ctors, not by any zero-init assumption. The DeviceBuffer default-ctor semantics live outside this unit; from this unit's vantage the members are correctly default-constructed.
