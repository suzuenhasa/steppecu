# Review findings — src__device__cuda__device_partial

Files: /home/suzunik/steppe/src/device/cuda/device_partial.cu, /home/suzunik/steppe/src/device/cuda/device_partial_impl.cuh, /home/suzunik/steppe/src/device/device_partial.hpp

## Group 4 — Type & numeric

- [4.2][LOW] src/device/device_partial.hpp:39,40 — The shape contract this handle carries (`int P`, `int n_block_local`) describes a `[P*P*n_block_local]` resident buffer (impl comment, device_partial_impl.cuh:17-18). No index arithmetic or allocation happens in this unit, so there is no overflow here; but the `int` shape fields are the *width source* for the consumers (cuda_backend.cu/p2p_combine.cu) that do compute `P*P*n_block_local`, where the product exceeds 2^31 at scale (P≈2500, n_block≈757). Suggested: leave as-is for this unit (correctly delegates to DeviceBuffer); ensure the consuming TUs widen to size_t/int64_t before the `P*P*n_block_local` product — flag belongs to those units, not this one.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/device_partial.hpp:8 — `#include <cstddef>` is unused: the header declares only `int`/`bool` scalars, `std::vector<int>` (`<vector>`), and `std::unique_ptr<Impl>` (`<memory>`); no `std::size_t`/`ptrdiff_t`/`offsetof`/`std::byte` etc. appears (grep over the file is empty). Harmless, but it's a stray include. Suggested: drop `#include <cstddef>` (or, if kept deliberately as a self-sufficiency safety net, leave it — pure hygiene, no correctness impact).
