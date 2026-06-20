# Review findings — src__device__cuda__device_buffer

Files: /home/suzunik/steppe/src/device/cuda/device_buffer.cuh

## Group 4 — Type & numeric

- [4.7][LOW] device_buffer.cuh:95-96 — `data()` / `const data()` hand out a raw `T*` that is a *device* pointer with no host-vs-device space distinction, so a caller can pass it to a host-side `memcpy`/deref without a compile error. Suggested: acceptable per steppe design (raw device ptr for cuBLAS/kernel args); a thin `DevicePtr<T>` wrapper would make the wrong space unpassable, but this is hygiene, not a bug.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

No Group 5 issues found.

## Group 6 — Naming

- [6.1][LOW] device_buffer.cuh:78,81 — the move-source parameter is named `o` (single-letter) in both the move-ctor and move-assign signatures; `other` is the conventional and more readable name for a move/copy source, and the file's own header prose already refers to it as `other` (line 12: "`buf = std::move(other)`"). Suggested: rename `o` to `other` for the move ctor/assign params. Note `n`, `e`, `T` are conventional/tightly-scoped (element count, short-lived `cudaError_t`, template param) and fine.
- [6.3][LOW] device_buffer.cuh:12 vs 78,81 — minor inconsistency between doc and code: the header comment spells the move source `other` while the parameter is coded as `o`. Suggested: align them (covered by the 6.1 rename to `other`).

## Group 7 — Duplication

- [7.1][LOW] device_buffer.cuh:78-79,84-85 — the move-source steal `ptr_ = std::exchange(o.ptr_, nullptr); size_ = std::exchange(o.size_, 0);` is repeated verbatim in the move-ctor (init list) and the move-assign body. Suggested: leave as-is — this is the canonical two-member move idiom (one form is a mem-initializer list, the other a post-`reset()` assignment), so a helper would not fold both call shapes cleanly and reduces clarity over the idiom; not worth extracting.

## Group 8 — Comments

No Group 8 issues found.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.
