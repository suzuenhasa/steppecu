# Review findings — src__device__f2_disk_format

Files: /home/suzunik/steppe/src/device/f2_disk_format.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/f2_disk_format.hpp:16 — `#include <cstddef>` is unused: no `std::size_t`/`std::ptrdiff_t`/`std::byte`/`offsetof` symbol appears in the file (verified by grep), and `sizeof(double)` on lines 42/46 is a built-in operator needing no include. `<cstdint>` (line 17) IS required (uint32_t/int32_t/uint64_t/uint8_t). Suggested: drop the `<cstddef>` include.

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/device/f2_disk_format.hpp:37 — header size `64` is a bare literal in `static_assert(sizeof(F2DiskHeader) == 64, ...)`, and the same value is repeated as prose in the comments on lines 24 ("64-byte fixed header"), 26 ("sizeof == 64"), and the `f2_offset == 64` invariant on line 32. The struct is self-validating via the static_assert, but the value is not a named constant that writers/readers (other files) can reference, so the on-disk header size lives as a duplicated magic literal. Suggested: add `inline constexpr std::size_t kF2DiskHeaderSize = 64;` (or `= sizeof(F2DiskHeader)`) and use it in the static_assert + offset documentation.
- [5.3][LOW] src/device/f2_disk_format.hpp:28 — the format version value `1` exists only as a comment ("// 1") on the `version` field; there is no `kF2DiskVersion` named constant. The writer and the M7 reader must each hardcode `1` independently, which is a drift risk if the format ever bumps. (`dtype` is fine — it has `kF2DiskDtypeFp64`.) Suggested: add `inline constexpr std::uint32_t kF2DiskVersion = 1u;` alongside the other format constants and have both ends reference it.

## Group 6 — Naming

No Group 6 issues found.

## Group 7 — Duplication

- [7.1][LOW] src/device/f2_disk_format.hpp:40-47 — `f2_block_offset` and `vpair_block_offset` are copy-pasted blocks differing only by the base-offset constant (`h.f2_offset` vs `h.vpair_offset`); the `P·P·b·sizeof(double)` per-block stride arithmetic is identical in both. Suggested: extract a private `slab_offset(base, h, b)` helper (or a `block_offset(h.f2_offset, h, b)` style) that both call, passing the base offset.
- [7.3][LOW] src/device/f2_disk_format.hpp:41-42,45-46 — the triple `static_cast<std::uint64_t>(h.P) * static_cast<std::uint64_t>(h.P) * static_cast<std::uint64_t>(b)` widening cast chain is repeated verbatim in both functions (and the widen-before-multiply is the correct overflow guard at P*P*b scale). Suggested: folded automatically once 7.1's shared helper computes the slab stride once.

## Group 8 — Comments

- [8.2][LOW] src/device/f2_disk_format.hpp:12,22 — both comments cite `fstats.hpp:18` for the "Storage is FP64 in EVERY precision mode" claim, but the "STORAGE IS FP64 IN EVERY PRECISION MODE" statement actually begins on line 17 of include/steppe/fstats.hpp (line 18 is the "knob is an OPERATION mode..." continuation). Off-by-one cross-reference. Suggested: update both citations to `fstats.hpp:17` (and prefer the full `include/steppe/fstats.hpp` path, since the file is in include/, not the local src/device/ dir).
- [8.2][LOW] src/device/f2_disk_format.hpp:7 — the header comment cites `fstats.hpp:38-46` as the anchor for the `i + P·j + P·P·b` layout it claims to be byte-identical to, but that exact index formula is stated on line 37 of include/steppe/fstats.hpp; the cited range starts one line after the load-bearing line. Suggested: widen the citation to `fstats.hpp:37-46` (and use the full include/ path).

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.
