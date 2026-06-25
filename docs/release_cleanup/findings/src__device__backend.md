# src__device__backend
Files: /home/suzunik/steppe/src/device/backend.hpp
Subsystem: backend

## Findings

### G6
- [G6.backend][LOW] backend.hpp:951 — In `transpose_to_canonical`, the local byte accumulator is named `packed` (`std::uint8_t packed = 0;`), the same identifier as the destination member `out.packed` (the output vector) referenced a few lines later at 973 (`out.packed[...] = packed;`). The name collision is harmless (different scopes) but reading the inner loop, `packed` as a single byte vs `out.packed` as the byte vector reads as the same concept at two granularities. Suggested: rename the local to `byte` / `out_byte`.

### G8
- [G8.backend][LOW] backend.hpp:964-966 — The `switch (view.encoding)` has only `case TileEncoding::Identity:` plus a `default:` that both fall to `canon = code;`, with the comment "later formats remap here" (line 960). With a single enumerator today this is intentional format-readiness, not dead code, but the `default:` silently maps any future-added-but-unhandled encoding to Identity rather than failing loudly. Suggested: leave as-is for now (documented seam), or drop the `default:` so a newly added enumerator triggers a `-Wswitch` warning at the port site.

No other issues found (groups checked: G2-G10). The unit is a CUDA-free header (POD result/request structs + the abstract `ComputeBackend` interface); G11-G22 do not apply (is_cuda=false). The `(void)param;` casts in the non-pure throwing base bodies are intentional unused-parameter suppression, not dead code (G3). Index/stride widths are scale-safe: `DecodeResult::M`/`dstat_block_reduce` use `long M`, `SweepSurvivors::enumerated` and tile sizes use `std::size_t`, and `RatioJackArray` strides are `long` — no int-index-overflow-before-widening at P~2500/M~584k (G4). Literals (`fudge=1e-4`, `min_z=3.0`, `top_k=1000000`, the `6,4,2,0` shift) are named config fields or derived from `core::kCodesPerByte`/`core::kBitsPerCode`, not hardcoded magic (G5).
