# src__device__vram_budget
Files: /home/suzunik/steppe/src/device/vram_budget.hpp
Subsystem: backend

## Findings

### G6
- [G6.naming][LOW] src/device/vram_budget.hpp:176,179 — The local that holds the budget-derived quotient is named `fit_blocks` ("blocks that fit"), and the doc comment at line 159-160 also uses `fit_blocks` ("Capping `fit_blocks` at `kMaxGridZ`"). In this `qpadm`-heavy codebase "fit" is a heavily overloaded term (the Phase-2 qpAdm *fit* engine, on this very branch). A reader scanning for fit-engine state could misread it; the value is purely "how many jackknife blocks fit in one chunk", unrelated to model fitting. Suggested: rename to `blocks_that_fit` / `fitting_blocks` (or `budget_blocks`) and update the matching doc line to remove the collision with the fit-engine vocabulary.

### G8
- [G8.comment][LOW] src/device/vram_budget.hpp:71,88,118 — Comments name the second resident tensor inconsistently: "f2+Vpair" / "vpair" / "f2 AND vpair" (lowercase v) at 71/88 vs. the capitalized "Vpair"/"Vpairg" used at 71 (header) and in `per_block_chunk_elems`/config (line 95 uses `Vpairg`). The mixed casing of the same logical tensor (`vpair` vs `Vpair`) is minor prose drift within one file. Suggested: pick one spelling (`vpair`) for the tensor throughout the comments.

### G9
- [G9.const][LOW] src/device/vram_budget.hpp:67-69,132-137,169-186 — `budget_bytes`, `chunk_budget_bytes`, and `max_blocks_per_chunk` are `inline` but NOT `constexpr`, while every other helper in the file (`nonneg`, `resident_tensor_bytes`, `per_block_chunk_elems`, `per_block_chunk_bytes`) is `constexpr`. The only thing blocking constexpr is the `double` multiply in `budget_bytes` (line 68), which IS a constant expression in C++20; the size_t arithmetic in the other two is trivially constexpr-able once `budget_bytes` is. Making them `constexpr` would let the host unit tests assert budget results at compile time and keep the helper family uniform. Note: this is a cleanliness/consistency observation, not a bug. Suggested: mark all three `constexpr` (the `double` math is constexpr-legal in C++20).

No further issues. Notes verified clean:
- G4 (numeric/scale): all products (`resident_tensor_bytes` 2·P²·n_block·8, `per_block_chunk_elems` 4·P·s_pad + 4·P²) are widened to `std::size_t` via `nonneg()` BEFORE multiplying (lines 86-89, 105-108), so the documented P~2500 / n_block~757 overflow-before-widening hazard is correctly avoided. The size_t→int narrowing in `max_blocks_per_chunk` is bounded by `min(..., kMaxGridZ=65535, nb_total)` in size_t before the cast (lines 179-185), so the cast cannot wrap; `int(65535)` is safe. `budget_bytes` truncates a non-negative `double` product to size_t — no negative-wrap path (free is size_t, fraction is asserted (0,1]).
- G3: no dead/unused code; all four includes are used (`<algorithm>`→`std::min/std::max`, `<cstddef>`→`std::size_t`, launch_config→`kMaxGridZ`, config→the five k-constants); no unread computed values.
- G5: every literal is a named config constant (`kResidentTensorCount`, `kChunkInputStacks`, `kChunkOutputStacks`, `kMaxVramUtilizationFraction`, `kCublasWorkspaceBytes`, `kMaxGridZ`); the duplicated "4" coefficients and the 2× resident factor are explicitly single-sourced in config.hpp (verified: kChunkInputStacks=4, kChunkOutputStacks=4, kResidentTensorCount=2, kMaxGridZ=65535) — no drift.
- G10: every local is initialized at declaration (lines 86-87, 105-107, 134-135, 172-184); no declare-then-assign or zero-init assumptions.
- G2 / G7 not applicable beyond the above (no CUDA APIs — CUDA-FREE header by design; no copy-pasted blocks — the clamp-then-widen and floor-fraction idioms are already extracted into `nonneg`/`budget_bytes`).
