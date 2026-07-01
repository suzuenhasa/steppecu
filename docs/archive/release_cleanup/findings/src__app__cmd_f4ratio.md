# src__app__cmd_f4ratio
Files: /home/suzunik/steppe/src/app/cmd_f4ratio.cpp, /home/suzunik/steppe/src/app/cmd_f4ratio.hpp
Subsystem: app-cli

## Findings

### G5
- [G5.src__app__cmd_f4ratio][LOW] cmd_f4ratio.cpp:138,87,92,95-96 — the tuple arity `5` is an unnamed literal repeated across the resolve loop (`c < 5`), the `--pops` shape check (`% 5`, `/ 5`) and the group-of-5 indexing (`5 * k + 0..4`). It is the intrinsic arity of an f4-ratio 5-tuple and is documented in the header, so drift risk is low, but a single named `constexpr std::size_t kTupleArity = 5;` would make the relationship between the `std::array<…, 5>` type, the modulus, and the unrolled column loop explicit. Suggested: introduce one named constant and reference it in the loop bound / modulus / divisor.

### G6
- [G6.src__app__cmd_f4ratio][LOW] cmd_f4ratio.cpp:132-135 — the resolved-label vectors `l1`..`l5` are terse positional names; they are correct (mirroring pop1..pop5) but `l4` vs `l1` is easy to transpose at the `push_back` / `emit` sites (lines 147-151, 190/198). The emitter parameter names (`p1_labels`..`p5_labels`) are clearer. Suggested: rename to `p1_labels`..`p5_labels` to match the sink and reduce transposition risk.

### G10
- [G10.src__app__cmd_f4ratio][LOW] cmd_f4ratio.cpp:137 — `std::array<int, 5> idx{};` is value-initialized then every element is overwritten in the immediately-following `c < 5` loop (line 144); the zero-init is dead. Harmless (and arguably defensive), not a correctness issue. Suggested: leave as-is or drop the `{}` since all five slots are unconditionally assigned before use.
