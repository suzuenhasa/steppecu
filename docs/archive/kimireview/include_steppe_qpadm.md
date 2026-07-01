I read through this carefully. This is **not slop** — it's clearly the work of someone who understands both the genomics domain and C++ API design. A senior developer would find a lot to like, but would also flag a few structural and stylistic concerns.

## What's genuinely good

- **The CUDA-free seam is the right call.** Forward-declaring `device::DeviceF2Blocks` and `device::Resources` and keeping the header strictly standard C++ means this compiles into core, CLI, and bindings without dragging in the CUDA toolkit. That's a deliberate, valuable architectural decision and the comments explain the contract clearly (lines 12–15, 38–40).
- **Named constants and frozen defaults.** `fudge = 1e-4`, `als_iterations = 20`, `rank_alpha = 0.05`, and `JackknifePolicy::All` are all named and documented as AT2 parity constants (lines 48–52, 57–100). No bare magic literals.
- **Value semantics throughout.** `QpAdmModel` and `QpAdmResult` are plain aggregates, vector-friendly, and avoid constructors that could throw. Using `std::vector<int>` for indices and `std::vector<double>` for results keeps ownership simple and explicit (lines 106–163).
- **Per-model `Status` instead of exceptions for domain outcomes.** This is exactly the right design for a search over thousands of models: rank-deficient or non-SPD results are recorded and the caller continues (lines 121–123, 154–157). The comment references `architecture.md §10`, which shows they are following project conventions.
- **`std::span` in the batched/search APIs.** The `run_qpadm_search` and `run_qpwave` overloads take `std::span<const QpAdmModel>` and `std::span<const int>` (lines 192–205, 226–237), which is modern, non-owning, and avoids unnecessary copies.
- **Host-oracle parity overloads.** Providing both device-resident and `F2BlockTensor` overloads of each entry point (lines 170–181, 192–205, 226–237) makes bit-exact CPU reference testing straightforward. The comments explain which path is primary and which is the parity door.
- **Avoiding `vector<bool>`.** Using `std::vector<char>` for `popdrop_feasible` with a 0/1 convention (line 152) is a deliberate, defensible choice given the CUDA-free/value-type goals.

## What a senior developer would flag

**`QpAdmResult` is a kitchen-sink struct.** It accumulates single-model fields, qpWave rank-sweep fields, rankdrop tables, and popdrop tables all in one type (lines 124–163). The comments frame this as "non-breaking" and "appended," which is true, but it also means every `QpAdmResult` carries fields that are irrelevant for most call sites. A senior reviewer would ask whether `QpWaveResult` should really be separate while `QpAdmResult` still embeds the full qpWave table.

**The `-1` sentinel defaults.** `target = -1`, `model_index = -1`, `rank = -1` (lines 61, 69, 109, 118, 162) are pragmatic for a value type, but they are still magic sentinel values. A strongly-typed wrapper like `std::optional<int>` or a dedicated `Index` type would make invalid states unrepresentable. Given the "vector-friendly value" goal, this is a trade-off, not a clear bug.

**Comment density is extreme.** Lines 1–23 are essentially a design doc, and nearly every struct member has a multi-line doxygen block. Most of the comments are genuinely informative (e.g., the `allow_negative_weights` rationale on lines 71–76), but a senior reader will start to wonder whether the code is compensating for unclear naming. The header is as much prose as code.

**Some naming is informal.** `fudge` (line 61) is documented as a ridge constant, but the word itself is colloquial. `JackknifePolicy::FeasibleOnly` is clear; `fudge` less so. In a public header that the comments repeatedly call a "frozen contract," more precise terminology like `ridge` or `ridge_epsilon` would be expected.

**Duplication between `QpAdmResult` and `QpWaveResult`.** The rankdrop fields (`rankdrop_f4rank`, `rankdrop_dof`, `rankdrop_chisq`, etc.) appear in both types (lines 146–147 and 216–217). If these are the same conceptual table, a small shared value type would reduce drift risk.

**Implicit conversion hazard across overloads.** `run_qpadm` is overloaded on `const device::DeviceF2Blocks&` vs. `const F2BlockTensor&` (lines 170–181). Since these are unrelated types, ambiguity is unlikely today, but if either type ever grows an implicit conversion constructor to the other, the overload set becomes a footgun. Worth a note, not an action.

**"standard-C++ only" but uses C++20 `std::span`.** The comment on line 12 says "standard-C++ only," but `std::span` is a C++20 feature. The project presumably targets C++20, but the header's self-description is slightly imprecise.

## The "slop" test

**Not slop.** Slop is:
- Magic numbers without explanation
- Copy-pasted code with stale comments
- No error checking
- Obviously wrong algorithms that happen to pass tests

This has none of that. Every default has a name and a rationale. The comments are dense but they explain *why*, not just *what*. The API layering is coherent.

## What it actually looks like

This looks like **a carefully written public contract header by a domain expert who values backwards compatibility and explicit design decisions.** The author is clearly thinking in terms of frozen APIs, parity with a reference implementation (AT2), and separation between host and device compute. It is not the work of someone throwing code together; it is the work of someone who expects this header to outlive several milestones.

A senior C++ reviewer would say: "Solid design, but the result struct is doing too much and the comments need a haircut." A senior domain reviewer would say: "The shapes are right and the defaults match the reference." The main risk is that the header's verbosity and kitchen-sink result type will make future refactors harder than they need to be.

**Verdict:** B+ to A-. Competent, well-reasoned API design with some structural overgrowth that a senior reviewer would want to prune before calling it pristine.
