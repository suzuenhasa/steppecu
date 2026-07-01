I read through this carefully. This is **not slop** — it's clearly written by someone who understands dependency injection, RAII, and the value of keeping the CUDA runtime at arm's length. A senior reviewer would find a lot to like, but would also flag some copy/paste drift and a comment style that occasionally substitutes architecture.md citations for clarity.

## What's genuinely good

- **The CUDA-free seam is real and disciplined.** The `.cpp` includes no `<cuda_runtime.h>`, touches the GPU only through `make_cuda_backend` and `visible_device_count()`, and holds the backend through `std::unique_ptr<ComputeBackend>`. That's a genuinely good layering decision, not just a comment claiming it.
- **One count query, used everywhere.** `visible_device_count()` is called exactly once (line 101) and the result is passed into both `resolve_device_order` and `validate_device_order`. The comment about avoiding "throwaway device-0 backend / leaked cudaSetDevice(0)" is borne out by the code.
- **Fail-fast before binding.** `validate_device_order` runs before any `make_cuda_backend(ordinal)` call, so an invalid or duplicated ordinal is caught with a host-side predicate instead of a deep CUDA runtime throw. Line 117 before the loop on line 128 is the right order.
- **`std::exchange` for duplicate detection** (line 86) is a nice modern-C++ touch — concise and avoids a clunky `if (seen[...])` read-modify-write.
- **Exception safety is structurally correct.** `Resources resources` and `PerGpuResources entry` are stack locals containing RAII-owning members. A mid-loop throw unwinds through `std::vector` and `std::unique_ptr` destructors, so the claim about leaking no handle or VRAM holds.
- **`std::span<const int>` for `validate_device_order`** (line 70) is the right abstraction: unit-testable without forcing a `std::vector` and CUDA-free by construction.

## What a senior developer would flag

**The stale function-name prefix in error messages:**

```cpp
// line 80-83
throw std::runtime_error(
    "steppe::device::build_resources: configured device ordinal " +
    std::to_string(ord) + ...
);
```

This is inside `validate_device_order`, not `build_resources`. The prefix is copy-pasted from the caller. It's a small thing, but in a file this heavily annotated it's exactly the kind of drift that makes a senior reviewer distrust the docs. Either say `validate_device_order` here, or factor the prefix into a shared helper.

**Comment density bordering on cargo-cult documentation:**

Lines 39-46, 71-74, 96-100, 111-116, and 123-127 all cite `architecture.md §9`, `§11.4`, `§12`, cleanup group-7, etc. The citations are accurate, but the file often repeats the same contract three times in three adjacent functions. A senior reviewer would ask: *who is this for?* If the answer is "future maintainers," a single crisp comment at the top plus precise names in code would be more readable than §-references on every other line.

**The `seen` vector type:**

```cpp
// line 77
std::vector<char> seen(span_len, 0);
```

It works, and `char` is arguably better than `std::vector<bool>` for this pattern. But `std::vector<unsigned char>` or `std::vector<std::uint8_t>` would make the intent clearer. As written, it faintly signals "I wasn't sure which small-integer type to use."

**Copy/move ergonomics in the loop:**

```cpp
// lines 129-133
PerGpuResources entry;
entry.device_id = ordinal;
entry.backend = make_cuda_backend(ordinal);
entry.caps = entry.backend->capabilities();
resources.gpus.push_back(std::move(entry));
```

This is fine, but `resources.gpus.emplace_back(ordinal, make_cuda_backend(ordinal), ...)` or a constructor on `PerGpuResources` would be more direct and less error-prone if someone later adds a field and forgets to set it. The struct is currently default-constructible and partially assigned.

**Implicit assumption that `capabilities()` is non-throwing:**

```cpp
// line 132
entry.caps = entry.backend->capabilities();   // the ONE probe, owned here
```

The header doc says this is the contract, and the `.cpp` comment repeats it. A senior reviewer would want that contract encoded somewhere enforceable — ideally `capabilities()` itself marked `noexcept` in `backend.hpp` — because the whole "budget GeForce degrades gracefully" logic depends on it.

**`span_len` is defensively over-computed:**

```cpp
// lines 75-76
const std::size_t span_len =
    (visible > 0) ? static_cast<std::size_t>(visible) : 0;
```

`visible` is an `int`; if it's negative the subsequent `ord >= visible` check already rejects every non-negative ordinal, so the ternary doesn't change behavior. Harmless, but it reads like uncertainty about whether `std::vector<char>(negative)` throws or asserts.

## The "slop" test

**Not slop.** Slop would be:
- raw `new`/`delete` for device handles
- a `cudaGetDeviceCount` call mid-loop
- duplicated validation logic
- error messages like "something went wrong"

None of that is here. The code is careful, the ownership is right, and the comments — while verbose — are mostly accurate rather than stale.

## What it actually looks like

This looks like **solid production infrastructure written by someone who thinks in architectural contracts first and code second.** The author clearly cares about layering, RAII, fail-fast behavior, and testability. The device-resource builder is doing the right things in the right order: count, validate, bind, probe, return.

The main weakness is **editorial**. The file is over-annotated with cross-references to `architecture.md`, and that density makes genuine issues (like the `build_resources:` prefix in `validate_device_order`) easier to miss. A senior reviewer would probably say: "Good bones — clean up the prose and tighten the error-message helper, then it's ready."

**Verdict:** B+ — competent, layered, and exception-safe, with a copy/paste drift in error strings and documentation that could use a ruthless edit pass.