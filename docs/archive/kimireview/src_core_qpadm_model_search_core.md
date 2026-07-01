I read `model_search_core.cpp` plus its header. This is a tiny file — basically one deterministic shard-planning function — but it’s a good microcosm of the codebase’s strengths and a few recurring senior-dev nitpicks.

## What's genuinely good

- **The algorithm is correct and the invariant is well-stated.** The count-balanced contiguous tiling (`base = n / G`, `rem = n % G`, first `rem` devices get `+1`) covers `[0, n_models)` with no gaps or overlaps, and the post-condition is explicitly called out at lines 13-15 and 29-30.
- **Clean separation of concerns.** The header advertises this as HOST-PURE / CUDA-FREE / GPU-FREE-TESTABLE, and the `.cpp` honors that: no `cuda_runtime.h`, no `printf`, no device globals, just standard C++.
- **Modern C++ hygiene.** It uses `std::vector`, `reserve`, value-type `ModelShard`, and `[[nodiscard]]` on the declaration. The only failure mode is signaled with an exception rather than a magic error code.
- **No output/CSV/emit spaghetti.** Unlike a lot of genomics compute code, this doesn’t try to log, print, or write files. It just returns a plan.

## What a senior developer would flag

**`G == 0` is a runtime exception, not a type-system impossibility.**

```cpp
// model_search_core.cpp:10-12
if (G == 0) {
    throw std::runtime_error("plan_model_shards: G must be >= 1");
}
```

`std::runtime_error` is the wrong exception family here — this is a precondition violation, so `std::invalid_argument` is more accurate. More importantly, `G` is `std::size_t`, which *allows* zero by construction. If the domain really is “at least one device,” a `NonZeroDeviceCount` strong type or at least `std::size_t` with an explicit contract would make the bug unrepresentable.

**Narrowing `g` to `int` is a latent footgun.**

```cpp
// model_search_core.cpp:26
shards.push_back(ModelShard{static_cast<int>(g), lo, hi});
```

`ModelShard::g` is declared `int` in the header (line 27), but `g` iterates up to `G`, which is `std::size_t`. On platforms with 64-bit `size_t` and 32-bit `int`, a shard plan with `G > INT_MAX` silently truncates the device index. For genomics that will never happen in practice, but a senior reviewer will still ask why `g` isn’t just `std::size_t` — especially since the rest of the math is `size_t`-clean.

**The struct exposes raw fields with no invariant enforcement.**

```cpp
// model_search_core.hpp:26-30
struct ModelShard {
    int         g;
    std::size_t lo;
    std::size_t hi;
};
```

This is fine as a plain aggregate, but there’s no guarantee that `lo <= hi` or that `g` matches the vector index. The planner happens to produce correct values, but downstream code has to trust that. A small constructor or a `bool valid() const` would make the contract machine-checkable.

**Minor style nits:**
- Parameter name `G` is capitalized, which conflicts with the usual `snake_case` convention elsewhere in the project.
- `shards.push_back(ModelShard{...})` at line 26 could be `shards.emplace_back(g, lo, hi)` — not a bug, just slightly less idiomatic C++11.

## The "slop" test

**Not slop.** There are no magic numbers, no stale copy-pasted comments, no `TODO`s, no raw `malloc`, no mixed `printf`/`std::cout`, and the math is obviously correct. The comments are dense, but they explain *why* the tiling works and what the host re-sort relies on, which is exactly the kind of context a shard planner needs.

## What it actually looks like

This looks like **competent, disciplined utility code written by someone who understands both the algorithm and the value of keeping GPU details out of pure index math.** It’s a little overcautious in prose and a little undercautious in types, but the thinking is clear.

## Verdict

**B+.** Ship it, but tighten the precondition (`std::invalid_argument`, and ideally a non-zero `G` type) and either justify the `int` device index or widen it to `std::size_t`. For a job-application showcase, this is the kind of file that earns trust rather than embarrassment.