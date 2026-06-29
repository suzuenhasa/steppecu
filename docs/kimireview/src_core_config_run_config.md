# Code review: `src/core/config/run_config.hpp`

This is a **configuration value-object header** with the right architectural intent, but it doesn't fully enforce the invariants it advertises. A senior reviewer would see competent C++ hygiene and a clean CUDA-free seam, then immediately notice that the "immutable, validated" contract has a public back door.

## What's genuinely good

- **The immutability intent is front-and-center.** All public accessors are `const`, `noexcept`, and marked `[[nodiscard]]` (lines 46–124). That's the correct surface for a frozen config object.
- **CUDA-free boundary is respected.** The includes stick to `steppe/config.hpp`, `steppe/qpadm.hpp`, `io/ind_reader.hpp`, and `core/config/cli_args.hpp` (lines 27–34) — no device headers leak into core config. The comment at lines 17–23 explains *why*, and the design matches the architecture doc.
- **Value semantics, no ownership games.** Fields are plain `std::string` and `std::vector<std::string>` (lines 148–178). No raw pointers, no `new`/`delete`, no `std::unique_ptr` theater. For a DTO-like config this is exactly right.
- **Default member initializers are documented.** Lines 157, 163, 164, 167, 169, 172, 173, 175, 176, 177, and 178 tie defaults to doc sections or explain sentinel behavior. `blgsize_cm_ = kDefaultBlockSizeCm` is a nice named-constant example.
- **Friendship with `ConfigBuilder` is the right seam.** Line 140 keeps the mutable path private while letting the builder populate the object. That's a textbook pattern for "validate once, freeze."

## What a senior developer would flag

**The public default constructor undermines the whole premise.**

```cpp
37  /// The immutable, validated run configuration. Constructed ONLY by ConfigBuilder::
38  /// build() (which validates); the public ctor is intentionally the
39  /// aggregate-from-builder path. Const accessors only — no field is publicly mutable.
40  class RunConfig {
41  public:
42      RunConfig() = default;
```

Line 42 exposes a default constructor that produces a `RunConfig` with `command_ == Command::None`, empty paths, and default sentinel values. Nothing in this file validates it. The comment on line 39 says the public ctor is "intentionally the aggregate-from-builder path," but that's wrong — this class is **not** an aggregate because a user-declared constructor exists, and the default ctor is available to *any* caller, not just the builder. A senior reviewer would either make this constructor `private` or delete it and give `ConfigBuilder` an explicit factory right.

**Comment/contract drift on the same point:**

```cpp
136  // ConfigBuilder is the ONLY constructor of a validated RunConfig (it sets these
137  // fields after build()-validation). Friendship keeps the fields const-after-build
138  // without a public mutating surface (architecture.md §9 "config is const").
139  friend class ConfigBuilder;
```

Lines 136–139 claim `ConfigBuilder` is the *only* constructor of a validated config, but line 142's public default ctor contradicts that. The phrase "without a public mutating surface" is also slightly off: the default ctor *is* a public mutation surface for creating an unvalidated object.

**Sentinel values are used without named constants.**

```cpp
166  int max_sources_ = -1;          // -1 ⇒ "up to the whole pool" (app default)
168  int sweep_top_k_ = -1;          // f4/f3-sweep --top-k (>0 ⇒ TopK mode; -1 ⇒ MinZ mode)
```

`-1` is a common "unset" sentinel, but it appears for two unrelated concepts and isn't given a name like `kUnlimitedSources` or `kSweepTopKDisabled`. For a config object this public, named constants would make downstream code far more readable and less error-prone.

**Magic string default:**

```cpp
157  std::string format_ = "csv";   // cli-bindings.md §4.4 default
```

A `constexpr std::string_view kDefaultOutputFormat = "csv";` would be cleaner and avoid a runtime string construction if this header is included everywhere.

**The accessor surface is huge and mostly homogeneous.**

There are ~40 getters spanning lines 46–124. They are correct, but the sheer repetition suggests this object is acting as a flat data bag rather than a typed seam. That's not a bug, but a senior reviewer would ask whether commands like `f4`, `qpgraph`, and `extract-f2` really need to share one monolithic config, or whether a variant/sum-type of per-command configs would better express the domain. Right now every command carries every other command's knobs.

**`Command::None` as a valid state.**

```cpp
142  Command         command_ = Command::None;
```

Combined with the public default ctor, `Command::None` lets callers hold a "config" that doesn't correspond to any command. If the design truly is "validate once and freeze," `Command::None` should be unreachable after construction.

## The "slop" test

**Not slop.** There are no unexplained magic numbers, no copy-pasted accessors with stale comments, no raw resource management, and no obviously wrong invariants. The comments are dense but they explain *why* the class exists and how it fits into the architecture. This is deliberate code.

## What it actually looks like

This looks like **competent scaffolding written by someone who understands the project boundaries but hasn't fully tightened the screws on the public API.** The author clearly gets the value of immutability, validation seams, and CUDA-free layering, but they left the default constructor public and let the object sprawl into a catch-all DTO. In a job-application portfolio, the architectural commentary and `[[nodiscard]]/noexcept` discipline would impress; the leaky constructor and 40-getter surface would make a senior interviewer pause and ask, "How do you enforce that this object is always valid?"

## Verdict

**B.** Strong architectural intent and clean value semantics, but the public default constructor and un-named sentinels undermine the very invariants the file claims to enforce.

**Bottom line:** Good bones, loose seams — tighten the constructor and the sentinels and it becomes A-grade config plumbing.
