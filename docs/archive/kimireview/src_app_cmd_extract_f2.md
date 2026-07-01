I read through this carefully. This is **not slop** — it's clearly written by someone who understands the architecture and the genomics domain, but a senior developer would flag several maintainability and convention issues. The command works, the CUDA-free seam discipline is real, and the threading is handled responsibly; the problems are mostly layering, duplication, and stale numbering from a refactor.

## What's genuinely good

- **The CUDA-free seam discipline is enforced.** The header comment (lines 1–6) explicitly states no CUDA headers, and the file keeps its word: GPU access is only through `device::build_resources`, `device::resolve_output_tier`, and `steppe::run_extract_f2`. That's good architecture.
- **The background hash thread is done correctly.** Lines 298–311 capture the path by value, join on a RAII guard, and never detach. The `ThreadJoiner` destructor at lines 308–311 prevents the unjoined-thread destructor footgun:
  ```cpp
  struct ThreadJoiner {
      std::thread& t;
      ~ThreadJoiner() { if (t.joinable()) t.join(); }
  } geno_hash_joiner{geno_hash_thread};
  ```
- **Error mapping is deliberate.** The CLI catches library exceptions and maps them to concrete exit codes (`kExitInvalidConfig`, `kExitIoError`, `kExitRuntimeError`) instead of leaking raw exceptions or collapsing everything into one code.
- **Dry-run sizing uses the real resolver.** The `--dry-run` path calls the same `device::resolve_output_tier` the real run uses (line 265), so the estimate isn't a cheap approximation — it's the same policy on an upper-bound M. That's thoughtful.
- **Explicit pop validation is fast-fail.** Lines 118–129 check unknown population names against the resolved partition and return `kExitInvalidConfig` with the offending label. Good UX for a CLI.

## What a senior developer would flag

**Duplicate `tier_label` overloads — copy-paste drift.**

Lines 74–92 contain two nearly identical functions:

```cpp
[[nodiscard]] const char* tier_label(device::OutputTier t) {
    switch (t) { ... }
    return "resident";
}

[[nodiscard]] const char* tier_label(steppe::ExtractTier t) {
    switch (t) { ... }
    return "resident";
}
```

These are the same mapping for two different enum types. A senior dev would ask why the CLI even needs to know about both `device::OutputTier` and `steppe::ExtractTier`. The library should return one public tier type, or there should be a single conversion point, not two switch statements maintained by hand. This is exactly the kind of duplication that drifts out of sync.

**Stale section numbering from a refactor.**

The section comments jump from `---- 1.` (line 155) to `---- DRY RUN` (line 224) to `---- 5-8b.` (line 313) to `---- 9.` (line 363). Steps 2–4, 5–8a are missing. That's because the actual decode/filter/f2 compute chain was lifted into `steppe::run_extract_f2` (the comment at lines 313–322 admits this), but the numbering wasn't updated. Stale numbering is a small thing, but it signals "this file was edited in a hurry after a refactor."

**Signed/unsigned mixing around `M`.**

```cpp
long M = 0;
...
M = static_cast<long>(std::min(reader.header().n_snp, snptab.count));
...
if (snptab.count < static_cast<std::size_t>(M)) { ... }
```

`M` is `long`, `snptab.count` is `std::size_t`, and `n_snp` is presumably `std::size_t`. The comparisons require casts and invite subtle bugs if `M` ever becomes negative. A `std::size_t` with explicit guards would be cleaner.

**`std::printf` / `std::fprintf` everywhere.**

The file uses C-style I/O exclusively. That's defensible for a simple CLI, but in a C++20 codebase a senior reviewer would expect either `std::cout`/`std::cerr` or a tiny formatting helper. More importantly, the formatting is ad-hoc and repeated (e.g., `config.blgsize_cm() / kCentimorgansPerMorgan` is computed inline at lines 253 and 413). A helper like `format_morgans(double cm)` would reduce duplication and make unit testing easier.

**Magic constant in the slab-size estimate.**

```cpp
const std::size_t slab_bytes =
    static_cast<std::size_t>(P) * static_cast<std::size_t>(P) * 2u * sizeof(double);
```

The `2u` is explained in the format string as "f2 + vpair", but it should be a named constant (e.g., `kEntriesPerBlock` or `kF2PlusVPair`).

**Output parameter in `validate_explicit_pops`.**

```cpp
[[nodiscard]] bool validate_explicit_pops(const io::PopSelection& sel,
                                          const io::IndPartition& part,
                                          std::string& offending);
```

In modern C++ this should return `std::optional<std::string>` or `std::expected<bool, std::string>`. The output parameter forces the caller to declare a variable and makes the API less obvious.

**Dry-run device probe failure returns success.**

At lines 258–285, if `device::build_resources` or `resolve_output_tier` throws, the catch prints `"  device:     probe failed: ..."` and then falls through to `return cfg::kExitOk;`. For a planning aid that's arguably fine, but a senior dev would flag that a failed probe is silently OK — especially if a GPU *is* supposed to be present.

**Thread joiner can throw during stack unwinding.**

The `ThreadJoiner` destructor calls `t.join()`, which will rethrow any exception that escaped the worker lambda. If the main thread is already unwinding due to another exception, this calls `std::terminate`. In practice `sha256_file` failures are rare, but the design is not noexcept-safe. The worker should either catch internally and set an error flag, or the joiner should handle the exception explicitly.

**Plumbing leak between config and library enums.**

Lines 323–326 manually translate `cfg::PloidyMode` to `ExtractPloidy` with a nested ternary. Same for precision handling at lines 360–361. These conversions belong in the library or in a thin adapter, not scattered in the CLI command.

## The "slop" test

**Not slop.** Slop is:
- Magic numbers without explanation
- Copy-pasted code with stale comments
- No error checking
- Obviously wrong algorithms that happen to pass tests

This file has a few stale comments and one clear duplication (`tier_label`), but the logic is correct, the error handling is deliberate, and the architecture is coherent. The comments are verbose but mostly explain *why*, not just *what*.

## What it actually looks like

This looks like **solid production CLI glue written by someone who understands the domain and the build-time CUDA-free constraint, but who hasn't fully cleaned up after a major refactor.** The hard work was correctly moved into the library (`steppe::run_extract_f2`), but the command file still carries the scar tissue: duplicated tier-label helpers, section numbers that no longer match the code, and a few C-style habits (`printf`, output parameters, `std::getenv`) that are out of place in a modern C++20 project.

A senior C++ reviewer would say: "Correct and well-architected at the seams, but it needs a cleanup pass before I'd call it showcase-ready." A senior CUDA reviewer would have little to complain about because there's no CUDA in here — which, for this file, is the point.

**Verdict:** B. Competent, ship-with-cleanup code. The duplication and stale numbering are the main things that would embarrass in a portfolio review.
