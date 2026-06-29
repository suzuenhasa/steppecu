# Review of `src/app/extract_f2_core.cpp`

I read through this carefully. This is a **competent, workmanlike library extraction** of the CLI f2 extraction pipeline. The refactor from `cmd_extract_f2.cpp` to a throwing library entry point is the right architectural move, and the code shows solid domain awareness. But a senior reviewer would notice some copy-paste residue, a few brittle patterns, and an API that still leaks CLI-shaped assumptions.

## What's genuinely good

- **The architectural intent is correct and well-explained.** Moving the decode→filter→assign_blocks→compute chain out of the CLI and replacing `fprintf`/`exit` with exceptions + `F2ExtractResult` is exactly the kind of layering that makes the project testable and binding-friendly (lines 73–90).
- **Early validation is sensible and user-friendly.** Empty path checks, explicit-pop validation with a clear message, and a hard failure when no GPU is available all happen before any heavy I/O (lines 82–90, 57–69).
- **The CPU parity oracle path is deliberately preserved,** not deleted. That protects the golden tests and gives a host reference for the on-device filter (lines 224–291). That's disciplined.
- **Resource ownership is mostly clean.** `std::vector`, `std::move`, `std::span`, and `std::string` do the heavy lifting. The raw pointers inside `DecodeTileView` are views into containers that outlive them, which is acceptable for a C-style view struct (lines 154–163).
- **`std::span` is used well** to pass SNP tables and block arrays to the backend without exposing allocation details (lines 206–212, 295–296).

## What a senior developer would flag

**The `resources.gpus.front().backend` idiom and the `resident` check are semantically muddled:**

```cpp
steppe::ComputeBackend& backend = *resources.gpus.front().backend;      // line 108
const bool resident = (backend.capabilities().device_count > 0);        // line 196
```

If `resources.gpus` is the vector of *GPU* devices, why does the CPU fallback live behind a `gpus.front()` dereference? The name implies GPU-only, but the code then branches on `device_count`. This works only because the resource abstraction happens to put a CPU backend there when no GPU exists, which is non-obvious and fragile. A senior reviewer would want `resources` to expose a clearer "primary compute backend" accessor rather than reaching into `.gpus.front()`.

**Exception-type choices are inconsistent:**

```cpp
if (snptab.count < static_cast<std::size_t>(M)) {
    throw std::runtime_error(...);        // line 120
}
```

A caller/input mismatch between `.snp` and `.geno` is an argument error, not a runtime system failure. The empty-input case correctly uses `std::invalid_argument` (line 83), but this path and the "no GPU" path use `std::runtime_error`. That makes catch-site error classification harder than it needs to be.

**Copy-paste drift is visible in the section numbers and line-number references:**

The header says the chain was "lifted VERBATIM from cmd_extract_f2.cpp:157-498" (line 5), the maxmiss comment references "cmd_extract_f2.cpp:358-416" (line 176), and the lockstep subset references "cmd_extract_f2.cpp:427-449" (line 269). Meanwhile the inline section numbers jump from `1` to `5` to `7` to `8`/`8b`. Those line anchors will rot the first time someone edits the CLI file, and the skipped section numbers are a tell that this was pasted from a longer file and not fully renumbered. This is the kind of stale-comment debt seniors notice.

**The `M_kept <= 0` failure is duplicated almost verbatim:**

```cpp
if (M_kept <= 0) {
    throw std::invalid_argument(
        "extract_f2: every SNP was filtered out (0 of " + std::to_string(M) +
        " kept) — relax the filters");
}
```

It appears at lines 214–218 and again at 262–266. A small helper such as `ensure_some_snps_kept(M, M_kept)` would remove the duplication and make the message a single source of truth.

**`std::vector<bool>` for the keep mask:**

```cpp
std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, class_filter, mem);   // line 239
```

`vector<bool>` is a packed proxy container; it is usually avoided in performance-sensitive code because every access goes through a proxy reference. For a mask this is probably fine, but it is a code-smell that a senior C++ reviewer would flag. `std::vector<char>` or a dedicated bitset would be more explicit.

**The ploidy-counting logic assumes values are exactly 1 or 2:**

```cpp
for (int pl : ploidy_for_counts) (pl == kPloidyPseudoHaploid ? n_ph : n_dip)++;   // line 174
```

Anything other than pseudo-haploid silently counts as diploid. That's probably valid for the current enum, but it's an implicit invariant that should be explicit (e.g., an assertion or a switch).

**`view.ploidy = kPloidyDiploid` as an "unused" fallback:**

```cpp
view.ploidy = kPloidyDiploid;  // uniform fallback (unused when a vector/flag is set).   // line 163
```

Comments that say "this is unused" are a warning sign. If it really is unused, the API should not require it; if it is used as a fallback by some backend path, then the comment is misleading and a bug could hide there.

**The tier switch has no default:**

```cpp
switch (dev_f2.tier) {
    case device::OutputTier::Resident: out.tier = ExtractTier::Resident; break;
    case device::OutputTier::HostRam:  out.tier = ExtractTier::HostRam;  break;
    case device::OutputTier::Disk:     out.tier = ExtractTier::Disk;     break;
}
```

If someone adds a new `OutputTier` value, this becomes UB / a compiler warning at best (lines 321–325). A `default: std::unreachable()` or a `static_assert` on the enum count would be safer.

**`SIZE_MAX` as a "read everything" sentinel:**

```cpp
const io::SnpTable snptab = io::read_snp_table(fmt, snp, SIZE_MAX);   // line 103
```

`SIZE_MAX` is a magic value from the caller's perspective. A named constant such as `io::kReadAllSnps` would document intent.

## The "slop" test

**Not slop.** Slop would be un-commented magic numbers, no error handling, obviously wrong filtering, or copy-pasted code with stale logic. This file has none of those. The comments are dense but mostly explain *why* decisions were made and *what* invariants must hold. The few warts above are maintainability issues, not correctness disasters.

## What it actually looks like

This looks like a **solid, mid-level refactoring pass by someone who understands both the genomics domain and the project's layering goals.** The move from CLI exit codes to a library exception contract is the right call, and the author clearly cares about keeping the CPU and GPU paths bit-exact. The rough edges — copy-pasted section numbers, duplicated error messages, the `gpus.front()` backend access, and a few implicit invariants — are exactly the kind of things a senior reviewer would ask to be cleaned up before merging, but they wouldn't make anyone doubt the author's competence.

A senior C++ reviewer would say: "Good shape, but tighten the ownership semantics, deduplicate the error handling, and scrub the stale line-number references." A domain reviewer would say: "The math and parity logic look right; just make the CPU/GPU dispatch less accident-prone."

## Verdict

**B+ — a respectable, production-bound refactor that still needs a polish pass before it could be called exemplary.**
