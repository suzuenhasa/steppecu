I read through this carefully. This is **not slop** — it's a small, focused command wrapper that mostly follows the project's CUDA-free app-layer rules and gets the domain contract right. A senior developer would find it competent, but would flag the hand-rolled output formatter, mixed I/O conventions, and a couple of exit-code/naming warts.

## What's genuinely good

- **The command contract is explicit.** The header comment (lines 2–11) explains exactly what `steppe dates` does, why it reads the genotype triple directly instead of the f2 cache, and how the GPU is reached only through the CUDA-free `run_dates` seam. That kind of framing saves reviewers time and shows architectural awareness.
- **Good input validation.** Lines 79–96 fail fast on missing `--prefix`, missing `--target`, and wrong `--left` arity, with specific error messages and the right exit code (`kExitInvalidConfig`).
- **Proper GPU/resource plumbing.** It uses `device::build_resources(config.device())`, checks `resources.gpus.empty()` before calling into the compute seam, and passes the resources handle through to `run_dates` (lines 110–118). The app layer stays CUDA-free.
- **Clear fault-vs-domain-outcome separation.** The final return uses `cfg::exit_code_for(result.status)` (line 142), and the header comment explicitly calls out the "degenerate run = NaN date + exit 0" contract. That's the right CLI behavior for this codebase.
- **`status_text` covers the whole enum.** Lines 36–46 map every `steppe::Status` value, with a defensive `"unknown"` fallback. No missing cases.
- **RAII output handling.** The `--out` file path is opened as an `std::ofstream` and checked with `if (!out)` before writing (lines 134–140).

## What a senior developer would flag

**A bespoke, hand-rolled output formatter instead of reusing the shared emit primitives.**

`emit_dates` (lines 48–74) builds CSV/TSV/JSON from scratch:

```cpp
auto d = [](double v) -> std::string {
    if (std::isnan(v)) return "NA";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f", v);
    return std::string(buf);
};
```

This duplicates work that `result_emit.hpp` already does for the other commands (`fmt_double`, `json_double`, `csv_quote`, `status_str`, etc.). It also hard-codes `"%.6f"` and a 64-byte buffer as magic numbers. More importantly, the JSON path manually quotes strings without escaping:

```cpp
os << "  \"target\": \"" << target << "\",\n";
```

If a population label ever contains a quote, backslash, or control character, the output will be invalid JSON. The shared emit utilities presumably handle this; a local one-off does not. A senior reviewer would ask why this command has its own emitter at all.

**Mixed I/O conventions: `std::fprintf(stderr, ...)` for diagnostics, C++ streams for output.**

Lines 80, 84, 92, 112, 120, 126, and 136 all use C-style stdio:

```cpp
std::fprintf(stderr, "steppe dates: --prefix PREFIX.{geno,snp,ind} is required\n");
```

But the actual result goes through `std::cout` or `std::ofstream` (lines 132 and 140). The same mismatch appears in `cmd_f4.cpp` and `cmd_f3.cpp`, so it's a project-wide convention drift, but it's still a real code-smell in a file that claims "main() owns stdout/stderr." Pick streams everywhere or centralize diagnostics; mixing the two makes testing and redirection harder.

**The `try` block is too broad and maps device errors to the wrong exit code.**

```cpp
} catch (const std::exception& e) {
    std::fprintf(stderr, "steppe dates: input/device error: %s\n", e.what());
    return cfg::kExitIoError;
}
```

Line 121 returns `kExitIoError` for *any* exception thrown by `build_resources` or `run_dates`, including device OOM. The project's `exit_code.hpp` contract has a dedicated `kExitDeviceOom` for that. Collapsing "missing file", "bad genotype format", and "GPU out of memory" into one code hurts debugging and violates the fault taxonomy the CLI claims to enforce.

**`config.qpdstat_prefix()` for the dates `--prefix` is a naming leak.**

Lines 79 and 98 both call `config.qpdstat_prefix()`. The `dates` command has nothing to do with `qpdstat`; the field name is clearly inherited from a shared config struct. It's not a bug, but it reads like a placeholder that never got renamed, and it would make a senior reviewer wonder what other command fields are overloaded this way.

**Format validation happens after the expensive GPU work.**

Lines 124–129 parse `--format` only after `run_dates` has already executed. If the user misspells `--format`, they pay for the full GPU run before getting an error. `cmd_qpdstat.cpp` does the same thing, so it's a shared pattern, but it's still user-hostile.

**Minor nits.**

- `std::ios::binary | std::ios::trunc` on line 134 is unusual for textual CSV/TSV/JSON output. Harmless on Linux, but worth a comment.
- The `status_text` helper here probably duplicates `status_str` or similar inside `result_emit.cpp`; another sign that this command should be using the shared emit layer.
- The `d` lambda is fine, but using `std::snprintf` for a simple 6-decimal formatter is more C-ish than the rest of the C++20 app layer.

## The "slop" test

**Not slop.** Slop would be unhandled errors, magic numbers with no explanation, copy-pasted stale comments, or output code that only works by accident. None of that is here. The comments are dense but accurate, validation is thorough, and the compute is properly delegated. The bespoke emitter is a maintainability issue, not slop — it's clearly intentional, just not as clean as it could be.

## What it actually looks like

This looks like **solid, workmanlike command-layer code written by someone who understands the `steppe` architecture and the DATES domain well.** It follows the CUDA-free seam rules, validates inputs sensibly, and respects the domain-outcome contract. The weak spots are polish and integration: a hand-rolled JSON/CSV emitter that duplicates shared utilities, a `qpdstat`-shaped config field leaking into `dates`, and a catch-all error path that conflates I/O faults with device faults.

A senior C++ reviewer would say: "Competent and correct, but why didn't you reuse `result_emit`? And fix the exit-code mapping." A senior CUDA reviewer wouldn't have much to say about this file specifically — which is a compliment to the layering — but the device-error handling would catch their eye.

## Verdict

**B** — Correct, well-documented command glue that follows the project's architectural rules, but the custom output formatter, mixed `fprintf`/stream conventions, and loose error-classification keep it from being showcase-quality code.
