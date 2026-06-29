I read through this carefully. This is **not slop** — it's clearly written by someone who understands the project's command-layer architecture and knows how to glue the CUDA-free seams together. A senior developer would find it competent and mostly clean, but would flag a few convention inconsistencies and a naming leak.

## What's genuinely good

- **The command contract is explicit and well-documented.** The header comment (lines 2–19) states exactly what `f3` is (sibling of `f4`, not a `qpAdm` fork), what the GPU path is, the input options, and the output schema. That kind of clarity at the top of a command file saves reviewers a lot of time.
- **Good reuse of existing components.** The file leans on `read_f2_dir`, `PopResolver`, `build_resources`, `upload_f2_blocks_to_device`, `run_f3`, and `emit_f3_result` instead of reinventing them. The `--all-triples` sweep mode delegates to a single shared helper:

  ```cpp
  if (config.sweep_all_combinations()) {
      return run_fstat_sweep(config, /*k=*/3, "f3");
  }
  ```

  That's the right way to avoid copy-paste drift between `f3`, `f4`, and `qpwave`.
- **Domain-vs-fault error handling is mostly right.** Invalid config maps to `kExitInvalidConfig`, I/O failures to `kExitIoError`, and device errors are caught and mapped to `kExitRuntimeError`. The final domain outcome is passed through `exit_code_for(result.status)` (line 202), which respects the "record-and-continue, exit 0" contract.
- **Defensive input validation.** `build_triple_names` checks for mismatched `--pop1/--pop2/--pop3` lengths, empty columns, and non-multiple-of-3 `--pops` lists. Name resolution fails fast with a clear message per bad population.
- **`std::span` used to pass the triples table into `run_f3`.** Line 170 avoids raw pointer/size pairs:

  ```cpp
  result = run_f3(dev_f2, std::span<const std::array<int, 3>>(triples), opts,
                  resources);
  ```

## What a senior developer would flag

**Mixed output conventions: `std::fprintf(stderr, ...)` for errors, C++ streams for results.**

The file uses C-style stdio for every diagnostic:

```cpp
std::fprintf(stderr, "steppe f3: --f2-dir is required\n");
```

but then emits the actual result through `std::cout` or `std::ofstream`:

```cpp
emit_f3_result(std::cout, fmt, result, l1, l2, l3);
```

Line 191:

```cpp
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
```

A codebase should pick one style. Since the output is fundamentally stream-oriented, the diagnostics should probably be stream-oriented too (or the output should use `FILE*`). Mixing the two in the same command is a small but real convention mismatch.

**Magic number `3` for the triple arity is everywhere.**

Lines 82, 87, and 90:

```cpp
if (pops.size() % 3 != 0) { ... }
const std::size_t n = pops.size() / 3;
triples.push_back({pops[3 * k + 0], pops[3 * k + 1], pops[3 * k + 2]});
```

Yes, `f3` is literally "three populations," but a named constant (`kTripleSize` or similar) would make the intent machine-checkable and reduce future drift if a different arity variant is ever added.

**`QpAdmOptions` is a strange name inside an `f3` command.**

Line 157:

```cpp
const QpAdmOptions opts = config.qpadm_options();
```

The comment explains that only the struct default matters here (`fudge` defaults to 0), but a reader has to wonder why an `f3` command is pulling `qpadm_options()`. If the options are genuinely shared, a neutral name like `FitOptions` or `FStatOptions` would be cleaner. This is a naming/dependency leak, not a logic bug.

**`std::ios::binary` for CSV/TSV output.**

Line 191 again:

```cpp
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
```

Binary mode suppresses newline translation, which is fine on Linux but unusual for textual CSV/TSV. It's harmless, but a reviewer would pause and ask why `binary` is needed. A short comment would remove the doubt.

**The `--all-triples` branch bypasses the local `--f2-dir` validation.**

Line 101 returns into `run_fstat_sweep` before line 106 checks `config.f2_dir().empty()`. Presumably `run_fstat_sweep` validates its own inputs, but it's slightly odd that the same command has two paths with two different validation orders. If a user calls `steppe f3 --all-triples` with no `--f2-dir`, the error message and exit code come from a different file. Minor, but worth unifying.

**`build_triple_names` uses a C-style `bool` + `std::string& err` out-parameter.**

Line 52:

```cpp
[[nodiscard]] bool build_triple_names(const cfg::RunConfig& config,
                                      std::vector<std::array<std::string, 3>>& triples,
                                      std::string& err)
```

The project already has `steppe::Status` (included on line 38). Using `Status` or `std::expected`-like type would be more idiomatic and consistent with the rest of the seam. As a private helper it's not a disaster, but it stands out.

**The `std::span` cast is verbose.**

Line 170:

```cpp
std::span<const std::array<int, 3>>(triples)
```

A simple `std::span{triples}` is sufficient and reads better. This is a nitpick, not a bug.

## The "slop" test

**Not slop.** Slop would be:
- Magic numbers without any surrounding explanation
- Copy-pasted compute or output logic
- Missing error handling on I/O or GPU calls
- Stale comments that no longer describe the code

None of that is here. The comments are dense but accurate, the compute path is delegated, and errors are handled at every boundary.

## What it actually looks like

This looks like **solid, production-grade command glue written by someone who knows the `steppe` architecture well.** The author understands the layering rules (CUDA-free app layer, CUDA-free seams, stdout/stderr ownership), understands the difference between domain outcomes and runtime faults, and avoids the classic trap of duplicating the `f2`-dir load or result formatting code.

A senior C++ reviewer would say: "Competent, well-structured, but tighten up the output style and fix the `QpAdmOptions` name leak." A CUDA reviewer wouldn't have much to say about this file specifically, because the GPU details are properly pushed down into the device seams.

The main things holding it back from an A are the **fprintf/streams inconsistency** and the **qpadm-shaped option struct leaking into f3**. Those are polish issues, not competence issues.

## Verdict

**B+** — Clean, correct command glue with a couple of convention and naming warts that a senior reviewer would ask to fix before calling it "showcase" quality.
