I read through this carefully. This is **not slop** — it's clearly written by someone who understands the architecture and the genomics domain, but a senior developer would flag the command-layer plumbing as repetitive and slightly underbaked relative to the compute-layer discipline.

## What's genuinely good

- **The `kTupleArity` single-source-of-truth is exactly right.** Line 53 freezes the 5-tuple shape once, and every downstream use (column count, `--pops` modulus, `std::array` arity, loop bounds) derives from it. The comment even cites the coding-standard clause. This is the kind of small decision that prevents copy-paste drift.
- **Clean architectural layering.** The command file has no CUDA headers; it reaches the GPU only through the CUDA-free seams `device::build_resources`, `device::upload_f2_blocks_to_device`, and `steppe::run_f4ratio` (lines 169-180). That matches the project's stated layering and is the correct way to keep the app buildable without nvcc.
- **Good use of modern C++ for the domain data.** `std::array<std::string, kTupleArity>` for tuple names and `std::array<int, kTupleArity>` for resolved indices (lines 123, 138) are exactly the right shapes; no raw allocations, no C-style arrays, no pointer arithmetic.
- **RAII and exit-code discipline.** The output `std::ofstream` (line 200) and `device::Resources` are value types with deterministic cleanup, and the final return maps the domain `result.status` through `cfg::exit_code_for(result.status)` (line 211) rather than inventing ad-hoc exits at the end.
- **Honest comments about design intent.** The header comment and inline notes explain *why* f4-ratio is a sibling of f4/f3 (not a qpAdm fork), why the GPU path is the deliverable, and why domain outcomes are exit-0 while faults are nonzero. That's domain competence on display.

## What a senior developer would flag

**Mixed output conventions: `std::fprintf(stderr, ...)` vs `std::cout`/`std::ofstream`.**

```cpp
std::fprintf(stderr, "steppe f4-ratio: --f2-dir is required\n");          // line 113
std::fprintf(stderr, "steppe f4-ratio: %s\n", dir.error.c_str());        // line 118
...
emit_f4ratio_result(std::cout, fmt, result, l1, l2, l3, l4, l5);        // line 198
```

The file comment says "main() owns stdout/stderr (architecture.md §10)," but the implementation splits stderr into C-style `fprintf` and stdout into C++ streams. A senior reviewer would ask for a unified logging/reporting primitive — especially because the stderr prefix `"steppe f4-ratio: "` is hand-rolled on **six** separate lines (113, 118, 126, 132, 149, 185, plus two more for format/file errors at 192 and 202). That repetition is exactly where copy-paste drift and inconsistent punctuation will appear in the next command someone ports.

**Output-format and output-file validation happen *after* the expensive GPU work.**

```cpp
// lines 162-187: build_resources, upload_f2_blocks_to_device, run_f4ratio
// lines 189-195: parse_output_format(...)
// lines 197-207: open --out file
```

If the user passes `--format=parquet` or `--out=/no/such/dir/out.csv`, the code happily burns GPU cycles first and only then fails. Cheap config validation belongs at the top of `run_f4ratio_command`, before the device work.

**Error-code taxonomy is inconsistent at the edges.**

- `PopResolver` construction failure returns `kExitIoError` (line 133), but a name-resolution failure inside the loop returns `kExitInvalidConfig` (line 150). Both are "you gave me a bad population name," so why different exits?
- `--f2-dir` missing returns `kExitInvalidConfig` (line 114), which is right; but the catch-all device catch returns `kExitRuntimeError` (line 186), which is also right. The resolver cases are the ones that look arbitrary.

**Five parallel label vectors are a code-smell.**

```cpp
std::vector<std::string> l1, l2, l3, l4, l5;          // line 140
l1.reserve(tuple_names.size()); l2.reserve(...);       // lines 141-143
l1.push_back(resolver.label_at(idx[0]));               // lines 155-159
```

This is the classic "structure of arrays" pattern at the app layer for no performance reason. A single `std::vector<TupleLabels>` (or even just passing the original `tuple_names` through to the emitter) would be clearer and harder to get out of sync. As written, it's five reserve calls, five push_backs, and five parameters every time `emit_f4ratio_result` is called.

**`QpAdmOptions` for f4-ratio is semantically odd.**

```cpp
const QpAdmOptions opts = config.qpadm_options();      // line 166
```

The comment says "fudge defaults to 0 ... opts here is the default," but the f4-ratio command is explicitly *not* qpAdm. A senior reviewer would want either a dedicated `F4RatioOptions` type or a clearly named subset, not a repurposed qpAdm config bag. Right now the next reader has to trust the comment that nothing qpAdm-specific leaks through.

**`std::ios::binary` for CSV/TSV output.**

```cpp
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);  // line 200
```

This is probably deliberate to suppress CRLF translation on Windows, but for CSV and especially JSON it's an unusual choice that will confuse anyone expecting plain text. A comment explaining the intent would help.

**The catch block is `std::exception` only.**

```cpp
} catch (const std::exception& e) {                    // line 181
```

If a CUDA or STL helper ever throws something not derived from `std::exception`, the process terminates with no diagnostics. In a command-line tool the practical risk is low, but a senior reviewer would add a `catch (...)` fallback that logs "unknown exception" and returns `kExitRuntimeError`.

**Minor: the comment says output sink is "REUSED verbatim from cmd_f3.cpp/cmd_f4.cpp" (line 13-15), but the file-vs-stdout branching logic is still inline here.** The *emitter primitive* is reused, not the sink orchestration. Stale or overstated comments erode trust in the accurate ones.

## The "slop" test

**Not slop.** Slop would be magic 5s scattered through the code, unchecked `config` accessors, duplicated f2-block upload logic, or hand-rolled CSV formatting. None of that is here. The constants are centralized, errors are checked, the GPU path is reused through proper seams, and the comments explain *why* the code is shaped the way it is. The repetition and validation ordering are workmanship issues, not competence issues.

## What it actually looks like

This looks like **solid, architecture-aware app-layer glue written by someone who knows the genomics domain and respects the project's layering rules.** The hard parts — f2-block I/O, name resolution, device upload, and the actual f4-ratio statistic — are delegated to the right modules. The weaknesses are all in the boring command scaffolding: error-message repetition, validation ordering, and a few too many parallel containers. A senior C++ reviewer would say: "The design is correct, the math plumbing is in the right place, but tidy up the command boilerplate before we add a fourth command that copy-pastes this pattern."

**Verdict:** Respectable production code with identifiable maintainability debt. **B+**. The compute-side choices are A- work; the command-side repetition and late validation keep it from an A.
