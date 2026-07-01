I read through this carefully. This is **not slop** — the architecture is sound and the author clearly understands the project's layering rules, but a senior developer would flag the surface finish: it's a C++20 CLI file that still reports errors like a 1990s C program, and the repeated ad-hoc printing would not impress in a job-application showcase.

## What's genuinely good

- **Respect for the CUDA-free seam.** The file includes no CUDA headers and reaches the GPU only through `device::build_resources`, `device::upload_f2_blocks_to_device`, and `steppe::run_qpadm` (lines 27–30, 106–117). That shows the author read `architecture.md` and `cli-bindings.md` and actually followed the layering.
- **Clear fault-vs-domain separation.** Device faults are caught and return `kExitRuntimeError` (lines 106–124), while model-level outcomes like `RankDeficient` are passed through `exit_code_for(result.status)` and can still exit 0 (lines 153–156). The comment on lines 119–121 explains the contract accurately.
- **RAII and no manual ownership.** `device::Resources`, `device::DeviceF2Blocks`, and `std::ofstream` are all RAII-managed; there are no raw pointers, `new`/`delete`, or `fopen`/`fclose` pairs.
- **Small performance care.** `left_labels.reserve(model.left.size())` on line 137 is the right instinct, and the resolve helper is marked `[[nodiscard]]` on line 41.
- **Section comments map the pipeline.** Lines 76, 87, 98, and 126 break the function into readable phases that match the design doc narrative.

## What a senior developer would flag

**Mixing C-style `std::fprintf` with C++ streams in the same file.**

```cpp
std::fprintf(stderr, "steppe qpadm: --target is required\n");          // line 47
std::fprintf(stderr, "steppe qpadm: %s\n", t.error.c_str());          // line 60
emit_qpadm_result(std::cout, fmt, result, target_label, left_labels);  // line 142
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc); // line 144
emit_qpadm_result(out, fmt, result, target_label, left_labels);        // line 150
```

The project says main owns stdout/stderr, not that it must use `fprintf`. Pick one style. A modern C++ CLI would use streams (or a tiny logging helper) so formatting is type-safe and consistent. Right now the file includes both `<cstdio>` and `<iostream>`/`<ostream>` just to talk out of both sides of its mouth.

**Ad-hoc error strings copy-pasted everywhere.**

The prefix `"steppe qpadm: "` is hard-coded on lines 47, 51, 55, 60, 62, 64, 78, 83, 90, 109, 122, 130, and 146. That is exactly the kind of repetition that drifts into stale copy-paste bugs when someone renames the command or changes the punctuation. A one-line helper:

```cpp
void error(std::string_view msg) { std::cerr << "steppe qpadm: " << msg << '\n'; }
```

would centralize the prefix and make the file shorter and less error-prone.

**`model.model_index = 0;` is a naked magic number.**

```cpp
model.target = t.index;
model.left = l.indices;
model.right = r.indices;
model.model_index = 0;   // line 69
```

What does `0` mean? First model in a batch? A default? It deserves a named constant or at least a comment, especially because the rest of the model fields are carefully resolved.

**The `resources.gpus.empty()` check contradicts its own comment.**

```cpp
device::Resources resources = device::build_resources(config.device());
if (resources.gpus.empty()) {                                         // line 108
    std::fprintf(stderr,
                 "steppe qpadm: no CUDA device available (steppe is a GPU "
                 "product; a CUDA-capable GPU is required)\n");
    return cfg::kExitRuntimeError;
}
```

The comment on lines 99–103 says `build_resources` "enumerates / binds a CUDA device and throws when none is visible." If that's true, the empty check is dead code. If it's not true, the comment is wrong. A senior reviewer will stop and ask which contract is real.

**Defensive `parse_output_format` branch is almost certainly unreachable.**

```cpp
if (!parse_output_format(config.format(), fmt)) {
    // ConfigBuilder::build() already validates --format, so this is defensive.  // line 129
```

The comment admits it. Defensive code is fine, but returning `kExitInvalidConfig` from a supposedly validated input is a smell — it either means the validation boundary is unclear, or this is leftover paranoia. In a showcase file, it looks like the author wasn't sure who owns validation.

**Binary mode for textual CSV/TSV/JSON output is odd.**

```cpp
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc); // line 144
```

`std::ios::binary` prevents newline translation. That may be intentional for reproducible byte output, but for a default CSV/JSON emitter it is surprising and should be called out in a comment if it is deliberate.

**Minor: the resolve helper does input validation that the config parser probably already does.** Lines 46–57 check for empty `--target`, `--left`, and `--right`. That is not wrong, but it duplicates validation that likely belongs in `ConfigBuilder`, and the resulting error messages are slightly different from the parser's.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted code with stale comments, missing error handling, or algorithms that are obviously wrong. This file has none of those. The repeated `"steppe qpadm: "` literals are the closest thing to copy-paste drift, but they are at least consistent. The comments are accurate and explain *why*, not just *what*.

## What it actually looks like

This looks like **solid, mid-level C++ written by someone who understands the project architecture but hasn't fully committed to a modern C++ style.** The layering, error-code philosophy, and resource handling are all correct. The problems are almost entirely at the surface: error reporting is scattered and C-ish, literals are repeated, and one or two comments contradict the code.

A senior reviewer would say: "The bones are good — fix the error reporting and clean up the literals and this is ready to ship." For a job-application showcase, those surface issues matter more than they would in day-to-day production, because a reviewer reading this in isolation will assume the style reflects how the author writes everywhere.

## Verdict

**B.** Competent, architecture-aware CLI glue with a messy presentation layer. Clean up the fprintf/stream split and centralize the error strings and it becomes a strong showcase piece.
