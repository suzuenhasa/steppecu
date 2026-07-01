I read through this carefully. This is **not slop** — it's clearly written by someone who understands the `steppe` architecture and the f4/qpAdm distinction. A senior developer would find it competent and mostly correct, but would flag the inconsistent I/O conventions and a few stylistic warts.

## What's genuinely good

- **The domain framing is precise.** The header comment (lines 3-19) correctly distinguishes f4 as a sibling of qpwave, not a fork of qpAdm: no target, no ALS, no rank. It also explains the deliverable (GPU path), the data flow, and the output schema. This shows real knowledge of the genomics pipeline, not just cargo-culted comments.
- **Clean architectural layering.** The file is "PLAIN C++20, app-only, NO CUDA header" (line 16) and reaches the GPU only through CUDA-free seams (`device/resources.hpp`, `device_f2_blocks.hpp`, `steppe/f4.hpp`). That's exactly the right separation for an app-layer command.
- **`build_quartet_names` is well-structured.** It handles the two input styles (row-aligned `--pop1/2/3/4` vs. grouped `--pops`), validates shapes, gives useful error messages, and returns `[[nodiscard]] bool` with an out-parameter reason.
- **Reuses existing paths instead of copy-pasting.** The comment at lines 13-14 claims the f2-dir load, name resolution, build/upload chain, and output sink are reused from `cmd_qpwave.cpp`; the sweep mode at line 105 routes to `run_fstat_sweep`. That shows good discipline against duplication.
- **Modern C++ surface details.** Uses `std::span<const std::array<int, 4>>` (line 174), range-for where appropriate (line 140), and RAII file handling (line 195).
- **Clear fault-vs-domain-outcome separation.** The comment at lines 176-179 and the final `exit_code_for(result.status)` (line 206) show the author understands the CLI contract: exceptions/faults → nonzero, domain results → exit 0 with record-and-continue.

## What a senior developer would flag

**Mixed I/O conventions: `std::fprintf(stderr, ...)` everywhere, but `std::cout` / `std::ofstream` for output.**

The file includes both `<cstdio>` and `<iostream>` / `<ostream>` / `<fstream>`. Errors go to `stderr` via `fprintf`:

```cpp
std::fprintf(stderr, "steppe f4: --f2-dir is required\n");          // line 110
std::fprintf(stderr, "steppe f4: %s\n", dir.error.c_str());        // line 115
std::fprintf(stderr, "steppe f4: device error: %s\n", e.what());   // line 180
```

But normal output goes through streams:

```cpp
emit_f4_result(std::cout, fmt, result, l1, l2, l3, l4);            // line 193
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc); // line 195
```

A senior reviewer would ask: why two output systems in one file? The comment says "main() owns stdout/stderr (architecture.md §10)" (line 18), but this file is *not* `main()` — it's the command implementation, and it's writing directly to both. Either pick streams everywhere or centralize logging through an app-provided sink. The current split is a real convention mismatch and makes testing/re-routing harder.

**The `try` block covers too much.**

```cpp
try {
    device::Resources resources = device::build_resources(config.device());
    // ... validation ...
    device::DeviceF2Blocks dev_f2 = device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
    result = run_f4(dev_f2, std::span<const std::array<int, 4>>(quartets), opts, resources);
} catch (const std::exception& e) {
    std::fprintf(stderr, "steppe f4: device error: %s\n", e.what());
    return cfg::kExitRuntimeError;
}
```

Lines 163-182 wrap build, upload, and compute in one catch-all. It's not wrong, but it collapses "no CUDA device", "OOM during upload", and "compute failure" into a single `kExitRuntimeError`. For debugging and telemetry, finer-grained stage handling would be better. The `resources.gpus.empty()` check inside the try is also a bit awkward — it's a domain condition, not an exception.

**Awkward label decomposition.**

```cpp
std::vector<std::string> l1, l2, l3, l4;
l1.reserve(quartet_names.size()); l2.reserve(quartet_names.size());
l3.reserve(quartet_names.size()); l4.reserve(quartet_names.size());
```

Lines 137-139 split a naturally 4-column structure into four separate vectors, then push into each in lockstep. A `std::vector<std::array<std::string, 4>>` (mirroring `quartet_names`) would be cleaner and harder to desynchronize.

**Minor loop/index style issues.**

In `build_quartet_names` (lines 71-73 and 90-93) and name resolution (lines 136-154), the code `reserve`s then `push_back`s in a loop. A senior might suggest `resize(n)` and index assignment, or `std::transform`, to avoid repeated growth checks and reserve/push asymmetry.

Also at line 142-149:

```cpp
for (int c = 0; c < 4; ++c) {
    const ResolveResult rr = resolver.resolve(q[static_cast<std::size_t>(c)]);
    idx[static_cast<std::size_t>(c)] = rr.index;
}
```

The `static_cast<std::size_t>` dance is unnecessary noise for a fixed 4-element array. Just loop with `std::size_t c = 0; c < 4; ++c`.

**`QpAdmOptions` used for f4 is mildly confusing.**

Line 161 calls `config.qpadm_options()` inside an f4 command. The comment at lines 159-160 explains it's only for the `fudge` default (0 for bare f4 vs. 1e-4 for qpAdm), but the name leakage is still a bit odd. Not a bug, but it hints that the options struct is overloaded.

**Output parsing/emitting logic is likely duplicated across commands.**

Lines 185-202 handle `--format` parsing, `--out` file opening, and `emit_f4_result`. This exact shape probably appears in `cmd_qpwave.cpp` and friends. A shared helper (`emit_to_output(config, result, labels)`) would cut the duplication and reduce the chance of one command mishandling the output path.

## The "slop" test

**Not slop.** Slop is magic numbers without explanation, copy-pasted code with stale comments, missing error handling, or algorithms that only pass tests by accident. This file has none of that. The comments are dense but accurate; the error messages are specific; the code reuses existing paths instead of duplicating them.

The closest thing to slop-adjacent is the mixed `fprintf`/`iostream` output, which feels like two different conventions collided rather than like thoughtless code.

## What it actually looks like

This looks like **solid app-layer CLI code written by someone who knows the genomics domain and the project architecture well.** The structure is clear, the validation is thorough, and the GPU layering is correct. It's the kind of file you'd be happy to see in a PR — correct, well-explained, and unlikely to break the build.

A senior C++ reviewer would say: "Competent, but standardize your I/O and consider a few cleanups." A senior CUDA reviewer wouldn't have much to say here because the CUDA details are properly hidden behind seams; that's actually a compliment to the design.

**Verdict:** B+ — correct and well-architected, but the stderr/stdout convention split and minor stylistic warts keep it from being pristine showcase code.
