I read through this carefully. This is **not slop** — it's clearly written by someone who understands the architecture and the domain problem (don't materialize a multi-terabyte C(P,k) table, keep the survivors on the GPU). But a senior C++ reviewer would flag a handful of stylistic inconsistencies and a couple of real correctness/robustness issues that would keep it out of the "showcase" tier without cleanup.

## What's genuinely good

- **The overall flow is sane and well-layered.** `read_f2_dir` → `PopResolver` → `build_resources` → `upload_f2_blocks_to_device` → `run_f4_sweep`/`run_f3_sweep` → `emit_sweep`. The file is doing exactly one thing: orchestrating a GPU-backed sweep. No CUDA headers leak in, which respects the layering comment.
- **Resource management is mostly RAII.** `device::Resources`, `DeviceF2Blocks`, and the `std::ofstream` destructors do the cleanup. The device upload is inside a `try`/`catch` that translates exceptions into a clean stderr message and a proper exit code.
- **The design intent is documented honestly.** The comments about O(K) host memory, the reservoir on the device, and why `--top-k` is the default for bare `--all-combinations` make it clear the author understands the memory cliff they're avoiding.
- **Output routing is cleanly prioritized.** `--shard-dir` → `--out` → stdout is a sensible precedence, and the shard filename (`survivors.<ext>`) is a reasonable convention.

## What a senior developer would flag

**Mixing `std::ostream` with `std::fprintf` everywhere.** The file opens with the claim that "main() owns stdout/stderr," but then the implementation spews directly to `stderr` with `fprintf` at lines 80, 85, 91, 100, 134, 145, 151, 160, 173, 184, 190, 200, 208:

```cpp
std::fprintf(stderr, "steppe %s: --f2-dir is required\n", cmd);
```

At the same time, the data path uses a `std::ostream&`. A senior reviewer would ask: why is the error/logging path not going through the same stream abstraction or a small `log_error()` helper? The repeated `"steppe %s: ..."` prefix is ad-hoc copy-paste boilerplate.

**Hand-rolled JSON/CSV/TSV with no escaping.** `emit_sweep` writes labels directly into the output:

```cpp
os << "\"pop" << (c + 1) << "\":\""
   << resolver.label_at(r.keys[i][static_cast<std::size_t>(c)]) << "\",";
```

If a population label contains a quote, comma, tab, or newline, the output is malformed. For CSV/TSV this is a real footgun with user-provided labels; for JSON it's a correctness bug. Either use a real serializer or at least escape the labels.

**Stale top-of-file comment about capped sweeps.** Line 14 says:

> A capped/empty sweep is a clean exit (record-and-continue); only device/IO faults return nonzero.

But lines 149–156 treat a capped sweep as a hard failure:

```cpp
if (result.capped) {
    std::fprintf(stderr, ..."refusing to enumerate %zu combinations..."...);
    return cfg::kExitInvalidConfig;
}
```

The later inline comment ("A CAPPED sweep is a hard, actionable refusal") matches the code, so the header comment is just stale. Sloppy.

**Raw `int k` as an arity selector with no validation.** `run_fstat_sweep` accepts `int k` and branches on `k == 4`, otherwise it assumes f3:

```cpp
result = (k == 4) ? run_f4_sweep(dev_f2, req, resources)
                  : run_f3_sweep(dev_f2, req, resources);
```

The only callers pass 4 and 3, so it's safe in practice, but a senior would prefer an `enum class Arity { F3, F4 }` or at least an `assert(k == 3 || k == 4)` to make the contract explicit. Right now a bad caller silently gets the f3 path.

**`std::ios::binary` for text output.** Lines 182 and 198 open CSV/TSV/JSON files with `std::ios::binary | std::ios::trunc`. Binary mode avoids newline translation, which is fine on Linux but unusual for textual formats and will produce LF-only files even on Windows. More importantly, it signals that the author may not be distinguishing text from binary output.

**Trusting vector sizes match `r.survivors`.** `emit_sweep` loops over `r.survivors` and indexes into `r.keys`, `r.est`, `r.se`, `r.z`, `r.p` without checking that the vectors are the same size. The device layer is supposed to enforce this, but the app layer is the seam where that invariant should be asserted or at least commented.

**Minor C++20 nitpicks:**
- `const char* cmd` could be `std::string_view`.
- `OutputFormat fmt = OutputFormat::Csv; if (!parse_output_format(...))` is fine, but the default assignment before parsing is unnecessary; an `std::optional`-returning parse would be cleaner.
- `<ostream>` is included alongside `<iostream>`; one of them is redundant.

## The "slop" test

**Not slop.** The code is structured, the error paths are handled, the comments explain *why* decisions were made, and there's no copy-pasted duplication with stale comments or magic numbers. The worst sins here are stylistic inconsistency and a hand-rolled serializer that doesn't escape strings — not algorithmic negligence.

## What it actually looks like

This looks like **competent research/engineering glue code written by someone who knows the domain and the project architecture.** It correctly avoids the big mistakes: no full combinatorial materialization, RAII for device resources, centralized output formatting, and sensible exit codes. But it also has the rough edges of code that was written to get a feature working rather than to be a public example: mixed C and C++ I/O style, a homegrown JSON/CSV emitter that skips escaping, and a stale comment that contradicts the actual behavior.

A senior reviewer would say: "Solid logic and good layering — but clean up the stderr boilerplate, fix the escaping, and delete the stale comment before I'd point an intern at this as a model."

**Verdict:** Respectable production code with fixable rough edges. **B** — would be B+ after addressing the escaping, unifying the I/O style, and correcting the stale comment.
