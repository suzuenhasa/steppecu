# Code review: `src/app/cmd_qpdstat.cpp`

This is **not slop**, but it's also not a file you'd hold up as your cleanest work. The author clearly understands the domain (D-statistics, f2-path parity, AT2 conventions) and makes disciplined reuse of existing compute paths. What hurts it is duplication that has been copy-pasted rather than abstracted, and a style that oscillates between modern C++ and C-style `fprintf` logging.

## What's genuinely good

- **Strong domain reasoning in comments.** The header comment (lines 1–22) is unusually precise about why `qpdstat` on the f2 path equals `f4`, why `--prefix` is Part B, and what the GPU seams are. That kind of context is valuable in genomics compute code.
- **Reuses the hard stuff rather than reimplementing it.** `run_f4`, `run_dstat`, `emit_f4_result`, `upload_f2_blocks_to_device`, and `run_fstat_sweep` are all called instead of duplicated. The `DstatResult` → `F4Result` shim (lines 218–221) is a pragmatic way to avoid writing a second emitter.
- **Correct separation of domain outcomes vs. faults.** The code returns `exit_code_for(result.status)` for domain results (exit 0 even when the math fails) and returns nonzero only for I/O, config, or runtime faults. The comments cite the CLI bindings doc, which shows architectural discipline.
- **Modern C++ where it matters.** Uses `std::span`, `std::array`, `std::vector::reserve`, and `std::find` appropriately. No raw owning pointers or manual memory management.
- **The branching structure is clear.** `--prefix` Part B, `--all-quartets` sweep, and the default f2-dir path are cleanly separated at lines 244, 251, and 256.

## What a senior developer would flag

**`fprintf(stderr, ...)` mixed with `std::cout`/`std::ofstream`.** The file uses iostreams for normal output (`std::cout`, `std::ofstream`) but C-style `fprintf` for all error logging:

```cpp
129: std::fprintf(stderr, "steppe qpdstat: %s\n", qerr.c_str());
159: std::fprintf(stderr, "steppe qpdstat: input error: %s\n", e.what());
228: std::fprintf(stderr, "steppe qpdstat: cannot open --out file: %s\n",
                 config.out_file().c_str());
```

In a C++20 project this is a convention mismatch. Pick one family and stick to it; if the architecture says `main()` owns stdout/stderr, that doesn't require `fprintf`. A small helper like `log_error(config, fmt, ...)` would also remove the repeated `"steppe qpdstat: "` prefix boilerplate.

**Heavy duplication between the `--prefix` path and the f2-dir path.** The same quartet-name building, resolver validation, index resolution, and label-backfill sequence appears twice (lines 125–187 vs. 266–302), with only minor differences:

- The f2-dir branch reserves `l1..l4` (line 285); the prefix branch doesn't.
- The prefix branch uses `pop_labels` from `read_ind_partition`; the f2-dir branch uses `dir.dir.pop_labels`.
- Otherwise the loops are nearly identical.

A senior reviewer would ask: why isn't there a `resolve_quartets(quartet_names, pop_labels, ...)` helper shared with `cmd_f4.cpp`? The header comment even admits the builder mirrors `cmd_f4.cpp`'s `build_quartet_names`.

**`std::find` linear scan to build the population union (line 142):**

```cpp
141: for (const std::string& nm : q) {
142:     if (std::find(pop_union.begin(), pop_union.end(), nm) == pop_union.end())
143:         pop_union.push_back(nm);
144: }
```

This is O(P × Q) and, while the populations are small, it's the wrong tool. A `std::unordered_set` (or sort + unique) is the idiomatic choice and makes intent obvious. The comment says "any order; read_ind sorts" — but the code itself doesn't make that clear.

**Magic number `4` everywhere.** Quartet size is hard-coded repeatedly:

```cpp
64:  std::vector<std::array<std::string, 4>>& quartets,
95:  if (pops.size() % 4 != 0) {
103: quartets.push_back({pops[4 * k + 0], pops[4 * k + 1],
104:                     pops[4 * k + 2], pops[4 * k + 3]});
289: for (int c = 0; c < 4; ++c) {
```

A named constant (`constexpr std::size_t kQuartetSize = 4;`) would document the invariant and make the code easier to change if a different statistic ever needed a similar shape.

**`parse_output_format` is duplicated across both branches (lines 213–217 and 334–338).** Exactly the same logic, same error message. This should happen once at the top of `run_qpdstat_command` before branching.

**`QpAdmOptions opts = config.qpadm_options()` (line 309) is semantically odd in an f4/qpdstat path.** The name leaks a different command's config struct into this file. The comment acknowledges it uses the struct default, but a reader will do a double-take. A more generic name like `F4Options` would reduce confusion.

**Output file opened with `std::ios::binary` (lines 226 and 344).** The emitter writes CSV/TSV/JSON — text formats. Opening in binary mode works but is incongruous and can cause line-ending surprises if the emitter ever writes `\n` on Windows. If the intent is to avoid CRLF translation, that's worth a comment.

**The shim copy is field-by-field (lines 218–221):**

```cpp
218: F4Result shim;
219: shim.p1 = result.p1; shim.p2 = result.p2; shim.p3 = result.p3; shim.p4 = result.p4;
220: shim.est = result.est; shim.se = result.se; shim.z = result.z; shim.p = result.p;
221: shim.status = result.status; shim.precision_tag = result.precision_tag;
```

This is fine, but if the two structs are intentionally mirrors, a constructor or conversion operator would be safer against future field additions. The current code is a maintenance footgun: add a field to `DstatResult` and the emitter may silently get a default value.

**Stale / misleading comments.** Line 8 says the file differs "ONLY in ... (c) the not-yet-impl message wording." The message *is* implemented. The header also calls this a "THIN wrapper," but the file is 358 lines and contains substantial logic — not thin by any reasonable definition.

**No validation that `read_ind_partition` actually found all requested populations in the `--prefix` branch.** If a population is missing, the resolver will fail later with a generic message, but an earlier explicit check would give a better user-facing error.

## The "slop" test

**Not slop.** The code has:
- Error handling on every I/O and config path.
- Clear comments explaining *why*, not just *what*.
- No obviously wrong algorithms.
- Deliberate reuse of compute kernels and emitters.

The duplication and `fprintf` usage are maintenance problems, not sloppiness. This is the work of someone who knows the domain and shipped a working feature.

## What it actually looks like

This looks like **competent, deadline-driven command plumbing** written by a developer who deeply understands the genomics semantics but is less concerned with C++ elegance. It prioritizes correctness and reuse over abstraction: the hard compute paths are delegated, but the glue between them is copy-pasted and patched rather than factored into shared helpers. A reviewer gets the sense that `cmd_f4.cpp` was open in one pane and this file was typed in the other.

A senior C++ reviewer would say: "Solid logic, ship after you centralize the error logging and deduplicate the quartet resolution." A domain reviewer would be happy; a style reviewer would wince at the `fprintf` and magic numbers.

## Verdict

**B.** Genuinely good domain work and reuse, but the duplication, `fprintf`/iostream mix, and magic-number quartet size keep it out of the A tier. In a job-application showcase, this file would impress with its correctness and domain reasoning, then undermine itself with copy-paste drift.
