I read through this carefully. This is **not slop** — it's clearly written by someone who understands the project architecture and the qpWave/qpadm relationship. But a senior developer would flag some real convention drift and a few sloppy presentation details that undercut an otherwise competent file.

## What's genuinely good

- **The architecture comments at the top are excellent.** The header block explains exactly what this command is (rank/cladality test), how it differs from `cmd_qpadm.cpp` (no target; left[0] is the reference), and that the f2-dir load/upload chain is reused rather than duplicated. That shows system-level thinking.
- **Proper CUDA-free layering.** The code never includes a CUDA header; it reaches the GPU only through `device::build_resources`, `device::upload_f2_blocks_to_device`, and `steppe::run_qpwave`. That matches the project's stated §4 layering policy.
- **Clear fault-vs-domain distinction.** The try/catch block around the GPU pipeline is explicitly for faults (nonzero exit), while `result.status` is handled by `exit_code_for` and returns 0 for domain outcomes like `RankDeficient` or `NonSpd`. The comments explicitly call this out, and the code honors it.
- **Defensive invariant for `right_n`.** The `STEPPE_ASSERT(!right_idx.empty(), ...)` before `right_idx.size() - 1` is good practice — it documents the R0 convention locally instead of relying on a config check 50 lines away.
- **Modern C++ touches.** Using `std::span<const int>` to pass index lists and `std::vector<std::string>` with `reserve`/`push_back` shows competent, non-C-ish C++.

## What a senior developer would flag

**Mixed output conventions: `fprintf(stderr, ...)` alongside `std::cout` and `std::ofstream`.**

```cpp
std::fprintf(stderr, "steppe qpwave: --f2-dir is required\n");
// ...
emit_qpwave_result(std::cout, fmt, result, left_labels, right_n);
// ...
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
```

The project claims "main() owns stdout/stderr" and "the library never prints," but this file *is* the command implementation and it uses C-style `fprintf` for errors and C++ streams for output. A senior reviewer would ask: pick one error channel style. Either use `std::cerr` consistently or use `FILE*` consistently. The `fprintf`-to-`stderr` + `std::cout` mix looks like half a migration.

**Inconsistent error-handling formatting at lines 81–83:**

```cpp
if (!l.ok) { std::fprintf(stderr, "steppe qpwave: %s\n", l.error.c_str()); return cfg::kExitInvalidConfig; }
if (!r.ok) { std::fprintf(stderr, "steppe qpwave: %s\n", r.error.c_str()); return cfg::kExitInvalidConfig; }
```

These are one-liners while every other validation block in the same function is braced across multiple lines. That's pure presentation sloppiness — copy-paste drift or a late edit that didn't match the surrounding style. Lines 57–60 and 67–79 look professional; 81–83 look rushed.

**Error-code taxonomy is inconsistent.**

Line 58 maps a failed `read_f2_dir` to `kExitIoError`. Line 68 maps a failed `PopResolver` (also reading pop labels from the same directory) to `kExitIoError`. But line 81 maps a failed `resolve_all` on `--left` to `kExitInvalidConfig`, even though a missing/duplicate population name is arguably a user input error — fine — *except* the same failure on `--right` at line 83 also returns `kExitInvalidConfig`. That's actually consistent between left/right, but it's inconsistent with the directory-read errors. A reviewer would want a brief rationale: why is a bad f2-dir an I/O error but a bad population name a config error? The distinction exists, but it's not obvious from the code.

**`parse_output_format` treated as defensive rather than authoritative.**

```cpp
if (!parse_output_format(config.format(), fmt)) {
    // ConfigBuilder::build() already validates --format, so this is defensive.
    std::fprintf(stderr, ...);
    return cfg::kExitInvalidConfig;
}
```

A senior reviewer would ask: if the config builder already validated it, why does this function need the fallback at all? If the answer is "belt and suspenders," that's okay, but it also hints that `RunConfig` might be carrying around invalid state. Either trust your config type or fix the type's contract.

**Stale/cross-file comment at line 131:**

```cpp
// nr convention: right[0] == R0, so right_n == right.size()-1 (== metadata.nr).
```

`metadata.nr` is not visible in this file. The parenthetical refers to a data structure from `cmd_qpadm.cpp` or the engine internals. It's not wrong, but it's a copy-paste artifact that makes the reader look for something that isn't here.

**The `QpAdmOptions opts = config.qpadm_options();` line.**

Line 93 is correct but the comment "fudge / rank_alpha drive qpWave" is terse to the point of being unhelpful. It assumes the reader knows what `fudge` and `rank_alpha` mean. The rest of the file explains domain conventions carefully; this line drops the ball.

## The "slop" test

**Not slop.** Slop is unmarked magic numbers, duplicated logic with stale comments, unchecked allocations, or algorithms that only work by accident. This file has none of that. The logic is straightforward, the comments are mostly accurate, and the architecture is clean. The issues above are polish and convention, not competence.

## What it actually looks like

This looks like **solid production CLI glue written by someone who understands the genomics domain and the project's layering rules, but who hasn't fully enforced a single house style on themselves.** The hard parts — CUDA-free seams, fault-vs-domain handling, rank-sweep semantics — are handled well. The soft parts — brace style, error-channel consistency, cross-file comment leakage — are what you'd expect from a mid-level developer shipping a working command rather than a senior tightening up a showcase file.

A senior reviewer would probably say: "Correct and well-architected, but run `clang-format`, settle on streams vs. `fprintf`, and delete the `metadata.nr` ghost reference before we merge."

## Verdict

**B+.** Competent, shipable code with a few presentation warts that would not impress in a job-application showcase unless cleaned up.
