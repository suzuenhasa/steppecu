I read `/home/suzunik/steppe/src/app/cli_parse.cpp` carefully and compared it to the tone in `docs/kimiexample.md`. This file is **not slop**, but it's also not pristine senior C++. It's competent, well-organized wiring code with a few control-flow and idiom rough edges that would make a senior reviewer pause.

## What's genuinely good

- **Strong architectural discipline.** The top comment (lines 1–13) is explicit: CLI11 lives *only* here, this TU is pure host, no CUDA. That's a clean layering decision and the file respects it.
- **Flag deduplication via helpers.** `add_common_flags`, `add_output_flags`, `add_qpadm_option_flags`, `add_f2_dir_flag`, etc. show the author actually looked for shared surface area instead of copy-pasting every subcommand. The parameterization of help strings to keep `--help` byte-identical across callers (lines 145–152, 163–170) is careful, unglamorous work.
- **Modern C++ structure where it matters.** `build_config` returns `std::optional<RunConfig>` (line 60), CLI11 lambdas capture stable stack storage, and the subcommand `CliArgs` instances are declared up front so the bound callbacks don't dangle (lines 323–336).
- **CLI11 features used correctly.** The paired-flag syntax `--allow-neg,!--no-allow-neg` (line 117), `->delimiter(',')` on vector options (line 169), and `require_subcommand(0, 1)` (line 318) are idiomatic.
- **Callback pattern is consistent.** Every subcommand does the same `build_config → nullopt exits kExitInvalidConfig → run_*_command → std::exit` dance. Easy to read once you see the first one.

## What a senior developer would flag

**`std::exit` everywhere instead of returning codes through the call stack.**

Lines 353, 375, 411, 439, 458, 484, 525, 552, 573, 593, 610, 640, 667, 743, 744 all call `std::exit(...)`. This is a C-style exit-in-the-middle pattern. It mostly works here because `run_cli` is the top-level host function and subcommand callbacks own the terminal action, but it means RAII cleanup in `main()` or a surrounding scope is bypassed. A senior C++ reviewer would prefer the callbacks return an `int` or `std::variant<RunConfig, ExitCode>` and let `run_cli` return naturally. The comment even notes CLI11 callbacks "std::exit before reaching here" as if that's a feature — it's a constraint the author imposed on themselves.

**Inconsistent output/error channel idioms.**

- Line 71: `std::fprintf(stderr, ...)`
- Line 753: `std::printf(...)`

Both are C stdio inside a C++ file that already includes `<string>` and `<vector>` and uses CLI11. There's no reason not to use `std::cerr << ...` and `std::cout << ...`, or at least route through a single project outputter. The file's own comment says "print the builder's reason to stderr" but then does it with `fprintf`.

**`build_config` violates separation of concerns.**

Line 60–76 prints the error message directly to `stderr` inside the config-building helper:

```cpp
auto result = builder.build();
if (!result.has_value()) {
    std::fprintf(stderr, "steppe: invalid configuration: %s\n",
                 builder.error_message().c_str());
    return std::nullopt;
}
```

A cleaner design returns the error string and lets the caller decide how to render it. Here the caller doesn't know *why* the build failed except by side effect. That makes testing harder and couples config construction to stderr.

**Redundant `std::move`.**

Line 75:

```cpp
return std::move(result.value());
```

Returning `result.value()` from a local `std::optional` already triggers move elision or a move for the contained type. The `std::move` is not wrong, but it's the kind of thing a senior reviewer marks with "minor pessimization / code smell" in a nit round.

**Inconsistent error handling for the `--ploidy` callback.**

Line 711–715:

```cpp
else throw CLI::ValidationError(
    "--ploidy", "must be auto, 1 (pseudo-haploid), or 2 (diploid); got '" + v + "'");
```

Everything else defers validation to `ConfigBuilder::build()`. Throwing a CLI11 exception here works, but it breaks the uniform "parse first, validate in build_config" model without explanation. A senior would ask: why is this flag special?

**Stale/copy-paste drift in comments.**

Line 354–356:

```cpp
// The real GPU qpAdm fit [...] qpadm-rotate + extract-f2 are likewise wired; only
// qpwave (M(cli-2)) remains a scaffold no-op.
```

But lines 446–464 show `qpwave` is fully wired to `run_qpwave_command`. The comment is stale scaffolding from an earlier milestone. Several callback end-comments also repeat "Mirrors how `qpadm` dispatches" verbatim across half a dozen subcommands; functional but lazy.

**Comment density is high even for simple bindings.**

Lines 145–152 explain why `add_f2_dir_flag` takes a `const char* help`, and similar justifications appear repeatedly. The intent is good (preserve `--help` byte identity), but after the third helper the noise-to-signal ratio drops. A senior reviewer would say: trust the code, or factor these into a tiny schema table instead of twenty near-identical lambdas.

**Minor: `CLI11_PARSE` macro mixed with manual `std::exit`.**

Line 747:

```cpp
CLI11_PARSE(app, argc, argv);
```

`CLI11_PARSE` prints errors and exits on parse failure. Then every subcommand callback also exits. Then the fall-through prints help and returns normally. This triple-mode exit flow is readable but brittle: if someone later adds code after `CLI11_PARSE` expecting to run after a subcommand, it won't.

## CUDA-specific footguns

None in this file — and that's correct. This is a pure host TU. The header comment enforces it and the implementation honors it. No raw `cudaMalloc`, no grid/occupancy logic, no stream handling. So no demerits here; the file stays in its lane.

## Memory management / ownership

Clean. The `CLI::App` is owned by CLI11. `CliArgs` instances are stack-allocated and stable for the lifetime of the bound lambdas. No raw owning pointers, no leaks visible. The helpers take `CLI::App*` because that's what `add_subcommand` returns, which is reasonable.

## API design

The public surface is just `run_cli(int, char**)`, which is good. However, the file pulls in 13 command headers (lines 23–34) just to dispatch to `run_*_command` functions. That's a lot of coupling for a dispatcher. If this were my review, I'd suggest a tiny command registry map (`Command` enum → function pointer) to avoid the header explosion, but for a research CLI this is acceptable.

## The "slop" test

**Not slop.** The code is intentional, tested-looking, and free of magic numbers without explanation, copy-pasted broken logic, or unchecked fallbacks. The verbose comments sometimes overshoot, but they explain *why* decisions were made, not just restate the code.

## What it actually looks like

This looks like **competent research-engineering CLI wiring by someone who understands the domain and the library.** It's methodical, consistent, and respectful of the project layering. The C-idiom exits and C-style I/O are the main tells that the author is more comfortable with systems/C-style code than modern C++ shop conventions. A senior C++ developer would have no trouble reading it but would push back on the control-flow design.

## Verdict

**B+, ship after tightening control flow and error reporting.** The flag bindings and subcommand structure are solid. Fix the `std::exit` soup, move error rendering out of `build_config`, and scrub the stale "scaffold no-op" comment. The rest is polish.