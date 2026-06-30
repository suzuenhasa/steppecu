# steppe — generated API reference

Two opt-in, behavior-neutral documentation targets, both behind the CMake option
`STEPPE_BUILD_DOCS` (default `OFF`; `cmake/Docs.cmake`). Neither defines a library
target or touches the compute/precision path — they only render the docstrings that
already live in the headers and the Python facade.

| Target        | Tool    | Input                | Output                            |
| ------------- | ------- | -------------------- | --------------------------------- |
| `docs`        | Doxygen | `include/steppe/*.hpp` | `<build>/docs/api/html/index.html` |
| `docs-python` | pdoc    | `bindings/steppe`    | `<build>/docs/api/python/`        |

## C++ (`docs`) — runs locally, no CUDA

**Key property:** Doxygen parses the public headers as *text* — it needs **no compiler
and no CUDA toolkit**. So, unlike every other target in this project (which requires a
Blackwell `sm_120` box), the `docs` target builds on the **local dev box**.

```sh
cmake -S . -B build -DSTEPPE_BUILD_DOCS=ON
cmake --build build --target docs
# open build/docs/api/html/index.html  (Precision, run_qpadm, QpAdmResult, F2BlockTensor, ...)
```

`docs/Doxyfile.in` is a template: CMake substitutes `PROJECT_NAME` / `PROJECT_NUMBER`
(single-sourced from the top-level `project(VERSION ...)`, the same source the wheel
version and the CLI/extract `STEPPE_VERSION` use) and the input/output paths via
`configure_file(... @ONLY)`. Override the HTML location with
`-DSTEPPE_DOXYGEN_OUTPUT_DIR=<dir>` (e.g. point it at `docs/api` to commit the HTML).

If Doxygen is not installed the target is a **soft no-op** (`find_package(Doxygen QUIET)`),
so a normal build is never affected.

## Python (`docs-python`) — needs the built extension

```sh
cmake --build build --target docs-python
# -> build/docs/api/python/  (read_f2, qpadm, dates, QpAdmResult.weights, ...)
```

`docs-python` runs `pdoc -o <out> bindings/steppe` over the facade. Unlike Doxygen, pdoc
**imports** the package, and `steppe/__init__.py` imports the compiled `_core` extension —
so run this where a built `_core` is importable (on the box, or after a wheel build), not
on the no-CUDA local box. The target is a soft no-op when `pdoc` is not on `PATH`.

The heavier Sphinx + Breathe path (architecture.md §6) is a deferred A+ follow-on and is
not wired here.
