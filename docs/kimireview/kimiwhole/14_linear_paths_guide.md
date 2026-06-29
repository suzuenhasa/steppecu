# Guide: Making Every Path Linear and Precise in `steppe`

This guide describes how to transform the current "bouncy" control flow into clean, start-to-finish pipelines. The goal is that a reader can follow any operation from entry to exit without zig-zagging between layers or holding more than 2–3 files in their head.

---

## 1. The principle

A linear path means:

- **One entry point** per operation.
- **One direction of travel** through layers (usually down: CLI → app → core → device).
- **No repeated decisions** along the way.
- **No hidden ordering** — every prerequisite is either enforced by types or explicit in the function signature.
- **No mixed abstraction** — a function either orchestrates or computes; it does not do both.

The current codebase violates this in three places:

1. CLI commands duplicate the same orchestration.
2. `core` does genotype I/O.
3. The device backend is one monolithic interface that every feature grows.

Fix those three and most paths become linear.

---

## 2. The canonical command path

### Current path (bouncy)

```
main.cpp
  → cli_parse.cpp (picks subcommand)
    → cmd_f4.cpp
      → read_f2_dir()          [app/io]
      → build_resources()      [device]
      → upload_f2_blocks()     [device]
      → run_f4()               [core]
        → primary_backend()    [device, via core]
        → assemble_f4()        [device]
        → jackknife_diag()     [device]
      → parse_output_format()  [app]
      → open out file          [app]
      → emit_f4_result()       [app]
```

Problems:

- `cmd_f4.cpp` is ~210 lines of orchestration repeated in every command.
- `run_f4()` in `core` has to know device seams (`assemble_f4`, `jackknife_diag`).
- Output formatting is duplicated and divergent.

### Linear path (target)

```
main.cpp
  → cli_parse.cpp
    → run_fit_command<F4Config>(cfg)   [app/command_runner.hpp]
      → F2Engine::open(cfg.f2_dir)     [f2]
      → ComputeContext ctx(resources, f2) [device]
      → F4::run(ctx, quartets)         [core — pure math]
      → OutputSink::emit(result)       [app/output]
```

The command-specific file (`cmd_f4.cpp`) becomes ~30 lines:

```cpp
int run_f4_command(const RunConfig& cfg) {
    auto quartets = parse_quartets(cfg);
    return run_fit_command(cfg, [&](const ComputeContext& ctx) {
        return steppe::f4::run(ctx.f2(), quartets, ctx.precision());
    });
}
```

---

## 3. Concrete refactor #1: the command runner

Create `src/app/command_runner.hpp`:

```cpp
#pragma once
#include <steppe/config.hpp>
#include <steppe/error.hpp>
#include "app/output_sink.hpp"
#include "device/resources.hpp"
#include "f2/f2_engine.hpp"

namespace steppe::app {

struct ComputeContext {
    device::Resources& resources;
    const f2::F2DirHandle& f2;
    Precision precision;
    int device_id;
};

// Generic orchestrator for any command that needs f2 blocks.
template<typename ComputeFn, typename EmitFn>
int run_f2_command(const RunConfig& cfg,
                   ComputeFn&& compute,
                   EmitFn&& emit) {
    // 1. Load f2.
    auto f2_or = f2::Engine::open(cfg.f2_dir());
    if (!f2_or) return exit_code_for(f2_or.error());

    // 2. Build device resources.
    auto resources_or = device::build_resources(cfg, *f2_or);
    if (!resources_or) return exit_code_for(resources_or.error());

    // 3. Upload f2 blocks (if needed).
    auto ctx_or = upload_f2_if_needed(*resources_or, *f2_or);
    if (!ctx_or) return exit_code_for(ctx_or.error());

    // 4. Compute.
    auto result = std::forward<ComputeFn>(compute)(*ctx_or);
    if (!result) return exit_code_for(result.error());

    // 5. Emit.
    OutputSink sink(cfg.out_file(), cfg.output_format());
    auto emit_err = std::forward<EmitFn>(emit)(sink, *result);
    if (emit_err) return exit_code_for(*emit_err);

    return kExitSuccess;
}

} // namespace steppe::app
```

What this removes:

- Duplicate `--f2-dir` validation.
- Duplicate `--out`/`stdout` opening.
- Duplicate no-GPU checks.
- Duplicate exception → exit-code mapping.

Every fit command becomes a thin adapter:

```cpp
// cmd_f4.cpp
int run_f4_command(const RunConfig& cfg) {
    auto quartets = resolve_quartets(cfg);
    return run_f2_command(cfg,
        [&](const ComputeContext& ctx) {
            return steppe::f4::run(ctx, quartets);
        },
        [&](OutputSink& sink, const F4Result& r) {
            return emit_f4_result(sink, r);
        });
}
```

---

## 4. Concrete refactor #2: the f2 engine

Create `src/f2/f2_engine.hpp` as the single authority for f2 lifecycle:

```cpp
#pragma once
#include <steppe/error.hpp>
#include <steppe/fstats.hpp>
#include <filesystem>
#include <memory>

namespace steppe::f2 {

struct F2DirHandle {
    std::filesystem::path dir;
    F2BlockTensor host_tensor;
    F2DirMeta meta;
};

struct F2CacheKey {
    std::string geno_hash;
    std::string ind_hash;
    std::string snp_hash;
    BlockPartitionRule partition;
    Precision precision;
    auto operator<=>(const F2CacheKey&) const = default;
};

class Engine {
public:
    // Open an existing f2 dir. Returns NotFound if missing.
    [[nodiscard]] static Result<F2DirHandle> open(std::filesystem::path dir);

    // Create an f2 dir from raw genotypes. Writes the STPF2BK1 bundle.
    [[nodiscard]] static Result<F2DirHandle> build(
        const GenotypeSource& source,
        const PopulationSelection& pops,
        const F2BuildOptions& opts);

    // Lazy open-or-build. Looks up by cache key; builds if missing.
    [[nodiscard]] static Result<F2DirHandle> fetch(
        const F2CacheKey& key,
        const GenotypeSource& source,
        const PopulationSelection& pops);

    // Validate that an f2 dir is compatible with a command's requirements.
    [[nodiscard]] static Status validate(const F2DirHandle& h,
                                         const F2Requirements& req);
};

} // namespace steppe::f2
```

What this gives you:

- One place decides whether f2 exists.
- One place builds f2.
- One place validates compatibility (population set, block parameters, precision).
- Commands no longer check `--f2-dir` manually.

Move these files into `src/f2/`:

- `src/app/f2_dir_io.cpp` → `src/f2/f2_dir_io.cpp`
- `src/app/f2_dir_writer.cpp` → `src/f2/f2_dir_writer.cpp`
- `src/app/extract_f2_core.cpp` → `src/f2/f2_builder.cpp`
- `src/core/stats/read_canonical_tile.cpp` → `src/f2/f2_builder.cpp` or `src/extract/`

`core` should depend on `f2::F2BlockTensor` only, not on `io`.

---

## 5. Concrete refactor #3: split the backend seam

Replace the 1,857-line `ComputeBackend` god interface with role interfaces:

```cpp
// src/device/decode_backend.hpp
struct DecodeBackend {
    virtual ~DecodeBackend() = default;
    virtual DeviceDecodeResult decode_af_compact_filter(...) = 0;
    virtual DeviceDecodeResult decode_af_compact_autosome(...) = 0;
};

// src/device/precompute_backend.hpp
struct PrecomputeBackend {
    virtual ~PrecomputeBackend() = default;
    virtual F2BlockTensor compute_f2_blocks(...) = 0;
    virtual DeviceF2Blocks upload_f2_blocks(...) = 0;
};

// src/device/fit_backend.hpp
struct FitBackend {
    virtual ~FitBackend() = default;
    virtual QpAdmResult fit_qpadm(...) = 0;
    virtual QpGraphResult fit_qpgraph(...) = 0;
    virtual F4Result assemble_f4(...) = 0;
    virtual F3Result assemble_f3(...) = 0;
};

// src/device/sweep_backend.hpp
struct SweepBackend {
    virtual ~SweepBackend() = default;
    virtual std::vector<QpAdmResult> fit_models_batched(...) = 0;
};
```

A concrete backend implements only the roles it cares about:

```cpp
class CudaBackend : public DecodeBackend,
                    public PrecomputeBackend,
                    public FitBackend,
                    public SweepBackend {
    // ...
};

class CpuBackend : public FitBackend {
    // CPU only needs to assemble stats from precomputed f2.
};
```

Then `core` depends on narrow interfaces, not the whole backend:

```cpp
// src/core/qpadm/f4.hpp
namespace steppe::f4 {
    Result<F4Result> run(const device::FitBackend& backend,
                         const F2BlockTensor& f2,
                         std::span<const Quartet> quartets);
}
```

What this removes:

- `core` no longer includes `backend.hpp`.
- Adding a new stat does not require touching the device seam unless it needs new hardware primitives.
- The backend monolith can be split into multiple `.cu` files (`cuda_decode.cu`, `cuda_f2.cu`, `cuda_fit.cu`, etc.).

---

## 6. Concrete refactor #4: the output sink

Create `src/app/output_sink.hpp`:

```cpp
#pragma once
#include <steppe/error.hpp>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

namespace steppe::app {

enum class OutputFormat { csv, tsv, json };

class OutputSink {
public:
    explicit OutputSink(std::optional<std::string> path,
                        OutputFormat fmt);

    std::ostream& stream();
    [[nodiscard]] Status finalize();  // flush + check

    OutputFormat format() const { return fmt_; }

private:
    OutputFormat fmt_;
    std::optional<std::ofstream> file_;
    std::ostream* os_;
};

// Format primitives
[[nodiscard]] std::string fmt_double(double v, int precision = 17);
[[nodiscard]] std::string csv_escape(std::string_view s);
[[nodiscard]] std::string json_escape(std::string_view s);

class CsvRowWriter {
public:
    explicit CsvRowWriter(std::ostream& os, char sep = '\t');
    void header(std::initializer_list<std::string_view> cols);
    void row(std::initializer_list<std::string_view> cells);
    void row_double(std::initializer_list<double> cells);
private:
    std::ostream& os_;
    char sep_;
};

class JsonWriter {
public:
    explicit JsonWriter(std::ostream& os);
    void begin_object();
    void end_object();
    void field(std::string_view key, std::string_view value);
    void field(std::string_view key, double value);
    // ...
private:
    std::ostream& os_;
    bool first_ = true;
};

} // namespace steppe::app
```

Then every command emits through it:

```cpp
Status emit_f4_result(OutputSink& sink, const F4Result& r) {
    if (sink.format() == OutputFormat::csv) {
        CsvRowWriter w(sink.stream());
        w.header({"f4", "se", "z", "p"});
        for (const auto& v : r.values) {
            w.row_double({v.est, v.se, v.z, v.p});
        }
    }
    return sink.finalize();
}
```

What this removes:

- 12 copies of the `--out`/`stdout` block.
- Divergent number formatting.
- Hand-rolled JSON with incomplete escaping.
- Silent write failures.

---

## 7. Concrete refactor #5: one error type, one exit-code mapping

Create `src/core/error.hpp`:

```cpp
#pragma once
#include <steppe/error.hpp>
#include <string>

namespace steppe {

struct Error {
    Status status;
    std::string message;

    static Error io(std::string msg) {
        return {Status::IoError, std::move(msg)};
    }
    static Error cuda(cudaError_t err, std::string msg) {
        return {Status::CudaRuntime, std::move(msg) + ": " + cudaGetErrorString(err)};
    }
    static Error invalid(std::string msg) {
        return {Status::InvalidConfig, std::move(msg)};
    }
};

template<typename T>
using Result = std::expected<T, Error>;

} // namespace steppe
```

Then every library call returns `Result<T>`:

```cpp
Result<F2DirHandle> Engine::open(std::filesystem::path dir);
Result<F4Result> f4::run(const ComputeContext& ctx, ...);
```

And the CLI maps it once:

```cpp
// src/app/main.cpp
cfg::ExitCode exit_code_for(const steppe::Error& e) {
    switch (e.status) {
        case Status::Ok:             return kExitSuccess;
        case Status::InvalidConfig:  return kExitInvalidConfig;
        case Status::IoError:        return kExitIoError;
        case Status::CudaRuntime:    return kExitCudaError;
        case Status::DeviceOom:      return kExitDeviceOom;
        case Status::NotImplemented: return kExitNotImplemented;
        default:                     return kExitRuntimeError;
    }
}
```

What this removes:

- `try/catch` control flow in library code.
- Swallowed `std::runtime_error`.
- Wrong exit codes for disk-full or OOM.
- Lossy exception → status conversion.

---

## 8. How the paths look after these refactors

### f4 path

```
main
  → cli_parse
    → cmd_f4::run (parse quartets)
      → run_f2_command (generic)
        → F2Engine::open
        → build_resources
        → upload_f2_if_needed
        → f4::run (core, pure math)
        → OutputSink::emit
```

### extract-f2 path

```
main
  → cli_parse
    → cmd_extract_f2::run (parse source + pops)
      → F2Engine::build
        → GenoReader (io)
        → decode backend
        → assign blocks
        → precompute backend
        → f2_dir_writer
```

### qpadm rotation path

```
main
  → cli_parse
    → cmd_rotate::run (parse model list)
      → run_f2_command (generic)
        → F2Engine::open
        → build_resources (one ctx per GPU)
        → qpadm::search (core)
          → per-device precompute OR shared f2
          → sweep backend
        → OutputSink::emit
```

Each path is:

- 5–7 steps.
- Mostly downward through layers.
- Reuses the same generic orchestrator.
- Returns a typed error if anything fails.

---

## 9. The "do not do this" list

To keep paths linear, avoid:

- **Callback into higher layers.** `device/backend.hpp` should not mention `core::qpadm`.
- **Functions that both decide and compute.** Split `run_extract_f2_command` into validation, planning, execution, and output phases.
- **Repeated conditionals.** If you find yourself writing `if (out.empty()) std::cout ...` more than once, make a helper.
- **Implicit ordering hidden in comments.** If two calls must happen in sequence on the same object, encode it in a type (`F2Uploader` returns `DeviceF2Blocks`, which can only be obtained by uploading).
- **Mixed I/O in core.** `core` should receive `F2BlockTensor`, not open `.geno` files.

---

## 10. Suggested order of attack

1. **OutputSink** — smallest change, biggest visible payoff. Every command becomes shorter.
2. **Command runner** — removes the duplicate 5-step dance.
3. **Status/Result** — stops the exception/status schizophrenia.
4. **F2Engine** — centralizes f2 lifecycle and removes `core` → `io` dependency.
5. **Split backend seam** — the largest change, but unlocks long-term maintainability.

Do steps 1–3 first. They are low-risk and immediately make the code feel more linear. Steps 4–5 are structural and should follow.
