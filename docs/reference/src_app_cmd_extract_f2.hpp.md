# `cmd_extract_f2.hpp` reference

## 1. Purpose

`src/app/cmd_extract_f2.hpp` declares the single entry point for the
`steppe extract-f2` command. That command is steppe's one-time precompute step:
it reads raw genotype data and turns it into a directory of **f2 blocks** — the
per-genome-block building blocks that every f-statistic and model fit is later
computed from. The header itself is tiny (one function declaration); its value
is the contract and the pipeline that declaration stands for.

The reason this is its own command is a compute-once / fit-many split. Scanning
the genotypes and building the f2 blocks is the expensive part, so steppe does it
once and writes the result to a directory. The model-fitting commands then read
that directory back many times, cheaply, without ever touching the genotype files
again — the same precompute-and-cache f2 workflow steppe reproduces[^at2].

---

## 2. Where this command sits in the layering

steppe's code is split into layers, and there is a strict rule about which layer
may talk to which. The input/output code (readers for the various genotype
formats) and the GPU compute code are kept apart; the small application layer is
the **only** layer permitted to wire the io code into the compute code.

`extract-f2` is the one place in the whole codebase where that wiring actually
happens — it is the sole io→compute seam. Everything else either reads files or
computes on the GPU, but only this command hands freshly read genotypes into the
GPU pipeline.

### No CUDA in this header

This is plain C++20 and deliberately pulls in no CUDA header. The GPU is reached
only through CUDA-free seams:

- the resource-building step that stands up the GPU context,
- the backend's decode step that turns genotypes into allele frequencies through
  those resources,
- the multi-GPU f2-blocks compute entry point.

Keeping CUDA out of this header lets the application layer be included and built
without dragging in the GPU toolchain, and an automated source check enforces that
no CUDA header leaks in here.

### GPU-only

There is no CPU fallback. A machine with no visible CUDA device does not silently
compute on the CPU — it surfaces a clear "no CUDA device" fault and stops. (The
CPU code that exists elsewhere in steppe is a test-and-parity oracle, never a
user-facing runtime.)

---

## 3. What the command does

Running `extract-f2` executes a fixed pipeline:

1. **Up-front sizing and validation reads.** Before committing to any GPU work,
   it reads just enough of the dataset to resolve the problem's dimensions and to
   validate the request — that the named populations exist and the referenced
   files are present. Bad inputs fail here, early, before anything is allocated.
2. **Delegates the genotype → f2-blocks chain** to the library entry point
   `steppe::run_extract_f2`. That chain is: decode the genotypes → apply the
   quality-control filters → assign SNPs to genome blocks → compute the f2 values
   per block. The command drives it but does not reimplement it.
3. **Writes the result** using the f2-blocks directory writer, producing the
   output `<dir>`.
4. That `<dir>` is exactly what the fit commands consume — e.g. `steppe qpadm`
   pointed at it with `--f2-dir`.

The command owns its own standard-output and standard-error writing (progress and
diagnostics); the process's top-level `main` does not print on its behalf.

---

## 4. `run_extract_f2_command`

```cpp
[[nodiscard]] int run_extract_f2_command(const config::RunConfig& config);
```

The command's whole public surface is this one function.

| Aspect | Detail |
|---|---|
| **Input** | A `steppe::config::RunConfig` — the already-parsed, frozen run configuration. The function does not parse arguments itself; it acts on a settled config. |
| **Returns** | The process exit code (a `steppe::config::CliExitCode`, returned as `int`). A clean extract returns `0`. Any fault returns a nonzero code. |
| **`[[nodiscard]]`** | The return value is the process exit code, so the caller must not discard it — the attribute makes ignoring it a compile-time warning. |

### Fault behavior

Faults return a nonzero exit code rather than throwing out of the command. The
documented failure sources are:

- **InvalidConfig** — a bad population name or a missing input file (caught during
  the up-front validation reads).
- **DeviceOom** — the GPU ran out of memory for the computation.
- **File, format, or CUDA-runtime errors** — anything raised while reading the
  data or running the GPU pipeline.

### `--dry-run`

With `--dry-run`, the command does **no** compute. It performs only the resolution
step and reports the resolved sizes, the chosen memory tier, and the precision
policy, then returns `0`. This lets you see what a run would do — how big it is and
which path it would take — without spending any GPU time.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
