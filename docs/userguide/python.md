# Python API

steppe ships a small Python package that wraps the same GPU engines the command line drives, so you can build f2 caches and run fits from a script or a notebook instead of the shell — and get results back as objects and pandas DataFrames instead of CSV text.

## What it is

`import steppe` gives you a thin, pandas-friendly facade over the compiled extension. Every command-line tool has a matching function (`steppe.qpadm`, `steppe.extract_f2`, `steppe.f4`, …), and each returns a typed result object with tidy accessors. `pandas` is an **optional** dependency: `import steppe` works without it, and the `.weights` / `.to_dataframe()` accessors only import it when you actually call them.

Like the CLI, this is a **GPU product** — there is no CPU fallback. On a machine without a CUDA 13 device, the fit calls raise a clear "no CUDA device" error.

## Install

The Python API comes from the wheel (install route 3 in the [README](../../README.md)):

```bash
pip install steppe-0.1.0-cp312-abi3-linux_x86_64.whl
python -c "import steppe; print(steppe.__version__)"
```

Needs Python 3.12+ and the CUDA 13 runtime; `numpy` is the only hard dependency. Install `pandas` too if you want DataFrame output (`pip install pandas`).

## The core idea

Build or load an **f2 cache once**, then run as many fits against it as you like — the handle keeps the GPU-resident tensor alive, so repeated fits don't re-upload anything. This is the same "compute once, fit many" shape as [extract-f2](./extract-f2.md), just in-process.

```python
import steppe

f2 = steppe.read_f2("f2_dir/")            # load an existing STPF2BK1 f2 dir
res = steppe.qpadm(
    f2,
    target="England_BellBeaker",
    left=["Czechia_EBA_CordedWare", "Turkey_N"],
    right=["Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"],
)
print(res.weights)          # pandas DataFrame: [target, left, weight, se, z]
print(res.p, res.chisq, res.f4rank)
```

To go straight from genotypes, build the cache in-process instead of loading one:

```python
f2 = steppe.extract_f2(
    "/path/to/v66.p1_HO.aadr.patch.PUB",   # reads PREFIX.{geno,snp,ind}
    pops=["England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N",
          "Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"],
)
res = steppe.qpadm(f2, target="England_BellBeaker",
                   left=["Czechia_EBA_CordedWare", "Turkey_N"],
                   right=["Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"])
```

`extract_f2(..., out="f2_dir/")` writes a reusable f2 dir to disk and returns the path (then `read_f2("f2_dir/")` reloads it); with no `out=`, it returns an in-memory handle you feed straight to the fit functions.

## What's available

Each maps to a command-line tool; the [feature pages](./README.md) explain the statistics themselves.

| Function | Does |
|---|---|
| `read_f2(dir)` | Load an existing f2 dir into a handle |
| `extract_f2(prefix, pops=…)` | Build an f2 cache from genotypes ([extract-f2](./extract-f2.md)) |
| `qpfstats(prefix, pops=…)` | Build a *smoothed* f2 cache ([qpfstats](./qpfstats.md)) |
| `qpadm(f2, target=, left=, right=)` | Admixture-model fit ([qpadm](./qpadm.md)) |
| `qpwave(f2, left=, right=)` | Rank-sufficiency test ([qpwave](./qpwave.md)) |
| `qpadm_search(f2, target=, pool=, right=)` | Rotation over a source pool ([qpadm-rotate](./qpadm-rotate.md)) |
| `qpgraph(f2, graph=)` / `qpgraph_search(f2, pops=)` | Fit / search admixture graphs ([qpgraph](./qpgraph.md)) |
| `f4` / `f3` / `f4ratio` / `qpdstat` / `dstat` | Standalone [f-statistics](./f-statistics.md) |
| `dates(prefix, target=, left=)` | Admixture dating ([dates](./dates.md)) |
| `export_f2_rds(dir, out)` / `import_f2_rds(dir, out)` | The [RDS bridge](./data-and-formats.md) to/from ADMIXTOOLS 2 |

## Reading the results

Fit functions return typed result objects, not raw dicts:

- `qpadm(...)` → a result with `.weights` (a DataFrame `[target, left, weight, se, z]`), the summary scalars `.p` / `.chisq` / `.f4rank` / `.feasible`, and the `.rankdrop` / `.popdrop` diagnostic tables.
- `qpwave(...)` → `.per_rank` (the ascending-rank sweep) and `.rankdrop`.
- Each result also has `.to_dataframe()` for the whole thing as one frame.

Because `pandas` is imported lazily, the DataFrame accessors raise a clear message if pandas isn't installed — everything else works without it.

## Gotchas

- **GPU required.** No CUDA 13 device → the fit calls raise a clear error; there is no CPU path.
- **`pandas` is optional.** `import steppe` and the raw fits work without it; only the DataFrame accessors need it.
- **`extract_f2` has two return modes** — an in-memory handle (no `out=`) vs. a written f2 dir (`out="dir/"` returns the path). Use the handle when you'll run fits right away; write a dir when you want to reuse the cache later or from the CLI.
- **`blgsize` is in Morgans** (`0.05` = 5 cM), same footgun as everywhere else.
- **`pops` order doesn't matter for `extract_f2`** — the population axis is sorted internally (matching `pops.txt`), so pass them in any order.

## See also

- [Getting started](./README.md) and the [command cheatsheet](./commands.md)
- [extract-f2](./extract-f2.md), [qpadm](./qpadm.md), [data-and-formats](./data-and-formats.md)
- Reference: [`bindings_internal_bind_common.hpp.md`](../reference/bindings_internal_bind_common.hpp.md), [`bindings_bind_fstats.cpp.md`](../reference/bindings_bind_fstats.cpp.md)
