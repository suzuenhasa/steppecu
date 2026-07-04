# Getting started

steppe is a GPU reimplementation of [ADMIXTOOLS 2](https://github.com/uqrmaie1/admixtools) —
qpAdm, qpWave, qpGraph, f-statistics, and DATES — that runs on an NVIDIA GPU and matches
ADMIXTOOLS 2 numerically. If you already know these tools, the commands and results will feel
familiar; steppe just runs them on the GPU, which is what makes the big batches — thousands of
models, whole-dataset sweeps — practical to run. Same math, same numbers.

This page gets you from nothing to your first real fit, then points you at the rest of the
user guide.

## Install

The shortest path is the one-liner. It downloads the CLI, checks that you have a CUDA 13
runtime (with a clear message if you don't), and installs a launcher that sets up the library
path for you:

```bash
curl -fsSL https://raw.githubusercontent.com/suzuenhasa/steppecu/main/install.sh | bash

source ~/.bashrc   # the installer added ~/.local/bin to PATH — load it here (or open a new terminal)
steppe --help
```

The `source ~/.bashrc` step matters: the installer puts `steppe` on your `PATH` by editing
`~/.bashrc`, but your current shell won't see that until you reload it (or just open a new
terminal). After that, `steppe` works from anywhere.

**One hard requirement: a CUDA 13 GPU.** steppe is GPU-only — there is no CPU fallback. The
binaries link the CUDA 13 runtime (`libcudart.so.13`), so a CUDA 12 box will not run them. If
you see `libcudart.so.13: cannot open shared object file`, your machine is on CUDA 12 or CUDA
isn't on the loader path.

Prefer to install by hand? There are three other routes — a standalone prebuilt CLI, a Python
wheel, and building from source. See the [README](../../README.md) for those; this page won't
duplicate them.

## Your first run

The installer stages a tiny but **real** 9-population AADR f2 cache, so you can fit a model
immediately — no data download, no cache to build:

```bash
steppe qpadm --f2-dir ~/.local/share/steppe/example_9pop \
  --target England_BellBeaker \
  --left Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian
```

This models the target `England_BellBeaker` as a mixture of two sources
(`Czechia_EBA_CordedWare` and `Turkey_N`), using the six right-hand populations as outgroups.
You should get weights close to:

```
Czechia_EBA_CordedWare  ~0.87
Turkey_N                ~0.13
tail p                  ~0.41
```

**How to read it.** The **weights** are the estimated ancestry proportions of the target from
each source — here, Bell Beaker ancestry that is mostly steppe-related (à la Olalde et al.
2018). The **p-value** (tail probability) tests whether the model fits: a larger p means the
data are consistent with this many sources, and a very small p means the model is rejected. A
run is marked **feasible** when the fit is acceptable and every weight is a sensible proportion
(none negative). So `p ~0.41` with two positive weights is a clean, feasible two-way fit.

## A typical real workflow

With your own data, steppe works over an **f2-blocks directory** — a precomputed cache of
pairwise f2 statistics per genome block. You build that cache once, then run any number of
analyses against it. Three steps:

1. **Get data.** Obtain a genotype dataset (AADR or your own) in a supported format —
   PACKEDANCESTRYMAP, EIGENSTRAT, PLINK, or TGENO. See [data and formats](./data-and-formats.md).
2. **Build the f2 cache.** Run `extract-f2` on your genotype prefix to produce the f2-blocks
   directory. See [extract-f2](./extract-f2.md).
3. **Run an analysis.** Point qpAdm (or any other command) at that `--f2-dir`. See
   [qpadm](./qpadm.md).

Almost every command below needs an f2-dir from step 2 first, so start there.

## User guide index

- **[extract-f2](./extract-f2.md)** — build the f2-blocks cache from a genotype prefix; the
  starting point for nearly everything else.
- **[qpadm](./qpadm.md)** — model a target population as a mixture of source populations, with
  a fit p-value and jackknife standard errors.
- **[qpwave](./qpwave.md)** — test how many independent ancestry streams relate a set of left
  populations to the right outgroups (the rank test behind qpAdm).
- **[qpadm-rotate](./qpadm-rotate.md)** — sweep many source combinations for one target from a
  pool of candidates, ranking the feasible models.
- **[qpgraph](./qpgraph.md)** — fit and score admixture graphs (topologies with drift and
  admixture edges).
- **[dates](./dates.md)** — estimate admixture dates from the decay of ancestry linkage
  (DATES).
- **[f-statistics](./f-statistics.md)** — compute f2, f3, f4/D-statistics and related tests
  from an f2-dir.
- **[sweeps](./sweeps.md)** — enumerate large batches of statistics (e.g. all quartets) in one
  GPU-bound pass.
- **[qpfstats](./qpfstats.md)** — turn a genotype dataset into a smoothed f2 directory you can
  then feed to qpadm/f4/qpgraph.
- **[data and formats](./data-and-formats.md)** — supported genotype formats and how steppe
  reads them.
- **[precision and tiers](./precision-and-tiers.md)** — the `--precision` matmul modes
  (emu40/emu32/fp64/tf32) and when to reach for each.
