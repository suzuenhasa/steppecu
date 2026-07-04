<h1 align="center">
  <img alt="Steppe" src="https://raw.githubusercontent.com/suzuenhasa/steppecu/main/docs/assets/steppe-logo.png" width="680">
</h1>

<p align="center"><em>A GPU based f-statistics and qpAdm toolset.</em></p>

> 🚧 **Work in progress — research preview.** Steppe is under active development. The compute
> paths that are done are validated to bit/tolerance parity with ADMIXTOOLS 2 on real AADR
> data, but install/onboarding, docs, and some APIs are still rough and may change. **Not a
> stable release yet** — expect sharp edges, and please don't rely on it for production work.

**[What it does](#what-it-does) · [Install](#install) · [Quick start](#quick-start) · [Commands](#commands) · [Python](#python) · [Performance](#performance) · [Docs](#documentation)**

## About

One of my favorite hobbies is finding ways to make software faster, and Steppe grew out of that
curiosity — I wanted a faster tool that would work on my not-so-modern desktop, and it was a great excuse to learn
CUDA along the way. Somewhere in there it turned into something I was excited to share, and I hope
it's useful to other people too.

It's still very much a personal project, so you'll occasionally find opinions, experiments, and
design decisions that reflect how I like to work. At the same time, I've done my best to make Steppe
reliable, approachable, and useful well beyond my own tinkering. ^^

If you use it, I genuinely hope it makes your work a little easier. And if you have ideas, questions,
or spot something that could be better, please let me know.

## What it does

Steppe computes f-statistics (the pairwise-f2 block tensor) and fits admixture models —
qpAdm / qpWave / qpGraph / DATES, plus the standalone f3 / f4 / f4-ratio / D-statistic family —
entirely **on the GPU**. The idea it rests on is ADMIXTOOLS's own: *precompute once, fit many*. One
streaming pass over the genotypes builds an f2 cache; after that the tensor stays resident in VRAM
and every fit runs device-side, so a model fit does zero host round-trips. That's most of why it's
fast.

> **Steppe is a GPU product.** A CUDA-13 GPU is required at runtime; there is no CPU runtime path
> (the bundled `CpuBackend` is a test/parity oracle only).

---

## Requirements

| | |
|---|---|
| **GPU** | NVIDIA GPU, compute capability **`sm_75`–`sm_120`** (Turing → Blackwell: RTX 20/30/40/50-series, T4, A100, A6000/A40, L4/L40, H100/GH200, RTX PRO 6000, …). The prebuilt binary and wheel are a **fatbinary** covering all of these arches. |
| **CUDA** | **CUDA 13.x runtime** + a matching driver (verified on 13.0 and 13.1). This is a **hard requirement** — the binaries link `libcudart.so.13` / `libcublas.so.13` / `libcusolver.so.12`, so a **CUDA 12 box will not run them**. The runtime is resolved at load from your CUDA install (`/usr/local/cuda/lib64` or `LD_LIBRARY_PATH`); it is **not** bundled. |
| **OS** | Linux x86_64. |
| **GPUs used** | single-GPU (`--device 0`). |
| **Python** (wheel only) | **3.12+** for the prebuilt wheel — a `cp312-abi3` **stable-ABI** wheel, so one file runs on 3.12, 3.13, 3.14+; 3.9+ if you build from source. `numpy` is the only hard dep; `pandas` is optional/lazy (the DataFrame accessors). |
| **Input formats** | EIGENSTRAT, PACKEDANCESTRYMAP/GENO, TGENO, ANCESTRYMAP, PLINK (`.bed`/`.bim`/`.fam`) — auto-detected from the `--prefix` triple; all decode bit-identically. f2 dirs (STPF2BK1) are cacheable and reusable (not AT2 `.rds`-compatible). |

---

## Install

**Fastest — one command.** Downloads the CLI, checks your CUDA 13 runtime (with a clear message
if it's missing, not the opaque loader error), and installs a launcher that sets
`LD_LIBRARY_PATH` for you so you never have to:

```bash
curl -fsSL https://raw.githubusercontent.com/suzuenhasa/steppecu/main/install.sh | bash
#  -> ~/.local/bin/steppe   (override the dir:  ... | bash -s -- --dir /usr/local/bin)

source ~/.bashrc   # the installer added ~/.local/bin to PATH — load it here (or open a new terminal)
steppe --help
```

Or do it by hand — three manual paths below; the prebuilt CLI + wheel are attached to the
[v0.1.0 release](https://github.com/suzuenhasa/steppecu/releases/tag/v0.1.0) and download over
plain `curl` (no GitHub auth). **(1) is simplest: one binary, no Python, no build.**

### 1. Prebuilt CLI — no build
A single standalone executable (a fatbinary for every supported GPU):

```bash
curl -L -o steppe https://github.com/suzuenhasa/steppecu/releases/download/v0.1.0/steppe
chmod +x steppe
./steppe --version
```

It links the system CUDA 13 runtime at load. If you see `libcudart.so.13: cannot open
shared object file`, your box has CUDA 12 (or CUDA isn't on the loader path) — Steppe needs
CUDA 13. Point the loader at it if needed: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64`.

### 2. Prebuilt wheel — the Python API
```bash
curl -L -O https://github.com/suzuenhasa/steppecu/releases/download/v0.1.0/steppe-0.1.0-cp312-abi3-linux_x86_64.whl
pip install ./steppe-0.1.0-cp312-abi3-linux_x86_64.whl
python -c "import steppe; print(steppe.__version__)"
```
Needs Python 3.12+ and the CUDA 13 runtime; `numpy` is the only hard dependency. (It's a
`cp312-abi3` stable-ABI wheel — the *same* file installs on 3.12, 3.13, 3.14+.)

### 3. Build from source — any CUDA-13 arch / Python
Needs the full **CUDA 13 toolkit** (nvcc) + CMake ≥ 3.28 + Ninja.
```bash
git clone https://github.com/suzuenhasa/steppecu && cd steppecu
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON
cmake --build build                 # → build/bin/steppe
```
Defaults to `sm_120`. Build for **the GPU in the box**: add `-DCMAKE_CUDA_ARCHITECTURES=native`.
Build a portable **fatbinary**: `-DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"`.
Build the **Python wheel**: `pip wheel . --no-deps` (add `--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native` for your GPU).

---

## Quick start

**First result in seconds.** The installer stages a tiny **real** 10-population AADR f2 cache, so
you can fit a qpAdm model immediately — no data download, no build:

```bash
steppe qpadm --f2-dir ~/.local/share/steppe/example \
  --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian
#  -> weights (Russia_Samara_EBA_Yamnaya ~0.73, Turkey_N ~0.27) + tail p ~0.11 — a real fit.
#     Corded Ware ≈ 73% steppe (Yamnaya) + 27% Anatolian farmer — the migration this tool is named
#     for (Haak et al. 2015). That steppe ancestry then flows on into Bell Beaker — see docs/userguide/.
```

Built from source, or want it by hand? Grab the same example directly:
```bash
mkdir example && curl -fsSL \
  https://github.com/suzuenhasa/steppecu/releases/download/v0.1.0/example_f2.tar.gz \
  | tar xz -C example
```

**Your own data?** Steppe works over an **f2-blocks directory** — a cache you build once from a
genotype prefix, then run any number of fits against. The full workflow (data → `extract-f2` →
`qpadm` → sweeps), every command's flags with runnable examples, and a marimo notebook walkthrough
all live in the **[user guide](docs/userguide/)** — start with **[getting started](docs/userguide/)**.

---

## Commands

`steppe --help` lists the **14 subcommands** (all compute runs on the GPU); the
**[user guide](docs/userguide/)** has a page per command explaining every flag with runnable examples.

| Command | What it does |
|---|---|
| `extract-f2` | Precompute the f2_blocks dir from a genotype prefix (the precompute-once / fit-many artifact). |
| `qpadm` | Single-model qpAdm fit over an f2_blocks dir. |
| `qpadm-rotate` | qpAdm rotation sweep over a source pool (batched on the GPU). |
| `qpwave` | qpWave rank-sufficiency sweep (no target; `left[0]` is the reference). |
| `qpgraph` | Fit a single fixed admixture graph (`--graph` edge-list). |
| `qpgraph-search` | Exhaustive bounded admixture-graph topology search (returns the global-best). |
| `f4` | Standalone f4(p1,p2;p3,p4) — est/se/z/p per quartet (`--all-quartets` for a sweep). |
| `f3` | Standalone f3(C;A,B) / outgroup-f3 — est/se/z/p per triple (`--all-triples` for a sweep). |
| `f4-ratio` | f4-ratio admixture proportion `alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4)`. |
| `f4-sweep` | GPU all-quartets f4 scan over C(P,4) with an on-device \|z\|/top-k filter (survivors only). |
| `f3-sweep` | GPU all-triples f3 scan over C(P,3) with an on-device filter. |
| `qpdstat` | D-statistic: `--f2-dir` reports f4 (f2-path), or `--prefix` reports the genotype-path normalized-D. |
| `qpfstats` | Genotype-path joint f2 smoother (the AADR-missingness remedy); emits an f2 dir. |
| `dates` | Admixture dating (DATES): generations since admixture from the ancestry-covariance decay. |

**Precision & output.** `--precision emu40` (default: emulated-FP64 / Ozaki for the
matmul-heavy paths, native FP64 for the cancellation-sensitive stats) · `emu32` · `fp64`
(native) · `tf32` (screening only). f4/f3/qpDstat and the jackknife SEs run in native FP64 and
are **bit-identical across GPU arches**. Output: `--format csv|tsv|json`, `--out FILE`
(stdout if omitted), `--device 0`.

---

## Python

The Python facade mirrors the CLI: build or load an f2 handle, then run any fit against it —
results come back as pandas-friendly objects (`.weights`, …).

```python
import steppe
f2  = steppe.read_f2("f2_dir/")
res = steppe.qpadm(f2, target="England_BellBeaker",
                   left=["Czechia_EBA_CordedWare", "Turkey_N"],
                   right=["Mbuti", "Han", "Papuan", "Karitiana", "Iran_GanjDareh_N", "Israel_Natufian"])
print(res.weights)   # DataFrame: [target, left, weight, se, z]
```

The full API — every function, the result objects, the DataFrame accessors — is on the
**[Python page](docs/userguide/python.md)** of the user guide. `pandas` is optional (imported lazily).

---

## Performance

Making things fast is the whole point, so here's what **one RTX 5090** does on real AADR data
(single GPU, the release binary; measured on a shared box, so these are conservative).

**Precompute — stream genotypes → f2 cache** (`extract-f2`, the one expensive step):

| Panel | SNPs | genotypes | 100 pops | 500 pops |
|---|---|---|---|---|
| Human Origins | 584 K | 3.8 GB | 5.5 s | 18 s |
| 1240k | 1.23 M | 6.7 GB | 9 s | 37 s |
| 2M | 2.14 M | 12 GB | 12 s | 58 s |

The genotype matrix streams out-of-core, and large caches **auto-tier** (the f2 blocks spill from
VRAM → host RAM → disk as needed), so panels far bigger than the card's memory still build.

**Fits & stats** — single call on a 500-population cache: qpAdm, qpWave, f4, f3, f4-ratio, and D all
land in **~3 seconds**; DATES in ~2 s; `qpfstats` in under a second.

**Model search** — a qpAdm **rotation of 4,047 candidate models** in **2.8 s** (~1,400 models/s), and
an admixture-graph topology search of **25,560 graphs** in **14.5 s** (~1,770 graphs/s).

**Sweeps** — every combination, filtered on-device to the top survivors:

| Sweep | statistics computed | time |
|---|---|---|
| f3, all triples (500 pops) | 20.7 million | 4 s |
| f4, all quartets (500 pops) | 2.57 **billion** | 2.9 min |
| **f4, all quartets (700 pops)** | **9.92 billion** | **12 min** |

→ roughly **14 million f4-statistics per second** — 9.9 billion of them in about twelve minutes, on
one GPU.

---

## Accuracy

Steppe is checked against ADMIXTOOLS 2 (the reference implementation) on real AADR data: f4 / f3 /
qpAdm weights, chi-square, p, and the block-jackknife standard errors match to bit/tolerance parity.
The point of a reimplementation is to be a *faster* ADMIXTOOLS, not a different one — so correctness
comes first, and the goldens live under `tests/`.

---

## Citing Steppe

If it helped your work, please cite the software:

> Steppe (v0.1.0). https://github.com/suzuenhasa/steppecu

And please **also cite ADMIXTOOLS 2 and the underlying methods** — they're the science Steppe reruns;
see [Acknowledgments](#acknowledgments) for the references.

---

## Documentation

- **[docs/userguide/](docs/userguide/)** — **start here.** Getting started, a page per feature
  (what every flag does + commands to run), the command cheatsheet, and the Python API.
- **[docs/examples/](docs/examples/)** — a worked end-to-end example + a marimo notebook walkthrough.
- **[docs/reference/](docs/reference/)** — per-module internals, one file per source file (for contributors).

---

## Acknowledgments

Steppe stands on the shoulders of giants, and would not have been possible without the work of great
people in this field. Here are the projects and papers Steppe builds on or was inspired by:

**ADMIXTOOLS 2** — I used this to check Steppe against constantly. The f-statistics and qpAdm fits are
the parity oracle that I validated Steppe against, bit for bit. If you use Steppe, please also cite
their work, which can be found here: <https://doi.org/10.7554/eLife.85492> ·
[article](https://elifesciences.org/articles/85492) · [GitHub](https://github.com/uqrmaie1/admixtools).

**ADMIXTOOLS** — the f2/f3/f4/D and the qp* methods (qpAdm, qpWave, qpGraph, qpDstat) all come from the
original ADMIXTOOLS methods worked out by Nick Patterson, David Reich, and colleagues. Steppe is a new
engine, but the hard part was based on their work: <https://doi.org/10.1534/genetics.112.145037>.

**Dating / DATES / ALDER** — Steppe reimplements the DATES admixture-dating method, which is based on
the earlier approach in ALDER that Steppe's dating lineage builds on. You can find the relevant papers
here: <https://doi.org/10.7554/eLife.77625>, <https://doi.org/10.1126/science.aat7487>,
<https://doi.org/10.1534/genetics.112.147330>.

**EIGENSOFT, convertf, PLINK** — Steppe supports various genotype formats and can regenerate parity
goldens, including PLINK. You can find the relevant information here:
<https://doi.org/10.1371/journal.pgen.0020190>, <https://doi.org/10.1038/ng1847>,
<https://doi.org/10.1186/s13742-015-0047-8>, <https://www.cog-genomics.org/plink/2.0/>,
<https://doi.org/10.1086/519795>.

**igraph** — Steppe mirrors ADMIXTOOLS 2's graph data-structure conventions on-device with igraph. I
can't find a DOI to cite, but here's the website: <https://igraph.org>.

**AADR datasets** — Steppe's validations and benchmarks were run on real ancient-DNA genotypes from the
Allen Ancient DNA Resource. I'd also like to point out that the AADR datasets are curated, managed, and
made freely available. If you found their efforts contributed to anything you've done, please show the
team your support! Here are the relevant citations: <https://doi.org/10.1038/s41597-024-03031-7>,
<https://doi.org/10.7910/DVN/FFIDCW>.

**The name "Steppe"** — the name was coined for the steppe-ancestry migration story, and the one paper
that specifically inspired it was the Haak 2015 study of *Massive migration from the steppe as a source
for Indo-European languages in Europe*. It was the first study I was able to reproduce and used as an
example, and is the study I use in the quickstart along with the Olalde study:
<https://doi.org/10.1038/nature14317>, <https://doi.org/10.1038/nature25738>.

**Claude Code** — a different kind of thank-you. My pair-programming partner for this whole build:
GPU kernels, chasing parity bugs bit-for-bit, the proxy/model scanner, the docs you're reading, and a
hundred late-night refactors — all of it side by side in the terminal. A real G, and a genuinely good
bro to build with. 🤝

---

## License

Steppe is released under the **GPL-3** license (see [LICENSE](LICENSE)). I read through ADMIXTOOLS 2's source to understand how it works so
I could build my own GPU version, and releasing Steppe under the same license is the right thing to do.
If I've credited anything here imperfectly, please tell me — I want to get it right. ^^
