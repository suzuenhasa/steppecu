# steppe

**A GPU/CUDA-13 reimplementation of [ADMIXTOOLS 2](https://github.com/uqrmaie1/admixtools) + qpAdm / DATES.**

> 🚧 **Work in progress — research preview.** steppe is under active development. The compute
> paths that are done are validated to bit/tolerance parity with ADMIXTOOLS 2 on real AADR
> data, but install/onboarding, docs, and some APIs are still rough and may change. **Not a
> stable release yet** — expect sharp edges, and please don't rely on it for production work.

steppe computes f-statistics (the pairwise-f2 block tensor) and fits admixture models —
qpAdm / qpWave / qpGraph / DATES, plus the standalone f3 / f4 / f4-ratio / D-statistic
family — entirely **on the GPU**. The f2 tensor stays resident in VRAM and the fits run
device-side, so a model fit does zero host round-trips. Results are validated to
bit/tolerance parity against ADMIXTOOLS 2 on real ancient-DNA (AADR) data.

> **steppe is a GPU product.** A CUDA-13 GPU is required at runtime; there is no CPU
> runtime path (the bundled `CpuBackend` is a test/parity oracle only).

---

## Requirements

| | |
|---|---|
| **GPU** | NVIDIA GPU, compute capability **`sm_75`–`sm_120`** (Turing → Blackwell: RTX 20/30/40/50-series, T4, A100, A6000/A40, L4/L40, H100/GH200, RTX PRO 6000, …). The prebuilt binary and wheel are a **fatbinary** covering all of these arches. |
| **CUDA** | **CUDA 13.x runtime** + a matching driver (verified on 13.0 and 13.1). This is a **hard requirement** — the binaries link `libcudart.so.13` / `libcublas.so.13` / `libcusolver.so.12`, so a **CUDA 12 box will not run them**. The runtime is resolved at load from your CUDA install (`/usr/local/cuda/lib64` or `LD_LIBRARY_PATH`); it is **not** bundled. |
| **OS** | Linux x86_64. |
| **GPUs used** | single-GPU (`--device 0`). |
| **Python** (wheel only) | **3.12** for the prebuilt wheel (`cp312`); 3.9+ if you build from source. `numpy` is the only hard dep; `pandas` is optional/lazy (the DataFrame accessors). |
| **Input formats** | EIGENSTRAT, PACKEDANCESTRYMAP/GENO, TGENO, ANCESTRYMAP, PLINK (`.bed`/`.bim`/`.fam`) — auto-detected from the `--prefix` triple; all decode bit-identically. f2 dirs (STPF2BK1) are cacheable and reusable (not AT2 `.rds`-compatible). |

---

## Install

Three ways; **(1) is simplest: one binary, no Python, no build.** The prebuilt CLI + wheel are
attached to the [v0.1.0 release](https://github.com/suzuenhasa/steppecu/releases/tag/v0.1.0)
and download over plain `curl` — no GitHub auth needed.

### 1. Prebuilt CLI — no build
A single standalone executable (a fatbinary for every supported GPU):

```bash
curl -L -o steppe https://github.com/suzuenhasa/steppecu/releases/download/v0.1.0/steppe
chmod +x steppe
./steppe --version
```

It links the system CUDA 13 runtime at load. If you see `libcudart.so.13: cannot open
shared object file`, your box has CUDA 12 (or CUDA isn't on the loader path) — steppe needs
CUDA 13. Point the loader at it if needed: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64`.

### 2. Prebuilt wheel — the Python API
```bash
curl -L -O https://github.com/suzuenhasa/steppecu/releases/download/v0.1.0/steppe-0.1.0-cp312-cp312-linux_x86_64.whl
pip install ./steppe-0.1.0-cp312-cp312-linux_x86_64.whl
python -c "import steppe; print(steppe.__version__)"
```
Needs Python 3.12 + the CUDA 13 runtime; `numpy` is the only hard dependency.

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

steppe works over an **f2-blocks directory** — a precomputed cache of pairwise f2 per
genome block. Build it once, then run any number of stats/fits over it.

```bash
S=./steppe        # or just `steppe` if it's on your PATH

# 1) build the f2 cache from a genotype prefix (reads PREFIX.{geno,snp,ind})
$S extract-f2 --prefix /data/v66_HO --auto-top-k 200 --maxmiss 0.5 --device 0 --out-dir f2_dir

# 2) qpAdm — model a target as a mixture of sources (jackknife SEs, JSON out)
$S qpadm --f2-dir f2_dir --target England_BellBeaker \
  --left  Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --jackknife 2 --format json

# 3) sweep EVERY f4 quartet on the GPU, keep the top-K most extreme |z|
$S f4 --all-quartets --f2-dir f2_dir --top-k 1000000 --sure --shard-dir sweep_out --device 0
```

The full copy-paste command sheet for every subcommand — including the 2M-SNP AADR panel and
measured sweep timings (e.g. C(700,4) = 9.9 B quartets in ~12 min on one RTX 5090) — is in
**[docs/commands.md](docs/commands.md)**.

---

## CLI

`steppe --help` lists the **14 subcommands**; `steppe <cmd> --help` documents each one's flags.
All compute runs on the GPU.

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
are **bit-identical across GPU arches**; the emulated path is accuracy-approximate (~1e-15,
well below the statistical noise floor). Output: `--format csv|tsv|json`, `--out FILE`
(stdout if omitted), `--device 0`.

---

## Python

The Python facade mirrors the CLI: `read_f2` / `extract_f2` / `qpfstats` build or load an f2
handle, then `qpadm` / `qpwave` / `qpgraph` / `qpgraph_search` / `f4` / `f3` / `f4ratio` /
`qpdstat` / `dstat` / `dates` / `qpadm_search` consume it. Results return as pandas-friendly
objects (`.weights`, `.table`, …) and the f2 tensor as a NumPy array.

```python
import steppe

f2 = steppe.read_f2("f2_dir/")           # load an STPF2BK1 f2 dir
res = steppe.qpadm(
    f2,
    target="England_BellBeaker",
    left=["Czechia_EBA_CordedWare", "Turkey_N"],
    right=["Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"],
)
print(res.weights)          # DataFrame: [target, left, weight, se, z]
print(res.p, res.chisq, res.f4rank)
```

`steppe.extract_f2(prefix, pops=[...])` builds an f2 handle straight from a genotype prefix on
the GPU (pass `out=...` to also write an STPF2BK1 dir). `pandas` is imported lazily, so
`import steppe` works without it.

---

## Documentation

- **[docs/commands.md](docs/commands.md)** — the runnable copy-paste command sheet (kept live).
- Deeper design/reference docs (architecture spec, feature matrix, run-sheet, study
  reproductions, perf notes) live under **[docs/archive/](docs/archive/)**.

---

## License

MIT — see [LICENSE](LICENSE). steppe is an independent reimplementation of the published
ADMIXTOOLS / ADMIXTOOLS 2 + DATES methods; ADMIXTOOLS 2 is used only as a test oracle (no
GPL source is vendored).
