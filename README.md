# steppe

**A GPU/CUDA-13 reimplementation of [ADMIXTOOLS 2](https://github.com/uqrmaie1/admixtools) + qpAdm.**

steppe computes f-statistics (the f2 block tensor) and fits admixture models —
qpAdm / qpWave / qpGraph / DATES, plus the standalone f3 / f4 / f4-ratio / D-statistic
family — entirely **on the GPU**. The f2 block tensor stays resident in VRAM and the
fits run device-side, so a model fit does zero host round-trips. Results are validated
against ADMIXTOOLS 2 goldens on real AADR data.

> steppe is a **GPU product**: a CUDA-capable GPU is required at runtime. There is no
> CPU runtime path (the bundled `CpuBackend` is a test/parity oracle only).

---

## Requirements

| | |
|---|---|
| **GPU** | CUDA-capable NVIDIA GPU, **Blackwell `sm_120`** (e.g. RTX 5090 / RTX PRO 6000). The shipped wheel is built for `sm_120`; rebuild with `STEPPE_CUDA_ARCH=<arch>` for another GPU. |
| **CUDA** | **CUDA 13** runtime + a matching driver (verified green on toolkit 13.0). The runtime is resolved at load from your CUDA 13 install — it is **not** bundled in the wheel. |
| **GPUs used** | **single-GPU** (`--device 0`). Multi-GPU is deferred; run single-GPU. |
| **Input format** | **TGENO, PACKEDANCESTRYMAP/GENO, EIGENSTRAT, PLINK (`.bed`/`.bim`/`.fam`), and ANCESTRYMAP** — the format is auto-detected from the magic/extension of the `--prefix` triple. All decode bit-identically (validated against the TGENO decode on real AADR). f2 dirs (STPF2BK1) are cacheable and reusable; they are **not** AT2 `.rds`-compatible. |
| **Python** | 3.9+. `numpy` is the only hard dependency; `pandas` is an optional/lazy dependency used only by the DataFrame accessors. |

---

## Install

steppe ships as **one GPU-only wheel** (scikit-build-core + nanobind, built for `sm_120`).
The CUDA 13 runtime is a system requirement, not a bundled or pip dependency.

```bash
pip install steppe          # the prebuilt sm_120 wheel
```

`import steppe` succeeds wherever CUDA 13 is on the loader path (`/usr/local/cuda/lib64`
or `LD_LIBRARY_PATH`); a GPU call on a box without a CUDA device raises a clear
"no CUDA device" error.

To build from source for a different arch:

```bash
pip wheel . --config-settings=cmake.define.STEPPE_CUDA_ARCH=<arch>
```

---

## CLI

The `steppe` binary exposes **14 subcommands**. `steppe --help` lists them; each
`steppe <cmd> --help` documents its flags. All compute runs on the GPU.

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

A typical end-to-end flow:

```bash
# 1) precompute the f2 dir once from a TGENO prefix
steppe extract-f2 --prefix /data/v66_HO --pops England_BellBeaker,Czechia_EBA_CordedWare,Turkey_N,Mbuti,Israel_Natufian,Iran_GanjDareh_N,Han,Papuan,Karitiana \
  --out f2_dir/ --blgsize 0.05 --maxmiss 0

# 2) fit qpAdm over it
steppe qpadm --f2-dir f2_dir/ \
  --target England_BellBeaker \
  --left  Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Israel_Natufian,Iran_GanjDareh_N,Han,Papuan,Karitiana \
  --format csv
```

---

## Python

The Python facade mirrors the CLI: `read_f2` / `extract_f2` / `qpfstats` build or load an
f2 handle, then `qpadm` / `qpwave` / `qpgraph` / `qpgraph_search` / `f4` / `f3` / `f4ratio`
/ `qpdstat` / `dstat` / `dates` / `qpadm_search` consume it. Results return as
pandas-friendly objects (`.weights`, `.table`, …) and the f2 tensor as a NumPy array.

```python
import steppe

f2 = steppe.read_f2("f2_dir/")           # load an STPF2BK1 f2 dir (no GPU upload yet)
res = steppe.qpadm(
    f2,
    target="England_BellBeaker",
    left=["Czechia_EBA_CordedWare", "Turkey_N"],
    right=["Mbuti", "Israel_Natufian", "Iran_GanjDareh_N",
           "Han", "Papuan", "Karitiana"],
)
print(res.weights)   # pandas DataFrame: [target, left, weight, se, z]
print(res.p, res.chisq, res.f4rank)
```

`steppe.extract_f2(prefix, pops=[...])` builds an f2 handle straight from a TGENO prefix on
the GPU (pass `out=...` to also write an STPF2BK1 dir). `pandas` is imported lazily, so
`import steppe` works without it.

---

## Documentation

- **[docs/RUN-SHEET.md](docs/RUN-SHEET.md)** — the canonical "how do I run everything" command sheet.
- **[docs/feature-matrix.md](docs/feature-matrix.md)** — every shipped feature, its CLI/Python surface, golden-match status, and a real-AADR wall-clock.
- **[docs/architecture.md](docs/architecture.md)** — the canonical design spec (GPU-first f2 tensor + device-resident fit engine).

---

## License

MIT — see [LICENSE](LICENSE). steppe is an independent reimplementation of the published
ADMIXTOOLS / ADMIXTOOLS 2 + DATES methods; ADMIXTOOLS 2 is used only as a test oracle (no
GPL source is vendored).
