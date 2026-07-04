# steppe examples — `read_f2 → qpadm → inspect` quick-starts

Two runnable quick-starts that fit a real-AADR qpAdm model and print the weights + tail
`p`. They are **living API canaries**: each walks the same public surface a user would, so
it stops compiling/running if that surface drifts. They add **no new compute** — they call
the frozen `read_f2` / `qpadm` (Python) and `read_f2_dir` / `build_resources` / `run_qpadm`
(C++) entry points the CLI uses — and **assert nothing; they just print**.

steppe is a **GPU product**: both examples run on a CUDA box (sm_120 / CUDA 13) only. Off
GPU the fit raises a clear *"no CUDA device"* message. **The local RTX 2070 / CUDA 11.8 is
the wrong arch and cannot build or run the C++ example** — build/run it on the box.

## Just want to run something? `f2_9pop/`

`f2_9pop/` here is a **ready-to-run** STPF2BK1 f2-dir (a tiny real-AADR 9-pop cache) — no build,
no staging. With the CLI installed, point `--f2-dir` straight at it:

```bash
steppe qpadm --f2-dir docs/examples/f2_9pop \
  --target England_BellBeaker --left Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian
```

(The one-liner installer stages the same cache to `~/.local/share/steppe/example`, and it's
also a release asset: `example_f2.tar.gz`.) The two scripts below are instead **living API
canaries** for the *built bindings* — read on for those.

## The model — real-AADR `golden_fit0`

Both examples fit the committed real-AADR golden (`tests/reference/goldens/at2`):

| field  | value |
| ------ | ----- |
| target | `England_BellBeaker` |
| left   | `Czechia_EBA_CordedWare`, `Turkey_N` |
| right  | `Mbuti`, `Israel_Natufian`, `Iran_GanjDareh_N`, `Han`, `Papuan`, `Karitiana` |

No synthetic data: the f2 tensor is the committed binary fixture
`tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin` (P=9, 710 jackknife blocks), the
same real-AADR f2 the parity gate uses.

## Getting an f2-dir

`read_f2` / `read_f2_dir` consume an **f2-dir** — a directory with `f2.bin` (STPF2BK1) +
`pops.txt` (one population per line, in P-axis index order). The committed fixture is a raw
`.bin`, so stage it into an f2-dir first (the Python quickstart does this for you):

```bash
# write the committed golden_fit0 fixture out as an f2-dir
python examples/python/quickstart_qpadm.py --stage /tmp/fit0_f2dir
```

Or point either example at a **real extract-f2 dir** instead, e.g. the box's
`/workspace/data/aadr/f2_fit0_corrected` (the canonical directory-path golden).

## Python quickstart

```bash
# zero-arg default: stage the committed golden_fit0 fixture into a temp dir and fit it
python examples/python/quickstart_qpadm.py

# or point at an existing f2-dir
python examples/python/quickstart_qpadm.py /tmp/fit0_f2dir
```

It needs the built bindings on `sys.path`; the script auto-discovers `build-rel/bindings`
(or set `STEPPE_BINDINGS_DIR`). `pandas` is optional (the weights print as a DataFrame when
present, a plain list otherwise).

## C++ quickstart (box only)

```bash
# configure + build on the box (examples + CLI both ON — the C++ canary needs steppe::access)
cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_EXAMPLES=ON
cmake --build build-rel --target quickstart_qpadm

# run it on the SAME f2-dir
./build-rel/bin/quickstart_qpadm /tmp/fit0_f2dir
```

`STEPPE_BUILD_EXAMPLES` defaults **OFF**. The C++ target links `steppe::access` (the shared
f2-dir reader + pop→index resolver), which only exists when `STEPPE_BUILD_CLI` (or
`STEPPE_BUILD_PYTHON`) is also ON; with neither, the build prints a status line and skips the
C++ target (the Python quickstart still works).

## Expected output — self-check

On the **staged committed fixture** (`f2_fit0_9pop.bin`, the f2-OBJECT-path values — the same
the Python and C++ parity gates assert):

```
left                          weight
Czechia_EBA_CordedWare      0.868755
Turkey_N                    0.131245
p=0.411881  chisq=3.956821  dof=4  f4rank=1
```

(Weights match to ~1e-6; `p` to ~1e-3.) Pointing instead at the canonical **directory-path**
golden (`/workspace/data/aadr/f2_fit0_corrected`) gives the directory-path values
`weight ≈ [0.868751, 0.131249]`, `p ≈ 0.4072` — the documented ~1e-5 read-arg difference
between the f2-object path and the f2-directory path (golden_fit0.json `reproduction.note`).
