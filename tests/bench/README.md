# steppe validation & benchmark harness (FST + PCA)

Turns the 8-row validation table into a **re-runnable artifact**: run it and get the
table, on **real AADR data**, in **~5 minutes**. It drives the `steppe` CLI, `plink2`, and a
`numpy<2` venv (for the scikit-allel / sklearn / numpy oracles) — it is **not** a new binary
subcommand and it does not touch feature logic.

Scope v1 = **FST** and **PCA** (the two reformulated stats). Each feature emits its own
8-row table (`out/fst_table.{md,tsv}`, `out/pca_table.{md,tsv}`) plus `out/bench_meta.json`.

## How to run

On the box (`ssh box5090`, repo at `/workspace/steppe`, Release build, `--device 0`):

```bash
cd /workspace/steppe
LD_LIBRARY_PATH=/usr/local/cuda/lib64 TMPDIR=/workspace/tmp \
python3 tests/bench/run_bench.py \
  --steppe /workspace/steppe/build-recheck/bin/steppe \
  --fit9   /workspace/data/aadr/fixtures_eig/v66_fit9_ped \
  --full   /workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB \
  --out    tests/bench/out
```

Or via CMake (same thing; needs the box + venv, so it is a custom target, **not** wired into
the default `ctest` run):

```bash
cmake --build build-recheck --target bench
```

Flags: `--feature fst|pca|both` (default both), `--quick` (smaller top-K/N points for a
smoke run), `--device N`, `--seed N` (default 1). The runner **SKIPs cleanly (exit 0)** if no
GPU or the data are absent, like the existing gates. It exits non-zero only if a PASS/FAIL
row FAILs.

The first run **bootstraps** the `numpy<2` venv at `/workspace/tmp/pca_oracle_venv`
(`numpy<2` + `scikit-allel` + `scikit-learn`, ~1-2 min, cached thereafter). Scratch
(thinned/injected beds, `.raw`, plink2 output) lives under `$TMPDIR/bench_scratch` and is
wiped at the start of every run (disk-tight box); the venv is preserved.

## The 8 rows (per feature)

| # | dimension | what it measures |
|---|-----------|------------------|
| 1 | parity_vs_reference | FST: max\|WC_FST steppe−plink2\| per-SNP + genome ratio (plink2 `--fst method=wc report-variants`, **live**). PCA: subspace Gram rel + per-PC \|r\| + var-explained diff vs scikit-allel patterson PCA (**live**, small N). |
| 2 | scaling_K / scaling_N | FST: all-pairs wall + throughput at K=10/25/50/100 pops (full 1240K subset). PCA: wall at N≈500/1000/2000/4000 samples. |
| 3 | scaling_SNPs | wall at M=100k/300k/584k SNPs (thinned fit9_ped). |
| 4 | missingness | inject 0/5/10/20 % missingness into **real** fit9_ped; confirm finite/monotone/(FST) ==plink2. |
| 5 | small_sample | N=1/2/5. FST: both-pops-singleton ⇒ WC `n_bar≤1` ⇒ valid=0 flag. PCA: `-k≤N-1`, rank flag vs graceful, **no crash**. |
| 6 | cpu_vs_gpu | FST: numpy FP64 WC oracle vs GPU, max-abs + mean-abs (~1e-12). PCA: (a) `--precision fp64` vs `emu40`; (b) numpy `eigh` oracle vs GPU. |
| 7 | runtime_vs_reference | steppe vs the reference tool, same work, walls + threads + hardware. |
| 8 | memory | peak GPU MiB (nvidia-smi `-lms 200`) + peak host RSS (`/usr/bin/time -v`). |

## Scale (MODERATE — the whole harness is ~5 min)

Measured on box5090 (RTX 5090, EPYC 9654): FST all-pairs K=50 = 2.7 s, K=100 = 4.1 s (the
tile read amortizes over pairs); PCA on a 1240K subset ≈ 2-6 s. No full 3898-pop /
22k-sample / genome-wide max runs — those are the balls-out runs this harness deliberately
avoids.

## measured vs cited (honest labels — enforced by the emitter `status` column)

- **plink2** (fast C): always **live** — parity, missingness cross-check, runtime, M-curve → `measured` / `PASS`.
- **scikit-allel / sklearn** (slow Python): **live only at N≲514** (parity + runtime rows). Any larger-N reference number is **`cited`** from the frozen gate docs, never freshly run at scale.
- **No fully-synthetic panels.** The only synthetic operation is missingness injected into real fit9_ped genotypes (`inject_missing.py`).

Every row carries an explicit `status` (`PASS` / `FAIL` / `measured` / `cited` / `flag`), so a
reader never mistakes an extrapolation for a measurement. Re-running overwrites `out/`
deterministically (fixed `--seed 1` for thinning/injection).

## Files

- `bench_common.py` — shared library (discovery, timing/memory, plink2 wrappers, pop-table
  parsers, vectorized `.raw` loader, numpy-FP64 WC-FST oracle, Patterson+`eigh` PCA oracle,
  idempotent scratch builders, the Row/Table emitter). No side effects on import.
- `bench_fst.py` / `bench_pca.py` — `main(cfg) -> Table`, one file per feature.
- `inject_missing.py` — deterministic missingness injection into a PLINK `.bed`.
- `run_bench.py` — entrypoint (venv bootstrap, cache warm, dispatch, meta.json).
