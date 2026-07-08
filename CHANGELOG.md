# Changelog

Notable changes to Steppe. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
each entry is grounded in the commit that landed it.

## Unreleased

The low-coverage / relatedness / population-statistics family, plus three GPU reformulations. Every
feature is gated on **real AADR data** against its field-standard reference tool (never AT2 for these
— they're a separate compute path).

### Added — commands

- **`fst`** — per-SNP Weir & Cockerham 1984 FST between two populations, on the GPU; gated
  machine-precision vs `plink2 --fst method=wc` (max\|diff\| 5.0e-7). (`edccab2`)
- **`fst --all-pairs`** — the whole population×population FST **matrix** in one GPU pass (decode-once
  sufficient stats → `sweep_unrank` over every pair → P×P matrix); per-cell bit-exact vs the
  single-pair path. (`0798fc0`)
- **`sfs`** — 2D joint site-frequency spectrum (folded/unfolded); bit-exact vs scikit-allel
  `joint_sfs`. (`e09cfab`)
- **`pca`** — Patterson-2006 genotype PCA (SYRK covariance + eigensolve) with an optional
  self-contained interactive HTML scatter; \|r\|=1.0 vs scikit-allel/sklearn. (`c3ea681`)
- **`pcangsd`** — low-coverage PCA / individual allele frequencies from a beagle genotype-likelihood
  file (the PCAngsd EM); vs `pcangsd` r=0.99999. (`dc37b08`)
- **`ibd`** — ancIBD IBD-segment detection between imputed ancient individuals (5-state FB over GP);
  vs `pip ancIBD` per-pair total-IBD r=0.99999, segment overlap 99.84%. (`848eb66`)
- **`roh`** — hapROH runs-of-homozygosity for an ancient genome vs a phased panel ((K+1)-state
  copying HMM); byte-exact vs `pip hapROH` on a real ancient. (`5f80ea9`)
- **`ingest --likelihoods`** — parse VCF `FORMAT/GL|PL|GP` into a GPU-resident `[site×sample×3]`
  likelihood tensor (the shared input for `pcangsd` and `ibd`); 100% bit-exact decode. (`8b97ee3`)
- **`ingest`** — first-class VCF reference-build detection (GRCh37/38 auto-detected) + a GRCh37-direct
  path. (`2688136`)

### Performance — GPU reformulations

- **`fst --all-pairs`** beats `plink2 --fst` at scale and the margin widens with the population count:
  **3.95×** at 109 pops, **19.9×** at 267, **29.5×** at 502; the full 3,898-pop AADR matrix (7.6M
  pairs) in **22 min** vs plink2's projected ~18.5 hr. Crossover ~60 pops (below it plink2 wins).
  (`0798fc0`)
- **`pca`** now runs the **full 23,089-sample AADR cohort**: a Halko randomized top-k eigensolver
  replaces the full-spectrum `Dsyevd` (extra workspace 8.5 GB → ~190 MB), lifting the OOM ceiling
  (23k cohort: 6m24s, 18 GB peak; top-10 matches the full solver). (`e220cff`)
- **`roh`** batch throughput **2.84×** (GPU duty 57→77%) via a panel-resident + look-ahead stream
  pipeline; output bit-identical. ~90× the single-threaded hapROH crawl. (`b67bef6`)

### Added — tooling / docs

- **`tests/bench/`** — a committed, re-runnable validation harness (`make bench`) that emits the
  8-row table (parity / K-curve / M-curve / missingness / small-N / CPU-vs-GPU error / runtime-vs-
  reference / memory) for FST + PCA on real data in ~2.4 min, no balls-out reference runs. (`fb151f0`)
- Per-file `docs/reference/*.md` for the new source, and per-command guides in `docs/userguide/`.

### Fixed

- **`roh`** rejects up front when `--n-ref` is large enough (>~5000) to drive the copying-HMM's
  column-0 prior negative, instead of silently producing wrong ROH. (`173ca74`)
