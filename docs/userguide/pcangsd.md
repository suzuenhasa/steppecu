# `steppe pcangsd` reference

## 1. What it is

`steppe pcangsd` runs a **PCAngsd** principal component analysis on the GPU —
PCA for **low-coverage** samples, straight from genotype **likelihoods** instead
of hard genotype calls. You hand it a Beagle GL file (one triple of likelihoods
per individual per site: homozygous-reference, heterozygous, homozygous-alternate),
and it estimates per-individual allele frequencies, builds a likelihood-weighted
sample-by-sample covariance, eigendecomposes it, and hands back the top-`e`
principal-component coordinates for each sample.

It answers the same plain question `steppe pca` does — *given a pile of
individuals, how do they group, and along what axes of variation?* — but for the
case where you **can't trust a hard genotype call**. When coverage is low (a
handful of reads per site, ancient DNA, shallow sequencing), the single most
likely genotype is often wrong, and a plain PCA on those calls smears the
structure. PCAngsd keeps the whole genotype-likelihood triple at every site and
lets the uncertainty flow through the whole fit, so the axes stay clean where a
hardcall PCA would blur.

This is the second consumer of steppe's genotype-likelihood tensor (the first is
`ingest --likelihoods`). Like `ibd` and `roh`, it is a self-contained command —
it reads a Beagle file and does its own linear algebra; it does not touch the f2
cache or the qpAdm fit engine.

---

## 2. The method, and the tool it matches

### The EM, faithfully ported

The math is ported from the reference **pcangsd** package (Meisner &
Albrechtsen 2018; the Rosemeis/pcangsd source), step for step:

- **emMAF** — a per-site EM that estimates the population allele frequency from
  the likelihoods, followed by a **minor-allele-frequency filter** that drops
  sites below `--maf`.
- **the Fumagalli init** — the starting individual-allele-frequency guess, from
  the population frequencies.
- **the main loop** — update each individual's allele frequency, form a truncated
  rank-`e` SVD of the centered likelihood matrix via its `EᵀE` gram, reconstruct,
  and repeat to convergence. This is the PCAngsd idea: the low-rank structure and
  the individual allele frequencies are estimated **together**, each refining the
  other.
- **the covariance** — a final GL-weighted, individual-allele-frequency-centered
  covariance with PCAngsd's corrected diagonal (`dCov`), then the eigendecomposition
  → PCA.

The gram SYRK and the eigen solve reuse the exact cuBLAS / cuSOLVER path that
`steppe pca` uses.

### The precision split

This follows steppe's standard fit-precision policy. The gram **SYRK** is
matmul-heavy, so it runs the **emulated-FP64 default** (the Ozaki matmul
accelerator); you can override it with `--precision`. The **EM elementwise math**
(a genotype-likelihood cancellation carve-out) and **every eigendecomposition**
are **native FP64**, always.

### One honest difference from the reference

steppe runs the main loop as **native FP64 plain fixed-point EM**. The reference
pcangsd runs **float32 with SQUAREM acceleration**. They converge to the **same
fixed point** — the answers match — but two things follow from it:

- The agreement is a **concordance**, not bit-exact (see below). float32 vs FP64
  and different iteration schemes reach the same place by slightly different steps.
- The plain EM takes **more iterations** to get there than SQUAREM does, and it
  is unaccelerated, so at a tight `--tole` it commonly runs to the `--iter` cap.
  steppe reports honestly which happened — "converged in N iters" vs "stopped at
  the --iter cap" — rather than always claiming convergence.

### What it is gated against

Concordance was checked on box5090 (Release, `sm_120`, real data) against
**pcangsd 1.36.4** (a numpy venv, `--eig 2`) — this matches the reference pcangsd
package, **not ADMIXTOOLS2**, which has no genotype-likelihood path at all. The
data was the popgen.dk PCAngsd tutorial **Demo2**: **100 low-coverage samples ×
50,000 SNPs** (49,492 kept after the MAF 0.05 filter), HapMap CEU / YRI / JPT.
Results:

- Beagle decode **600,000 / 600,000 GL values bit-exact** vs an independent read.
- Covariance **Frobenius relative-error ~5e-7**.
- PCA **PC1 and PC2 sign-aligned |Pearson r| = 1.000000**.
- Individual allele frequencies **r = 0.99999948**.

`nsys` confirmed the EM, covariance, and eigen steps are all GPU kernels over the
device-resident GL tensor — no host loop. The CpuBackend is a reference oracle
only.

---

## 3. Inputs

- **`--beagle FILE`** (required) — a Beagle genotype-likelihood file, gzipped
  (`.beagle.gz`) or plain. The layout is the tensor: after a header line, each row
  is a site (marker id, the two alleles), then **3 GL columns per individual** —
  homozygous-A1, heterozygous, homozygous-A2. steppe reads it straight into its
  shipped likelihood tile and uploads it to the device.

There is **no panel join** — PCAngsd is self-contained on the Beagle markers. The
individuals are processed in Beagle header order, and that order is the row order
of every output (it is the join key the gate used).

---

## 4. Outputs

steppe writes PCAngsd-convention files under the `--out PREFIX`:

- **`PREFIX.cov`** — the `N × N` GL-weighted covariance matrix.
- **`PREFIX.eigenvec`** — the `N × e` PC coordinates (one row per sample,
  eigenvector × √eigenvalue; the PC sign is arbitrary per axis).
- **`PREFIX.eigenval`** — the `e` eigenvalues, descending.

Two optional, off-by-default extras:

- **`--emit-freq`** → **`PREFIX.freq`** — the per-site population allele-2
  frequency (length = kept sites).
- **`--emit-iaf`** → **`PREFIX.pi`** — the `M × N` individual allele-2 frequencies
  (**large** — one value per kept site per sample).

Matrices are whitespace tables with no header — TSV by default, CSV with
`--format csv`. To stderr (kept out of the files) steppe prints a one-line
diagnostic: samples × kept/total sites, the MAF used, the PC count, whether it
converged or hit the cap plus the iteration count and final RMSE, and the PC1
variance-explained percentage.

---

## 5. The CLI

```
steppe pcangsd --beagle FILE -e E --out PREFIX
               [--iter N] [--tole T] [--maf M] [--maf-iter N] [--maf-tole T]
               [--emit-freq] [--emit-iaf]
               [--precision emu|fp64] [--format tsv|csv] [--device N]
```

| flag | meaning |
| --- | --- |
| `--beagle FILE` | Beagle GL file, `.beagle.gz` or plain (**required**) |
| `-e`, `--eig E` | number of PCs / IAF rank (**required**, ≥ 1; auto `-e` MAP-test is deferred) |
| `--out PREFIX` | output prefix; writes `PREFIX.{cov,eigenvec,eigenval}` (**required**) |
| `--iter N` | main-loop iteration cap (default 100) |
| `--tole T` | main-loop convergence RMSD (default 1e-5) |
| `--maf M` | per-site minor-allele-frequency filter (default 0.05) |
| `--maf-iter N` | emMAF iteration cap (default 500) |
| `--maf-tole T` | emMAF convergence RMSD (default 1e-6) |
| `--emit-freq` | also write `PREFIX.freq` (per-site population allele-2 freq) |
| `--emit-iaf` | also write `PREFIX.pi` (individual allele-2 freqs; large) |
| `--precision` | gram SYRK precision `emu` (default) or `fp64`; EM + eigen are always native FP64 |
| `--format` | matrix separator `tsv` (default) or `csv` |
| `--device N` | CUDA device ordinal (default auto) |

The exact invocation used for the timing below (Demo2 GL file, top-2 PCs):

```
steppe pcangsd --beagle /workspace/tmp/Demo2input.gz -e 2 --out t_pcangsd --device 0
```

**Measured wall-clock:** on that real Demo2 GL file (100 low-coverage samples ×
50,000 SNPs, 49,492 kept after MAF 0.05), steppe runs in **3.66 s** (median of 3;
720 MB peak RSS) on a single RTX 5090, converging in 100 plain-EM FP64 iterations
(final RMSE 9.93e-06). The reference pcangsd 1.36.4 finished this same tiny demo
in **~1 s** (16 CPU threads, harvested from the gate's own pcangsd log), so on
this small demo the reference is faster (~0.27×). That is expected: pcangsd
reaches the fixed point in a handful of float32 SQUAREM iterations, where steppe
runs 100 plain-EM FP64 iterations to the same place. This is a correctness demo,
not a throughput benchmark — a larger-sample GL panel is where the GPU EM would
pull ahead, but no such head-to-head was run.

---

## 6. Honest caveats

- **Concordance, not bit-exact.** steppe is native FP64 + plain EM; pcangsd is
  float32 + SQUAREM. They hit the same fixed point (covariance Frobenius ~5e-7,
  PC |r| = 1.000000, IAF r = 0.99999948), but the numbers are not identical to the
  last bit — by design.
- **On this tiny demo the reference is faster.** See §5. steppe's plain unaccelerated
  EM runs more iterations than SQUAREM does; on the 100-sample Demo2 that makes it
  slower. Size `--iter` up for tighter tolerances, and don't read the demo timing
  as a throughput claim.
- **`-e` is required — no auto rank yet.** You must pass the number of PCs. The
  automatic `-e` selection (PCAngsd's MAP test) is a documented follow-up, not
  built in v1.
- **Core PCAngsd only.** v1 is the headline output: the GL-weighted covariance,
  the PCA, and the individual allele frequencies. PCAngsd's admixture (Q/P),
  selection scan, HWE / inbreeding, and kinship extensions are documented
  follow-ups and are **not** built.
- **Beagle GL input only.** The input is a Beagle likelihood file. VCF `PL`/`GL`
  ingestion into this path is a documented follow-up.
- **PC signs are arbitrary.** As in every PCA tool, the sign of each individual PC
  is not meaningful — only the geometry is. A comparison against another tool must
  sign-align per PC first.
- **For hard genotype calls, use `steppe pca` instead.** `steppe pcangsd` is for
  low-coverage genotype likelihoods. If you have confident hard calls
  (`.geno`/`.snp`/`.ind`), the standard Patterson PCA is the other command.
