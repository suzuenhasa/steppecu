# Command cheatsheet

Copy-paste one-liners for every steppe subcommand. They assume `steppe` is installed and on your
PATH — see [getting started](./README.md).

The **f2-based** examples run against the bundled 10-population example that the installer stages at
`~/.local/share/steppe/example`, so they work the moment you've installed steppe. The
**genotype-based** ones (`extract-f2`, `qpfstats`, `dates`) need your own data — see
[data and formats](./data-and-formats.md).

> The bundled example contains: `Czechia_EBA_CordedWare`, `England_BellBeaker`,
> `Russia_Samara_EBA_Yamnaya`, `Turkey_N`, `Mbuti`, `Han`, `Papuan`, `Karitiana`,
> `Iran_GanjDareh_N`, `Israel_Natufian`.

Every command takes `--device <n>` (default `0`) and, where it produces a result,
`--format csv|tsv|json` (default `csv`) and `--out FILE`. They're omitted below for brevity.
(`roh` also accepts a **comma-separated device list** — `--device 0,1,2,3` — to shard an ancient
cohort across multiple GPUs; the output is byte-identical to a single-GPU run.)

---

## qpAdm — fit an admixture model

Target as a mixture of `--left` sources, tested against the `--right` outgroups. See [qpadm](./qpadm.md).

```bash
# Corded Ware = Yamnaya (steppe) + Anatolian farmer — the Haak 2015 steppe migration (~73% / 27%)
steppe qpadm --f2-dir ~/.local/share/steppe/example \
  --target Czechia_EBA_CordedWare \
  --left  Russia_Samara_EBA_Yamnaya,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --jackknife 2 --format json
```

> **Measured** (one RTX 5090, `emu40`): a 3-source / 8-outgroup fit against a real 77-population
> AADR-1240K f2 cache runs in **~0.72 s** (peak host RAM ~0.6 GB, GPU ~12 GB). A single `f3` /
> `f4` / `qpdstat` triple over the same cache is **~0.47 s** (GPU ~0.5 GB). Timings are the median
> of 3 timed reps after one warmup, measured with `/usr/bin/time`.

## qpWave — rank test (no target)

Tests how many ancestry streams relate the `--left` set to the outgroups; `left[0]` is the
reference. See [qpwave](./qpwave.md).

```bash
steppe qpwave --f2-dir ~/.local/share/steppe/example \
  --left  Czechia_EBA_CordedWare,Turkey_N,England_BellBeaker \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian
```

## qpAdm-rotate — many competing models in one pass

Enumerates source subsets of `--pool` for one target in a single batched GPU run. See
[qpadm-rotate](./qpadm-rotate.md).

```bash
steppe qpadm-rotate --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker \
  --pool   Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian \
  --right  Mbuti,Han,Papuan,Karitiana \
  --min-sources 1 --max-sources -1 --format csv
```

## scan — guided, gated, best-first proxy/model search

Searches the pool for a better source model, gated + ranked best-first (with a marked winner),
plus a relatedness shortlist, swap suggestions, and an outgroup-admissibility check. See
[scan](./scan.md).

```bash
# guided search: gate + rank, winner marked "selected, not confirmed"
steppe scan --f2-dir ~/.local/share/steppe/example \
  --target England_BellBeaker \
  --pool  Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian \
  --right Mbuti,Han,Papuan,Karitiana --strategy beam

# --prerank: which sources are closest to the target (outgroup-f3 shortlist)
steppe scan --f2-dir ~/.local/share/steppe/example --target England_BellBeaker \
  --pool  Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian \
  --right Mbuti,Han,Papuan,Karitiana --prerank

# --suggest-swaps: for failing models, "drop the culprit, add a related source" (refit-verified)
steppe scan --f2-dir ~/.local/share/steppe/example --target England_BellBeaker \
  --pool  Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian,Han \
  --right Mbuti,Papuan,Karitiana --min-sources 2 --max-sources 2 --suggest-swaps

# --right-search: do the outgroups actually distinguish the sources? (anti-circularity)
steppe scan --f2-dir ~/.local/share/steppe/example --target England_BellBeaker \
  --pool  Czechia_EBA_CordedWare,Turkey_N --right Mbuti,Han,Papuan,Karitiana \
  --no-allow-clade --right-search check
```

---

## Standalone f-statistics

See [f-statistics](./f-statistics.md).

```bash
# f4(p1,p2 ; p3,p4)
steppe f4 --f2-dir ~/.local/share/steppe/example \
  --pop1 England_BellBeaker --pop2 Czechia_EBA_CordedWare --pop3 Han --pop4 Iran_GanjDareh_N

# f3(C ; A, B)
steppe f3 --f2-dir ~/.local/share/steppe/example \
  --pop1 Czechia_EBA_CordedWare --pop2 England_BellBeaker --pop3 Turkey_N

# f4-ratio  alpha = f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)
steppe f4-ratio --f2-dir ~/.local/share/steppe/example \
  --pops Czechia_EBA_CordedWare,Mbuti,Turkey_N,Han,England_BellBeaker

# D-statistic (f2-path: reports f4 with Z / p)
steppe qpdstat --f2-dir ~/.local/share/steppe/example \
  --pop1 Mbuti --pop2 Han --pop3 Czechia_EBA_CordedWare --pop4 Turkey_N
```

---

## Sweeps — score every combination on the GPU

Enumerate every quartet / triple over a pop set, keep only survivors (`--top-k` or `--min-z`).
`--sure` lifts the safety cap on huge sweeps. See [sweeps](./sweeps.md).

```bash
# every f4 quartet over the example (C(9,4) = 126), keep the top 50 by |z|
steppe f4-sweep --f2-dir ~/.local/share/steppe/example --top-k 50

# every f3 triple, keep |z| >= 3
steppe f3-sweep --f2-dir ~/.local/share/steppe/example --min-z 3

# the same sweep via f4/f3 with --all-quartets/--all-triples + sharded CSV output
steppe f4 --all-quartets --f2-dir ~/.local/share/steppe/example \
  --top-k 50 --shard-dir ./sweep_out
```

> **At scale:** on a real panel this is the headline path — C(700,4) ≈ **9.9 billion** quartets
> over a 700-population f2 cache, top-1,000,000 survivors, in about **12 minutes on one RTX 5090**.
> Add `--sure` for sweeps past the safety cap.

---

## Admixture graphs

Fit a graph you provide, or search topologies over a leaf set. See [qpgraph](./qpgraph.md).

```bash
# fit a single graph topology (edge-list: "parent child" per line)
steppe qpgraph --f2-dir ~/.local/share/steppe/example --graph mygraph.txt --format json

# search topologies over a bounded leaf set (>= 3 pops)
steppe qpgraph-search --f2-dir ~/.local/share/steppe/example \
  --pops England_BellBeaker,Czechia_EBA_CordedWare,Turkey_N,Mbuti,Han,Karitiana \
  --max-nadmix 1 --numstart 10 --format json
```

---

## From genotypes — needs your own data

These read a genotype prefix (`PREFIX.{geno,snp,ind}`); download a panel first
(see [Get data](#get-data) below and [data and formats](./data-and-formats.md)).

```bash
# build an f2 cache (the "compute once" step) — pick pops by count, keep >=50% covered SNPs
steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB \
  --auto-top-k 700 --maxmiss 0.5 --out f2_dir

# preview cost first (SNPs kept, blocks, tier, output size) — no compute
steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB --auto-top-k 700 --maxmiss 0.5 --dry-run

# then run any f2-based command above against ./f2_dir instead of the example:
steppe qpadm --f2-dir f2_dir --target ... --left ... --right ...

# --- measured: building an f2 cache from a real AADR 1240K panel ---
# 77-population worldwide set, 1,117,641 of 1,233,013 SNPs kept, 713 jackknife blocks:
#   ~6.1 s  (peak host RAM ~5.5 GB, GPU ~13.7 GB) on one RTX 5090, emu40.
# It only reads the selected populations' individuals (TGENO is individual-major), so
# a small pop set off a huge panel stays cheap.

# qpfstats — build a SMOOTHED f2 cache (feed it to qpadm/f4/qpgraph like any f2 dir)
steppe qpfstats --prefix v66.p1_HO.aadr.patch.PUB \
  --pops Czechia_EBA_CordedWare,Russia_Samara_EBA_Yamnaya,Turkey_N,Mbuti,Han,Papuan,Karitiana \
  --out-dir smoothed_f2

# dates — admixture dating (needs a real cM genetic map in the .snp)
steppe dates --prefix v66.p1_HO.aadr.patch.PUB \
  --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N
```

---

## VCF ingestion — genotype a sample against a panel

`steppe ingest` reads one sample's `.vcf.gz` (gVCF ref-blocks included) and genotypes it at a
panel's sites, harmonizing strand/allele polarity. Point it at an AADR `.snp` panel plus a GRCh38
FASTA and an `rsID -> pos38` lift map (native mode), or a pre-built 7-column target table (legacy
mode). It emits any mix of: a per-site report, the built target table, a canonical 2-bit tile, or
the sample merged into the panel as a size-1 population (which then feeds `extract-f2`). Host-only
except `--emit-tile` (the one GPU seam). See [data and formats](./data-and-formats.md).

```bash
# native: genotype a sample at panel sites -> per-site report (no GPU)
steppe ingest --vcf sample.vcf.gz \
  --panel v66.p1_1240K.aadr.patch.PUB.snp --fasta GRCh38.fa --lift rsid_pos38.tsv \
  --sample MYSAMPLE --report sample_report.tsv

# build only the GRCh38 target table (no --vcf needed)
steppe ingest --panel v66.p1_1240K.aadr.patch.PUB.snp --fasta GRCh38.fa --lift rsid_pos38.tsv \
  --emit-targets native_targets.tsv

# merge the sample into a panel as a size-1 population (then extract-f2 the merged prefix)
steppe ingest --vcf sample.vcf.gz \
  --panel v66.p1_1240K.aadr.patch.PUB.snp --fasta GRCh38.fa --lift rsid_pos38.tsv \
  --sample MYSAMPLE \
  --merge-into v66.p1_1240K.aadr.patch.PUB --emit-merged merged_panel

# legacy: a pre-built 7-col target table instead of panel/fasta/lift
steppe ingest --vcf sample.vcf.gz --targets targets.tsv --report sample_report.tsv

# bit-exact canonical 2-bit tile (the one GPU path)
steppe ingest --vcf sample.vcf.gz --targets targets.tsv --emit-tile sample.tile --device 0
```

Tune the gVCF confidence floors with `--min-dp` (default 8) and `--min-gq` (default 20).
`ingest-concord` (GPU-free) diffs an ingest report against an oracle dosage table and asserts
per-site `{call, dosage, source, drop_reason}` match:

```bash
steppe ingest-concord --a sample_report.tsv --b oracle_dosage.tsv
```

> **Measured** (one RTX 5090, host-only step). Genotyping a **30x WGS gVCF** at the AADR
> 1240K sites (GRCh38 FASTA + rsID→pos38 lift map, ~1.1M sites emitted) takes **~16.8 s** to a
> per-site report, or **~19.0 s** with `--merge-into` (which also writes the full ~7.1 GB merged
> TGENO panel). Peak host RAM ~0.65 GB; no GPU used. Single timed run, `/usr/bin/time`.

### GL mode — read per-site genotype likelihoods (`--likelihoods`)

`--likelihoods` switches ingest into a **multi-sample** GL reader: it pulls one FORMAT
likelihood field (`--gl-field PL|GL|GP`, default `PL`) into a normalized
`[n_site × n_sample × 3]` likelihood tensor — the substrate the `pcangsd` and `ibd` paths
consume. Unlike the genotyping path it does NOT apply the FILTER/DP/GQ floor (a GL is soft
information). `--emit-likelihoods` writes the `STPGL1` tensor artifact and uploads it
device-resident with a residency checksum (the one GPU seam in GL mode).

```bash
# read PL into a normalized GL tensor at a target table's sites, write the STPGL1 artifact
steppe ingest --likelihoods --vcf sample.vcf.gz --targets targets.tsv \
  --gl-field PL --emit-likelihoods sample.stpgl1 --device 0
```

> **Measured** (one RTX 5090, `--device 0`). Reading `PL` from a **30x WGS gVCF**
> (13.9M records) joined to the 1.1M-site 1240K target table — 475,265 sites with PL present →
> a device-resident GL tensor — takes **~12.3 s** (peak host RAM ~0.72 GB). The wall is
> dominated by scanning the 13.9M-record VCF; the tensor upload is device-resident. Median of 3
> timed reps after one warmup, `/usr/bin/time`.

---

## Relatedness — READv2 kinship (needs your own data)

`steppe readv2` detects related pairs from a genotype prefix (`PREFIX.{geno,snp,ind}`, read as
pseudo-haploid), via an all-pairs windowed-mismatch sweep on the GPU. Every individual is its own
group; the output is an 8-column per-pair table with a relatedness degree + normalized P0.
Autosomes only by default (READv2 convention).

```bash
# all-pairs relatedness over a panel (median-normalized 1000-SNP windows)
steppe readv2 --prefix v66.p1_1240K.aadr.patch.PUB \
  --window-snps 1000 --norm median --auto-only --out relatedness.csv

# restrict to a subset of Genetic IDs, drop thin pairs (<10% comparable sites)
steppe readv2 --prefix v66.p1_1240K.aadr.patch.PUB \
  --samples keep_ids.txt --min-overlap 0.1 --out relatedness.csv

# huge cohort: stream per-pair rows to shard files, lift the C(N,2) safety cap
steppe readv2 --prefix v66.p1_1240K.aadr.patch.PUB --shard-dir ./readv2_out --sure
```

> **Measured** (one RTX 5090, `--samples` subset of a real AADR 1240K panel). 38 samples
> (703 pairs) → **~1.0 s**; 1,000 samples (499,500 pairs) → **~2.1 s**. Peak host RAM ~0.5–0.7 GB,
> GPU ~0.7–1.0 GB. `--samples` reads only the kept individuals' columns, so the panel size barely
> matters — the all-pairs GPU sweep dominates. Single timed run, `/usr/bin/time`.

`readv2-concord` (GPU-free) diffs one READv2 table against a reference and asserts the degree
confusion matrix + P0 concordance — the "READv2 ruler" for validating a run:

```bash
steppe readv2-concord --a relatedness.csv --b reference.readv2.csv \
  --degree-agreement-min 0.95 --p0-within-tol-min 0.90 --coverage-min 1.0
```

### KING-robust kinship — the diploid counterpart (`steppe kinship`)

`steppe readv2` is pseudo-haploid only. For **diploid** data (consumer / biobank / modern
panels) use `steppe kinship`, the GPU KING-robust between-family estimator (Manichaikul et al.
2010). It reuses the same decode-once + all-pairs GPU sweep, but the unit is a per-individual
Genetic ID and it consumes the het signal: per pair it counts hetHet, opposite-homozygotes
(IBS0), and each side's hets over the shared non-missing autosomal sites, then reports
`phi = (hetHet − 2·IBS0) / (het_i + het_j)` and a degree band (`dup / 1st / 2nd / 3rd /
unrelated`). Duplicate/MZ → φ ≈ 0.5, unrelated → φ ≈ 0. Gated vs `plink2 --make-king-table`.

```bash
# all-pairs KING over a diploid panel
steppe kinship --prefix diploid_panel --all-pairs --out kinship.tsv

# biobank scale: only emit related pairs, or run a targeted pair list
steppe kinship --prefix diploid_panel --all-pairs --min-kinship 0.0442 --out related.tsv
steppe kinship --prefix diploid_panel --pairs pairs.txt --out kinship.tsv
```

The output is a plink2 `.kin0`-shaped per-pair table: `id1 id2 nsnp hethet ibs0 phi degree`
(one row per pair; `id1 ≤ id2`). `--all-pairs` is the C(N,2) sweep (`--sure` lifts the maxcomb
cap); `--pairs FILE` (an `id1<ws>id2` list) is the targeted / large-N path.

---

## Chromosome painting — Li-Stephens haplotype copying (needs your own data)

`steppe paint` runs a GPU Li-Stephens haplotype-copying forward-backward: it paints each
**recipient** haplotype (`--prefix`) as a mosaic of copied stretches from a **donor** panel
(`--donors`), then summarizes the copying posterior two ways. Both triples must be
**PRE-PHASED HAPLOID** — every haploid column is one haplotype, and steppe ships no phaser, so a
diploid input is refused up front. Recipients and donors must share one `.snp` marker set with a
real cM genetic map. Point `--prefix` and `--donors` at the same panel to paint it against itself
(all-vs-all, self donor left out).

Two faces:

- **paint** (default) — the ChromoPainter-style *coancestry* summary: expected copied `chunks` +
  `length_cM` per recipient, aggregated by donor ancestry/pop `--labels` (add `--full` for the
  per-donor matrix instead).
- **`--face localanc`** — the per-SNP *local-ancestry* posterior: for each recipient, at each
  marker, the probability it copied from each ancestry label (carries `chrom`/`pos_bp` so a
  FLARE/RFMix-style aligner needs no external join).

```bash
# coancestry: paint a set of recipients against a labelled donor panel, aggregate by label
steppe paint --prefix recipients_phased --donors donor_panel_phased \
  --labels donor_labels.txt --Ne 20000 --theta auto --out coancestry.csv

# panel-vs-self all-vs-all (self haplotype left out automatically), full per-donor matrix
steppe paint --prefix panel_phased --donors panel_phased \
  --full --recip-batch 256 --out coancestry_full.csv

# local ancestry: per-SNP ancestry posterior for each recipient
steppe paint --prefix recipients_phased --donors donor_panel_phased \
  --labels donor_labels.txt --face localanc --format json --out localanc.json
```

`--labels` is one label per donor **haplotype column**, in `.ind` order (for phased diploid donors
that is two identical entries per individual); omit it to fall back to `.ind` population labels.
Tune the copying model with `--Ne` (recombination scale, default 20000) and `--theta`
(emission/miscopy rate, default `auto` = Watterson over the K donors). `--self-copy` lets a
haplotype copy itself (default off = leave-one-out in the panel-vs-self case). `--recip-batch`
(default 256) is the VRAM knob — recipient haplotypes resident per GPU wave; the full K×M posterior
never leaves the device. `--sure` lifts the O(N·K·M) cost guard for a long job.

> **Measured** (one RTX 5090, real phased 1000G haplotype panels). A coancestry `paint` run
> (recipients × a labelled donor panel of a few thousand SNPs) and a per-SNP `--face localanc`
> run both land around **~0.45 s** (peak host RAM ~0.44 GB, GPU ~0.5–0.7 GB). Median of 3 timed
> reps after one warmup, `/usr/bin/time`. Cost scales with recipients × donors × SNPs, so a
> genome-wide painting of a large panel is far larger than these smoke-sized panels.

---

## Population statistics — FST, SFS, PCA (needs your own data)

These read a genotype prefix (`PREFIX.{geno,snp,ind}`, EIGENSTRAT/PLINK/…) and compute a classic
population statistic straight on the GPU. Point `--prefix` at your own panel.

```bash
# Weir & Cockerham 1984 FST between two populations (per-SNP table; else genome-wide summary row)
steppe fst --prefix v66_fit9_ped --pops Han,Papuan --method wc --per-snp --out fst.tsv

# all-pairs: the whole population×population FST matrix in ONE GPU pass (the large-panel win)
steppe fst --all-pairs --prefix v66.p1_1240K.aadr.patch.PUB --min-n 5 --out fst_matrix.tsv

# 2D joint site-frequency spectrum between two populations (--fold = per-pop minor-allele SFS)
steppe sfs --prefix v66_fit9_ped --pops Han,Papuan --fold --out sfs.tsv

# Patterson-2006 genotype PCA, top-k PCs (SYRK covariance + cuSOLVER eigen)
steppe pca --prefix v66_fit9_ped -k 12 --out pca.tsv
```

`fst` is pairwise Weir-Cockerham (`--method wc`; `--per-snp` for the per-SNP
`num/den/fst/valid` table, else the genome-wide ratio-of-averages summary row) — gated vs
`plink2 --fst method=wc`. `--all-pairs` instead computes the full **P×P** genome-wide FST matrix
over the selected populations (`--pops A,B,…` or `--min-n N`; `--sure` lifts the pair-count cap)
in one GPU pass — every cell bit-exact vs the single-pair path; this is the large-panel workload
where the GPU pulls ahead. `sfs` builds the integer `(2nA+1)×(2nB+1)` joint spectrum over
complete-data sites (`--fold` for the polarity-free per-pop minor-allele SFS, default unfolded) —
gated bit-exact vs scikit-allel `joint_sfs`. `pca` takes `-k` PCs and optionally `--eigenvalues`
(the scree table) or `--emit-html` (a self-contained interactive scatter) — gated vs
scikit-allel/sklearn PCA (NOT ADMIXTOOLS 2). Its eigensolve is a Halko randomized top-k solver, so
PCA now runs the **full AADR cohort** (23,089 samples × 1.23M SNPs, ~6m24s, 18 GB peak) — the
old full-spectrum solver OOM'd there.

> **Measured** (one RTX 5090, `--device 0`, on a real AADR v66 fixture — 430 samples ×
> ~584k variants; Han vs Papuan). `fst` (per-SNP, 425,234 valid sites) ~**3.15 s**; `sfs`
> (folded 154×47 over 380,265 complete sites) ~**2.17 s**; `pca` (`-k 12` over 522,483
> polymorphic SNPs) ~**2.53 s** (peak host RAM ~0.5–0.7 GB). Median of 3 timed reps after one
> warmup, `/usr/bin/time`. HONEST NOTE: at this gate size a mature multithreaded CPU tool wins —
> `plink2 --fst` finishes sub-second on the same pair — because steppe pays a fixed
> GPU-context-init + full-decode cost per run. steppe's GPU advantage is a large-model / scale
> story, not this small fixture — and that story is **`--all-pairs`**: the whole pop×pop matrix
> runs **29.5× faster than plink2** at 502 populations (125,751 pairs), and the entire **3,898-pop
> AADR matrix (7.6M pairs) in ~22 min** vs plink2's projected ~18.5 hr (crossover ~60 pops — below
> it plink2 still wins).

---

## Low-coverage & ancient DNA — PCAngsd, IBD, ROH (needs your own data)

The genotype-likelihood / imputed-haplotype family for low-coverage and ancient samples. Each
matches a specific reference tool (gated) and is self-contained (no f2 dir).

```bash
# PCAngsd: low-coverage PCA from a beagle GL file, top-e PCs (iterative IAF-EM + GL covariance)
steppe pcangsd --beagle input.beagle.gz -e 2 --out pcangsd_out

# ancIBD: IBD-segment detection between imputed ancient individuals (phased GT + GP)
steppe ibd --gp-vcf imputed.vcf.gz --targets targets.tsv --map map.cM --af panel_af.tsv \
  --min-cm 8 --out ibd.tsv

# hapROH: runs-of-homozygosity for ancient target(s) vs a phased reference panel
steppe roh --prefix TARGET_triple --ref-panel phased_panel_triple --n-ref 2504 --out roh.csv
# ...shard an ancient cohort across GPUs (targets split across the listed devices, output identical):
steppe roh --prefix cohort_triple --ref-panel phased_panel_triple --device 0,1,2,3 --out roh.csv
```

`pcangsd` reads a 3-GL-per-individual beagle file, runs the individual-allele-frequency EM
(`--iter`/`--tole`), builds a GL-weighted covariance and takes `-e` PCs, writing
`PREFIX.{cov,eigenvec,eigenval}` — gated vs the `pcangsd` package. `ibd` runs a per-pair 5-state
forward-backward over imputed GP at target sites (needs `--map` in cM and a panel `--af`),
calling segments ≥ `--min-cm` — gated vs `pip ancIBD`. `roh` runs the (K+1)-state copying FB of
ancient target(s) against a phased reference panel (`--n-ref` reference individuals ⇒
K = 2·n_ref haplotypes; the default 2504 is safe, a very large `--n-ref` above ~5000 is
rejected up front), calling ROH segments — byte-identical to `pip hapROH/hapsburg`. A cohort runs
as a **wave-batched grid** (hundreds of ancients resident on the GPU at once) with a
**multi-threaded host prep**; pass a comma-separated **device list** (`--device 0,1,2,3`) to shard
the cohort across GPUs — task-parallel, output byte-identical to one GPU. `STEPPE_ROH_THREADS` caps
the prep thread pool and `STEPPE_ROH_TIMING=1` prints the per-stage prep timing.

> **Measured** (one RTX 5090, `--device 0`). `pcangsd` on the popgen.dk Demo2 beagle
> (100 low-coverage samples × ~50k SNPs, `-e 2`, 100 plain-EM FP64 iters) ~**3.66 s** (peak host
> RAM ~0.72 GB; median of 3 after warmup). `ibd` on the ancIBD Hazelton chr20 tutorial
> (28.5k sites × 15 pairs) ~**1.8 s** for the GPU forward-backward — this figure is from the gate
> run (that dataset was since cleaned off the disk-tight box, so it is not a fresh re-time).
> `roh` (real 1000G chr3 panel, K=5008, one RTX 5090): a **512-ancient cohort in ~27 s**
> (~1,100 ancients/min), and it shards across GPUs — a 2,048-ancient cohort runs 92.6 s (1 GPU) →
> 57 s (2) → 37 s (4), byte-identical at every device count. The win is scale-shaped: the
> wave-batched grid lifted the kernel off ~1% GPU utilization, and multi-GPU adds a further
> ~1.6–2.5× (task-parallel, capped by the shared read/decode prep, not the GPUs). HONEST NOTE: on the
> tiny Demo2 fixture the reference `pcangsd` (10 SQUAREM float32 iters, 16 CPU threads) finishes
> ~1 s — steppe runs more, heavier FP64 iterations to the same fixed point, so it is slower on
> this smoke size; the GPU win is a scale story.

---

## Interop — move an f2 cache to/from ADMIXTOOLS 2 (`.rds`)

`steppe-rds` (installed alongside the CLI, GPU-free) converts between steppe's f2 dir and
ADMIXTOOLS 2's `read_f2()` `.rds` format, so you can cross-check a fit in R. Import needs
`pip install steppe[rds]`. See [data and formats](./data-and-formats.md).

```bash
# steppe f2 dir  ->  an AT2 read_f2() .rds dir you can open in R
steppe-rds export ~/.local/share/steppe/example ./exported_rds

# an AT2 .rds dir  ->  a steppe f2 cache
steppe-rds import ./some_at2_rds_dir ./imported_f2_dir
```

---

## Cache / dataset manager — inspect and verify f2 caches

An f2 cache is a directory (`f2.bin` + `pops.txt` + an optional `meta.json`). The built-in
`steppe cache` subcommand inspects them header-only, never touching the multi-GB payload:

```bash
steppe cache ls   ./caches                   # tabulate every cache under a root
steppe cache show ./caches/aadr_1240K_f2     # header facts + integrity mark + meta.json
steppe cache verify ./caches/aadr_1240K_f2   # re-hash the payload against its content-address
```

> **Measured** (host-only, no GPU). `cache ls` over a directory of caches ~**0.10 s**, `cache show`
> ~**0.09 s**, and `cache verify` (re-hashing a 64.5 MB f2 payload) ~**0.15 s**; peak host RAM
> ~0.26 GB. These are header/payload reads, so they stay sub-second regardless of population count.
> Median of 3 timed reps after one warmup, `/usr/bin/time`.

`steppe-cache` (installed alongside the CLI, GPU-free, stdlib-only) does the same plus pop
listing, panel discovery, and AADR fetch:

```bash
steppe-cache ls ./caches --long --sort size                # tabulate (--index memoizes the scan)
steppe-cache show ./caches/aadr_1240K_f2 --json            # one cache's record as JSON
steppe-cache pops ./caches/aadr_1240K_f2                   # population labels, one per line
steppe-cache verify ./caches/aadr_1240K_f2 --check-sources # also re-hash geno/snp/ind
steppe-cache datasets --dir .                              # which AADR panels (1240K|HO|2M) are present
steppe-cache get 1240K ./aadr                             # fetch an AADR panel (wraps download-aadr.sh)
```

From Python (GPU-free):

```python
import steppe
steppe.list_caches("./caches")                # a record per cache under a root
steppe.cache_info("./caches/aadr_1240K_f2")   # full parsed record (like `show --json`)
steppe.verify_cache("./caches/aadr_1240K_f2", check_sources=True)   # True iff all checks pass
```

---

<a name="get-data"></a>
## Get data — AADR panels (Harvard Dataverse, doi:10.7910/DVN/FFIDCW)

Each panel is a `.geno`/`.snp`/`.ind` triple. `-C -` resumes a partial download; the `.geno` is
the big file. (These are v66 file IDs; see [data and formats](./data-and-formats.md) for the
resumable, version-proof download script.)

```bash
# HO panel (~4.0 GB) — smallest, fastest to build
curl -L -C - -o v66.p1_HO.aadr.patch.PUB.geno https://dataverse.harvard.edu/api/access/datafile/13994808
curl -L -C - -o v66.p1_HO.aadr.patch.PUB.snp  https://dataverse.harvard.edu/api/access/datafile/13994527
curl -L -C - -o v66.p1_HO.aadr.patch.PUB.ind  https://dataverse.harvard.edu/api/access/datafile/13994526

# 1240K panel (~7.1 GB)
curl -L -C - -o v66.p1_1240K.aadr.patch.PUB.geno https://dataverse.harvard.edu/api/access/datafile/13994829
curl -L -C - -o v66.p1_1240K.aadr.patch.PUB.snp  https://dataverse.harvard.edu/api/access/datafile/13994514
curl -L -C - -o v66.p1_1240K.aadr.patch.PUB.ind  https://dataverse.harvard.edu/api/access/datafile/13994513

# 2M panel (~12 GB) — most SNPs
curl -L -C - -o v66.p1_2M.aadr.patch.PUB.geno https://dataverse.harvard.edu/api/access/datafile/13994830
curl -L -C - -o v66.p1_2M.aadr.patch.PUB.snp  https://dataverse.harvard.edu/api/access/datafile/13994517
curl -L -C - -o v66.p1_2M.aadr.patch.PUB.ind  https://dataverse.harvard.edu/api/access/datafile/13994516
```

Then feed the prefix (the path without the extension) to `extract-f2` above.
