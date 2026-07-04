# Command cheatsheet

Copy-paste one-liners for every steppe subcommand. They assume `steppe` is installed and on your
PATH — see [getting started](./README.md).

The **f2-based** examples run against the bundled 9-population example that the installer stages at
`~/.local/share/steppe/example_9pop`, so they work the moment you've installed steppe. The
**genotype-based** ones (`extract-f2`, `qpfstats`, `dates`) need your own data — see
[data and formats](./data-and-formats.md).

> The bundled example contains: `England_BellBeaker`, `Czechia_EBA_CordedWare`, `Turkey_N`,
> `Mbuti`, `Han`, `Papuan`, `Karitiana`, `Iran_GanjDareh_N`, `Israel_Natufian`.

Every command takes `--device <n>` (default `0`) and, where it produces a result,
`--format csv|tsv|json` (default `csv`) and `--out FILE`. They're omitted below for brevity.

---

## qpAdm — fit an admixture model

Target as a mixture of `--left` sources, tested against the `--right` outgroups. See [qpadm](./qpadm.md).

```bash
steppe qpadm --f2-dir ~/.local/share/steppe/example_9pop \
  --target England_BellBeaker \
  --left  Czechia_EBA_CordedWare,Turkey_N \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian \
  --jackknife 2 --format json
```

## qpWave — rank test (no target)

Tests how many ancestry streams relate the `--left` set to the outgroups; `left[0]` is the
reference. See [qpwave](./qpwave.md).

```bash
steppe qpwave --f2-dir ~/.local/share/steppe/example_9pop \
  --left  Czechia_EBA_CordedWare,Turkey_N,England_BellBeaker \
  --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian
```

## qpAdm-rotate — many competing models in one pass

Enumerates source subsets of `--pool` for one target in a single batched GPU run. See
[qpadm-rotate](./qpadm-rotate.md).

```bash
steppe qpadm-rotate --f2-dir ~/.local/share/steppe/example_9pop \
  --target England_BellBeaker \
  --pool   Czechia_EBA_CordedWare,Turkey_N,Iran_GanjDareh_N,Israel_Natufian \
  --right  Mbuti,Han,Papuan,Karitiana \
  --min-sources 1 --max-sources -1 --format csv
```

---

## Standalone f-statistics

See [f-statistics](./f-statistics.md).

```bash
# f4(p1,p2 ; p3,p4)
steppe f4 --f2-dir ~/.local/share/steppe/example_9pop \
  --pop1 England_BellBeaker --pop2 Czechia_EBA_CordedWare --pop3 Han --pop4 Iran_GanjDareh_N

# f3(C ; A, B)
steppe f3 --f2-dir ~/.local/share/steppe/example_9pop \
  --pop1 Czechia_EBA_CordedWare --pop2 England_BellBeaker --pop3 Turkey_N

# f4-ratio  alpha = f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)
steppe f4-ratio --f2-dir ~/.local/share/steppe/example_9pop \
  --pops Czechia_EBA_CordedWare,Mbuti,Turkey_N,Han,England_BellBeaker

# D-statistic (f2-path: reports f4 with Z / p)
steppe qpdstat --f2-dir ~/.local/share/steppe/example_9pop \
  --pop1 Mbuti --pop2 Han --pop3 Czechia_EBA_CordedWare --pop4 Turkey_N
```

---

## Sweeps — score every combination on the GPU

Enumerate every quartet / triple over a pop set, keep only survivors (`--top-k` or `--min-z`).
`--sure` lifts the safety cap on huge sweeps. See [sweeps](./sweeps.md).

```bash
# every f4 quartet over the example (C(9,4) = 126), keep the top 50 by |z|
steppe f4-sweep --f2-dir ~/.local/share/steppe/example_9pop --top-k 50

# every f3 triple, keep |z| >= 3
steppe f3-sweep --f2-dir ~/.local/share/steppe/example_9pop --min-z 3

# the same sweep via f4/f3 with --all-quartets/--all-triples + sharded CSV output
steppe f4 --all-quartets --f2-dir ~/.local/share/steppe/example_9pop \
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
steppe qpgraph --f2-dir ~/.local/share/steppe/example_9pop --graph mygraph.txt --format json

# search topologies over a bounded leaf set (>= 3 pops)
steppe qpgraph-search --f2-dir ~/.local/share/steppe/example_9pop \
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

# qpfstats — build a SMOOTHED f2 cache (feed it to qpadm/f4/qpgraph like any f2 dir)
steppe qpfstats --prefix v66.p1_HO.aadr.patch.PUB \
  --pops Czechia_EBA_CordedWare,Russia_Samara_EBA_Yamnaya,Turkey_N,Mbuti,Han,Papuan,Karitiana \
  --out-dir smoothed_f2

# dates — admixture dating (needs a real cM genetic map in the .snp)
steppe dates --prefix v66.p1_HO.aadr.patch.PUB \
  --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N
```

---

## Interop — move an f2 cache to/from ADMIXTOOLS 2 (`.rds`)

`steppe-rds` (installed alongside the CLI, GPU-free) converts between steppe's f2 dir and
ADMIXTOOLS 2's `read_f2()` `.rds` format, so you can cross-check a fit in R. Import needs
`pip install steppe[rds]`. See [data and formats](./data-and-formats.md).

```bash
# steppe f2 dir  ->  an AT2 read_f2() .rds dir you can open in R
steppe-rds export ~/.local/share/steppe/example_9pop ./exported_rds

# an AT2 .rds dir  ->  a steppe f2 cache
steppe-rds import ./some_at2_rds_dir ./imported_f2_dir
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
