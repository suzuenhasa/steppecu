# f-statistics: f4, f3, f4-ratio, qpdstat

Four related point statistics you can compute on their own, without fitting a
model: `f4`, `f3`, `f4-ratio`, and `qpdstat`. Each takes an f2 cache (or, for
`qpdstat`, raw genotypes) and reports an estimate, a standard error, a z-score,
and — for most of them — a p-value.

These are the "just give me the number" tools. If you want to fit an admixture
model instead, see [qpadm](./qpadm.md) and [qpwave](./qpwave.md).

Three of the four (`f4`, `f3`, `f4-ratio`) and one mode of `qpdstat` read a
**precomputed f2 cache** — a directory built once by
[extract-f2](./extract-f2.md). Build that first, then point these commands at it
with `--f2-dir`. The only exception is `qpdstat --prefix`, which reads raw
genotype files directly (see that subsection).

steppe reproduces ADMIXTOOLS 2 numerically, so the formulas, the block-jackknife
error, and the sign/z/p conventions below all match what AT2 reports.

---

## f4

`f4(p1, p2; p3, p4)` for one or many population quartets — estimate, SE, z, p.

### What it does

For each quartet of populations it computes a single f4 value and its
block-jackknife standard error. There is no model fitting and no rank test — just
the point estimate and its error, batched over every quartet you ask for. The
estimate comes straight from four f2 values:

```
f4(p1,p2;p3,p4) = 0.5 * ( f2(p2,p3) + f2(p1,p4) - f2(p1,p3) - f2(p2,p4) )
```

Reach for it to test treeness / gene flow: an f4 that is significantly non-zero
(large `|z|`) is evidence that the four populations do not fit a simple tree. You
give the quartets one of three ways — column-parallel with `--pop1..--pop4`, as
flat groups of four with `--pops`, or sweep all quartets with `--all-quartets`.

### Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | The f2_blocks directory built by extract-f2. Required. | — |
| `--pop1 TEXT ...` | Quartet column 1 (p1). Row-aligned with `--pop2/3/4`. | — |
| `--pop2 TEXT ...` | Quartet column 2 (p2). | — |
| `--pop3 TEXT ...` | Quartet column 3 (p3, the f4 "R0"). | — |
| `--pop4 TEXT ...` | Quartet column 4 (p4, the f4 "R1"). | — |
| `--pops TEXT ...` | Quartet(s) as names in flat groups of 4: `p1,p2,p3,p4[,p1,p2,p3,p4,...]`. Alternative to the four `--popN` lists. | — |
| `--all-quartets` | Sweep every quartet `C(P,4)` over the `--pops` subset (empty `--pops` means the whole f2 dir). | off |
| `--min-z FLOAT` | Sweep only: keep items with `|z| >=` this. On-device filter. Mutually exclusive with `--top-k`. | 3.0 |
| `--top-k INT` | Sweep only: keep the K items with the largest `|z|` (bounded device-side reservoir). Mutually exclusive with `--min-z`. | — |
| `--sure` | Sweep only: lift the safety cap on the number of combinations. A sweep bigger than the cap refuses to run without this. | off |
| `--shard-dir TEXT` | Sweep only: write the survivor table to a CSV under this directory (created if absent) instead of stdout/`--out`. | — |
| `--out TEXT` | Output file. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | csv |
| `--device TEXT` | CUDA device(s): `auto`, `<ordinal>`, or `<ordinal>,<ordinal>`. GPU only — there is no `cpu`. | auto |
| `--precision TEXT` | Matmul precision for the covariance step: `emu40`, `emu32`, `fp64`, `tf32`. The estimate itself is always native FP64. | emu40 |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

### Examples

A single quartet:

```
steppe f4 --f2-dir /path/to/steppe_f2 \
  --pop1 England_BellBeaker --pop2 Czechia_EBA_CordedWare \
  --pop3 Han --pop4 Iran_GanjDareh_N \
  --format csv --device 0
```

Expect one row: `est`, `se`, `z`, `p` for that quartet.

Sweep every quartet in a 500-population f2 dir, keep the 1,000,000 most extreme
by `|z|`, write shards to disk:

```
steppe f4 --all-quartets --f2-dir /path/to/f2_500 \
  --top-k 1000000 --sure --shard-dir /tmp/sweep500 --device 0
```

The `--sure` is needed because `C(500,4)` is far past the safety cap.

### Gotchas

- Needs an f2 cache first — build one with [extract-f2](./extract-f2.md) and pass
  it as `--f2-dir`.
- **The quartet order matters.** `f4(p1,p2;p3,p4)` changes sign and meaning when
  you reorder the four populations — this is not a symmetric statistic.
- `--pop1..--pop4` are row-aligned columns: the k-th quartet is
  `(pop1[k], pop2[k], pop3[k], pop4[k])`. Give them equal lengths, or use `--pops`
  in flat groups of four instead. Do not mix the two styles.
- `--min-z` and `--top-k` are mutually exclusive — pick one to bound a sweep.
- The safety cap on `--all-quartets` is deliberate: a big population set is a huge
  number of quartets. You must pass `--sure` to run past it.
- `--min-z`/`--top-k`/`--sure`/`--shard-dir` only do anything in sweep mode
  (`--all-quartets`); they are ignored for an explicit quartet list.
- There is a dedicated `f4-sweep` command for the all-quartets case as well;
  `f4 --all-quartets` is the same sweep exposed on the point-statistic command.

---

## f3

`f3(C; A, B)` for one or many population triples — estimate, SE, z, p.

### What it does

The three-population sibling of f4. For each triple it reports one value and its
block-jackknife error, from three f2 values:

```
f3(C; A, B) = 0.5 * ( f2(C,A) + f2(C,B) - f2(A,B) )
```

It is symmetric in A and B (swapping them gives the same number). There are two
common ways to use it, and only the interpretation differs:

- **Outgroup-f3** — `f3(Outgroup; A, B)`: how much drift A and B share, seen from
  a distant outgroup. Bigger means more shared history.
- **Admixture-f3** — `f3(Target; Src1, Src2)`: a **negative** value is evidence
  that the target is a mixture of the two sources.

### Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | The f2_blocks directory built by extract-f2. Required. | — |
| `--pop1 TEXT ...` | Triple column 1 (C, the apex/outgroup/target). Row-aligned with `--pop2/3`. | — |
| `--pop2 TEXT ...` | Triple column 2 (A). | — |
| `--pop3 TEXT ...` | Triple column 3 (B). | — |
| `--pops TEXT ...` | Triple(s) as names in flat groups of 3: `C,A,B[,C,A,B,...]`. Alternative to the three `--popN` lists. | — |
| `--all-triples` | Sweep every triple `C(P,3)` over the `--pops` subset (empty `--pops` means the whole f2 dir). | off |
| `--min-z FLOAT` | Sweep only: keep items with `|z| >=` this. Mutually exclusive with `--top-k`. | 3.0 |
| `--top-k INT` | Sweep only: keep the K items with the largest `|z|`. Mutually exclusive with `--min-z`. | — |
| `--sure` | Sweep only: lift the combination-count safety cap. | off |
| `--shard-dir TEXT` | Sweep only: write the survivor CSV under this directory instead of stdout/`--out`. | — |
| `--out TEXT` | Output file. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | csv |
| `--device TEXT` | CUDA device(s): `auto`, `<ordinal>`, or `<ordinal>,<ordinal>`. GPU only. | auto |
| `--precision TEXT` | Matmul precision for the covariance step: `emu40`, `emu32`, `fp64`, `tf32`. The estimate is always native FP64. | emu40 |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

### Examples

A single triple (here read as admixture-f3 — a negative `est` is the signal):

```
steppe f3 --f2-dir /path/to/steppe_f2 \
  --pop1 Czechia_EBA_CordedWare \
  --pop2 Russia_Samara_EBA_Yamnaya --pop3 Turkey_N --device 0
```

Sweep every triple in a 500-pop dir, keep the top 100,000 by `|z|`:

```
steppe f3 --all-triples --f2-dir /path/to/f2_500 \
  --top-k 100000 --sure --shard-dir /tmp/f3sweep500 --device 0
```

### Gotchas

- Needs an f2 cache first — see [extract-f2](./extract-f2.md).
- **The apex (C, `--pop1`) is special; A and B are interchangeable.** For
  admixture-f3, C is the target and A/B are the candidate sources. Put the right
  population in `--pop1`.
- `--min-z` and `--top-k` are mutually exclusive.
- `--all-triples` obeys the same safety cap as f4's sweep — pass `--sure` to run a
  large sweep, and the sweep-only flags do nothing for an explicit triple list.
- There is also a dedicated `f3-sweep` command; `f3 --all-triples` is the same
  sweep on the point command.

---

## f4-ratio

The admixture-proportion statistic `alpha = f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)`.

### What it does

An f4-ratio divides one f4 by another that shares three of its four populations
(`p1`, `p2`, `p4`) — only the third slot differs (`p3` numerator, `p5`
denominator). The result `alpha` estimates an admixture proportion. It reports
`alpha`, `se`, and `z` — there is deliberately **no p column**, matching AT2's
`qpf4ratio` output.

Each input is a 5-tuple `(p1, p2, p3, p4, p5)`. `alpha` is not a plain ratio of
totals — it is a weighted block-jackknife of the per-block ratios, which is what
gives you a meaningful standard error.

### Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | The f2_blocks directory built by extract-f2. Required. | — |
| `--pop1 TEXT ...` | 5-tuple column 1 (p1). Row-aligned with `--pop2/3/4/5`. | — |
| `--pop2 TEXT ...` | 5-tuple column 2 (p2). | — |
| `--pop3 TEXT ...` | 5-tuple column 3 (p3, the numerator's 3rd slot). | — |
| `--pop4 TEXT ...` | 5-tuple column 4 (p4, the shared 4th slot). | — |
| `--pop5 TEXT ...` | 5-tuple column 5 (p5, the denominator's 3rd slot). | — |
| `--pops TEXT ...` | Tuple(s) as names in flat groups of 5: `p1,p2,p3,p4,p5[,...]`. Alternative to the five `--popN` lists. | — |
| `--out TEXT` | Output file. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | csv |
| `--device TEXT` | CUDA device(s): `auto`, `<ordinal>`, or `<ordinal>,<ordinal>`. GPU only. | auto |
| `--precision TEXT` | Matmul precision: `emu40`, `emu32`, `fp64`, `tf32`. `alpha` is always native FP64. | emu40 |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

### Examples

One 5-tuple, passed as a flat group of five with `--pops`:

```
steppe f4-ratio --f2-dir /path/to/steppe_f2 \
  --pops Czechia_EBA_CordedWare,Mbuti,Russia_Samara_EBA_Yamnaya,Han,Turkey_N \
  --device 0
```

Expect one row: `alpha`, `se`, `z`. Here the tuple is
`(p1=Czechia_EBA_CordedWare, p2=Mbuti, p3=Russia_Samara_EBA_Yamnaya, p4=Han,
p5=Turkey_N)`.

### Gotchas

- Needs an f2 cache first — see [extract-f2](./extract-f2.md).
- **The 5-tuple order is fixed and load-bearing.** The numerator is
  `f4(p1,p2;p3,p4)` and the denominator is `f4(p1,p2;p5,p4)`; `p3` and `p5` are the
  only slots that differ, and `p4` is shared. Getting the order wrong silently
  computes a different ratio.
- No p-value column — that is intentional and matches the reference. Use `z`.
- No sweep mode here: give it explicit 5-tuples (`--pops` in groups of 5, or the
  five row-aligned `--popN` lists). Do not mix the two styles.

---

## qpdstat

A D-type statistic with **two modes**: f2-path f4 (with `--f2-dir`) or
genotype-path normalized-D (with `--prefix`).

### What it does

`qpdstat` has two distinct behaviors depending on which input you give it, and
they report different numbers:

- **`--f2-dir` (f2-path):** reads a precomputed f2 cache and reports **f4** for
  each quartet — the same f4 value as the `f4` command, following AT2's f2-path D
  convention (f4 plus z/p). This is the fast path when you already have an f2 dir.
- **`--prefix` (genotype-path):** reads the raw genotype triple
  `PREFIX.{geno,snp,ind}` directly and reports the **normalized D** magnitude —
  the numerator `(a-b)(c-d)` divided by heterozygosity-style normalization terms,
  jackknifed over blocks (`allsnps=TRUE`). This is a genuinely different number
  from f4: normalized D is typically ~10x larger in magnitude, though the z-scores
  track each other closely.

Reach for `--f2-dir` when you already have an f2 cache and want the fast f4 with a
D-style report. Reach for `--prefix` when you want the classic normalized D
straight from genotypes, matching AT2's `qpdstat` genotype behavior.

### Flags

| flag | what it does | default |
|---|---|---|
| `--f2-dir TEXT` | f2_blocks directory. In this mode qpdstat reports **f4** (the AT2 f2-path convention). Mutually exclusive with `--prefix`. | — |
| `--prefix TEXT` | Genotype triple prefix `PREFIX.{geno,snp,ind}` for the **normalized-D** magnitude (Part B; `allsnps=TRUE` block-jackknife). Mutually exclusive with `--f2-dir`. | — |
| `--pop1 TEXT ...` | Quartet column 1 (p1). Row-aligned with `--pop2/3/4`. | — |
| `--pop2 TEXT ...` | Quartet column 2 (p2). | — |
| `--pop3 TEXT ...` | Quartet column 3 (p3, the "R0"). | — |
| `--pop4 TEXT ...` | Quartet column 4 (p4, the "R1"). | — |
| `--pops TEXT ...` | Quartet(s) as names in flat groups of 4: `p1,p2,p3,p4[,...]`. | — |
| `--all-quartets` | Sweep every quadruple `C(P,4)` over the `--pops` subset (empty means the whole f2 dir). | off |
| `--min-z FLOAT` | Sweep only: keep items with `|z| >=` this. Mutually exclusive with `--top-k`. | 3.0 |
| `--top-k INT` | Sweep only: keep the K items with the largest `|z|`. Mutually exclusive with `--min-z`. | — |
| `--sure` | Sweep only: lift the combination-count safety cap. | off |
| `--shard-dir TEXT` | Sweep only: write the survivor CSV under this directory instead of stdout/`--out`. | — |
| `--out TEXT` | Output file. | stdout |
| `--format TEXT` | Output format: `csv`, `tsv`, or `json`. | csv |
| `--device TEXT` | CUDA device(s): `auto`, `<ordinal>`, or `<ordinal>,<ordinal>`. GPU only. | auto |
| `--precision TEXT` | Matmul precision: `emu40`, `emu32`, `fp64`, `tf32`. The estimate is always native FP64. | emu40 |
| `--config TEXT` | **Reserved — not yet supported** (passing one currently errors). | — |

### Examples

f2-path (reports f4 + z/p) for one quartet, given as a flat group of four:

```
steppe qpdstat --f2-dir /path/to/steppe_f2 \
  --pops Mbuti,Han,Czechia_EBA_CordedWare,Turkey_N --device 0
```

Genotype-path normalized D straight from the genotype triple:

```
steppe qpdstat --prefix /path/to/dataset \
  --pops Mbuti,Han,Czechia_EBA_CordedWare,Turkey_N --device 0
```

(reads `/path/to/dataset.geno`, `.snp`, `.ind`).

### Gotchas

- **`--f2-dir` and `--prefix` are mutually exclusive and report different
  numbers.** `--f2-dir` gives you f4; `--prefix` gives you normalized D (~10x
  larger). Don't compare their estimates directly — compare z-scores.
- The `--f2-dir` mode needs an f2 cache built first (see
  [extract-f2](./extract-f2.md)). The `--prefix` mode does not — it reads
  genotypes directly.
- Genotype-path (`--prefix`) pins ploidy to **diploid** for parity and uses the
  "all SNPs" mask — no max-missingness, no MAF, no monomorphic-dropping filter;
  autosomes-only is on. It does not use the automatic per-sample ploidy detection
  that extract-f2 uses (that could flip the sign of a near-zero D).
- Quartet order matters, exactly as for `f4`.
- The sweep flags (`--all-quartets`, `--min-z`, `--top-k`, `--sure`,
  `--shard-dir`) apply to the f2-path; use `--sure` to run a large sweep.

---

## See also

- [extract-f2](./extract-f2.md) — build the f2 cache that `--f2-dir` needs.
- [qpadm](./qpadm.md) / [qpwave](./qpwave.md) — fit an admixture model instead of
  a single statistic.
- Reference docs (internals and exact semantics):
  [`f4.hpp`](../reference/include_steppe_f4.hpp.md),
  [`f3.hpp`](../reference/include_steppe_f3.hpp.md),
  [`f4ratio.hpp`](../reference/include_steppe_f4ratio.hpp.md),
  [`dstat.hpp`](../reference/include_steppe_dstat.hpp.md).
