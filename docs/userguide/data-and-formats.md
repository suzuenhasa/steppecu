# Getting data: download, formats, and the RDS bridge

How to get genotype data onto disk, which file formats steppe reads, and how to move an f2 cache back and forth with ADMIXTOOLS 2 in R.

## What it does

This is the "before you run anything" page. Every steppe analysis starts from a **genotype triple** on disk — a `.geno`/`.snp`/`.ind` set (or a PLINK `.bed`/`.bim`/`.fam` set) — which you feed to [`extract-f2`](./extract-f2.md) to build the f2 cache that the fit commands read. Here we cover three practical things:

1. **Downloading an AADR panel** with the bundled `docs/download-aadr.sh` script (the Allen Ancient DNA Resource, from Harvard Dataverse).
2. **The genotype formats steppe reads** — EIGENSTRAT, PLINK, ANCESTRYMAP, PACKEDANCESTRYMAP, and the AADR TGENO layout — and how it picks the right reader.
3. **The `steppe-rds` command**, a GPU-free converter that translates an f2 directory to and from ADMIXTOOLS 2's R `.rds` format, so you can re-run a fit in R and compare.

None of this needs a GPU. Only the actual compute (`extract-f2` and the fit commands) touches CUDA.

---

## 1. Downloading AADR data

`docs/download-aadr.sh` fetches one AADR v66 panel from Harvard Dataverse (`doi:10.7910/DVN/FFIDCW`). It resolves the file IDs from the dataset API at run time, so it keeps working when the AADR publishes a new version; downloads are resumable and MD5-checked.

```
bash docs/download-aadr.sh [PANEL] [OUTDIR]
```

| argument | what it does | default |
|---|---|---|
| `PANEL` | Which panel to fetch: `1240K`, `HO`, or `2M`. | `1240K` |
| `OUTDIR` | Where to put the files. | `./aadr_<PANEL>` |

The three panels differ in SNP coverage and size — the `.geno` file is the big one:

| panel | roughly | notes |
|---|---|---|
| `1240K` | ~7 GB | the 1240K capture set |
| `HO` | ~4 GB | Human Origins (fewer SNPs) |
| `2M` | ~12 GB | the ~2-million-SNP set |

Each download lands four files in `OUTDIR` (plus the shared `v66.p1__files.md5sum` manifest used for verification):

```
aadr_1240K/
  v66.p1_1240K.aadr.patch.PUB.geno    # packed genotypes (the big one)
  v66.p1_1240K.aadr.patch.PUB.snp     # one row per SNP
  v66.p1_1240K.aadr.patch.PUB.ind     # one row per individual (pop labels)
  v66.p1_1240K.aadr.PUB.anno          # rich per-sample annotation
  v66.p1__files.md5sum                # checksum manifest
```

The `.geno`/`.snp`/`.ind` share the prefix `v66.p1_1240K.aadr.patch.PUB` — that prefix is exactly what you hand to `extract-f2 --prefix`. (The `.anno` file is human-readable metadata; steppe reads pop labels from `.ind`, not `.anno`.)

```
bash docs/download-aadr.sh 1240K ./aadr_1240K

steppe extract-f2 --prefix ./aadr_1240K/v66.p1_1240K.aadr.patch.PUB \
  --auto-top-k 200 --maxmiss 0.5 --device 0 --out-dir f2_dir
```

If a download is interrupted, just re-run the same command — `curl -C -` resumes each partial file, and the MD5 pass at the end tells you if anything is still broken.

---

## 2. Genotype formats steppe reads

steppe reads the EIGENSOFT/PLINK format family that ADMIXTOOLS 2 uses. You point it at a dataset one of two ways:

- **`--prefix PREFIX`** — steppe looks for `PREFIX.geno` + `PREFIX.snp` + `PREFIX.ind`. If there is no `.geno` but a `PREFIX.bed` exists, it switches to the PLINK triple `PREFIX.bed` + `PREFIX.bim` + `PREFIX.fam` instead.
- **`--geno` / `--snp` / `--ind`** — name the three files explicitly (each overrides what `--prefix` would have picked). Use this when your files do not share a clean common prefix.

Within a `.geno` file, the exact on-disk layout is **auto-detected from the file's magic bytes** — you do not tell steppe which format it is. The formats it recognizes:

| format | layout on disk | detected by |
|---|---|---|
| **TGENO** | packed, individual-major (one record per sample). The real AADR v66 layout. | `TGENO` magic |
| **PACKEDANCESTRYMAP / GENO** | packed, SNP-major (one record per SNP). | `GENO` magic |
| **EIGENSTRAT** | ASCII `.geno` of `0`/`1`/`2`/`9` genotype lines, SNP-major. | text probe (`9` = missing) |
| **ANCESTRYMAP** | legacy unpacked text triple, one line per (SNP, individual) pair. | text probe (`-1` = missing) |
| **PLINK** | binary `.bed` (SNP-major) + `.bim` + `.fam`. | `.bed` magic `0x6c 0x1b 0x01` |

All of these decode to the same canonical genotype matrix, so downstream results are identical regardless of which format you started from. A few things happen automatically so the numbers match across formats: PLINK's `.bim`/`.fam` columns are re-mapped into the same SNP/individual tables the EIGENSTRAT readers produce (PLINK stores the population label in the `.fam` phenotype column), and PLINK's reversed bit order and different 2-bit code values are normalized during decode.

One sharp edge worth knowing: a plain packed **`GENO`** (SNP-major PACKEDANCESTRYMAP) file is *recognized but reported rather than decoded* on the packed fast path, which targets TGENO — steppe would rather refuse than silently read a file along the wrong axis. EIGENSTRAT, ANCESTRYMAP, and PLINK `.bed` all decode normally via their own readers. See the reference docs below for the exact behavior.

Once you have a dataset in any of these formats, the next step is the same for all of them: build an f2 cache with [`extract-f2`](./extract-f2.md).

---

## 3. `steppe-rds` — the f2 ↔ ADMIXTOOLS 2 `.rds` bridge

`steppe-rds` moves an f2 cache between steppe's on-disk format (an `STPF2BK1` directory: `f2.bin` + `pops.txt` + `meta.json`) and ADMIXTOOLS 2's `read_f2()` `.rds` directory. It is a pure on-disk format translation — **no GPU** — so you can build f2 on the GPU with `extract-f2`, then open the exact same blocks in R to double-check a fit. steppe's f2 values already equal ADMIXTOOLS 2's on the same blocks; this only changes the file layout.

```
steppe-rds export <f2_dir> <out_rds_dir>    # steppe STPF2BK1  -> AT2 read_f2() .rds dir
steppe-rds import <rds_dir> <out_f2_dir>    # AT2 read_f2() .rds dir -> steppe STPF2BK1
```

| flag / argument | what it does | default |
|---|---|---|
| `export` \| `import` | Direction of the conversion (one is required). | — |
| `<f2_dir>` (export) | The steppe f2 directory to read (holds `f2.bin` + `pops.txt`). | required |
| `<out_dir>` (export) | Output directory for the ADMIXTOOLS 2 `read_f2()` `.rds` cache. | required |
| `--counts {ones,vpair}` | What to write in the `.rds` `counts` column: `ones` reproduces ADMIXTOOLS 2's own f2 family; `vpair` writes steppe's per-block valid-pair counts. | `ones` |
| `<rds_dir>` (import) | The ADMIXTOOLS 2 `read_f2()` `.rds` directory (per-pop subdirs). | required |
| `<out_dir>` (import) | Output directory for the steppe `STPF2BK1` f2 cache. | required |

**Export needs no extra dependency** — it writes `.rds` with a stdlib serializer. **Import reads `.rds` via `pyreadr`**, which is an optional extra: `pip install steppe[rds]`.

### Examples

```
# EXPORT: a steppe f2 dir -> an AT2 read_f2() .rds dir you can open in R
steppe-rds export ./f2_dir ./exported_rds

# IMPORT: an AT2 read_f2() .rds dir -> a steppe f2 cache (needs pyreadr)
steppe-rds import ./from_at2_rds ./imported_f2_dir
```

Then in R (ADMIXTOOLS 2), `read_f2("./exported_rds")` loads it as a `[P, P, n_block]` array — off-diagonal f2 is bit-identical to steppe, and `f4`/`qpadm` in R match steppe's native fit.

If the wheel is not pip-installed (e.g. on a dev box where the package is only on `PYTHONPATH`), invoke the same entry point as a module:

```
python3 -m steppe._rds export ./f2_dir ./exported_rds
```

You can also drive it from Python instead of the CLI:

```python
import steppe
f2 = steppe.read_f2("./f2_dir")            # or extract_f2(...)
steppe.export_f2_rds(f2, "./exported_rds")     # per-pop subdirs + block_lengths_f2.rds
steppe.import_f2_rds("./from_at2_rds", "./imported_f2_dir")   # needs pyreadr
```

---

## Gotchas

- **`extract-f2` first.** Every fit and f-statistic command reads an f2 cache, not raw genotypes. Download data, run [`extract-f2`](./extract-f2.md), *then* run the analysis. `steppe-rds` also operates on f2 directories, not genotypes.
- **`--prefix` picks the format for you.** With `--prefix`, a present `.geno` always wins; PLINK (`.bed`/`.bim`/`.fam`) is only chosen when there is no `.geno`. If you have both kinds sitting under the same prefix, name the files explicitly with `--geno`/`--snp`/`--ind`.
- **A packed `GENO` (SNP-major PACKEDANCESTRYMAP) file is reported, not decoded** on the packed path — it is detected so steppe can refuse rather than mis-decode. EIGENSTRAT, ANCESTRYMAP, and PLINK `.bed` decode normally.
- **AADR TGENO and ADMIXTOOLS 2's own reader don't mix.** ADMIXTOOLS 2 v2.0.10 cannot correctly read AADR v66 TGENO — it silently misreads it. Let steppe read the TGENO directly; do not build f2 goldens by pointing old ADMIXTOOLS 2 at a TGENO file.
- **`steppe-rds import` needs `pyreadr`** (`pip install steppe[rds]`); export does not. Import is also `vpair`-lossy: ADMIXTOOLS 2's `counts` column is `1.0`, not real counts, so `vpair` is filled with a nonzero sentinel. This is harmless for f4/qpAdm/qpGraph, which don't use it.
- **`.rds` files are keyed under the alphabetically-smaller pop** (`<p1>/<p2>_f2.rds`), and diagonal self-pairs are `0.0`. If you inspect the exported directory by hand, that is why each pair appears once, under one pop.
- **Re-run the downloader to repair, don't restart from scratch.** An interrupted download resumes with `curl -C -`; a re-run also re-verifies MD5s.

## See also

- [extract-f2](./extract-f2.md) — build the f2 cache from a genotype triple (the next step after downloading data).
- [qpadm](./qpadm.md), [qpwave](./qpwave.md), [qpgraph](./qpgraph.md), [dates](./dates.md) — the fit commands that read the f2 cache.
- `docs/reference/src_io_geno_reader.hpp.md` — the reader that opens `.geno`/`.bed` files and auto-detects the format.
- `docs/reference/src_io_eigenstrat_format.hpp.md` — the on-disk format literals (TGENO vs GENO layouts, the 2-bit code mapping, header sizes).
- `docs/reference/src_io_plink_reader.hpp.md` — the `.bim`/`.fam` parsers and how PLINK columns map onto the shared tables.
- `bindings/steppe/_rds.py` — the GPU-free `.rds` (de)serializer behind `steppe-rds` and `export_f2_rds`/`import_f2_rds`.
