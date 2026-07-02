# `src/io/CMakeLists.txt` reference

## 1. Purpose

This file builds `steppe_io`, the static library that reads genotype data files
off disk and turns them into plain in-memory data — genotype tiles plus per-SNP
and per-sample metadata. It is the front end of the pipeline: everything the rest
of steppe knows about the raw input files comes through here.

The library is built as an **isolated leaf**. Two rules define that isolation, and
both are enforced by how this build file is written:

1. It produces only plain data structures (a genotype tile plus per-SNP and
   per-sample metadata). It never runs any computation on that data.
2. It depends on nothing in the compute layers. There is no CUDA in it, no GPU
   backend, and no reference back up into the core math library. The dependency
   arrows point only downward, into the small CUDA-free shared surface.

Because `steppe_io` computes nothing itself, another layer has to connect its
output to the GPU compute code. That wiring lives in the `app` layer, and the
`app` layer is the only place allowed to do it. The reader library and the
compute library never reference each other directly.

---

## 2. What each source file compiles

`steppe_io` is built from nine `.cpp` files. Each one owns a single, narrow slice
of the read path.

| Source file | What it does |
|---|---|
| `eigenstrat_format.cpp` | Parses the header of a packed `.geno` file and derives the record stride (how many bytes each SNP row occupies on disk). |
| `ind_reader.cpp` | Reads the `.ind` file into population labels and the sample selection. It follows the same rules as the ADMIXTOOLS 2 reader it is validated against, so the two agree on exactly which samples are selected. |
| `snp_reader.cpp` | Reads the `.snp` file: for each SNP it pulls the ID, chromosome, genetic position (in Morgans), reference allele, and alternate allele. |
| `plink_reader.cpp` | Reads the PLINK format pair: the `.bim` file into a SNP table and the `.fam` file into the individual partition (which samples belong to which population). |
| `genotype_source.cpp` | The front door. Given a `--prefix`, it figures out which on-disk format the triple of files is in, resolves the three file paths, and hands back the SNP table and individual partition. |
| `geno_reader.cpp` | Reads tiles of raw genotype bytes from a `.geno` or PLINK `.bed` file. It gathers the raw bytes only — it does **not** decode them into genotype values. Decoding happens later, on the GPU. |
| `ploidy_detect.cpp` | Auto-detects, per sample, whether the data is pseudo-haploid, reproducing the same adjustment ADMIXTOOLS 2 applies. |
| `filter/include_exclude.cpp` | Resolves an explicit include/exclude SNP set and an external `prune.in` list. These lists are **read** from disk; steppe never computes them (for example, it never computes linkage-disequilibrium pruning itself). |
| `filter/snp_filter.cpp` | Builds a per-SNP keep-or-drop mask by applying the quality filters: pooled minor-allele frequency, per-SNP missingness, SNP class (transition vs. transversion, palindrome handling), population membership, and the various on/off flags. |
| `filter/mind_prepass.cpp` | The optional per-sample missingness pre-pass for the `--mind` filter. It runs only when that filter is actually active, because deciding which samples exceed the missingness threshold requires a streaming pass over every SNP first. |

### Header-only filter pieces

Two parts of the filtering logic have no `.cpp` file and so do not appear in the
source list: the per-item filter decision and the filter plan. These are
header-only — small predicates and plain structs — so they compile into whatever
translation unit includes them rather than into their own object file. They are
still part of this same `io` library; they are simply implemented inline in
headers.

---

## 3. The isolation rule (what it links, and what it deliberately does not)

`steppe_io` links exactly two things, and the choice is deliberate:

- **`steppe::api`** (public, propagated to anything that uses `steppe_io`) — the
  CUDA-free public surface: the configuration types and named constants. This is
  the only shared code the reader is allowed to see.
- **`steppe::warnings`** (private, not propagated) — the shared compiler-warning
  settings.

Just as important is what it does **not** link. It does not link the core math
library, the core-internal code, or the GPU device library. That exclusion is the
whole mechanism behind the isolation described in section 1. Because those
libraries are not on the link line, a source file in `io` physically cannot
reach a core domain rule or pull in a CUDA header — the code would fail to
compile. The leaf stays a leaf by construction, not by convention or good
intentions.

There is one shared domain rule the reader path will eventually need: the rule
that partitions SNPs into genome blocks. The design keeps that rule in one place
(a host-only piece of the core library) and has the `app` layer read it, rather
than letting `io` re-derive its own copy. That way the block-partitioning rule
lives in exactly one place, and `io` never grows a dependency on core just to
reach it.

---

## 4. Include-path convention and language standard

**Include root.** Headers that belong to this library are included with an `io/`
prefix (for example, `#include "io/snp_reader.hpp"`). To make that work, the
include root is set to `src/` — the directory one level above this one — rather
than to `src/io/` itself. This is the same convention the other libraries use:
device headers are included as `device/...` and core headers as `core/...`. Every
library rooting its includes at `src/` keeps the `#include` lines uniform across
the whole codebase.

**Language standard.** The library is compiled as C++20, and that requirement is
public, so any target that links `steppe_io` also compiles as at least C++20.
