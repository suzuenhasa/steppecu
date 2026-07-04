# `f2_dir_writer.hpp` reference

## 1. Purpose

`src/app/f2_dir_writer.hpp` declares the code that **writes** an f2-blocks
directory to disk. That directory is the compute-once, fit-many artifact: a heavy
precompute pass measures the pairwise `f2` statistics for a set of populations and
saves the result here, and many later model fits read it back without recomputing.

This header is the write side of that exchange. A separate reader (in
`f2_dir_io.hpp`) loads a directory back into memory. The two are designed as exact
inverses: the byte layout written here is precisely what the reader expects, so a
directory round-trips by construction. Earlier work shipped only the reader; this
header is the writer that finally lets steppe produce the directories the reader
consumes.

The on-disk format is shaped to match the reference implementation's f2-blocks
directories[^at2], so the two tools speak the same interchange format.

The header is plain C++20 with no CUDA code. It reaches only into a small,
CUDA-free description of the binary file's header layout (the shared home for the
file's magic-number and version stamps), so the writer and reader can never drift
apart on those stamps. Everything else it uses is the standard library plus
steppe's own `Status` type and the in-memory f2 tensor type.

---

## 2. The directory layout

A written directory contains exactly three files.

| File | What it holds |
|---|---|
| `f2.bin` | The numeric payload. A 64-byte header, followed by the f2 values, followed by the per-block pairwise-valid SNP counts (the "vpair" region), followed by the per-block SNP sizes. The values are laid out block-major on the outside and column-major within each block — byte-for-byte identical to the in-memory f2 tensor's layout, which is why the reader round-trips it without any conversion. |
| `pops.txt` | The population labels, one per line, in the same order as the f2 tensor's population axis. This is the name-to-index map: line 1 is population index 0, and so on. |
| `meta.json` | Provenance. Records how the payload was produced — steppe version, the precision mode actually used, the jackknife block size, the number of blocks, the population count, which filters were applied, hashes of the source files, and identifying hashes of `pops.txt` and `f2.bin` themselves. |

The `f2.bin` header carries a magic string and a binary-format version so a reader
can reject a file it does not understand. `meta.json` carries its own, separate
version number for the provenance schema (see the next section).

---

## 3. The two independent version numbers

There are two version numbers in play, and they intentionally do **not** move
together.

- **`kF2MetaSchemaVersion`** (value `1`, defined in this header) versions the
  *shape of the `meta.json` provenance* — the set of fields the writer emits and a
  reader expects. It is the value stamped into `meta.json` as
  `"meta_schema_version"`. Bump it when the provenance field set changes.
- **`kF2DiskVersion`** (defined elsewhere, alongside the binary header layout)
  versions the *bytes of `f2.bin`*. Bump it when the on-disk numeric layout changes.

These evolve independently and must not be conflated. A change to the provenance
fields bumps the first; a change to the binary payload bumps the second. The reader
gates strictly on the binary version (`f2.bin`) — that is the load-bearing check.
`meta.json` and its schema version are advisory provenance: useful for
reproducibility and debugging, but not something the reader enforces correctness
against.

---

## 4. The real-vpair rule

The "vpair" region of `f2.bin` records, for each jackknife block, the real count of
SNP pairs that were valid in the precompute — the actual `F2BlockTensor.vpair`
values, **not** zeros.

This matters because of how missing blocks are detected downstream. The fit reads
the per-block *sizes* to do its work, so at first glance the vpair counts look
unused. But the missing-block / not-available path detects a dropped pair-block by
checking whether `vpair == 0`. Real vpair counts are therefore what drive the
correct dropping of a block when a maximum-missingness filter is in effect. Writing
zeros there — which an early test helper did, precisely because the fit itself only
needed the sizes — would defeat that detection. The writer must persist the true
counts.

---

## 5. `F2DirMeta` — the provenance record

`F2DirMeta` is the set of values serialized into `meta.json`. Every field is a
plain string or number resolved by the application before the write, so the writer
is a pure serializer with no compute dependency of its own.

### Identity and shape

| Field | Type | Default | Meaning |
|---|---|---|---|
| `steppe_version` | `string` | — | The steppe project version that produced the directory. |
| `precision_tag` | `string` | — | The precision mode that was *actually engaged* (emulated / native double / TF32), taken from the run's resources — not merely the mode requested. |
| `precision_mantissa_bits` | `int` | `0` | The engaged mantissa-bit count. Meaningful for the emulated-double mode. |
| `blgsize_cm` | `double` | `0.0` | The jackknife block size, in centimorgans. |
| `n_block` | `int` | `0` | The number of jackknife blocks in `f2.bin`. |
| `P` | `int` | `0` | The population count — the size of the f2 tensor's population axis. |
| `n_snp_total` | `long` | `0` | SNPs read from the `.snp` file, before any filtering. |
| `n_snp_kept` | `long` | `0` | SNPs kept after filtering — the SNP set the precompute actually used. |

### Filter flags

These echo the resolved filter settings so a run can be reproduced exactly.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `maf_min` | `double` | `0.0` | Minimum minor-allele frequency applied. |
| `geno_max_missing` | `double` | `1.0` | Maximum per-SNP missing fraction (the `maxmiss` filter[^at2]). |
| `mind_max_missing` | `double` | `1.0` | Maximum per-sample missing fraction. |
| `autosomes_only` | `bool` | `false` | Whether only autosomes were kept. |
| `drop_monomorphic` | `bool` | `false` | Whether no-variation SNPs were dropped. |
| `transversions_only` | `bool` | `false` | Whether only transversion SNPs were kept. |

### Source dataset provenance

| Field | Type | Default | Meaning |
|---|---|---|---|
| `geno_sha256` | `string` | empty | SHA-256 of the genotype file (empty if not computed). |
| `snp_sha256` | `string` | empty | SHA-256 of the `.snp` file (empty if not computed). |
| `ind_sha256` | `string` | empty | SHA-256 of the `.ind` file (empty if not computed). |
| `geno_path`, `snp_path`, `ind_path` | `string` | — | The paths of the three source files. |
| `hash_source_files` | `bool` | `false` | Whether to hash the source files (see below). |
| `pop_selection` | `string` | — | A human-readable echo of how populations were chosen, e.g. `"explicit:England_BellBeaker,..."`, `"auto-top:9"`, or `"min-n:30"`. The resolved labels themselves live in `pops.txt`; this records the *request*. |

#### The source-hash opt-in

Hashing the source files is opt-in and off by default, because it is expensive:
hashing the whole genotype file dominates the wall time (roughly 37 seconds of a
41-second run on a 6.7 GB genotype file). That hash is a provenance value, not a
correctness requirement, so it is skipped unless asked for.

- When `hash_source_files` is **false**, the writer does not hash any source file
  whose SHA is still empty. `meta.json` then records the empty SHAs together with
  `"source_hash_computed": false`, so a consumer can tell the absence is
  *deliberate* rather than a forgotten or failed hash.
- When `hash_source_files` is **true**, the writer fills any empty source SHA by
  hashing the file at its path.

Either way, a caller may pre-fill `geno_sha256` from a background thread — computing
the big hash in parallel with the GPU pipeline — and the writer will then skip
re-hashing it. This is what `sha256_file` (section 8) exists to support.

---

## 6. `F2DirWriteResult` — the outcome of a write

`F2DirWriteResult` reports whether the write succeeded and, on success, the
identifying hash of the payload.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ok` | `bool` | `false` | True on success. |
| `status` | `Status` | `Ok` | The status code. |
| `error` | `string` | empty | Empty on success; a human-readable reason on failure. |
| `f2_cache_id` | `string` | empty | The identifier of the payload, `"sha256:<hex>"` of `f2.bin`. Valid only when `ok` is true. |

The library never prints; it returns the failure reason as a string and lets the
application decide how to surface it. A write or I/O failure is reported through the
`InvalidConfig` status even though it is I/O in nature: an unwritable output
directory is treated as a configuration-level fault the user must fix, which mirrors
how the reader classifies the same kind of problem.

---

## 7. `write_f2_dir` — the write entry point

```cpp
[[nodiscard]] F2DirWriteResult write_f2_dir(
    const std::filesystem::path& dir,
    const F2BlockTensor& f2,
    const std::vector<std::string>& pop_labels,
    const F2DirMeta& meta);
```

Writes the three files described in section 2 into `dir`, creating the directory if
it does not already exist. `f2.bin` is written with real vpair counts (section 4)
in the exact byte layout the reader expects. `pops.txt` gets the labels in
population-axis index order. `meta.json` gets the provenance.

The number of blocks and the population count stamped into the output are taken from
the `f2` tensor's own shape, not from the `meta` struct — the writer trusts the
tensor, not the caller-supplied metadata, for those two values.

The call fails (returning `ok = false` with a reason) on a shape mismatch (the
number of labels does not equal the tensor's population count), an unwritable path,
or a degenerate tensor. The return value is marked must-use so a failed write cannot
be silently ignored.

---

## 8. `sha256_file` — whole-file hashing

```cpp
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);
```

Returns the SHA-256 of an entire file as a 64-character lowercase hex digest, or an
empty string if the file cannot be opened. The digest is byte-for-byte compatible
with the standard `sha256sum` tool (it follows the FIPS 180-4 standard).

This is the *same* hashing routine the writer uses internally for the payload's
`f2_cache_id` and for the source-dataset hashes — a single hashing implementation
shared across the file. It is exposed here so the extract-f2 command can hash the
large source genotype file on a **background thread**, overlapping the tens-of-
seconds whole-file hash with the GPU decode-and-f2 pipeline, then pre-fill
`F2DirMeta.geno_sha256` before calling `write_f2_dir` (which then skips re-hashing
it — see the opt-in in section 5).

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
