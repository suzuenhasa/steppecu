# `f2_dir_io.hpp` reference

## 1. Purpose

`src/app/f2_dir_io.hpp` declares the reader for an **f2_blocks directory** — the
on-disk artifact that lets steppe compute the expensive f2 statistics once and then
run many model fits against them without recomputing. The header exposes one entry
point, `read_f2_dir`, plus the two small structs it returns.

The design splits steppe into two phases with a file-on-disk seam between them: a
first "precompute" phase writes an f2_blocks directory, and a later "fit" phase reads
it back. This header is the **read** side of that seam. The matching writer (which
also owns writing the optional provenance file) is a separate, later piece of work;
what ships here is the reader alone, and the tests write directories for it to read.

The header is deliberately plain C++20 with **no CUDA**. It only reaches into a
CUDA-free header for the on-disk binary layout, and otherwise uses the standard
library. That keeps it in the application layer, which is not allowed to depend on
GPU code — the loaded data lands in a host-side tensor, and any move to the GPU
happens later, elsewhere.

---

## 2. The f2_blocks directory layout

An f2_blocks directory is a folder holding up to three files. Only the first two are
required to read it.

### `f2.bin` — the numeric payload

A single binary file in steppe's own on-disk f2 format (magic tag `STPF2BK1`). Its
byte layout is, in order:

1. **A 64-byte header** describing the payload (its full definition lives in the
   CUDA-free on-disk-format header this file includes).
2. **The f2 values** — `P × P × nb` doubles, where `P` is the number of populations
   and `nb` is the number of genome blocks.
3. **The paired-variance values** — another `P × P × nb` block, the per-entry
   variance that travels alongside each f2 value.
4. **The block sizes** — `nb` 32-bit integers, one SNP count per block.

The ordering inside the two large arrays is **block-major on the outside,
column-major within each block**: an entry for population pair `(i, j)` in block `b`
sits at index `i + P*j + P*P*b`. This is chosen to be *byte-identical* to the layout
of the in-memory host tensor (`F2BlockTensor`), whose own indexing helpers use the
exact same formula. Because the two layouts match exactly, reading the whole file is
equivalent to a straight memory copy into the host tensor — no per-element
reshuffling is needed.

### `pops.txt` — the population labels

A plain-text file listing the `P` population labels, **one per line, in P-axis index
order**. The numeric engine works purely with population *indices* and carries no
names — names are an application-layer concern — so this file is the name-to-index
map the engine itself lacks. Line `k` (starting from 0) is the name of the population
at index `k`.

### `meta.json` — provenance (optional)

A JSON file recording how the directory was produced. It is **optional for reading**:
the fit engine needs only `f2.bin` and `pops.txt`. The reader in this header does not
parse it, and a missing `meta.json` is not an error. Writing it is the writer's job.

---

## 3. `F2Dir` — the loaded directory

`F2Dir` is the successfully-loaded result: the host-side numeric tensor plus its
population names.

| Field | Type | Meaning |
|---|---|---|
| `f2` | `steppe::F2BlockTensor` | The host tensor read from `f2.bin` — the f2 values, paired variances, and block sizes, all in the canonical `i + P*j + P*P*b` layout. |
| `pop_labels` | `std::vector<std::string>` | The `P` population labels from `pops.txt`, in P-axis index order. Index `k` in this vector names population `k` on the tensor's P axis. |

Together these two fields reunite the index-only numeric data with the names it needs
to be interpreted.

---

## 4. `F2DirResult` — the return value

`read_f2_dir` never throws for a bad directory; instead it returns an `F2DirResult`
that reports either success or a described failure. This lets the application print a
human-readable reason itself — the library never prints.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ok` | `bool` | `false` | `true` on success, `false` on any failure. This is the flag to check first. |
| `status` | `Status` | `Status::Ok` | `Ok` on success; a fault category on failure (see below). |
| `error` | `std::string` | empty | Empty on success; on failure, the human-readable reason. The application prints this to standard error. |
| `dir` | `F2Dir` | — | The loaded directory. **Valid only when `ok` is true.** |

### The fault category

A malformed directory is reported as an **`InvalidConfig`** fault, not a raw I/O
fault. The reasoning: a bad or unreadable cache directory is treated as a
configuration-level problem the user must fix, rather than a transient I/O hiccup.
When the application maps this back to a process exit code, it becomes the I/O-error
exit code by the application's fallthrough rule, so a bad directory still surfaces as
a clear failure to the caller.

---

## 5. `read_f2_dir` — the reader entry point

```cpp
[[nodiscard]] F2DirResult read_f2_dir(const std::filesystem::path& dir);
```

Given a directory path, this:

1. Reads `<dir>/f2.bin` and parses the `STPF2BK1` payload into a host
   `F2BlockTensor`.
2. Reads `<dir>/pops.txt` into the list of `P` population labels, in index order.
3. **Validates that the number of lines in `pops.txt` equals the tensor's `P`** — the
   name-to-index map must cover the whole P axis exactly, no more and no fewer.

It does **not** read or require `meta.json`.

### What counts as a failure

Any of the following returns `ok = false` with a filled-in `error` reason:

- `f2.bin` is missing or unreadable.
- `pops.txt` is missing or unreadable.
- `f2.bin` has a bad magic tag or an unrecognized version.
- `f2.bin`'s payload is truncated (shorter than its header says it should be).
- `pops.txt`'s line count does not equal the tensor's `P`.

The `[[nodiscard]]` marker means the returned result must be inspected — a caller
cannot silently ignore whether the read succeeded.
