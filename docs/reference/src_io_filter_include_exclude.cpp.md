# `include_exclude.cpp` reference

## 1. Purpose

`src/io/filter/include_exclude.cpp` turns a user's include/exclude SNP-id lists —
plus an optional external `prune.in` file — into a single per-SNP yes/no test that
the reader can call while streaming genotypes. It answers one question for each SNP:
"given what the user asked to keep and drop, is this SNP wanted?"

The file has no GPU code and no dependency on the core or device layers. It is
pure host C++ that only touches the standard library, and it turns file-read
problems into a thrown `std::runtime_error` for the caller to handle. steppe
*reads* an externally supplied, already-LD-pruned SNP list; it never computes
linkage disequilibrium itself.

There are three pieces:

1. A small helper, `read_snp_id_list`, that reads a `prune.in`-style file of SNP
   IDs into a list.
2. The `SnpMembership` constructor, which resolves the include list, exclude list,
   and `prune.in` into two hash sets — a keep-set and a drop-set.
3. `SnpMembership::passes`, which applies those two sets to one SNP ID.

The exact membership rule and the public contract of these functions live in the
header; the sections below explain the parts of the implementation that are not
obvious from the signatures — chiefly why the file reader checks three separate
stream-error flags instead of the usual one.

---

## 2. Reading a `prune.in` file (`read_snp_id_list`)

A `prune.in` file is one SNP ID per line. `read_snp_id_list` opens the file, reads
it line by line, and appends each line's first whitespace-delimited token to the
output list. Taking only the first token means a file with extra trailing columns
still parses cleanly, and blank lines are skipped automatically (extracting a token
from a blank line fails, so nothing is appended for that line).

If the file cannot be opened at all, the function throws immediately.

### The fail-fast guard on unreadable-but-openable paths

The subtle part is the check *after* the read loop:

```cpp
if (in.bad() || (in.fail() && !in.eof())) { throw ...; }
```

This exists because the obvious open-time check, `if (!in)`, is not enough on Linux
(libstdc++/POSIX). Constructing an input file stream on a **directory** — or on a
FIFO/socket that the process can `open()` but not actually read from — *succeeds*.
The stream looks fine, so the open-time `if (!in)` guard passes. The failure only
shows up on the first attempt to read, which sets the stream's `badbit`.

To catch that, the post-loop guard has to tell two very different end states apart:

- **A normal end of input.** When `std::getline` reaches the real end of a regular
  file, it extracts no characters and sets `eofbit` together with `failbit` — but
  **not** `badbit`. This is success, and must not throw.
- **A hard read failure.** A directory or other non-regular node sets `badbit` (a
  genuine stream error), or fails for some reason while not yet at end-of-file.

So the guard throws when either `badbit` is set (a hard error) **or** the stream
failed for a reason other than reaching EOF (`fail() && !eof()`). Normal EOF sets
`fail` but also sets `eof`, so it slips past the guard as intended.

Without this guard, pointing `prune_in_path` at a directory would silently produce
an **empty** keep-set. That is worse than a crash: an empty keep-set imposes no
include constraint (see section 4), so the reader would quietly ignore the user's
`prune.in` and run over every SNP — dropping the intended constraint on a run whose
whole point is to match a reference result exactly. Failing fast turns that silent
mistake into a clear error.

---

## 3. Resolving the keep-set and drop-set (`SnpMembership` constructor)

The constructor takes a `FilterConfig` and builds two hash sets of SNP IDs once, up
front, so the per-SNP test later is a constant-time lookup rather than a scan. (A
`.snp` file can hold on the order of half a million IDs, so an O(1) lookup matters.)

**Keep-set = `include_snp_ids` ∪ `prune.in` IDs.** Both sources are unioned into a
single set. The `prune.in` file, if a path is given, is read here via
`read_snp_id_list` — read, never computed — and its IDs are merged in. Folding both
into one set means the per-SNP test only has to answer a single question ("is this
SNP wanted?") instead of consulting two separate lists. If the `prune.in` path is
set but unreadable, the read throws and construction fails (see section 2).

**Drop-set = `exclude_snp_ids`.** These IDs are collected into a separate set. They
override the keep-set — an ID listed here is dropped even if it would otherwise be
kept. This matches the usual `--exclude` / `.missnp` convention.

If all three inputs (`include_snp_ids`, `exclude_snp_ids`, `prune_in_path`) are
empty, both sets end up empty and the membership becomes a complete no-op — the
default behavior on the standard path, where nothing is filtered by ID.

---

## 4. The membership test (`passes`)

`passes` applies the two resolved sets to a single SNP ID and returns whether it is
wanted. It is declared `noexcept` — it only does hash-set lookups and never throws.
Two rules, in order:

1. **Exclude wins.** If the drop-set is non-empty and contains the ID, the SNP fails
   immediately, regardless of the keep-set. (The drop-set is only consulted when it
   is non-empty, so the common no-exclude case skips the lookup entirely.)
2. **Include applies only when a keep-set exists.** If the keep-set is non-empty and
   does **not** contain the ID, the SNP fails. But if the keep-set is *empty*, it
   imposes no include constraint at all — every ID is allowed through this step.
   This is what makes "no include list and no `prune.in`" mean "keep everything"
   rather than "keep nothing."

If neither rule rejects the ID, it passes. Put together: a SNP passes when it is not
in the drop-set **and** (the keep-set is empty **or** the SNP is in the keep-set).
