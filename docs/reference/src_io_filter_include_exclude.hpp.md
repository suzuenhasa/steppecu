# `include_exclude.hpp` reference

## 1. Purpose

`src/io/filter/include_exclude.hpp` turns a user's SNP include/exclude choices
into a single yes/no test that the SNP reader can ask once per SNP. A caller may
supply three things: an explicit list of SNP IDs to keep, an explicit list of SNP
IDs to drop, and the path to an external `prune.in` file (a linkage-disequilibrium
pruned list of SNPs to keep). This header resolves all three into one object that
answers a single question — "does this SNP ID pass?" — so that the reading code
never has to re-derive the rule for combining those three inputs.

Two important boundaries:

- **The `prune.in` file is read, never computed.** steppe does not compute
  linkage disequilibrium itself. It only accepts a pruned list that some external
  tool produced, and treats it as an additional set of IDs to keep.
- **This is a host-only, dependency-light header.** It is plain C++20 with no GPU
  code and no dependency on the numeric core. Its only project dependency is the
  configuration struct that carries the include/exclude lists and the `prune.in`
  path. Any problem reading a file surfaces to the caller as a standard
  `std::runtime_error`.

Building the membership in one place means the reader (`snp_filter`) does not
re-implement the "how do include, exclude, and prune.in combine?" logic on its
own — there is exactly one definition of that rule, described next.

---

## 2. The membership rule

Membership is built from two resolved sets:

- **keep-set** = the union of the include IDs and the `prune.in` IDs. If *either*
  of those inputs is non-empty, a SNP must be in this union to pass. If *both* are
  empty, there is no keep constraint at all.
- **drop-set** = the exclude IDs. Any ID in this set fails, and this overrides the
  keep-set — exclude always wins. This matches the usual `--exclude` / `.missnp`
  convention where an exclusion is final.

Putting those together, a SNP with ID `id` passes if and only if:

> (keep-set is empty **OR** `id` is in the keep-set) **AND** `id` is **not** in the
> drop-set.

Read that in two halves. The first half is the keep test: an empty keep-set means
"no include constraint, everything is eligible," while a non-empty keep-set means
"only these IDs are eligible." The second half is the drop test: it can only ever
remove a SNP, and it applies regardless of whether the SNP was in the keep-set.

The important consequence of "exclude wins" is that listing the same ID in both
the include list and the exclude list drops it. The drop test is applied after the
keep test and cannot be overridden.

---

## 3. SnpMembership

`SnpMembership` is the resolved object. You build it once from the configuration,
then query it per SNP ID. Internally it holds the keep-set and drop-set as hash
sets, so each per-SNP query is constant time — this matters because a `.snp` file
can hold on the order of 584,000 IDs, and the reader asks the question once for
every one of them.

### Construction

```cpp
explicit SnpMembership(const FilterConfig& cfg);
```

The constructor reads the include IDs, exclude IDs, and (if a path is set) the
`prune.in` file out of the configuration and resolves them into the two sets:

- The include IDs and the `prune.in` IDs are unioned into the keep-set.
- The exclude IDs become the drop-set.
- The `prune.in` file, if its path is non-empty, is opened and read (one SNP ID
  per whitespace-delimited line, blank lines skipped) and its IDs are added to the
  keep-set.

**Fail-fast on a bad `prune.in`.** If a `prune.in` path is set but the file cannot
be opened, *or* it opens but cannot be read, the constructor throws
`std::runtime_error`. The "opens but cannot be read" case is deliberate: on this
platform a directory (or a FIFO or similar non-regular file) opens successfully
but fails on the first read. Rather than silently ending up with an empty keep-set
— which would quietly change which SNPs pass — the constructor fails loudly. See
[section 4](#4-read_snp_id_list) for the shared read routine that enforces this.

**The no-op default.** When the include list, the exclude list, and the
`prune.in` path are all empty/unset, the resulting membership imposes no
constraint at all: every SNP passes. This is the ordinary default when the user
has not asked for any SNP-ID filtering.

### Querying

```cpp
[[nodiscard]] bool passes(const std::string& snp_id) const noexcept;
```

Returns whether `snp_id` passes the rule from [section 2](#2-the-membership-rule):
`(keep-set empty OR id in keep-set) AND id not in drop-set`. When both sets are
empty this always returns `true`. It never throws.

### Fast-path and diagnostics helpers

| Member | Returns | What it's for |
|---|---|---|
| `is_noop()` | `bool` | True exactly when the membership imposes no constraint — both the keep-set and the drop-set are empty. This lets the reader skip the per-SNP `passes` query entirely on the ordinary no-filter path, so the common case pays nothing. |
| `keep_count()` | `std::size_t` | The number of IDs in the resolved keep-set (include ∪ `prune.in`). For diagnostics and tests. |
| `drop_count()` | `std::size_t` | The number of IDs in the resolved drop-set (exclude). For diagnostics and tests. |

All three are `noexcept` and do not allocate.

### Fields

| Field | Type | Meaning |
|---|---|---|
| `keep_set_` | `std::unordered_set<std::string>` | The include IDs unioned with the `prune.in` IDs. Empty means no keep constraint. |
| `drop_set_` | `std::unordered_set<std::string>` | The exclude IDs. Overrides the keep-set — an ID here always fails. |

---

## 4. read_snp_id_list

```cpp
void read_snp_id_list(const std::string& path, std::vector<std::string>& out);
```

This is the shared routine that reads a `prune.in`-style SNP-ID list file. It is
exposed separately (rather than kept private to the constructor) so that tests and
other code can reuse the exact same parsing and error behavior.

What it does:

- Reads one SNP ID per line. The **first whitespace-delimited token** of each line
  is taken as the ID, so lines with trailing columns are tolerated (only the first
  field is used).
- Skips blank lines.
- **Appends** the IDs to `out` — it does not clear `out` first, so a caller can
  accumulate IDs from more than one source into the same vector.

Error behavior (the same fail-fast contract the constructor relies on): it throws
`std::runtime_error` if the file cannot be opened, **or** if the file opens but the
first read fails. The second case covers a path that names a directory, a FIFO, or
another non-regular node — on this platform those open successfully but set the
stream's error state on the first read. Failing fast there prevents the silent
"empty list" outcome, which would otherwise look like a valid-but-empty prune list
and quietly change which SNPs are kept.
