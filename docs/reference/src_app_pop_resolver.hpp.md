# `pop_resolver.hpp` reference

## 1. Purpose

`src/app/pop_resolver.hpp` turns population *names* into population
*indices*. The compute engine never deals in names — every model refers to a
population by its position (its index) along the "P axis" of the underlying
statistics data. Names are strictly an app-level, human-facing concern.

This header is the single place that bridges the two worlds. It reads the list
of population labels (one per line, in the file `pops.txt` inside a dataset
directory) and builds a lookup that maps each label to its index. When a user
names a target, a left group, or a right group on the command line, that name is
resolved through this map before anything reaches the compute engine.

It is deliberately plain C++20 with no GPU code. It only uses the standard
library and one small error-status type, so it can be included freely by the
command-line tool and the app layer without pulling in any of the GPU stack.

This resolver replaces an older approach where a test carried a hardcoded
name-to-index map baked directly into its source. Centralizing that mapping here
is the main reason the app access layer exists.

---

## 2. The index-only seam and fail-fast contract

Two design decisions shape everything in this file.

**Names stop here.** The boundary between the app and the compute engine is
index-only. Once a name has been resolved to an index, the name is no longer
needed downstream; the engine works purely with integer positions. This file is
the only crossing point, so name handling never leaks into the compute code.

**An unknown name is a hard, up-front error.** If a user asks for a population
that is not in `pops.txt`, that is treated as a configuration mistake and the
run stops immediately — before any GPU work begins — reporting the exact
offending label so the user can see precisely which name was wrong. In code
terms this surfaces as a failure carrying the status value `InvalidConfig` and a
human-readable reason string. There is no silent skipping and no best-effort
partial match.

---

## 3. Result types

Resolution never throws and never returns a bare integer that a caller might
mistake for a valid index. Instead it returns a small result struct that carries
both the outcome and, on failure, the reason. There are two, one for a single
name and one for a list.

### `ResolveResult` — one name

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ok` | `bool` | `false` | True only when the name was found. Check this first. |
| `status` | `Status` | `Ok` | The error status on failure (`InvalidConfig` for an unknown name); `Ok` on success. |
| `error` | `string` | empty | Empty on success. On failure, a message naming the offending label, for example `unknown population 'X' ...`. |
| `index` | `int` | `-1` | The resolved P-axis index. Only meaningful when `ok` is true; stays `-1` on failure. |

### `ResolveListResult` — a list of names

Used to resolve a whole group of labels (a target, a left set, or a right set)
in one call.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ok` | `bool` | `false` | True only when every name in the list was found. |
| `status` | `Status` | `Ok` | The error status if any name failed; `Ok` when all succeed. |
| `error` | `string` | empty | Empty on success; on failure, the message for the first name that could not be resolved. |
| `indices` | `vector<int>` | empty | The resolved indices, in the **same order** as the input labels. Only meaningful when `ok` is true. |

---

## 4. The `PopResolver` class

`PopResolver` owns the label-to-index map. It is built once from the population
labels in index order, after which each lookup is a constant-time hash-map
probe.

### Construction

```cpp
explicit PopResolver(const std::vector<std::string>& labels_in_index_order);
```

The single argument is the list of labels exactly as they appear in `pops.txt`,
in line order. Line 0 becomes index 0, line 1 becomes index 1, and so on. The
constructor builds two things from this: a forward array (index to label, for
printing names back onto result rows) and a reverse hash map (label to index,
for resolution).

A **duplicate label** in the input makes the resolver invalid rather than
throwing. If the same name appears twice, the reverse map would be ambiguous
(which index does that name mean?), so construction records the conflict and
marks the object unusable. Callers must check validity before resolving.

### Checking validity

| Method | Returns | Meaning |
|---|---|---|
| `valid()` | `bool` | True only if the map was built with no duplicate-label conflict. Check this immediately after construction. |
| `error()` | `const string&` | The reason the resolver is invalid; empty when `valid()` is true. |

The intended pattern is to construct the resolver, check `valid()`, and stop the
run with `error()` if it is false — before attempting any resolution.

### Reading the mapping

| Method | Returns | Meaning |
|---|---|---|
| `size()` | `int` | The number of populations, which equals the P-axis length and the `pops.txt` line count. |
| `label_at(int index)` | `const string&` | The label at a given index. Used to put names back onto result rows for output. Bounds-checked: an out-of-range index throws. |

### Resolving names

| Method | Returns | Behavior |
|---|---|---|
| `resolve(const string& label)` | `ResolveResult` | Looks up one name. An unknown name yields `ok=false`, status `InvalidConfig`, and a reason naming the label plus a hint about the dataset directory. |
| `resolve_all(const vector<string>& labels)` | `ResolveListResult` | Resolves a list in order. The **first** unknown name short-circuits the whole call: it stops and returns a failure naming that one label, so the user sees the exact offending name rather than a vague "some name failed." On success `indices` matches the input order one-for-one. |

---

## 5. Invariants worth remembering

- **Index order is authoritative.** The order of labels handed to the
  constructor defines the index of every population. That order comes from
  `pops.txt` and must not be reshuffled, or every downstream index would refer
  to the wrong population.
- **A resolver may be born invalid.** Construction never throws on a duplicate
  label; it produces an invalid object instead. Always gate use on `valid()`.
- **Resolution failures are reported, not thrown.** `resolve` and `resolve_all`
  return their outcome in the result struct. The only method that can throw is
  `label_at`, and only for an out-of-range index.
- **First failure wins in a list.** `resolve_all` reports the first unresolved
  label and does not continue, matching the "name the exact offending label and
  fail fast" contract.
