# `cmd_common.hpp` reference

## 1. Purpose

`src/app/cmd_common.hpp` is a tiny header of shared helpers for the `steppe`
command-line handlers. steppe ships a bunch of subcommands — `steppe f4`,
`steppe qpdstat`, `steppe f2`, and so on — and each one is written as its own
handler function. Those handlers kept re-typing the same two chores, so this
header pulls each chore into one place. That is the whole reason the file
exists: one copy of a shared step means the copies cannot quietly drift apart
and start behaving differently from one subcommand to the next.

Nothing here does any actual population-genetics math. It is pure
command-plumbing: check that a GPU is present before we try to use one, and turn
the population names the user typed on the command line into tidy groups the math
code can consume. It is header-only, lives in the app layer, and contains no
CUDA — it only ever runs on the host while a command is starting up.

There are no `— reference §N` markers in the source, so the sections below just
follow the file top to bottom.

---

## 2. The no-GPU guard (`require_first_gpu`)

```
require_first_gpu(const device::Resources& resources, const char* prefix) -> bool
```

steppe is a GPU product — every real subcommand needs a CUDA device to do its
work. This is the one shared check that every handler runs first, before it
touches the device, so the "you have no GPU" message reads the same everywhere.

It looks at `resources.gpus` (the list of CUDA devices steppe found at startup).
If that list is non-empty, it returns `true` and the handler carries on. If it is
empty, it prints the canonical diagnostic to standard error —

```
steppe <prefix>: no CUDA device available (steppe is a GPU product; a
CUDA-capable GPU is required)
```

— where `prefix` is the subcommand's own name (`f4`, `qpdstat`, …) so the user
can see which command bailed. Then it returns `false`, which tells the caller to
stop.

The one subtle contract here is that this function **must not throw**. Each
caller runs this guard inside a `try { }` block that catches exceptions and turns
them into a specific exit code. If this function threw instead of returning
`false`, that `try` would catch the throw and remap it to the *wrong* exit code,
so the "no GPU" case would look like some other kind of failure. Returning a
plain `bool` keeps the no-GPU exit path clean and predictable. The `[[nodiscard]]`
marker is there so a caller can't accidentally ignore the answer and march on to
use a device that isn't there.

---

## 3. Building the quartet name table (`build_quartet_names`)

```
build_quartet_names(const config::RunConfig& config,
                    std::vector<std::array<std::string, 4>>& quartets,
                    std::string& err,
                    const char* tool, const char* group_noun) -> bool
```

The `f4` and `qpdstat` commands both work on **quartets** — ordered groups of
four population names, written `(p1, p2, p3, p4)`, that name the four populations
an f4-style statistic compares. Both commands let the user specify those quartets
in one of two ways, and both need the exact same parsing. So the parser lives
here once, and each command passes in just the two words that differ between them
(via `tool` and `group_noun`).

The function fills `quartets` with one `{p1, p2, p3, p4}` row per quartet and
returns `true` on success. On any problem it writes a human-readable message into
`err` and returns `false`, leaving the handler to report it.

### The two input styles

**Column style — `--pop1 … --pop4`.** The user gives four parallel lists, one per
column, and quartet *k* is formed by taking the *k*-th name from each list. The
function first checks whether any of the four columns is non-empty (`have_cols`).
If so, it takes that as the user's intent and validates the columns:

- All four columns must be the **same length** ("row-aligned"). If they aren't,
  `err` reports the four lengths it actually got, so the user can see which column
  is short.
- The length must not be zero.

When those hold, it walks *k* from 0 to *n*−1 and pushes `{p1[k], p2[k], p3[k],
p4[k]}` — reading across the columns, one quartet per row.

**Flat style — `--pops`.** If none of the four columns was given, the function
falls back to a single flat `--pops` list and reads it in groups of four: names 0
through 3 are the first quartet, names 4 through 7 the second, and so on. It
checks that:

- the list isn't empty — otherwise `err` explains that the command needs quartets
  and shows both accepted spellings; and
- the count is a **multiple of four**, since each quartet is exactly four names.
  If it isn't, `err` names the offending count.

Then it slices the list into groups of four and pushes each as a quartet.

### Why the two little `const char*` parameters

`tool` and `group_noun` are the *only* things that differ between the `f4` and
`qpdstat` versions of this parsing. `tool` is the command name that gets dropped
into the error messages (so the user sees "`f4` needs quartets…" versus
"`qpdstat` needs quartets…"), and `group_noun` is the word for a group of four in
the "must be a multiple of 4" message — `f4` calls it a **quartet** while
`qpdstat` calls it a **quadruple**. Passing those two words in, rather than
copy-pasting the whole parser into each command with slightly different wording,
is exactly the "one source of truth so the copies cannot drift" idea this header
is built around. Like the guard above, it is `[[nodiscard]]` so a caller can't
skip checking whether parsing succeeded.
