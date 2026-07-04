# `cmd_qpgraph.cpp` reference

## 1. Purpose

`src/app/cmd_qpgraph.cpp` implements two command-line subcommands:

- **`steppe qpgraph`** — fit a single, fixed admixture graph to a set of observed
  f-statistics and report the fitted edge lengths, admixture weights, and a
  goodness-of-fit score.
- **`steppe qpgraph-search`** — enumerate every rooted admixture topology over a
  bounded set of populations, fit them all at once, and report the best one.

Both commands follow the same shape: read the precomputed `f2_blocks` directory,
build the GPU resources, upload the f2 blocks so they stay resident on the GPU, run
the fit, then write the result out as CSV, TSV, or JSON.

This file is deliberately plain C++20 with no CUDA in it. The GPU is reached only
through a few CUDA-free seams (the resource builder, the device f2-block type, and
the `run_qpgraph` / `run_qpgraph_search` entry points). That keeps the command layer
compiling without the GPU toolchain and keeps all CUDA specifics behind the library
boundary. The `main()` function owns stdout and stderr; these command functions only
return a process exit code.

The value types they pass around — `QpGraphEdge`, `QpGraphOptions`,
`QpGraphResult`, `QpGraphSearchOptions`, `QpGraphSearchResult` — are defined in the
public headers, not here. This file is the glue that turns command-line input into
those types and turns the result back into text.

---

## 2. The `--graph` edge-list file format

`steppe qpgraph` reads the graph topology from the file named by `--graph`. The
parser (`read_edge_list`) accepts the same 2-column edge list the reference
writes[^at2], and is forgiving about the small formatting differences between a
hand-written file and one produced by R's `write.csv`.

The rules for each line:

- **Two tokens per line**, a parent and a child (`parent child`), naming a single
  directed edge from parent to child.
- **Separators may be whitespace or commas.** Every comma on a line is turned into a
  space before the line is split, so space-separated, tab-separated, and
  comma-separated (CSV) edge lists all parse the same way.
- **Surrounding double quotes are stripped** from each token. R's `write.csv` quotes
  the labels (`"parent","child"`), so the quotes are removed to recover the bare
  label.
- **Blank lines are skipped.** A line with no first token is ignored.
- **Comment lines are skipped.** A line whose first token begins with `#` is ignored.
- **A header row is skipped.** Only the very first data line is checked: if its two
  tokens (compared case-insensitively) are `from`/`to` or `parent`/`child`, that line
  is treated as a header and dropped. This lets a CSV that starts with a column
  header parse cleanly, without accidentally dropping a real edge later in the file
  that happens to use those words.

Two conditions make the parse fail (the function returns false and fills in an error
message):

- A non-blank, non-comment line that has only one token — the message reports the
  line number and that two columns are required.
- A file that yields no edges at all — reported as "no edges."

On success the edges are appended, in file order, to the caller's vector.

---

## 3. Command flow

### `run_qpgraph_command` (single graph)

The steps, in order:

1. **Validate required inputs.** `--f2-dir` and `--graph` must both be set;
   otherwise the command fails with an invalid-config exit code and a message.
2. **Read the f2 directory.** If it cannot be read, the command fails with an I/O
   exit code. The directory supplies both the f2 blocks and the population-label
   order (the list of populations, in the order they appear in the directory's
   `pops.txt`). That label order is the axis the graph's leaf names are matched
   against.
3. **Parse the edge list** (see section 2). A parse failure is an invalid-config
   exit.
4. **Assemble the fit options** (see section 4).
5. **Dispatch to the GPU** through the shared helper (see section 5), which builds
   resources, uploads the f2 blocks, and runs `run_qpgraph`.
6. **Surface a structural graph failure.** If the fit comes back with an
   invalid-config status, that means the graph itself is not usable — for example a
   leaf name that is not one of the f2 populations, or a graph that is not a valid
   rooted acyclic topology. This is reported as a clear error with a nonzero exit,
   not written out as a silent result row.
7. **Emit the result** as CSV, TSV, or JSON (see section 6), then return the exit
   code that corresponds to the fit's status.

### `run_qpgraph_search_command` (topology search)

Same overall shape, with three differences:

- It requires **`--f2-dir`** and **`--pops`** rather than a `--graph` file. There is
  no topology to read, because the search generates the topologies itself.
- `--pops` must list **at least 3 population labels**. This is the bounded leaf set
  the search enumerates topologies over; fewer than three leaves has no meaningful
  set of topologies to search.
- It runs `run_qpgraph_search` instead of `run_qpgraph`, and emits the
  search-specific result (see section 7).

The invalid-config status here means the leaf set is unusable — a named population
that is not in the f2 set, or fewer than three leaves.

---

## 4. Fit options and their defaults

The fit options are taken from the parsed command-line configuration and handed to
the fit engine. The defaults are chosen to reproduce the `qpgraph()`
results[^at2], so a run with no extra flags matches the reference.

For **`steppe qpgraph`** (`QpGraphOptions`):

| Option | Source flag | Meaning |
|---|---|---|
| `fudge` | `--fudge` | A small diagonal term added to stabilize the fit. This value is shared with the qpAdm options (it is the same `fudge` field), so the two commands stabilize the same way. |
| `diag_f3` | `--diag-f3` (via `qpgraph_diag_f3`) | A small value added to the diagonal of the f3 statistics used in the fit. |
| `numstart` | `--numstart` (via `qpgraph_numstart`) | How many random restarts the optimizer runs; the best-scoring restart wins. More restarts reduce the chance of settling in a poor local optimum. |
| `constrained` | `--constrained` (via `qpgraph_constrained`) | Whether admixture weights are constrained (for example, kept in a valid range and summing correctly at a node). |

For **`steppe qpgraph-search`** (`QpGraphSearchOptions`):

- `pops` — the bounded leaf set from `--pops`.
- `max_nadmix` — from `--max-nadmix`; the largest number of admixture events any
  enumerated topology may have. The search covers every topology from zero
  admixtures up through this many.
- `fit` — a nested copy of the same four single-graph options above (`fudge`,
  `diag_f3`, `numstart`, `constrained`), applied to each topology it fits.

The concrete default numbers live with the configuration and the option structs, not
in this file; this file only copies them across. The point that matters here is that
they are wired to line up with the `qpgraph()` golden.

---

## 5. The shared device-fit dispatch

Both commands do exactly the same GPU-side sequence — build resources, refuse to run
if there is no GPU, upload the f2 blocks resident, and run the fit inside one
error-handling block. That sequence lives once, in the `dispatch_device_fit`
template helper, so the two command bodies do not repeat it.

What it does:

1. **Build the device resources** from the run's device configuration.
2. **Guard the no-GPU case.** steppe is a GPU product: if no CUDA-capable device is
   available, the command prints a clear message and returns a runtime-error exit
   code. It does not fall back to a CPU path.
3. **Upload the f2 blocks resident on device 0.** The first GPU in the resource list
   is chosen, and the f2 blocks are copied to it and kept there for the whole fit
   rather than being streamed in and out.
4. **Run the fit.** The caller passes in a small function (`run_fit`) that receives
   the uploaded device blocks and the resources and calls the right entry point
   (`run_qpgraph` for a single graph, `run_qpgraph_search` for the search). The
   result is written into the caller's result object.

The whole sequence is wrapped in one try/catch. On success the helper returns
"nothing" (an empty optional) to signal that the caller should continue. If anything
throws — a device fault, an out-of-memory, or another runtime error — it prints a
`device error` message tagged with the command name and returns the exit code that
matches the caught exception. That exit-code mapping is what turns a genuine
device out-of-memory into the dedicated out-of-memory exit code rather than a generic
error (see section 8).

The `prefix` argument is just the command name (`"qpgraph"` or `"qpgraph-search"`)
used in the diagnostic messages so the user can tell which command failed.

---

## 6. Single-graph output format

`emit` writes a `QpGraphResult` in one of three formats. The separator is a comma
for CSV, a tab for TSV; JSON is a single object.

### Status names

The internal fit status is mapped to a short lowercase string used in both output
forms:

| Status | Emitted as |
|---|---|
| Ok | `ok` |
| Non-SPD covariance | `nonspd` |
| Rank deficient | `rankdeficient` |
| Invalid config | `invalid_graph` |
| anything else | `error` |

### CSV / TSV

Two labeled sections, each introduced by a `# section:` comment line:

- **`# section: edges`** — one header row `from,to,type,weight`, then one row per
  edge. Admixture edges come first, tagged `type = admix`, and carry the fitted
  mixture weight in the `weight` column. Drift edges follow, tagged `type = edge`,
  and carry the fitted edge length in the `weight` column. So the `weight` column
  means "mixture proportion" for admix rows and "edge length" for edge rows.
- **`# section: summary`** — one header row and one data row with the overall
  `score`, the `restart_spread` (how much the score varied across restarts), the
  worst residual z-score, the pair of populations that produced that worst residual
  (`worst_pop2`, `worst_pop3`), and the `status` string.

### JSON

A single object with these keys:

- `score`, `restart_spread`, `worst_residual_z` — the top-level fit quality numbers.
- `worst_pair` — a two-element array of the populations behind the worst residual.
- `status` — the status string from the table above.
- `admix` — an array of objects, one per admixture edge, each with `from`, `to`,
  `weight`, and a `low`/`high` confidence band. If the result carries no band for an
  entry, `low` and `high` fall back to the point weight itself.
- `edges` — an array of objects, one per drift edge, each with `from`, `to`, and
  `length`.

Both forms are produced entirely within this function; there is no shared
result-formatting primitive behind them.

---

## 7. Topology-search output format

`emit_search` writes a `QpGraphSearchResult`. It reports how large the search space
was, which topology won, how close the runner-up came, and how fast the whole sweep
ran.

Fields reported:

- `n_trees` — the number of base (zero-admixture) tree topologies.
- `n_admix1` — the number of one-admixture topologies.
- `n_candidates` — the total number of topologies fit.
- `best_score` — the score of the winning topology (lower is a better fit).
- `second_best_score` — the score of the runner-up, so the gap to the best is
  visible.
- `best_nadmix` — how many admixture events the winning topology has.
- `heuristic_recovered` — whether a faster heuristic search would have found the same
  global best. This is a check that the exhaustive search and a cheaper strategy
  agree.
- `fit_all_wall_ms` — the wall-clock time to fit every topology, in milliseconds.
- `topologies_per_s` — the resulting throughput.
- the winning topology's edges.

### CSV / TSV

- **`# section: search`** — one header row and one data row carrying every field
  above except the edges. `heuristic_recovered` is written as `1` or `0`.
- **`# section: best_edges`** — the winning topology's edge list on one line, encoded
  as `from>to` per edge joined by `;`.

### JSON

A single object with the same fields. `best_hash` (a stable identifier for the
winning topology) and `best_edges` (the same `from>to;...` encoding) are included as
strings; `heuristic_recovered` is a JSON boolean.

---

## 8. Exit codes and error surfacing

Every path returns a specific process exit code so a caller or script can tell what
happened without parsing text:

- **Missing required flags** (`--f2-dir`, `--graph`, or too few `--pops`) →
  invalid-config exit.
- **f2 directory cannot be read** → I/O-error exit.
- **Edge-list parse failure** → invalid-config exit.
- **No GPU available** → runtime-error exit, with a message that a CUDA-capable GPU
  is required.
- **A device fault during the fit** → the exit code that matches the caught
  exception. A genuine device out-of-memory maps to the dedicated out-of-memory exit
  code rather than a generic runtime error, so an out-of-memory is distinguishable
  from other failures.
- **A structural graph / leaf-set problem** (the fit returns an invalid-config
  status) → invalid-config exit, with a message explaining that a leaf is not an f2
  population or the topology is unusable. This is surfaced as an explicit fault, not
  written out as a silent result.
- **A clean fit** → the exit code derived from the fit's status. An `Ok` fit exits 0;
  a non-fatal but noteworthy status (such as a non-SPD covariance or a rank-deficient
  system) maps to its own nonzero code while the result is still written out.

The emit step can itself fail (for example, the output file cannot be opened or
written); that failure short-circuits with the emit helper's own exit code before the
status-based code is returned.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
