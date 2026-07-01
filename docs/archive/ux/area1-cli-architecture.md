# Area 1 — Terminal & CLI Experience: Engineering Architecture

> **Scope split.** `docs/EXPERIENCE-PLAN.md` is the **product** plan for Area 1 (§2 "Area 1 — Terminal & CLI Experience": *what* and *why* — the feature table, the delight rationale, the pre-ship sprint's Phase A). **This document is the engineering architecture** (*how*): the tech-stack decision, the fit against the real `src/app` code, the folder scaffolding, the exact interface signatures, the packaging, and the build order. It references the plan; it does not restate the feature tables.

> **Area role.** Area 1 is the one experience area that is **primarily C++**, not Python-over-the-facade — because the CLI *is* a C++ binary (`src/app`, `add_executable(steppe_app)`). Area 1 **owns Shared-Infra (i): the single CLI output seam** (`emit_to_destination` in `src/app/cmd_emit.hpp`). Every visual feature in the plan's Area-1 table — auto/pretty output, significance color, the qpAdm model line, `--explain`, fuzzy "did you mean", progress bars, sparklines, shell completion, examples-rich help, the timing line — lands **inside the existing `steppe` binary**, through one seam, with **zero new distributable package** and **zero third-party dependency added to the CLI subtree** beyond the CLI11 it already fetches.

---

## 1. The hard constraints this architecture is designed around

Grounded in the real code (line refs verified in this repo):

1. **One output seam, no other path to stdout.** `emit_to_destination(config, prefix, write)` (`cmd_emit.hpp:60`) is the sole funnel: parse `--format` once → select `std::cout` or open `--out` (`binary|trunc`) → invoke the command's `write(std::ostream&, OutputFormat)` → `finish_emit()` flush+`good()` check. All 14 subcommands route through it (`cmd_qpadm.cpp:138`, `cmd_fstat_sweep.cpp:203`, …). **One Table branch here reaches every command for free.**

2. **The goldens are the regression guard, and they diff CSV/JSON byte-for-byte.** Pretty output must be a **new** `OutputFormat`, never a mutation of the existing `emit_qpadm_result`/`emit_f4_result`/… serializers. `--format csv|tsv|json` must stay byte-identical.

3. **`--format` is validated in two places** and both reject unknown tokens today: `config_builder.cpp:317` (the authoritative validator: `csv|tsv|json`) and `parse_output_format` (`result_emit.hpp:43`, the defensive re-map inside `emit_to_destination`). Adding `auto`/`table`/`pretty` means **extending both**.

4. **`std::exit()` fires inside each CLI11 callback** (`cli_parse.cpp:389` etc.) — the process terminates *inside* the callback, so **there is no post-emit in-process hook**. A completion/timing banner must be printed by `run_<cmd>_command` **before it returns**, or by the pretty serializer itself. There is no place "after emit, before exit" to bolt onto.

5. **Zero progress plumbing exists.** `grep isatty|progress|ProgressCallback|on_progress` over `src/app` and `include/` = **0 hits**. Compute is a single blocking call (`run_qpadm(...)` `cmd_qpadm.cpp:119`, `run_qpadm_search(...)` `cmd_rotate.cpp:180`) with **no callback seam**. Coarse phase markers are cheap (in the `cmd_*.cpp` bodies); a true streaming `%` bar needs a **new core callback** — a real engine change.

6. **The wheel does not build the CLI.** `pyproject.toml` sets `-DSTEPPE_BUILD_CLI=OFF`, `build.targets=["_core"]`. `pip install steppe` gives `import steppe` but **no `steppe` on PATH**. Whether Area 1's C++ experience reaches a `pip` user is an explicit packaging decision (§7), not an existing seam.

7. **`PopResolver` already holds the label vector + `by_name_` map** (`pop_resolver.hpp:70-71`) and fails fast on an unknown name (`pop_resolver.cpp:30`). The fuzzy scorer drops in **exactly at that failure site**. It is compiled into `steppe::access` (`src/access/CMakeLists.txt`), the shared lib linked by **both** the CLI and the Python `_core` — so one scorer serves both.

---

## 2. Tech-stack decision (researched, with alternatives weighed)

The plan poses the choice explicitly: *hand-rolled ANSI vs a header-only lib (tabulate / ftxui / fmt) vs a thin Python pretty layer; where completion + fuzzy-validation live.* The decisions:

### 2.1 Rendering engine → **hand-rolled ANSI + a tiny internal `render/` module. No new third-party dep.**

| Option | Verdict | Why |
|---|---|---|
| **Hand-rolled ANSI (chosen)** | ✅ | The existing `result_emit.cpp` *already* aligns columns and formats doubles via `ostream <<` + `fmt_double`/`csv_field` primitives. A pretty table is the **same populated struct** with width-aware padding + per-cell SGR wrapping — a ~200-line `ansi.hpp` + `table.hpp` reusing `fmt_double` for byte-consistency with the CSV numbers. The *value* of Area 1 is the **domain coloring rules** (`\|Z\|≥3` bold-red, qpAdm `p≥0.05` green) and the **model-line sentence**, not box-drawing — those are ours to write regardless of library. |
| [**tabulate**](https://github.com/p-ranav/tabulate) (header-only static tables) | ❌ | Reasonable and MIT, but (a) it injects a vendored dep into an `src/app` subtree that today fetches **exactly one** dep (CLI11) — the CMake comment (`src/app/CMakeLists.txt`) is emphatic that CLI11 + CPM stay *private to the app subtree*; (b) steppe's cells are domain-specific (per-cell significance color, NA sentinels, right-justified sci-notation doubles that must match `fmt_double`, the unicode weight bar) — we'd fight tabulate's generic cell model as much as use it; (c) net LOC saved is near zero once the coloring logic (the actual work) is written. |
| [**FTXUI**](https://github.com/ArthurSonzogni/FTXUI) (functional TUI framework) | ❌ **here**, ✅ **for Area 4/6** | FTXUI is an *event-loop TUI* (components, Flexbox, keyboard/mouse). Massive overkill for **static** aligned tables, and a heavy dep for a subtree kept lean. It is exactly the right tool for the **interactive** `steppe pops pick` fzf picker (Area 4) and a live TUI (Area 6) — call it out **there**, not for Area 1's static rendering. |
| [**{fmt}**](https://github.com/fmtlib/fmt) | ❌ | We're already on `CMAKE_CXX_STANDARD 20`, so `std::format` is available *in principle*; and the existing code uses `snprintf`/`ostream` (no `std::format`, no `fmt` anywhere — grep-confirmed). Adding `fmt` for what is a `std::setw`/`snprintf` padding job is a dependency for no gain. Stay with the existing `ostringstream` + `fmt_double` primitives (a `<format>`-availability check is Open Question 3). |
| **Thin Python pretty layer** | ❌ | The CLI *is* the C++ binary; a Python renderer would **not** get the C++ pretty output and would duplicate every serializer. Python pretty output belongs to Area 2 (`steppe.plot` `backend="terminal"` via plotext/rich) for *notebook/facade* users — a separate surface, not the CLI. |

**Net:** Area 1 adds **no** third-party library to the C++ CLI. The CLI subtree stays CLI11-only; the base wheel stays `numpy`-only.

### 2.2 Fuzzy "did you mean" scorer → **hand-rolled bounded Levenshtein + token-overlap in `steppe::access`. No new dep.**

- The vocabulary is small (≤ 4,265 AADR Group IDs; a per-f2-dir `pops.txt` is far smaller) and the query fires **once on a cold error path**, not in a hot loop. A bounded-band DP Levenshtein + shared-token Jaccard is ~50 lines, dependency-free, and matches the plan's "Levenshtein drops straight in."
- [**rapidfuzz-cpp**](https://github.com/rapidfuzz/rapidfuzz-cpp) (MIT, fast, `token_set_ratio`/`token_sort_ratio`) is the **named escalation path**, *not* the default: reserve it for Area 4's *interactive* fzf-style picker where the filter re-runs on every keystroke over 4,265 labels and latency matters. For a one-shot "did you mean" the hand-rolled scorer wins on zero-dep.
- **Home:** `src/app/fuzzy.{hpp,cpp}`, compiled into **`steppe::access`** (alongside `pop_resolver.cpp`, which already lives in `src/app/` but is compiled by `steppe_access`). This gives the **one** scorer to both the CLI `resolve()` failure path **and** the Python facade / Area 4 `pops.suggest()`.

### 2.3 Shell completion → **`steppe completions {bash,zsh,fish}` prints a static script + a hidden `steppe __complete` dynamic helper.**

- **CLI11 has no built-in completion** ([issue #343](https://github.com/CLIUtils/CLI11/issues/343), confirmed for 2.4.x). So we generate it ourselves. The subcommand list, flags, and enum values are all enumerable from the `CLI::App` model at runtime → a `steppe completions <shell>` subcommand emits a hand-authored, shell-native script.
- **Dynamic pop-name completion** follows the **Cobra `__complete` pattern**: the emitted script calls a hidden `steppe __complete -- <current line>` which parses the `--f2-dir`/`--f2` already on the line, reads its `pops.txt` (self-contained; `read_f2_dir` already does this), and prints candidate labels. When Area 4's catalog exists, `__complete` upgrades to complete **AADR Group IDs** with sample counts (Shared-Infra dependency, §6).

### 2.4 Progress / throughput bars → **stderr-only, `isatty(2)`-gated, two tiers.**

- **Tier 0 (ships now, no core change):** coarse phase markers (`read-dir` → `upload-f2` → `fit` → `emit`) + the final timing line, printed from the `cmd_*.cpp` bodies, gated on `isatty(fileno(stderr)) && !quiet`. Purely app-layer.
- **Tier 1 (a real core change, phased):** a `ProgressCallback = std::function<void(const ProgressEvent&)>` threaded into the **batched** entry points (`run_qpadm_search`, `run_fstat_sweep`, `run_dates`, `extract-f2`) so a live `14.5M quartets/s — 41% — ETA 1m44s` bar is possible. This touches golden-gated compute headers — see Open Question 2.

### 2.5 Sparklines & `--explain` → hand-rolled, dependency-free.

- Braille (U+2800) / block-element sparklines in `render/sparkline.hpp` (DATES decay, sweep `\|Z\|` histogram, rotation p-strip). Trivial 8-dot mapping; works over SSH with no Python. (Python `plotext` terminal plots are Area 2's territory, for facade users.)
- `--explain` is pure C++ string-building from the already-computed result fields (`p`, `feasible`, `rankdrop_p_nested`, weights-in-`[0,1]`) — `render/explain.{hpp,cpp}`.

---

## 3. Fit decision: **EXTEND-EXISTING** (the C++ emit seam), with an internal `render/` module

Area 1 does **not** create a new distributable package. It:
- **extends** the output seam (`OutputFormat` enum + `emit_to_destination` + the `config_builder` validator),
- **adds an internal, non-distributable `src/app/render/` module** compiled into the existing `steppe_app` target,
- **adds one shared scorer** to the existing `steppe::access` lib,
- **adds two subcommands** (`completions`, `examples`) to the existing `CLI::App`.

All of it lives inside the existing C++ CLI target and existing shared libs. No new pip package; no new distributable. (Enum choice: `extend-existing`; the `render/` subdir is internal scaffolding within that target, not a package.)

---

## 4. Folder scaffolding (actual tree + file names)

```
src/app/
  render/                         # NEW internal module — pure host C++20, NO CUDA, compiled into steppe_app
    ansi.hpp                      # SGR codes, ColorMode {Auto,Always,Never}, isatty gates, NO_COLOR honoring
    ansi.cpp
    table.hpp                     # width-aware aligned table renderer; reuses fmt_double for numeric cells
    table.cpp
    significance.hpp              # the coloring RULES: |Z|/p -> Style (z>=3 bold-red, 2-3 yellow, p>=.05 green)
    model_line.hpp                # the qpAdm "Target = 0.74*Steppe + 0.26*Anatolia | p=0.31 | FEASIBLE" sentence
    model_line.cpp
    explain.hpp                   # --explain narrator (plain-language reading from result fields)
    explain.cpp
    sparkline.hpp                 # braille/block sparklines (dates decay / sweep |Z| hist / rotation p-strip)
    sparkline.cpp
    progress.hpp                  # Tier-0 stderr Progress + phase markers + the timing line (isatty(2)-gated)
    progress.cpp
  pretty_emit.hpp                 # NEW: emit_<stat>_pretty(...) — one per result struct, sibling of result_emit.hpp
  pretty_emit.cpp                 #      reuses the SAME populated QpAdmResult/F4Result/... — no compute, no re-format
  fuzzy.hpp                       # NEW: did-you-mean scorer (bounded Levenshtein + token overlap)
  fuzzy.cpp                       #      compiled into steppe::access (shared with the Python facade)
  cmd_completions.cpp             # NEW: `steppe completions {bash,zsh,fish}` + hidden `steppe __complete`
  cmd_completions.hpp
  cmd_examples.cpp                # NEW: `steppe examples` cheat-sheet + per-subcommand help footers
  cmd_examples.hpp
  completions/                    # NEW: hand-authored shell script templates emitted by `steppe completions`
    steppe.bash
    steppe.zsh
    steppe.fish

  # --- EXTENDED existing files (no new file) ---
  cmd_emit.hpp                    # + resolve Auto->Table|Csv on isatty(stdout); route Table to pretty_emit
  result_emit.hpp                 # + OutputFormat::{Table} ; parse_output_format accepts auto/table/pretty
  cli_parse.cpp                   # + add_output_flags: --color/--no-color ; add_common_flags: --progress/--quiet/--verbose
                                  # + register cmd_completions / cmd_examples subcommands
  pop_resolver.cpp                # + on resolve() miss, call fuzzy::suggest() to build the "did you mean" reason

src/core/config/
  config_builder.cpp              # + accept format tokens {auto,table,pretty} (currently rejects non csv/tsv/json)
  cli_args.hpp / run_config.hpp   # + color_mode / progress / verbosity fields (mirrors the existing format/out_file)

tests/app/                        # snapshot tests for the NEW format only (csv/tsv/json goldens are untouched)
  test_pretty_snapshot.cpp        # golden-free: assert structure/color-escapes under a forced-TTY shim
  test_fuzzy_suggest.cpp          # unit: "Anatolia_N" -> ranks Anatolia_Neolithic first
  test_completions.cpp            # `steppe completions bash` emits a sourceable script; __complete parses --f2-dir
```

**CMake wiring (edits, not new targets):**
- `src/app/CMakeLists.txt`: add `render/*.cpp`, `pretty_emit.cpp`, `cmd_completions.cpp`, `cmd_examples.cpp` to `add_executable(steppe_app ...)`. Embed `completions/steppe.{bash,zsh,fish}` as string resources (a generated header) so the binary is self-contained.
- `src/access/CMakeLists.txt`: add `../app/fuzzy.cpp` to `add_library(steppe_access ...)` (so the Python `_core` gets the scorer too).

---

## 5. Key interfaces (signatures + how they plug into existing code)

### 5.1 Extend the format enum + the Auto resolution (the foundation)

```cpp
// result_emit.hpp — ADD Table to the existing enum (Csv/Tsv/Json stay first, byte-identical).
enum class OutputFormat { Csv, Tsv, Json, Table };

// parse_output_format ALSO accepts "auto"/"table"/"pretty".  "auto" is NOT a wire format:
// it is a pre-format token that cmd_emit resolves against isatty(stdout).  This keeps the
// pure-format enum {Csv,Tsv,Json,Table} the serializers switch on.
```

```cpp
// cmd_emit.hpp — the ONE change that reaches all 14 commands.  emit_to_destination gains an
// Auto->concrete resolution BEFORE it selects the destination, and routes Table to pretty_emit.
//   * format == "auto"  &&  out_file empty  &&  isatty(fileno(stdout))  ->  Table
//   * else (piped, --out, or explicit csv/tsv/json)                     ->  the existing path
// Serializers are unchanged; only the funnel learns Table.  Piped output is byte-identical.
```

### 5.2 The pretty serializers — reuse the SAME populated structs

```cpp
// pretty_emit.hpp — one per result struct, mirroring result_emit.hpp's signatures EXACTLY so the
// cmd_*.cpp write-lambda just switches on fmt.  NO new compute, NO re-derived numbers.
namespace steppe::app {

void emit_qpadm_pretty (std::ostream&, const QpAdmResult&,  const std::string& target,
                        const std::vector<std::string>& left_labels, const render::RenderOpts&);
void emit_rotation_pretty(std::ostream&, std::span<const QpAdmResult>, const std::string& target,
                        const std::vector<std::vector<std::string>>& left_per_model, int right_n,
                        const render::RenderOpts&);
void emit_qpwave_pretty (std::ostream&, const QpWaveResult&, const std::vector<std::string>& left,
                        int right_n, const render::RenderOpts&);
void emit_f4_pretty     (std::ostream&, const F4Result&, /* p1..p4 label vectors */, const render::RenderOpts&);
void emit_f3_pretty     (std::ostream&, const F3Result&, /* p1..p3 */, const render::RenderOpts&);
void emit_f4ratio_pretty(std::ostream&, const F4RatioResult&, /* p1..p5 */, const render::RenderOpts&);
void emit_sweep_pretty  (std::ostream&, const SweepResult&, const PopResolver&, int k, const render::RenderOpts&);
}  // the cmd_*.cpp write-lambda becomes:  if (fmt==Table) emit_*_pretty(...); else emit_*_result(...);
```

### 5.3 The qpAdm model line (the headline)

```cpp
// render/model_line.hpp — the "qpAdm as a sentence" builder.  Pure function over the fields the
// engine already fills (weight[], p, feasible-from-weights, status).  Returns a styled string.
std::string qpadm_model_line(const QpAdmResult& r,
                             const std::string& target,
                             const std::vector<std::string>& left_labels,
                             const render::RenderOpts& opt);
// -> "Target = 0.74*Steppe + 0.26*Anatolia | p=0.31 | FEASIBLE"   (green/red by p and feasibility)
```

### 5.4 Coloring rules + ANSI gate

```cpp
// render/ansi.hpp
enum class ColorMode { Auto, Always, Never };
struct RenderOpts { ColorMode color = ColorMode::Auto; bool unicode = true; int width = 0; };
// resolve():  Always -> on ; Never -> off ; Auto -> isatty(fileno(stdout)) && getenv("NO_COLOR")==nullptr
bool color_enabled(ColorMode, int fd);
std::string sgr(std::string_view body, Style);   // wraps body in the SGR only if color_enabled

// render/significance.hpp — the domain rules (the ACTUAL value of Area 1)
Style style_for_z(double z);          // |z|>=3 -> BoldRed ; 2<=|z|<3 -> Yellow ; else -> Default
Style style_for_qpadm_p(double p);    // p>=0.05 -> Green (not rejected) ; else -> Red
Style style_for_weight(double w);     // w<0 || w>1 -> Red (infeasible)
```

### 5.5 Fuzzy "did you mean" — plugged into `PopResolver`

```cpp
// fuzzy.hpp  (compiled into steppe::access; shared by CLI + Python facade)
struct Suggestion { std::string label; int distance; double score; };
std::vector<Suggestion> suggest(std::string_view query,
                                const std::vector<std::string>& vocab,   // PopResolver's labels_
                                std::size_t k = 3, int max_distance = 4);

// pop_resolver.cpp — at the resolve() MISS (currently pop_resolver.cpp:30), enrich the reason:
//   "unknown population 'Anatolia_N'. Did you mean: Anatolia_Neolithic, Iran_N? (3 close of 711)"
// The by_name_ miss path already has labels_ in hand — one call, no new state.
```

### 5.6 Progress (Tier 0 now) + the core callback (Tier 1, phased)

```cpp
// render/progress.hpp — Tier 0, app-only, stderr, isatty(2)-gated.  No core change.
class Progress {
public:
    explicit Progress(bool enabled /* = isatty(fileno(stderr)) && !quiet */);
    void phase(std::string_view label);                 // "read-dir" -> "upload-f2" -> "fit" -> "emit"
    void tick(std::uint64_t done, std::uint64_t total, double rate_per_s);  // "41% - 14.5M/s - ETA ..."
    void done(std::string_view summary);                // "fit in 0.91s on RTX 5090, emu40"
};

// Tier 1 (PHASED — a real engine API add, Open Question 2): a core-level callback the batched
// entry points invoke.  Lives in a CUDA-FREE header so it does not violate the app layering.
//   using ProgressCallback = std::function<void(const ProgressEvent&)>;   // {done,total,rate}
//   run_qpadm_search(..., ProgressCallback on_progress = {});             // default no-op
//   run_fstat_sweep (..., ProgressCallback on_progress = {});
// The cmd_*.cpp passes a lambda that drives render::Progress::tick.  Golden-gated headers change.
```

### 5.7 Completion + examples subcommands

```cpp
// cmd_completions.cpp
int run_completions_command(const cfg::RunConfig&);   // prints completions/steppe.<shell> to stdout
int run_complete_command  (const cfg::RunConfig&);     // hidden `steppe __complete -- <line>`:
                                                       //   parse --f2-dir on <line> -> read_f2_dir -> print pops.txt labels
// cmd_examples.cpp
int run_examples_command  (const cfg::RunConfig&);     // a top-level cheat-sheet; also drives the
                                                       //   per-subcommand help footer (CLI::App::footer(...))
```

### 5.8 New flags (attach to the existing binders in `cli_parse.cpp`)

```cpp
// add_output_flags  (cli_parse.cpp:98): default --format becomes "auto"; add:
//   --color {auto,always,never}   (also honors NO_COLOR env)
// add_common_flags  (cli_parse.cpp:83): shared by every subcommand:
//   --progress / --no-progress    (default: on when isatty(stderr))
//   --quiet / -q                  (suppress the stderr timing/phase lines)
//   --verbose / -v                (the "fit in 0.91s on RTX 5090, emu40" line)
// config_builder.cpp:317: extend the format validator to accept {auto,table,pretty}.
```

---

## 6. Shared-infrastructure dependencies

Area 1 **owns Shared-Infra (i): the CLI output seam** (`cmd_emit.hpp`). Its contract to Areas 2/3/7 is *"the golden CSV/JSON stays byte-identical"* — those areas consume the Python **result classes**, not this C++ seam, so ownership here is mostly the invariant that the pretty format never perturbs the machine formats.

Area 1 **depends on** (references, does **not** re-scaffold):

| Dependency | Owner | What Area 1 needs it for | Blocking? |
|---|---|---|---|
| **AADR pops catalog — `steppe.pops`** | **Area 4** | (a) dynamic completion of AADR **Group IDs** (beyond a single f2-dir's `pops.txt`); (b) the **enriched** "did you mean" with sample counts (`"…Russia_Samara_EBA_Yamnaya (n=10, cov 1.2x)?"`). | **No.** The CLI-local variants — `pops.txt` completion via `read_f2_dir`, and did-you-mean against `PopResolver::labels_` — are **self-contained and ship first**. The catalog is a strict *upgrade*, wired at the same call sites. |
| Result classes / DataFrames | facade (existing) | Not consumed by the C++ CLI; noted only because Areas 2/3/7 consume them off the seam Area 1 owns. | n/a |
| f2 cache / `steppe serve` daemon / provenance `run.json` | Areas 5/6/5 | **None** for Area 1's core CLI features. The timing line and phase markers are self-contained. (Live re-fit CLI would ride the daemon, but that is Area 6.) | No |

The scorer in `steppe::access` is itself **shared infrastructure Area 1 contributes**: Area 4's `pops.suggest()` and the Python facade reuse the same `fuzzy::suggest`.

---

## 7. Packaging

**Packaging extra: none.** Area 1's features compile **into the `steppe` C++ binary**; there is no heavy Python dep to gate behind a `steppe[...]` extra. The base wheel stays `numpy`-only.

**The real packaging decision (Open Question 1): how does `pip install steppe` yield a `steppe` command?** Today `-DSTEPPE_BUILD_CLI=OFF` (`pyproject.toml`) means it does not. Two paths:

- **(A) Flip `STEPPE_BUILD_CLI=ON` and ship the C++ binary as a wheel script.** The pretty output, completion, sparklines all come "for free" because they're *in* the binary. Cost: the wheel build must compile the CLI, which links `steppe::device` (needs the CUDA toolchain at **build** time — though the CLI itself reaches the GPU only through CUDA-free seams). Distribution via `[tool.scikit-build] wheel.scripts` or installing the binary into the wheel's `scripts/` dir. **Recommended** for the full C++ experience.
- **(B) A pure-Python `steppe.__main__:main` `console_scripts` dispatcher over the facade.** Keeps the wheel pure-Python-launchable (GPU work still server-side), but a Python CLI would **not** inherit the C++ pretty output and would have to reimplement it — defeating the "one seam" design.

**Recommendation:** the C++ binary is the CLI (A); if `pip`-installability without a build-time CUDA toolchain is required for 0.1.0, ship the binary via the GitHub Release / conda channel and keep (B) only as a thin `steppe` → "call the binary or the facade" shim. Version stays single-sourced from `project(VERSION)` (drives the wheel, `steppe --version`, `steppe.__version__`) — unchanged.

---

## 8. Intra-area build order

1. **Format/color plumbing (foundation).** `OutputFormat::Table` + `parse_output_format` accepts `auto/table/pretty`; `config_builder.cpp:317` validator extended; `cmd_emit.hpp` Auto→Table resolution on `isatty(stdout)`; `--color/--progress/--quiet/--verbose` flags. *Zero golden risk* (new format; csv/tsv/json untouched). Nothing renders until this lands.
2. **`render/` primitives.** `ansi.{hpp,cpp}` (SGR + `color_enabled` + `NO_COLOR`), `table.{hpp,cpp}` (width-aware, reuses `fmt_double`), `significance.hpp` (the |Z|/p color rules).
3. **Pretty serializers + the qpAdm model line (headline).** `pretty_emit.{hpp,cpp}`; wire each `cmd_*.cpp` write-lambda to branch on `fmt==Table`. Model line first (highest delight-per-line).
4. **Fuzzy "did you mean."** `fuzzy.{hpp,cpp}` into `steppe::access`; enrich `pop_resolver.cpp` resolve-miss reason. (Reused by Area 4 later.)
5. **`--explain` + result-aware one-liners.** `render/explain.{hpp,cpp}`.
6. **Progress Tier 0 + timing line.** `render/progress.{hpp,cpp}`; phase markers + `done()` banner in the `cmd_*.cpp` bodies (printed *before* `run_<cmd>_command` returns — the `std::exit` constraint). *Then* Tier 1 core `ProgressCallback` (deferred; Open Question 2).
7. **Sparklines.** `render/sparkline.{hpp,cpp}` → DATES decay, sweep `\|Z\|` histogram, rotation p-strip.
8. **Completion + examples.** `cmd_completions.cpp` (+ `completions/steppe.{bash,zsh,fish}`, hidden `__complete`), `cmd_examples.cpp` (+ per-subcommand `footer()`).

Maps onto the plan's Phase A (foundation + headline first). Steps 6b–8 are the plan's "steady value" stretch.

---

## 9. Open questions

1. **CLI-in-wheel distribution (§7):** flip `STEPPE_BUILD_CLI=ON` to bundle the C++ binary (build-time CUDA toolchain required, full C++ experience) vs a pure-Python `console_scripts` shim (no build-time CUDA, but no C++ pretty output). Which distribution model for 0.1.0?
2. **Tier-1 progress core seam:** adding `ProgressCallback = std::function<void(const ProgressEvent&)>` to `run_qpadm_search`/`run_fstat_sweep`/`run_dates`/`extract-f2` is a real, golden-gated *engine* API change (default-arg no-op keeps parity, but it edits compute headers). Land it for 0.1.0, or ship only Tier-0 coarse phase markers and defer live `%` bars post-ship?
3. **`std::format` vs `snprintf`/`ostream`:** does the Blackwell build box toolchain (GCC/Clang version) ship complete libstdc++ `<format>`? If yes, `render/` may use it; if not, stay on the existing `ostringstream` + `fmt_double` primitives (safer, matches current code).
4. **Unicode fallback:** the weight bar + braille sparklines assume a UTF-8 TTY. Auto-detect via `LANG`/`LC_*` and fall back to ASCII, or gate behind an explicit `--unicode/--ascii`? (`RenderOpts.unicode` is the hook; the *policy* is open.)
5. **Dynamic-completion source before Area 4:** ship `__complete` reading only the on-line `--f2-dir`'s `pops.txt` now, upgrading to the AADR Group-ID catalog (with counts) when Area 4 lands — confirm this staging is acceptable.
6. **Golden the pretty layout?** Recommend **not** goldening `--format table` early (only csv/tsv/json stay byte-diffed) so the human layout can evolve; add structure/escape **snapshot** tests (`tests/app/test_pretty_snapshot.cpp`, forced-TTY shim) once the layout stabilizes. Confirm.
