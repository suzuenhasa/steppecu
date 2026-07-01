# Review — `io/filter/include_exclude` (`SnpMembership` + `read_snp_id_list`)

Unit under review:
- `/home/suzunik/steppe/src/io/filter/include_exclude.hpp` (72 lines)
- `/home/suzunik/steppe/src/io/filter/include_exclude.cpp` (65 lines)

Context read in full: `include/steppe/config.hpp` (`FilterConfig`), `src/io/filter/snp_filter.{hpp,cpp}` (the sole production consumer), `src/io/snp_reader.hpp` (`SnpTable`, the id source), `src/io/CMakeLists.txt` (layering), `tests/unit/test_filters.cpp` and `tests/reference/test_filter_oracle.cu` (the only callers besides snp_filter). Architecture `§1`, `§2`, `§4`, `§5` (S−2/S−1/S0′), `§12`; ROADMAP `§4`, `§6` M2.

---

## Role & layering

This unit resolves the user's `--extract` (include), `--exclude`, and an externally-supplied `prune.in` into one `SnpMembership` object with two `std::unordered_set<std::string>` — a **keep-set** (`include ∪ prune.in`) and a **drop-set** (`exclude`) — plus a per-id `passes()` predicate and a `is_noop()` fast-path flag. `read_snp_id_list` is the file reader for the `prune.in`-style list (whitespace-token-per-line). It is the *only* place in the filter pipeline that does file I/O and the *only* place that can throw, which makes its correctness/edge-case behavior disproportionately important relative to its size.

Layering is **correct and clean** and matches `architecture.md §4` / `src/io/CMakeLists.txt` exactly:
- Pure host C++20; no CUDA, no `core`/`device` include, no upward edge. It includes only `<string>`, `<unordered_set>`, `<vector>`, `<fstream>`, `<sstream>`, `<stdexcept>` and the CUDA-free `steppe/config.hpp`. `steppe_io` links only `steppe::api` + `steppe::warnings` (PRIVATE) — so this TU physically cannot reach a core domain rule or a CUDA header. Verified against the CMake comment block ("LEAF: only the CUDA-free public surface").
- It implements the architecture's "prune.in is READ, NEVER computed" rule (`§1`) and the S−1 "externally-supplied `prune.in` is read, not computed → resolved include/exclude sets folded back into the plan" contract (`§5`, S−1 row). It does **not** compute LD. Good.
- The "membership rule composed in one place so `snp_filter` does not re-derive it" (`§2` single-source, `§8` DRY) is honored: `snp_filter.cpp:99–103` consumes `mem.passes(id)` / `mem.is_noop()` and never re-implements the union/override logic. This is the right decomposition.

So the unit is small, well-placed, and largely well-documented. But "small and well-placed" is not the same as "9.5/10". There is **one substantiated edge-case bug** (a `prune.in` path that is a directory or other openable-but-unreadable node is silently swallowed, not thrown), a **material testability gap** (the entire file-read + throw + prune.in-union path has zero test coverage — the riskiest code in the unit is untested), a few **robustness/normalization gaps** (CRLF/`\r` and embedded-whitespace asymmetry between this reader and the `.snp` id source; duplicate/blank/comment handling unspecified), and several smaller polish items (`reserve`, return-by-value reader form, `std::string_view` query, capability-tier logging hook). None of these are CUDA/precision issues — this is a host leaf — but the bar is a senior 9.5–10 and the bug + the missing tests keep it below that.

---

## Score: 7.5/10 — solid, correct-on-the-happy-path leaf with one real swallow-the-error edge bug and a conspicuous untested failure path

A genuinely good, idiomatic small unit (clean layering, single-source membership rule, dense and accurate doc comments, sensible no-op fast path). It loses points for: (a) `read_snp_id_list` only checking `if (!in)` after construction, which on libstdc++/POSIX does **not** detect a directory (or other openable-but-non-readable) path — the documented "throws if the file cannot be opened" contract is violated for that case and the user silently gets an empty keep-contribution (HIGH, parity-affecting); (b) the prune.in / read / throw path having **no test at all** while the architecture promises it as a first-class S−1 input (MED, but central to M2's "definition of done"); (c) normalization/robustness gaps and small polish. Fix the directory/error-detection bug, add the missing prune.in tests, and tighten normalization and the score moves to ~9.5.

---

## Findings

### (1) Correctness & bugs

**1.1 [HIGH] `read_snp_id_list` does not detect an openable-but-unreadable `prune.in` (directory, FIFO, etc.) — silently yields an empty list instead of throwing.** `include_exclude.cpp:14–28`. The open-failure guard is `if (!in)` immediately after `std::ifstream in(path);` (line 15–19). On libstdc++/POSIX, constructing an `ifstream` on a **directory** path succeeds — `operator bool` and `is_open()` both return true — and the failure only manifests on the first read, which sets `badbit`. I verified this empirically on this box (g++ 11.4, libstdc++):

```
operator bool after ctor: 1
is_open: 1
getline succeeded: 0   line=''
fail after getline: 1   bad: 1
```

So for `cfg.prune_in_path = "/some/dir"` (or a path the process can `open()` but not `read()`), `read_snp_id_list` does **not** throw; the `while (std::getline(...))` loop simply never iterates, `out` stays empty, and `SnpMembership` happily builds with `keep_set_` getting *zero* prune ids. The header explicitly promises the opposite: *"Throws std::runtime_error if `cfg.prune_in_path` is set but the file cannot be opened"* (`.hpp:38–39`, `.hpp:66`). This is a **contract violation and a silent-wrong-result hazard**: a user who points `--prune-in` at the wrong (directory) path gets a *no include constraint at all* for that source rather than a loud failure — directly against the `architecture.md §2` fail-fast principle and dangerous on a parity run where the prune set was supposed to restrict the SNPs.

Why it matters: parity. If the prune.in silently contributes nothing, `is_noop()` may even return `true` when the user clearly requested a constraint, and `snp_filter` skips membership entirely (`snp_filter.cpp:99`). The run completes "successfully" with the wrong SNP set.

Concrete fix: after the read loop, check the stream for a hard read error and throw, e.g.

```cpp
std::ifstream in(path);
if (!in) { throw std::runtime_error("...cannot open SNP-id list (prune.in): " + path); }
std::string line;
while (std::getline(in, line)) { /* ... */ }
if (in.bad()) {  // openable but not readable (e.g. a directory) — fail-fast
    throw std::runtime_error(
        "io::filter::read_snp_id_list: read failed on SNP-id list (not a regular file?): " + path);
}
```

(`in.bad()` is the right test: `getline` hitting EOF normally sets `eofbit|failbit` but *not* `badbit`; a directory/unreadable node sets `badbit`. The empirical run above confirms `bad()==1` for the directory case.) Optionally, harden the open itself by rejecting the path if `std::filesystem::is_directory(path)` before constructing the stream — but the post-loop `bad()` check is the minimal, portable fix. Severity HIGH; effort S; before-M4.5: **yes** (it is a correctness/parity hazard in the M2-owned path and the fix is two lines).

**1.2 [LOW] Empty-string id is a legal, silently-accepted membership key.** `include_exclude.cpp:34–49` and `passes` `:52–63`. If `cfg.include_snp_ids` (or `exclude_snp_ids`) contains an empty string `""`, it is inserted as a real key, and a SNP whose id is `""` would match it. `snp_filter.cpp:100–101` *deliberately* substitutes `kEmptyId` (an empty string) for short/alleles-only records, then calls `mem.passes(id)`. So an empty-string entry in the include/exclude lists silently couples to "alleles-only records" in the consumer. This is almost certainly never what a user means and is an undocumented cross-coupling between two files. Not a crash, but a latent surprise. Fix: either reject/skip empty ids at build time (and document it) or document that empty ids are reserved/ignored. The reader path is fine here — `ls >> id` cannot produce an empty token — so this only applies to the in-memory `cfg` lists. Severity LOW; effort S; before-M4.5: no.

**1.3 [LOW] No de-duplication accounting / no detection of include∩exclude beyond "exclude wins".** Correct behavior (exclude overrides) is implemented and tested (`test_filters.cpp:178–187`). But there is no diagnostic when a user's include and prune.in disagree, or when exclude fully empties the keep-set (`keep_count() > 0`, `drop_count() > 0`, yet every kept id is also dropped → *zero* SNPs pass). That is a legitimate config that produces an all-empty mask downstream and an opaque "0 SNPs" result. Not a bug per se, but a senior bar would surface a warning (see capability-tier finding 11.1). Severity LOW; effort S; before-M4.5: no.

**1.4 [INFO / decision gap, not a bug] The `is_noop()` ⇄ prune.in interaction is correct but subtle.** A `prune_in_path` that resolves to a *non-empty* keep-set makes `is_noop()` false (good). A `prune_in_path` pointing at an **empty but readable** file resolves to an empty keep-set, leaving `is_noop()` true — i.e. "I asked to prune but the prune file was empty" silently becomes "no constraint." Combined with 1.1, this is the same fail-open family. Arguably an empty prune.in *should* keep zero SNPs (intersection with the empty set), not all of them — but ADMIXTOOLS / PLINK `--extract emptyfile` semantics are themselves ambiguous, so I am flagging this as a **documentation/decision gap**, not a definite bug: the header should state explicitly what an empty-but-present prune.in means. Severity LOW; effort S; before-M4.5: no.

### (2) Edge cases & failure modes

**2.1 [HIGH] (same root as 1.1) directory / unreadable prune.in path.** The single most important edge case for this unit (it is the only I/O code) is mishandled. See 1.1 for the empirical evidence and fix.

**2.2 [MED] CRLF and whitespace-in-id asymmetry between `read_snp_id_list` and the `.snp` id source.** `read_snp_id_list` uses `std::istringstream ls(line); ls >> id;` (`:22–24`). `operator>>` into `std::string` uses the stream's locale `ctype` facet to delimit; in the default classic ("C") locale, **`\r` (0x0D) is whitespace** — confirmed via the `isspace` C-locale table and the `operator>>(istream&, string&)` extraction rule ("extraction stops at a character for which `ctype::is(space, ch)` is true; that character is put back"). So a `prune.in` written on Windows (CRLF) is handled **correctly here**: the trailing `\r` is treated as a delimiter and excluded from the token. Good — and worth keeping (see Good patterns).

The risk is the **other side**: the `.snp` ids that `passes()` is compared against are read by `snp_reader` (not this unit) and stored in `SnpTable::id`. If that reader retained a trailing `\r` (e.g. via a bare `std::getline` + substr), then `"rs123"` (from prune.in, no `\r`) would *not* match `"rs123\r"` (from the .snp), and every SNP would silently fail membership. This unit cannot fix `snp_reader`, but the header should **document the normalization contract** ("ids are compared byte-for-byte; both the prune.in token and the `.snp` id must be whitespace-trimmed identically") so the invariant is single-sourced rather than implicit. It is exactly the kind of cross-file assumption that bites on real, mixed-origin data (AADR panels are distributed in varied formats). Severity MED; effort S; before-M4.5: no (but cheap to document now).

**2.3 [LOW] Comment lines / metadata lines in `prune.in` are treated as ids.** The reader takes the *first whitespace token* of every non-blank line (`:24`). A `prune.in` with a `#`-comment header, or a file that is actually an `--extract`-style list with a header row, will silently insert `#chrom` / `#` etc. as a SNP id. That id simply never matches a real SNP, so it is harmless to the result *but* it inflates `keep_count()` and masks an obviously-wrong input file. The header says "blank lines skipped" but is silent on comments. ADMIXTOOLS / PLINK prune.in files are id-per-line with no comments, so this is low-risk for the parity path, but a fail-fast / `#`-skip would be more robust. Severity LOW; effort S; before-M4.5: no.

**2.4 [INFO — no fix] No size / overflow concerns, correctly.** The `.snp` can be ~584k ids (per the header comment) and a prune.in similar; `std::unordered_set<std::string>` and `std::vector<std::string>` handle that fine; no integer-width risk (all `std::size_t`). N/A as a bug — noting it because the prompt asks for the overflow/width sweep: there is genuinely nothing to fix here. The one micro-inefficiency is the intermediate `std::vector<std::string> ids` (finding 8.2).

**2.5 [INFO — no fix] `passes` is correctly `noexcept` and total.** `unordered_set::find` does not throw for a valid set; `std::hash<std::string>` does not throw. The empty-set short-circuits (`!drop_set_.empty()`, `!keep_set_.empty()`) are correct and make the no-op path branch-cheap. No edge case (empty id, very long id, non-ASCII id) breaks `passes`. Good. N/A.

### (3) Numerical / precision vs §12

**N/A — by construction.** This is a pure string-membership host leaf: no floating point, no accumulation, no GEMM, no Ozaki/native-FP64 distinction, no determinism-sensitive reduction. `§12` does not touch it. The one tangential determinism note: `unordered_set` iteration order is unspecified, but this unit never *iterates* the sets (only `find`/`insert`/`size`/`empty`), so there is no order-dependence leaking into any downstream fixed-order combine. Correct. (If a future `keep_count()`-driven diagnostic ever *prints* the sets, it must sort first to stay reproducible — worth a one-line comment if/when that is added.)

### (4) CUDA idioms / RAII / streams / launch config vs §7

**Mostly N/A — host leaf, no CUDA.** The applicable RAII subset is satisfied: `std::ifstream` is RAII (closed on scope exit, including on the thrown-exception path at `:39` — the partially-built `SnpMembership` and the local `ids`/`in` all unwind cleanly; no leak, no half-open handle). `SnpMembership` owns its two sets by value (RAII, no manual lifetime). No global mutable state (`§2`). The one `static const std::string kEmptyId` is in the *consumer* (`snp_filter.cpp:100`), not here, so this unit has none. Good. Nothing to add for `§7`.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**No magic numbers. Clean (genuine pass, not a skip).** There is not a single numeric literal in either file — correctly, because there is no geometry, threshold, or tunable here; the *thresholds* live in `FilterConfig`, which is itself the magic-number single-home per ROADMAP §4. This unit is a model of "no literal survives except true math constants" — there are no constants at all. The only string literals are the two `std::runtime_error` messages, which are appropriate.

### (6) Decomposition / single-responsibility / function size vs §2

**6.1 [INFO] Good decomposition.** `read_snp_id_list` (one job: parse a token-per-line file) is correctly factored out of the constructor and exposed for reuse/test (`.hpp:64–68`). The constructor composes three sources into two sets in ~20 lines. `passes` is 11 lines, single responsibility. All three functions are well under any size threshold and each does exactly one thing. The membership rule is single-sourced (`§2`, `§8`). No complaints on decomposition.

**6.2 [LOW] The "insert all of X into set Y" loop is repeated three times.** `:34–36`, `:40–42`, `:47–49` are three near-identical loops. Trivial, but `set.insert(range.begin(), range.end())` for the vector cases (or a one-line `insert_all` helper) would be DRY-er and let the include/exclude cases use range-insert directly. Very minor. Severity LOW; effort S; before-M4.5: no.

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density

**7.1 [LOW] `read_snp_id_list` uses out-param-append semantics; a return-by-value form would be safer and `[[nodiscard]]`-friendly.** `.hpp:64–68`, `.cpp:14`. It returns `void` and *appends* to `std::vector<std::string>& out` (documented at `.hpp:68` "into `out` (appended)" — good that it is documented). But append-not-replace is a foot-gun if a caller passes a non-empty vector expecting replacement, and a `void` reader cannot be `[[nodiscard]]`. The append form is only used internally at `:38–42`, where it is always a fresh local. Consider making the primary API `[[nodiscard]] std::vector<std::string> read_snp_id_list(const std::string& path)` (return-by-value, RVO-friendly, `[[nodiscard]]`-able) and keeping the append overload only if a caller truly needs accumulation. Severity LOW; effort S; before-M4.5: no.

**7.2 [LOW] `passes(const std::string&)` could take `std::string_view` — but only with transparent lookup.** `:52`, `.hpp:46`. The query never needs ownership and never stores the argument. A `std::string_view` parameter would let `snp_filter` (and tests) query with non-owning views. Caveat: `unordered_set<std::string>::find(string_view)` requires C++20 heterogeneous lookup, which needs a transparent hash+equal (a custom `Hash`/`KeyEqual` with `is_transparent`); without that, a `string_view` arg would silently construct a temporary `std::string` per call (no win). So this is worth doing only *with* the transparent-lookup set type. Given the `.snp` is ~584k ids queried once each, the per-query temporary is not a hotspot today, but for a model-space search that re-runs the filter, heterogeneous lookup is the clean answer. Severity LOW; effort M (needs transparent hash/equal); before-M4.5: no.

**7.3 [INFO] Const-correctness, naming, comment density are excellent.** `passes`/`is_noop`/`keep_count`/`drop_count` are all `[[nodiscard]] ... const noexcept` (`.hpp:46,51,56,57`) — exactly right. Member names (`keep_set_`, `drop_set_`) match the documented vocabulary. Comment density is high and *accurate* (I checked each claim against the code) and ties back to architecture sections with §-references — better than most of the codebase. The trailing-comment style (`///<`) on members matches the surrounding `config.hpp`/`snp_filter.hpp` style.

**7.4 [LOW] Header comment claims "O(1) per-SNP query" — true on average, worst-case O(n).** `.hpp:30–32`. `unordered_set::find` is *average* O(1), *worst-case* O(size) under adversarial hashing. For the parity path this is fine (ids are rs-numbers, well-distributed under `std::hash<std::string>`), and no untrusted adversary supplies the .snp, so this is not a DoS concern. But the comment states "O(1)" unqualified; "amortized/average O(1)" would be precise. Severity LOW (pedantic); effort S; before-M4.5: no.

### (8) Performance

**8.1 [LOW] No `reserve` on the two sets, despite known sizes and a codebase convention of reserving.** `:34–49`. The keep-set will hold `include_snp_ids.size() + ids.size()` and the drop-set `exclude_snp_ids.size()` — all knowable before insertion. The rest of `io` reserves consistently (`ind_reader.cpp:95,125`, `geno_reader.cpp:99,100`, `mind_prepass.cpp:24`, `eigenstrat_format.cpp:26`). For a ~584k-id keep-set, the missing `keep_set_.reserve(...)` means several rehashes (each O(n)) during construction. Cheap, one-pass, build-once cost — not on the per-SNP hot path — so the impact is modest, but it is a free win and a consistency fix. Fix: `keep_set_.reserve(cfg.include_snp_ids.size());` before the include loop, then grow by the prune count after reading; `drop_set_.reserve(cfg.exclude_snp_ids.size());`. Severity LOW; effort S; before-M4.5: no.

**8.2 [LOW] The prune.in path allocates an intermediate `std::vector<std::string> ids` then moves each into the set.** `:38–42`. This is one extra container's worth of allocation for the prune ids. It exists because `read_snp_id_list` is the reusable/testable reader (append-to-vector). Acceptable trade — the indirection buys testability — but if a return-by-value-or-into-set variant is added (7.1), the constructor could insert directly into `keep_set_` and skip the vector. The `std::move(id)` at `:41` is correct and already avoids the string copy. Severity LOW; effort S; before-M4.5: no.

**8.3 [INFO] `passes` ordering is already optimal for the no-op/common case.** It checks `drop_set_` first (usually empty → one `empty()` test, short-circuit) then `keep_set_`. The `!set.empty()` guards mean the no-op membership pays only two `empty()` checks, and `snp_filter` additionally skips `passes` entirely via `is_noop()` (`snp_filter.cpp:99`). Good layered fast-path; no change needed.

### (9) Layering / API / ABI vs §4

**9.1 [INFO] Layering is exemplary — see Role & layering.** Host-only, `io`-leaf, links only `steppe::api`. No core/device dependency, no CUDA. The API surface (`SnpMembership` value type + free `read_snp_id_list`) is minimal and CUDA-free. ABI: it is a static lib (`steppe_io STATIC`), so no cross-DSO ABI concern. Nothing to fix.

**9.2 [LOW] `read_snp_id_list` is named generically but is specifically a `prune.in`-convention reader.** Naming/scope nit: the name says "snp_id_list" but the doc and behavior are the *first-token-per-line, blank-skipped, prune.in convention*. Fine as the one supported format, but if `--extract` files (possibly different conventions) are ever read through this same function, the convention coupling (first token only, comments-as-ids) becomes a hazard (see 2.3). Consider documenting it firmly as "prune.in-style", or adding a small enum/flag if a second convention appears. Severity LOW; effort S; before-M4.5: no.

### (10) Testability vs §13

**10.1 [MED] The entire file-read + throw + prune.in-union path is UNTESTED.** This is the most important testing gap. Searching the whole tree:
- `tests/unit/test_filters.cpp:153–189` covers `SnpMembership` for **include / exclude / exclude-overrides / no-op** — all from in-memory `cfg` vectors. Good.
- `tests/reference/test_filter_oracle.cu` constructs `SnpMembership` six times but always with **empty or default membership** (it exercises MAF/geno/mono/tv/autosome, never include/exclude/prune).
- **`read_snp_id_list` has zero direct tests.** **`cfg.prune_in_path` (the file branch at `:37–43`) is never exercised by any test.** There is **no `prune.in` fixture anywhere** (`find ... -iname '*prune*'` → nothing).

So the only code in this unit that does I/O, that can throw, and that the architecture lists as a first-class S−1 input is completely uncovered — including the very contract (1.1) that is currently wrong. ROADMAP §6's "definition of done" requires the new code path to have an equivalence test at the tight tier and to be sanitizer-clean; the prune.in path meets neither because it is untested. Add: (a) a temp-file round-trip test for `read_snp_id_list` (multi-token lines → first token taken; blank lines skipped; CRLF line → `\r` stripped; trailing-column tolerance); (b) a throw-on-missing-file test; (c) **a throw-on-directory test** (which currently FAILS and pins finding 1.1); (d) a `SnpMembership` test that unions a prune.in file with `include_snp_ids` and confirms both contribute and that `exclude` still overrides a prune-supplied id. Severity MED; effort M; before-M4.5: **yes** (M2 is marked DONE but this path is the unfinished corner of it).

**10.2 [LOW] No test pins the `is_noop()` ⇄ prune.in interaction.** Per 1.4, the "prune requested but file empty → is_noop true" behavior is a decision that should be pinned by a test once the decision is documented. Severity LOW; effort S; before-M4.5: no.

**10.3 [INFO] The unit is otherwise highly testable** — pure functions, value semantics, free-function reader, sizes exposed via `keep_count()`/`drop_count()` "for diagnostics / tests" (`.hpp:55–57`). The design *enables* the tests; they just have not been written for the file path.

### (11) Capability tiers (PRO-6000 vs budget-5090) — TODO "Keeping it GPU-dominant"

**11.1 [LOW] This unit is tier-agnostic but should emit a tagged diagnostic line.** `read_snp_id_list` / `SnpMembership` is pure host I/O and string work — it runs identically on the full-host RTX PRO 6000 path and the budget vast 5090 path; there is no GPU capability to probe, no GDS, no P2P, nothing to degrade. So strictly: **no per-tier code path is needed here.** However, the TODO "CAPABILITY-TIER" rule is that *every* notable runtime decision should be **explicitly logged with a tag**, and this unit makes two user-visible decisions that today happen silently: (i) "read N prune.in ids from `<path>`" and (ii) "resolved keep=K, drop=D; membership is/ is-not a no-op." On a parity run those numbers are exactly what an operator needs to confirm the right SNP set was used (and they would have caught finding 1.1 immediately — "read 0 prune ids" is a red flag). There is no logging facility referenced here at all. Recommend: thread the project's logger (or a `std::function<void(std::string_view)>` sink injected via the resources, per `§9` injected-resources) and emit a tagged line like `[io.filter.prune] read 12345 ids from <path>` and `[io.filter.membership] keep=12345 drop=2 noop=false`. This is the unit's one real capability-tier touchpoint: **observability/tagging**, not hardware degradation. Severity LOW; effort M; before-M4.5: no.

**11.2 [INFO] GDS / large-file note.** If a future capable-tier path reads a very large external keep-list (millions of ids) the streaming `getline` reader is fine on both tiers (it is bandwidth-trivial vs the genotype stream). No GDS / cuFile is warranted here — this is metadata, not the genotype matrix. Correctly out of scope. N/A.

---

## Considered & rejected

- **"`std::getline` retaining `\r` corrupts prune.in ids on Windows-origin files."** *Rejected as a bug in this unit.* I verified that `operator>>` into `std::string` delimits on the classic-locale `ctype` whitespace set, and `\r` (0x0D) *is* in that set. Since the reader extracts via `ls >> id` (not by taking the whole `line`), the trailing `\r` is a delimiter and is excluded from the token. So `read_snp_id_list` is *robust* to CRLF — this is a strength, not a bug. (The residual risk is the asymmetry with how the `.snp` ids are read elsewhere — captured as 2.2.)
- **"Constructor should `clear()` the sets first."** *Rejected.* The sets are freshly default-constructed members; there is nothing to clear. (The append-semantics caveat applies to `read_snp_id_list`'s out-param, not the constructor — captured as 7.1.)
- **"`passes` should lock / be thread-safe."** *Rejected.* `SnpMembership` is built once then queried read-only; concurrent `find` on a `const unordered_set` is safe (no mutation). The architecture's filter runs per-tile on the host without sharing a mutable membership. No data race.
- **"Throw a typed exception instead of `std::runtime_error`."** *Rejected (consistency wins).* The header documents `std::runtime_error` and the codebase surfaces io errors as `std::runtime_error` to the caller (stated in `.hpp:10`). A bespoke exception type would be inconsistent with the rest of `io` for no benefit here.
- **"`unordered_set` is non-deterministic and threatens §12 parity."** *Rejected.* Iteration order is unspecified, but this unit only does `find`/`insert`/`size`/`empty`, never order-dependent iteration, so no nondeterminism leaks downstream. (Flagged only as a forward-looking caveat in §3 if a printing diagnostic is ever added.)
- **"`read_snp_id_list` should `reserve` `out`."** *Rejected as not worth it.* The line count is unknown without a pre-pass; the cost is dominated by the per-line string allocations, not vector growth, and the vector is a transient local. The `reserve` win that *is* worth taking is on the two `unordered_set`s (8.1).
- **"`P`/`M`/index arithmetic overflow."** Not applicable to this unit (that concern lives in `snp_filter`/`derive_per_snp_summary`, reviewed separately). This unit has no index arithmetic. N/A.
- **"Add a CUDA/GPU membership kernel for large keep-lists."** *Rejected.* Membership is metadata-scale and host-cheap; a kernel would violate the `io`-leaf no-CUDA layering for no measurable gain.

---

## What it takes to reach 10/10

1. **Fix 1.1 (HIGH, before-M4.5):** detect the openable-but-unreadable (directory/FIFO) `prune.in` and throw — add the post-loop `if (in.bad()) throw ...` (and/or a `std::filesystem::is_directory` pre-check). This makes the documented "throws if the file cannot be opened" contract true for *all* unreadable cases, not just non-existent ones. Restores fail-fast / parity safety.
2. **Close the test gap 10.1 (MED, before-M4.5):** add the prune.in fixture + tests — `read_snp_id_list` round-trip (multi-token line, blank-skip, CRLF, trailing columns), throw-on-missing, **throw-on-directory** (pins 1.1), and a `SnpMembership` prune∪include + exclude-overrides-prune test. Brings the M2-DONE path to the §6 definition of done.
3. **Document the normalization & empty-file contracts (2.2, 1.4):** state in the header that prune.in tokens and `.snp` ids are compared byte-for-byte after identical whitespace-trim, and define what an empty-but-present prune.in means (no-op vs keep-zero). Single-source the cross-file invariant.
4. **Robustness polish:** skip/reject `#` comment lines or document that the first token of every non-blank line is taken verbatim (2.3); decide on empty-string-id handling (1.2).
5. **Performance/idiom polish:** `reserve` both sets to their known sizes (8.1), matching the rest of `io`; consider a return-by-value `read_snp_id_list` overload (7.1) and direct insertion into the keep-set (8.2).
6. **Observability/capability-tier (11.1):** inject a log sink and emit tagged `[io.filter.prune] read N ids` / `[io.filter.membership] keep=K drop=D noop=…` lines so a parity operator can confirm the resolved SNP set (and so 1.1-style "read 0 ids" is loud).
7. **API ergonomics (7.2):** if the model-space search re-runs membership, switch the sets to transparent (`is_transparent`) hash/equal and make `passes` take `std::string_view` to kill per-query temporaries.

Items 1–3 are the ones that actually move the needle from 7.5 to ~9.5; 4–7 take it to 10.

---

## Good patterns to keep

- **Single-source membership rule.** The keep/drop union+override logic lives *only* here; `snp_filter` consumes `passes()`/`is_noop()` and never re-derives it (`§2`, `§8`). This is exactly the DRY/single-responsibility shape the architecture asks for.
- **Clean `io`-leaf layering.** No CUDA, no core/device dependency, links only `steppe::api`. A textbook isolated leaf.
- **No-op fast path.** `is_noop()` + the `!empty()` short-circuits in `passes` make the default (parity) path branch-cheap and let the consumer skip the per-SNP query entirely. Good layered optimization.
- **Token-extraction reader is CRLF-robust by construction.** Using `ls >> id` (not the raw `line`) means Windows-origin prune.in files parse correctly with no special-casing — a quietly correct choice. Keep it (and keep it documented).
- **Excellent, accurate, §-referenced doc comments.** Every nontrivial claim in the header maps to real behavior and to an architecture section; comment density matches the surrounding files. Const-correctness and `[[nodiscard]] ... const noexcept` on the query surface are spot-on.
- **No magic numbers.** Thresholds live in `FilterConfig`; this file has zero numeric literals. Model `§4`/ROADMAP §4 compliance.
- **RAII / exception-safety on the throw path.** The `ifstream` and the partially-built sets unwind cleanly when `read_snp_id_list` throws mid-construction; no leak, no half-open handle.
