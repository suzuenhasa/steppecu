# steppe — COMMENT STYLE STANDARD

Status: **DRAFT for user approval.** Companion to `NAMING-STYLE-STANDARD.md`. Governs the
comment-hygiene pass (one agent per file, comment-only). **DESCRIPTIVE-first:** it codifies the
conventions steppe already follows; it is prescriptive only where it resolves an inconsistency
or a tone problem.

---

## 1. Purpose & scope

Governs **code comments** in `src/`, `include/`, `bindings/` (C++/CUDA + the Python facade).
It does **not** govern the design docs (those are prose by design). Goal: factual, lean,
consistent comments — the **file header carries the substance**; functions are commented only
where something is non-obvious. This is a **comment-only** pass: it changes **no code**.

## 2. Tone — the core rule

State **plain facts**: what it is, what to expect, the non-obvious why. A comment sells nothing.

**Remove** marketing / self-praise / superlatives:
- Out: "fastest", "the first / only GPU X", "blazing(ly)", "world-class", "state-of-the-art",
  "cutting-edge", "seamless", "unmatched", "N× faster than AT2", and standalone performance
  claims.
- A *technical* use of a word is fine — `// pick the fastest tier that fits VRAM` is a real
  selection criterion, not a boast. The ban is on **self-praise**, not on the vocabulary.
- Benchmarks and comparisons live in `docs/perf/`, never in a code comment.

**Keep** honest, useful bluntness — a comment that **warns or informs the reader** earns its
place even with personality:
- Keep: `// Do NOT reorder these two lines — the stream must drain first or the next memcpy
  returns cudaErrorInvalidValue (measured on box5090).`
- Keep: `// Not sure why this ordering is required, but removing it breaks the qpadm_rotation
  golden.`
- A little cheek is allowed. **The test:** does it *warn / inform* (keep) or *sell* (cut)?

**No** essays. **No** changelog / history in comments (git has it). **No** restating what the
code plainly does.

## 3. Altitude — where the information lives

**File header (the primary home):** a few sentences — the file's role, the key
invariants/contracts/gotchas, the non-obvious justification. This is where "the information"
lives. Not an essay, not a tutorial.

**Functions / blocks:** a comment **only** when something is non-obvious. Otherwise nothing —
the name and the code carry it.

**"Important" (the keep-a-comment test)** — a comment is warranted iff it records one of:
- a non-obvious **invariant / contract** (ordering, units, ownership, lifetime, "must run after X");
- a **parity** constraint (§12) or a numeric footgun (cancellation-sensitive, NEVER-emulated);
- a **gotcha / footgun** (a non-obvious failure mode, a measured quirk);
- a **why-not-the-obvious-way** (why this design over the naive one);
- an **external reference** (an AT2 source line, a paper, an RFC).

**Not** important = restating the code, obvious steps, a comment that just names the next line.

**Load-bearing comments are always kept** (tighten if verbose, never delete): parity pins,
"don't touch" warnings, the AT2 citations, fail-fast rationale.

## 4. Format — codify + resolve the inconsistencies

- **`///`** = a doc comment on a **declaration** — the file-header block in a header, and any
  public class / function / field in `include/steppe/` or a cross-TU seam. Doxygen-style.
  *(7,769 in use — the dominant doc style; keep it.)*
- **`//`** = an **implementation** comment — inline in `.cpp`/`.cu`, and the file-header block
  in an implementation `.cpp`/`.cu`.
- **Section separators:** `// ---- short label ----` (regular `//`, terse), used *sparingly* to
  mark phases inside a genuinely long function. *(452 in use, already consistent — keep this
  form; keep the label short, e.g. `// ---- S4: Qinv ----`, never a 60-char banner. Do not use
  `///` for separators.)*
- **Doxygen tags:** use `@param` / `@return` on the **public API** (`include/steppe/`) and
  wherever a param/return is non-obvious; do **not** tag every trivial internal function.
  `@brief` is optional — the first sentence is the brief. *(183 `@param` / 20 `@return` today.)*
- **File-header block:** keep the existing `// <path>` / `//` / `// <role + invariants>` shape;
  trim it to substance.
- One space after `//` / `///`. **No** decorative fills (`// ==========` / `// **********`
  banners); a bare `// ----` separator is the only rule-line allowed.

## 5. Provenance tokens & doc references

**Remove** internal-review tags from comments: `[N.N]`, `[Gx.y]`, `cleanup X-N`, `group-N`,
milestone / ticket tokens (~119 across 28 files). The fix stays; the internal ticket reference
goes.

**Internal steppe doc references** (`architecture.md §X`, `docs/design/…`, internal findings):
do **not** keep them in the comment — those docs are internal and will not ship, so the pointer
would dangle for a reader. State the fact / invariant **directly** instead, and **log the
reference** (file:line → the concept it pointed at) to the external-reference task list (§9), so
the concept gets covered in the *public* documentation.

**Genuine external citations** (AT2 source lines like `qpsubs.c:1698`, papers, RFCs): **keep**
them in the comment — they point at accessible sources and are real parity provenance — and
also **log** them to §9 so the reference map is complete.

## 6. Python docstrings (`bindings/steppe`) — different rules

Docstrings are the **user-facing API reference**, not implementation comments — **keep and
improve** them (a clear one-line summary + args/returns where non-obvious). Do **not** trim them
like internal `.cu` notes. The same applies to public-header (`include/steppe/`) contract
comments: they are API, not scratch.

## 7. Examples (before → after)

- **Marketing → factual:**
  `// The world's first & only GPU qpAdm — 70× faster than AT2 via the blazing Ozaki path.`
  → `// qpAdm fit over the f2 blocks. Emulated-FP64 (Ozaki) matmul path (§12).`
- **Essay → header:** a 12-line header narrative → 3 sentences (role · the one invariant · the
  justification).
- **Load-bearing → kept verbatim:**
  `// NEVER EmulatedFp64 here — cancellation-sensitive; native FP64 (§12).`
- **Honest-cheeky → kept:**
  `// The stream MUST drain here or the next memcpy returns cudaErrorInvalidValue (measured).`
- **Obvious → deleted:** `// loop over the SNPs` above `for (s = 0; s < M; ++s)`.
- **Provenance → stripped:** `// ... the RAII is load-bearing ([16.1]).`
  → `// ... the RAII is load-bearing.`

## 8. How the pass is applied

One agent per file, **comment-only** (touches **no** code). Per comment: **keep-tighten** or
**delete** (no per-file archive dump — git is the archive; genuine design rationale is promoted
to a design doc only if it is real "why", which is rare). The pass also emits the §9
external-reference rows for its file. Per-area commit, gated on: a **comment-only diff proof**
(strip comments from the file before/after → the non-comment code must be byte-identical) + the
file still builds. Goldens do not need re-running — a proven comment-only change is bit-identical
by construction — but the build catches a botched edit (a deleted `*/`, a broken line). The §12
parity law is loaded for every agent so load-bearing comments are preserved.

## 9. The external-reference task list (a pass byproduct)

While rewriting, the pass collects every "see X" reference — internal doc pointers (removed from
the code) and external citations (kept) — into a single list at
`docs/cleanup/EXTERNAL-REFERENCES.md`, one row per reference:

| source (file:line) | referenced | concept to document publicly |

This is **not shipped**. It is the **input for the public documentation**: it tells the author
which concepts the code leans on (the §12 parity law, the block partition, the f2 estimator, the
AT2 mirroring) that a user-facing doc should explain. The author elaborates from it.
