# steppe — Kimi-Review Action Plans (index)

These three plans are the **decision-ready, not-yet-done** work distilled from triaging the
external Kimi code review (`docs/kimireview/ASSESSMENT.md`). They cover only §3 of that
assessment — *genuinely open, in-scope-for-this-release* items, each re-verified against the
code at HEAD with real `file:line`. The just-finished HIGH/MED/LOW source-hygiene campaign
(`a2f9d64..HEAD`, ASSESSMENT §2) is **already landed and excluded**. So is every §4-rejected
item — 2nd compute stream, NCCL all-reduce, multi-GPU release gate, excise-the-oracle,
`steppe::Index`, `JsonWriter` reshape, capability pure-virtuals, kernel-path fuzzing,
mock/no-GPU backend — because they fight the §12 parity law, the deterministic-reduction
design, the protected math vocabulary, the kernel ABI, or the single-GPU / GPU-only product
shape. Standing constraint: **nothing builds locally** (RTX 2070 / CUDA 11.8) — every
GPU/golden/CLI check runs on `box5090`; only doc/host-unit items verify off-box.

## The three plans

- **`01-open-worth-doing.md`** — the High + Med non-CI work, in five clusters:
  **A** §12 parity-law integrity (wire `cusolverDnSetDeterministicMode` + assert + fix the
  stale §12 claim — the one HIGH that sits *inside* the frozen parity law);
  **B** error-handling/exit-code taxonomy (post-write `good()` torn-write → exit; OOM
  `5→3`; replace try/catch-as-capability-detection with a capability query);
  **C** IO/output correctness (dedup the byte-identical genotype decode front-end into one
  `core::` helper; route the bypassing emitters through shared `csv_field`/`json_quote`;
  unify `--out` vs `--out-dir`);
  **E** build/supply-chain hygiene (pin CPM `EXPECTED_HASH`; wire `STEPPE_SANITIZER` —
  feeds the CI plan); **D** doc-vs-as-built honesty (`architecture.md` core→io / C++-surface /
  CMake≥3.28 sync; single-source the version).
- **`02-ci-plan.md`** — the proper phased CI/CD, two tracks. **Phase 0 (Cluster A):** a
  host-only **CUDA-free** PR lane — `STEPPE_ENABLE_CUDA` (defaults ON, box configure stays
  byte-identical), test-guards + CTest LABELS, `host-ci.yml` on `ubuntu-latest`, and
  check-only clang-format / clang-tidy / arch-grep §4 gate / cmake-lint. **Phase 1–2
  (Cluster B):** the GPU **parity** lane via an ephemeral vast.ai JIT self-hosted runner,
  hybrid topology (PR=host smoke, nightly/on-demand=GPU parity), secrets/AADR data tiering,
  the §12 determinism CI assertion, benches kept *off* the gate; persistent-box and hosted-GPU
  options are recorded as DEFER / REJECT. Everything is behavior-neutral by construction
  (check-only linters, default-ON CUDA); parity acceptance stays the box's job.
- **`03-low-polish.md`** — sixteen XS–S polish items, fifteen behavior-neutral.
  **A** tooling (`STEPPE_WERROR=OFF`+ccache, CTest labels, pytest/ruff/mypy, meta.json
  schema-version+checksum, one `regenerate_goldens.sh`); **B** small robustness (NVTX wiring,
  `error_code::message()` forwarding, geno-hash thread guard, drop a `mutable`, pool cuSOLVER
  workspace); **C** API/DX portfolio polish (`Precision` factories, `f2_at` accessor,
  `f2_to_numpy`→`memcpy`, `examples/`, Doxygen+pdoc). Only `regenerate_goldens.sh` is
  golden-adjacent, and only on deliberate human invocation behind a version preflight.

## Recommended cross-plan sequence

Front-load the cheap, parity-safe, high-leverage work, then the CI engine, then polish:

1. **§12 reconcile** — `01` A1 (HIGH, inside the parity law; small wire+assert or doc-fix).
2. **CI Phase 0** — `02` Cluster A: the host-only CUDA-free lane + check-only gates.
3. **Error/exit taxonomy + dedup** — `01` B1 (torn-write), B2 (OOM 5→3), C1 (decode dedup,
   a 4-way parity-divergence risk), C2 (escaping).
4. **Doc-sync + version** — `01` D1/D2 (nearly free, judged for a portfolio release);
   land `01` E2 (`STEPPE_SANITIZER`) here so the CI sanitizer lanes have their seam.
5. **CI Phase 1** — `02` Cluster B: the vast.ai GPU parity runner + hybrid wiring + the
   §12 determinism assertion (golden-gated).
6. **Polish in batches** — `03`: header-sugar batch, XS robustness, then `examples/`/Doxygen.

## Top-level priority snapshot

| Tier | Items |
|---|---|
| **Do first (HIGH, parity-safe)** | §12 determinism reconcile (`01`A1); CI Phase 0 host lane (`02`A1–A3); torn-write exit (`01`B1) |
| **Next (MED, high cost/value)** | OOM 5→3 (`01`B2); genotype decode dedup (`01`C1); emitter escaping (`01`C2); capability-query (`01`B3); doc-sync + version (`01`D1/D2); `STEPPE_SANITIZER` (`01`E2); GPU parity CI (`02` Cluster B) |
| **Polish (P1–P3 batches)** | all of `03` — Werror/ccache + CTest labels first; golden regen last |

**One line:** parity-law integrity and a CI floor first, then honest fault taxonomy +
the decode dedup, then the GPU parity runner, then polish — nothing here moves a committed
golden except the human-run `regenerate_goldens.sh`.
