export const meta = {
  name: 'release-readiness-scope',
  description: 'SCOPE-OUT the release-readiness audit (READ-ONLY, NO code changes) — the user declared a COMPLETION CHECKPOINT and is pivoting to a RELEASE-prep audit: revise/clean/harden everything, then (later planning stage) build forward features (population creation/simulation, imputation, msprime, etc.). THIS workflow produces the SCOPE, not the execution: fan out one auditor per release dimension, each VERIFYING the REAL codebase state (file:line — the user hates forgetfulness/unverified claims), cataloguing the gaps/cleanup/work for release, and triaging each item RELEASE-BLOCKER vs POST-RELEASE with a rough effort. Synthesis writes a master RELEASE-SCOPE.md: the prioritized cleanup/hardening roadmap (blockers first), the effort, the recommended sequence, + a HIGH-LEVEL forward-roadmap section (the post-release feature direction: popgen/sim/imputation/msprime — note + rough-shape only, do NOT deep-design now). GROUND TRUTH: steppe is a GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; the full tool suite (fit engine, f4/f3/f4-ratio/qpDstat, the all-quartets sweep, qpfstats, qpGraph single-fit + topology-search, DATES, the Python bindings + wheel) is BUILT, golden-gated, GPU-bound, host-clean (the host-compute audit campaign), perf-passed, and feature-smoke-tested (docs/feature-matrix.md). main @ cb4e9df. The standing constraints (do not relitigate, scope WITHIN them): single-GPU first (multi-GPU PARKED, 5090 no P2P); EmulatedFp64{40} default + native carve-out; CUDA 13+; REAL AADR only / no synthetic for any RESULT; TGENO-only decode currently (the AADR-TGENO-corrupt history); the CpuBackend is test-oracle-only; GPU-only product. NO code changes — audit + scope + the master doc only.',
  phases: [ { title: 'Parallel release-readiness audit (one auditor per dimension, verified against the codebase)' }, { title: 'Synthesize the master RELEASE-SCOPE roadmap + the forward-feature shape + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const STD = [
  'PROJECT: steppe = GPU/CUDA-13 (Blackwell sm_120) reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ cb4e9df. THE SUITE IS BUILT + golden-gated + GPU-bound + host-clean + perf-passed + feature-smoke-tested: the qpAdm/qpWave fit engine + the S8 rotation; standalone f4/f3/f4-ratio/qpDstat (f2- and genotype-path); the all-quartets GPU sweep; qpfstats (genotype-path joint f2 smoother); qpGraph single-graph fit + topology SEARCH (heterogeneous fleet); DATES (cuFFT LD-decay); Python bindings (steppe._core + the facade) + a GPU-only scikit-build-core wheel. Docs: docs/architecture.md (canonical spec), docs/RUN-SHEET.md (runnable command sheet, predates the 4 newest tools), docs/feature-matrix.md (the smoke + wall-clock matrix), docs/research/* (the GPU-shape designs + the host-compute audit). The user now wants to AUDIT everything for RELEASE.',
  'YOUR JOB (read-only, NO code changes): audit YOUR dimension by READING the REAL codebase (cite file:line — do NOT assert from assumption; the user has repeatedly caught unverified/forgetful claims). Catalogue every gap / cleanup / hardening item for a clean RELEASE. For EACH item: what it is (file:line), why it matters for release, RELEASE-BLOCKER vs POST-RELEASE, a rough effort (S/M/L), and the fix shape. Be honest + specific; flag anything broken/half-done/inconsistent/dead.',
  'STANDING CONSTRAINTS (scope WITHIN these, do NOT relitigate): single-GPU first (multi-GPU PARKED — 5090 no P2P); EmulatedFp64{40} default + native-FP64 carve-out (matmul vs cancellation/reduction); CUDA 13+; REAL AADR only / NO synthetic for any reported result; TGENO-only decode currently (the AADR-TGENO-corrupt history — older GENO/EIGENSTRAT readers are a known gap); CpuBackend = test-oracle ONLY (GPU-only product). The known deferred items: L1-L4 (bounded per-model diagnostics), the 6-pop topology-search scale-up, non-negative constrained-weights qpAdm, older readers.',
  'The box ' + SSH + ' has the built binary + the build system if you need to inspect build/test/run reality. NO code changes — this is a SCOPING audit; the output is findings + the scope.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['dimension','current_state','items','clean_bill','notes'],
  properties: {
    dimension: { type: 'string' },
    current_state: { type: 'string', description: 'the verified current state of this dimension (file:line), what exists + works' },
    items: { type: 'array', description: 'the gaps/cleanup/hardening items for release', items: {
      type: 'object', additionalProperties: false, required: ['item','where','why','priority','effort','fix_shape'],
      properties: {
        item: { type: 'string' }, where: { type: 'string', description: 'file:line / artifact' },
        why: { type: 'string', description: 'why it matters for release' },
        priority: { type: 'string', description: 'RELEASE-BLOCKER | POST-RELEASE | NICE-TO-HAVE' },
        effort: { type: 'string', description: 'S | M | L' },
        fix_shape: { type: 'string' },
      },
    } },
    clean_bill: { type: 'string', description: 'what is already release-ready in this dimension (verified)' },
    notes: { type: 'string' },
  },
}

phase('Parallel release-readiness audit (one auditor per dimension, verified against the codebase)')
const dims = [
  { key: 'correctness-goldens-tests', p: 'AUDIT correctness + the golden/test suite for release. READ the ctest suite (tests/CMakeLists.txt + tests/reference/*) + the goldens (tests/reference/goldens/at2 + dates). Assess: golden INTEGRITY (we found ONE defective golden golden_qpgraph_toposearch_spotcheck.csv — are there OTHER mis-paired/stale/orphaned goldens? un-read goldens? the convertf-PA-vs-TGENO SNP-set tiers — documented + consistent?); test COVERAGE gaps (commands/paths without a golden or a CPU==GPU parity test; error/edge-case coverage; the STEPPE_THOROUGH vs default split); the golden PROVENANCE/reproducibility (how each golden was generated, the AADR-TGENO-corrupt caveat). Scope the correctness-hardening for release.' },
  { key: 'code-quality-cleanup', p: 'AUDIT code quality + cleanup for release. READ across src/ + the repo. Catalogue: DEAD/orphaned code, commented-out blocks, TODO/FIXME/HACK/XXX markers, the §4 layering adherence (cmd/binding TUs CUDA-free; kernels in .cu), naming/style consistency; the ROLLED-BACK overbuild branch wip/fstats-massive-overbuild (mine for parts or delete?); the agentscripts/*.js workflow clutter (~30 files — keep/archive/gitignore?); the leftover stashes (the obsolete wip:sweep-cli-FAILED); the tests/tools/ one-off stagers (untracked); atlas_results/ (untracked); any half-built/disabled code. Scope the cleanup.' },
  { key: 'cli-surface', p: 'AUDIT the CLI surface for release. READ src/app/cli_parse.cpp + the cmd_*.cpp. Assess CONSISTENCY across the 12 subcommands: flag naming (--device/--precision/--jackknife/--f2-dir/--prefix/--pops/--out etc — uniform?), help-string quality + accuracy (we just fixed a stale qpdstat one — any other stale/missing/inconsistent help?), error messages + exit codes (clear? consistent?), the --format options, defaults, the dispatch ergonomics. Is the CLI release-grade for an external user? Scope the CLI polish.' },
  { key: 'python-api-packaging', p: 'AUDIT the Python API + the wheel/packaging for release. READ bindings/steppe/__init__.py (the facade + __all__ + docstrings + type hints) + bindings/module.cpp + pyproject.toml + the scikit-build-core/wheel config. Assess: API completeness/consistency (we just added qpgraph_search — any other gaps/asymmetries vs the CLI?), docstrings/type-hints/the pandas soft-dep, the wheel (GPU-only, the CUDA-13 runtime dependency, the platform tag, manylinux feasibility, pip-installability, the import-without-CUDA failure mode), versioning in pyproject. Is it PyPI-publishable? Scope the packaging for release.' },
  { key: 'perf-scale', p: 'AUDIT performance + scale for release. READ docs/perf/* + docs/feature-matrix.md + the host-compute-audit. Assess: the known perf shapes (the decode I/O-bound, the rotation f2-LOAD-bound, the sweep GPU-compute-bound [nsys-proven], qpfstats 8.1s/40-pop); remaining perf WARTS worth fixing for release vs post-release (the decode streaming/overlap? the rotation f2-load amortization?); the scaling envelope + the OOM/M5-tiering robustness at scale; the deferred perf-ish items. Are any a release-blocker? Scope perf for release (honest: most is done — what truly remains).' },
  { key: 'robustness-formats', p: 'AUDIT robustness + input formats + error-handling for release. READ the Status taxonomy (error.hpp), the M5 device/host/disk tiering (tier_select), input validation, the io readers. Assess: the TGENO-ONLY decode constraint (older GENO/EIGENSTRAT/PACKEDANCESTRYMAP readers — a RELEASE-BLOCKER for external users who have those formats? the AADR-TGENO-corrupt context); edge cases (empty/degenerate inputs, missing pops, all-NaN blocks, OOM); the domain-outcome-as-value (RANK_DEFICIENT etc) handling; reproducibility/determinism. Scope robustness for release.' },
  { key: 'build-ci-distribution', p: 'AUDIT the build system + CI + distribution for release. READ the CMake (presets, options STEPPE_BUILD_CLI/PYTHON, the CUDA arch sm_120/120), pyproject, any CI config. Assess: the build reproducibility + the box-only build reality (is there CI? should there be?); the dependency/toolchain story (CUDA 13, cuBLAS/cuSOLVER/cuFFT/CUB, nanobind); platform/arch support (sm_120 only? other Blackwell/Hopper?); the GPU-only distribution model + how an external user builds/installs; the binary/wheel artifacts. Scope build+release-engineering.' },
  { key: 'documentation', p: 'AUDIT the documentation for release. READ docs/architecture.md, docs/RUN-SHEET.md, docs/feature-matrix.md, docs/research/*, any README. Assess: is there a top-level README + install/quickstart? is the RUN-SHEET current (it predates qpfstats/qpgraph/qpgraph-search/dates)? user-facing docs (per-command usage, examples, the parity/methods documentation, the precision policy, the TGENO caveat) vs internal research docs; API docs; a CHANGELOG. What docs does a RELEASE need that do not exist / are stale? Scope the docs for release.' },
  { key: 'release-eng-licensing-roadmap', p: 'AUDIT release engineering + licensing + triage the deferred items + SHAPE the forward roadmap. Assess: VERSIONING (pyproject says 0.1.0 — the release version + a CHANGELOG); LICENSING (is there a LICENSE? steppe reimplements ADMIXTOOLS 2 math — attribution/derivation; the AADR data licensing for the goldens/tests; the no-AT2-runtime story); what RELEASE means (PyPI wheel? a GitHub release? a preprint/methods note? the support matrix). TRIAGE the known deferred items (L1-L4, the 6-pop scale-up, constrained-weights qpAdm, older readers) as RELEASE-BLOCKER vs POST-RELEASE. THEN HIGH-LEVEL shape the FORWARD roadmap the user named (post-release planning stage): population creation/SIMULATION, IMPUTATION, msprime integration, + other popgen features — rough shape + how they fit the steppe GPU architecture (the genotype-stat seam, the fleet, the f2 cache), NOT a deep design. Scope all of this.' },
]
const audits = await parallel(dims.map(d => () => tryAgent(['You are the release-readiness auditor for: ' + d.key + ' (read-only, verified, NO code changes). ' + d.p, STD].join('\n'), { schema: AUDIT_SCHEMA, label: 'audit:' + d.key, phase: 'Parallel release-readiness audit (one auditor per dimension, verified against the codebase)' })))
const ok = audits.filter(Boolean)
const totalItems = ok.reduce((n, a) => n + (a.items ? a.items.length : 0), 0)
const blockers = ok.reduce((n, a) => n + (a.items ? a.items.filter(i => /BLOCKER/i.test(i.priority)).length : 0), 0)
log('audits: ' + ok.length + '/' + dims.length + '; items: ' + totalItems + '; release-blockers: ' + blockers)
if (ok.length === 0) { log('--- all audits died — HALT'); return { halted: true } }

phase('Synthesize the master RELEASE-SCOPE roadmap + the forward-feature shape + commit')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['committed','blocker_count','blockers','release_sequence','forward_roadmap','headline','note'],
  properties: {
    committed: { type: 'string', description: 'commit hash + doc path' },
    blocker_count: { type: 'number' },
    blockers: { type: 'string', description: 'the RELEASE-BLOCKER items across all dimensions (the must-do-before-release list)' },
    release_sequence: { type: 'string', description: 'the recommended ordered cleanup/hardening sequence to reach release' },
    forward_roadmap: { type: 'string', description: 'the high-level post-release feature direction (popgen/sim/imputation/msprime) shape' },
    headline: { type: 'string', description: 'the one-paragraph release-readiness verdict (how close are we, what is the critical path)' },
    note: { type: 'string' },
  },
}
const synth = await tryAgent([
  'You are synthesizing the release-readiness audit into a master scope. The per-dimension audits:\n<<<\n' + JSON.stringify(ok) + '\n>>>', STD, '',
  'WRITE docs/RELEASE-SCOPE.md: (1) a one-paragraph release-readiness VERDICT (how close, the critical path); (2) the RELEASE-BLOCKERS (the must-do-before-release items across all dimensions, grouped, with effort); (3) the full dimension-by-dimension scope (each dimension: current state + the items, priority-sorted); (4) the recommended ORDERED sequence to reach release; (5) a CLEAN BILL (what is already release-ready); (6) a FORWARD ROADMAP section (high-level: the post-release features the user named — population creation/simulation, imputation, msprime integration + other popgen — rough shape + the steppe-architecture fit, explicitly marked PLANNING-STAGE / not-yet-scoped). De-dup across auditors. Then cd ' + R + ' && git add ONLY docs/RELEASE-SCOPE.md, commit (docs(RELEASE-SCOPE): release-readiness audit + the prioritized cleanup/hardening roadmap (blockers vs post-release) + the forward feature roadmap — the scoping for the release undertaking) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Return the structured synthesis.',
].join('\n'), { schema: SYNTH_SCHEMA, label: 'synth:scope', phase: 'Synthesize the master RELEASE-SCOPE roadmap + the forward-feature shape + commit' })
if (synth === null) { log('--- synth died — HALT'); return { halted: true, audits: ok } }
log('RELEASE SCOPE: ' + synth.committed + ' — blockers=' + synth.blocker_count)
return { audits: ok, synth }
