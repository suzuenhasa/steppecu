export const meta = {
  name: 'feature-cleanups',
  description: 'THREE small cleanups surfaced by the feature smoke (83a008b) — fast, NO golden runs (just build + Python smoke + targeted ctest). (1) qpgraph_search Python FACADE: the CLI command + the engine steppe._core.run_qpgraph_search EXIST, but there is NO public Python wrapper + no __all__ entry (it is the lone un-wrapped engine; qpgraph + qpadm_search ARE wrapped). ADD a qpgraph_search() facade in bindings/steppe/__init__.py mirroring the qpadm_search facade (~line 784) + a QpGraphSearchResult accessor class mirroring the other Result classes + the __all__ entries (qpgraph_search + QpGraphSearchResult). It calls _core.run_qpgraph_search and exposes the global-best topology + score + the per-candidate vector + the argmin. (2) STALE qpdstat HELP: the qpdstat subcommand description AND the --prefix flag help in src/app/cli_parse.cpp still say "normalized-D ... not yet implemented (Part B)" — WRONG/stale (the genotype-path normalized-D IS implemented, GPU, golden-gated test_cli_dstat_geno green). FIX the help strings to describe the implemented Part B (genotype-path normalized-D from --prefix PREFIX.{geno,snp,ind}). (3) STALE CMake ARG: the cli_dstat_geno add_test in tests/CMakeLists.txt carries a stale convertf-PA 3rd arg the test ignores (and --prefix is TGENO-only, errors clearly on a PA/GENO prefix); align/remove it so it reflects the raw TGENO prefix the gate actually uses. REQUIREMENTS: NO golden regen (no admixtools/R, no re-running goldens — these are facade/help/cmake fixes); additive Python facade (mirror the existing pattern, no engine change); GPU-FIRST (the search still runs on the fleet, unchanged); SINGLE-GPU --device 0; CUDA 13+; REAL AADR for the Python smoke. VERIFY (FAST): build-rel (CLI + Python) + (a) Python smoke — import steppe; steppe.qpgraph_search on the real 5-pop f2 returns the global-best matching the CLI/golden (1590 candidates, best ~0.36555, argmin); steppe.qpdstat still works; (b) the qpdstat --help no longer says "not yet implemented"; (c) TARGETED ctest -R "dstat|qpgraph|py" to confirm no regression. Discipline: implement (3 fixes) + build + Python smoke + targeted ctest -> verify (qpgraph_search callable from Python == golden, qpdstat help fixed, no regression) -> commit. FAIL-PROTOCOL: commit ONLY on green; on BAD (qpgraph_search facade returns wrong/non-sane, a regression) HALT + report + defer; NEVER silently fail/revert; on failure git stash push -u + HALT. NEVER touch a golden.',
  phases: [ { title: 'Implement the 3 cleanups + build + Python smoke + targeted ctest' }, { title: 'Verify qpgraph_search-from-Python + qpdstat help + no regression + commit-or-HALT' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -16 && echo === HELP === && ./build-rel/bin/steppe qpdstat --help 2>&1 | grep -iE \"normalized|implement|part.b|prefix\" | head && echo === TARGETED CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel -R \"dstat|qpgraph|py_\" --output-on-failure 2>&1 | tail -20; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 83a008b. The feature smoke (docs/feature-matrix.md) confirmed all 12 CLI commands + the Python API run on real AADR, and surfaced 3 small cleanups (a Python facade gap + 2 stale strings). These are FAST fixes — NO golden runs.',
  'CLEANUP 1 — qpgraph_search Python facade: bindings/steppe/__init__.py exposes qpgraph (~line 580) + qpadm_search (~line 784) but NOT qpgraph_search (the topology-search engine _core.run_qpgraph_search EXISTS, just unwrapped + absent from __all__ lines 21-43). ADD a qpgraph_search() facade mirroring qpadm_search + a QpGraphSearchResult accessor class + the __all__ entries. Additive, no engine change.',
  'CLEANUP 2 — stale qpdstat help: src/app/cli_parse.cpp — the qpdstat subcommand description + the --prefix flag help say "normalized-D ... not yet implemented (Part B)". WRONG — Part B (genotype-path normalized-D from --prefix) is implemented + golden-gated (test_cli_dstat_geno). Fix the strings to describe the working Part B.',
  'CLEANUP 3 — stale CMake arg: tests/CMakeLists.txt cli_dstat_geno add_test carries a stale convertf-PA 3rd arg (ignored by the test; --prefix is TGENO-only). Align/remove so it reflects the raw TGENO prefix the gate uses.',
  'REQUIREMENTS: NO golden regen; additive Python facade (mirror the existing pattern); the search/fit math UNCHANGED; GPU-FIRST (search on the fleet, unchanged); SINGLE-GPU --device 0; CUDA 13+; REAL AADR for the Python smoke; no synthetic.',
  'VERIFY (FAST): build-rel (CLI+Python) + Python smoke (import steppe; steppe.qpgraph_search on the real 5-pop f2 returns the global-best matching the golden — 1590 candidates, best ~0.36555, argmin; steppe.qpdstat works) + qpdstat --help no longer says "not yet implemented" + TARGETED ctest -R "dstat|qpgraph|py_". NOT a full golden-regen.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout/clean; NEVER touch a golden. commit ONLY on green (qpgraph_search callable from Python + matches the golden, qpdstat help fixed, no regression). On BAD (facade wrong / regression) HALT + report + DEFER. On any failure ' + STASH + ' "wip:feature-cleanups-FAILED-<reason>" + HALT.',
  'SINGLE-GPU --device 0. RELEASE build-rel. Box ' + SSH + '; nvcc -> ' + PATHENV + '. §4 layering. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Implement the 3 cleanups + build + Python smoke + targeted ctest')
const fixer = await tryAgent([
  'You are doing the 3 feature cleanups (additive Python facade + 2 stale-string fixes; NO golden runs). Do NOT commit. Do NOT git checkout/clean.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+smoke (' + BUILD + ').', '',
  'READ bindings/steppe/__init__.py (the qpadm_search facade ~784 + qpgraph ~580 + __all__ ~21 + the Result classes) + bindings/module.cpp (the _core.run_qpgraph_search binding + the result it returns) + include/steppe/qpgraph_search.hpp (QpGraphSearchResult fields) + src/app/cli_parse.cpp (the qpdstat subcommand + --prefix help) + tests/CMakeLists.txt (cli_dstat_geno add_test). IMPLEMENT: (1) the qpgraph_search() Python facade + QpGraphSearchResult class + __all__ entries (mirror qpadm_search; expose global-best topology + score + argmin + the per-candidate vector); (2) fix the stale qpdstat help strings to describe the working genotype-path normalized-D (Part B, --prefix PREFIX.{geno,snp,ind}); (3) align/remove the stale convertf-PA CMake arg on cli_dstat_geno. Build (CLI+Python) + Python smoke (import steppe; steppe.qpgraph_search on the real 5-pop f2 -> 1590 candidates, best ~0.36555, argmin matches; steppe.qpdstat works) + the qpdstat --help check + TARGETED ctest -R "dstat|qpgraph|py_". SANITY: qpgraph_search callable from Python + returns the golden global-best; the help no longer says "not yet implemented"; no regression; NO golden touched. Report files, the facade, the smoke result (qpgraph_search Python output), the help fix, the ctest. Do NOT commit. BAD -> STOP + report.',
].join('\n'), { label: 'implement:cleanups', phase: 'Implement the 3 cleanups + build + Python smoke + targeted ctest' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true } }
await tryAgent(['BUILD-REPAIR for the feature cleanups. Accumulated edits (do NOT clean/revert/touch-a-golden; git stash only if forced). Reach a CLEAN Release build (CLI+Python) + green TARGETED ctest -R "dstat|qpgraph|py_" + a working steppe.qpgraph_search Python smoke, patching only trivial errors. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x. NON-trivial -> STOP + report. Report build + smoke + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement the 3 cleanups + build + Python smoke + targeted ctest' })

phase('Verify qpgraph_search-from-Python + qpdstat help + no regression + commit-or-HALT')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','qpgraph_search_python_works','qpgraph_search_matches_golden','qpdstat_help_fixed','cmake_arg_fixed','no_regression','no_golden_touched','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if steppe.qpgraph_search is callable from Python + returns the global-best matching the golden (1590 candidates, best ~0.36555, argmin), the qpdstat help no longer says "not yet implemented", the stale CMake arg is fixed, no dstat/qpgraph/py regression, no golden touched, single-GPU' },
    fail_severity: { type: 'string' },
    qpgraph_search_python_works: { type: 'boolean' },
    qpgraph_search_matches_golden: { type: 'string', description: 'the Python qpgraph_search global-best vs the golden (count/best/argmin)' },
    qpdstat_help_fixed: { type: 'boolean' },
    cmake_arg_fixed: { type: 'boolean' },
    no_regression: { type: 'boolean' }, no_golden_touched: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the 3 feature cleanups. implementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — the qpgraph_search facade is additive (mirrors qpadm_search, no engine change), the qpdstat help strings are fixed (no "not yet implemented"), the CMake arg is aligned, no golden modified. (2) ' + BUILD + ' — Python smoke: steppe.qpgraph_search on the real 5-pop f2 returns the golden global-best (1590 candidates, best ~0.36555, argmin); the qpdstat --help shows the working Part B; TARGETED ctest -R "dstat|qpgraph|py_" green. (3) single-GPU, real AADR. PASS only if qpgraph_search works from Python + matches the golden + the help is fixed + no regression. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed bindings/source/cmake (NEVER git add dot; never aadr/ atlas_results/ tests/tools/; never a golden), commit (fix(features): qpgraph_search Python facade (was CLI-only) + fix stale qpdstat Part-B help + the cli_dstat_geno CMake arg — surfaced by the feature smoke; no golden runs) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs/feature-matrix.md (mark qpgraph_search Python Y; the qpdstat help fixed). ',
  'ON BAD: DO NOT git checkout/clean, DO NOT revert, DO NOT touch a golden. ' + STASH + ' "wip:feature-cleanups-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:cleanups', phase: 'Verify qpgraph_search-from-Python + qpdstat help + no regression + commit-or-HALT' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ feature cleanups ' + verdict.commit_hash + ' — qpgraph_search Py: ' + verdict.qpgraph_search_matches_golden)
else log('--- feature cleanups FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
