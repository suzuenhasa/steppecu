export const meta = {
  name: 'wheel-mpy2',
  description: 'M(py-2): the GPU-only Python WHEEL + extract_f2 from Python. M(py-1) already built the nanobind module (steppe._core + the pure-Python facade bindings/steppe/__init__.py: read_f2/qpadm/qpwave/qpadm_search/f4/f3/f4ratio/qpdstat/read_fstats) + the pyproject.toml/scikit-build-core scaffold. M(py-2) = (1) WIRE extract_f2 from Python — a binding to the genotype->f2 extract entry (mirror the existing read_f2/run_* binding pattern; reuse the steppe::access lib + the extract-f2 core), returning an f2 dir on disk and/or an in-memory F2Blocks, GPU-only; (2) FINISH + BUILD the scikit-build-core wheel — pyproject invokes CMake with the CUDA toolchain, compiles the _core CUDA extension, packages ONE GPU-only wheel; nail the CUDA-runtime handling (declare a CUDA-13 runtime dependency rather than bundle the toolkit; the _core.so resolves libcudart at load), the platform tag, and that pandas stays a lazy soft-dep; (3) VERIFY: build the wheel, pip install it into a CLEAN venv on the box, and SMOKE-TEST the INSTALLED wheel — import steppe; reproduce the committed golden_fit0 through the installed API (read_f2 + qpadm, weights rtol 1e-6); run steppe.extract_f2 on a small REAL pop subset and confirm the f2 matches a direct read. Discipline: design -> implement -> build+install+smoke -> commit. GPU-ONLY (no CPU runtime), SINGLE-GPU, REAL AADR (no synthetic), golden-gated. FAIL-PROTOCOL: NEVER git checkout/clean; on failure git stash push -u + HALT; verdict classifies severity.',
  phases: [ { title: 'Design extract_f2 binding + the wheel build (verify the scaffold)' }, { title: 'Implement extract_f2 + finish the wheel + build' }, { title: 'Build wheel + clean-venv pip install + smoke-test + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -15 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -20; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'
const PA = '/workspace/data/aadr/converted_pa/v66_HO_pa'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 4c54a06. M(py-1) (b6902f5) built the nanobind module: bindings/module.cpp -> steppe._core (marshalling-only) + the pure-Python facade bindings/steppe/__init__.py (read_f2/qpadm/qpwave/qpadm_search + f4/f3/f4ratio/qpdstat + read_fstats -> pandas/numpy). The pyproject.toml + scikit-build-core scaffold landed with M(py-1). The build is via CMake STEPPE_BUILD_PYTHON=ON (nanobind 2.13 via pip CMake config).',
  'GOAL M(py-2): (1) extract_f2 from Python — bind the genotype->f2 extract entry (the extract-f2 core; reuse steppe::access + the existing extract path) so Python can BUILD an f2 dir from a genotype prefix + pops, GPU-only; mirror the existing read_f2/run_* binding shape (capsule/path-return, NOT a giant copy). (2) the GPU-only WHEEL via scikit-build-core: finish pyproject so `python -m build` / `pip wheel` produces ONE wheel that builds + bundles _core; handle the CUDA-13 runtime as a RUNTIME dependency (the _core.so dynamically resolves libcudart at load — do NOT bundle the toolkit; document the CUDA-13 requirement); pandas stays a lazy soft-dep; the platform tag is a linux GPU wheel.',
  'VERIFY (the gate): build the wheel, create a CLEAN venv on the box, pip install the wheel, and SMOKE-TEST the INSTALLED package (not the source tree): `import steppe`; read the committed golden_fit0 f2 + run steppe.qpadm -> weights rtol 1e-6 vs golden_fit0; run steppe.extract_f2 on a SMALL real pop subset from the convertf-PA ' + PA + ' and confirm the resulting f2 matches a direct read_f2 to ~1e-12. Real AADR, no synthetic. The existing pytest (tests/python) must stay green; the FULL STEPPE_THOROUGH ctest stays green.',
  'GPU-ONLY (no CPU runtime; no-device -> clear fault). SINGLE-GPU --device 0 (multi-GPU PARKED). RELEASE build. NO golden regen. Box ' + SSH + '; nvcc -> ' + PATHENV + '. The box has python3.12 + pip; install build/scikit-build-core deps as needed. §4 layering (the binding TU CUDA-free).',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout -- . / git clean -fd. On ANY failure ' + STASH + ' "wip:wheel-FAILED-<reason>" + HALT. NON-trivial blocker (e.g. the CUDA-wheel packaging genuinely needs a decision) -> STOP + report (defer). Classify minor vs bad.',
  'nothing builds locally; all build/test on the box.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Design extract_f2 binding + the wheel build (verify the scaffold)')
const design = await tryAgent([
  'You are designing M(py-2) (verify-before-implement; NO code changes). READ: bindings/module.cpp + bindings/steppe/__init__.py (the M(py-1) binding pattern — read_f2/run_* + the capsule/path-return idiom), the extract-f2 core entry (the genotype->f2 path; src/core + src/app/cmd_extract_f2.cpp + steppe::access), the existing pyproject.toml + the scikit-build-core scaffold + the CMake STEPPE_BUILD_PYTHON wiring, how _core.so links/resolves libcudart. Determine: (1) the extract_f2 Python API (signature: prefix + pops + options -> f2 dir path and/or F2Blocks; GPU-only; reuse which core entry) + the binding shape; (2) the wheel plan (what pyproject/scikit-build-core changes finish it; the CUDA-runtime-as-dependency strategy; the platform tag; the clean-venv install + smoke-test plan). Cite file:line. Return the design; do NOT implement. If the CUDA-wheel packaging has a genuine fork (bundle vs depend, manylinux feasibility), surface it.',
  STD,
].join('\n'), { label: 'design:wheel', phase: 'Design extract_f2 binding + the wheel build (verify the scaffold)' })
if (design === null) { log('--- wheel design died — HALT'); return { halted: true } }

phase('Implement extract_f2 + finish the wheel + build')
const fixer = await tryAgent([
  'You are implementing M(py-2) per this design:\n<<<\n' + design + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT: (1) the extract_f2 binding (Python steppe.extract_f2 -> the genotype->f2 entry, GPU-only, reuse the existing core; mirror the M(py-1) shape, no dup compute). (2) finish the scikit-build-core wheel (pyproject so a wheel builds + packages _core; CUDA-13 runtime as a dependency; pandas lazy). Build + full STEPPE_THOROUGH ctest + the existing pytest green. SANITY: STEPPE_BUILD_PYTHON build clean; steppe.extract_f2 works from the source tree (matches a direct read_f2). Report files changed, the extract_f2 API, the wheel/pyproject state, the FULL ctest + pytest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:wheel', phase: 'Implement extract_f2 + finish the wheel + build' })
if (fixer === null) { log('--- wheel fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for M(py-2). Accumulated edits (do NOT clean/revert; git stash only if forced). Reach a CLEAN Release build (STEPPE_BUILD_PYTHON=ON) + green ctest + green pytest, patching only trivial wiring. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report build + ctest + pytest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement extract_f2 + finish the wheel + build' })

phase('Build wheel + clean-venv pip install + smoke-test + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','extract_f2_works','wheel_builds','clean_install_smoke','golden_through_wheel','gpu_only','ctest_green','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if extract_f2 from Python works (matches a direct read_f2), the scikit-build-core wheel BUILDS, a CLEAN-venv pip install of the wheel imports + reproduces golden_fit0 through the INSTALLED API (weights rtol 1e-6) + steppe.extract_f2 on real data works, GPU-only, full ctest + pytest green, single-GPU, no synthetic' },
    fail_severity: { type: 'string' },
    extract_f2_works: { type: 'boolean' }, wheel_builds: { type: 'boolean' },
    clean_install_smoke: { type: 'boolean', description: 'a CLEAN venv pip-installed the wheel + import steppe + the smoke test passed (the INSTALLED package, not the source tree)' },
    golden_through_wheel: { type: 'boolean' }, gpu_only: { type: 'boolean' }, ctest_green: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the extract_f2 API + the wheel name/tag + the install smoke result; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for M(py-2) (adversarial). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — extract_f2 reuses the existing extract core (no dup), the wheel/pyproject is sound, GPU-only, binding TU CUDA-free. (2) ' + BUILD + ' — ctest + pytest green. (3) THE WHEEL GATE: on the box, build the wheel (python -m build or pip wheel), create a FRESH venv, `pip install` the built wheel, then in that venv run a SMOKE TEST against the INSTALLED steppe (NOT the source tree): import steppe; reproduce golden_fit0 via steppe.read_f2+steppe.qpadm (weights rtol 1e-6); steppe.extract_f2 on a small real pop subset == a direct read_f2. (4) GPU-only, single-GPU, real AADR. PASS only if the CLEAN-venv installed wheel passes the smoke test. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed binding/pyproject/cmake/python/doc files (NEVER git add dot; never aadr/ atlas_results/), commit (feat(M(py-2)): extract_f2 from Python + the GPU-only scikit-build-core wheel — clean-venv pip install reproduces golden_fit0 through the installed API + extract_f2 on real AADR; GPU-only, single-GPU) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (RESUME/TODO/RUN-SHEET: M(py-2) done, the pip-install + extract_f2 usage). ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:wheel-FAILED" (capture the ref). Classify fail_severity. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:wheel', phase: 'Build wheel + clean-venv pip install + smoke-test + commit' })
if (verdict === null) { log('--- wheel verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ M(py-2) wheel ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- M(py-2) FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
