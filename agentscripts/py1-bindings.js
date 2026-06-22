export const meta = {
  name: 'py1-bindings',
  description: 'M(py-1): a minimal viable nanobind Python module exposing the EXISTING steppe entries to numpy/scipy/pandas — GPU-only, golden-gated via pytest. NO msprime. Expose: run_qpadm / run_qpwave / run_qpadm_search (results as Python objects -> pandas-friendly: weights/se/z/p/feasible/status per model) + read_f2_dir (the f2 tensor + pops -> numpy). nanobind via CPM (mirror how CLI11 is pulled in), GPU-only (no CPU runtime), built under the existing STEPPE_BUILD_PYTHON option (currently an OFF stub — make it real). The design is already researched: docs/research/interop-usecases.md (MUST = results->pandas + an f2/Q-V-N numpy entry; nanobind NOT PyCUDA) + docs/design/cli-bindings.md (the bindings contract + the M(py-1)/M(py-2) split). Discipline: design/verify -> implement -> verify. GATE: a pytest that reproduces golden_fit0 (qpadm) THROUGH the Python API (weights/p match the committed golden within tier) + the f2 numpy array round-trips (matches read_f2_dir); ideally also a qpwave + a rotate call. NO AT2 (the committed goldens are the gate). FAIL-PROTOCOL (user-mandated): on ANY failure DO NOT git checkout/clean (never destroy) — git stash push -u the attempt (preserved + reviewable) and HALT; the verdict classifies severity so the orchestrator decides. M(py-2) (extract_f2 from Python + the scikit-build-core wheel) is the NEXT workflow, not this one. SINGLE-GPU; REAL data; commit-on-green; HALT-on-fail-with-stash.',
  phases: [ { title: 'Design the bindings (verify)' }, { title: 'Implement the nanobind module + build' }, { title: 'Verify golden-through-Python + commit-or-stash' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
// BUILD: configure with the Python bindings ON, build, and (if the module built) run pytest.
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -25; " + CORECLEAN + "'"
// PRESERVE-not-destroy: stash (incl untracked) instead of the destructive checkout/clean.
const STASH = 'cd ' + R + ' && git stash push -u -m'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 9b914e2. The CLI is complete (extract-f2/qpadm/qpwave/qpadm-rotate wired, golden-gated). The C++ entries exist: run_qpadm / run_qpwave / run_qpadm_search (include/steppe/qpadm.hpp), read_f2_dir (the f2-dir loader). There is a STEPPE_BUILD_PYTHON CMake option that is currently an OFF stub (no bindings/ module yet) — VERIFY its exact state in code.',
  'GOAL M(py-1): a minimal nanobind Python module (GPU-only) exposing run_qpadm / run_qpwave / run_qpadm_search (results -> Python objects that are pandas-friendly: per-model weights/se/z/p/chisq/dof/feasible/status, the rankdrop) + read_f2_dir (the f2 tensor + pops -> numpy float64 array). nanobind via CPM (mirror how CLI11/other deps are pulled in CMake). NO CPU runtime (steppe is GPU-only; memory cpu-is-test-only). NO msprime. The design is researched in docs/research/interop-usecases.md + docs/design/cli-bindings.md — FOLLOW the contract (results->pandas, f2->numpy, nanobind not PyCUDA, GPU-only wheel; the 4 spike risks: VRAM ownership/deleter, stream sync, fp64 truncation, column-major layout).',
  'GATE (no AT2): a pytest (tests/python/ or similar) that (1) loads the committed golden_fit0 f2 fixture via the Python read_f2_dir-equivalent (or a built f2 dir), runs qpadm THROUGH the Python API, and asserts the weights/p match the committed golden_fit0.json within the existing tier; (2) the f2 numpy array round-trips / matches the C++ f2 (shape + values); (3) ideally a qpwave + a qpadm_search (rotate) call returns sensible structured results. Reproduce a GOLDEN through Python — that is the proof the binding is correct.',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER run git checkout -- . or git clean -fd (they DESTROYED work before). On ANY failure, PRESERVE the attempt with: ' + STASH + ' "wip:py1-FAILED-<short-reason>" (this stashes tracked+untracked, restorable via git stash list/pop, reviewable later) and HALT. Do NOT revert/destroy. The fixer likewise must NOT destroy; if build-repair hits a NON-trivial error it STOPS + reports (the verdict stashes).',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data / committed goldens only; no synthetic. Box ' + SSH + '; nvcc -> ' + PATHENV + '. The box has python3 (used it for the atlas); pip-install pytest / python3-dev / nanobind deps if missing. nothing builds locally. NAMING-STYLE-STANDARD + the §4 layering (bindings are a separate subtree like app/, must not leak CUDA headers into a pure-host TU where the arch-grep forbids it).',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Design the bindings (verify)')
const design = await tryAgent([
  'You are a senior engineer DESIGNING the M(py-1) nanobind bindings (verify-before-implement; NO code changes). READ: docs/research/interop-usecases.md + docs/design/cli-bindings.md (§6 bindings contract, the M(py-1)/M(py-2) split, the spike risks), include/steppe/qpadm.hpp (the run_qpadm/run_qpwave/run_qpadm_search signatures + QpAdmResult/QpWave result/QpAdmModel/QpAdmOptions types — what to surface to Python), the f2-dir reader (read_f2_dir + DeviceF2Blocks/F2BlockTensor — the f2 array shape/layout), the CMake (the STEPPE_BUILD_PYTHON option current state — grep it; how CLI11/CPM deps are pulled), src/app/cmd_qpadm.cpp (the load->resolve->run->result flow to mirror in a binding).', STD, '',
  'PRODUCE the design: (1) the module layout (a bindings/ or python/steppe subtree, the nanobind target, how STEPPE_BUILD_PYTHON wires it, nanobind via CPM); (2) the API surface — the exact Python functions/classes (qpadm/qpwave/qpadm_search taking an f2-dir path + pops + options, returning pandas-friendly structured results; read_f2 returning numpy + pops) and how the C++ result types map to Python (the 4 spike risks: f2 as fp64 numpy column-major, no VRAM exposed in M(py-1) [results are host-side], etc.); (3) the pytest gate (reproduce golden_fit0 through Python); (4) the GPU-only + layering constraints. Cite file:line. Return the design (do NOT implement).',
].join('\n'), { label: 'design:py1', phase: 'Design the bindings (verify)' })
if (design === null) { log('--- design died — HALT (nothing to stash)'); return { halted: true } }

phase('Implement the nanobind module + build')
const fixer = await tryAgent([
  'You are implementing the M(py-1) nanobind bindings per this design:\n<<<\n' + design + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (never destroy — if you must reset, git stash). Work from HEAD.', STD, '',
  'IMPLEMENT the nanobind module + the API surface + the pytest gate + the CMake wiring (STEPPE_BUILD_PYTHON=ON builds it, nanobind via CPM). Reuse the existing run_*/read_f2_dir entries (NO duplicated compute). ' + RSYNC + ' then ' + BUILD + '. Build the module + run pytest. SANITY: pytest reproduces golden_fit0 through Python (weights/p within tier) + the f2 numpy round-trips. If the box lacks python3-dev/pytest/nanobind, install them (pip/apt). Report every file added/changed, the API surface, the pytest result, and the build. Do NOT commit. If build-repair hits a NON-trivial blocker (a real design/build problem), STOP + report (do NOT thrash or destroy) — the verdict will stash.',
].join('\n'), { label: 'implement:py1', phase: 'Implement the nanobind module + build' })
if (fixer === null) { log('--- fixer died — HALT (work in tree; verdict will assess)'); return { halted: true } }

phase('Verify golden-through-Python + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','golden_through_python','f2_numpy_ok','no_duplication','gpu_only','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the nanobind module builds (STEPPE_BUILD_PYTHON=ON), pytest reproduces golden_fit0 qpadm THROUGH the Python API within tier, the f2 numpy array matches the C++ f2, no duplicated compute, GPU-only, existing ctest still green' },
    fail_severity: { type: 'string', description: 'if pass=false: "minor" (salvageable/understood — orchestrator may stash + move to the next standalone) or "bad" (broke something / risky / ambiguous — orchestrator should HALT + defer to the user). empty if pass' },
    golden_through_python: { type: 'boolean', description: 'pytest reproduced golden_fit0 qpadm via the Python API within tier' },
    f2_numpy_ok: { type: 'boolean', description: 'read_f2 returns a numpy array matching the C++ f2 (shape + values)' },
    no_duplication: { type: 'boolean', description: 'bindings call the existing run_*/read_f2_dir; no copied compute' },
    gpu_only: { type: 'boolean', description: 'no CPU-runtime path introduced' },
    build_clean: { type: 'boolean' },
    commit_hash: { type: 'string', description: 'the commit hash on PASS; empty on fail' },
    stash_ref: { type: 'string', description: 'on FAIL: the git stash message/ref where the attempt was preserved; empty on pass' },
    note: { type: 'string', description: 'the API surface delivered + the pytest result; on FAIL the exact blocker + why minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for M(py-1) bindings (adversarial). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff --stat + review — the nanobind module reuses run_*/read_f2_dir (no dup compute), GPU-only, layering respected. (2) ' + BUILD + ' — the module builds under STEPPE_BUILD_PYTHON=ON, pytest reproduces golden_fit0 through Python within tier, the f2 numpy matches, and the existing ctest stays green. (3) confirm no AT2, no synthetic. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new/changed bindings + CMake + pytest + doc files (NEVER git add dot; never aadr/ atlas_results/ build_run.sh handoff-*.md), commit with a ROADMAP §6 message (M(py-1): nanobind Python bindings — qpadm/qpwave/qpadm_search -> pandas-friendly + f2 -> numpy, GPU-only, golden_fit0-gated through Python) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (RESUME/TODO: M(py-1) done; M(py-2) wheel next). ',
  'ON FAIL: DO NOT git checkout/clean. PRESERVE the attempt: ' + STASH + ' "wip:py1-FAILED" (stashes tracked+untracked, reviewable). Capture the stash ref. Classify fail_severity ("minor" if the bindings are close/the blocker is understood + isolated; "bad" if it broke the build/ctest, is risky, or needs a design decision). Return the structured verdict (pass=false, fail_severity, stash_ref, the exact blocker).',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:py1', phase: 'Verify golden-through-Python + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ M(py-1) bindings ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- M(py-1) FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
