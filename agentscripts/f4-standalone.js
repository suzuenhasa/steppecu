export const meta = {
  name: 'f4-standalone',
  description: 'STEP 3 stat #1: standalone f4(A,B;C,D) — the cheapest standalone (per docs/research/standalone-fstats.md): it is a linear combination of f2 entries [f4(A,B;C,D)=(f2(A,D)+f2(B,C)-f2(A,C)-f2(B,D))/2] per block + the EXISTING block-jackknife for the SE; reuse assemble_f4 (backend.hpp:604,618) + jackknife_cov (backend.hpp:634) + steppe::access (the shared f2-dir reader + PopResolver from M(py-1)) + the CLI flag helpers + the result emitter + the M(py-1) nanobind module. Its golden ALREADY EXISTS in-tree: tests/reference/goldens/at2/csv/golden_fit0_f4.csv (pop1,pop2,pop3,pop4,est,se,z,p over the 9-pop fit0 set). DELIVER: (1) a run_f4 core entry (single quartet + BATCHED over a quartet list / all-quartets-over-the-f2-dir, reusing the f2 contraction + jackknife — NO dup compute); (2) a `steppe f4` CLI subcommand (reuse add_f2_dir_flag/add_output_flags/add_common_flags; --pops A,B,C,D for one, or all-quartets over the dir; csv/tsv/json: pop1..pop4,est,se,z,p); (3) a Python binding steppe.f4(...) on the M(py-1) module (-> pandas-friendly); (4) GOLDEN-GATE reproducing golden_fit0_f4.csv (est rtol ~1e-6, se/z/p at the existing loose tier) via a new cli_f4 ctest + a pytest. NO AT2 (golden_fit0_f4 is the gate). Discipline: design/verify -> implement -> verify. FAIL-PROTOCOL (user-mandated): NEVER git checkout/clean — on failure git stash push -u the attempt (reviewable) + HALT; the verdict classifies severity (minor=move-on / bad=defer). SINGLE-GPU; REAL data; commit-on-green.',
  phases: [ { title: 'Design f4 (verify reuse + golden)' }, { title: 'Implement run_f4 + CLI + binding + build' }, { title: 'Verify golden_fit0_f4 + commit-or-stash' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -30; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 93b3217. The CLI is complete (extract-f2/qpadm/qpwave/qpadm-rotate). M(py-1) nanobind bindings landed (steppe._core + bindings/steppe; the f2-dir reader + PopResolver are now a shared steppe::access host lib reused by CLI + bindings). The qpAdm/qpWave fit engine computes the f4 contraction INTERNALLY (assemble_f4) + the weighted block-jackknife (jackknife_cov).',
  'GOAL: standalone f4(A,B;C,D). MATH: f4(A,B;C,D) = mean_SNP (a-b)(c-d) = (f2(A,D)+f2(B,C)-f2(A,C)-f2(B,D))/2, computed PER jackknife block from the f2 tensor, then block-jackknifed for the SE (z=est/se, p the 2-sided normal). This is a thin linear combination of the f2 blocks + the EXISTING jackknife — reuse, do NOT reimplement the contraction or the SE. Per docs/research/standalone-fstats.md the only genuinely new code is the quartet gather + the linear combo.',
  'REUSE (verify file:line, no dup): assemble_f4 / the f4 identity (backend.hpp:597-602,604,618); jackknife_cov (backend.hpp:634); steppe::access (the shared f2-dir reader + PopResolver from M(py-1)); the CLI flag helpers (add_f2_dir_flag/add_output_flags/add_common_flags in cli_parse.cpp) + the cmd_qpadm.cpp load->resolve->run->emit pattern; the result emitter (result_emit.cpp helpers); the M(py-1) nanobind module (add steppe.f4).',
  'GOLDEN-GATE (no AT2): tests/reference/goldens/at2/csv/golden_fit0_f4.csv (header pop1,pop2,pop3,pop4,est,se,z,p,weight; the f4 quartets over the 9-pop golden_fit0 set, the committed f2_fit0_9pop fixture). Reproduce est at rtol ~1e-6 + se/z/p at the existing loose tier. A new cli_f4 ctest (mirror test_cli_qpadm) + a pytest (steppe.f4 via Python). Single-quartet AND batched (all-quartets-over-the-dir, which is what the golden is).',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER git checkout -- . / git clean -fd (they destroyed work before). On ANY failure PRESERVE via ' + STASH + ' "wip:f4-FAILED-<reason>" (tracked+untracked, restorable) + HALT. The fixer must NOT destroy; a NON-trivial blocker -> STOP + report (the verdict stashes).',
  'SINGLE-GPU (--device 0; multi-gpu PARKED). RELEASE build. REAL data / committed goldens only; no synthetic; no AT2. Box ' + SSH + '; nvcc -> ' + PATHENV + '. NAMING-STYLE-STANDARD + §4 layering. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Design f4 (verify reuse + golden)')
const design = await tryAgent([
  'You are designing the standalone f4 (verify-before-implement; NO code changes). READ: backend.hpp (assemble_f4 + the f4 identity + jackknife_cov — the exact signatures + what they need), include/steppe/qpadm.hpp (the result/option types + where a run_f4 entry should live), src/core/qpadm/* (how assemble_f4 is invoked in the fit, whether a thin run_f4 can reuse it or needs a small f4-from-f2 helper + the jackknife), the steppe::access lib (the shared f2-dir reader + PopResolver from M(py-1)), src/app/cmd_qpadm.cpp + cli_parse.cpp (the CLI pattern + flag helpers), src/app/result_emit.cpp (the emitter helpers), the M(py-1) bindings module (how to add steppe.f4), tests/reference/goldens/at2/csv/golden_fit0_f4.csv (the exact columns/quartets to reproduce) + test_cli_qpadm.cpp (the golden e2e pattern).', STD, '',
  'PRODUCE the design: (1) the run_f4 entry (signature: f2 + a quartet or quartet-list -> est/se/z/p per quartet; reuse assemble_f4 + jackknife_cov, or the minimal f4-from-f2 + the existing jackknife — name exactly what to reuse vs the small new gather/combo); (2) the CLI `f4` subcommand (single --pops A,B,C,D + the batched all-quartets-over-the-dir mode that matches the golden; reuse the flag helpers + emitter); (3) the steppe.f4 Python binding; (4) the golden-gate (cli_f4 + pytest, the tiers). Confirm NO duplicated contraction/SE. Cite file:line. Return the design (do NOT implement).',
].join('\n'), { label: 'design:f4', phase: 'Design f4 (verify reuse + golden)' })
if (design === null) { log('--- design died — HALT'); return { halted: true } }

phase('Implement run_f4 + CLI + binding + build')
const fixer = await tryAgent([
  'You are implementing the standalone f4 per this design:\n<<<\n' + design + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (never destroy; git stash if you must reset). Work from HEAD.', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT run_f4 (reuse assemble_f4/jackknife_cov + steppe::access — no dup), the `steppe f4` CLI subcommand (reuse the flag helpers + emitter; single + batched all-quartets), the steppe.f4 Python binding (on the M(py-1) module), and the gate (cli_f4 ctest + a pytest) reproducing golden_fit0_f4.csv. Build + full STEPPE_THOROUGH ctest. SANITY: cli_f4 reproduces golden_fit0_f4 (est rtol 1e-6); `steppe f4 --f2-dir <fit0 dir> --pops England_BellBeaker,Czechia_EBA_CordedWare,Han,Iran_GanjDareh_N` matches the golden row; the existing goldens/cli/pytest stay green. Report every file added/changed, the reuse, the gate result, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report (do NOT thrash/destroy).',
].join('\n'), { label: 'implement:f4', phase: 'Implement run_f4 + CLI + binding + build' })
if (fixer === null) { log('--- fixer died — HALT (work in tree)'); return { halted: true } }
await tryAgent(['BUILD-REPAIR for the f4 standalone. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON) + green ctest, patching only trivial -Werror / CMake wiring of the new cmd_f4 + cli_f4. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement run_f4 + CLI + binding + build' })

phase('Verify golden_fit0_f4 + commit-or-stash')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','golden_f4_reproduced','python_f4_ok','no_duplication','goldens_green','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if `steppe f4` reproduces golden_fit0_f4.csv (est rtol ~1e-6, se/z/p at tier) via cli_f4, steppe.f4 works via Python (pytest), no duplicated contraction/SE, full STEPPE_THOROUGH ctest green, Release build clean, single-GPU, no synthetic, no AT2' },
    fail_severity: { type: 'string', description: 'if pass=false: "minor" (salvageable/understood -> orchestrator stashes + moves to the next stat) or "bad" (broke build/ctest, risky, needs a decision -> orchestrator HALTs + defers). empty if pass' },
    golden_f4_reproduced: { type: 'boolean' }, python_f4_ok: { type: 'boolean', description: 'steppe.f4 reproduces the golden via Python' },
    no_duplication: { type: 'boolean', description: 'reused assemble_f4/jackknife_cov/access; no copied contraction/SE' },
    goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string', description: 'hash on PASS; empty on fail' },
    stash_ref: { type: 'string', description: 'on FAIL: the git stash message/ref; empty on pass' },
    note: { type: 'string', description: 'the f4 est-vs-golden + the reuse; on FAIL the blocker + why minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the standalone f4 (adversarial). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff --stat + review — run_f4 reuses assemble_f4/jackknife_cov/access (NO dup contraction/SE), the f4 CLI reuses the flag helpers/emitter, the binding rides the M(py-1) module. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest green incl the new cli_f4 + the existing goldens; cli_f4 reproduces golden_fit0_f4.csv (est rtol ~1e-6); the pytest steppe.f4 passes. (3) no AT2, no synthetic, GPU-only. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new/changed source/test/cmake/binding/doc files (NEVER git add dot; never aadr/ atlas_results/ build_run.sh handoff-*.md), commit with a ROADMAP §6 message (STEP3 f4: standalone f4(A,B;C,D) CLI `steppe f4` + steppe.f4 binding — reuses assemble_f4 + the block-jackknife [no dup], golden_fit0_f4-gated) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (RESUME/TODO: f4 done; next f3). ',
  'ON FAIL: DO NOT git checkout/clean. PRESERVE: ' + STASH + ' "wip:f4-FAILED" (capture the ref). Classify fail_severity (minor if f4 is close/the blocker is understood + isolated; bad if it broke the build/ctest or needs a design decision). Return the structured verdict (pass=false, fail_severity, stash_ref, the blocker).',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:f4', phase: 'Verify golden_fit0_f4 + commit-or-stash' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ f4 ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- f4 FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { verdict }
