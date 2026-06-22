export const meta = {
  name: 'wire-qpadm-rotate-and-olalde',
  description: 'Wire the qpadm-rotate CLI to the existing batched rotation engine (M(cli-3)), properly + golden-gated, THEN run the Olalde 2018 competing-sources rotation and compare to the published Table 6. NO AT2 runs needed: golden_rot.json (the existing 84-model AT2-parity rotation fixture) is the WIRING gate, and Olalde SI Table 6 (published) is the STUDY comparison. Discipline: design/verify -> implement -> verify(golden_rot through the CLI) -> only-then run Olalde. The engine run_qpadm_search (src/core/qpadm/model_search.cpp:253, ~2866 models/sec, batched on the GPU, f2 resident) is COMPLETE + golden-gated; the CLI subcommand qpadm-rotate is a scaffold no-op (src/app/cli_parse.cpp:216 run_not_yet_implemented). The CLI flags already parse (--target --pool --min-sources --max-sources --right --jackknife --format). WIRE (no dup): route the qpadm-rotate subcommand to: load the f2 dir (reuse the qpadm subcommand f2-dir loader), enumerate source-subsets of --pool of size [min_sources,max_sources] into QpAdmModels (target + each subset as left), call run_qpadm_search, emit the per-model table (target, left set, weights, p, chisq, feasible) in csv/tsv/json (reuse/extend the qpadm result formatter). GOLDEN-GATE: add a CLI e2e test that runs `steppe qpadm-rotate` reproducing golden_rot exactly (its target England_BellBeaker + the 8-source pool + 6 right + min/max-sources from golden_rot_generate.R) and matches golden_rot.json per-model at the existing tier (this is the parity gate; uses the EXISTING golden, NO AT2 re-run). Then RUN OLALDE: read Olalde 2018 SI Table 6 (the competing-sources qpAdm results, feasible models flagged) for the published reference; build the f2 dir (target England_BellBeaker; pool = a steppe source [Russia_Samara_EBA_Yamnaya] + rotated farmers [Turkey_N, Spain_EN, England_N, Czechia_EBA_CordedWare] + WHG [Luxembourg_Loschbour_Mesolithic, Russia_Karelia_Mesolithic]; right = Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana) on the v66 1240K; run qpadm-rotate over 1-3 source subsets; compare steppe FEASIBLE-model set (+ key weights) to Olalde Table 6; write docs/studies/olalde2018-rotation.md + commit. SINGLE-GPU (--device 0; multi-gpu parked); RELEASE; REAL v66; HALT-on-fail (do NOT run Olalde if the wiring gate fails); resumable on 529.',
  phases: [ { title: 'Design the wiring (verify)' }, { title: 'Implement qpadm-rotate + build' }, { title: 'Verify wiring vs golden_rot + commit' }, { title: 'Run Olalde rotation + compare + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 2321892. The batched rotation ENGINE run_qpadm_search (src/core/qpadm/model_search.cpp:253) is COMPLETE + golden-gated (test_qpadm_rotation.cu vs golden_rot.json, 84 models, ~2866/sec, f2 resident, batched on the GPU). The qpadm-rotate CLI subcommand is a SCAFFOLD no-op (src/app/cli_parse.cpp:216 run_not_yet_implemented); its flags already parse (--target --pool --min-sources --max-sources --right --fudge --jackknife --format --device --precision).',
  'NO AT2 RUNS ANYWHERE: the wiring is gated by the EXISTING golden_rot.json (no AT2 re-run); the Olalde study comparison is against the PUBLISHED Olalde 2018 SI Table 6. Do not invoke admixtools/R.',
  'WIRE PROPERLY (no duplication): the qpadm subcommand (src/app/cmd_qpadm* / cli_parse.cpp) already loads an f2 dir + runs run_qpadm + formats results. For qpadm-rotate: reuse that f2-dir loader + result formatter; build the model list by enumerating subsets of --pool of size [min_sources,max_sources] (each subset = the left/sources, with --target the target), call run_qpadm_search (the batched engine) ONCE over the whole list, emit a per-model table (target, left set, weights, p, chisq, dof, f4rank, feasible) in csv|tsv|json. Match how golden_rot defines a rotation (read scripts/golden_rot_generate.R + golden_rot.json: the target, the source pool, the right set, min/max sources, and how feasibility is decided).',
  'GOLDEN-GATE (the verify): add a CLI e2e test (pattern off tests/cli/test_cli_qpadm or test_cli_extract_qpadm) that runs `steppe qpadm-rotate` with golden_rot exact inputs and matches golden_rot.json per-model at the existing rotation tier. The existing test_qpadm_rotation (engine-level) + all goldens must stay green.',
  'DATA for Olalde: REAL v66 1240K at ' + PREFIX + '. SINGLE-GPU only (--device 0; multi-gpu PARKED). RELEASE -DSTEPPE_BUILD_CLI=ON; nothing builds locally; clear core dumps. NAMING-STYLE-STANDARD + the CLI/output schema (docs/design/cli-bindings.md). Binary ' + BIN + '.',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+thorough-ctest (' + BUILD + '). Do NOT commit (the verdict commits). NO synthetic. SINGLE-GPU. NO AT2.'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Design the wiring (verify)')
const design = await tryAgent([
  'You are a senior engineer DESIGNING the qpadm-rotate CLI wiring (verify-before-implement; NO code changes this phase). READ: src/app/cli_parse.cpp (the qpadm-rotate scaffold + flags + how qpadm/qpwave are wired), src/app/cmd_qpadm* (the f2-dir load + run_qpadm + the result formatter), include/steppe/qpadm.hpp (run_qpadm_search signature + QpAdmModel/QpAdmResult/QpAdmOptions/JackknifePolicy), src/core/qpadm/model_search.cpp (the engine), tests/reference/test_qpadm_rotation.cu + tests/reference/goldens/at2/golden_rot.json + scripts/golden_rot_generate.R (the rotation model space + feasibility + the per-model fields), tests/cli/test_cli_qpadm.cpp (the CLI e2e golden pattern).', STD, '',
  'PRODUCE the precise wiring spec: (1) exactly where/how to replace the qpadm-rotate scaffold (cli_parse.cpp:216) with the real path — the f2-dir loader to reuse, the subset-enumeration of --pool into QpAdmModels (target + each [min,max]-size subset as left), the run_qpadm_search call, and the per-model output formatter to reuse/extend (csv/tsv/json fields incl feasible). (2) The exact golden_rot inputs (target, pool, right, min/max sources) the new CLI e2e test must pass, and which golden_rot.json fields to compare at which tier. (3) Any type/seam details. (4) Confirm NO duplication of the fit/format logic. Cite file:line. Return the spec (do NOT implement).',
].join('\n'), { label: 'design:rotate-cli', phase: 'Design the wiring (verify)' })
if (design === null) { log('--- design died — HALT'); return { halted: true } }

phase('Implement qpadm-rotate + build')
const fixer = await tryAgent([
  'You are a senior engineer WIRING the qpadm-rotate CLI to the run_qpadm_search engine, per this design spec:\n<<<\n' + design + '\n>>>\n\nDo NOT commit. Start clean: ' + CLEAN + '.', STD, '', DEVLOOP, '',
  'IMPLEMENT the wiring exactly per the spec (reuse the qpadm f2-dir loader + result formatter; enumerate --pool subsets [min,max] -> QpAdmModels; ONE run_qpadm_search call; per-model table out). Add the CLI e2e golden test (steppe qpadm-rotate reproduces golden_rot.json). Build + full STEPPE_THOROUGH ctest. SANITY (no commit): run `steppe qpadm-rotate` on a small real set + confirm it emits a sensible per-model table. Report every file changed, the wiring, the test, and the FULL ctest. Do NOT commit.',
].join('\n'), { label: 'implement:rotate-cli', phase: 'Implement qpadm-rotate + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true } }
await tryAgent(['BUILD-REPAIR for the qpadm-rotate wiring. Accumulated edits (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON) + green ctest, patching only trivial -Werror / type mechanics. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement qpadm-rotate + build' })

phase('Verify wiring vs golden_rot + commit')
const WIRE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','golden_rot_through_cli','no_duplication','goldens_green','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if steppe qpadm-rotate CLI reproduces golden_rot.json per-model at the existing tier (the new CLI e2e test), engine test_qpadm_rotation + all goldens green, no duplicated fit/format logic, THOROUGH ctest green, Release build clean, single-GPU, NO AT2 used' },
    golden_rot_through_cli: { type: 'boolean', description: 'steppe qpadm-rotate reproduces golden_rot (the 84 models) through the CLI' },
    no_duplication: { type: 'boolean', description: 'reused the engine + the qpadm f2-loader/formatter; no copied fit/format logic' },
    goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the wiring + the CLI e2e golden gate result; for FAIL the exact issue' },
  },
}
const wire = await tryAgent([
  'You are the INDEPENDENT VERDICT for the qpadm-rotate CLI wiring (adversarial). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff — confirm the CLI routes to run_qpadm_search (NO duplicated fit/format), subset enumeration is correct, the per-model table is emitted, and the CLI e2e golden test is real (not weakened). (2) ' + BUILD + ' — THOROUGH ctest green incl. the new qpadm-rotate CLI e2e + test_qpadm_rotation + all goldens. (3) confirm `steppe qpadm-rotate` reproduces golden_rot.json through the CLI (the parity gate; NO AT2 re-run — golden_rot is the existing fixture). PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (M(cli-3): wire qpadm-rotate CLI to the batched run_qpadm_search engine [no dup] — golden-gated through the CLI vs golden_rot) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. ON FAIL: ' + CLEAN + ' and report the exact issue. Return the structured verdict.',
].join('\n'), { schema: WIRE_SCHEMA, label: 'verify:rotate-cli', phase: 'Verify wiring vs golden_rot + commit' })
if (wire === null || !wire.pass) { log('--- WIRING GATE FAILED (' + (wire ? wire.note : 'verdict died') + ') — HALT, NOT running Olalde'); return { halted: true, wire } }
log('+++ qpadm-rotate WIRED + golden-gated ' + wire.commit_hash)

phase('Run Olalde rotation + compare + commit')
const OLALDE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','table6','steppe_feasible','comparison','one_liner','commit_hash','note'],
  properties: {
    done: { type: 'boolean' },
    table6: { type: 'string', description: 'the Olalde 2018 SI Table 6 published competing-sources result read (which models feasible + key weights); if exact per-cell numbers unreadable, say so + use the feasible-model-set + headline' },
    steppe_feasible: { type: 'string', description: 'the steppe qpadm-rotate result on v66 1240K: which models feasible (target England_BellBeaker, the source-pool subsets), key weights/p' },
    comparison: { type: 'string', description: 'steppe feasible-model SET (+ key weights) vs Olalde Table 6 — match? the headline (e.g. the 2-way steppe+farmer feasible at ~50% steppe); differences + cause (proxy/version/SNP set)' },
    one_liner: { type: 'string', description: 'verified copy-paste ssh one-liner: extract-f2 the Olalde pool union + steppe qpadm-rotate' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'maxmiss used; #models enumerated; single-GPU/real confirmed; any caveat' },
  },
}
const olalde = await tryAgent([
  'You are reproducing the Olalde 2018 COMPETING-SOURCES ROTATION on steppe (the qpadm-rotate CLI is now wired + golden-gated, commit ' + wire.commit_hash + '). NO AT2 — compare to the PUBLISHED Olalde 2018 SI Table 6.', STD, '',
  'STEP 1: read Olalde 2018 (Nature 25738) SI Table 6 / section 8 (web/SI; try the supplementary tables/Excel) for the published competing-sources qpAdm result — which source combos are FEASIBLE for the British Beaker/BA target + the key weights (e.g. ~50% steppe in the 2-way). If exact per-cell p-values are unreadable from the image-PDF, say so and use the feasible-model SET + the headline numbers. STEP 2 on the box (single-GPU, REAL v66 1240K): build the f2 dir for the union {target England_BellBeaker; pool Russia_Samara_EBA_Yamnaya,Turkey_N,Spain_EN,England_N,Czechia_EBA_CordedWare,Luxembourg_Loschbour_Mesolithic,Russia_Karelia_Mesolithic; right Mbuti,Israel_Natufian,Iran_GanjDareh_N,Han,Papuan,Karitiana} (verify the labels exist in the .ind; map/substitute + note if any drifted; --maxmiss 0, relax to 0.5 if singleton collapse). STEP 3: `steppe qpadm-rotate --f2-dir <dir> --target England_BellBeaker --pool <the 7 pool pops> --right <the 6> --min-sources 1 --max-sources 3 --jackknife 1 --format csv` -> the per-model feasible set. STEP 4: COMPARE steppe feasible set + key weights to Olalde Table 6. Clean up the f2 dir + cores. Return the structured result incl a VERIFIED copy-paste ssh one-liner.',
  'THEN write ' + R + '/docs/studies/olalde2018-rotation.md (the rotation setup, the paper->v66 pop mapping incl any switches, the published Table 6 vs the steppe feasible-model set + weights, the one-liner) and commit ONLY that doc (+ nothing else; NEVER git add dot) with a ROADMAP §6 message (docs(studies): Olalde 2018 competing-sources rotation reproduced via steppe qpadm-rotate vs published Table 6) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. REAL data; single-GPU; NO AT2.',
].join('\n'), { schema: OLALDE_SCHEMA, label: 'run:olalde-rotation', phase: 'Run Olalde rotation + compare + commit' })
if (olalde === null) { log('--- Olalde run died — HALT (wiring already committed ' + wire.commit_hash + ')'); return { halted: true, wire } }
log('+++ Olalde rotation: ' + String(olalde.comparison).slice(0, 200));
return { wire, olalde }
