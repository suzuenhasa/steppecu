export const meta = {
  name: 'build-cli-m0-m1',
  description: 'STEP 2 productization, first build: M(cli-0) the steppe CLI scaffold + M(cli-1) `steppe qpadm` over an existing f2_blocks dir — per the APPROVED design docs/design/cli-bindings.md (defaults accepted). Two units, SEQUENTIAL, commit-between, HALT-on-fail. M(cli-0): a src/app CLI (CLI11) executable `steppe` behind STEPPE_BUILD_CLI=ON — --version/--help, subcommand stubs (qpadm/qpwave/qpadm-rotate/extract-f2), a minimal CUDA-FREE ConfigBuilder/RunConfig/CliArgs in core (precedence compiled<TOML<env STEPPE_*<CLI; validating build()), Status->exit-code map, GPU-only (NO --device cpu — CPU is the dev/test oracle only, memory cpu-is-test-only). NO compute. M(cli-1): `steppe qpadm --f2-dir DIR --target T --left a,b --right r0,..` — define the minimal f2-dir READ format (the STPF2BK1 .bin + pops.txt name<->index sidecar + meta.json; READ side only, the writer is M(cli-4)), resolve pop NAMES->indices via pops.txt, build_resources(DeviceConfig) -> upload_f2_blocks_to_device -> run_qpadm (GPU), emit tidy CSV (default) + JSON. RULES: GPU is the deliverable; app is a PLAIN CXX target (CUDA PRIVATE to steppe_device — the arch-grep must hold, app must NOT include a CUDA header); follow the cli-bindings.md contract + existing standards/NAMING-STYLE-STANDARD; NO SYNTHETIC DATA + follow the existing real-AADR golden test pattern (the CLI test builds an f2-dir from the COMMITTED real golden fixture + its pop list and asserts the CLI output == golden_fit0/golden_fit1_NRBIG — NO synthetic, NO smoke test); verify CLI11/CUDA-13 API claims vs docs; commit between; be aware of stale code (re-verify vs HEAD); core dumps cleared per build.',
  phases: [ { title: 'M(cli-0) scaffold' }, { title: 'M(cli-1) qpadm-on-dir' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
// build with the CLI enabled; default ctest (real-AADR goldens) must stay green + the new CLI test passes.
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const ARCHGREP = SSH + " 'cd /workspace/steppe && echo ARCH-GREP: app must not include CUDA; && ! grep -rnE \"cuda_runtime|cublas_v2|cusolver|<cuda\" src/app 2>/dev/null && echo APP-CUDA-FREE-OK || echo APP-CUDA-LEAK'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) C++20 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ a714bbc. The fit BACKEND is FINISHED + golden-gated. This is STEP 2 (productization): the CLI. THE CONTRACT is docs/design/cli-bindings.md (READ IT — the command set, the API mapping, the f2-dir format, the layering, the milestone defs M(cli-0)/M(cli-1)). Standards: docs/architecture.md (§4 layering [app/bindings->api->core->device; CUDA PRIVATE to steppe_device; core CUDA-free; io leaf], §9 ConfigBuilder/RunConfig precedence, §10 Status/no-printf-in-lib, §15 deps CLI11 2.4.x), docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md.',
  'GPU-ONLY (memory cpu-is-test-only): steppe is a GPU product; the CLI targets the GPU. NO --device cpu supported mode (the CpuBackend is the dev/test parity oracle ONLY). The CLI calls the GPU path: build_resources(DeviceConfig) -> upload_f2_blocks_to_device / the resident f2 -> run_qpadm. A no-GPU box surfaces a clear "no CUDA device" error.',
  'LAYERING (compiler-enforced, MUST hold): src/app is a PLAIN C++20 CXX target (NOT a CUDA target, NO LANGUAGES CUDA), links steppe::core/io/api/device, and reaches the GPU ONLY through the CUDA-free seams (resources.hpp build_resources; qpadm.hpp run_qpadm; device_f2_blocks.hpp upload_f2_blocks_to_device; f2_disk_format.hpp). app must NOT #include any CUDA header (cuda_runtime/cublas_v2/cusolverDn) — that is a hard build failure + the arch-grep gate. No printf/cout in library code (the app owns stdout; core uses the log sink). Domain Status (RankDeficient/NonSpd/ChisqUndefined) -> the CLI emits a status column + exits 0 (record-and-continue); only faults (InvalidConfig/DeviceOom/file errors) -> nonzero exit.',
  'NO SYNTHETIC DATA / existing real-golden test pattern (do NOT deviate, no smoke tests): the CLI test builds an f2-dir from the COMMITTED REAL golden fixture (tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin + the golden_fit0 pop list) — i.e. write the .bin + a pops.txt + meta.json into a temp f2-dir — then runs the built `steppe qpadm` binary and asserts the parsed CSV/JSON output == golden_fit0.json (weights/se/z/p/chisq/dof/rankdrop/popdrop) within the same tiers the existing tests use; add golden_fit1_NRBIG as the large case. REAL data only. Follow the test_qpadm_parity.cu golden-assert pattern.',
  'GATE: Release build clean (warnings-as-errors) with -DSTEPPE_BUILD_CLI=ON; the existing default ctest STILL green vs the real-AADR goldens (no regression — the CLI is additive); the NEW CLI test(s) pass (M(cli-1) reproduces golden_fit0 + NRBIG THROUGH the CLI on the GPU); the app-is-CUDA-free arch-grep holds; CLI11 wired (CPM per architecture §15). RELEASE/NDEBUG/-Werror; STEPPE_ASSERT-only params -> [[maybe_unused]]. Core dumps cleared per build.',
  'BOX = box5090 (2x RTX 5090, sm_120, CUDA 13). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . RELEASE only. NOTHING builds locally. Commit between M(cli-0) and M(cli-1) (the verdict commits each).',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD at item start (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+ctest (' + BUILD + '); arch-grep (' + ARCHGREP + '). Do NOT commit (the verdict commits). NO synthetic data. Wire new targets/tests into the CMake (src/app/CMakeLists.txt behind STEPPE_BUILD_CLI; the CLI test into tests/CMakeLists.txt).'

const ITEMS = [
  { id: 'M(cli-0)', title: 'M(cli-0) scaffold', research: true,
    fix: 'Build the steppe CLI SCAFFOLD per cli-bindings.md §4/§6 + architecture §9. (1) src/app/ + src/app/CMakeLists.txt: a `steppe` executable (CLI11 via CPM "gh:CLIUtils/CLI11@2.4.2" per architecture §379/§15), PLAIN CXX target behind option STEPPE_BUILD_CLI (already an OFF stub in cmake/SteppeOptions.cmake:22-23 — wire add_subdirectory(src/app) behind it in the top CMakeLists). --version/--help; subcommand STUBS for qpadm/qpwave/qpadm-rotate/extract-f2 (parse args, print "not yet implemented" + exit, EXCEPT qpadm which M(cli-1) implements). (2) a minimal CUDA-FREE ConfigBuilder/RunConfig/CliArgs in core (architecture §9: precedence compiled<TOML<env STEPPE_*<CLI; a validating build() that maps onto DeviceConfig/QpAdmOptions/PopSelection/FilterConfig and fail-fasts on bad config; the VRAM/precision checks route through the CUDA-free build_resources/BackendCapabilities seam, NEVER a direct CUDA call). GPU-only: --device accepts device ordinals (e.g. 0 / 0,1 / auto), NO cpu. (3) Status->exit-code map (Ok/domain=0 for the record-and-continue model-level outcomes at the search layer; InvalidConfig/DeviceOom/file-error=nonzero). NO compute in this milestone.',
    gate: 'Release build clean with -DSTEPPE_BUILD_CLI=ON; the `steppe` binary builds + --version/--help work; a config unit test (precedence order + build() rejects bad config) following the existing test_config/unit pattern passes; the app-is-CUDA-free arch-grep holds; existing default ctest still green.' },
  { id: 'M(cli-1)', title: 'M(cli-1) qpadm-on-dir', research: false,
    fix: 'Implement `steppe qpadm --f2-dir DIR --target T --left a,b[,c] --right r0,r1,..[--rank R --fudge F --jackknife 0|1|2 --rank-alpha A --precision emu40|fp64 --device 0 --out FILE --format csv|json]` per cli-bindings.md §4. (1) Define + implement the minimal f2-dir READ format: <dir>/f2.bin (the existing STPF2BK1 numeric blob, f2_disk_format.hpp) + <dir>/pops.txt (the P pop labels in P-axis index order = the name<->index map) + <dir>/meta.json (provenance). READ side only (the extract-f2 WRITER is M(cli-4)). (2) a pop-name resolver: map --target/--left/--right NAMES -> indices via pops.txt (unknown name => Status::InvalidConfig fail-fast with the name). (3) load the f2 (host F2BlockTensor from the .bin) -> build_resources(DeviceConfig from the CLI) -> upload_f2_blocks_to_device (GPU) -> run_qpadm(DeviceF2Blocks, QpAdmModel{target,left,right}, QpAdmOptions, Resources). (4) emit tidy CSV (default; columns mirroring the committed golden CSVs so AT2 scripts work) + --format json (mirroring golden_fit0.json schema); --out FILE or stdout; emit the status column. GPU path is the deliverable. Follow NAMING-STYLE-STANDARD + the §10 no-printf-in-lib (printing is in app only).',
    gate: 'a CLI test (NEW tests/cli/test_cli_qpadm or extend tests/CMakeLists.txt) that: writes the COMMITTED real golden fixture (f2_fit0_9pop.bin + golden_fit0 pop list) into a temp f2-dir (f2.bin+pops.txt+meta.json), runs the built `steppe qpadm` for the golden_fit0 model (target=England_BellBeaker, left=Czechia_EBA_CordedWare,Turkey_N, right=the 6 outgroups), parses the CSV+JSON output, and asserts weights/se/z/p/chisq/dof/rankdrop/popdrop == golden_fit0.json within the existing tiers — RUNNING ON THE GPU; + the golden_fit1_NRBIG large case. NO synthetic. Build clean; existing ctest still green; arch-grep holds.' },
]

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item','pass','diff_real','gpu_only','layering_ok','follows_contract','no_synthetic','goldens_green','new_test_passes','build_clean','commit_hash','note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if: real CLI code per cli-bindings.md + GPU-only (no --device cpu) + app is CUDA-free (arch-grep holds) + follows the contract/standards + NO synthetic (CLI test uses the real golden fixture) + existing default ctest still green vs real goldens + the new CLI/config test passes (M(cli-1) reproduces golden_fit0+NRBIG through the CLI on the GPU) + Release build clean (-Werror, -DSTEPPE_BUILD_CLI=ON)' },
    diff_real: { type: 'boolean' }, gpu_only: { type: 'boolean', description: 'no --device cpu / no CPU runtime path exposed' },
    layering_ok: { type: 'boolean', description: 'src/app is a plain CXX target, includes NO CUDA header (arch-grep APP-CUDA-FREE-OK)' },
    follows_contract: { type: 'boolean', description: 'matches docs/design/cli-bindings.md + NAMING-STYLE-STANDARD' },
    no_synthetic: { type: 'boolean' }, goldens_green: { type: 'boolean', description: 'existing default ctest green vs real-AADR goldens (no regression)' },
    new_test_passes: { type: 'boolean', description: 'the new config unit test (M(cli-0)) / CLI golden test (M(cli-1)) passes' },
    build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed + the gate result + the CLI test evidence; for FAIL exactly what blocked it' },
  },
}

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  let research = ''
  if (it.research) {
    research = await tryAgent(['You are researching the BEST approach for steppe ' + it.id + ' BEFORE editing. READ-ONLY: read docs/design/cli-bindings.md (the contract), architecture.md §4/§7/§9/§15 (layering, the CMake target wiring, CLI11/CPM, the ConfigBuilder design), cmake/SteppeOptions.cmake (the STEPPE_BUILD_CLI stub) + the top CMakeLists + src/*/CMakeLists.txt (how targets are wired), include/steppe/config.hpp (DeviceConfig/etc the ConfigBuilder maps to). Verify CLI11 2.4.x usage + the CPM pattern vs the CLI11 docs (web-search if needed). Do NOT edit/build. Return the concrete approach (the CMake wiring for a plain-CXX app target behind STEPPE_BUILD_CLI, the ConfigBuilder shape + where build() lives [core, host-testable], the subcommand skeleton) so the fixer executes without re-deriving.', STD].join('\n'), { label: 'research:' + it.id, phase: it.title }) || ''
  }
  const fixer = ['You are a senior C++ engineer building ONE step-2 CLI milestone for steppe. Do NOT commit (the verdict commits). FIRST clean HEAD: ' + CLEAN + ' . FIRST READ the contract docs/design/cli-bindings.md.', STD, '', DEVLOOP, '',
    'ITEM ' + it.id + ': ' + it.fix, '', 'THE GATE: ' + it.gate,
    research ? ('\nRESEARCH (use this):\n<<<\n' + String(research).slice(0, 3500) + '\n>>>') : '',
    '', 'Build it per the contract + standards. app = PLAIN CXX, NO CUDA header (arch-grep must pass), GPU-only (no --device cpu). Verify CLI11/CUDA-13 API claims vs docs (cite). Build + ctest + arch-grep until clean+green. Report: every file added/changed; the CMake wiring; the new test + what it asserts (real golden, no synthetic); the FULL build + ctest + arch-grep output. Do NOT commit. If you cannot reach green, report exactly what blocked it — do NOT fake it.',
  ].join('\n')
  const fix = await tryAgent(fixer, { label: 'fix:' + it.id, phase: it.title })
  if (fix === null) { ledger.push({ item: it.id, pass: false, note: 'fixer died' }); log('--- ' + it.id + ' fixer died — HALT'); break }

  await tryAgent(['You are BUILD-REPAIR for ' + it.id + '. The fixer accumulated edits (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON), patching ONLY trivial -Werror (unused param/var; [[maybe_unused]] for STEPPE_ASSERT-only params; a missing include). DO: ' + RSYNC + ' then ' + BUILD + ' . If a trivial error fires, fix minimally + rebuild, LOOP up to 4x. Do NOT change logic, do NOT revert. If it fails for a NON-trivial reason (a design/API error, a CLI11/CPM fetch failure, a layering violation), STOP + report. Report final build + the trivial patches.', STD].join('\n'), { label: 'repair:' + it.id, phase: it.title })

  const verdict = await tryAgent(['You are the INDEPENDENT VERDICT for ' + it.id + ' of steppe (adversarial). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
    'THE ITEM: ' + it.fix, '', 'THE GATE: ' + it.gate,
    '', 'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — confirm real CLI code per cli-bindings.md (not stub-where-it-should-compute), GPU-only (no --device cpu), app CUDA-free, follows the contract/standards, NO synthetic (CLI test uses the real golden fixture). (2) RE-RUN: ' + BUILD + ' ; ' + ARCHGREP + ' . (3) PASS only if ALL: real + GPU-only + layering_ok (arch-grep APP-CUDA-FREE-OK) + contract-conformant; existing default ctest still green vs real goldens; the new test passes (M(cli-1): golden_fit0 + NRBIG reproduced THROUGH the CLI on the GPU); Release build clean -DSTEPPE_BUILD_CLI=ON. ',
    'ON PASS: cd ' + R + ' && git add ONLY this item changed/new files (src/app, the CMake, the new test, include/core config files; NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (the CLI milestone + the gate result; cite the CLI11/CUDA-doc if relevant) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.',
    'ON FAIL: ' + CLEAN + ' (leave repo green) + report exactly what blocked it. Return the structured verdict.',
  ].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:' + it.id, phase: it.title })
  if (verdict === null) { ledger.push({ item: it.id, pass: false, note: 'verdict died' }); log('--- ' + it.id + ' verdict died — HALT'); break }
  ledger.push(verdict)
  if (verdict.pass) log('+++ ' + it.id + ' committed ' + verdict.commit_hash + ' — ' + verdict.note)
  else { log('--- ' + it.id + ' FAILED (' + verdict.note + ') — reverted; HALT'); break }
}
const passed = ledger.filter(x => x.pass).length
log('build-cli M(cli-0)+M(cli-1): ' + passed + '/' + ITEMS.length + ' committed')
return { ledger, passed }
