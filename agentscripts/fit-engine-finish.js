export const meta = {
  name: 'fit-engine-finish',
  description: 'FINISH the fit-engine backend (step 1 of backend-first): the 6 FINISH-NOW items from docs/design/fit-engine-finish-punchlist.md, each its OWN unit (research-if-needed -> fixer -> build-repair -> adversarial verdict -> commit), STRICTLY SEQUENTIAL (build mutex + overlapping files), commit between, HALT-on-fail (correctness; do NOT skip-continue). Order (dependency + momentum): F5 remove the dead opts.constrained flag; F3 add Status::ChisqUndefined + the dof<=0 guard (host AND CUDA path); F2 the M(fit-5) domain-outcome TEST (degenerate REAL-AADR models -> RankDeficient/NonSpdCovariance/ChisqUndefined VALUES, on BOTH backends); F6 widen the G1==G2 determinism memcmp to the full QpAdmResult; F4 pin a REAL AT2 qpwave() golden + test run_qpwave on both backends; F1 missing-block/NA handling — INVESTIGATE first (can an empty block occur in the maxmiss=0 resident f2?), then prove+document+gate OR implement NA-aware LOO on BOTH CpuBackend (parity) AND the GPU kernels. RULES: a CPU-for-parity version is NOT complete until PROPERLY adapted to the GPU; analyze the EXISTING code + follow existing standards/patterns (NAMING-STYLE-STANDARD + architecture §-rules + the existing kernel/test idioms); VERIFY CUDA/cuBLAS/cuSOLVER API claims vs the CUDA 13.x docs; NO SYNTHETIC DATA + do NOT deviate from the existing real-AADR golden test pattern (no synthetic fixtures, no smoke tests); be aware of STALE code — re-verify every punch-list claim against HEAD before editing; capable-path priority + 5090 fallback; core dumps cleared per build.',
  phases: [ { title: 'F5 dead-flag' }, { title: 'F3 ChisqUndefined' }, { title: 'F2 domain-outcome test' }, { title: 'F6 determinism widen' }, { title: 'F4 qpwave golden' }, { title: 'F1 NA/missing-block' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === DEFAULT CTEST (GPU-vs-golden, REAL AADR) === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -50; " + CORECLEAN + "'"
const THOROUGH = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && STEPPE_THOROUGH=1 ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -50; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) C++20 reimplementation of ADMIXTOOLS 2. Branch phase2-fit-engine == main @ b407ec0. The qpAdm fit engine S3-S8 is BUILT + golden-gated; this FINISHES the backend (docs/design/fit-engine-finish-punchlist.md). Standards: docs/architecture.md (§4 layering [io leaf; CUDA PRIVATE to steppe_device; core CUDA-free], §10 errors, §12 precision/PARITY, §13/§18 testing/DoD), docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md (§3.2 protected parity vocab = RENAME FORBIDDEN; member _ convention; kPascalCase constants).',
  'HARD RULE — CPU-then-GPU COMPLETENESS (memory build-the-real-gpu-implementation): if a fix has a CpuBackend (oracle) side AND a GPU side, doing the CPU version for parity is NOT complete until the GPU version is PROPERLY adapted. F3 (the dof<=0 guard) and F1 (NA handling) BOTH touch the host fit AND the CUDA path/kernels — finish BOTH. The CpuBackend is the oracle; the GPU is the deliverable.',
  'ANALYZE EXISTING CODE + FOLLOW STANDARDS: read the actual current code before editing (the punch-list may cite lines that drifted — VERIFY against HEAD). Follow the existing patterns (the Status enum shape, the QpAdmResult fields, the test_qpadm_parity/rotation golden-test structure, the kernel idioms, the §-cross-ref doc-comments). VERIFY any CUDA/cuBLAS/cuSOLVER/C++-stdlib API-behavior claim against the official CUDA 13.x docs (ToolSearch select:WebSearch,WebFetch) and cite it. You MAY research the problem first (a research sub-step) to pick the best approach.',
  'NO SYNTHETIC DATA — and do NOT deviate from the existing test patterns: every accuracy/parity check uses the REAL-AADR AT2 goldens (golden_fit0 9-pop, golden_fit1_NRBIG nr=39, golden_rot 84-model) under tests/reference/goldens/at2/. New tests MUST follow the EXISTING golden-test pattern (a pinned AT2 golden + the real f2 fixture, asserted to tier) — NO synthetic matrices, NO "smoke tests", NO made-up inputs. A DEGENERATE test model (F2) is built from REAL AADR pops arranged degenerate (e.g. a left set with a duplicated/collinear real source), NOT a synthetic matrix. R + admixtools 2.0.10 / R 4.3.3 are on the box to pin new real goldens (F4, and F1 if needed).',
  'GATE: the default ctest (39/39, GPU-vs-real-golden) AND STEPPE_THOROUGH qpadm (the CpuBackend oracle bit-identical) MUST stay green for every item (a finish item must not regress the existing goldens); plus the item\'s own new test/assertion passes. RELEASE/NDEBUG/-Werror; STEPPE_ASSERT-only params -> [[maybe_unused]]. Core dumps cleared per build.',
  'BOX = box5090 (2x RTX 5090, sm_120, CUDA 13, P2P DISABLED = fallback; capable-path PRO6000/P2P/CUDA13+ is priority where a fix touches that). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists, RELEASE only. NOTHING builds locally. Commit between items (the verdict commits each).',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD at item start (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+default-ctest (' + BUILD + '); thorough (' + THOROUGH + '). Do NOT commit (the verdict commits). NO synthetic data. Wire any NEW test into tests/CMakeLists.txt.'

const ITEMS = [
  { id: 'F5', title: 'F5 dead-flag', research: false,
    fix: 'REMOVE (or unambiguously mark not-implemented) the dead public `QpAdmOptions::constrained` field (include/steppe/qpadm.hpp:73, documented "reserved" but NEVER read in any solve path — it is NOT AT2 non-negative-weights; the only "constrained" in the solve is the unconditional Σw=1 equality constraint, unrelated). DECISION (per the user): REMOVE the dead field now (AT2 non-negative constrained-weights is a step-3 feature, not a finish item). Verify (grep) no code reads it before removing; if any test/struct-init sets it, drop that too. This is API hygiene before the public field ships in bindings. Host-only, behavior-neutral.',
    gate: 'build clean + default ctest 39/39 + thorough green (pure removal of an unread field — goldens unchanged).' },
  { id: 'F3', title: 'F3 ChisqUndefined', research: false,
    fix: 'Add `Status::ChisqUndefined` to include/steppe/error.hpp (spec §10 / architecture.md:676 names STEPPE_ERR_CHISQ_UNDEFINED "dof<=0 or chi^2 not computable"; currently absent so a dof<=0 model returns NaN `p` with status Ok — a consumer trap). Then add a `dof<=0 => Status::ChisqUndefined` guard: in the HOST path (qpadm_fit.cpp where pchisq_upper(chisq,dof) is called ~:98) AND the CUDA assemble/result path (cuda_backend.cu assemble_result) AND the model-batched path so the GPU returns it too (CPU-then-GPU completeness). Follow the existing Status taxonomy + the per-model status plumbing. This is an enum add (§16 deliberate MINOR-bump) — update any exhaustive switch on Status. Behavior-neutral for normal models (dof>0).',
    gate: 'build clean + default ctest 39/39 (normal models still Ok, goldens unchanged) + thorough green. (The dof<=0 status itself is asserted by F2.)' },
  { id: 'F2', title: 'F2 domain-outcome test', research: false,
    fix: 'Add the M(fit-5) DOMAIN-OUTCOME TEST (the contract gate fit-engine.md:432 / architecture.md §13:803 / §18:939 require but NO test exists — every status assertion currently checks ==Ok). Following the EXISTING test_qpadm_parity.cu golden-test pattern (REAL f2_fit0_9pop.bin fixture, the real AADR pops), construct DEGENERATE models from REAL pops and assert the API returns the STATUS VALUE (not a crash/NaN) on BOTH the CpuBackend oracle AND the CudaBackend (GPU): (a) a collinear left set (e.g. a left with a DUPLICATED real source pop, or two perfectly-correlated reals) -> Status::RankDeficient; (b) a model that yields a non-SPD covariance -> Status::NonSpdCovariance; (c) a dof<=0 model (e.g. nr < nl-1 / over-parameterized) -> Status::ChisqUndefined (from F3). NO synthetic matrices — only real AADR pops arranged degenerate. Wire into tests/CMakeLists.txt (or extend the existing qpadm test). Assert: status == the expected value, no crash, no NaN leak into a reported field. Run on both backends (default GPU + STEPPE_THOROUGH CpuBackend).',
    gate: 'the new domain-outcome test passes (correct status VALUES on both backends, no crash); the existing goldens stay green (default + thorough).' },
  { id: 'F6', title: 'F6 determinism widen', research: false,
    fix: 'WIDEN the G1==G2 determinism memcmp in test_qpadm_rotation.cu (~:452-467) to cover the FULL QpAdmResult — it currently memcmps only model_index/status/f4rank/weight/p/chisq/se, missing z, dof, est_rank, rank_chisq/rank_dof, and the rankdrop_*/popdrop_* arrays (all reported fields). Compare EVERY reported field G=1 vs G=2 (bit-identical). Test-only hardening. If the widened compare reveals a REAL nondeterminism in a previously-unchecked field, that is a genuine bug — STOP and report it (do not loosen the compare to pass).',
    gate: 'the widened G1==G2 memcmp passes bit-identical across all QpAdmResult fields (real data, 84-model rotation); all existing tests green.' },
  { id: 'F4', title: 'F4 qpwave golden', research: false,
    fix: 'Pin a REAL AT2 qpwave() golden + test the first-class `run_qpwave` entry (qpadm.hpp:226-237, impl qpadm_fit.cpp:266-309) which today has ZERO test coverage (its no-target-prepend / left[0]-is-reference semantic is unvalidated). On the box (R + admixtools 2.0.10 / R 4.3.3): pick a REAL well-determined 2-way qpWave model over real AADR pops present in the .ind (e.g. reuse the golden_fit0 pop set as a qpwave left/right split), run admixtools::qpwave(boot=FALSE), capture est_rank + the rankdrop table + §12 metadata, and write a golden (JSON + the matching real f2 fixture, EXACTLY like golden_fit0.json/its fixture) into tests/reference/goldens/at2/. Add a test (existing golden-test pattern) that calls run_qpwave DIRECTLY on BOTH backends and gates est_rank/rankdrop/chisq vs the AT2 golden. NO synthetic data. Run DETACHED + poll on the box for the R step (flaky net).',
    gate: 'run_qpwave matches the pinned REAL AT2 qpwave golden within tier on BOTH the CpuBackend (thorough) AND the GPU; default + thorough ctest green; the golden + fixture committed.' },
  { id: 'F1', title: 'F1 NA/missing-block', research: true,
    fix: 'Missing-block / NA handling (est_to_loo_nafix) — design OQ-12 said add it BEFORE the at-scale search; S8 shipped without it, so it is overdue. RESEARCH FIRST (verify against HEAD + the block-partition code + real AADR): CAN an empty block (vpair[i,j,b]==0 for a pair, or a block with 0 kept SNPs) actually occur in the maxmiss=0 GLOBAL-INTERSECTION resident f2 path? Read block_partition_rule.* (how n_block is formed — are zero-SNP blocks emitted?), the f2_blocks build, and how AT2 (allsnps=FALSE) treats an empty block. Cite the evidence. THEN: (PATH A — if empty blocks provably CANNOT occur) PROVE it, DOCUMENT the invariant (close OQ-12), add an internal assertion/test that no resident block has vpair==0 for the gated path, and gate the boundary — cheap, no new math, no AT2 golden needed. (PATH B — if empty blocks CAN occur) IMPLEMENT the NA-aware LOO: a vpair[i,j,b]==0 entry is NA and EXCLUDED from the LOO/jackknife (NOT imputed 0, which biases toward 0), matching AT2 est_to_loo_nafix — on BOTH the CpuBackend oracle (parity) AND the GPU kernels (the qpadm_fit_kernels loo paths), and validate the existing goldens still pass + add a REAL-AADR test that exercises a missing-block case (pin an AT2 golden with a sparse pop set if one is needed; real data only). If PATH B needs an AT2 missing-block golden that cannot be cleanly pinned, STOP and report the finding + the scope question (do not fake it).',
    gate: 'PATH A: the no-empty-block invariant proved + asserted + OQ-12 closed; existing goldens green. PATH B: NA-aware LOO on CPU+GPU, existing goldens green, a real-AADR missing-block test passes (CpuBackend==GPU). Either way default + thorough green, no synthetic data.' },
]

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['item','pass','diff_real','follows_standards','cpu_and_gpu_done','no_synthetic','goldens_green','thorough_green','build_clean','commit_hash','note'],
  properties: {
    item: { type: 'string' },
    pass: { type: 'boolean', description: 'true ONLY if: the diff genuinely closes the finish item per the punch-list + analyzes/follows existing code+standards + (if it has a CPU+GPU side) BOTH are done (CPU-for-parity is not complete without the GPU) + NO synthetic data / existing real-golden test pattern followed + default ctest 39/39 green vs real goldens + STEPPE_THOROUGH green (oracle) + Release build clean + the item own new test/assertion passes' },
    diff_real: { type: 'boolean', description: 'real change closing the item (not a stub/comment-only); re-verified vs HEAD (no stale-line edits)' },
    follows_standards: { type: 'boolean', description: 'follows the existing patterns/NAMING-STYLE-STANDARD/§-rules; CUDA API claims doc-verified' },
    cpu_and_gpu_done: { type: 'boolean', description: 'if the item has a CpuBackend AND GPU side (F3/F1), BOTH are properly done; else N/A=true' },
    no_synthetic: { type: 'boolean', description: 'NO synthetic data; new tests follow the existing real-AADR golden pattern (no smoke tests, no synthetic fixtures/matrices)' },
    goldens_green: { type: 'boolean' }, thorough_green: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed (CPU+GPU split if applicable) + the gate result + the new test; for F1 which PATH (A prove+gate / B impl) + why; for FAIL exactly what blocked it' },
  },
}

const ledger = []
for (const it of ITEMS) {
  phase(it.title)
  let research = ''
  if (it.research) {
    research = await tryAgent([
      'You are researching the BEST approach for steppe finish-item ' + it.id + ' BEFORE any edit. READ-ONLY (read the actual code at HEAD + verify the punch-list claims; web-search the relevant AT2/CUDA behavior if useful via ToolSearch select:WebSearch,WebFetch). Do NOT edit, do NOT build.', STD, '',
      'THE ITEM: ' + it.fix, '',
      'Determine the correct approach with EVIDENCE (file:line + cited AT2/CUDA behavior). For F1 specifically: answer definitively whether an empty/NA block can occur in the maxmiss=0 resident f2 (read block_partition_rule.*, the f2_blocks build, how n_block is formed) -> recommend PATH A (prove+gate) or PATH B (NA-aware LOO on CPU+GPU). Return the recommended approach + the evidence so the fixer executes it without re-deriving.',
    ].join('\n'), { label: 'research:' + it.id, phase: it.title }) || ''
  }
  const fixer = [
    'You are a senior CUDA/C++ engineer applying ONE fit-engine FINISH item to steppe. Do NOT commit (the verdict commits). FIRST clean HEAD: ' + CLEAN + ' .', STD, '', DEVLOOP, '',
    'ITEM ' + it.id + ': ' + it.fix, '', 'THE GATE: ' + it.gate,
    research ? ('\nRESEARCH (use this, do not re-derive):\n<<<\n' + String(research).slice(0, 3000) + '\n>>>') : '',
    '', 'VERIFY the punch-list claims against the ACTUAL code at HEAD first (lines may have drifted — find the real sites). Apply the fix following the existing standards/patterns; if it has a CPU + GPU side, do BOTH (CPU-for-parity is not complete without the GPU). Verify any CUDA/cuBLAS API claim vs the CUDA 13.x docs (cite). Build + run the gates (default + thorough). Report: every file:line changed (CPU vs GPU split if applicable); the new test + what it asserts (real-AADR pattern, no synthetic); the CUDA-doc citation if any; the FULL build + ctest output. Do NOT commit. If you cannot reach green, do NOT fake it — report exactly what blocked it (for F1, which PATH + why).',
  ].join('\n')
  const fix = await tryAgent(fixer, { label: 'fix:' + it.id, phase: it.title })
  if (fix === null) { ledger.push({ item: it.id, pass: false, note: 'fixer died' }); log('--- ' + it.id + ' fixer died — HALT'); break }

  // build-repair: trivial -Werror only, no full revert over a one-liner
  await tryAgent(['You are BUILD-REPAIR for ' + it.id + '. The fixer accumulated edits (do NOT clean/revert). Reach a CLEAN Release build, patching ONLY trivial -Werror (unused param/var; [[maybe_unused]] for STEPPE_ASSERT-only params). DO: ' + RSYNC + ' then ' + BUILD + ' . If a trivial -Werror fires, fix minimally + rebuild, LOOP up to 4x. Do NOT change fix logic, do NOT revert. If it fails for a NON-trivial reason, STOP + report. Report final build + the trivial patches.', STD].join('\n'), { label: 'repair:' + it.id, phase: it.title })

  const verdict = await tryAgent([
    'You are the INDEPENDENT VERDICT for ' + it.id + ' of steppe (adversarial). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
    'THE ITEM: ' + it.fix, '', 'THE GATE: ' + it.gate,
    '', 'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — confirm a REAL fix per the punch-list (not stub/comment-only/stale-line), following existing standards; if it has a CPU+GPU side BOTH are done; NO synthetic data + the new test follows the existing real-AADR golden pattern (no smoke test). (2) RE-RUN: ' + BUILD + ' ; ' + THOROUGH + ' . (3) PASS only if ALL: real + standards-conformant; default ctest 39/39 green vs real goldens; STEPPE_THOROUGH green (oracle); Release build clean; the item own new test/assertion passes; CPU+GPU both done where applicable; NO synthetic. ',
    'ON PASS: cd ' + R + ' && git add ONLY this item changed/new source+test+doc+golden files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (the finish item + the gate result + CUDA-doc cite if any; for F1 the PATH taken) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.',
    'ON FAIL: ' + CLEAN + ' (leave repo green) + report exactly what blocked it. Return the structured verdict.',
  ].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:' + it.id, phase: it.title })
  if (verdict === null) { ledger.push({ item: it.id, pass: false, note: 'verdict died' }); log('--- ' + it.id + ' verdict died — HALT'); break }
  ledger.push(verdict)
  if (verdict.pass) log('+++ ' + it.id + ' committed ' + verdict.commit_hash + ' — ' + verdict.note)
  else { log('--- ' + it.id + ' FAILED (' + verdict.note + ') — reverted; HALT (correctness, no skip)'); break }
}

const passed = ledger.filter(x => x.pass).length
log('fit-engine-finish: ' + passed + '/' + ITEMS.length + ' committed')
return { ledger, passed }
