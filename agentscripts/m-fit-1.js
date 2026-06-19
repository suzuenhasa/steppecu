export const meta = {
  name: 'm-fit-1',
  description: 'M(fit-1): the FIRST qpAdm GLS fit — steppe computes a real population-model result and is validated against the AT2 golden (af6a8c2). Build S3 (f4 matrix X from f2_blocks) + S4 (weighted block-jackknife covariance Q) + S6 (ALS weight solve, AT2-exact) + S7 (jackknife SE), for ONE well-determined 2-way model, on the CpuBackend REFERENCE (native FP64), behind a GPU-FIRST architecture (batched/resident-capable types + ComputeBackend virtuals the CUDA backend will later implement — NOT a CPU-shaped transliteration of AT2 R loops; CpuBackend is only the parity oracle). Gate: steppe est/se/z/p match the af6a8c2 golden (England_BellBeaker = Czechia_CordedWare + Turkey_N: weights 0.5589/0.4411, se 0.22591, z 2.474/1.953, p 0.32682) within §12 tolerance, with the intermediate X and Q cross-checked vs the golden. Design/Contracts -> coupled implement on box5090 -> INDEPENDENT verdict (parity vs golden) commits. HALT-on-fail.',
  phases: [
    { title: 'Design', detail: 'freeze the GPU-first qpAdm seam/types + the f2->f4->Q->weights->SE math (AT2-exact ALS/fudge/LOO/block-weight) + the parity strategy vs af6a8c2' },
    { title: 'Implement', detail: 'core/qpadm/ + qpadm.hpp + CpuBackend reference fit + the AT2-golden parity test, on box5090 to green parity' },
    { title: 'Verify', detail: 'independent verdict: build+ctest green + steppe qpAdm matches the af6a8c2 golden (weights/X/Q/p) within tolerance; commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45'"
const QPADM_TEST = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ctest --test-dir build-rel -R qpadm --output-on-failure 2>&1 | tail -50'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell reimplementation of ADMIXTOOLS 2 f-statistics. Phase-1 precompute DONE (f2_blocks device-resident, DeviceF2Blocks). Phase 2 = the qpAdm fit engine (S3-S8); this is M(fit-1), the FIRST fit milestone. Branch phase2-fit-engine @ af6a8c2. The DESIGN is docs/design/fit-engine.md (read it — esp. §1 architecture, the first-milestone contract, and the OPEN QUESTIONS OQ-1/OQ-3). The AT2 GOLDEN oracle is committed at tests/reference/goldens/at2/ (golden_fit0.json + csv/{weights,X,Q,rankdrop,popdrop,f4}.csv + README with the §12 metadata).',
  'GPU-FIRST ARCHITECTURE (BINDING — the user directive, memory gpu-first-architecture-not-cpu-shape): design for OPTIMAL GPU processing, NOT a transliteration of AT2 R loops. Reproduce AT2 MATH for parity, but the TYPES + SEAM must be GPU-first: f2_blocks stays device-resident (the fit reads DeviceF2Blocks in VRAM); the data structures + ComputeBackend virtuals must be BATCHED-CAPABLE (so S8 can batch across thousands of models, and S7 can batch the ~n_block jackknife replicates on-device) and RESIDENT-CAPABLE — never per-model/per-block host loops as the architecture. The CpuBackend in this milestone is ONLY the parity reference/oracle (it implements the SAME ComputeBackend virtuals the CUDA backend will later implement); do NOT bake a CPU-shaped data flow the GPU would have to retrofit.',
  'THE M(fit-1) SCOPE (per the design first-milestone contract): S3 f4 matrix X[nl x nr] per block from f2 (X[i,j]=(f2(Li,R0)+f2(L0,Rj)-f2(L0,R0)-f2(Li,Rj))/2; L0=target prepended to left); S4 weighted block-jackknife covariance Q[m x m], m=nl*nr (est_to_loo totals->LOO, xtau pseudo-values, Q = xtau*xtau^T/numblocks, then diag += fudge*tr(Q), fudge=1e-4, invert); S6 GLS WEIGHTS via AT2 ALS — CRITICAL (OQ-1): AT2 is NOT a single Cholesky; it is svd(X) -> rank-r A,B refined by alternating opt_A/opt_B (default 20 iters, fudge ridge), then the constrained weight solve (solve(crossprod(x), crossprod(x,y)), normalized). REPRODUCE AT2 verbatim for parity. S7 SE from LOO weight replicates (re-fit per leave-one-block-out, jackknife variance). ONE model, full rank, no missing blocks. NATIVE FP64 (the reference; §12 cancellation-sensitive). OQ-3: the jackknife BLOCK WEIGHT — block_sizes[b] (AT2 block_lengths) vs per-pair Vpair — get it right; the golden Q is the cross-check (if Q matches, the weight is right).',
  'THE GOLDEN (af6a8c2, the af6a8c2 model): target=England_BellBeaker; left(sources)=Czechia_EBA_CordedWare, Turkey_N; right(outgroups)=Mbuti, Israel_Natufian, Iran_GanjDareh_N, Han, Papuan, Karitiana. blgsize=0.05, maxmiss=0, auto_only, 708 blocks, 500893 SNPs. Golden: weight CordedWare=0.5589062489 (se 0.2259118618, z 2.474), Turkey_N=0.4410937511 (se 0.2259118618, z 1.953); rank-1 chisq 4.6352 dof 4 p 0.32682; X (f4 vector) + Q (5x5 covariance) in the csv. admixtools 2.0.10, R 4.3.3.',
  'PARITY STRATEGY (the design resolves the exact form): the cleanest M(fit-1) test ISOLATES the FIT math from any steppe-f2-vs-AT2-f2 difference — feed steppe qpAdm the SAME f2 AT2 used (the golden f2 blocks are on box5090 at /workspace/data/aadr/f2_fit0_FINAL/; OR reconstruct X from the golden and test S5/S6/S7; OR derive a 9-pop f2 with steppe and ALSO compare steppe-X vs golden-X to localize). Compare steppe est/se/z/p vs the golden: WEIGHTS tight (rtol ~1e-6 to AT2 0.5589/0.4411), se/z looser (~1e-3, derived), p close; cross-check the intermediate X and Q vs the golden csv (this localizes S3/S4 vs S6/S7). State the chosen f2-source explicitly.',
  'BOX = box5090 (vast 2x RTX 5090, UP). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists. RELEASE build (the fit is CPU-side in M(fit-1) but builds with the project on the box; NOTHING builds locally). The golden f2 + the AADR data are on the box. Flaky network -> long steps detached + poll.',
  'KEY EXISTING SEAMS: include/steppe/fstats.hpp (F2BlockTensor), src/device/device_f2_blocks.hpp (DeviceF2Blocks + to_host), src/device/backend.hpp (ComputeBackend — ADD the qpAdm-supporting virtuals here, batched-capable), src/device/cpu/cpu_backend.cpp (the CpuBackend reference impl), src/core/ (NEW core/qpadm/ orchestration, host-pure, CUDA-free), include/steppe/ (NEW qpadm.hpp public CUDA-free QpAdmModel/QpAdmResult), tests/ (NEW test_qpadm_parity against the golden), tests/reference/goldens/at2/ (the golden), src/device/config.hpp (Precision).',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); the qpAdm parity test (' + QPADM_TEST + '). Iterate until the qpAdm parity test passes vs the af6a8c2 golden. Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA/C++ architect. Produce the EXACT FROZEN contract for M(fit-1) — the first qpAdm GLS fit, GPU-FIRST architecture + CpuBackend reference, validated against the af6a8c2 AT2 golden. READ-ONLY (read docs/design/fit-engine.md, the golden at tests/reference/goldens/at2/, the existing seams; do NOT edit, do NOT touch the box).', STD, '',
  'Specify with file:line + exact C++ signatures: (1) the PUBLIC GPU-FIRST types — include/steppe/qpadm.hpp: QpAdmModel (index-based: target + left + right as pop indices into f2_blocks; batched-capable = a list of models for S8 later) + QpAdmResult (weights/se/z + chisq/dof/p + rank stats). CUDA-free. (2) the ComputeBackend virtuals to ADD (backend.hpp) for the fit primitives — BATCHED-CAPABLE signatures (the CUDA backend will implement them batched/resident; the CpuBackend implements them as the reference): the f4-gather/X build, the jackknife Q, the SVD/ALS, the GLS solve. (3) the core/qpadm/ orchestration (host-pure, CUDA-free, drives ComputeBackend) — the S3->S4->S6->S7 data flow. (4) the EXACT math per AT2 (cite the design / AT2 source): the f4 X identity + index convention (target prepended to left); the est_to_loo + xtau + Q (m=nl*nr) + fudge 1e-4 + invert; the ALS weight solve (svd -> opt_A/opt_B 20 iters + constrained solve + normalize) — REPRODUCE AT2, this is OQ-1; the jackknife block weight (block_sizes vs Vpair, OQ-3 — pick the one that reproduces the golden Q). (5) the PARITY TEST design: the f2-source (isolate the fit — feed the golden/AT2 f2, OR steppe-derived + X-cross-check), and the tolerances (weights rtol ~1e-6, se/z ~1e-3, p; X/Q cross-check vs the golden csv). The implementer makes NO design decisions. Be explicit GPU-first (batched/resident types) vs CpuBackend-reference (this milestone implements the reference).',
].join('\n'), { label: 'design:m-fit-1', phase: 'Design' })

phase('Implement')
const impl = await agent([
  'You are a senior CUDA/C++ engineer. Implement M(fit-1) per the FROZEN design and dev-loop on box5090 until the qpAdm parity test passes vs the af6a8c2 golden. EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(missing — derive from docs/design/fit-engine.md + the golden)') + '\n>>>', '', DEVLOOP, '',
  'Build: include/steppe/qpadm.hpp (GPU-first QpAdmModel/QpAdmResult) + the ComputeBackend virtuals (batched-capable) + core/qpadm/ orchestration + the CpuBackend reference impl (native FP64, AT2-exact ALS/Q/SE) + tests/test_qpadm_parity reading tests/reference/goldens/at2/. Build Release + ctest, then run the qpAdm parity test. Iterate until: build clean (warnings-as-errors), ctest green (no regression), AND the qpAdm parity test PASSES — steppe weights match the golden (0.5589/0.4411) within rtol~1e-6, se/z within ~1e-3, p close, and the intermediate X and Q match the golden csv. If X matches but weights do not, the ALS (OQ-1) is wrong; if Q does not match, the block-weight (OQ-3) is wrong — fix accordingly. Report: git --no-pager diff --stat; files added/changed; build/ctest; the qpAdm parity output (steppe vs golden, per quantity); the f2-source used. Do NOT commit. If parity cannot be reached, report exactly which quantity diverges + by how much — do NOT fabricate a pass.',
].join('\n'), { label: 'impl:m-fit-1', phase: 'Implement' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','gpu_first_types','build_green','weights_match','X_match','Q_match','p_match','steppe_weights','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real non-empty diff adding the GPU-first qpAdm seam + core/qpadm + CpuBackend reference fit + the parity test; Release build clean (warnings-as-errors); ctest green; AND steppe qpAdm matches the af6a8c2 golden — weights within rtol~1e-6 (0.5589/0.4411), X and Q cross-checked vs the golden csv, p close' },
    diff_is_real: { type: 'boolean', description: 'you re-ran git diff: qpadm.hpp + core/qpadm + the CpuBackend fit + the parity test are genuinely present (not a stub/sham)' },
    gpu_first_types: { type: 'boolean', description: 'the types/seam are GPU-FIRST (batched-capable QpAdmModel list + ComputeBackend virtuals the CUDA backend can implement batched/resident; reads DeviceF2Blocks), NOT a CPU-shaped per-model host transliteration' },
    build_green: { type: 'boolean' },
    weights_match: { type: 'boolean', description: 'steppe weights == golden 0.5589062489/0.4410937511 within rtol~1e-6' },
    X_match: { type: 'boolean', description: 'steppe f4 matrix X == the golden X csv (S3 correct)' },
    Q_match: { type: 'boolean', description: 'steppe jackknife covariance Q == the golden Q csv (S4 + the block-weight OQ-3 correct)' },
    p_match: { type: 'boolean', description: 'steppe model p-value close to the golden 0.32682' },
    steppe_weights: { type: 'string', description: 'the YOU-measured steppe weights/se/z/p vs the golden' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the per-quantity steppe-vs-golden comparison + the f2-source; for FAIL exactly which quantity diverges + by how much (X? Q? weights? p?)' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT for M(fit-1) (maximally adversarial — prior steppe fix-passes have SHAMMED; this is the FIRST qpAdm fit and must genuinely match the AT2 golden). The implementer reported:\n<<<\n' + (impl || '(impl died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM non-empty AND genuinely contains: include/steppe/qpadm.hpp (GPU-first QpAdmModel/QpAdmResult), the ComputeBackend qpAdm virtuals (batched-capable), core/qpadm/ orchestration, the CpuBackend reference impl (AT2-exact ALS + Q + SE), and tests/test_qpadm_parity. CHECK the types are GPU-FIRST (batched/resident-capable, reads DeviceF2Blocks), not a CPU-shaped transliteration. If stub/sham/empty => pass=false. (2) RE-RUN yourself: ' + BUILD + ' ; ' + QPADM_TEST + ' . (3) PASS only if ALL: diff real + GPU-first; Release build clean; ctest green; AND steppe qpAdm matches the af6a8c2 golden — weights within rtol~1e-6 (CordedWare 0.5589062489, Turkey_N 0.4410937511), the intermediate X matches the golden X csv, the Q matches the golden Q csv, and the model p is close to 0.32682. Inspect the parity-test output yourself; if X matches but weights do not, the ALS is wrong (OQ-1, fail); if Q is off, the block-weight is wrong (OQ-3, fail). \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-new/changed source+test files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (the first steppe qpAdm fit + the measured vs-golden parity) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly which quantity diverges.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:m-fit-1', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ M(fit-1) LANDED ' + verdict.commit_hash + ' — first steppe qpAdm fit matches AT2 golden (weights=' + verdict.weights_match + ' X=' + verdict.X_match + ' Q=' + verdict.Q_match + ' p=' + verdict.p_match + '): ' + verdict.steppe_weights)
else log('--- M(fit-1) FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, impl, verdict }
