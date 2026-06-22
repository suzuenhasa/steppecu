export const meta = {
  name: 'fix-jk1-se-vram-budget',
  description: 'Fix the jackknife-1 (feasible-only SE) single-GPU VRAM OOM in the qpadm rotation so big rotations get standard errors, not just point estimates. MEASURED: jk0 (point estimate) scales clean to pool_200 (1.33M models, peak 31.6/32GB); jk1 OOMs between pool_50 and pool_60 because the SE pass-2 (the feasible-only LOO re-fit over survivors — device_survivor_blocks + the x_loo/dLoo/dXtau arenas, src/device/cuda/cuda_backend.cu ~:1416-1431) allocates a SECOND arena set while the pass-1 buffers are live, and the per-chunk VRAM budgeter (fit_one_bucket / per_model_bytes / max_blocks_per_chunk, ~:607-616 + vram_budget.hpp) does NOT account for the pass-2 footprint -> it sizes chunks for pass-1 only -> pass-2 over-commits -> cudaMalloc OOM at device_buffer.cuh. THE FIX (NO code duplication): make the EXISTING per-chunk budgeter SE-aware so the EXISTING chunked machinery tiles small enough for BOTH passes when jackknife>=1 — i.e. fold the SE pass-2 arena bytes into per_model_bytes / the budget term (so max_blocks_per_chunk shrinks under jk1/jk2), AND/OR route the SE pass-2 through the SAME chunk loop, AND/OR scope the pass-1 buffers to free before pass-2 — whichever the code shows is correct, REUSING the existing budgeter+chunker (do NOT copy the chunking logic into a parallel SE path). PARITY-NEUTRAL (architecture.md §12): chunk size moves NO bits — the fit must be BIT-IDENTICAL before/after (the small pools that already ran jk1, e.g. pool_25/pool_50, must give byte-identical results). GOLDEN-GATE: test_qpadm_rotation + golden_rot + cli_rotate + all goldens stay green (bit-identical). VERIFY the unlock: jk1 rotation now COMPLETES single-GPU at pool_100/150/200 (was OOM), bit-identical to the pre-fix pool_50 jk1, and report the new jk1 ceiling + rate. design/verify -> implement(no dup) -> build-repair -> verify(parity + jk1 scales) + commit. SINGLE-GPU (--device 0; multi-gpu parked); RELEASE; REAL v66 1240K; HALT-on-fail; resumable on 529.',
  phases: [ { title: 'Design the SE-budget fix (verify, no-dup plan)' }, { title: 'Implement + build' }, { title: 'Verify parity + jk1 scaling + commit' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 315f345. qpadm-rotate is wired to the batched run_qpadm_search engine. The rotation already TILES the BLOCK dimension via a per-chunk VRAM budget (cuda_backend.cu ~:607-616 + max_blocks_per_chunk in vram_budget.hpp + size_buckets). jk0 (point estimate) scales to pool_200 (1.33M models). The f2 input is RESIDENT in VRAM (fits at these pool sizes).',
  'THE BUG: jk1 (feasible-only SE) and jk2 (all SE) OOM single-GPU above ~pool_50. The SE pass-2 (the LOO re-fit over feasible survivors — device_survivor_blocks + x_loo/dLoo/dXtau, cuda_backend.cu ~:1357-1431) allocates a SECOND arena set while pass-1 buffers are live, and the per-chunk budgeter (fit_one_bucket / per_model_bytes) does NOT count the pass-2 footprint -> chunks sized for pass-1 only -> pass-2 over-commits -> cudaMalloc OOM (device_buffer.cuh). (There is a comment near :612 noting a PRIOR ~2x under-budget fix — this is the same class: one more uncounted buffer.)',
  'THE FIX — NO DUPLICATION: reuse the EXISTING per-chunk budgeter + chunk loop. Make the budget SE-AWARE: fold the SE pass-2 arena bytes into per_model_bytes / the budget term so max_blocks_per_chunk returns a SMALLER value when jackknife>=1 (so the existing chunked machinery tiles small enough for both passes); AND/OR run the SE pass-2 through the SAME chunk loop; AND/OR scope the pass-1 buffers to free before pass-2 (so they do not coexist). Pick what the code shows is correct + minimal; do NOT copy the chunking/budget logic into a parallel SE-only path. Keep the per-block/per-model byte coefficients single-sourced (the existing single-source pattern).',
  'PARITY-NEUTRAL (architecture.md §12): chunk size moves NO bits. The fit result MUST be BIT-IDENTICAL before/after — the pools that already ran jk1 (pool_25, pool_50) must produce byte-identical weights/SE/p. Smaller chunks must sum/combine to the same values (the jackknife/LOO is chunk-order-deterministic; preserve the existing reduction order).',
  'SINGLE-GPU only (--device 0; multi-gpu PARKED). RELEASE -DSTEPPE_BUILD_CLI=ON. REAL v66 1240K at ' + PREFIX + '. nothing builds locally; clear core dumps. Binary ' + BIN + '. Verify CUDA/cuBLAS/cuSOLVER VRAM claims against the docs.',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+thorough-ctest (' + BUILD + '). Do NOT commit (the verdict commits). NO synthetic. SINGLE-GPU.'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Design the SE-budget fix (verify, no-dup plan)')
const design = await tryAgent([
  'You are a senior CUDA engineer DESIGNING the jk1 SE-pass VRAM-budget fix (verify-before-implement; NO code changes this phase). READ src/device/cuda/cuda_backend.cu: the per-chunk VRAM budget for the main fit (~:594-720, fit_one_bucket, the max_blocks_per_chunk call, per_model_bytes / the per-block byte coefficient, size_buckets), AND the SE pass-2 (~:1318-1450: device_survivor_blocks, the feasible-only LOO re-fit, the x_loo/dLoo/dXtau / survivor arenas — their exact sizes m, nb_s, and WHEN they are allocated vs the pass-1 buffers). Also src/device/vram_budget.hpp (max_blocks_per_chunk + the per-block coefficients) and include/steppe/config.hpp (the named buffer coefficients). And how jackknife policy (JackknifePolicy 0/1/2) flows into the fit.', STD, '',
  'PRODUCE the precise NO-DUP fix plan: (1) WHERE the SE pass-2 arenas are allocated + their byte formula, and whether they coexist with pass-1. (2) The exact change to make the EXISTING budgeter SE-aware (add the pass-2 term to per_model_bytes / the budget so max_blocks_per_chunk shrinks under jk>=1) AND/OR scope pass-1 to free first AND/OR route pass-2 through the existing chunk loop — pick the minimal correct option that REUSES the existing machinery (name the functions to reuse; confirm NO duplicated chunk/budget logic). (3) Why it is PARITY-NEUTRAL (the reduction/order is preserved; chunk size changes no values). (4) The verify plan: which small pool (pool_50) proves bit-identical jk1, and that pool_100/150/200 jk1 now fit. Cite file:line. Return the plan (do NOT implement).',
].join('\n'), { label: 'design:se-budget', phase: 'Design the SE-budget fix (verify, no-dup plan)' })
if (design === null) { log('--- design died — HALT'); return { halted: true } }

phase('Implement + build')
const fixer = await tryAgent([
  'You are a senior CUDA engineer implementing the jk1 SE-pass VRAM-budget fix per this plan:\n<<<\n' + design + '\n>>>\n\nDo NOT commit. Start clean: ' + CLEAN + '.', STD, '', DEVLOOP, '',
  'IMPLEMENT exactly per the plan — make the EXISTING per-chunk budgeter SE-aware (NO duplicated chunk/budget logic; reuse fit_one_bucket / max_blocks_per_chunk / the single-sourced coefficients). Keep it PARITY-NEUTRAL. Build + full STEPPE_THOROUGH ctest (golden_rot/test_qpadm_rotation/cli_rotate/all goldens must stay bit-identical green). SANITY (no commit): on the box single-GPU, build a pool_100 f2 dir on the 1240K (target England_BellBeaker, ~100-pop pool, 6 right, maxmiss 0.5) and run qpadm-rotate --jackknife 1 --min-sources 1 --max-sources 3 -> confirm it now COMPLETES (was OOM at pool_60) + report wall + peak VRAM; and confirm a pool_50 jk1 run is byte-identical to before the fix (parity). Report every file changed, the no-dup approach, and the FULL ctest. Do NOT commit.',
].join('\n'), { label: 'implement:se-budget', phase: 'Implement + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true } }
await tryAgent(['BUILD-REPAIR for the jk1 SE-budget fix. Accumulated edits (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON) + green ctest, patching only trivial -Werror. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial (a parity break or logic error) -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement + build' })

phase('Verify parity + jk1 scaling + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','no_duplication','parity_bit_identical','jk1_scales','jk1_detail','goldens_green','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the SE-budget fix reuses the existing budgeter/chunker (NO dup), is parity-neutral (jk1 pool_50 byte-identical pre/post), jk1 now COMPLETES at pool_100/150/200 single-GPU (was OOM), THOROUGH ctest + golden_rot + cli_rotate green, Release build clean, single-GPU, no synthetic' },
    no_duplication: { type: 'boolean', description: 'reused fit_one_bucket/max_blocks_per_chunk + single-sourced coefficients; no parallel SE chunk/budget logic copied' },
    parity_bit_identical: { type: 'boolean', description: 'jk1 pool_50 (or golden_rot) results byte-identical pre/post the fix (chunking moves no bits)' },
    jk1_scales: { type: 'boolean', description: 'jk1 rotation now completes at pool_100/150/200 single-GPU (was OOM ~pool_60)' },
    jk1_detail: { type: 'string', description: 'the new jk1 ceiling + wall + peak VRAM at pool_100/150/200; vs the old OOM at ~pool_60' },
    goldens_green: { type: 'boolean' }, build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the no-dup approach (which existing fn made SE-aware); the parity check; the jk1 numbers; for FAIL the exact issue' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the jk1 SE-pass VRAM-budget fix (adversarial; this touches the rotation fit path). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff — confirm the fix made the EXISTING budgeter SE-aware (NO duplicated chunk/budget logic; reused fit_one_bucket/max_blocks_per_chunk/single-sourced coeffs) and is parity-neutral (no reduction-order change). (2) ' + BUILD + ' — THOROUGH ctest green incl golden_rot/test_qpadm_rotation/cli_rotate (bit-identical). (3) PARITY: run a pool_50 jk1 rotation on the 1240K and confirm byte-identical results to before the fix (or that golden_rot jk-path is unchanged). (4) THE UNLOCK: run qpadm-rotate --jackknife 1 at pool_100, pool_150, pool_200 on the 1240K single-GPU -> confirm they now COMPLETE (were OOM above ~pool_50) + report wall + peak VRAM + the new jk1 ceiling. PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (fix: budget the jk1/jk2 SE pass-2 arenas in the existing per-chunk VRAM budgeter [no dup] -> rotation SEs scale single-GPU past pool_50; parity-neutral) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . ALSO update docs/perf/1240k-sweep.md (the jk1 SE path now scales to pool_NNN single-GPU; the old ~pool_50 OOM is fixed). Capture the hash + the jk1 numbers.',
  'ON FAIL: ' + CLEAN + ' and report the exact issue (parity break? still OOM? where?). Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:se-budget', phase: 'Verify parity + jk1 scaling + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ jk1 SE scales ' + verdict.commit_hash + ' — ' + verdict.jk1_detail)
else log('--- FAILED (' + verdict.note + ')')
return { verdict }
