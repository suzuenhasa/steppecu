export const meta = {
  name: 'm4.5-persistent-pin-d2h',
  description: 'Make multi-GPU genuinely PARALLEL: kill the per-call cudaHostRegister/cudaHostUnregister on the D2H result slices that serializes the two devices on the driver-wide lock (nsys measured: only 22.7% GPU-overlap; G2 GPU-union 909ms vs G1 510ms @P=512; ~690ms/iter register/unregister serial tail; introduced by a84b85b). FIX: pin the D2H destination ONCE (persistent pinned staging per backend via cudaHostAlloc, reused across calls + cheap concurrent host copy; OR register the result region once) so the two devices D2Hs overlap; + move remaining per-call device cudaMalloc/cudaFree off the hot path. Design -> implement on box5090 to green parity -> INDEPENDENT verdict gating on memcmp parity bit-identical + an nsys-measured GPU-overlap JUMP / G2-union-wall DROP. HALT-on-fail.',
  phases: [
    { title: 'Design', detail: 'freeze the persistent-pinned-D2H seam (no per-call register), parity-preserving' },
    { title: 'Implement', detail: 'coupled core on box5090 to green parity + an nsys overlap sanity-check' },
    { title: 'Verify', detail: 'independent verdict: parity bit-identical + nsys GPU-overlap jumped / G2-union dropped; commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -40'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/test_f2_multigpu_parity 2>&1 | tail -50'"
const NSYS_PROF = SSH + " '" + PATHENV + " && cd /workspace/steppe && nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true -o /tmp/fix512 ./build-rel/bin/bench_f2_multigpu /workspace/data/aadr 512 2>&1 | tail -15'"
const NSYS_STATS = SSH + " '" + PATHENV + " && nsys stats --report cuda_gpu_trace --report cuda_api_sum /tmp/fix512.nsys-rep 2>&1 | tail -130'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell reimpl of ADMIXTOOLS 2 f-statistics. M4.5 multi-GPU f2 precompute. f2_blocks[P x P x n_block] FP64. Branch m4.5-d2h-speed (has the a84b85b host-staged-direct-D2H fix we are now improving).',
  'THE MEASURED PROBLEM (nsys, box5090, P=512, docs/cleanup/m4.5/parallelism-check.md): the multi-GPU is NOT truly parallel — only 22.7% GPU-overlap; G2 GPU-union wall 909ms vs G1 510ms (the 2nd GPU makes the GPU timeline SLOWER). The COMPUTE+H2D phase IS parallel (~206ms overlap, work split 0.94:1 — fan-out + shard are correct). The serial part is a ~570ms D2H tail with ZERO device overlap. Cause is NOT PCIe/bytes (real copies only 50+64ms): it is per-call cudaHostRegister/cudaHostUnregister of the D2H destination slices in compute_f2_blocks_into (cuda_backend.cu:352-353 -> pinned_buffer.cuh:164,201 RegisteredHostRegion) — ~690ms/iter — taking the DEVICE-WIDE DRIVER LOCK so the two worker threads cannot pin concurrently (dev1 D2H cannot start until dev0 finishes pinning). Unlike the H2D inputs (PinnedRegistryCache, amortized) the result is pinned RAW EVERY call (the result vector base pointer changes per call so a naive cache never hits, comment pinned_buffer.cuh:236-240).',
  'THE FIX: eliminate the per-call register/unregister so the two devices D2Hs run as concurrent DMAs. Preferred: each backend owns a PERSISTENT pinned staging buffer (cudaHostAlloc once, sized to the largest partial, reused across calls — PinnedBuffer at pinned_buffer.cuh:59 already exists); D2H into it, then a cheap host memcpy into the caller result slice (the memcpy is CPU bandwidth, NOT the driver lock, and runs concurrently on the two worker threads). Alternative if cleaner: register the WHOLE result region ONCE before the fan-out (one cudaHostRegister, not per-device-per-call) and pass stable slices, deleting the per-call RegisteredHostRegion. The DESIGN agent picks whichever is cleanest, parity-safe, and demonstrably shows the overlap win on the bench (which reallocates the result each call — so a PER-BACKEND persistent staging that survives across calls is the robust choice that shows the win). ALSO move the remaining per-call device cudaMalloc/cudaFree (cuda_backend.cu:455,459-462,473 the run-long tensors) off the hot path — allocate once per backend, reuse (same as the slab pre-size at :516-565 already did) — secondary, do if low-risk.',
  'PARITY LAW (architecture.md §12, NON-NEGOTIABLE): test_f2_multigpu_parity memcmp host-staged G==2 vs single-GPU on derived_acc (P=50) + derived_full (P=768, may be VRAM-gated/skipped on the 32GB 5090 — note it). Pinning/staging moves the SAME doubles to the SAME disjoint offsets — bit-identical. A host memcpy from pinned staging to the result slice copies exact bytes. No accumulation, disjoint slices. last_combine_path=HostStaged on this no-P2P box.',
  'BOX = box5090 (2x RTX 5090 sm_120, 32GB ea, CONSUMER no-P2P, CUDA 13.0.88), LIVE. ' + SSH + ' (alias); nvcc/nsys not on PATH -> ' + PATHENV + ' . nsys CUDA-tracing works (not the restricted ncu counters). build-rel exists. RELEASE only. Single-GPU OOMs ~P700 so G1-vs-G2 at P<=512. NOTHING builds locally. THE HONEST METRIC IS THE nsys GPU-UNION WALL / OVERLAP %, not the bench end-to-end (which is dominated by harness subset-repack + 3.18GB alloc + GPU-idle gaps, and last_multigpu_timings is never written).',
  'KEY FILES: src/device/cuda/cuda_backend.cu (compute_f2_blocks_into D2H + the per-call RegisteredHostRegion at :352-353; the per-call device allocs; add a persistent pinned staging member to CudaBackend), src/device/cuda/pinned_buffer.cuh (PinnedBuffer cudaHostAlloc at :59, RegisteredHostRegion at :151, PinnedRegistryCache at :246), src/core/fstats/f2_blocks_multigpu.cpp + f2_blocks_multigpu_core.cpp (the host-staged orchestrator + fan-out), include/steppe/fstats.hpp (F2BlockTensor public type — keep stable), src/device/backend.hpp (the seam), tests/reference/bench_f2_multigpu.cu, tests/reference/test_f2_multigpu_parity.cu.',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); parity (' + PARITY + '); nsys overlap check (profile ' + NSYS_PROF + ' then stats ' + NSYS_STATS + ' — confirm the per-call cudaHostRegister/Unregister tail is gone and the two devices D2Hs overlap / the G2 GPU-union wall dropped toward G1). Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA/C++ architect. Produce the EXACT FROZEN contract for the persistent-pinned-D2H fix that removes the per-call cudaHostRegister/Unregister serialization. READ-ONLY (read parallelism-check.md + the key files incl. how a84b85b added compute_f2_blocks_into and how the H2D PinnedRegistryCache amortizes; do NOT edit, do NOT touch the box).', STD, '',
  'Decide and specify with file:line + signatures: (1) the pinning strategy — persistent per-backend pinned staging (cudaHostAlloc, sized to max partial, reused; D2H into it then concurrent host memcpy to the result slice) vs register-the-result-once-before-fan-out — pick the one that is cleanest, parity-safe, AND demonstrably shows the overlap win on the bench (which reallocates the result each call, so a per-backend persistent staging that survives across calls is favored); justify; (2) the exact change to compute_f2_blocks_into (cuda_backend.cu) — delete the per-call RegisteredHostRegion (:352-353), D2H into the persistent staging, host-copy into the dst slice; the new CudaBackend member + its lifetime; (3) whether the host memcpy reintroduces a serial cost and why it is still a net win (CPU bandwidth, no driver lock, concurrent across the two worker threads — vs 690ms serial register); (4) the optional move of per-call device allocs off the hot path (only if low-risk); (5) parity invariants to preserve (disjoint slices, exact bytes, public F2BlockTensor unchanged). The implementer makes NO design decisions.',
].join('\n'), { label: 'design:persistent-pin', phase: 'Design' })

phase('Implement')
const impl = await agent([
  'You are a senior CUDA/C++ engineer. Implement the persistent-pinned-D2H fix per the FROZEN design and dev-loop on box5090 until host-staged parity is memcmp BIT-IDENTICAL and an nsys check shows the per-call register/unregister tail is GONE and the two devices D2Hs overlap. EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(missing — derive from parallelism-check.md + the persistent-staging direction)') + '\n>>>', '', DEVLOOP, '',
  'Build Release + ctest + parity (host-staged bit-identical). Then nsys-profile a G2 run at P=512 and CONFIRM from the timeline: the cudaHostRegister/cudaHostUnregister ~690ms serial tail is gone, the two devices D2H windows now OVERLAP, and the G2 GPU-union wall dropped from ~909ms toward G1 ~510ms (report the new overlap % and union wall). Report: git --no-pager diff --stat; files changed + what; build/ctest; parity output; the nsys before/after (overlap %, G2 union wall, the register/unregister API time). Do NOT commit. If blocked, report exactly what — do NOT fabricate.',
].join('\n'), { label: 'impl:persistent-pin', phase: 'Implement' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','parity_bit_identical_hoststaged','overlap_pct_after','g2_union_wall_ms_after','g1_union_wall_ms','register_tail_eliminated','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real non-empty diff removing the per-call cudaHostRegister/Unregister (persistent pinned staging or register-once) + Release build clean + ctest green + host-staged parity memcmp BIT-IDENTICAL + the nsys-measured GPU-overlap at P=512 JUMPED substantially above the 22.7% baseline (target >50%) OR the G2 GPU-union wall dropped substantially below 909ms toward G1 510ms' },
    diff_is_real: { type: 'boolean', description: 'you re-ran git diff: the per-call RegisteredHostRegion on the D2H dst is gone, replaced by persistent pinned staging / register-once; public F2BlockTensor stable' },
    parity_bit_identical_hoststaged: { type: 'boolean' },
    overlap_pct_after: { type: 'number', description: 'YOU-measured nsys GPU-overlap % at P=512 (baseline 22.7)' },
    g2_union_wall_ms_after: { type: 'number', description: 'YOU-measured G2 GPU-union wall ms at P=512 (baseline 909)' },
    g1_union_wall_ms: { type: 'number', description: 'G1 GPU-union wall ms at P=512 (baseline 510)' },
    register_tail_eliminated: { type: 'boolean', description: 'the ~690ms/iter per-call cudaHostRegister/Unregister serial tail is gone from the nsys API timeline' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'before/after overlap % + G2 union wall + the register-tail status; for FAIL exactly why (parity / no overlap gain / sham)' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (maximally adversarial — prior fix-passes on this project have SHAMMED, and the metric here is subtle: the bench END-TO-END is harness-contaminated, so you MUST measure the nsys GPU-union wall / overlap, NOT the bench wall). The implementer reported:\n<<<\n' + (impl || '(impl died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM non-empty AND the per-call RegisteredHostRegion on the D2H destination is GONE (replaced by persistent pinned staging or a register-once), public F2BlockTensor stable. If missing/empty => SHAM => pass=false. (2) RE-RUN yourself: ' + BUILD + ' ; ' + PARITY + ' ; then nsys: ' + NSYS_PROF + ' ; ' + NSYS_STATS + ' . From the cuda_gpu_trace per-device timeline compute the GPU-overlap % and the G2 GPU-union wall at P=512; from cuda_api_sum confirm the cudaHostRegister/cudaHostUnregister time collapsed. (3) PASS only if ALL: diff real; build clean; ctest green; host-staged parity memcmp BIT-IDENTICAL (last_combine_path=HostStaged; note if derived_full is VRAM-gated/skipped on 32GB); AND the GPU-overlap jumped substantially above 22.7% (target >50%) OR the G2 GPU-union wall dropped substantially below 909ms toward 510ms; AND the register/unregister tail is eliminated. \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-changed source/test files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (measured before/after overlap % + G2 union wall) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly why.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:persistent-pin', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ PERSISTENT-PIN D2H LANDED ' + verdict.commit_hash + ' — overlap ' + verdict.overlap_pct_after + '% (was 22.7), G2-union ' + verdict.g2_union_wall_ms_after + 'ms (was 909, G1 ' + verdict.g1_union_wall_ms + '), parity=' + verdict.parity_bit_identical_hoststaged)
else log('--- FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, impl, verdict }
