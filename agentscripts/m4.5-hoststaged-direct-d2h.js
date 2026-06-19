export const meta = {
  name: 'm4.5-hoststaged-direct-d2h',
  description: 'SPEED fix for the host-staged multi-GPU combine (the only path on consumer no-P2P boxes, measured SLOWER than single-GPU: 0.71-0.83x on box5090). Each device currently D2H-copies its partial to a SEPARATE pageable host buffer, then a redundant full host copy_n merges them. FIX: pin ONE host result, have each device D2H its compact partial DIRECTLY into its disjoint block-slice of it (concurrent, pinned) — delete the separate partials buffer AND the copy_n. Parity-SAFE (disjoint placement, identical bytes/offsets). Also pin the device-resident final D2H (parity-neutral; perf verified later on the PRO box). Design -> coupled implement (on box5090 to GREEN parity) -> INDEPENDENT verdict (memcmp parity bit-identical + measured G2host BEATS G1). Release build. HALT-on-fail.',
  phases: [
    { title: 'Design', detail: 'freeze the pinned-result + direct-D2H-into-disjoint-slice seam (host-staged), parity-preserving' },
    { title: 'Implement', detail: 'coupled core on box5090 to green parity' },
    { title: 'Verify', detail: 'independent verdict: parity bit-identical + G2host beats G1 (measured); commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/test_f2_multigpu_parity 2>&1 | tail -55'"
// box5090 = consumer, 32GB/GPU, no P2P: single-GPU OOMs ~P700, so compare where BOTH fit (256/512). 768 = G1 OOM (G2 only).
const BENCH = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/bench_f2_multigpu /workspace/data/aadr 256 512 768 2>&1 | grep -vE \"P2P combine unavailable\"'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimpl of ADMIXTOOLS 2 f-statistics. f2_blocks[P x P x n_block] FP64 (the per-block jackknife tensor). Branch m4.5-d2h-speed (off main + the device-resident combine 867a4bf + the scaling harness + the architecture audit).',
  'THE PROBLEM (measured on box5090, 2x RTX 5090, consumer, NO P2P): multi-GPU is SLOWER than single-GPU. G1 vs G2-host-staged: P=256 883 vs 1063ms (0.83x); P=512 2653 vs 3733ms (0.71x); P=768 G1 OOMs, G2host 8007ms. On a consumer no-P2P box the gate degrades the device-resident path to host-staged (G2res==G2host confirmed), so HOST-STAGED is the only multi-GPU path here and it is the slow one. (The rtxbox host-staged was likewise 0.72-0.76x — a real architectural cost, not a box quirk.)',
  'ROOT CAUSE (architecture audit docs/cleanup/m4.5/architecture-audit.md Flaw 3 + why-multigpu-slow.md): the host-staged combine has each device D2H-copy its compact partial to its OWN SEPARATE pageable host buffer (partials[g], allocated inside compute_f2_blocks), and THEN combine_f2_partials_host allocates a full result and does a redundant full host copy_n to merge. So: pageable D2Hs (~4 GB/s vs ~25+ pinned) + an extra full-tensor host copy the single-GPU path never pays.',
  'THE FIX (parity-SAFE by construction): the shards are BLOCK-ALIGNED + DISJOINT (each global block owned by exactly one device; validate_f2_partials checks the tiling covers [0,n_block)). So each device can D2H its compact partial DIRECTLY into its disjoint block-slice [slab*b0, slab*b1) of ONE shared, PINNED host result — no separate partials buffer, NO copy_n. (a) Orchestrator (host-staged branch of f2_blocks_multigpu.cpp) pre-allocates the full result F2BlockTensor with PINNED f2/vpair host storage; (b) each per-device worker D2Hs its compact f2/vpair slabs straight into result.f2.data()+slab*b0 (and vpair) — pinned, concurrent across the 2 worker threads; (c) DELETE combine_f2_partials_host\'s full alloc + the copy_n for this path (block_sizes still placed host-side, fixed g order); (d) the per-device compute must accept a destination host pointer+offset (a host-staged analog of the compute_f2_blocks_resident seam from 867a4bf) OR the worker copies from a pinned staging — prefer the direct-into-destination D2H to avoid any extra copy. ALSO (parity-neutral, perf verified later on the PRO box): pin the device-resident combine\'s final D2H buffer (p2p_combine.cu result vectors) — same pinning primitive.',
  'PINNING: use the existing pinned-buffer machinery (src/device/cuda/pinned_buffer.cuh / the PinnedRegistry the H2D inputs already use, P4/L2) — cudaHostRegister or cudaMallocHost the result f2/vpair once per call. The registration cost is amortized by the D2H bandwidth win (audit: P=768 final D2H 1720ms pageable -> ~143ms pinned). If F2BlockTensor.f2/vpair (std::vector<double>) cannot be pinned in place, register the vector\'s buffer (cudaHostRegister) for the D2H window then unregister — whichever is cleanest; keep the public F2BlockTensor type unchanged for consumers.',
  'PARITY LAW (architecture.md §12, NON-NEGOTIABLE): test_f2_multigpu_parity memcmp G==2 (host-staged) vs single-GPU on derived_acc (P=50) + derived_full (P=768), EmulatedFp64{40}. Direct D2H into disjoint slices moves the SAME doubles to the SAME offsets — bit-identical. Disjoint shards => no overlap, no accumulation, order-independent. NOTE box5090 has NO P2P (can_access_peer=false): the parity test\'s P2P/device-resident assertion is gated/skipped here (T1 VRAM/peer gate) — the HOST-STAGED arm is what must stay bit-identical; confirm last_combine_path=HostStaged on this box.',
  'BOX = box5090 (2x RTX 5090 sm_120, 32GB ea, CONSUMER no-P2P, CUDA 13.0.88). ' + SSH + ' (alias); nvcc not on PATH -> ' + PATHENV + ' . Single-GPU OOMs ~P700 (32GB), so the G1-vs-G2 comparison is at P=256/512 (both fit); at P=768 only G2 runs. build-rel exists. RELEASE only. NOTHING builds locally. The vast box can drop long ssh — for the bench, if a run is long, detach + poll a logfile.',
  'KEY FILES: src/core/fstats/f2_blocks_multigpu.cpp (the host-staged branch + the gate), src/core/fstats/f2_blocks_multigpu_core.cpp (compute_multigpu_partials — the jthread fan-out where each worker D2Hs), src/core/fstats/f2_combine.cpp (combine_f2_partials_host — the alloc + copy_n to delete for this path), src/device/cuda/cuda_backend.cu (compute_f2_blocks + the resident variant from 867a4bf — model the host-staged-direct variant on it; the D2H site), src/device/cuda/pinned_buffer.cuh (pinning), src/device/cuda/p2p_combine.cu (pin its final D2H), include/steppe/fstats.hpp (F2BlockTensor — keep public type stable), src/device/backend.hpp (the ComputeBackend seam), tests/reference/bench_f2_multigpu.cu (the harness), tests/reference/test_f2_multigpu_parity.cu (the gate).',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); parity (' + PARITY + '); bench (' + BENCH + '). Iterate until host-staged parity BIT-IDENTICAL. Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA/C++ architect. Produce the EXACT, FROZEN contract for the host-staged direct-D2H speed fix. READ-ONLY (read the audit doc + the key files incl. how 867a4bf added the resident path; do NOT edit, do NOT touch the box).', STD, '',
  'Specify PRECISELY with file:line + C++ signatures: (1) how the result F2BlockTensor gets PINNED host storage for the D2H (register-in-place vs a pinned buffer the worker D2Hs into then the result owns — keep the public F2BlockTensor type stable for consumers); (2) the per-device compute seam for host-staged-direct — does the worker call a compute variant that D2Hs into a caller-provided host pointer+offset (mirror compute_f2_blocks_resident from 867a4bf), or compute-then-D2H-into-slice in the worker? pick one and give the signature; (3) the exact f2_blocks_multigpu.cpp host-staged-branch rewrite: pre-allocate the pinned result, compute the disjoint offsets (slab*b0 from the shards), fan out so each worker writes its slice, and DELETE the combine_f2_partials_host alloc+copy_n for this path (keep block_sizes placement, fixed g order); (4) what stays for the no-P2P degrade path correctness; (5) the parity-neutral pinning of the device-resident final D2H (p2p_combine.cu) using the same primitive; (6) the lifetime/concurrency/exception-safety plan across the jthread join (two workers D2H into disjoint slices of the same pinned buffer concurrently — safe; no overlap). Flag every spot the result bytes/offsets must stay identical (parity). The implementer makes NO design decisions.',
].join('\n'), { label: 'design:hoststaged-direct', phase: 'Design' })

phase('Implement')
const impl = await agent([
  'You are a senior CUDA/C++ engineer. Implement the host-staged direct-D2H speed fix per the FROZEN design and dev-loop on box5090 until host-staged parity is memcmp BIT-IDENTICAL and the bench shows G2host improved. EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(design missing — derive from audit Flaw 3 + 867a4bf resident pattern)') + '\n>>>', '', DEVLOOP, '',
  'Build Release + ctest + run test_f2_multigpu_parity (confirm host-staged == single-GPU bit-identical on BOTH datasets; last_combine_path=HostStaged on this no-P2P box). Then run the bench (256 512 768) to confirm G2host dropped (target: G2host BEATS G1 at P=256/512 where G1 fits — was 0.83x/0.71x). Report: git --no-pager diff --stat; files changed + what; FULL build result; ctest summary; the parity output; the bench table. Do NOT commit. If blocked, report exactly what — do NOT fabricate.',
].join('\n'), { label: 'impl:hoststaged-direct', phase: 'Implement' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass', 'diff_is_real', 'parity_bit_identical_hoststaged', 'g1_512_ms', 'g2host_512_ms', 'g2host_beats_g1', 'g2host_256_ms', 'g1_256_ms', 'commit_hash', 'note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real non-empty diff implementing the pinned direct-D2H host-staged combine (separate partials + copy_n DELETED for this path) + Release build clean + ctest green + host-staged parity memcmp BIT-IDENTICAL (both datasets) + the measured G2host at P=512 dropped meaningfully and BEATS (or at least ties within ~5%) G1' },
    diff_is_real: { type: 'boolean', description: 'you re-ran git diff yourself: the pinned result + direct-D2H-into-disjoint-slice are present, the copy_n merge is gone for the host-staged path, public F2BlockTensor type unchanged' },
    parity_bit_identical_hoststaged: { type: 'boolean' },
    g1_512_ms: { type: 'number' }, g2host_512_ms: { type: 'number', description: 'YOU-measured G2 host-staged @ P=512 (baseline was 3733ms / 0.71x)' },
    g2host_beats_g1: { type: 'boolean', description: 'true if g2host_512_ms < g1_512_ms (multi-GPU finally faster than single-GPU on this box)' },
    g1_256_ms: { type: 'number' }, g2host_256_ms: { type: 'number' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'before/after G2host vs G1 @ 256 & 512 + the speedup; for FAIL, exactly why (parity break / no improvement / sham)' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (maximally adversarial — prior fix-passes on this project have SHAMMED). The implementer reported:\n<<<\n' + (impl || '(impl agent died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself (do NOT trust the report): (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM non-empty AND genuinely contains the pinned result + each device D2H direct into its disjoint slice + the host-staged copy_n/separate-partials merge DELETED (public F2BlockTensor type stable). If missing/empty => SHAM => pass=false. (2) RE-RUN the box yourself: ' + BUILD + ' ; ' + PARITY + ' ; ' + BENCH + ' . (3) PASS only if ALL: diff real; Release build clean; ctest green; host-staged parity test_f2_multigpu_parity memcmp BIT-IDENTICAL on both datasets (last_combine_path=HostStaged on this no-P2P box); AND the YOU-measured G2host@512 dropped meaningfully vs the 3733ms baseline AND now beats (or ties within ~5% of) G1@512 (was 2653ms). \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-changed source/test files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (measured before/after G2host vs G1 @ 256/512) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly why.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:hoststaged-direct', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ HOST-STAGED DIRECT-D2H LANDED ' + verdict.commit_hash + ' — G2host@512 ' + verdict.g2host_512_ms + 'ms vs G1 ' + verdict.g1_512_ms + 'ms (beats_g1=' + verdict.g2host_beats_g1 + ', parity=' + verdict.parity_bit_identical_hoststaged + ')')
else log('--- FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, impl, verdict }
