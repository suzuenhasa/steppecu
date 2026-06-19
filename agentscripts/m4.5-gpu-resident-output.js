export const meta = {
  name: 'm4.5-gpu-resident-output',
  description: 'GET f2_blocks OFF THE CPU. The precompute computes the answer on the GPU (~411ms @P=512) then is FORCED by its return type (host std::vector F2BlockTensor) to copy the whole 6.36GB result into CPU RAM (~1840ms of the ~2253ms wall) — for a host consumer that does not exist yet. This is GPU software; the result must STAY ON THE GPU. FIX: make the precompute produce a DEVICE-RESIDENT result (a VRAM handle), NO forced D2H; the host F2BlockTensor becomes an OPT-IN .to_host()/export used only by tests, the disk cache, or a host caller that explicitly asks. Single-GPU fully device-resident is the headline win on box5090; keep multi-GPU partials resident. Design -> implement on box5090 to green parity -> INDEPENDENT verdict gating on parity bit-identical (device result materialized ONCE for the test) + the device-resident pipeline measured WITHOUT the CPU round-trip (the ~1840ms host tail GONE). HALT-on-fail.',
  phases: [
    { title: 'Design', detail: 'freeze the device-resident output seam: VRAM handle return, opt-in host materialization, no forced D2H' },
    { title: 'Implement', detail: 'coupled core on box5090 to green parity + a bench mode timing the device-resident pipeline' },
    { title: 'Verify', detail: 'independent verdict: parity bit-identical + device-resident pipeline measured without the CPU round-trip; commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -40'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/test_f2_multigpu_parity 2>&1 | tail -50'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell reimpl of ADMIXTOOLS 2 f-statistics. GPU software. M4.5 multi-GPU f2 precompute. f2_blocks[P x P x n_block] FP64 (f2 + Vpair). Branch m4.5-gpu-resident-output (off the d2h-speed work @ 94c6d8e).',
  'THE PROBLEM (measured, the whole point): compute_f2_blocks[_multigpu] computes the answer on the GPU (~411ms GPU-union @P=512) then is FORCED by its RETURN TYPE to dump the entire 6.36GB result into CPU RAM. The return type is F2BlockTensor with std::vector<double> f2/vpair (HOST memory; include/steppe/fstats.hpp:47-60), so the function CANNOT return until the data is in a CPU vector. That host materialization (out.f2.resize(total) zero-fills 6.36GB then it is overwritten — f2_blocks_multigpu.cpp:197-198; + D2H + the staging memcpy) is ~1840ms of the ~2253ms wall @P=512. It exists ONLY to satisfy the host return type; NOTHING currently consumes the host result except the test harness (no fit engine, no qpadm/, no app/ — ROADMAP Phase 2 unbuilt). This is pure waste in GPU software.',
  'THE DIRECTIVE (the user, emphatic): GET IT OFF THE CPU. The result must STAY ON THE GPU. The host return type was a CUDA-free-seam portability DEFAULT (fstats.hpp:10-15), never a requirement — it is the bug.',
  'THE FIX: the precompute produces a DEVICE-RESIDENT result and returns a VRAM HANDLE (no forced D2H, no host alloc/zero/copy on the hot path). The host F2BlockTensor becomes an OPT-IN materialization (e.g. a to_host() on the handle, or an explicit export fn) used ONLY by: the test harness (to D2H once for the memcmp), the future M7 disk cache, and a host/CLI caller that explicitly asks. The device-resident handle already exists internally: DevicePartial (src/device/device_partial.hpp; compute_f2_blocks_resident, cuda_backend.cu:300) — promote a device-resident full result to the PRIMARY output. SINGLE-GPU (G==1) is the clean headline win on box5090 (no P2P): the result is already on the one GPU after the GEMM; just KEEP it resident and return the handle — no D2H at all. For MULTI-GPU: keep each device partial resident (DevicePartial); on a P2P box assemble device-resident (the existing p2p_combine path); on the no-P2P 5090, document that full single-tensor assembly across 2 GPUs still needs P2P or a host bounce, but the per-device partials stay resident (no premature D2H). The headline metric is: the device-resident pipeline runs WITHOUT the ~1840ms CPU round-trip.',
  'PARITY LAW (architecture.md §12, NON-NEGOTIABLE): the COMPUTED bits are unchanged — this only changes WHERE the result lives (VRAM vs a forced host copy) and WHEN it is materialized to host. test_f2_multigpu_parity must stay memcmp BIT-IDENTICAL: it materializes the device-resident result to host ONCE (the opt-in .to_host()) and compares to the single-GPU reference. derived_acc (P=50) + derived_full (P=768, may be VRAM-gated on the 32GB 5090 — note it). No recompute, no reorder.',
  'BOX = box5090 (2x RTX 5090 sm_120, 32GB ea, CONSUMER no-P2P, CUDA 13.0.88), LIVE. ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists. RELEASE only. Single-GPU OOMs ~P700 (use P<=512 for G1; P=256/512 fit). NOTHING builds locally. THE HONEST METRIC: the wall of the device-resident precompute (no host materialization), vs the old host-returning path — show the CPU round-trip is gone.',
  'KEY FILES: include/steppe/fstats.hpp (F2BlockTensor host type — keep as the OPT-IN host export type, do NOT make it the forced return), src/device/backend.hpp (the ComputeBackend seam — the device-resident return), src/device/device_partial.hpp + src/device/cuda/device_partial*.cu* (the VRAM handle, extend to a full result), src/device/cuda/cuda_backend.cu (compute_f2_blocks / _resident / _into — make the resident result the primary, opt-in to_host), src/core/fstats/f2_blocks_multigpu.cpp + f2_blocks_multigpu_core.cpp (orchestrator: return device-resident, materialize host only on request), src/device/cuda/p2p_combine.cu (device-resident assembly on P2P boxes), tests/reference/test_f2_multigpu_parity.cu (materialize-once to compare), tests/reference/bench_f2_multigpu.cu (add a device-resident timing mode that does NOT materialize to host).',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); parity (' + PARITY + '). Iterate until parity BIT-IDENTICAL (the test materializes the device result once). Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA/C++ architect. Produce the EXACT FROZEN contract for making the f2 precompute output DEVICE-RESIDENT (the result stays in VRAM, no forced D2H; host materialization opt-in). READ-ONLY (read why-d2h.md + parallelism-check.md + the key files incl. DevicePartial from 867a4bf; do NOT edit, do NOT touch the box).', STD, '',
  'Specify with file:line + C++ signatures: (1) the device-resident RESULT type — extend DevicePartial or add a DeviceF2Blocks handle owning the full [P^2*n_block] f2/vpair in VRAM + shape; move-only; with an opt-in to_host()->F2BlockTensor (the ONLY place the D2H + host alloc happens); (2) the ComputeBackend seam change — the primary compute returns the device-resident handle; how compute_f2_blocks (host) becomes a thin wrapper = resident-compute + opt-in to_host (so existing host callers/tests still work but the hot path does NOT force it); (3) the orchestrator (f2_blocks_multigpu.cpp) returning device-resident: G==1 returns the one device handle (no D2H — the headline win); G==2 keeps per-device DevicePartials resident, assembles device-resident on a P2P box (p2p_combine), and on the no-P2P 5090 documents the assembly limitation while NOT doing a premature D2H; (4) how the parity test materializes the device result ONCE (.to_host()) to memcmp vs single-GPU; (5) how the bench times the device-resident pipeline WITHOUT host materialization (a new cell/flag) so we can SEE the ~1840ms CPU tail is gone; (6) parity invariants (computed bits unchanged; only residency/timing of materialization changes). The implementer makes NO design decisions. Be explicit about what stays host (the opt-in export, the disk cache, the CLI) vs what becomes device-resident (the precompute->fit handoff).',
].join('\n'), { label: 'design:gpu-resident', phase: 'Design' })

phase('Implement')
const impl = await agent([
  'You are a senior CUDA/C++ engineer. Implement the device-resident output per the FROZEN design and dev-loop on box5090 until parity is bit-identical and a bench mode shows the device-resident pipeline runs WITHOUT the CPU round-trip. EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(missing — derive from why-d2h.md: device-resident handle primary, host opt-in)') + '\n>>>', '', DEVLOOP, '',
  'Prioritize the SINGLE-GPU device-resident path (the clean, unambiguous win on this no-P2P box: compute keeps the result in VRAM, returns the handle, NO D2H) and the opt-in to_host used by the parity test. Keep multi-GPU partials resident. Build Release + ctest + parity (bit-identical; the test materializes once). Then time the device-resident pipeline vs the old host-returning path at P=256/512 (single-GPU) and report the wall WITHOUT host materialization — show the ~1840ms CPU tail is gone (the device-resident compute should be ~hundreds of ms, near the GPU time, not ~2253ms). Report: git --no-pager diff --stat; files changed + what; build/ctest; parity output; the device-resident-vs-host timing. Do NOT commit. If blocked, report exactly what — do NOT fabricate.',
].join('\n'), { label: 'impl:gpu-resident', phase: 'Implement' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','parity_bit_identical','host_roundtrip_eliminated','g1_resident_ms_512','g1_host_ms_512','speedup_vs_host','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real non-empty diff making the precompute output DEVICE-RESIDENT (VRAM handle primary, host F2BlockTensor opt-in via to_host, no forced D2H on the hot path) + Release build clean + ctest green + parity memcmp BIT-IDENTICAL (test materializes once) + the device-resident single-GPU pipeline measured WITHOUT the CPU round-trip (wall near the GPU time, the ~1840ms host tail gone)' },
    diff_is_real: { type: 'boolean', description: 'you re-ran git diff: the result is genuinely returned device-resident (handle), the host alloc/zero/D2H is NOT on the hot path (opt-in to_host only), DevicePartial/DeviceF2Blocks used' },
    parity_bit_identical: { type: 'boolean', description: 'device result materialized once == single-GPU reference, memcmp bit-identical (note derived_full VRAM-gate on 32GB)' },
    host_roundtrip_eliminated: { type: 'boolean', description: 'the device-resident pipeline does NOT do the 6.36GB alloc/zero/D2H/copy on the hot path (verified in the diff + the timing)' },
    g1_resident_ms_512: { type: 'number', description: 'YOU-measured single-GPU DEVICE-RESIDENT precompute wall @P=512 (no host materialization)' },
    g1_host_ms_512: { type: 'number', description: 'the old single-GPU host-returning wall @P=512 (~2655 baseline)' },
    speedup_vs_host: { type: 'number', description: 'g1_host_ms_512 / g1_resident_ms_512 — how much getting it off the CPU saved' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'before/after wall + what stayed host (opt-in export only); for FAIL exactly why' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (maximally adversarial — prior fix-passes have SHAMMED, and reporting has been sloppy: measure the REAL device-resident wall yourself, do NOT trust the report). The implementer reported:\n<<<\n' + (impl || '(impl died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM non-empty AND the precompute genuinely returns a DEVICE-RESIDENT result (VRAM handle) with the host alloc/zero/D2H moved to an OPT-IN to_host (NOT forced on the hot path), public host F2BlockTensor preserved as the export type. If the forced host D2H is still on the hot path => the directive was not met => pass=false. (2) RE-RUN yourself: ' + BUILD + ' ; ' + PARITY + ' ; then time the SINGLE-GPU device-resident precompute @P=512 (no host materialization) vs the old host-returning wall. (3) PASS only if ALL: diff real; build clean; ctest green; parity memcmp BIT-IDENTICAL (test materializes the device result once; note derived_full VRAM-gate on 32GB); AND the device-resident single-GPU wall is near the GPU time (the ~1840ms CPU round-trip is GONE — a large speedup vs the ~2655ms host path). \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-changed source/test files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (measured device-resident vs host wall; what stays host = opt-in export only) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly why.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:gpu-resident', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ GPU-RESIDENT OUTPUT LANDED ' + verdict.commit_hash + ' — single-GPU resident ' + verdict.g1_resident_ms_512 + 'ms vs host ' + verdict.g1_host_ms_512 + 'ms (' + verdict.speedup_vs_host + 'x; CPU round-trip eliminated=' + verdict.host_roundtrip_eliminated + ', parity=' + verdict.parity_bit_identical + ')')
else log('--- FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, impl, verdict }
