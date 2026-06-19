export const meta = {
  name: 'm5-tiered-streaming',
  description: 'M5: ADAPTIVE (not mandatory) tiered f2_blocks output. The result goes to the FASTEST tier it FITS in: (0) fits VRAM -> stay DEVICE-RESIDENT (the existing 3.9x win, NO streaming engaged); (1) big box, fits host RAM -> stream blocks to host RAM; (2) otherwise -> stream blocks to DISK (small pinned staging -> disk file; runs on a laptop, RAM footprint stays tiny). Block-axis streaming (each block independent -> its own [P^2] slab, spillable), triple-buffered compute->D2H->write so the spill overlaps compute (fast at large P). Single-GPU first. Tier auto-selected from free VRAM + free host RAM, with a force-tier override for testing. Design-contract -> coupled implement on box5090 to green parity -> INDEPENDENT verdict gating on: parity bit-identical across tiers (read-back == reference) + P=512 STILL auto-selects device-resident (3.9x preserved) + the disk-tier wall overlaps compute. HALT-on-fail.',
  phases: [
    { title: 'Design', detail: 'freeze the tier-select policy + the pluggable sink + block-stream loop + disk format + parity invariants' },
    { title: 'Implement', detail: 'coupled core on box5090 to green parity (all tiers bit-identical, P=512 stays resident)' },
    { title: 'Verify', detail: 'independent verdict: parity all tiers + resident-tier preserved + disk overlap; commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -45'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/test_f2_multigpu_parity 2>&1 | tail -55'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell reimpl of ADMIXTOOLS 2 f-statistics, precompute-once/fit-many. f2_blocks[P x P x n_block] FP64 (f2 + Vpair), the per-block jackknife tensor. Branch m5-tiered-streaming (off the device-resident output 1f80c0c).',
  'WHERE WE ARE: the precompute now returns a DEVICE-RESIDENT handle (DeviceF2Blocks, src/device/device_f2_blocks.hpp; compute_f2_blocks_device / compute_f2_blocks = device-resident + opt-in .to_host()) — for small P that FITS VRAM this is the fast path (3.9x: 736ms vs 2876ms host-returning @P=512). BUT the result is 2*P^2*n_block*8 bytes (P=512 -> 3.18GB, P=2500 -> 76GB, P=4000 -> 194GB), so at large P it does NOT fit VRAM — and NOT every user has a big box (a normal machine has 16-64GB RAM, not 235GB). So we need OUT-OF-CORE streaming for large P, WITHOUT penalizing the small-P device-resident path.',
  'THE DESIGN (the user directive): ADAPTIVE tiered output — the result lives in the FASTEST tier it FITS in, selected AUTOMATICALLY: (TIER 0 RESIDENT) if the result + working set fits free VRAM -> keep it device-resident exactly as today (NO streaming, the 3.9x win preserved — streaming must be OPT-IN-by-need, never mandatory); (TIER 1 HOST) else if it fits free host RAM -> stream blocks into a host buffer; (TIER 2 DISK) else -> stream blocks to a DISK file via a SMALL persistent pinned staging buffer (device -> few-slab pinned staging -> disk write), so RAM footprint stays tiny and it runs on a laptop. The disk file is the precompute-once/fit-many artifact (the M7-style on-disk f2_blocks cache; what ADMIXTOOLS 2 also does) that the fit later reads tiles from.',
  'BLOCK-AXIS STREAMING (the mechanism for tiers 1+2): f2_blocks is block-major and each block is INDEPENDENT — block b reads only its SNP columns and produces its own [P^2] f2/vpair slab. So stream blocks (or small block-tiles): compute block b on GPU -> its [P^2] slab -> spill via the sink. TRIPLE-BUFFER for speed: compute block b+1 while D2H-ing b while writing b-1, so the spill OVERLAPS compute (at large P the GEMM dominates, so the spill hides -> stays fast). PERSISTENT pinned staging (pin ONCE, reuse — do NOT per-call cudaHostRegister; that was the serialization bug we just fixed in 94c6d8e).',
  'PARITY LAW (architecture.md §12, NON-NEGOTIABLE): block-axis streaming is EXACT by construction — each block is computed identically and independently; streaming only changes WHEN/WHERE a slab is written, never its bits. ALL tiers must produce a result that, read back, is memcmp BIT-IDENTICAL to the in-memory device-resident f2_blocks (and to the single-GPU reference). No recompute, no reorder, no precision change. The combine/compute math is untouched.',
  'BOX = box5090 (2x RTX 5090 sm_120, 32GB ea, CONSUMER no-P2P, CUDA 13.0.88), LIVE, 251GB host RAM / 235 avail (vast instances VARY — read free VRAM via cudaMemGetInfo and free host RAM via sysinfo at runtime, do NOT hardcode). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists. RELEASE only. Single-GPU FIRST (multi-GPU block-sharding is a follow-on). NOTHING builds locally.',
  'KEY FILES: src/device/device_f2_blocks.hpp + src/device/cuda/device_f2_blocks*.cu* (the device-resident handle = TIER 0, reuse), include/steppe/fstats.hpp (F2BlockTensor = the host materialization type), src/device/backend.hpp (the seam), src/device/cuda/cuda_backend.cu (compute_f2_blocks_device / run_f2_blocks_resident — the per-block compute to drive the stream from), src/device/cuda/pinned_buffer.cuh (PinnedBuffer cudaHostAlloc for the staging), src/core/fstats/f2_blocks_multigpu.cpp (the orchestrator — add the tier select + stream loop), src/device/resources.hpp (config / VRAM budget helpers, vram_budget.hpp), tests/reference/test_f2_multigpu_parity.cu (the parity gate — add per-tier read-back checks), tests/reference/bench_f2_multigpu.cu (timing).',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); parity (' + PARITY + '). Iterate until parity BIT-IDENTICAL on all exercised tiers. Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA/C++ architect. Produce the EXACT FROZEN contract for the adaptive tiered f2_blocks output (M5). READ-ONLY (read the device-resident handle from 1f80c0c, the block compute, the VRAM budget helpers, the architecture M5/M7 sections; do NOT edit, do NOT touch the box).', STD, '',
  'Specify with file:line + C++ signatures: (1) the TIER-SELECT policy — a function taking (P, n_block, free_vram, free_host_ram) -> {Resident, HostRam, Disk}: result_bytes = 2*P^2*n_block*8; Resident if result_bytes + the per-call working set fits a fraction of free VRAM; else HostRam if fits a fraction of free host RAM; else Disk. State the headroom factors + how free VRAM (cudaMemGetInfo) and free host RAM (sysinfo/sysconf) are read. PLUS a force-tier OVERRIDE (env var or config field) so tests can exercise Disk/HostRam at small P. (2) the unified RESULT type the precompute returns — an F2BlocksOut that holds EITHER a DeviceF2Blocks (Resident, the existing handle — UNCHANGED path), a host F2BlockTensor (HostRam), or a disk-cache descriptor (Disk: path + header + shape) — with a tile/slab accessor the fit + the parity test use to read it back; (3) the pluggable SINK interface (begin(P,n_block) / spill_block(b, device_f2_slab, device_vpair_slab) / finish()) with 3 impls; Resident BYPASSES the stream entirely (uses the existing device-resident compute); HostRam + Disk use a SMALL persistent pinned staging buffer + the triple-buffered compute->D2H->write pipeline; (4) the DISK file format (binary header {magic,P,n_block,dtype} + block-major [P^2] f2 then vpair, so the fit can pread a block by offset; note AT2-compat as a goal); (5) the block-stream loop driving the per-block compute (reuse run_f2_blocks_resident per block; do NOT recompute the math); (6) how the parity test forces each tier and reads back for memcmp vs the device-resident reference; (7) PARITY invariants (bit-identical, block-axis exact). The implementer makes NO design decisions. Make ABSOLUTELY explicit that TIER 0 (Resident) is the UNCHANGED device-resident path so P=512 keeps its 3.9x and never engages streaming.',
].join('\n'), { label: 'design:tiered-streaming', phase: 'Design' })

phase('Implement')
const impl = await agent([
  'You are a senior CUDA/C++ engineer. Implement the adaptive tiered f2_blocks output per the FROZEN design and dev-loop on box5090 until parity is bit-identical on all tiers AND P=512 still auto-selects the device-resident tier. EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(missing — derive: tier-select VRAM/RAM/disk; block-stream loop; persistent pinned staging; disk file; Resident=existing path unchanged)') + '\n>>>', '', DEVLOOP, '',
  'Build the tier-select + the pluggable sink (Resident bypasses streaming = the existing device-resident path UNCHANGED; HostRam + Disk = block-stream with persistent pinned staging + triple-buffer overlap) + the disk file + the parity-test tier hooks. Build Release + ctest + parity. Exercise EACH tier: confirm (a) P=512 auto-selects Resident (no streaming, 3.9x intact), (b) FORCED HostRam at a small P reads back bit-identical, (c) FORCED Disk at a small P reads back bit-identical, (d) the disk-tier wall overlaps compute (streamed wall ~ compute wall, not serialized). Report: git --no-pager diff --stat; files changed + what; build/ctest; parity output (all tiers); the tier-selection + disk-overlap measurements. Do NOT commit. If blocked, report exactly what — do NOT fabricate.',
].join('\n'), { label: 'impl:tiered-streaming', phase: 'Implement' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','parity_resident','parity_hostram','parity_disk','resident_tier_auto_selected_p512','disk_overlaps_compute','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real non-empty diff implementing the auto tier-select + the block-stream sink (Resident bypasses streaming, HostRam + Disk stream with persistent pinned staging + overlap) + Release build clean + ctest green + parity memcmp BIT-IDENTICAL on Resident AND forced-HostRam AND forced-Disk (read-back == reference) + P=512 auto-selects the Resident tier (3.9x NOT regressed, streaming NOT engaged) + the disk-tier wall overlaps compute' },
    diff_is_real: { type: 'boolean', description: 'you re-ran git diff: tier-select + pluggable sink + block-stream + disk file genuinely present; Resident path unchanged; persistent (not per-call) pinned staging' },
    parity_resident: { type: 'boolean', description: 'Tier 0 device-resident == single-GPU reference, bit-identical (unchanged)' },
    parity_hostram: { type: 'boolean', description: 'forced HostRam tier read-back == reference, memcmp bit-identical' },
    parity_disk: { type: 'boolean', description: 'forced Disk tier (write file, read back) == reference, memcmp bit-identical' },
    resident_tier_auto_selected_p512: { type: 'boolean', description: 'at P=512 the auto tier-select picks Resident (no streaming) and the 3.9x device-resident wall is preserved' },
    disk_overlaps_compute: { type: 'boolean', description: 'the disk-tier streamed wall ~ the compute wall (the write overlaps, not serialized)' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'per-tier parity + the tier-select behavior + the disk-overlap evidence; for FAIL exactly why' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (maximally adversarial — prior fix-passes have SHAMMED, and metrics here are subtle). The implementer reported:\n<<<\n' + (impl || '(impl died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM non-empty AND it genuinely implements: the auto tier-select (VRAM/host-RAM/disk), the pluggable sink (Resident bypasses streaming = existing device-resident path UNCHANGED; HostRam + Disk = block-stream with a PERSISTENT pinned staging buffer, NOT per-call register), the disk file format, the block-stream loop reusing the per-block compute (no math change), and the parity-test tier hooks. If missing/empty/sham => pass=false. (2) RE-RUN yourself: ' + BUILD + ' ; ' + PARITY + ' ; and exercise each tier (force HostRam + Disk at a small P; confirm P=512 auto-selects Resident). (3) PASS only if ALL: diff real; build clean; ctest green; parity memcmp BIT-IDENTICAL on Resident AND forced-HostRam AND forced-Disk (read-back == reference; note derived_full VRAM-gate on 32GB); P=512 auto-selects Resident with the 3.9x intact (streaming NOT engaged); AND the disk-tier wall overlaps compute (not serialized). \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-changed source/test files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (the tiers + per-tier parity + the disk-overlap evidence) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly why.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:tiered-streaming', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ M5 TIERED STREAMING LANDED ' + verdict.commit_hash + ' — parity resident=' + verdict.parity_resident + ' host=' + verdict.parity_hostram + ' disk=' + verdict.parity_disk + '; P512 auto-resident=' + verdict.resident_tier_auto_selected_p512 + '; disk overlaps compute=' + verdict.disk_overlaps_compute)
else log('--- FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, impl, verdict }
