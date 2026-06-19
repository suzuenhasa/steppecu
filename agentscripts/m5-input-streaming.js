export const meta = {
  name: 'm5-input-streaming',
  description: 'COMPLETE M5: SNP-tile INPUT streaming. The current feeder uploads/decodes ALL M SNPs at once (raw 3*P*M + outputs 4*P*M = 7*P*M resident, cuda_backend.cu:171 "does NOT tile — uploads all M"), so full-autosome (M=584k) OOMs the INPUT feeder at ~P512-768 on a 32GB card REGARDLESS of the M5 output streaming (measured: full-autosome sweep OOMs every tier at P>=768 in device_buffer.cuh cudaMalloc = the feeder, NOT the result). FIX: in the block-stream loop, decode/upload ONLY each block-tile s SNP columns from the host [P x M] inputs as that block is processed, so the GPU per-block working set is O(P*tile + P^2) not O(P*M). The full [P x M] inputs stay in HOST RAM (35GB @P=2500, fits); the result streams via the existing M5 output sink. This makes full-autosome ANY-P run on a normal card. Design-contract -> coupled implement on box5090 -> INDEPENDENT verdict gating on: full-autosome (derived_2500, M=584131, n_block=757) COMPLETES at P=1000/1500/2000/2500 (where it OOMd before) + parity bit-identical at P=512 (streamed-input == reference) + the GPU footprint is per-block (no 7*P*M feeder). HALT-on-fail.',
  phases: [
    { title: 'Design', detail: 'freeze the per-block-tile input decode/upload (replace the all-M feeder prologue in the stream loop)' },
    { title: 'Implement', detail: 'coupled core on box5090 to green parity + full-autosome high-P completing' },
    { title: 'Verify', detail: 'independent verdict: full-autosome 1000-2500 COMPLETES + P=512 parity bit-identical + per-block footprint; commit-green / revert+HALT' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -40'"
const PARITY = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/test_f2_multigpu_parity 2>&1 | tail -50'"
// full-autosome sweep on the REAL derived_2500 (P0=2500, M=584131, n_block=757). The win = high-P COMPLETES.
const FASWEEP = SSH + " 'cd /workspace/steppe && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && ./build-rel/bin/bench_f2_multigpu --tiered /workspace/data/aadr 512 1000 1500 2000 2500 2>&1 | grep -vE \"P2P combine unavailable\" | tail -40'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell reimpl of ADMIXTOOLS 2 f-statistics. f2_blocks[P x P x n_block] FP64 (f2+Vpair), per-block jackknife. Branch m5-input-streaming (off m5-tiered-streaming 176a07d, which added the adaptive Resident/HostRam/Disk OUTPUT tiering).',
  'THE PROBLEM (MEASURED, full-autosome derived_2500 M=584131 n_block=757, single-GPU box5090 32GB): the precompute OOMs at P>=768 — and the OOM is the INPUT FEEDER (device_buffer.cuh:74 cudaMalloc), NOT the result. The feeder uploads/decodes ALL M SNPs at once: raw inputs dQ_raw/dV_raw/dN_raw (3*P*M) + persisted feeder outputs (4*P*M) = 7*P*M doubles resident (cuda_backend.cu:171 comment "path does NOT tile — uploads all M"; :256 "One fused feeder over ALL SNPs"; tier_select.hpp resident_working_set = 7*P*M). At M=584k: P=512 feeder 16.7GB (fits), P=768 25GB (+overhead OOMs 32GB), P=1000 32.7GB. So full-autosome caps ~P512-768 on a 32GB card. The M5 OUTPUT streaming (block-stream the result to host/disk) does NOT help because the INPUT feeder OOMs first — the streaming path REUSES the all-M feeder prologue verbatim.',
  'THE FIX: SNP-tile INPUT streaming in the block-stream loop. Each block reads only its own SNP columns (block b spans SNP range [s0,s1), ~M/n_block ~= 772 SNPs at full autosome). So in stream_f2_blocks_impl, decode/UPLOAD ONLY each block-tile s columns of the host [P x M] Q/V/N as that block is processed (a per-block column slice Q[:, s0:s1] etc., P*(s1-s0) each), feed + GEMM + assemble that block s slab, spill it via the existing M5 output sink, then move to the next block — NEVER holding more than one block-tile s inputs (7*P*tile) + its [P^2] slab + scratch on the GPU. The full [P x M] Q/V/N stay in HOST RAM (the caller already passes them as MatView; 3*P*M*8 = 35GB @P=2500/M=584k, fits the 235GB host). GPU per-block working set becomes O(P*tile + P^2) ~= a few hundred MB at P=2500 — INDEPENDENT of M. This removes the feeder wall: full-autosome ANY-P runs on a normal card (GPU-side bounded by one block-tile; host bounded by [P x M] inputs; result by the output tier/disk).',
  'PARITY LAW (architecture.md §12, NON-NEGOTIABLE): each block s f2 is a sum over ITS SNPs only; decoding/feeding per-block-column is EXACT — same SNPs, same feeder math, same GEMM, same bits as the all-M path (which also gathers per-block before the GEMM). Streaming the INPUT changes only WHEN each block s columns are uploaded/decoded, never the values. test_f2_multigpu_parity must stay memcmp BIT-IDENTICAL (at P=512, where a non-streamed reference fits VRAM). No reorder, no precision change.',
  'BOX = box5090 (2x RTX 5090 sm_120, 32GB ea, CONSUMER, CUDA 13.0.88), LIVE, 235GB host RAM free. ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel exists. RELEASE only. Single-GPU. NOTHING builds locally. DATA: /workspace/data/aadr/derived_2500 is REAL full-autosome (P=2500, M=584131, n_block=757, Q/V/N 11.7GB each); the bench --tiered uses it and subsets DOWN per P.',
  'KEY FILES: src/device/cuda/cuda_backend.cu (run_f2_blocks_resident = the all-M feeder prologue at ~:202-256, and stream_f2_blocks_impl = the block-stream loop that must now decode per-block-tile instead of reusing the all-M prologue; the per-bucket gather already slices block columns — reuse that machinery to upload+feed per block), src/device/cuda/decode_af_kernel.cuh + f2_block_kernel.cuh (launch_f2_feeder — it takes Q/V/N + dims; invoke it per block-tile column slice), src/device/stream_f2_blocks.hpp (the streamed-tier seam), src/device/tier_select.hpp (UPDATE resident_working_set for the streamed path — it is now O(P*tile), not 7*P*M; the streamed tiers no longer carry the all-M feeder cost, so high-P no longer needs the resident feeder), src/core/fstats/f2_blocks_multigpu.cpp (orchestrator), tests/reference/test_f2_multigpu_parity.cu, tests/reference/bench_f2_multigpu.cu.',
].join('\n')

const DEVLOOP = 'DEV LOOP: edit locally; rsync (' + RSYNC + '); RELEASE build+ctest (' + BUILD + '); parity (' + PARITY + '); full-autosome sweep (' + FASWEEP + '). Iterate until parity BIT-IDENTICAL @P=512 AND full-autosome P>=1000 COMPLETES (no feeder OOM). Do NOT commit (the verdict commits). Clean revert: ' + CLEAN + '.'

phase('Design')
const design = await agent([
  'You are the lead CUDA/C++ architect. Produce the EXACT FROZEN contract for SNP-tile INPUT streaming in the block-stream loop. READ-ONLY (read run_f2_blocks_resident, stream_f2_blocks_impl, the per-bucket gather, launch_f2_feeder, the decode path; do NOT edit, do NOT touch the box).', STD, '',
  'Specify with file:line + signatures: (1) WHERE the all-M feeder prologue lives in run_f2_blocks_resident (the dQ_raw/dV_raw/dN_raw 3*P*M upload + the launch_f2_feeder over all M -> 4*P*M outputs) and how stream_f2_blocks_impl currently reuses it; (2) the NEW per-block-tile decode/upload: for each block b with SNP range [s0,s1), upload ONLY Q[:,s0:s1]/V/N (P*(s1-s0) each) from the host MatView, run launch_f2_feeder on that column slice, GEMM+assemble that block s [P^2] slab, spill via the existing sink, free the tile s buffers before the next block — reuse the per-bucket gather machinery if it already slices block columns (cite it); (3) the GPU per-block working-set bound (O(P*tile + P^2)) and confirm the host keeps the full [P x M] (the caller s MatView; no change to who owns it); (4) the tier_select.hpp update — the STREAMED tiers no longer carry the 7*P*M feeder, so resident_working_set for the streamed path is O(P*max_tile); the Resident tier (in-VRAM) still uses the all-M path unchanged (small P only); (5) PARITY invariants (per-block decode is exact; same SNPs/feeder/GEMM/bits); (6) how the parity test verifies bit-identity at P=512 (streamed-input path vs the reference) and how the bench confirms high-P completes. The implementer makes NO design decisions. Be explicit that the Resident (small-P) path is UNCHANGED and only the STREAMED path gains per-block input decode.',
].join('\n'), { label: 'design:input-streaming', phase: 'Design' })

phase('Implement')
const impl = await agent([
  'You are a senior CUDA/C++ engineer. Implement SNP-tile INPUT streaming per the FROZEN design and dev-loop on box5090 until parity is bit-identical at P=512 AND full-autosome P>=1000 COMPLETES (the feeder OOM is gone). EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
  'THE FROZEN DESIGN:\n<<<\n' + (design || '(missing — derive: per-block-tile column upload+feed in stream_f2_blocks_impl, host keeps [PxM], GPU per-block O(P*tile+P^2))') + '\n>>>', '', DEVLOOP, '',
  'Build Release + ctest + parity (P=512 bit-identical). Then run the FULL-AUTOSOME sweep (' + FASWEEP + ') and CONFIRM P=1000/1500/2000/2500 now COMPLETE (no device_buffer.cuh feeder OOM) — the result streams to the disk/host tier, the GPU footprint is per-block. Report: git --no-pager diff --stat; files changed + what; build/ctest; parity output; the full-autosome sweep table (which P complete now vs OOM before — P>=768 was ALL OOM before this fix). Do NOT commit. If a high-P still OOMs, report exactly where (feeder? result? host RAM? disk space?) — do NOT fabricate completion.',
].join('\n'), { label: 'impl:input-streaming', phase: 'Implement' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_is_real','parity_bit_identical_p512','max_p_completed','p2500_completes','feeder_wall_removed','sweep_table','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real non-empty diff implementing per-block-tile input decode in the stream loop (NOT the all-M feeder) + Release build clean + ctest green + parity memcmp BIT-IDENTICAL at P=512 (streamed-input == reference) + the full-autosome sweep now COMPLETES at P>=1000 (the feeder OOM that killed P>=768 before is gone) + ideally P=2500 completes' },
    diff_is_real: { type: 'boolean', description: 'you re-ran git diff: the per-block-tile column upload/feed is genuinely there, the all-M feeder is NOT on the streamed hot path; Resident small-P path unchanged' },
    parity_bit_identical_p512: { type: 'boolean', description: 'P=512 streamed-input result == single-GPU reference, memcmp bit-identical' },
    max_p_completed: { type: 'number', description: 'the highest full-autosome P (M=584131, n_block=757) that COMPLETES now (was ~512-768 before)' },
    p2500_completes: { type: 'boolean', description: 'full-autosome P=2500 (76GB result, streamed) completes without OOM' },
    feeder_wall_removed: { type: 'boolean', description: 'the GPU per-block footprint is O(P*tile+P^2) not 7*P*M — no feeder OOM at high P (verified by completion + the diff)' },
    sweep_table: { type: 'string', description: 'the full-autosome sweep table you measured (P | completes? | tier | wall), showing the high-P now completing' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'before (P>=768 all OOM) vs after (max P completing); for FAIL exactly where it still dies (feeder/result/host/disk)' },
  },
}
phase('Verify')
const verdictPrompt =
  'You are the INDEPENDENT VERDICT (maximally adversarial — prior passes have SHAMMED, and the user is furious about half-assed/toy results, so this MUST genuinely complete full-autosome at high P). The implementer reported:\n<<<\n' + (impl || '(impl died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO, yourself: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — CONFIRM non-empty AND it genuinely makes the block-stream loop upload+decode ONLY each block s SNP columns (per-block-tile) instead of the all-M feeder prologue; the host keeps [PxM]; the Resident small-P path unchanged. If still all-M on the streamed path => pass=false. (2) RE-RUN yourself: ' + BUILD + ' ; ' + PARITY + ' ; ' + FASWEEP + ' . (3) PASS only if ALL: diff real; build clean; ctest green; parity memcmp BIT-IDENTICAL at P=512 (streamed-input == reference); AND the full-autosome sweep now COMPLETES at P>=1000 (the feeder OOM that killed every tier at P>=768 before is GONE) — confirm by the actual sweep output (P=1000/1500/2000/2500 produce a wall time / complete, not OOM; note disk space for the 76GB result at P=2500). Record max_p_completed + whether P=2500 completes. \n\nON PASS: cd ' + R + ' && git add ONLY the genuinely-changed source/test files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md); commit with a ROADMAP §6 message (the full-autosome max-P before/after + P=512 parity) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: ' + CLEAN + ' ; report exactly where it still dies.\nReturn the structured verdict.'

async function tryAgent(p, opts) {
  let r = await agent(p, opts)
  if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) }
  return r
}
const verdict = await tryAgent(verdictPrompt, { schema: VERDICT_SCHEMA, label: 'verdict:input-streaming', phase: 'Verify' })

if (verdict && verdict.pass) log('+++ M5 INPUT STREAMING LANDED ' + verdict.commit_hash + ' — full-autosome max P completing=' + verdict.max_p_completed + ' (P=2500 ok=' + verdict.p2500_completes + '), P=512 parity=' + verdict.parity_bit_identical_p512)
else log('--- FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { design, impl, verdict }
