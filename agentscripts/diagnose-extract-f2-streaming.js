export const meta = {
  name: 'diagnose-extract-f2-streaming',
  description: 'Diagnose why the 1240K big-pop sweep OOMd at 500/700 pops when steppe HAS an M5 streamed-tier path (HostRam/Disk) that tiles the feeder per-block (no 7*P*M wall, per tier_select.hpp streamed_working_set_bytes). The sweep OOMd on the RESIDENT all-M feeder (7*P*M) -> it did NOT engage streaming. ANSWER: (A) CODE TRACE — does cmd_extract_f2 single-GPU compute actually CALL select_output_tier and BRANCH to stream_f2_blocks_impl (the streamed/Disk/HostRam path) for the single-GPU case, or only for multi-GPU / only honor it under --dry-run? Is the streamed path wired for single-GPU extract-f2? What is the STEPPE_FORCE_TIER env / config.force_tier / any --tier CLI knob, and does extract-f2 honor it? Why did select_output_tier pick Resident at 700pops x 1.1M SNPs when the 7*P*M feeder (~43GB) should FAIL the Resident test (result+feeder <= 0.70*free_vram)? (B) EMPIRICAL — re-run 500 + 700 pops single-GPU (--device 0) on the real 1240K, --maxmiss 0.5 (retained ~1.1M SNPs), with: (i) DEFAULT tier, (ii) STEPPE_FORCE_TIER=host, (iii) STEPPE_FORCE_TIER=disk. Report which COMPLETE, wall-clock, peak VRAM, and the output f2.bin size. This decides: did the sweep run the WRONG command (needed force-tier / streaming), or is there an auto-select WIRING GAP. SINGLE-GPU (multi-gpu parked); RELEASE; REAL 1240K; no-hash default. READ-ONLY (no code changes) — diagnose first. Synthesis -> the real 700-pop streaming number + whether it is a command issue or a wiring bug. HALT-on-fail; resumable on 529.',
  phases: [ { title: 'Trace wiring + force-tier empirical runs' }, { title: 'Synthesize: command-issue vs wiring-gap' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const ENSURE_BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -8 && grep -m1 CMAKE_BUILD_TYPE build-rel/CMakeCache.txt'"
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm, box5090 (2x RTX 5090 sm_120, 32GiB ea). The 1240K big-pop sweep (docs/perf/1240k-sweep.md) OOMd at 500/700 pops (maxmiss 0.5, ~1.1M SNPs) on the RESIDENT all-M FP64 feeder (7*P*M). But steppe HAS an M5 streamed-tier path that should avoid this.',
  'THE M5 TIER SYSTEM (src/device/tier_select.hpp): OutputTier { Resident, HostRam, Disk }. Resident holds the [P*P*n_block] f2 result + the all-M feeder (7*P*M doubles) in VRAM. The STREAMED tiers (HostRam/Disk) run stream_f2_blocks_impl which decodes ONE block-tile at a time -> GPU footprint O(P*tile + P^2*nb), NO 7*P*M term (streamed_working_set_bytes) -> they do NOT hit the feeder wall. select_output_tier picks Resident iff result+resident_working_set(P,M) <= 0.70*free_vram, else HostRam (result <= 0.60*free_host_ram), else Disk. STEPPE_FORCE_TIER (resident|host|disk) overrides (host_ram.cpp resolve_output_tier). cmd_extract_f2 includes tier_select.hpp + stream_f2_blocks.hpp.',
  'THE PARADOX: at 700 pops x ~1.1M SNPs the feeder is ~7*700*1.1M*8 = ~43GB, which should FAIL the Resident test (43GB+result > 0.70*32GB) and route to a STREAMED tier. Yet the sweep OOMd on the resident feeder. So either: (a) cmd_extract_f2 single-GPU path does NOT branch to stream_f2_blocks_impl on a streamed tier (only multi-GPU does, or only --dry-run consults the tier), or (b) STEPPE_FORCE_TIER is needed to engage it, or (c) an auto-select bug. DIAGNOSE which.',
  'SINGLE-GPU ONLY (--device 0; multi-gpu PARKED). RELEASE build. REAL 1240K: ' + PREFIX + '. no-hash default. nvcc -> ' + PATHENV + '. Clear cores. Binary ' + BIN + '. READ-ONLY (no code changes) — first diagnose; cite file:line.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Trace wiring + force-tier empirical runs')
const DIAG = [
  { key: 'wiring-trace', focus: 'CODE TRACE (read-only, cite file:line): in src/app/cmd_extract_f2.cpp + the f2-compute it calls (core/fstats/f2_blocks*.hpp / f2_blocks_multigpu.cpp + the CudaBackend f2_blocks entry in src/device/cuda/cuda_backend.cu + stream_f2_blocks_impl + block_sink.cu), determine: (1) does the SINGLE-GPU extract-f2 path CALL select_output_tier and actually BRANCH to stream_f2_blocks_impl (the streamed HostRam/Disk path) when the tier is HostRam/Disk — or does the single-GPU path ALWAYS run the resident all-M feeder (run_f2_blocks_resident) regardless of tier, with the streamed path only reachable via multi-GPU or never wired for single-GPU? (2) where/whether STEPPE_FORCE_TIER / config.force_tier is parsed + honored by cmd_extract_f2 (host_ram.cpp resolve_output_tier). (3) is select_output_tier even CALLED at the compute site, or only for the --dry-run estimate? Pin the exact branch with file:line. This explains the OOM.' },
  { key: 'force-tier-runs', focus: 'EMPIRICAL on box5090, SINGLE-GPU --device 0, REAL 1240K ' + PREFIX + ', --maxmiss 0.5 --blgsize 0.05 --auto-only, no --hash. First ' + RSYNC + ' then ' + ENSURE_BUILD + ' (confirm Release). Then run extract-f2 at 500 pops AND 700 pops (build nested real pop sets >=5 indiv from the .ind, OR reuse the sweep sets) under THREE tier settings each: (i) DEFAULT (no env), (ii) STEPPE_FORCE_TIER=host, (iii) STEPPE_FORCE_TIER=disk. For each: does it COMPLETE (rc=0) or OOM (rc=5)? wall-clock, peak VRAM (background nvidia-smi -l 1), output f2.bin size + SNPs kept. KEY: do the FORCED streamed tiers (host/disk) COMPLETE 700 pops where DEFAULT OOMd? Report the full matrix (P x tier -> complete/OOM, wall, VRAM). rm big f2 dirs after; clear cores. This decides command-issue vs wiring-gap.' },
]
const found = await parallel(DIAG.map(d => () => agent([
  'You are diagnosing the extract-f2 streaming-tier gap, PRECISE + (for the empirical lens) measured: ' + d.focus, '', STD,
  '', 'Cite file:line and report measured rc/wall/VRAM. Clear conclusion for this lens. No fabrication.',
].join('\n'), { label: 'diag:' + d.key, phase: 'Trace wiring + force-tier empirical runs' })))

phase('Synthesize: command-issue vs wiring-gap')
const spec = await agent([
  'You are the lead synthesizer. Combine the wiring trace + the force-tier empirical matrix to answer: did the big-pop sweep run the WRONG command (the streamed tier exists + works for single-GPU extract-f2 but needed STEPPE_FORCE_TIER / a relaxed auto-select), or is there a real WIRING GAP (the single-GPU extract-f2 path never branches to stream_f2_blocks_impl)?', '', STD,
  '', 'THE LENSES:\n' + found.filter(Boolean).map((f,i)=>'### '+DIAG[i].key+'\n'+f).join('\n\n'),
  '', 'Return a crisp verdict: (1) is the M5 streamed path WIRED + WORKING for single-GPU extract-f2? (2) does forcing host/disk COMPLETE 700 pops (the real number: wall + VRAM + f2.bin), or does it still fail? (3) is the DEFAULT auto-select correct (should it have streamed at 700pops without forcing — and if it picked Resident wrongly, what is the bug, file:line)? (4) the corrected bottom line for docs/perf/1240k-sweep.md: the single-GPU 700-pop f2 IS / IS NOT feasible via streaming, and the exact command to do it. Cite file:line + measured numbers. This is the answer to the user question -- are we running the right commands. NO code changes (just the diagnosis + the right command).',
].join('\n'), { label: 'synthesize:streaming', phase: 'Synthesize: command-issue vs wiring-gap' })
log('streaming diagnosis: ' + String(spec).slice(0, 260))
return { spec }
