export const meta = {
  name: 'sweep-1240k-bigpop',
  description: 'Extend the 1240K extract-f2 wall-clock sweep to MANY populations (up to ~700) now that the provenance-SHA bottleneck is removed (no-hash default; extract-f2 ~5.8s at 60 pops). Measure how extract-f2 scales as the f2 tensor goes O(P^2) and #individuals grows. SINGLE-GPU (--device 0; multi-gpu PARKED), RELEASE build, REAL 1240K, no-hash default. Phase 1 SWEEP: build NESTED real pop sets 60/120/250/500/700 (pops with >=5 indiv from the .ind; box has ~926 such); for EACH run extract-f2 --device 0 --blgsize 0.05 --maxmiss 0 --auto-only -> wall-clock + SNPs kept + blocks + #individuals + f2.bin size + peak VRAM. ALSO note the --maxmiss 0 SNP COLLAPSE at high P (zero-missing across 700 pops is brutal) and do ONE 700-pop run at a RELAXED maxmiss (e.g. 0.5) to show the realistic cost when SNPs are retained (so the f2 O(P^2) compute is actually exercised). f2.bin(P)=P^2*711*16B -> 700 pops ~= 5.6GB (fits VRAM 32GB + disk). Phase 2 VERIFY+DOC: confirm Release + single-GPU + real 1240K; spot-check; APPEND the up-to-700 scaling table to docs/perf/1240k-sweep.md (wall vs P, where O(P^2) f2 starts to dominate the read+decode, the maxmiss-0 SNP collapse, peak VRAM, and the headline "700 pops takes X s"). commit. HALT-on-fail; resumable on 529. If extract-f2 OOMs/fails at high P, report exactly where (a valuable finding).',
  phases: [ { title: 'Sweep extract-f2 to 700 pops' }, { title: 'Verify + append to perf doc' } ],
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
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm on box5090 (2x RTX 5090 sm_120, 32 GiB ea). Extending the 1240K extract-f2 perf sweep (docs/perf/1240k-sweep.md @ 0935fd6) to MANY pops. The provenance-SHA bottleneck is FIXED (no-hash is now the default; extract-f2 ~5.8s at 60 pops, was ~44s). Now measure the scaling to ~700 pops.',
  'SINGLE-GPU ONLY (--device 0; MULTI-GPU IS PARKED, memory multi-gpu-parked). RELEASE build (build-rel, -DSTEPPE_BUILD_CLI=ON) — debug voids timing (perf-bench-release-build). NO-HASH default (do NOT pass --hash). REAL 1240K: ' + PREFIX + '.{geno,snp,ind} (TGENO, 1,233,013 SNPs, 23,089 ind). The .ind 3rd column has the pops; ~926 pops have >=5 individuals.',
  'WHAT TO EXPECT / WATCH: f2.bin = P^2 * 711 blocks * 16 bytes -> 60 pops 41MB, 250 ~711MB, 500 ~2.8GB, 700 ~5.6GB (fits VRAM 32GB + disk). The f2 GEMM is O(P^2 * SNPs) — negligible at 60 but should START to show by 250-700. read+decode scales with #individuals (700 pops likely selects most of the 23k). --maxmiss 0 across 700 pops is a brutal joint-completeness filter -> SNPs kept likely COLLAPSES (60 pops already 682k of 1.23M); a collapsed SNP set makes f2 cheap-but-degenerate, so ALSO measure a relaxed-maxmiss 700 run to exercise the real O(P^2) f2.',
  'nvcc -> ' + PATHENV + '. Clear core dumps after each run. /usr/bin/time may be absent (use bash time/date math); od -c for magic. Capture peak VRAM via background nvidia-smi --query-gpu=memory.used --format=csv -l 1. The Release binary: ' + BIN + '.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Sweep extract-f2 to 700 pops')
const SWEEP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','build_type','sweep_table','time_700','snp_collapse','relaxed_700','peak_vram','where_time_goes','notes'],
  properties: {
    done: { type: 'boolean' },
    build_type: { type: 'string', description: 'CMAKE_BUILD_TYPE confirmed Release; single-GPU --device 0; no --hash' },
    sweep_table: { type: 'string', description: 'table: P (60/120/250/500/700) -> #individuals, SNPs kept (maxmiss 0), blocks, f2.bin size, extract-f2 WALL seconds' },
    time_700: { type: 'string', description: 'THE ANSWER: how long extract-f2 takes at 700 pops (maxmiss 0), single-GPU' },
    snp_collapse: { type: 'string', description: 'how SNPs-kept collapses with P under --maxmiss 0 (e.g. 60p=682k -> 700p=?)' },
    relaxed_700: { type: 'string', description: '700-pop run at a relaxed maxmiss (e.g. 0.5): SNPs kept + wall, to show the real O(P^2) f2 cost when SNPs are retained' },
    peak_vram: { type: 'string', description: 'peak VRAM at the heaviest run (700 pops); vs the 32GB/GPU limit' },
    where_time_goes: { type: 'string', description: 'at high P, is it read+decode (individuals) or the O(P^2) f2 GEMM that dominates? infer from the scaling (wall vs P vs SNPs vs #indiv)' },
    notes: { type: 'string', description: 'any OOM/failure/limit; surprises; confirm single-GPU + no-hash + Release' },
  },
}
const sweep = await tryAgent([
  'You are a CUDA performance engineer extending the 1240K extract-f2 sweep to ~700 pops, SINGLE-GPU, no-hash default. NO code changes. Start: ' + RSYNC + ' then ' + ENSURE_BUILD + ' (CONFIRM Release).', STD, '',
  'STEPS: (1) from the .ind, build NESTED real pop sets of 60, 120, 250, 500, 700 pops (each pop >=5 indiv; nest them). (2) For EACH, time extract-f2 (bash time or date math): ' + BIN + ' extract-f2 --prefix ' + PREFIX + ' --pops <set> --out /workspace/data/1240k_big/f2_<P> --device 0 --blgsize 0.05 --maxmiss 0 --auto-only — record wall, SNPs kept, blocks, #individuals, f2.bin size; poll nvidia-smi for peak VRAM on the 700-pop run. (3) ALSO run 700 pops at --maxmiss 0.5 (relaxed) to show SNPs-retained + the real O(P^2) f2 cost. (4) rm the big f2 dirs after (they are GBs); clear cores. Report the structured sweep — THE KEY ANSWER is the 700-pop wall + the scaling shape. If it OOMs/fails at some P, report exactly where (valuable).',
].join('\n'), { schema: SWEEP_SCHEMA, label: 'sweep:bigpop', phase: 'Sweep extract-f2 to 700 pops' })
if (!sweep || !sweep.done) { log('HALT: bigpop sweep failed — ' + (sweep ? sweep.notes : 'agent died')); return { halted: true, sweep } }
log('bigpop sweep: 700p = ' + sweep.time_700 + ' | ' + String(sweep.sweep_table).slice(0,120))

phase('Verify + append to perf doc')
const verdict = await tryAgent([
  'You are the INDEPENDENT VERIFIER for the up-to-700-pop extract-f2 sweep. The sweep agent reported:\n<<<\n' + JSON.stringify(sweep, null, 1) + '\n>>>', STD, '',
  'DO: (1) re-confirm Release + single-GPU (--device 0) + no --hash. (2) spot re-run ONE point (e.g. 250 pops) to confirm the wall is stable. (3) sanity: does the wall scale sensibly (read+decode with #indiv, f2 with P^2*SNPs)? is the maxmiss-0 SNP collapse real? plausible VRAM? (4) flag anything implausible or any silent failure/truncation.',
  'THEN APPEND to ' + R + '/docs/perf/1240k-sweep.md a new section "extract-f2 scaling to 700 pops (single-GPU, no-hash)": the P -> #indiv / SNPs / f2.bin / wall TABLE, the headline 700-pop time, where O(P^2) f2 starts to dominate read+decode, the --maxmiss 0 SNP collapse (+ the relaxed-maxmiss 700 contrast), and peak VRAM vs 32GB. Mark Release/single-GPU/real-1240K/no-hash. Then cd ' + R + ' && git add ONLY docs/perf/1240k-sweep.md (NEVER git add dot; never aadr/), commit (perf: extract-f2 scaling to 700 pops on 1240K — single-GPU, no-hash; O(P^2) f2 + maxmiss-0 SNP collapse) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Return the headline 700-pop time + the scaling shape + the hash.',
].join('\n'), { label: 'verify:bigpop-doc', phase: 'Verify + append to perf doc' })
if (verdict === null) { log('--- verify/doc died — HALT'); return { halted: true, sweep } }
log('+++ bigpop sweep doc: ' + String(verdict).slice(0, 220))
return { sweep, verdict }
