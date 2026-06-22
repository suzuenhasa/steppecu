export const meta = {
  name: 'sweep-1240k-wallclock',
  description: 'Wall-clock timing sweep of steppe on the 1240K panel (the LARGE real target; ~1.2M SNPs, ~23k ind). Measure f2 (extract-f2) AND the other stages (qpAdm fit, S8 rotation) on REAL 1240K data, RELEASE build. THE DATA: /workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB.{geno,snp,ind} (TGENO, 6.7GB geno, on the box, complete). MUST be the Release build (build-rel, -DSTEPPE_BUILD_CLI=ON) — the debug per-kernel cudaDeviceSynchronize voids all timing (memory perf-bench-release-build). REAL data only (no synthetic). Phase 1 SWEEP: confirm steppe reads the 1240K (magic TGENO, n_ind from .ind, n_snp from .snp); then time `extract-f2` across a nested pop-set size sweep (~5, ~15, ~30, ~60 pops drawn from the real .ind, anchored on the Haak set) -> wall-clock + SNPs-kept + blocks + peak VRAM each; then time `qpadm` fit on a resulting f2 dir (single model + a larger nr); then a `qpadm` rotation / multi-model run -> models/sec. Capture wall-clock via /usr/bin/time -v + steppe stdout stage timings + nvidia-smi peak VRAM. Phase 2 VERIFY+DOC: re-confirm the binary is the RELEASE build (not debug) + the data is the real 1240K (not synthetic, not HO) + spot re-time one run for stability + plausibility; write docs/perf/1240k-sweep.md with the per-stage x size table, where-the-time-goes (decode vs f2 GEMM vs fit), VRAM, scaling, and any failure/limit hit at this size. sweep -> verify(+commit the doc). HALT-on-fail; resumable on 529. GPU-only; the 2 RTX 5090 box.',
  phases: [ { title: 'Wall-clock sweep on 1240K' }, { title: 'Verify + write perf doc' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const ENSURE_BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -8 && echo BUILD_TYPE=$(grep -m1 CMAKE_BUILD_TYPE build-rel/CMakeCache.txt)'"
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm, on the 2x RTX 5090 box (box5090, sm_120). Full AT2 parity is DONE; this is a PERFORMANCE wall-clock sweep on the LARGE 1240K panel (the real production target; the HO golden is tiny). The CLI: `steppe extract-f2 --prefix P --pops a,b,.. --out DIR --blgsize 0.05 --maxmiss 0 --auto-only` (genotypes -> f2 dir) then `steppe qpadm --f2-dir DIR --target T --left .. --right ..`.',
  'THE DATA (REAL, on the box, complete): ' + PREFIX + '.{geno,snp,ind} — AADR v66 1240K panel, TGENO format (same as the HO steppe reads natively), .geno 6.7GB, ~1.2M SNPs (wc -l the .snp), ~23k individuals (wc -l the .ind). Pop labels are in the .ind 3rd column. The Haak pops exist here too (Mbuti, Han, Papuan, Karitiana, French, Sardinian, Russia_Samara_EBA_Yamnaya, Turkey_N, Czechia_EBA_CordedWare, Serbia_IronGates_Mesolithic, Israel_Natufian, Iran_GanjDareh_N, Russia_Kostenki_UP, Russia_MA1_HG.SG/Russia_Malta_UP, Russia_UstIshim_IUP, Czechia_BellBeaker, England_BellBeaker).',
  'HARD RULES: RELEASE build ONLY — build-rel with -DSTEPPE_BUILD_CLI=ON; the debug per-kernel cudaDeviceSynchronize VOIDS timing (memory perf-bench-release-build). Confirm BUILD_TYPE=Release before trusting any number. REAL 1240K data ONLY — never synthetic, never the HO panel (this sweep is specifically the 1240K). nvcc -> ' + PATHENV + '. Clear core dumps. box5090 has 2 RTX 5090 (sm_120); note single vs both GPUs used.',
  'MEASURE wall-clock honestly: wrap each run in `/usr/bin/time -v` (Elapsed wall + Max RSS), AND capture steppe stdout (it may print per-stage timings / SNP+block counts), AND peak VRAM (poll `nvidia-smi --query-gpu=memory.used --format=csv -l 1` in the background during a run, or steppe report). Report wall-clock seconds per stage per pop-set size.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Wall-clock sweep on 1240K')
const SWEEP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','build_type','dataset_geometry','extract_f2_sweep','qpadm_timing','rotation_timing','peak_vram','notes'],
  properties: {
    done: { type: 'boolean' },
    build_type: { type: 'string', description: 'the CMAKE_BUILD_TYPE confirmed from build-rel/CMakeCache.txt (MUST be Release) + steppe binary path used' },
    dataset_geometry: { type: 'string', description: '1240K n_snp (wc -l .snp), n_ind (wc -l .ind), .geno magic (TGENO?) — confirm steppe reads it' },
    extract_f2_sweep: { type: 'string', description: 'a table: pop-set size (~5/15/30/60) -> #individuals, SNPs kept, blocks, extract-f2 WALL-CLOCK seconds; note decode-vs-f2 split if steppe prints it' },
    qpadm_timing: { type: 'string', description: 'qpadm fit wall-clock on a 1240K f2 dir (single model + a larger nr if run)' },
    rotation_timing: { type: 'string', description: 'a qpadm rotation / multi-model run wall-clock -> models/sec at 1240K scale' },
    peak_vram: { type: 'string', description: 'peak VRAM used (per GPU) during the heaviest extract-f2; single vs both GPUs' },
    notes: { type: 'string', description: 'where the time goes; scaling with pop count / SNP count; any failure/limit/OOM hit at 1240K size; anything surprising' },
  },
}
const sweep = await tryAgent([
  'You are a CUDA performance engineer running a WALL-CLOCK timing sweep of steppe on the REAL 1240K panel. NO code changes — measure only.', STD, '',
  'STEPS: (1) ' + RSYNC + ' then ' + ENSURE_BUILD + ' — CONFIRM BUILD_TYPE=Release (if not, STOP — debug voids timing). (2) Confirm the 1240K geometry: ' + SSH + " 'wc -l " + PREFIX + ".snp " + PREFIX + ".ind; xxd -l 8 " + PREFIX + ".geno' (magic + counts). (3) From the .ind, build 4 NESTED real pop sets of ~5, ~15, ~30, ~60 pops (each pop with >=5 individuals; anchor on the Haak set, add more real v66 pops). (4) For EACH set: `/usr/bin/time -v " + BIN + " extract-f2 --prefix " + PREFIX + " --pops <set> --out /workspace/data/1240k_sweep/f2_<n> --blgsize 0.05 --maxmiss 0 --auto-only` -> record Elapsed wall, SNPs kept, blocks, MaxRSS; poll nvidia-smi for peak VRAM on the ~60-pop run. (5) `/usr/bin/time -v " + BIN + " qpadm --f2-dir <a 1240k f2 dir> --target <T> --left <a,b> --right <r..>` -> fit wall-clock. (6) a multi-model rotation (qpadm-rotate if built, else loop qpadm over ~20-50 target/source combos) -> models/sec. Clear core dumps after. Report the structured sweep — REAL numbers only, Release build confirmed. If steppe OOMs or fails at 1240K size, report exactly where (that is itself a valuable finding).',
].join('\n'), { schema: SWEEP_SCHEMA, label: 'sweep:1240k', phase: 'Wall-clock sweep on 1240K' })
if (!sweep || !sweep.done) { log('HALT: 1240K sweep failed — ' + (sweep ? sweep.notes : 'agent died')); return { halted: true, sweep } }
log('1240K sweep done: build=' + sweep.build_type + ' | f2: ' + String(sweep.extract_f2_sweep).slice(0,120))

phase('Verify + write perf doc')
const verdict = await tryAgent([
  'You are the INDEPENDENT VERIFIER for the 1240K wall-clock sweep (adversarial about perf hygiene). The sweep agent reported:\n<<<\n' + JSON.stringify(sweep, null, 1) + '\n>>>', STD, '',
  'DO: (1) re-confirm the binary is the RELEASE build: ' + SSH + " 'grep CMAKE_BUILD_TYPE /workspace/steppe/build-rel/CMakeCache.txt; ls -la /workspace/steppe/build-rel/bin/steppe' (Release, not Debug). (2) confirm the data was the REAL 1240K (the " + PREFIX + " prefix, ~1.2M SNPs) NOT the HO panel and NOT synthetic. (3) SPOT RE-TIME one extract-f2 run (e.g. the ~15-pop set) to check the wall-clock is stable/reproducible (within ~20%). (4) sanity: are the numbers plausible (decode of 6.7GB-backed records + f2 over ~1.2M SNPs)? flag anything implausible or any silently-dropped coverage.",
  'THEN write ' + R + '/docs/perf/1240k-sweep.md: the per-stage x pop-set-size wall-clock TABLE (extract-f2 / qpadm fit / rotation), the dataset geometry, where-the-time-goes (decode vs f2 GEMM vs fit), peak VRAM (single vs both GPUs), scaling observations (with pop count + the jump from HO~600K to 1240K~1.2M SNPs), and any limit/OOM/failure hit at 1240K. Mark clearly: Release build, REAL 1240K, box5090 2x RTX 5090. Then: cd ' + R + ' && git add ONLY docs/perf/1240k-sweep.md (create docs/perf/ if needed; NEVER git add dot; never aadr/), commit with a message (perf: 1240K wall-clock sweep — f2 + fit + rotation on real v66 1240K, Release, 2x RTX 5090) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash + return the headline numbers (extract-f2 sec at each size, fit sec, models/sec, peak VRAM).',
].join('\n'), { label: 'verify:perfdoc', phase: 'Verify + write perf doc' })
if (verdict === null) { log('--- verify/doc died — HALT'); return { halted: true, sweep } }
log('+++ 1240K perf doc written: ' + String(verdict).slice(0, 220))
return { sweep, verdict }
