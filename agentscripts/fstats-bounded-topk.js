export const meta = {
  name: 'fstats-bounded-topk',
  description: 'Fix the f-stat sweep OOM with a BOUNDED device-side TOP-K (the user: "just cap it at ~1m"). PROBLEM (verified): the committed sweep (eda83c5, run_fstat_sweep_device in cuda_backend.cu ~1585-1691) accumulates EVERY survivor in unbounded host std::vectors -> a full C(500,4)=2.57B sweep OOM-kills at 152 GB host RAM (GPU mostly idle, host-RAM-bound), and on real AADR |z|>6 keeps ~70% (NOT a real filter) so ~1.8B survive. The current --top-k is ALSO broken for this (it compacts ALL survivors then host-ranks -> still OOM). FIX: a TRULY BOUNDED top-K — maintain the running top-K device-side via a rising |z| threshold + CUB DeviceRadixSort/DeviceSelect (per chunk: compute all items on-device, select |z| above the running threshold, merge into a bounded buffer of ~2K, periodically CUB-sort + truncate to K + raise the threshold), so HOST RAM stays O(K) (~40 MB at K=1M) regardless of how many BILLIONS are computed. Keep it GPU-BOUND (the compute is still the on-device unrank+assemble_f4_quartets+diagonal-jackknife over ALL C(P,k); only ~K ever come back). Make K (default ~1,000,000) the DEFAULT for --all-quartets so a bare sweep cannot OOM; --top-k overrides K; --min-z still available; --sure still gates the maxcomb ENUMERATION (a TIME guard, since computing 2.57B on the GPU still takes minutes). RESTORE + finish the CLI wiring (it is in stash@{0} from the prior pass: cli_parse/cmd_f4/cmd_f3/cmd_qpdstat/config + cmd_fstat_sweep). VERIFY the CUB DeviceRadixSort/DeviceSelect API against the CUDA 13.x docs. Then the full 2.57B sweep COMPLETES -> WALL-CLOCK it (/usr/bin/time) with nvidia-smi proof -> report wall-clock + quartets/sec + GPU util + bounded RSS + the top-K output. Discipline: design-gate (the bounded-topk algorithm + CUB API + CLI restore) -> implement -> build+golden-gate -> wall-clock the COMPLETED sweep + docs + commit. NO golden regen (explicit-list goldens EXACT). SINGLE-GPU (multi-GPU PARKED), EMULATED-FP64 first, REAL AADR, no synthetic, ZERO hidden CPU compute. FAIL-PROTOCOL: NEVER git checkout/clean; on failure git stash push -u + HALT; verdict classifies severity.',
  phases: [ { title: 'Design the bounded top-K + CUB API + CLI restore (GATE)' }, { title: 'Implement bounded top-K + wire CLI + build + golden-gate' }, { title: 'WALL-CLOCK the COMPLETED 2.57B sweep + GPU-bound proof + docs + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -20 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -25; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'
const F2_500 = '/workspace/data/f2_500'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimpl of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine, main @ eda83c5 (the committed GPU sweep). The CLI wiring from the prior pass is in stash@{0} (wip:sweep-cli-FAILED-full-2.57B-OOM-host-RAM-committed-compute) — RESTORE it (git stash apply, do not drop until verified). The GPU sweep core (run_f4_sweep/run_f3_sweep, on-device unrank + assemble_f4_quartets + diagonal jackknife + |z| filter + CUB compaction) is committed.',
  'THE BUG (verified): run_fstat_sweep_device (src/device/cuda/cuda_backend.cu ~1585-1691) accumulates EVERY survivor in unbounded host std::vectors (h_est/h_se/h_z/h_c0..c3), writing only at the END -> a full C(500,4)=2.57B sweep OOM-kills at 152 GB host RAM (GPU mostly idle, host-RAM-bound). Real AADR is signal-rich: |z|>6 keeps ~70% (C(60,4): 344406/487635), so ~1.8B survive a full sweep -> the |z| filter is NOT a real bound. The current --top-k compacts ALL survivors then host-ranks -> ALSO OOMs.',
  'THE FIX — a TRULY BOUNDED device-side TOP-K: maintain the running top-K (default ~1,000,000) entirely on-device via a rising |z| threshold. Per chunk: the EXISTING on-device unrank+assemble_f4_quartets+diagonal-jackknife computes ALL items (GPU-bound, unchanged); compute |z|; CUB DeviceSelect keeps only items with |z| > the running threshold tau into a bounded device buffer (~2K capacity); when it exceeds ~2K, CUB DeviceRadixSort by |z| desc + truncate to K + raise tau to the new K-th |z|. At the end the device holds exactly the top-K; D2H ONLY those K. HOST RAM = O(K) (~40 MB at K=1M) regardless of billions computed. NEVER an unbounded host vector. Keep the compute on the GPU (emulated-FP64), only ~K come back -> still GPU-bound.',
  'DEFAULTS: --all-quartets/--all-triples DEFAULT to top-K = 1,000,000 (so a bare sweep CANNOT OOM); --top-k INT overrides K; --min-z FLOAT still applies as a pre-filter (raises tau floor); --sure still gates the maxcomb ENUMERATION (a TIME guard — computing 2.57B on the GPU takes minutes; refuse > kFstatMaxComb unless --sure). --shard-dir optional. The result is the K MOST-SIGNIFICANT rows (what is actually useful on signal-rich real data).',
  'VERIFY the CUB DeviceRadixSort + DeviceSelect APIs against the CUDA 13.x docs (CCCL CUB; the two-call temp-storage idiom; 64-bit NumItemsT). Do NOT guess. Reuse the existing CUB usage pattern already in the sweep.',
  'RESTORE the stashed CLI wiring (stash@{0}) and finish it (point --all-quartets at run_f4_sweep with the bounded top-K). Explicit-list mode (--pop1..4/--pops) byte-identical -> goldens EXACT, NO regen.',
  'WALL-CLOCK (the deliverable the user asked for): the full C(500,4)=2,573,031,125-quartet sweep with default top-K=1M now COMPLETES -> on box5090 single-GPU run /usr/bin/time -v ./build-rel/bin/steppe f4 --all-quartets --f2-dir ' + F2_500 + ' --sure (default top-K) while sampling nvidia-smi -> report wall-clock, quartets/sec = 2573031125/seconds, GPU util range, peak VRAM, peak host RSS (must be BOUNDED ~ a few GB, NOT 150GB), the top-K output size. REAL AADR, no synthetic.',
  'FAIL-PROTOCOL (USER-MANDATED): NEVER git checkout -- . / git clean -fd. On ANY failure ' + STASH + ' "wip:bounded-topk-FAILED-<reason>" + HALT. NON-trivial blocker -> STOP + report. Classify minor vs bad.',
  'SINGLE-GPU --device 0 (multi-GPU PARKED). RELEASE build-rel. EMULATED-FP64 first. Box ' + SSH + '; nvcc -> ' + PATHENV + '. ZERO hidden CPU compute for a reported number. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Design the bounded top-K + CUB API + CLI restore (GATE)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','bounded_topk_algorithm','cub_api_check','host_ram_bound','cli_restore','gpu_bound_preserved','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff a device-side bounded top-K (O(K) host RAM, GPU-bound, completes the full 2.57B sweep) is achievable reusing the existing on-device compute + CUB, with goldens exact. false -> structural blocker, HALT' },
    bounded_topk_algorithm: { type: 'string', description: 'the exact device-side bounded-top-K (rising-threshold + CUB DeviceSelect into a ~2K buffer + periodic DeviceRadixSort+truncate-to-K + raise tau); where it replaces the unbounded host vectors in run_fstat_sweep_device (file:line)' },
    cub_api_check: { type: 'string', description: 'CUB DeviceRadixSort + DeviceSelect APIs VERIFIED against CUDA 13.x docs (signatures, two-call temp-storage, 64-bit offsets) + a doc ref' },
    host_ram_bound: { type: 'string', description: 'the proof host RAM is O(K): K=1M * ~40B ~ 40MB + the ~2K device buffer; NO unbounded host vector remains' },
    cli_restore: { type: 'string', description: 'how the stashed CLI wiring (stash@{0}) is restored + pointed at the bounded-top-K sweep; the default top-K=1M for --all-quartets' },
    gpu_bound_preserved: { type: 'string', description: 'why it stays GPU-bound (the unrank+assemble_f4_quartets+jackknife over ALL items is unchanged on-device; only ~K come back; the top-K maintenance is periodic CUB sort, not per-item host work)' },
    blocker: { type: 'string', description: 'if NOT feasible: the structural blocker + options (NO CPU fallback)' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are designing the bounded device-side top-K fix (verify-before-implement; NO code changes). READ: src/device/cuda/cuda_backend.cu run_fstat_sweep_device (~1585-1691 — the unbounded host std::vector accumulation to replace; the existing CUB DeviceSelect usage to mirror), steppe/fstat_sweep.hpp (run_f4_sweep/run_f3_sweep + SweepRequest — the top_k field), the stashed CLI wiring (git stash show -p stash@{0}). VERIFY the CUB DeviceRadixSort + DeviceSelect APIs against the CUDA 13.x CCCL docs (WebSearch/WebFetch). Design the rising-threshold bounded top-K (device buffer ~2K, periodic sort+truncate, raise tau) that keeps HOST RAM O(K) while the full C(P,k) is computed on-device.', STD, '',
  'Return the structured design (feasible + the algorithm + the CUB API check + the host-RAM bound + the CLI restore + why GPU-bound is preserved). If a bounded device-side top-K is infeasible, set feasible=false + blocker (NO CPU fallback). Cite file:line + doc refs. Do NOT implement.',
].join('\n'), { schema: DESIGN_SCHEMA, label: 'design:topk', phase: 'Design the bounded top-K + CUB API + CLI restore (GATE)' })
if (design === null) { log('--- design died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- bounded top-K STRUCTURAL (defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('bounded top-K feasible; host RAM O(K): ' + String(design.host_ram_bound).slice(0,120))

phase('Implement bounded top-K + wire CLI + build + golden-gate')
const fixer = await tryAgent([
  'You are implementing the bounded device-side top-K + restoring the CLI wiring per this design:\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean. RESTORE the stashed CLI: cd ' + R + ' && git stash apply stash@{0} (do NOT drop it).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT: (1) replace the unbounded host std::vector accumulation in run_fstat_sweep_device with the device-side bounded top-K (rising threshold + CUB DeviceSelect into ~2K buffer + periodic DeviceRadixSort+truncate-to-K + raise tau; D2H only the final K). HOST RAM must be O(K). (2) restore + finish the stashed CLI wiring; --all-quartets/--all-triples DEFAULT top-K=1M; --top-k overrides; --min-z pre-filter; --sure gates the enumeration. (3) keep the compute on-device (emulated-FP64), goldens EXACT. Build + full STEPPE_THOROUGH ctest. SANITY: explicit goldens EXACT; a bounded sweep returns the top-K most-significant; a medium real sweep (e.g. C(130,4)) completes with BOUNDED host RSS (not growing with survivors). Report files changed, the bounded-top-K, the CLI, the FULL ctest. Do NOT commit. NON-trivial blocker -> STOP + report.',
].join('\n'), { label: 'implement:topk', phase: 'Implement bounded top-K + wire CLI + build + golden-gate' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for the bounded top-K + CLI. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build + green ctest, patching only trivial -Werror / CMake / CUB-API misuse (VERIFY against CUDA 13 docs). DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 5x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement bounded top-K + wire CLI + build + golden-gate' })

phase('WALL-CLOCK the COMPLETED 2.57B sweep + GPU-bound proof + docs + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','sweep_completes','wall_clock','quartets_per_sec','peak_host_rss','gpu_util','goldens_exact','host_ram_bounded','build_clean','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the FULL C(500,4)=2.57B sweep COMPLETES (no OOM) with BOUNDED host RSS (a few GB, NOT ~150GB), wall-clocked, GPU-bound reconfirmed, the top-K output correct, explicit goldens EXACT, ctest green, single-GPU' },
    fail_severity: { type: 'string' },
    sweep_completes: { type: 'boolean', description: 'the full 2.57B sweep ran to COMPLETION (exit 0, top-K written) — no OOM' },
    wall_clock: { type: 'string', description: 'the measured wall-clock of the COMPLETED full C(500,4)=2.57B sweep' },
    quartets_per_sec: { type: 'string', description: '2573031125 / wall_seconds' },
    peak_host_rss: { type: 'string', description: 'peak host RSS during the full sweep — MUST be bounded (O(K), a few GB), NOT 150GB' },
    gpu_util: { type: 'string', description: 'nvidia-smi GPU util range during the completed sweep (reconfirm GPU-bound)' },
    goldens_exact: { type: 'boolean' }, host_ram_bounded: { type: 'boolean' }, build_clean: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'the COMPLETED full-sweep wall-clock + quartets/sec + GPU util + peak RSS; on FAIL the blocker' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the bounded top-K + the wall-clock (adversarial; the prior pass OOM-killed at 152GB and I over-claimed GPU-bound on a partial window — this time the full 2.57B sweep MUST COMPLETE with bounded RSS, and the wall-clock MUST be of a COMPLETED run). The implementer reported:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — the unbounded host vectors are GONE (device-side bounded top-K), CLI restored, compute on-device, explicit-list untouched. (2) ' + BUILD + ' — ctest green + explicit goldens EXACT. (3) THE WALL-CLOCK (the deliverable): on box5090 single-GPU run /usr/bin/time -v ./build-rel/bin/steppe f4 --all-quartets --f2-dir ' + F2_500 + ' --sure (default top-K=1M) to COMPLETION while sampling nvidia-smi. CONFIRM: it COMPLETES (exit 0, no OOM), peak host RSS is BOUNDED (a few GB, NOT 150GB — independently check time -v Max RSS), the wall-clock + quartets/sec, GPU util range (GPU-bound). If it OOMs or does not complete -> FAIL. (4) single-GPU, real AADR. PASS only if the full sweep COMPLETES bounded + wall-clocked + GPU-bound + goldens exact. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/cli/config/test/doc files (NEVER git add dot; never aadr/ atlas_results/ handoff-*.md; NO golden files change), update docs/perf/fstats-sweep.md + TODO + REORIENT with the COMPLETED full-sweep wall-clock + quartets/sec + GPU util + bounded RSS + the bounded-top-K design + the CLI usage, commit (feat(f-stats): bounded device-side top-K (default 1M) -> the full C(500,4)=2.57B sweep COMPLETES on one 5090 in <X>s (<Y> quartets/s, GPU <util>%, host RSS <Z> bounded), no OOM; CLI --all-quartets wired; goldens exact; single-GPU) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. ',
  'ON FAIL: DO NOT git checkout/clean. ' + STASH + ' "wip:bounded-topk-FAILED" (capture the ref). Classify fail_severity (bad if it still OOMs / does not complete). Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:topk', phase: 'WALL-CLOCK the COMPLETED 2.57B sweep + GPU-bound proof + docs + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ bounded top-K ' + verdict.commit_hash + ' — full 2.57B sweep COMPLETES in ' + verdict.wall_clock + ' (' + verdict.quartets_per_sec + '), GPU ' + verdict.gpu_util + ', RSS ' + verdict.peak_host_rss)
else log('--- bounded top-K FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — ' + verdict.note)
return { design, verdict }
