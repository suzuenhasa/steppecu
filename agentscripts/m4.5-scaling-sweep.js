export const meta = {
  name: 'm4.5-scaling-sweep',
  description: 'AT-SCALE scaling sweep on rtxbox (2x RTX PRO 6000, 96GB ea, 169GB host): regenerate AADR derived data at 2500 pops, then an OOM-tolerant ascending P-sweep (768/1200/1600/2000/2500) measuring THREE paths per P — single-GPU, multi-GPU device-resident (the fast combine), multi-GPU host-staged — catching every OOM (device AND host) and reporting exactly where each path hits its ceiling. Answers "can we do 2500 pops on 2 GPUs": it shows single-GPU OOMs early, the fast path tops out ~P2000 (root holds full result+partial), and whether host-staged carries 2500 (devices hold shards, 76GB result in host RAM). Prep (regen-data || harness+build) -> Run -> Report -> docs/cleanup/m4.5/scaling-sweep.md.',
  phases: [
    { title: 'Prep', detail: 'parallel: regenerate derived_2500 data + extend the bench to an OOM-tolerant 3-path sweep, build Release' },
    { title: 'Run', detail: 'run the ascending P-sweep on the box, capture the full scaling table (long-running; detached + poll)' },
    { title: 'Report', detail: 'write the scaling table + ceiling analysis -> docs/cleanup/m4.5/scaling-sweep.md' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh rtxbox'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ rtxbox:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release >/tmp/cfg.log 2>&1 && cmake --build build-rel --target bench_f2_multigpu 2>&1 | tail -25'"

const STD = [
  'PROJECT: steppe = CUDA-13/Blackwell (sm_120) reimpl of ADMIXTOOLS 2 f-statistics, branch m4.5-scaling-sweep (off main, which now has the merged M4.5 multi-GPU). The single-node multi-GPU f2 precompute is DONE + bit-identical + faster at P=768 (1.10x). NOW: stress it at scale (up to 2500 populations) to find where it tops out.',
  'BOX = rtxbox (2x RTX PRO 6000 Blackwell sm_120, CUDA 13). MEASURED: 95.6 GB/GPU (97887 MiB), 169 GB host RAM avail. ' + SSH + '; nvcc not on PATH -> ' + PATHENV + ' . NOTHING builds locally; RELEASE build only (build-rel).',
  'THE MEMORY REALITY (why this is a scaling sweep, not a simple bench): the f2/vpair result is [P^2 * n_block] FP64 each; n_block~757 (full autosome). At P=2500 the result pair is ~76 GB. So: (1) single-GPU needs full result (~76GB) + inputs on ONE device ~110GB => OOMs >96GB; (2) the multi-GPU DEVICE-RESIDENT combine (the fast path, p2p_combine.cu) has the ROOT hold the full result + its own resident partial ~113GB => OOMs at the root (tops out ~P2000); (3) the multi-GPU HOST-STAGED combine keeps only per-device shards (~38GB ea) on-device and the 76GB result in HOST RAM — the path that scales, but at P=2500 host peak ~ inputs(35) + result(76) + both partials(76) may brush the 169GB host ceiling. The sweep EMPIRICALLY finds each ceiling by catching OOM.',
  'AADR has 4266 populations (raw .ind col 3), so 2500 is available. Current derived_full is only P=768. Regen needed: build_tgeno_matrix.py --auto-top 2500 into a NEW dir (do NOT clobber derived_full). The bench repacks subsets of the data DOWN to smaller P, so a P=2500 derived dir serves the whole sweep.',
  'KEY FILES: tests/reference/bench_f2_multigpu.cu (the bench harness to extend), src/device/resources.{hpp,cpp} (Resources + DeviceConfig: prefer_p2p_combine knob, build_resources, device_count), src/core/fstats/f2_blocks_multigpu.cpp (the gate: use_p2p selects device-resident vs host-staged), include/steppe/config.hpp (Precision, prefer_p2p_combine, enable_peer_access). /workspace/data/aadr/build_tgeno_matrix.py (the data generator).',
].join('\n')

phase('Prep')
const prep = await parallel([
  // --- regenerate derived data at 2500 pops ---
  () => agent([
    'You are a data engineer. Regenerate the AADR derived Q/V/N matrices at 2500 populations on rtxbox, into a NEW directory (do not touch derived_full / derived_acc). Box work only.', STD, '',
    'STEPS: (1) ' + SSH + " 'df -h /workspace; ls -la /workspace/data/aadr/raw/; python3 /workspace/data/aadr/build_tgeno_matrix.py --help 2>&1 | head -40'  — confirm >=40GB free, the exact raw geno/ind/snp filenames, and the generator flags. (2) Run the regen (adapt filenames): " + SSH + " '" + PATHENV + " && cd /workspace/data/aadr && nohup python3 build_tgeno_matrix.py --geno raw/<GENO> --ind raw/<IND> [--snp raw/<SNP> if required] --out derived_2500 --auto-top 2500 > /tmp/regen2500.log 2>&1 &' then POLL /tmp/regen2500.log (tail) until done (it may take many minutes — read the log periodically, do not hold one long ssh). (3) Verify: " + SSH + " 'cat /workspace/data/aadr/derived_2500/shape.txt; head -5 /workspace/data/aadr/derived_2500/meta.json; ls -la /workspace/data/aadr/derived_2500/'  — confirm P==2500 (or the achievable max if <2500 pops survive filtering — REPORT the actual P), and the Q/V/N .f64 files exist.",
    'Return: the actual P and M of derived_2500, the disk used, and the absolute data root path to hand the bench. If regen fails (disk, missing flag, fewer pops), report exactly what and the largest P you could build.',
  ].join('\n'), { label: 'prep:regen-data', phase: 'Prep' }),
  // --- extend the bench to the OOM-tolerant 3-path sweep + build ---
  () => agent([
    'You are a senior CUDA/C++ engineer. Extend tests/reference/bench_f2_multigpu.cu into an OOM-TOLERANT ascending scaling sweep over THREE paths per P, then build it Release on rtxbox. EDIT locally + dev-loop on the box; do NOT commit.', STD, '',
    'REQUIRED BEHAVIOR: for each P in the argv list (ascending), measure and print a row with THREE cells: (A) SINGLE-GPU (Resources with device_count==1); (B) MULTI-GPU DEVICE-RESIDENT (G==2, prefer_p2p_combine=true — the fast P2P combine); (C) MULTI-GPU HOST-STAGED (G==2, prefer_p2p_combine=false — forces combine_f2_partials_host). EACH cell wrapped in its OWN try/catch around a FRESHLY-constructed Resources + the compute call: on a CUDA out-of-memory (cudaErrorMemoryAllocation surfaced as the project CudaError) OR std::bad_alloc (the host 76GB result), print "OOM" for that cell and CONTINUE the sweep (do not abort) — construct fresh Resources per cell so a failed alloc does not poison the next; after a caught OOM call cudaGetLastError() to clear sticky state and let RAII free. Print a clean table: P | n_block | result_GB | G1 | G2-resident | G2-hoststaged | speedup(G2res/G1) | speedup(G2host/G1), with OOM cells marked. Use ITERS=3 (median) for the sweep to keep wall-clock sane at large P (note it in the header); a warm-up iter per cell as today. Point at the data root passed as argv[1] (the derived_2500 dir); the bench already repacks derived_full-style subsets DOWN to each P — confirm it handles a data dir whose native P is 2500 and subsets to the requested P (fix the subset/repack if it assumed P<=768). Compute n_block + result_GB from the loaded partition.',
    'Then build: ' + RSYNC + ' && ' + BUILD + ' — must compile clean (warnings-as-errors). Report your git --no-pager diff --stat + the changed hunks + the build result. Do NOT commit. (Coordinate: you own ONLY bench_f2_multigpu.cu; if you need a Resources/config accessor to set prefer_p2p_combine or device_count, use what already exists — build_resources/DeviceConfig — do not change the library.)',
  ].join('\n'), { label: 'prep:harness-build', phase: 'Prep' }),
]).then(a => a)

phase('Run')
const run = await agent([
  'You are a CUDA performance engineer. RUN the at-scale scaling sweep on rtxbox and capture the full table. Box work only; the bench binary (build-rel/bin/bench_f2_multigpu) and the derived_2500 data are ready from the Prep phase.', STD, '',
  'PREP RESULTS:\n<<< REGEN:\n' + (prep[0] || '(regen agent died)') + '\n>>>\n<<< HARNESS+BUILD:\n' + (prep[1] || '(harness agent died)') + '\n>>>', '',
  'Use the derived_2500 data root + the actual P the regen achieved (if regen got < 2500, cap the sweep there and SAY SO). Run the sweep DETACHED on the box (the P=2000/2500 runs are minutes each; do not hold one long ssh) and poll a logfile: ' + SSH + " '" + PATHENV + " && cd /workspace/steppe && nohup ./build-rel/bin/bench_f2_multigpu <DATA_ROOT> 768 1200 1600 2000 2500 > /tmp/sweep.log 2>&1 &' then tail /tmp/sweep.log until it prints the final row / DONE. If a cell hangs (not OOM, genuine hang) for >10min, note it. Capture the COMPLETE table verbatim.",
  'Return: the full verbatim sweep table (every P, all three cells incl. OOM markers), the actual n_block + result_GB per P, and a one-line read of where each path hit its ceiling. Paste the raw bench stdout as evidence — do NOT fabricate any number.',
].join('\n'), { label: 'run:sweep', phase: 'Run' })

phase('Report')
const report = await agent([
  'You are the lead. Write the at-scale scaling sweep results to ' + R + '/docs/cleanup/m4.5/scaling-sweep.md (Write tool). The MEASURED numbers are ground truth.', STD, '',
  'THE SWEEP RESULT:\n<<<\n' + (run || '(run agent died)') + '\n>>>', '',
  'The doc must contain: (1) the verbatim scaling table (P | n_block | result_GB | G1 | G2-resident | G2-hoststaged | speedups, OOM cells marked); (2) THE HEADLINE — can we do 2500 pops on 2 GPUs? which path carries it, and at what P each of the three paths hits its memory ceiling (cite the GB budget: 96GB/GPU, 169GB host, result=2*P^2*n_block*8); (3) the SCALING story — does the device-resident speedup hold/grow with P up to its ceiling; where multi-GPU is ENABLING (single-GPU OOMs but multi-GPU runs) vs FASTER; (4) the implication for design — the device-resident fast path is memory-capped at the root (full result resident), the host-staged path scales further (result in host RAM) but is slower and itself host-RAM-capped; note the obvious next lever for going past the host ceiling (sharded/streamed result, M5 out-of-core). Tag every number to the measured table.',
  'Return a tight 5-8 line executive summary: the max P each path reached, whether 2500 worked and on which path, and the scaling read.',
].join('\n'), { label: 'report:scaling', phase: 'Report' })

return { prep, run, report }
