export const meta = {
  name: 'reduce-kernel-fix',
  description: 'FIX the qpfstats wall — the dstat_block_reduce genotype-f4 kernel is MEMORY-BOUND and dominates the 40-pop qpfstats run (~45s of ~49s, sustained 100% but stalled on memory). ROOT CAUSE (audit + qpfstats-perf verdict, verified): dstat_kernel.cu is one-thread-per-(combo,block); each thread re-reads ITS 4 pops Q/V down the block independently, with ZERO reuse across the ~305k combos -> ~11.6 TB of uncoalesced column-stride global loads at 40 pops. THE FIX (pairwise-difference reuse): every numerator is (Q[a]-Q[b])(Q[c]-Q[d]) over only npop pops => only C(npop,2) distinct per-SNP differences d_ij=Q[i]-Q[j] (+ the pairwise joint-valid mask). Restructure to a SNP-TILED kernel: each CUDA block loads the npop pops Q/V for a tile of SNPs into SHARED MEMORY once, forms the C(npop,2) diffs + pairwise valid masks on-chip, then every combo per-SNP product is two shared-memory diff lookups * mask, accumulated per (combo,block). Converts ~11.6 TB of global re-reads into ONE pass of the ~191MB Q + ~191MB V plus on-chip reuse => COMPUTE-bound not memory-bound; the ~1.77e11 cheap products should run in ~1-2s. SHARED-MEMORY BUDGET: C(40,2)=780 doubles ~6KB (fine); for larger npop (C(60,2)=1770 ~14KB, C(90,2)=4005 ~32KB) tile over pops too / cap the on-chip diff set — the design must handle the budget + a fallback for huge npop. SHARED kernel: dstat_block_reduce serves BOTH qpfstats (~305k combos, the big win) AND qpDstat genotype (few quadruples, must stay correct/neutral) — keep both correct. GOLDEN-EXACT: the qpfstats genotype f2-tensor golden (rtol 1e-6) AND the qpDstat genotype golden MUST still hold + CpuBackend==CudaBackend + full STEPPE_THOROUGH ctest green; the reduction result must be bit-faithful (match the current (a-b)(c-d) accumulation order / tier). REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND; VERIFY (real code file:line + shared-memory/occupancy vs the CUDA 13 docs); REUSE no parallel bolt-on (enhance dstat_block_reduce / its kernel, do not fork a parallel path); native FP64 for the (a-b)(c-d) cancellation accumulation (NOT emulated — same §12 carve-out as the existing kernel); SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing. MEASURE before/after the 40-pop qpfstats with FINE-GRAINED SINGLE-GPU util (nvidia-smi -i 0 -lms 200 — NOT dmon -d 1, which aliases on the 2-GPU box) + nsys for the reduce-kernel time. GOAL: the ~45s reduce phase collapses; the 40-pop wall drops hard. Discipline: diagnose+plan (GATE — the SNP-tiled diff-reuse design + shared-mem budget + golden-preserving accumulation) -> implement (the new tiled reduce kernel) -> verify (goldens + before/after reduce-time/wall + ctest) -> commit. FAIL-PROTOCOL: verify->commit on green; on BAD (goldens shift / not faster / shared-mem blows up for the npop range) HALT + report + defer; NEVER silently fail/revert; on failure git stash push -u + HALT.',
  phases: [ { title: 'Diagnose + design the SNP-tiled diff-reuse reduce kernel (GATE: shared-mem budget + golden-preserving)' }, { title: 'Implement the tiled diff-reuse reduce kernel + build' }, { title: 'Verify goldens + before/after reduce-time + 40-pop wall + commit-or-HALT' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -22; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'
const TGENO = '/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB'
const POPS40 = 'Spanish,Han,YRI,Sweden_Viking,Poland_IA,Czechia_EBA_Unetice,Spain_C,ESN,TSI,GWD,CHS,CHB,JPT,ITU,Austria_N_LBK,PUR,STU,CEU,Russian,Tibetan,LWK,CLM,BEB,ACB,French,FIN,GIH,PJL,KHV,Turkey_N,Druze,Palestinian,Mongol,Kazakh,Sweden_N,Scotland_N,Spain_EBA,England_IA,Denmark_Viking,MSL'
const MEASURE = (out) => SSH + " 'cd /workspace/steppe && " + PATHENV + " && nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader -lms 200 > /tmp/u.log 2>&1 & U=$!; /usr/bin/time -v ./build-rel/bin/steppe qpfstats --prefix " + TGENO + " --pops \"" + POPS40 + "\" --out-dir " + out + " >/dev/null 2>/tmp/qt.log; kill $U 2>/dev/null; echo WALL:; grep Elapsed /tmp/qt.log; echo IDLE_SAMPLES:; awk \"{gsub(/ %/,\\\"\\\");if(\\$1==0)z++}END{print z\\\" of \\\"NR}\" /tmp/u.log'"

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ a81f9ad. qpfstats at 40 pops is ~49s; the device-resident decode seam (9ad33d9) removed the host SNP-filter + the 1.1GB D2H but the wall is dominated by the genotype-f4 REDUCE kernel (~45s of ~49s, sustained 100% but MEMORY-BOUND).',
  'ROOT CAUSE (verified): src/device/cuda/dstat_kernel.cu dstat_block_reduce_kernel is one-thread-per-(combo,block); each thread re-reads its 4 pops Q/V (column-major [P x M], stride P) down the block, ZERO reuse across the ~305k combos -> ~11.6 TB of uncoalesced global loads at 40 pops. Sustained 100% util = SMs stalled on memory, NOT compute.',
  'THE FIX (pairwise-difference reuse, SNP-tiled): each numerator (Q[a]-Q[b])(Q[c]-Q[d]) uses only C(npop,2) distinct per-SNP differences. New kernel: each block loads the npop pops Q/V for a SNP tile into SHARED MEMORY once, forms the C(npop,2) diffs d_ij=Q[i]-Q[j] + the pairwise joint-valid masks on-chip, then each combo per-SNP value = diff[pair1]*diff[pair2]*mask, accumulated per (combo,block). One global pass of ~191MB Q + ~191MB V; the products reuse on-chip -> compute-bound (~1.77e11 cheap products, ~1-2s). SHARED-MEM BUDGET: C(40,2)=780 doubles ~6KB ok; larger npop needs pop-tiling or an on-chip diff cap + a fallback — the design MUST state the budget + the fallback for the npop range steppe supports.',
  'SHARED KERNEL: dstat_block_reduce serves qpfstats (~305k combos — the win) AND qpDstat genotype (few quadruples — must stay correct + not regress). Keep both. Native FP64 for the (a-b)(c-d) accumulation (the §12 cancellation carve-out — NOT emulated). Match the existing accumulation order so the goldens hold.',
  'GOLDEN-EXACT (NON-NEGOTIABLE): the qpfstats genotype f2-tensor golden (rtol 1e-6) AND the qpDstat genotype golden MUST hold + CpuBackend==CudaBackend + full STEPPE_THOROUGH ctest green. If the tiled accumulation cannot reproduce the current result at the tier, HALT + defer.',
  'STANDING REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND; VERIFY (file:line + the shared-memory/occupancy limits vs CUDA 13 docs); REUSE no parallel bolt-on (enhance the existing kernel, no forked path); SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing.',
  'MEASURE before/after with FINE-GRAINED SINGLE-GPU util (nvidia-smi -i 0 --query-gpu=utilization.gpu -lms 200 — NOT dmon -d 1) + nsys for the reduce-kernel time specifically. Report the 40-pop wall + the reduce-kernel ms before vs after.',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER git checkout -- . / git clean -fd. verify->commit ONLY on green (goldens held + the reduce phase materially faster). On a BAD result (goldens shift, not faster, shared-mem infeasible) HALT + report + DEFER — do NOT silently fail/revert. On any failure ' + STASH + ' "wip:reduce-kernel-FAILED-<reason>" + HALT.',
  'SINGLE-GPU --device 0. RELEASE build-rel. REAL AADR; no synthetic. The 40-pop set + TGENO ' + TGENO + '. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Diagnose + design the SNP-tiled diff-reuse reduce kernel (GATE: shared-mem budget + golden-preserving)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','current_kernel','tiled_design','shared_mem_budget','huge_npop_fallback','golden_accumulation','qpdstat_path','cuda13','before_measure','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff the SNP-tiled diff-reuse kernel is buildable within the shared-mem budget for the supported npop range, preserving the qpfstats + qpDstat goldens. false -> HALT + defer' },
    current_kernel: { type: 'string', description: 'the current dstat_block_reduce_kernel structure (file:line) + why it is memory-bound (the per-combo re-read)' },
    tiled_design: { type: 'string', description: 'the SNP-tiled diff-reuse kernel: block/tile shape, the shared-mem diff array, the per-combo product loop, the per-(combo,block) accumulation' },
    shared_mem_budget: { type: 'string', description: 'the shared-mem math (C(npop,2) doubles + masks per tile) for npop=40 and the supported range; occupancy' },
    huge_npop_fallback: { type: 'string', description: 'the fallback/pop-tiling when C(npop,2) exceeds the shared-mem budget (large npop)' },
    golden_accumulation: { type: 'string', description: 'how the tiled accumulation reproduces the current (a-b)(c-d) result at the golden tier (operand order / native FP64)' },
    qpdstat_path: { type: 'string', description: 'how qpDstat genotype (few quadruples) stays correct + non-regressed with the shared kernel' },
    cuda13: { type: 'string', description: 'shared-memory / launch-bounds / occupancy verified vs CUDA 13.x' },
    before_measure: { type: 'string', description: 'the BEFORE 40-pop wall + reduce-kernel ms (nsys) + idle samples' },
    blocker: { type: 'string', description: 'if NOT feasible: the blocker (HALT + defer)' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are diagnosing + designing the pairwise-diff-reuse reduce kernel (verify-before-implement; NO code changes). READ src/device/cuda/dstat_kernel.cu (the current dstat_block_reduce_kernel) + dstat_kernel.cuh + the cuda_backend.cu callers (dstat_block_reduce + the qpfstats fused path) + how Q/V are laid out (column-major [P x M]). Design the SNP-tiled diff-reuse kernel (shared-mem C(npop,2) diffs, per-combo product reuse), state the shared-mem budget for npop=40 + the supported range + the huge-npop fallback, and how the accumulation preserves the qpfstats(rtol 1e-6) + qpDstat goldens. VERIFY shared-mem/occupancy vs CUDA 13.x. RE-MEASURE the BEFORE 40-pop baseline + the reduce-kernel time via nsys (run: ' + MEASURE('/tmp/red_before') + ' ; and an nsys profile of the reduce kernel). Return the structured plan. If the shared-mem budget is infeasible for the npop range OR the goldens cannot be preserved, feasible=false + blocker (HALT + defer).', STD,
].join('\n'), { schema: DESIGN_SCHEMA, label: 'diagnose:reduce', phase: 'Diagnose + design the SNP-tiled diff-reuse reduce kernel (GATE: shared-mem budget + golden-preserving)' })
if (design === null) { log('--- diagnose died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- reduce-kernel STRUCTURAL (HALT + defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('reduce-kernel feasible; budget: ' + String(design.shared_mem_budget).slice(0,120))

phase('Implement the tiled diff-reuse reduce kernel + build')
const fixer = await tryAgent([
  'You are implementing the SNP-tiled pairwise-diff-reuse reduce kernel per this plan (GOLDEN-EXACT; shared kernel for qpfstats + qpDstat):\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT the SNP-tiled diff-reuse reduce kernel (enhance dstat_block_reduce_kernel / its launcher — no forked path): block loads the npop pops Q/V for a SNP tile into shared memory, forms the C(npop,2) diffs + masks, each combo per-SNP value = diff[pair1]*diff[pair2]*mask accumulated per (combo,block), native FP64 matching the current accumulation order; the huge-npop fallback per the plan. Keep qpDstat genotype correct. Build + full STEPPE_THOROUGH ctest. SANITY: qpfstats-geno (rtol 1e-6) + qpdstat-geno goldens hold; CpuBackend==CudaBackend; a 40-pop qpfstats run shows the reduce phase collapsed + materially faster. Report files, the tiled kernel (file:line), the golden re-check, your before/after reduce-time + wall observation. Do NOT commit. BAD/NON-trivial (goldens shift) -> STOP + report (do NOT revert).',
].join('\n'), { label: 'implement:reduce', phase: 'Implement the tiled diff-reuse reduce kernel + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for the reduce-kernel fix. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build + green STEPPE_THOROUGH ctest (qpfstats + qpDstat genotype goldens MUST hold), patching only trivial errors / shared-mem-size / occupancy issues (VERIFY vs CUDA 13 docs). DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 6x. NON-trivial / goldens shift -> STOP + report. Report build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement the tiled diff-reuse reduce kernel + build' })

phase('Verify goldens + before/after reduce-time + 40-pop wall + commit-or-HALT')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','qpfstats_golden_held','qpdstat_golden_held','reduce_compute_bound','before_wall','after_wall','reduce_speedup','goldens_green','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if the qpfstats-geno (rtol 1e-6) AND qpDstat-geno goldens hold, CpuBackend==CudaBackend, ctest green, the reduce kernel is now compute-bound (diff-reuse, not per-combo re-read), the 40-pop wall is materially faster, single-GPU, real AADR' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (HALT + defer); ANY golden shift = bad' },
    qpfstats_golden_held: { type: 'boolean' }, qpdstat_golden_held: { type: 'boolean' },
    reduce_compute_bound: { type: 'boolean', description: 'the reduce loads Q/V once + reuses C(npop,2) diffs on-chip (not ~11.6TB re-reads)' },
    before_wall: { type: 'string', description: 'BEFORE 40-pop wall + reduce-kernel ms (nsys)' },
    after_wall: { type: 'string', description: 'AFTER 40-pop wall + reduce-kernel ms (nsys)' },
    reduce_speedup: { type: 'string', description: 'the reduce-kernel + overall-wall speedup at 40 pops' },
    goldens_green: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'before/after reduce-time + wall; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the reduce-kernel fix (adversarial; hold the qpfstats + qpDstat goldens AND prove the reduce is now compute-bound + faster). plan:\n<<<\n' + JSON.stringify(design) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — the reduce kernel loads Q/V per SNP-tile into shared memory + reuses the C(npop,2) diffs (no per-combo global re-read), native FP64 accumulation, qpDstat path intact, no forked bolt-on. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest green; qpfstats-geno (rtol 1e-6) + qpdstat-geno goldens reproduce; CpuBackend==CudaBackend. (3) THE PERF GATE: re-measure the 40-pop qpfstats AFTER (' + MEASURE('/tmp/red_after') + ') + nsys the reduce kernel; confirm the reduce-kernel time collapsed + the wall materially dropped vs ~49s. (4) single-GPU, real AADR. PASS only if BOTH goldens HELD and the reduce is faster/compute-bound. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/cmake (NEVER git add dot; never aadr/ atlas_results/), commit (perf(reduce): SNP-tiled pairwise-difference-reuse genotype-f4 kernel — load npop pops/SNP once + reuse C(npop,2) diffs on-chip (was ~11.6TB per-combo re-reads); 40-pop qpfstats reduce <before>-><after>; qpfstats + qpDstat goldens held) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs (the qpfstats 40-pop wall on real AADR + host-compute-audit). ',
  'ON BAD (goldens shift / not faster / shared-mem infeasible): DO NOT git checkout/clean, DO NOT revert. ' + STASH + ' "wip:reduce-kernel-FAILED" (capture the ref). Classify fail_severity=bad + the blocker for the user. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:reduce', phase: 'Verify goldens + before/after reduce-time + 40-pop wall + commit-or-HALT' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ reduce-kernel ' + verdict.commit_hash + ' — ' + verdict.reduce_speedup + ' — ' + verdict.note)
else log('--- reduce-kernel FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — HALT+defer — ' + verdict.note)
return { design, verdict }
