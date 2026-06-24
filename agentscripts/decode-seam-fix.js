export const meta = {
  name: 'decode-seam-fix',
  description: 'FIX the CRITICAL host-compute violation from the audit (docs/research/host-compute-audit.md): the genotype DECODE seam does SNP-filtering ON THE HOST and D2Hs the whole decoded tensor — shared by ALL genotype tools (extract_f2, qpDstat, DATES, qpfstats) and the structural cause of the multi-second GPU-idle decode head. C1: decode_af D2Hs ~1.1GB (Q/V/N) to the host. C2: the per-SNP keep-mask (autosome keep + maxmiss + pooled-MAF) is computed on the HOST (~tens of millions of host iters). M3/M4: the autosome/keep lockstep SUBSET of Q/V/genpos is done as host copies, replicated across the 4 consumers. THE FIX (the device-resident decode seam): (1) decode_af keeps Q/V/N RESIDENT in VRAM (no full D2H); (2) an ON-DEVICE per-SNP keep-mask kernel computes the autosome + maxmiss + pooled-MAF predicate; (3) CUB DeviceSelect::Flagged stream-compaction does the lockstep Q/V/genpos subset on-device (the proven sweep pattern); (4) the consumers (extract_f2 f2-GEMM, dstat/qpfstats dstat_block_reduce, dates curve) read the RESIDENT compacted Q/V/genpos — only the small final result crosses to host. GOAL: the GPU-idle decode head shrinks / overlaps, the host SNP-filtering is gone, every genotype tool benefits. GOLDEN-EXACT (NON-NEGOTIABLE, the risk is high — shared seam): ALL genotype-tool goldens MUST still hold at their tiers — qpDstat genotype golden, qpfstats genotype f2-tensor golden (rtol 1e-6), DATES date golden, extract_f2 (f2 == direct read), CpuBackend==CudaBackend, full STEPPE_THOROUGH ctest green. The keep-mask + compaction must reproduce the host SNP-keep set EXACTLY (same SNPs kept, same order) or the goldens shift. REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host per-SNP filter loop); VERIFY (real code file:line + the CUB DeviceSelect API vs CUDA 13 docs); REUSE no parallel bolt-on (the CUB DeviceSelect compaction already exists in the sweep — reuse it); EmulatedFp64 N/A here (decode is integer unpack + the AF reduction, native); SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing. MEASURE before/after with FINE-GRAINED SINGLE-GPU util (nvidia-smi -i 0 -lms 200 — NOT dmon -d 1, which ALIASES on the 2-GPU box). Discipline: diagnose+plan (GATE — map every consumer + the exact host keep-mask to reproduce on-device, confirm golden-preserving) -> implement (resident decode + on-device keep-mask + CUB compaction + rewire consumers) -> verify (ALL genotype goldens + before/after decode util/wall + ctest) -> commit. FAIL-PROTOCOL: verify->commit on green; on BAD (any genotype golden shifts / not faster / still host-filtering) HALT + report + defer; NEVER silently fail/revert; on failure git stash push -u + HALT.',
  phases: [ { title: 'Diagnose the decode seam + map all consumers (GATE: golden-preserving on-device plan)' }, { title: 'Implement: resident decode + on-device keep-mask + CUB compaction + rewire consumers + build' }, { title: 'Verify ALL genotype goldens + before/after decode util/wall + commit-or-HALT' } ],
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
const MEASURE = (out) => SSH + " 'cd /workspace/steppe && " + PATHENV + " && nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader -lms 200 > /tmp/u.log 2>&1 & U=$!; /usr/bin/time -v ./build-rel/bin/steppe qpfstats --prefix " + TGENO + " --pops \"" + POPS40 + "\" --out-dir " + out + " >/dev/null 2>/tmp/qt.log; kill $U 2>/dev/null; echo WALL:; grep Elapsed /tmp/qt.log; echo SINGLE_GPU_UTIL_200ms:; awk \"{gsub(/ %/,\\\"\\\");print}\" /tmp/u.log | tr \"\\n\" \" \"; echo; echo IDLE_SAMPLES:; awk \"{gsub(/ %/,\\\"\\\");if(\\$1==0)z++}END{print z\\\" of \\\"NR}\" /tmp/u.log'"

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ 245b1aa. The host-compute audit (docs/research/host-compute-audit.md, commit 317b645) flagged the genotype DECODE seam as the 2 CRITICAL violations (C1 the ~1.1GB Q/V/N D2H; C2 the per-SNP keep-mask/maxmiss/pooled-MAF on the HOST) + M3/M4 (the autosome/keep lockstep subset as host copies, replicated across the 4 genotype consumers). This is the structural cause of the multi-second GPU-idle decode head measured in qpfstats (and the ~14s at first measure).',
  'THE FIX (the device-resident decode seam): (1) decode_af keeps Q/V/N RESIDENT in VRAM (drop the full host D2H); (2) an ON-DEVICE per-SNP keep-mask kernel computes the predicate the host currently computes (autosome keep + maxmiss + pooled-MAF, matching the host logic EXACTLY); (3) CUB DeviceSelect::Flagged stream-compaction does the lockstep Q/V/genpos subset on-device — REUSE the exact CUB compaction the sweep already uses (cuda_backend.cu run_fstat_sweep_device); (4) the consumers read the RESIDENT compacted Q/V/genpos: the extract_f2 f2-GEMM, the dstat/qpfstats dstat_block_reduce, the dates curve. Only the small final result crosses to host.',
  'GOLDEN-EXACT (NON-NEGOTIABLE — shared seam, high blast radius): ALL genotype-tool goldens MUST still hold — qpDstat genotype golden, qpfstats genotype f2-tensor golden (rtol 1e-6), DATES date golden, extract_f2 (f2==direct read), CpuBackend==CudaBackend, full STEPPE_THOROUGH ctest green. The on-device keep-mask + CUB compaction MUST reproduce the host SNP-keep set EXACTLY (identical SNPs kept, identical order, identical assign_blocks over the kept axis) — any drift shifts every genotype golden. If the keep-set cannot be reproduced bit-exactly on-device, HALT + defer.',
  'STANDING REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host per-SNP filter loop); VERIFY (file:line + the CUB DeviceSelect API vs the CUDA 13.x docs); REUSE no parallel bolt-on (the sweep CUB compaction exists — reuse it; decode_af is already a GPU kernel — extend its residency, do not rewrite); SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing.',
  'MEASURE before/after with FINE-GRAINED SINGLE-GPU util: nvidia-smi -i 0 --query-gpu=utilization.gpu -lms 200 (NOT dmon -d 1 — it ALIASES/aggregates on the 2-GPU box and shows a FALSE 100/0; the qpfstats-perf verdict proved this). Report the 40-pop qpfstats decode-head idle samples + wall before vs after.',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER git checkout -- . / git clean -fd. verify->commit ONLY on green (ALL genotype goldens held + the decode head materially smaller/overlapped). On a BAD result (any golden shifts, not faster, still host-filtering) HALT + report + DEFER — do NOT silently fail/revert. On any failure ' + STASH + ' "wip:decode-seam-FAILED-<reason>" + HALT.',
  'SINGLE-GPU --device 0. RELEASE build-rel. REAL AADR; no synthetic. The 40-pop set + TGENO ' + TGENO + '. Box ' + SSH + '; nvcc -> ' + PATHENV + '. §4 layering. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Diagnose the decode seam + map all consumers (GATE: golden-preserving on-device plan)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','host_filter_spec','consumers','resident_plan','keepmask_kernel','compaction_plan','golden_risk','cuda13_apis','before_measure','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff the host keep-mask can be reproduced bit-exactly on-device + Q/V/genpos kept resident + the 4 consumers rewired, preserving ALL genotype goldens. false -> HALT + defer' },
    host_filter_spec: { type: 'string', description: 'the EXACT host keep-mask logic to reproduce on-device (autosome keep + maxmiss + pooled-MAF), file:line, including the kept-order + assign_blocks dependency' },
    consumers: { type: 'string', description: 'every genotype consumer of the decoded/compacted Q/V/genpos (extract_f2, dstat, qpfstats, dates) + how each is rewired to read the resident compacted tensor, file:line' },
    resident_plan: { type: 'string', description: 'how decode_af keeps Q/V/N resident (drop the full D2H), file:line for the current D2H' },
    keepmask_kernel: { type: 'string', description: 'the on-device per-SNP keep-mask kernel design' },
    compaction_plan: { type: 'string', description: 'the CUB DeviceSelect::Flagged lockstep Q/V/genpos compaction (reusing the sweep CUB pattern), preserving kept-order' },
    golden_risk: { type: 'string', description: 'the risk that any genotype golden shifts + how the bit-exact keep-set is guaranteed' },
    cuda13_apis: { type: 'string', description: 'the CUB DeviceSelect API VERIFIED vs CUDA 13.x docs' },
    before_measure: { type: 'string', description: 'the BEFORE 40-pop qpfstats decode-head idle samples + wall (fine-grained single-GPU)' },
    blocker: { type: 'string', description: 'if NOT feasible: the blocker (HALT + defer)' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are diagnosing + planning the device-resident decode seam fix (verify-before-implement; NO code changes). READ src/core/internal/decode_af.hpp + the decode_af kernel (decode_af_kernel.cu) + the host keep-mask/subset in the genotype tools (src/core/stats/dstat.cpp ~217-270 autosome keep + Q/V/genpos lockstep subset; the same in qpfstats/dates/extract_f2_core.cpp), the maxmiss/pooled-MAF filter logic, assign_blocks over the kept axis, and the EXISTING CUB DeviceSelect compaction in cuda_backend.cu run_fstat_sweep_device. Map EVERY consumer + the EXACT host keep-mask to reproduce on-device. VERIFY the CUB DeviceSelect API vs the CUDA 13.x docs. RE-MEASURE the BEFORE 40-pop baseline (run: ' + MEASURE('/tmp/dec_before') + '). Return the structured plan. If the host keep-set cannot be reproduced bit-exactly on-device (so a golden would shift), feasible=false + blocker (HALT + defer).', STD,
].join('\n'), { schema: DESIGN_SCHEMA, label: 'diagnose:decode', phase: 'Diagnose the decode seam + map all consumers (GATE: golden-preserving on-device plan)' })
if (design === null) { log('--- diagnose died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- decode-seam STRUCTURAL (HALT + defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('decode-seam feasible; consumers: ' + String(design.consumers).slice(0,140))

phase('Implement: resident decode + on-device keep-mask + CUB compaction + rewire consumers + build')
const fixer = await tryAgent([
  'You are implementing the device-resident decode seam per this plan (GOLDEN-EXACT across all 4 genotype tools):\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT: keep decode_af Q/V/N resident (drop the full D2H); an on-device per-SNP keep-mask kernel reproducing the host predicate EXACTLY; CUB DeviceSelect::Flagged lockstep Q/V/genpos compaction (reuse the sweep CUB pattern, preserve kept-order); rewire the consumers (extract_f2, dstat, qpfstats, dates) to read the resident compacted tensor. Build + full STEPPE_THOROUGH ctest. SANITY: ALL genotype goldens still reproduce (qpdstat-geno, qpfstats-geno rtol 1e-6, dates date, extract_f2==read); CpuBackend==CudaBackend; a 40-pop qpfstats run shows the decode head idle samples dropped + faster. Report files, the resident-decode + keep-mask + compaction (file:line), the consumer rewire, the golden re-check, your before/after decode-head observation. Do NOT commit. BAD/NON-trivial (any golden shifts) -> STOP + report (do NOT revert).',
].join('\n'), { label: 'implement:decode', phase: 'Implement: resident decode + on-device keep-mask + CUB compaction + rewire consumers + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for the decode-seam fix. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build + green STEPPE_THOROUGH ctest (ALL genotype goldens MUST hold), patching only trivial errors / CUB API misuse (VERIFY vs CUDA 13 docs). DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 6x. NON-trivial / any golden shifts -> STOP + report. Report build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement: resident decode + on-device keep-mask + CUB compaction + rewire consumers + build' })

phase('Verify ALL genotype goldens + before/after decode util/wall + commit-or-HALT')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','all_genotype_goldens_held','host_filter_gone','decode_resident','before_wall','after_wall','decode_head_improved','goldens_green','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if ALL genotype goldens still hold (qpdstat-geno, qpfstats-geno rtol 1e-6, dates, extract_f2), the host SNP-filter is gone (on-device keep-mask + CUB compaction, Q/V/genpos resident), CpuBackend==CudaBackend, ctest green, the 40-pop decode head materially improved, single-GPU, real AADR' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (HALT + defer); ANY golden shift = bad' },
    all_genotype_goldens_held: { type: 'boolean', description: 'qpDstat + qpfstats + DATES + extract_f2 goldens ALL still match' },
    host_filter_gone: { type: 'boolean', description: 'the per-SNP keep-mask + lockstep subset are now on-device (no host per-SNP loop, no full Q/V/N D2H)' },
    decode_resident: { type: 'boolean' },
    before_wall: { type: 'string', description: 'BEFORE 40-pop wall + decode-head idle samples (fine single-GPU)' },
    after_wall: { type: 'string', description: 'AFTER 40-pop wall + decode-head idle samples (fine single-GPU)' },
    decode_head_improved: { type: 'boolean' },
    goldens_green: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'before/after + what moved on-device + which goldens checked; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the decode-seam fix (adversarial; the blast radius is ALL genotype tools — hold EVERY genotype golden AND the perf). plan:\n<<<\n' + JSON.stringify(design) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — the per-SNP keep-mask + lockstep subset are on-device (CUB compaction, no host per-SNP loop), decode_af Q/V/N resident (no full D2H), the 4 consumers read the resident tensor, reuses the sweep CUB pattern. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest green; ALL genotype goldens reproduce (qpDstat-geno, qpfstats-geno rtol 1e-6, DATES date, extract_f2==read); CpuBackend==CudaBackend. (3) THE PERF GATE: re-measure the 40-pop qpfstats AFTER (' + MEASURE('/tmp/dec_after') + ') vs BEFORE; confirm the decode-head idle samples dropped + wall improved (fine single-GPU, not dmon). (4) single-GPU, real AADR. PASS only if EVERY genotype golden HELD and the decode improved. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/cmake (NEVER git add dot; never aadr/ atlas_results/), commit (perf(decode): device-resident decode seam — on-device per-SNP keep-mask + CUB lockstep compaction, Q/V/genpos kept in VRAM (the ~1.1GB D2H + host SNP-filter gone); shared by extract_f2/qpDstat/DATES/qpfstats; all genotype goldens held) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs/research/host-compute-audit.md (mark C1/C2/M3/M4 done) + RESUME/TODO. ',
  'ON BAD (any genotype golden shifts / not faster / still host-filtering): DO NOT git checkout/clean, DO NOT revert. ' + STASH + ' "wip:decode-seam-FAILED" (capture the ref). Classify fail_severity=bad + the blocker for the user. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:decode', phase: 'Verify ALL genotype goldens + before/after decode util/wall + commit-or-HALT' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ decode-seam ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- decode-seam FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — HALT+defer — ' + verdict.note)
return { design, verdict }
