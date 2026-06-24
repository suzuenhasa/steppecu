export const meta = {
  name: 'extract-f2-regimeB-fix',
  description: 'FINISH the decode-seam CRITICAL (audit C2/M3/M4, Stage 2): extract_f2 REGIME-B host SNP-filter -> on-device. Stage 1 (9ad33d9) moved the AUTOSOME-ONLY keep (qpfstats/qpDstat) on-device + kept Q/V/N resident + CUB compaction; it DELIBERATELY left the extract_f2 FULL filter (regime B) on the host because it is FP-sensitive. Regime B (src/app/extract_f2_core.cpp ~150-226 / src/core/internal/snp_filter.cpp): per-SNP keep = autosome + allele-class (multiallelic/strand-ambiguous/transversions, integer-exact char ops) + per-SNP geno missing-frac + pooled FOLDED MAF (Σ_pop Q·N / Σ_pop N -> folded_maf, with is_monomorphic relying on EXACT == 0.0) + the SEPARATE pop-coverage maxmiss (drop if frac of pops with N<=0 > maxmiss). It runs on the HOST per SNP over ~M SNPs x P pops. THE FIX: extend the Stage-1 device-resident decode seam with a REGIME-B on-device keep-mask kernel that reproduces snp_keep_decision EXACTLY (the shared filter_decision.hpp predicates compiled __host__ __device__, the per-SNP across-pop Σ_pop Q·N reduction in the IDENTICAL sequential p=0..P-1 order as snp_filter.cpp so the FP keep decision incl is_monomorphic==0.0 is BIT-IDENTICAL) + CUB DeviceSelect lockstep compaction of Q/V/N/genpos (regime A did Q/V only; extract_f2 needs N too), keeping the result resident into the f2-GEMM. REUSE the Stage-1 infrastructure (decode_af_resident / device_decode_result.hpp / the autosome keep-mask + CUB compaction framework) — EXTEND it, no parallel bolt-on. GOLDEN-EXACT (high risk — bit-exact keep-set or the f2 shifts): a FILTERED extract_f2 (exercise regime B: maf>0 and/or maxmiss>0 and/or transversions) MUST match its reference (AT2 extract_f2 / a direct host-filtered read) at the f2 tier; the default (unfiltered) extract_f2 + qpfstats/qpDstat goldens MUST still hold; CpuBackend==CudaBackend; full STEPPE_THOROUGH ctest green. If the on-device pooled-MAF keep-set cannot be reproduced bit-exactly, HALT + defer. REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host per-SNP filter loop); VERIFY (file:line + the CUB API vs CUDA 13 docs + the exact host predicate); REUSE no parallel bolt-on (extend the Stage-1 seam); native (decode/filter is integer unpack + the AF reduction, FP-deterministic per-SNP-one-thread); SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing. Discipline: diagnose+plan (GATE — the regime-B predicate reproduced bit-exactly on-device + the filtered golden) -> implement (regime-B keep-mask + N compaction, rewire extract_f2) -> verify (filtered + unfiltered extract_f2 + all genotype goldens + ctest) -> commit. FAIL-PROTOCOL: verify->commit on green; on BAD (any golden shifts / keep-set differs) HALT + report + defer; NEVER silently fail/revert; on failure git stash push -u + HALT.',
  phases: [ { title: 'Diagnose regime-B + the filtered golden (GATE: bit-exact on-device keep-set)' }, { title: 'Implement the regime-B keep-mask + N compaction + rewire extract_f2 + build' }, { title: 'Verify filtered + unfiltered extract_f2 + all genotype goldens + commit-or-HALT' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr --exclude atlas_results -e ssh ' + R + '/ box5090:/workspace/steppe/'
const PULL = 'rsync -az -e ssh box5090:/workspace/steppe/tests/reference/goldens/at2/ ' + R + '/tests/reference/goldens/at2/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON -DSTEPPE_BUILD_PYTHON=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -22; " + CORECLEAN + "'"
const STASH = 'cd ' + R + ' && git stash push -u -m'
const TGENO = '/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ 1a2bf5b. The host-compute audit (docs/research/host-compute-audit.md) decode-seam CRITICAL had two regimes: (A) autosome-only (qpfstats/qpDstat) DONE on-device Stage 1 (9ad33d9, decode_af_resident + autosome keep-mask kernel + CUB compaction + device_decode_result.hpp); (B) the extract_f2 FULL filter, STILL ON HOST (deliberately deferred — FP-sensitive). This fixes regime B.',
  'REGIME B (src/app/extract_f2_core.cpp ~150-226 + src/core/internal/snp_filter.cpp + filter_decision.hpp): per-SNP keep = autosome + allele-class (is_multiallelic/is_strand_ambiguous/transversions — integer char ops) + per-SNP geno missing-frac + pooled FOLDED MAF (snp_filter.cpp ~90-104: pooled_ref_af = Σ_pop Q·N / Σ_pop N, sequential p=0..P-1; folded_maf; is_monomorphic via EXACT ==0.0) + the SEPARATE pop-coverage maxmiss (extract_f2_core.cpp ~177-191: drop if frac of pops with N<=0 > maxmiss). Runs on the HOST per SNP.',
  'THE FIX: a REGIME-B on-device keep-mask kernel (one thread per SNP) reproducing snp_keep_decision EXACTLY — the shared filter_decision.hpp predicates compiled __host__ __device__ (already pure/constexpr — same single-source pattern as decode_af.hpp), the per-SNP across-pop Σ_pop Q·N reduction in the IDENTICAL p=0..P-1 order (one SNP per thread => order is bit-identical to the host => the keep decision incl is_monomorphic==0.0 is bit-identical) — + CUB DeviceSelect lockstep compaction of Q/V/N/genpos (Stage 1 compacted Q/V; extract_f2 needs N too). Keep the compacted tensor resident into compute_f2_blocks (the f2-GEMM). EXTEND the Stage-1 seam (decode_af_resident / device_decode_result.hpp), NO parallel bolt-on.',
  'GOLDEN-EXACT (NON-NEGOTIABLE, high blast radius): a FILTERED extract_f2 (regime B: maf>0 and/or maxmiss>0 and/or transversions) MUST match its reference at the f2 tier; the DEFAULT (unfiltered) extract_f2 golden + the qpfstats/qpDstat genotype goldens MUST still hold; CpuBackend==CudaBackend; full STEPPE_THOROUGH ctest green. The on-device keep-set MUST be bit-identical to the host keep-set (same SNPs, same order, same assign_blocks). If it cannot be reproduced bit-exactly, HALT + defer.',
  'STANDING REQUIREMENTS (locked): GPU-FIRST + GPU-BOUND (no host per-SNP filter loop); VERIFY (file:line + the host predicate + the CUB API vs CUDA 13.x docs); REUSE no parallel bolt-on (extend the Stage-1 decode seam); SINGLE-GPU --device 0 (multi-GPU PARKED); CUDA 13+; KEEP ON-DEVICE no warts/bouncing. The decode/filter is integer unpack + the AF reduction (native, per-SNP-one-thread FP-deterministic).',
  'FAIL-PROTOCOL (USER-MANDATED, STRICT): NEVER git checkout -- . / git clean -fd. verify->commit ONLY on green (the filtered + unfiltered extract_f2 + all genotype goldens held + the host SNP-filter gone). On a BAD result (any golden shifts / keep-set differs) HALT + report + DEFER — do NOT silently fail/revert. On any failure ' + STASH + ' "wip:extract-f2-regimeB-FAILED-<reason>" + HALT.',
  'SINGLE-GPU --device 0. RELEASE build-rel. REAL AADR; no synthetic. TGENO ' + TGENO + '. Box ' + SSH + '; nvcc -> ' + PATHENV + '. nothing builds locally.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

phase('Diagnose regime-B + the filtered golden (GATE: bit-exact on-device keep-set)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible','regimeB_spec','stage1_reuse','keepmask_kernel','n_compaction','golden_plan','bit_exact_risk','cuda13','blocker','notes'],
  properties: {
    feasible: { type: 'boolean', description: 'true iff regime-B keep-mask reproduces the host keep-set bit-exactly on-device + N compaction + extract_f2 rewired, preserving the (filtered + unfiltered) extract_f2 + genotype goldens. false -> HALT + defer' },
    regimeB_spec: { type: 'string', description: 'the EXACT host regime-B predicate (autosome + allele-class + missing-frac + pooled-MAF + maxmiss), file:line, incl the across-pop sum order + is_monomorphic==0.0' },
    stage1_reuse: { type: 'string', description: 'how the Stage-1 device-resident decode infra (decode_af_resident / device_decode_result.hpp / the keep-mask + CUB compaction) is EXTENDED for regime B (no bolt-on), file:line' },
    keepmask_kernel: { type: 'string', description: 'the on-device regime-B keep-mask kernel (one thread per SNP, shared filter_decision.hpp predicates __host__ __device__)' },
    n_compaction: { type: 'string', description: 'how N joins the Q/V/genpos lockstep CUB compaction (Stage 1 did Q/V only)' },
    golden_plan: { type: 'string', description: 'the filtered-extract_f2 golden (maf/maxmiss/transversions) + its reference (AT2 extract_f2 / direct host-filtered read), regenerated on box5090' },
    bit_exact_risk: { type: 'string', description: 'the FP risk (pooled-MAF / is_monomorphic) + how the on-device keep-set is guaranteed bit-identical to the host' },
    cuda13: { type: 'string', description: 'the CUB API verified vs CUDA 13.x' },
    blocker: { type: 'string', description: 'if NOT feasible: the blocker (HALT + defer)' },
    notes: { type: 'string' },
  },
}
const design = await tryAgent([
  'You are diagnosing + planning extract_f2 regime-B on-device (verify-before-implement; NO code changes). READ src/app/extract_f2_core.cpp (the regime-B host filter ~150-226) + src/core/internal/snp_filter.cpp + filter_decision.hpp (the predicates) + the Stage-1 device-resident decode seam (decode_af_resident, src/device/device_decode_result.hpp, the autosome keep-mask kernel + CUB compaction in cuda_backend.cu) + the extract_f2 f2-GEMM consumer. Plan the regime-B on-device keep-mask (bit-exact, shared predicates __host__ __device__, the pooled-MAF reduction in host order) + the N compaction + the extract_f2 rewire. VERIFY the CUB API vs CUDA 13.x. On box5090: prepare a FILTERED extract_f2 reference golden (maf>0/maxmiss>0/transversions) + confirm reproducibility; pull ' + PULL + ' . Return the structured plan. If the regime-B keep-set cannot be reproduced bit-exactly on-device, feasible=false + blocker (HALT + defer).', STD,
].join('\n'), { schema: DESIGN_SCHEMA, label: 'diagnose:regimeB', phase: 'Diagnose regime-B + the filtered golden (GATE: bit-exact on-device keep-set)' })
if (design === null) { log('--- diagnose died — HALT'); return { halted: true } }
if (!design.feasible) { log('--- extract_f2 regime-B STRUCTURAL (HALT + defer): ' + design.blocker); return { halted: true, deferred: true, design } }
log('regime-B feasible; reuse: ' + String(design.stage1_reuse).slice(0,140))

phase('Implement the regime-B keep-mask + N compaction + rewire extract_f2 + build')
const fixer = await tryAgent([
  'You are implementing extract_f2 regime-B on-device per this plan (GOLDEN-EXACT; extend the Stage-1 seam, no bolt-on):\n<<<\n' + JSON.stringify(design) + '\n>>>\n\nDo NOT commit. Do NOT git checkout/clean (git stash if forced).', STD, '', 'DEV LOOP: edit locally; ' + RSYNC + '; build+ctest (' + BUILD + ').', '',
  'IMPLEMENT the regime-B on-device keep-mask kernel (one thread per SNP, shared filter_decision.hpp predicates, the pooled-MAF Σ_pop Q·N in host p-order, is_monomorphic==0.0 exact) + the N lockstep CUB compaction (joining Q/V/genpos) + rewire extract_f2_core.cpp to the resident regime-B path (drop the host per-SNP filter + the full D2H). Build + full STEPPE_THOROUGH ctest. SANITY: a FILTERED extract_f2 matches its reference at the f2 tier; the unfiltered extract_f2 + qpfstats/qpDstat goldens hold; CpuBackend==CudaBackend; the host per-SNP regime-B filter no longer runs in production. Report files, the regime-B kernel + N compaction (file:line), the extract_f2 rewire, the golden re-checks, the FULL ctest. Do NOT commit. BAD/NON-trivial (keep-set differs / golden shifts) -> STOP + report (do NOT revert).',
].join('\n'), { label: 'implement:regimeB', phase: 'Implement the regime-B keep-mask + N compaction + rewire extract_f2 + build' })
if (fixer === null) { log('--- fixer died — HALT'); return { halted: true, design } }
await tryAgent(['BUILD-REPAIR for extract_f2 regime-B. Accumulated edits (do NOT clean/revert/destroy; git stash only if forced). Reach a CLEAN Release build + green STEPPE_THOROUGH ctest (extract_f2 + genotype goldens MUST hold), patching only trivial errors / CUB misuse (VERIFY vs CUDA 13). DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 6x. NON-trivial / golden shifts -> STOP + report. Report build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement the regime-B keep-mask + N compaction + rewire extract_f2 + build' })

phase('Verify filtered + unfiltered extract_f2 + all genotype goldens + commit-or-HALT')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','fail_severity','filtered_extract_matches','unfiltered_held','genotype_goldens_held','host_filter_gone','reuses_stage1','goldens_green','commit_hash','stash_ref','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if a filtered extract_f2 (regime B) matches its reference at the f2 tier, the unfiltered extract_f2 + qpfstats/qpDstat goldens hold, the regime-B host SNP-filter is gone (on-device keep-mask + N compaction, resident), extends the Stage-1 seam (no bolt-on), CpuBackend==CudaBackend, ctest green, single-GPU' },
    fail_severity: { type: 'string', description: 'if pass=false: minor or bad (HALT + defer); ANY golden/keep-set drift = bad' },
    filtered_extract_matches: { type: 'boolean' }, unfiltered_held: { type: 'boolean' }, genotype_goldens_held: { type: 'boolean' },
    host_filter_gone: { type: 'boolean', description: 'the regime-B per-SNP host filter no longer runs in the CudaBackend/production path' },
    reuses_stage1: { type: 'boolean', description: 'extends the Stage-1 device-resident decode seam, not a forked path' },
    goldens_green: { type: 'boolean' },
    commit_hash: { type: 'string' }, stash_ref: { type: 'string' },
    note: { type: 'string', description: 'what moved on-device + the goldens checked; on FAIL the blocker + minor-vs-bad' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for extract_f2 regime-B (adversarial; bit-exact keep-set — hold the filtered + unfiltered extract_f2 + the genotype goldens). plan:\n<<<\n' + JSON.stringify(design) + '\n>>>\nimplementer:\n<<<\n' + fixer + '\n>>>', STD, '',
  'DO: (1) git diff review — the regime-B keep-mask + N compaction are on-device (shared predicates, pooled-MAF in host order, no host per-SNP filter loop), extends the Stage-1 seam (no fork), the extract_f2 full D2H dropped. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest green; a FILTERED extract_f2 matches its reference at the f2 tier; the unfiltered extract_f2 + qpfstats/qpDstat goldens hold; CpuBackend==CudaBackend. (3) single-GPU, real AADR. PASS only if the filtered keep-set is bit-exact AND every golden HELD. ',
  'ON PASS: cd ' + R + ' && git add ONLY the new golden + changed source/test/cmake (NEVER git add dot; never aadr/ atlas_results/), commit (perf(decode): extract_f2 regime-B SNP-filter on-device (Stage 2) — pooled-MAF/maxmiss/allele-class keep-mask kernel + N lockstep CUB compaction, bit-exact keep-set; the decode-seam CRITICAL fully closed; goldens held) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Update docs/research/host-compute-audit.md (C2/M3/M4 fully done) + RESUME/TODO. ',
  'ON BAD (keep-set differs / any golden shifts): DO NOT git checkout/clean, DO NOT revert. ' + STASH + ' "wip:extract-f2-regimeB-FAILED" (capture the ref). Classify fail_severity=bad + the blocker for the user. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verify:regimeB', phase: 'Verify filtered + unfiltered extract_f2 + all genotype goldens + commit-or-HALT' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true, design } }
if (verdict.pass) log('+++ extract_f2 regime-B ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- extract_f2 regime-B FAILED [' + verdict.fail_severity + '] — stashed ' + verdict.stash_ref + ' — HALT+defer — ' + verdict.note)
return { design, verdict }
