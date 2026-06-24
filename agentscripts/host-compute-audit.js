export const meta = {
  name: 'host-compute-audit',
  description: 'AUDIT (read-only, NO code changes) the ENTIRE steppe production compute surface for HOST-SIDE compute that violates the GPU-first/GPU-bound/keep-on-device mandate — triggered because the qpfstats block-jackknife was found running on the CPU (a host long-double per-combo loop, ~217M iterations at 40 pops, the GPU 50% idle) DESPITE explicit GPU-only instructions. FIND EVERY OTHER INSTANCE. THE OFFENDER PATTERN: host-side FLOATING-POINT / statistical compute (reductions, jackknife, covariance, regression, normalization, per-item/per-block/per-SNP/per-combo/per-pair/per-iteration arithmetic) that runs in the PRODUCTION path (the run_* orchestration in src/core + src/app, and anything dispatched to CudaBackend) and SCALES WITH THE DATA (M SNPs, n_block, N items, npopcomb, npairs) — i.e. it should be a ComputeBackend GPU kernel but is being done on the host, serializing with / instead of the GPU. EXEMPT (NOT violations, do not flag): the CpuBackend (src/device/cpu/*) which is the TEST-ONLY parity oracle (memory cpu-is-test-only); CLI/arg/config parsing; ONE-TIME O(npop)/O(small) integer index/setup-array builds; IO read orchestration (the unavoidable genotype/f2 read, though FLAG if the decode/unpack itself is a slow host hot loop that could be on-device); tiny final result formatting/CSV emit. THE DISCRIMINATOR: does host arithmetic scale with the data dimension AND compete with / serialize against the GPU compute? If yes -> VIOLATION. Audit EVERY tool: the qpAdm/qpWave fit + the S8 rotation; f4/f3/f4-ratio + their (diag/ratio) jackknife; qpDstat A + B + dstat_jackknife; the all-quartets sweep; qpfstats (confirm the known jackknife offender + find more); qpGraph; DATES + its jackknife; extract_f2 + decode_af/het-correction; and the SHARED jackknife family (jackknife_cov, jackknife_diag, dstat_jackknife, ratio_jackknife, matrix_jackknife_est_col) + the core-vs-backend dispatch boundary (what numerical compute lives in src/core that should be a backend kernel). For each offender: file:line, what it computes, hot-path Y/N, the dimension it scales with, the scale at which it bites (does it matter at the 9-pop golden? at 40+ pops? at sweep scale?), severity, and the on-device fix. Each auditor also returns the CLEAN parts (confirmed on-device) so we get a clean bill. VERIFY by reading the real code (file:line) — do not guess. Synthesis writes a RANKED audit report (docs/research/host-compute-audit.md) + commits. NO fixes in this workflow (audit only; fixes follow per the user). SINGLE-GPU/EmulatedFp64/CUDA-13 context applies but this is a READ audit.',
  phases: [ { title: 'Parallel host-compute audit across every production path' }, { title: 'Synthesize the ranked audit report + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; a GPU product. Branch phase2-fit-engine == main @ 621a7db. The ComputeBackend seam (src/device/backend.hpp): CudaBackend = the PRODUCTION GPU path; CpuBackend (src/device/cpu/*) = the TEST-ONLY parity oracle (EXEMPT from this audit — it is SUPPOSED to be on the host). src/core/* = the backend-agnostic orchestration (run_qpadm/run_f4/run_f3/run_f4ratio/run_dstat/run_qpfstats/run_qpgraph/run_dates + the jackknife drivers). src/app/* = CLI.',
  'WHY THIS AUDIT: the qpfstats block-jackknife was found on the CPU — src/core/stats/qpfstats.cpp matrix_jackknife_est_col, a host long-double per-block loop called PER popcomb (~305k combos x 711 blocks at 40 pops = ~217M host iterations, GPU 50% idle / 100-0 alternation) — DESPITE explicit GPU-only/keep-on-device instructions. It passed the golden gate because the 9-pop golden is tiny (666 combos). FIND EVERY OTHER host-compute violation of the same kind BEFORE it bites at scale.',
  'THE OFFENDER PATTERN (a VIOLATION): host-side floating-point/statistical compute in the PRODUCTION path (src/core, src/app, or run regardless of backend) that SCALES WITH THE DATA (M SNPs / n_block / N items / npopcomb / npairs / n_restart) — reductions, block-jackknife, covariance, GLS, regression, normalization, het-correction, per-item/per-block/per-SNP/per-combo/per-pair arithmetic — that should be a ComputeBackend GPU kernel but is being done on the host (serializing with or instead of the GPU).',
  'EXEMPT (do NOT flag as violations): (1) the CpuBackend src/device/cpu/* (test-only oracle); (2) CLI/arg/config parsing; (3) ONE-TIME small integer index/setup-array builds (e.g. building a combo/quad index list, O(combos) once, is acceptable SETUP — flag only if it is large AND repeated/in the hot loop); (4) the unavoidable genotype/f2 READ orchestration (but DO flag if the decode/2-bit unpack or het-correction is a slow HOST hot loop that competes with the GPU and could be on-device); (5) tiny final result formatting/emit. THE DISCRIMINATOR: host arithmetic that scales with the data dimension AND serializes against / replaces the GPU compute = VIOLATION; integer setup / IO / formatting = OK.',
  'HARD-VERIFY: read the REAL code (file:line). Do NOT guess. For each candidate, confirm it runs in the PRODUCTION (CudaBackend) path — not only the CpuBackend. Classify severity by the scale at which it bites: CRITICAL (bites at production/40+ pop/sweep scale like the qpfstats jackknife), MEDIUM (bites at large but not golden scale), LOW (small fixed cost). NO code changes — audit only.',
  'Box ' + SSH + ' available if you need to read the AT2 reference for what SHOULD be computed where. SINGLE-GPU/EmulatedFp64/CUDA-13 are the standing constraints the production path must honor.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); for (let i=0;i<2&&r===null;i++){ log(opts.label+': transient null/500 — retry '+(i+1)); r = await agent(p, {...opts, label: opts.label+':retry'+(i+1)}) } return r }

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['area','offenders','clean','notes'],
  properties: {
    area: { type: 'string' },
    offenders: { type: 'array', description: 'host-compute violations found (empty if clean)', items: {
      type: 'object', additionalProperties: false, required: ['file_line','what','in_production_path','scales_with','bites_at','severity','fix'],
      properties: {
        file_line: { type: 'string', description: 'file:line of the host-compute offender' },
        what: { type: 'string', description: 'what it computes on the host' },
        in_production_path: { type: 'boolean', description: 'true = runs in the CudaBackend/production path (not only CpuBackend)' },
        scales_with: { type: 'string', description: 'the data dimension it scales with (M / n_block / N / npopcomb / npairs / restarts)' },
        bites_at: { type: 'string', description: 'the scale at which it bites (golden 9-pop? 40+ pop? sweep/billions? large panel?)' },
        severity: { type: 'string', description: 'CRITICAL | MEDIUM | LOW' },
        fix: { type: 'string', description: 'the on-device fix (which backend kernel / batched approach)' },
      },
    } },
    clean: { type: 'string', description: 'the parts CONFIRMED correctly on-device (the clean bill) for this area, file:line' },
    notes: { type: 'string' },
  },
}

phase('Parallel host-compute audit across every production path')
const areas = [
  { key: 'fit-engine', p: 'AUDIT the qpAdm/qpWave fit engine + the S8 rotation. READ src/core/qpadm/qpadm_fit.cpp + model_search.cpp (run_qpadm/run_qpwave/run_qpadm_search/fit_models_batched) + the jackknife driver src/core/qpadm/jackknife.hpp. Is the est_to_loo / xtau / Q / Qinv / the model-batched fit all dispatched to the GPU (jackknife_cov, the batched potrf/potrs), or is any per-model / per-block / per-rep arithmetic done on the host? Flag host compute that scales with n_block / n_model / n_rep.' },
  { key: 'f4-f3-f4ratio', p: 'AUDIT f4/f3/f4-ratio. READ src/core/qpadm/f4.cpp, f3.cpp, f4ratio.cpp. The assemble (assemble_f4_quartets/f3_triples) is on-device; but is the DIAGONAL jackknife (f4_diag_var / jackknife_diag) on-device, and is the f4-RATIO ratio-jackknife (ratio_jackknife) on the host or device? Flag any host per-item/per-block jackknife or normalization loop (esp. one that would bite at sweep scale / many items).' },
  { key: 'qpdstat', p: 'AUDIT qpDstat A + B. READ src/core/stats/dstat.cpp (run_dstat) + the dstat_jackknife (dstat.cpp:70-157, described as "host-pure num/den block-jackknife"). Is the num/den block-jackknife on the HOST? At what scale does it bite (D over many quadruples / the sweep)? Flag it + any host per-quadruple/per-block loop. Confirm the decode + the dstat_block_reduce kernel are on-device.' },
  { key: 'sweep', p: 'AUDIT the all-quartets/all-triples sweep. READ src/core/qpadm/fstat_sweep.cpp + cuda_backend.cu run_fstat_sweep_device + the unrank/zfilter/CUB-compaction kernels. This is believed GPU-bound (on-device unrank/compute/filter/CUB). CONFIRM there is no host enumeration / host per-quartet compute / host filter that scales with the C(P,4) count; flag any host hot loop.' },
  { key: 'qpfstats', p: 'AUDIT qpfstats (the KNOWN offender — confirm + find MORE). READ src/core/stats/qpfstats.cpp fully. Confirm matrix_jackknife_est_col (the host long-double per-combo jackknife) as the CRITICAL offender. THEN find ANY OTHER host compute: the recenter (per-pair over blocks?), the popcomb/design build (O(npopcomb) host — setup or hot?), the scatter, any host normalization. Flag everything that scales with npopcomb/npairs/n_block on the host.' },
  { key: 'qpgraph', p: 'AUDIT qpGraph. READ src/core/stats/qpgraph.cpp (or wherever run_qpgraph lives) + cuda_backend.cu qpgraph paths. The fleet optimizer + GLS are on-device; but is the f-stat BASIS assembly, the Qinv/jackknife, the path-table build, or the result/SE computed on the host in a way that scales (with npairs / n_block / n_restart / n_graph for future topology search)? Flag host compute; confirm the fleet + basis + Qinv are on-device.' },
  { key: 'dates', p: 'AUDIT DATES. READ src/core/stats/dates.cpp + cuda_backend.cu dates_curve + cpu_backend.cpp dates path. The cuFFT autocorrelation is on-device; but is the per-sample WEIGHT/residual computation, the grid-scatter, the leave-one-chrom JACKKNIFE (fixjcorr/weightjack), or the decay fit done on the host in a scaling loop? DATES is decode-bound, but flag any host per-sample/per-chrom/per-SNP arithmetic that should be on-device.' },
  { key: 'decode-extract', p: 'AUDIT the decode/extract front-end (shared by all genotype tools). READ src/core/internal/decode_af.hpp + f2_estimator.hpp (het_correction/f2_term) + src/io readers + src/app/extract_f2_core.cpp. The 2-bit unpack / allele-freq / het-correction: is it on-device (decode_af GPU) or a host hot loop over M SNPs x N inds? The ~14s decode at 40 pops suggests the decode may be host-bound — flag if the unpack/AF/het is host compute that should be a kernel, and whether it can overlap.' },
  { key: 'jackknife-family-boundary', p: 'AUDIT the SHARED jackknife family + the core-vs-backend dispatch BOUNDARY (the cross-cutting view). READ src/device/backend.hpp (which compute is a ComputeBackend virtual = dispatched to GPU, vs which is a free function in src/core run on the host), jackknife_cov, jackknife_diag, dstat_jackknife, ratio_jackknife, matrix_jackknife_est_col, src/core/internal/small_linalg.hpp usage in the PRODUCTION path. Map: which statistical compute is on-device (backend virtual) vs host (core free function). The host jackknife family (dstat/ratio/matrix_jackknife_est) is the prime suspect pattern — catalog every member, whether it runs in production, and the scale it bites. Flag small_linalg used in the production (non-CpuBackend) path.' },
]
const audits = await parallel(areas.map(a => () => tryAgent(['You are the host-compute auditor for: ' + a.key + ' (read-only, NO code changes). ' + a.p, STD].join('\n'), { schema: AUDIT_SCHEMA, label: 'audit:' + a.key, phase: 'Parallel host-compute audit across every production path' })))
const ok = audits.filter(Boolean)
const totalOffenders = ok.reduce((n, a) => n + (a.offenders ? a.offenders.length : 0), 0)
log('audits returned: ' + ok.length + '/' + areas.length + '; offenders flagged: ' + totalOffenders)
if (ok.length === 0) { log('--- all audits died — HALT'); return { halted: true } }

phase('Synthesize the ranked audit report + commit')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['committed','critical_count','critical_list','medium_count','clean_bill','recommended_fix_order','note'],
  properties: {
    committed: { type: 'string', description: 'the commit hash + the doc path' },
    critical_count: { type: 'number' },
    critical_list: { type: 'string', description: 'the CRITICAL offenders (host compute that bites at production scale) — file:line + what + fix' },
    medium_count: { type: 'number' },
    clean_bill: { type: 'string', description: 'the paths confirmed correctly on-device' },
    recommended_fix_order: { type: 'string', description: 'the order to fix the offenders (after the qpfstats jackknife already in flight)' },
    note: { type: 'string' },
  },
}
const synth = await tryAgent([
  'You are synthesizing the host-compute audit into a ranked report. The per-area audits:\n<<<\n' + JSON.stringify(ok) + '\n>>>', STD, '',
  'WRITE docs/research/host-compute-audit.md: a RANKED table of every host-compute violation (area | file:line | what | in_production_path | scales_with | bites_at | severity | fix), grouped CRITICAL / MEDIUM / LOW; a section noting the qpfstats jackknife (already being fixed) as the exemplar; the CLEAN BILL (paths confirmed on-device); and the recommended fix order. De-dup across auditors. Then cd ' + R + ' && git add ONLY docs/research/host-compute-audit.md, commit (audit(host-compute): sweep every production path for CPU work violating the GPU-bound mandate — triggered by the qpfstats host jackknife; ranked offenders + fix order + clean bill) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Return the structured synthesis.',
].join('\n'), { schema: SYNTH_SCHEMA, label: 'synth:audit', phase: 'Synthesize the ranked audit report + commit' })
if (synth === null) { log('--- synth died — HALT'); return { halted: true, audits: ok } }
log('HOST-COMPUTE AUDIT: ' + synth.committed + ' — CRITICAL=' + synth.critical_count + ' MEDIUM=' + synth.medium_count)
return { audits: ok, synth }
