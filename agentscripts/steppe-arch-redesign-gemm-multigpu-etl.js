export const meta = {
  name: 'steppe-arch-redesign-gemm-multigpu-etl',
  description: 'Redesign steppe arch: multi-GPU first-class, GEMM/tensor reformulation + fused kernels (reference is oracle not template, no JAX), genotype ETL/imputation/merge in-scope',
  phases: [
    { title: 'Research', detail: '3 parallel: multi-GPU+NCCL determinism, GEMM reformulation+fused kernels, streaming genotype ETL/impute/merge' },
    { title: 'Synthesize', detail: 'produce machine-parseable edit blocks across affected sections' },
    { title: 'Critique', detail: 'adversarial check: NCCL determinism, GEMM-f2 correctness/parity, precision coherence, scope creep' },
    { title: 'Finalize', detail: 'apply critique; emit final edit blocks' },
  ],
}

const DOC = '/home/suzunik/steppe/docs/architecture.md'

phase('Research')

const RESEARCH = [
  // 1. Multi-GPU single-node architecture + NCCL determinism
  'You are an NVIDIA multi-GPU systems expert. Design the SINGLE-NODE, MULTI-GPU execution model for `steppe` (a GPU qpAdm/admixtools2 reimplementation). Multi-GPU is now a DAY-ONE goal: parallelism + speedup WITH numerical parity to ADMIXTOOLS 2 are the objectives, not a deferred feature. The workload is precompute-once/fit-many: (a) an out-of-core streaming genotype pass that builds a small f2_blocks tensor [n_pop x n_pop x n_block], and (b) a massive model-space search of small dense-LA fits over the resident f2_blocks. Research and recommend, citing NVIDIA docs (CUDA C++ Programming Guide multi-GPU, NCCL docs, multi-GPU best practices) with URLs: (1) Process model: single-process-multi-GPU (one host process driving N devices via per-device streams/threads) vs multi-process. Recommend one for a single workstation and justify. (2) How to shard the streaming precompute across G GPUs: partition SNP tiles across GPUs, each accumulates a partial f2_blocks, then a collective reduction. Which collective (NCCL AllReduce/ReduceScatter) and how it is invoked. (3) Sharding the model-space search (S8) across GPUs after f2_blocks is replicated. (4) DETERMINISM across GPUs: is NCCL AllReduce bit-reproducible? Discuss NCCL reduction-order/algorithm nondeterminism, NCCL_ALGO, and how to obtain bit-stable cross-GPU results (e.g. fixed reduction tree, or reduce partials to host in a fixed order, or CCCL gpu_to_gpu reproducible-accumulation reductions). State plainly where cross-GPU bit-parity is achievable and where it is not. (5) Peer access (cudaDeviceEnablePeerAccess), NVLink vs PCIe implications, load balancing for uneven tiles. (6) Whether cuSOLVERMp / distributed dense LA is needed (the Q/X matrices are tiny — argue it is NOT needed and stays deferred). (7) Note CUDA 13 removed cooperative multi-device launch APIs. Return concrete design + a recommended config surface (device list, n_gpus). Tag claims [VERIFIED]/[UNCERTAIN] with URLs.',

  // 2. GEMM/tensor reformulation + fused kernels
  'You are an expert in GPU dense-linear-algebra reformulation of statistical kernels. The owner wants f-statistics computed by REFORMULATING into matrix multiplies (DGEMM/batched-GEMM on tensor cores), NOT by transliterating a scalar CPU loop, and wants aggressive kernel FUSION. NO array framework (NO JAX, NO CuPy) — native cuBLAS/cuSOLVER in C++/CUDA. Here is the owner reference reformulation for bias-corrected f2 with per-SNP missingness (their prototype): for a SNP block, with Q = zero-filled allele-freq matrix [n_pop x block], V = validity mask (1 valid/0 missing), N = per-SNP haploid sample sizes: G = Q @ Q^T ; Vpair = V @ V^T (shared-valid counts per pop pair) ; Q_sq = Q^2 (elementwise) ; het_corr = Q*(1-Q)/max(N-1,1)*V ; stack S = [Q_sq ; het_corr] (2n_pop x block) ; R = S @ V^T giving R_diag (sum p_i^2 over shared-valid) and H (sum het correction) ; then f2_numerator = R_diag + R_diag^T - 2*G - H - H^T ; f2 = where(Vpair>0, numerator/Vpair, 0). Tasks: (1) Formalize this as the production S2 kernel design using cuBLAS DGEMM and FP64 EMULATION (Ozaki) on the three GEMMs, batched/strided-batched over the n_block blocks; and a FUSED elementwise pre-pass kernel that produces Q (zero-filled), V, Q_sq, het_corr in one sweep over the decoded tile, feeding the GEMMs without materializing [SNP x pop x pop]. (2) CRITICAL numerical point: the catastrophic cancellation (R_diag + R_diag^T - 2G - H - H^T) happens AFTER the GEMMs, on the small n_pop x n_pop reduced matrices — so the GEMMs accumulate well-conditioned sums of (mostly nonneg) products (emulated-FP64 OK) while the cancellation/division is a tiny O(n_pop^2) elementwise step done in NATIVE FP64. Explain why this keeps the GEMM-reformulation coherent with a precision policy that says emulation governs only well-conditioned matmuls and cancellation stays native FP64. (3) Confirm S3 f3/f4 contraction and S4 jackknife covariance (SYRK) are also GEMM-shaped and should run emulated-FP64. (4) State the design PRINCIPLE crisply: reformulate statistics into dense tensor ops; fuse the elementwise feeders; the CPU reference backend is a correctness ORACLE that validates RESULTS, not a structural template the GPU mimics; thin __host__ __device__ scalar functions remain only for the reference and for per-element primitives, not for the production hot path. (5) Note the parity caveat: the exact admixtools2 pairwise-complete NaN-mean path may be needed as a slow validation oracle alongside the fast GEMM path. Use WebSearch to confirm any cuBLAS batched-GEMM / strided-batched API specifics. Return reformulation math, kernel/dispatch design, and the principle text.',

  // 3. Streaming genotype ETL: merge, filter, impute
  'You are a population-genetics data-engineering expert. `steppe` must now include genotype QC / data-munging as IN-SCOPE features (the owner explicitly wants them): (a) MERGING/COMBINING multiple datasets, (b) ON-THE-FLY FILTERING, (c) IMPUTATION / missing-data handling. These must work in an OUT-OF-CORE STREAMING tile pass (datasets exceed VRAM) and compose with the GEMM f2 kernel and multi-GPU sharding. Research and design, with parity to ADMIXTOOLS 2 / PLINK conventions (cite URLs): (1) MERGE: combining PLINK bed/bim/fam + EIGENSTRAT/PACKEDANCESTRYMAP sources — SNP intersection vs union, allele harmonization (ref/alt matching, strand flips for A/T C/G ambiguity), position/rsID keying, and conflict handling. What PLINK --bmerge and admixtools2 do. (2) FILTERING on the fly during streaming: MAF threshold, per-SNP missingness (geno), per-sample missingness (mind), SNP include/exclude lists, sample/population include lists, optional LD pruning; which are cheap streaming filters vs which need a pre-pass. (3) IMPUTATION / missing handling: mean-imputation vs the pairwise-complete valid-mask approach (which the f2 GEMM reformulation already supports via V and per-SNP N); when each is appropriate; what admixtools2 does (it uses pairwise-complete / per-pair valid counts, NOT imputation, by default) — so position imputation as an OPTIONAL mode and pairwise-complete as the parity default. (4) Where these live architecturally: an `io`/preprocessing layer that streams harmonized, filtered tiles (+ validity mask + per-SNP sample sizes) into the decode->freq->f2 pipeline, staying out-of-core and keeping the SNP->block rule consistent. (5) The new pipeline STAGE(S) this adds before/around S0-S2 and how they remain layering-legal (io leaf produces plain data; no upward deps). Return the preprocessing design, the parity defaults, and the new stage definitions. Tag [VERIFIED]/[UNCERTAIN] with URLs.',
]
const research = await parallel(RESEARCH.map((p,i)=>()=>agent(p,{label:['multi-gpu-nccl','gemm-reformulation','genotype-etl'][i], phase:'Research'})))
const bundle = research.map((r,i)=>'### RESEARCH '+['1 multi-GPU/NCCL','2 GEMM reformulation','3 genotype ETL'][i]+'\n\n'+(r||'(none)')).join('\n\n---\n\n')

const EDIT_FORMAT = [
  'OUTPUT FORMAT — your ENTIRE response is a sequence of edit blocks in EXACTLY this format and nothing else (no prose between blocks):',
  '',
  '@@@EDIT id=<short-kebab-id> mode=<replace|insert_after>',
  '@@@OLD',
  '<for mode=replace: the VERBATIM existing text to replace, copied EXACTLY from the current doc incl. all whitespace/backticks. for mode=insert_after: the VERBATIM existing text to insert AFTER (an anchor; it is kept).>',
  '@@@NEW',
  '<the new text. for replace: replaces OLD. for insert_after: inserted immediately after OLD (on its own following lines).>',
  '@@@ENDEDIT',
  '',
  'Rules: (1) The @@@OLD text MUST be an exact, unique substring of the current doc on disk — copy it character-for-character. (2) Keep edits MINIMAL-DIFF: replace the smallest span that does the job; prefer several small edits over one giant section rewrite. (3) Lines beginning with @@@ are reserved delimiters — never put @@@ at the start of a line inside OLD/NEW content. (4) Preserve the doc voice: opinionated, one-line rationales, §-cross-references, [UNCERTAIN] tags for unverified CUDA/NCCL facts, em-dashes. (5) Do NOT alter the precision policy that was just set (emulated-FP64 Ozaki default for matmuls, native FP64 for cancellation/elementwise + oracle, TF32 screening) except to make the GEMM-reformulation coherence explicit. (6) Every multi-GPU/NCCL determinism claim must be accurate per the research; mark uncertain ones [UNCERTAIN].',
].join('\n')

phase('Synthesize')

const SYNTH = [
  'You are the principal engineer for `steppe`. First, Read the current architecture doc at ' + DOC + ' in full. Then produce edit blocks implementing FOUR owner-mandated changes, coherently and with minimal diff:',
  '',
  'CHANGE A — Multi-GPU is a DAY-ONE goal (was deferred). Parallelism + speedup WITH parity is the objective. Single-node multi-GPU (single process, N devices). Update: §0 assumptions table (add a multi-GPU row; the project is multi-GPU single-node, not single-GPU-first), §1 vision + the "multi-node or multi-GPU HPC framework / Single-GPU first" NOT bullet (reframe: multi-GPU single-node is in scope and central; multi-NODE remains the deferred boundary; cuSOLVERMp still deferred because Q/X are tiny), §3 tech stack (add NCCL for collectives), §5 (add a multi-GPU parallelism note: shard SNP tiles across GPUs in S0-S2 then NCCL-reduce partial f2_blocks; shard the S8 model search across GPUs), §9 DeviceConfig + Resources (device LIST / n_gpus, per-GPU resources), §11 (turn §11.4 "Deferred: multi-GPU" into a real "Multi-GPU execution" subsection; note tile sharding + collective), §12 (cross-GPU determinism: NCCL reduction-order nondeterminism and how parity is obtained — fixed reduction order / gpu_to_gpu reproducible reductions; be honest where cross-GPU bit-parity is/ isn\'t achievable), §16 (ADR), §17 (scaffold: device/cuda/multi_gpu file + NCCL dep), §18 (DoD: multi-GPU parity bullet).',
  '',
  'CHANGE B — GEMM/tensor reformulation + fused kernels are CENTRAL; the CPU reference is an ORACLE, not a template; NO JAX/array-framework (native cuBLAS/cuSOLVER). Update: §2 (revise "Testability" and "Correctness before speed" and add/adjust a principle stating: reformulate statistics into dense tensor ops + fuse elementwise feeders; the reference validates RESULTS not structure; thin __host__ __device__ scalar funcs are for the reference/per-element primitives, not the production hot path), §5 S2 (replace the "outer product over pop axis per SNP" description with the 3-GEMM bias-corrected f2 reformulation: G=QQ^T, Vpair=VV^T, stacked [Q_sq;het_corr]@V^T -> R_diag,H, f2=(R_diag+R_diag^T-2G-H-H^T)/Vpair, batched over blocks, emulated-FP64 on the GEMMs, fused elementwise pre-pass), §7 "Host/device separation" (reframe so it does not read as "every kernel is a thin shell around a scalar function" — the production path is GEMM-reformulated + fused; the thin-shell model is for the reference and per-element primitives), §12 (make explicit: the f2 GEMMs accumulate well-conditioned product-sums = emulated-FP64-eligible, and the catastrophic cancellation R_diag+R_diag^T-2G-H-H^T is a tiny O(n_pop^2) NATIVE-FP64 step on the reduced matrix — this is WHY the reformulation is coherent with the precision policy), §18 (DoD: GEMM-reformulation reference-equivalence). Affirm "native cuBLAS, no array framework (no JAX/CuPy)" somewhere appropriate (e.g. §0 assumptions or §2).',
  '',
  'CHANGE C — Genotype QC / data-munging is IN SCOPE: merging/combining datasets, on-the-fly filtering (MAF/geno/mind/include-exclude lists), and imputation/missing-data handling. Update: §1 (REVERSE the "genotype-QC / data-munging tool ... we do not impute, filter on the fly, or rewrite datasets" NOT bullet — these are now FEATURES; keep any genuinely-still-excluded item minimal), §1 vision (mention preprocessing/merge), §4 repo layout (add io/ files: merge/harmonize, filter, impute), §5 (add the preprocessing stage(s) before S0-S2, streaming + layering-legal, producing harmonized/filtered tiles + validity mask + per-SNP sample sizes; default missing-data handling = pairwise-complete valid-mask for AT2 parity, imputation OPTIONAL), §11 (streaming preprocessing stays out-of-core), §16 (ADR), §17 (scaffold), §18 (DoD).',
  '',
  'CHANGE D — these must all stay mutually coherent and coherent with the existing layering (app/bindings->api->core->device, io isolated leaf), the precision policy, and the out-of-core streaming design.',
  '',
  EDIT_FORMAT,
  '',
  '==== RESEARCH (verified facts to build on) ====',
  bundle,
].join('\n')

const synthEdits = await agent(SYNTH, { label: 'synthesize-edits', phase: 'Synthesize' })

phase('Critique')

const CRITIQUE = [
  'You are a skeptical staff engineer. Read the current architecture doc at ' + DOC + '. Below are PROPOSED EDIT BLOCKS for a redesign (multi-GPU-first, GEMM reformulation, genotype ETL in-scope). Produce a terse, prioritized punch list of problems:',
  '- Any multi-GPU/NCCL determinism claim that is WRONG or overstated (e.g. claiming NCCL AllReduce is bit-deterministic by default — it generally is not). ',
  '- Any way the 3-GEMM f2 reformulation is numerically WRONG or fails to match admixtools2 (bias correction, pairwise-complete missingness via Vpair, the cancellation location).',
  '- Any INCOHERENCE with the just-applied precision policy (emulated-FP64 for matmuls, native FP64 for cancellation/elementwise + oracle, TF32 screening) — especially whether the cancellation is correctly placed on the small reduced matrix in native FP64.',
  '- SCOPE CREEP in the ETL/preprocessing (keep it to merge, on-the-fly filter, optional impute; pairwise-complete default).',
  '- Layering violations (io must stay an isolated leaf; core must not include CUDA; multi-GPU resources injected not global).',
  '- @@@OLD anchors that are NOT verbatim substrings of the current doc (these will fail to apply) — flag any you suspect.',
  '- Internal contradictions the edits introduce elsewhere in the doc that are not also edited.',
  'For each: section + what is wrong + the specific fix. Do not rewrite the edits; just the punch list.',
  '',
  '==== PROPOSED EDIT BLOCKS ====',
  synthEdits,
].join('\n')

const critique = await agent(CRITIQUE, { label: 'staff-critique', phase: 'Critique' })

phase('Finalize')

const FINAL = [
  'You are the principal engineer finalizing the redesign edits for ' + DOC + '. Read the current doc on disk. Below are the PROPOSED EDIT BLOCKS and a STAFF CRITIQUE punch list. Apply every valid critique item and emit the FINAL edit blocks.',
  'Critical correctness bars: (1) NCCL AllReduce is NOT bit-deterministic by default — say so and specify how parity is obtained (fixed reduction order / reduce partials in a fixed order on one device / gpu_to_gpu reproducible reductions), marking residual uncertainty [UNCERTAIN]. (2) The 3-GEMM f2 must be numerically correct and the catastrophic cancellation must sit on the small n_pop^2 reduced matrix in native FP64 (coherent with the precision policy). (3) ETL stays scoped (merge, on-the-fly filter, optional impute; pairwise-complete default for parity). (4) Every @@@OLD anchor MUST be a verbatim unique substring of the current doc — re-read the doc and copy exactly. (5) No layering violations.',
  '',
  EDIT_FORMAT,
  '',
  '==== PROPOSED EDIT BLOCKS ====',
  synthEdits,
  '',
  '==== STAFF CRITIQUE ====',
  critique,
].join('\n')

const finalEdits = await agent(FINAL, { label: 'finalize-edits', phase: 'Finalize' })

return { finalEdits, critique }
