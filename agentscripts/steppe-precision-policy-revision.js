export const meta = {
  name: 'steppe-precision-policy-revision',
  description: 'Verify CUDA 13.x cuBLAS FP64 emulation (Ozaki scheme) + TF32 facts and redesign the steppe precision policy (drop native-FP64-default)',
  phases: [
    { title: 'Research', detail: '3 parallel: cuBLAS FP64 emulation API, Ozaki+TF32+cuSOLVER precision, f-stat numerical implications' },
    { title: 'Verify', detail: 'adversarial fact-check of API names, GPU support, accuracy/determinism overclaims' },
    { title: 'Synthesize', detail: 'produce redesigned policy + exact replacement text for sections 0/1/3/9/12' },
  ],
}

phase('Research')

const PROMPTS = [
  // A: cuBLAS FP64 emulation (Ozaki) in CUDA 13.x — the load-bearing facts
  'You are an NVIDIA-library expert. Establish the CURRENT, verifiable facts about cuBLAS FP64 (double-precision) EMULATION via the Ozaki scheme in CUDA Toolkit 13.x. Use WebSearch and WebFetch against the NVIDIA cuBLAS documentation, CUDA release notes (12.9, 13.0, 13.1), and the NVIDIA developer blog post(s) on FP64 / double-precision matmul emulation. Determine precisely, citing exact URLs: (1) Which CUDA/cuBLAS version introduced FP64 matmul emulation, and via what mechanism (Ozaki scheme using INT8 or BF16 tensor-core GEMMs). (2) The EXACT API to enable it: function names and enums — e.g. cublasSetEmulationStrategy and any CUBLAS_EMULATION_STRATEGY_* values, cublasLt matmul descriptor emulation attributes, the relevant cublasComputeType_t (e.g. CUBLAS_COMPUTE_64F) and math-mode flags, and any environment variables (e.g. CUBLAS_EMULATION_STRATEGY). Quote the doc. If you are not certain of an exact symbol name, say so explicitly rather than guessing. (3) Which GPUs support it (Blackwell datacenter GB200/B200 sm_100, consumer RTX 50 / sm_120; does it also run on Hopper/Ada/Ampere?). (4) Accuracy: does emulated FP64 match IEEE native FP64, and is accuracy tunable (number of Ozaki slices / accuracy modes)? (5) Performance vs native FP64, especially on consumer GPUs whose native FP64 is 1/32–1/64 rate. (6) Is emulated-FP64 GEMM bitwise-reproducible run-to-run (determinism)? Return tightly-organized markdown: a claims table with [VERIFIED]/[UNCERTAIN] tags and an inline source URL per claim. Do NOT fabricate API names.',

  // B: Ozaki scheme internals + TF32 + cuSOLVER precision flow
  'You are an expert in mixed-precision numerical linear algebra on GPUs. Research three things and cite URLs. (1) The Ozaki scheme: how error-free transformation works (splitting high-precision matrices into multiple low-precision/integer slices, computing several low-precision GEMMs, accumulating exactly), how accuracy scales with slice count, and that it can reach or exceed FP64 accuracy. Use the Ozaki et al. papers and NVIDIA material. (2) TF32 (NVIDIA TensorFloat-32): exact format (8-bit exponent, 10-bit mantissa, ~19-bit total), how cuBLAS enables it (CUBLAS_TF32_TENSOR_OP_MATH / compute-type), its accuracy class (roughly half-precision mantissa with FP32 range), where it is appropriate, and whether TF32 GEMM is deterministic. (3) How cuSOLVER consumes precision in CUDA 13.x: do its dense factorizations (potrf Cholesky, gesvd/gesvdj SVD, geqrf QR, gels least-squares, gesv/gels mixed-precision iterative-refinement solvers like cusolverDnXgesv/Xgelsd) benefit from or expose FP64 EMULATION, or do they run native precision? Is there any cuSOLVER emulation/iterative-refinement path in 13.x? Be precise about which routines are affected. Return markdown with [VERIFIED]/[UNCERTAIN] tags and inline source URLs.',

  // C: numerical implications for f-statistics + recommended policy shape
  'You are a numerical-methods engineer working on population-genetics f-statistics (Patterson f2/f3/f4) and qpAdm. The pipeline: (i) allele-frequency accumulation across SNPs (element-wise, bandwidth-bound, NOT a matmul); (ii) f2/f4 statistics = differences of large nearly-equal allele-frequency products (CATASTROPHIC CANCELLATION risk); (iii) covariance assembly via SYRK/GEMM over jackknife blocks (matmul-heavy); (iv) a GLS solve (Cholesky) and an SVD-based rank test (small dense). Analyze, with reasoning: (1) Where in this pipeline FP64-accuracy is mandatory because of cancellation, and where lower precision is tolerable. (2) Whether Ozaki-emulated FP64 (accuracy approximately equal to native FP64) is acceptable as the DEFAULT for the matmul-heavy covariance/f4 assembly while element-wise accumulation and the cancellation-sensitive subtraction stay in native FP64 arithmetic. (3) When TF32 is acceptable as a user-selectable precision — e.g. fast model-space SCREENING where only candidate RANKING matters, not the final est/se/z/p — and why it must not be the precision of a reported number unless re-validated. (4) The determinism consequences: is an emulated-FP64 path still run-to-run reproducible under a single-stream constraint? Then PROPOSE a concrete precision policy with named modes (suggest: EmulatedFp64 as default workhorse for GEMM-heavy work; native Fp64 as the validation oracle and gold reference; Tf32 as an opt-in fast/approximate mode), spelling out which mode each pipeline stage uses by default and the tolerance/validation implications. Use WebSearch to confirm any factual claim about emulation accuracy or cancellation. Return markdown with reasoning and any citations.',
]

const research = await parallel(PROMPTS.map((p, i) => () => agent(p, { label: ['cublas-fp64-emulation','ozaki-tf32-cusolver','fstat-numerics'][i], phase: 'Research' })))
const bundle = research.map((r, i) => '### RESEARCH ' + ['A: cuBLAS FP64 emulation','B: Ozaki/TF32/cuSOLVER','C: f-stat numerics + policy'][i] + '\n\n' + (r || '(no output)')).join('\n\n---\n\n')

phase('Verify')

const VERIFY = [
  'You are a ruthless technical fact-checker for NVIDIA CUDA 13.x library APIs. Below are three research bundles about cuBLAS FP64 emulation (Ozaki scheme), TF32, and cuSOLVER precision. Independently verify the load-bearing claims with WebSearch/WebFetch against primary NVIDIA sources. Produce a punch list of: (a) any FABRICATED or wrong API symbol / enum / env-var names (cublasSetEmulationStrategy, CUBLAS_EMULATION_STRATEGY_*, compute types, etc.) — confirm exact spelling or flag as unverifiable; (b) overclaims that emulated FP64 is bit-identical to native FP64 (it is accuracy-approximate, not bit-identical — confirm); (c) wrong GPU-support claims (which archs actually run emulation); (d) wrong determinism claims for emulated-FP64 or TF32; (e) wrong claims about cuSOLVER using emulation. For each, state the corrected fact + source URL, or mark [UNCERTAIN — not verifiable from primary sources]. Be terse and high-signal. End with a short list of facts that are SAFE to assert vs facts that must be marked [UNCERTAIN] in the spec.',
  '',
  bundle,
].join('\n')

const verdict = await agent(VERIFY, { label: 'fact-check', phase: 'Verify' })

phase('Synthesize')

const CURRENT_SECTIONS = ` // current doc text the synthesizer must rewrite, preserving voice + cross-refs
==== CURRENT §0 (assumptions table: the Precision and Determinism rows) ====
| Precision | **FP64 for all statistic-bearing math** | f-statistics are differences after large cancellation; FP32/TF32 biases χ²/SE. |
| Determinism | **single-stream statistic path + CCCL run_to_run reductions + scoped deterministic cuSOLVER** | Bit-stable on a given GPU under the documented constraints (§12) → non-flaky regression gates vs ADMIXTOOLS 2 goldens. |

==== CURRENT §1 (two precision-bearing bullets) ====
IS bullet: "- A faithful, FP64, validated reimplementation of the f2 → f3/f4 → block-jackknife → qpWave/qpAdm GLS+SVD pipeline."
NOT bullet: "- A research sandbox for low-precision tricks. TF32/FP16 never produce a reported statistic; FP32 is not a publicly reachable precision for any reported path (§9)."

==== CURRENT §3 (two stack rows) ====
| cuBLAS / cuSOLVER | from toolkit | GEMM/SYRK + Cholesky/SVD/batched-LAPACK with the *scoped* reproducibility controls in §12. |
(Precision is not its own row in §3; it lives in §0/§9/§12.)

==== CURRENT §9 (the Precision enum, DeviceConfig.precision default, and the "Why Precision has only Fp64" paragraph) ====
enum class Precision { Fp64 };                // ONLY Fp64 is publicly constructible for reported paths.
                                              // TF32/FP32 screening is NOT a Precision; it is an opt-in
                                              // flag on the *search* API that never emits est/se/z/p (§12).
DeviceConfig: Precision precision = Precision::Fp64;   // statistics-grade, and the only option
Paragraph "**Why Precision has only Fp64.**" explains the removed Fp32 foot-gun and that TF32 screening is a boolean screen_tf32 flag, final number recomputed in FP64.

==== CURRENT §12 (the Precision policy paragraph — first paragraph of §12) ====
"**Precision policy.** **FP64 for everything statistic-bearing** — allele-frequency accumulation, f-stat sums, covariance Q, Cholesky/SVD/GLS. f2/f4 are differences of large nearly-equal quantities; FP32/TF32 rounding biases est and inflates/deflates χ². FP32 is acceptable only for bulk genotype storage/bandwidth-bound element-wise ops that immediately feed an FP64 accumulator. TF32 (10-bit mantissa) and FP16 never produce a reported statistic — TF32 is permitted only as a fast first-pass screen of candidate models (the screen_tf32 search flag, §9), with the final number always recomputed in FP64. For validation use cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH). Where consumer-Blackwell FP64-emulation modes are used, validate them against pedantic native FP64 before trusting them."
(Also: §12 tolerance policy references "TF32-screened intermediate" in the loose tier; the determinism guarantee subsections reference cuBLAS single-stream and run_to_run.)`

const SYNTH = [
  'You are the principal engineer for `steppe` revising its precision policy in the canonical architecture doc. THE DECISION (from the project owner): native FP64 must NOT be the default; FP64-accuracy via the Ozaki-scheme EMULATION (cuBLAS FP64 emulation) is the intended default for the matmul-heavy work; TF32 is also a supported, user-selectable precision. Keep native FP64 available as the validation oracle / gold reference.',
  '',
  'Use ONLY facts the fact-check pass marked safe to assert; everything else must be written as **[UNCERTAIN]** in the doc voice (the doc already uses that convention). Do NOT invent cuBLAS/cuSOLVER API symbol names — if the exact symbol is not verified, describe the mechanism and mark it [UNCERTAIN].',
  '',
  'Design the new policy with NAMED precision modes. Recommended shape (adjust to the verified facts): a Precision enum with at least EmulatedFp64 (default; Ozaki emulation for GEMM/SYRK-heavy covariance & f4 assembly), Fp64 (native; the oracle/reference, used for validation and on parts with fast native FP64), and Tf32 (opt-in fast/approximate; for model-space screening / when the user accepts lower accuracy — results carry a precision tag and a looser tolerance tier, never bit-compared to AT2 goldens). Make explicit that: element-wise allele-frequency accumulation and the cancellation-sensitive f2/f4 subtraction stay in NATIVE FP64 arithmetic regardless of mode (emulation applies to the matmuls, not the cancellation-prone elementwise math); emulated-FP64 must be validated against pedantic native FP64; and the determinism guarantee still holds under the single-stream constraint (state explicitly whether emulated-FP64 GEMM is run-to-run reproducible, per the facts).',
  '',
  'Produce EXACT, paste-ready replacement text for EACH of the following, clearly delimited with the headers shown. Match the doc voice (opinionated, one-line rationales, §-cross-references, [UNCERTAIN] tags). For the §9 code block, output the full revised enum + DeviceConfig precision field + a rewritten "Why the precision modes" paragraph that REPLACES the old "Why Precision has only Fp64" foot-gun paragraph. Output sections in this order, each under a line starting with ">>> REPLACEMENT:":',
  '>>> REPLACEMENT: §0 Precision row (markdown table row)',
  '>>> REPLACEMENT: §0 Determinism row (markdown table row) — only if it needs adjustment; otherwise restate it unchanged',
  '>>> REPLACEMENT: §1 IS bullet (the "faithful, FP64..." bullet)',
  '>>> REPLACEMENT: §1 NOT bullet (the "research sandbox for low-precision tricks" bullet)',
  '>>> REPLACEMENT: §3 cuBLAS/cuSOLVER row(s) (+ a new emulation note row if warranted)',
  '>>> REPLACEMENT: §9 Precision enum + DeviceConfig precision field (full code) ',
  '>>> REPLACEMENT: §9 "Why the precision modes" paragraph (replaces "Why Precision has only Fp64")',
  '>>> REPLACEMENT: §9 build()-validation sentence about precision (what build() now validates re precision/determinism)',
  '>>> REPLACEMENT: §12 Precision policy paragraph (full rewrite)',
  '>>> REPLACEMENT: §12 tolerance-policy precision clause (the part mentioning TF32-screened intermediates / which tier emulated vs tf32 lands in)',
  '>>> NOTES: any other spot in the doc that now contradicts the new policy and how to fix it (list section + change).',
  '',
  'Keep each replacement self-contained and minimal-diff where possible. Do not rewrite unrelated content.',
  '',
  '==== FACT-CHECK VERDICT ====',
  verdict,
  '',
  '==== RESEARCH BUNDLES ====',
  bundle,
  '',
  CURRENT_SECTIONS,
].join('\n')

const synthesis = await agent(SYNTH, { label: 'policy-synthesis', phase: 'Synthesize' })

return { verdict, synthesis }
