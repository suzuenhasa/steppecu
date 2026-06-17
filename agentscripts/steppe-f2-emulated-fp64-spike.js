export const meta = {
  name: 'steppe-f2-emulated-fp64-spike',
  description: 'Design+write a CUDA C++ experiment: does Ozaki emulated-FP64 survive the f2 catastrophic cancellation vs native FP64? (code only; build/run done by main loop on the 5090 box)',
  phases: [
    { title: 'Research', detail: '3 parallel: exact cuBLAS emulation API (read headers on box), numerical methodology, first code draft' },
    { title: 'Synthesize', detail: 'integrate exact API + methodology into one compilable .cu + build script' },
    { title: 'Verify', detail: 'adversarial review: column-major GEMM args, emulation actually engaged, reference truly higher-precision' },
  ],
}

const SSH = 'ssh -i ~/.ssh/id_vastai -p 28443 -o ConnectTimeout=20 root@82.221.170.234'

phase('Research')

const research = await parallel([
  // A. exact cuBLAS emulation API from the real headers on the box (read-only)
  () => agent([
    'Recon the EXACT cuBLAS FP64-emulation API on the target GPU box. Connect read-only with:',
    '  ' + SSH + " 'bash -lc \"<cmd>\"'",
    'The CUDA 13.0 headers are under /usr/local/cuda-13.0/targets/x86_64-linux/include/ (cublas_api.h, cublas_v2.h). grep/sed them to extract VERBATIM:',
    '(1) the full `cublasMath_t` enum (every value, incl. CUBLAS_DEFAULT_MATH, CUBLAS_PEDANTIC_MATH, and the FP64/FP32 *_EMULATED_*_MATH values);',
    '(2) the `cublasSetMathMode` prototype;',
    '(3) `cublasEmulationStrategy_t` enum values + `cublasSetEmulationStrategy` prototype + env-var notes in comments;',
    '(4) the mantissa-control API: `cublasSetFixedPointEmulationMantissaControl`, the `cudaEmulationMantissaControl_t` (or similarly named) enum values, and `cublasSetFixedPointEmulationMaxMantissaBitCount`;',
    '(5) the `cublasComputeType_t` value `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` and the `cublasGemmEx` prototype;',
    '(6) any header COMMENTS describing the required call sequence to ENGAGE FP64 fixed-point emulation, and whether plain `cublasDgemm` honors the math mode or whether `cublasGemmEx`/cublasLt with the emulated compute type is required.',
    'Use grep -n -A/-B and sed -n ranges; quote the exact lines. Do NOT write, compile, or run anything on the box — read-only. Return a tight reference sheet of verbatim signatures/enums + the recommended minimal call sequence to make a double-precision GEMM use Ozaki emulation, citing the header lines.',
  ].join('\n'), { label: 'recon-cublas-api', phase: 'Research' }),

  // B. numerical methodology
  () => agent([
    'Design the numerical methodology for an experiment deciding whether Ozaki emulated-FP64 survives the catastrophic cancellation in the Patterson f2 statistic, computed via a GEMM reformulation.',
    'f2[i,j] = mean over SNPs s of [ (p_i,s - p_j,s)^2 - p_i,s(1-p_i,s)/(N_i-1) - p_j,s(1-p_j,s)/(N_j-1) ].',
    'GEMM reformulation over a block: G=Q@Q^T, Vpair=V@V^T, R=[Qsq;Hc]@V^T -> R_diag,H ; numerator = R_diag + R_diag^T - 2G - H - H^T ; f2 = numerator/Vpair. The cancellation: Sum p_i^2 + Sum p_j^2 - 2 Sum p_i p_j (each O(M*pbar^2)) yields the far-smaller Sum (p_i-p_j)^2.',
    'Specify precisely: (1) GROUND TRUTH reference — argue for CPU long double (80-bit x87) computing (p_i-p_j)^2 PER SNP then summing (no cancellation in the reference), and whether long double suffices for M up to 1e6 or whether double-double/Kahan is needed; how to validate the reference itself. (2) A synthetic data generator that DIALS cancellation severity: ancestral a_s ~ U(0.05,0.95); per-pop p_{i,s}=clamp(a_s+sigma*z_{i,s}, 1e-6, 1-1e-6), z~N(0,1); sweeping sigma from 1e-4 (severe: tiny f2, many digits lost) to 1e-1 (mild). Give the rough mapping sigma -> expected f2 magnitude -> decimal digits cancelled -> bits of accuracy needed. (3) METRICS: max & median RELATIVE error of native-FP64-GEMM and emulated-FP64-GEMM vs the long-double reference per sigma; "digits of agreement"; tie a PASS/FAIL threshold to steppe tolerance tiers (point estimate est ~1e-9..1e-6 relative). (4) Realistic dims to test: P in {10,30,100}, M in {1e4,1e5,1e6}. (5) State crisply what result JUSTIFIES emulated-FP64 for the f2 GEMMs and what CONDEMNS it. Return the methodology and the exact parameter sweep table.',
  ].join('\n'), { label: 'numerics-methodology', phase: 'Research' }),

  // C. first code draft
  () => agent([
    'Write a single self-contained CUDA C++17 program `f2_emu_spike.cu` (targeting sm_120 / RTX 5090, CUDA 13, link -lcublas) that empirically compares the f2 GEMM reformulation in three ways and prints accuracy + timing.',
    'Methods: (1) NATIVE FP64 — cublasDgemm with default math mode. (2) EMULATED FP64 — same GEMMs but with cuBLAS FP64 fixed-point (Ozaki) emulation engaged on the handle. (3) REFERENCE — CPU long double, computing (p_i-p_j)^2 per SNP then summing (cancellation-free ground truth).',
    'Program structure: CLI args `P M sigma [missing_frac]`. Generate column-major (cuBLAS-native) Q [P x M] allele freqs via ancestral a_s~U(0.05,0.95), p=clamp(a_s+sigma*z, 1e-6,1-1e-6); V (validity, all 1 unless missing_frac>0); N (haploid sample size, e.g. 100); Qsq=Q.*Q; Hc=Q.*(1-Q)./max(N-1,1).*V; stacked S=[Qsq;Hc] (2P x M). GEMMs (mind column-major + transposes): G[P x P]=Q*Q^T, Vpair[P x P]=V*V^T, R[2P x P]=S*V^T. Numerator+divide in a small double kernel or on host. Run methods (1) and (2); time the 3 GEMMs with cudaEvent for each; compute reference (3) on CPU. Print a table row: P, M, sigma, mean|f2|, maxRelErr_native, medRelErr_native, maxRelErr_emu, medRelErr_emu, t_native_ms, t_emu_ms.',
    'Include CUDA_CHECK and CUBLAS_CHECK macros (throw or exit on error). For the EMULATION-enabling sequence, write the most-likely-correct calls (cublasSetMathMode to the FP64 fixed-point emulation mode; cublasSetEmulationStrategy PERFORMANT; dynamic mantissa control) and mark them with a clear // TODO[verify-against-headers] comment so the synthesize step can correct them against the real header signatures. Also emit `build_run.sh`: `nvcc -O3 -std=c++17 -arch=sm_120 f2_emu_spike.cu -lcublas -o f2_emu_spike` then a sigma sweep (1e-4,1e-3,1e-2,1e-1) at a couple of (P,M). Return BOTH files complete, each in its own fenced code block with the filename on the line before the block.',
  ].join('\n'), { label: 'code-draft', phase: 'Research' }),
])
const [api, methodology, draft] = research.map(r => r || '(missing)')

phase('Synthesize')

const synth = await agent([
  'You are finalizing a CUDA C++ experiment. Integrate the three inputs below into a FINAL, compilable `f2_emu_spike.cu` plus `build_run.sh`, targeting CUDA 13 / sm_120 (RTX 5090), linking -lcublas.',
  'Use the EXACT cuBLAS emulation API from the RECON sheet (correct enum/function names and the recommended call sequence to engage FP64 fixed-point emulation) — replace the draft TODO with the verified calls. Implement the methodology (sigma sweep, long-double cancellation-free reference, max/median relative-error metrics, timing). Get the cuBLAS COLUMN-MAJOR GEMM transposes/dimensions right for G=Q*Q^T, Vpair=V*V^T, R=S*V^T. Keep it single-file, dependency-light (just <cublas_v2.h>, std headers), and make it print a clean results table.',
  'Output EXACTLY two fenced code blocks and nothing else: first the filename line `f2_emu_spike.cu` then its block; then `build_run.sh` then its block.',
  '',
  '==== RECON: exact cuBLAS emulation API ====',
  api,
  '',
  '==== METHODOLOGY ====',
  methodology,
  '',
  '==== DRAFT CODE ====',
  draft,
].join('\n'), { label: 'synthesize-code', phase: 'Synthesize' })

phase('Verify')

const verified = await agent([
  'You are a meticulous CUDA/numerics reviewer. Below is a final `f2_emu_spike.cu` + `build_run.sh`. Find and FIX every defect, then emit the corrected files. Check specifically:',
  '(1) cuBLAS COLUMN-MAJOR correctness: for column-major storage, are the op/transpose flags and m,n,k,lda,ldb,ldc right for G=Q*Q^T (P x P from Q which is P x M), Vpair, and R=S*V^T? This is the most likely bug.',
  '(2) Is FP64 emulation ACTUALLY engaged (correct math-mode/strategy calls per the verified API), and does the program FAIL LOUDLY (not silently fall back to native) if the emulation calls return an error or are unsupported? Add a check that the emulated path is really different from native (e.g. report if emu==native bit-for-bit, which would indicate emulation was ignored).',
  '(3) Is the long-double reference genuinely cancellation-free and higher-precision than the GEMM path, and computed independently (not from the GPU sums)?',
  '(4) Memory: for P=100,M=1e6, Q is 8e8 bytes (~0.8GB) — fits 32GB; confirm no accidental P x M x P materialization; confirm host long-double reference cost is acceptable (O(P^2 * M)).',
  '(5) Relative-error metric: guard divide-by-zero when a reference f2 is ~0; use |ref|>floor.',
  '(6) Compile correctness for nvcc -std=c++17 -arch=sm_120 (no C++20-isms, headers present, no undefined symbols).',
  'Emit EXACTLY two fenced code blocks and nothing else: `f2_emu_spike.cu` then its block, then `build_run.sh` then its block. If a defect cannot be fixed without the live headers, leave a clearly-marked // FIXME with the exact uncertainty.',
  '',
  '==== FILES TO REVIEW ====',
  synth,
].join('\n'), { label: 'verify-code', phase: 'Verify' })

return { verified, api, methodology }
