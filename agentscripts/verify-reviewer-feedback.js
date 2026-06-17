export const meta = {
  name: 'verify-reviewer-feedback',
  description: "Verify a reviewer's critique of the f2 emulation spike against real evidence: cuBLAS-13 headers, our actual measured runs on the 5090 box, the current docs/code, and the numerics.",
  phases: [
    { title: 'Verify', detail: '3 parallel: symbol/build claim, did-it-run/evidence claim, numerics + current-state claim' },
    { title: 'Synthesize', detail: 'per-claim verdict + what it changes + paper-frontier assessment' },
  ],
}

const BOX = 'ssh -i ~/.ssh/id_vastai -p 63709 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 root@108.255.76.60'
const HDR = '/usr/local/cuda-13.0/targets/x86_64-linux/include/cublas_api.h'
const LOCAL = '/home/suzunik/steppe'

const CONTEXT = [
  'CONTEXT — what was ACTUALLY done (the reviewer may be evaluating the ORIGINAL spike, not the current state; verify against this):',
  '- The experiment RAN on a remote 2x RTX 5090 / CUDA 13 box (' + BOX + '), NOT the local RTX2070/CUDA11.8 machine. Real AADR v66 HO genotype data.',
  '- We swept FIXED mantissa bits and measured (real AADR, P=50, vs a long-double cancellation-free reference): dynamic-mantissa Ozaki ≈ native (~1.8e-11); fixed 24b=2.5e-6, 32b=8.6e-9, 40b=2.2e-11 (≈native), 48b=1e-12.',
  '- We measured throughput (real AADR, P=2416 and P=4266, 100k SNPs): dynamic-mantissa Ozaki = PARITY with native FP64 (NO speedup — "the trap"); FIXED-slice 32b = 14.5–17.5× faster, 40b = 11–13×, all at native-grade accuracy. We engaged emulation with cublasSetEmulationStrategy(EAGER) + cublasSetFixedPointEmulationMantissaControl(FIXED) + MaxMantissaBitCount(bits).',
  '- After those measurements we REVISED docs/architecture.md §12: the f2 GEMMs now USE fixed-slice Ozaki (default 40-bit), and dynamic-mantissa is explicitly REJECTED as the trap. The earlier "f2 GEMMs must stay native FP64" framing was corrected.',
  '- Files: ' + LOCAL + '/experiments/f2_emu_spike/{f2_emu_spike.cu (the original spike the reviewer likely saw), f2_prec_acc.cu (the fixed-bit ACCURACY sweep 24/32/40/48/53), f2_timing.cu (the fixed-bit THROUGHPUT sweep)}; ' + LOCAL + '/src/device/cuda/f2_block_kernel.cu (M0 production kernel); ' + LOCAL + '/docs/{architecture.md §12/§0, ROADMAP.md §0}.',
].join('\n')

const CLAIMS = [
  "REVIEWER'S CLAIMS TO VERIFY:",
  'C1 (symbol/build): CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH may not be a real cublasMath_t spelling (reviewer found it in docs only as a compute type, not that math-mode enum); it is used UNCONDITIONALLY (not under STEPPE_HAVE_EMU_TUNING) in f2_emu_spike.cu, so if the name is wrong the file would not compile.',
  'C2 (no evidence): the experiment "can\'t run in the environment you describe (CUDA 11.8, no sm_120 nvcc), so there is no empirical evidence — only an experiment design."',
  'C3 (wrong axis — the central claim): the catastrophic cancellation is in the FINAL SUBTRACTION (held native FP64 in both arms); the 3 GEMMs are well-conditioned (κ~1 dot products of like-signed [0,1] terms); so emu vs native differ ONLY by relErr_GEMM, and κ amplifies BOTH arms equally. At DEFAULT (dynamic) mantissa, Ozaki targets ≥53 bits so relErr_emu ≈ relErr_native ≈ 2^-53 → the experiment would report PASS (emu≈native), CONTRADICTING a "must stay native FP64" conclusion. The danger only appears at REDUCED fixed mantissa (<53), but the file sets MaxMantissaBitCount=0 (auto) and never sweeps bits.',
  'C4 (best-case inputs): allele frequencies are in [0,1] so p² and products are in [0,1] — tiny dynamic range → few slices → emulation is BOTH fast AND accurate; the "f2 is a hard case for emulation" intuition is backwards for this input distribution; emulation might even BEAT native (Ozaki inner accumulation exact per slice vs native FP64 error growing with M).',
  'C5 (paper): no paper as-is (it tests a textbook expectation on synthetic data with unrun code); but the NUMERICS angle — "how many Ozaki mantissa bits does a cancellation-prone reduction need as a function of κ before the est-tier tolerance breaks" (sweep bits × κ × M, map the application-driven tolerance frontier) — is a real, less-scoopable contribution; and a genomics tools-track paper (GPU f-stats beating ADMIXTOOLS 2) is the lower-novelty option.',
].join('\n')

phase('Verify')
const checks = await parallel([
  () => agent([CONTEXT, '', CLAIMS, '',
    'VERIFY C1 (symbol/build). SSH the box and grep ' + HDR + ' (and cublas_v2.h / library_types.h) for the EXACT verbatim definitions of: CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (in the cublasMath_t enum), CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, cublasSetMathMode, cublasSetEmulationStrategy + cublasEmulationStrategy_t values, cublasSetFixedPointEmulationMantissaControl + the mantissa-control enum. Quote the lines. Then determine whether the code actually COMPILED AND RAN: check that the M0 CMake build + ctest passed and the spike binaries built (read ' + LOCAL + '/docs/ROADMAP.md, and on the box look for /workspace/steppe/build artifacts and the spike binaries under /workspace/steppe/experiments/f2_emu_spike/). VERDICT: is C1 TRUE (symbol missing / would not compile) or FALSE (symbol exists, code built+ran)? Give the verbatim header line as proof.',
  ].join('\n'), { label: 'C1-symbol-build', phase: 'Verify' }),

  () => agent([CONTEXT, '', CLAIMS, '',
    'VERIFY C2 (no empirical evidence). SSH the box: confirm GPUs are RTX 5090 sm_120 (nvidia-smi), nvcc is CUDA 13, and that the real AADR derived matrices exist (/workspace/data/aadr/derived_acc, derived_big2, derived_all — shape.txt/Q.f64). Read ' + LOCAL + '/docs/ROADMAP.md §0 and architecture.md §12 for the recorded MEASURED numbers. Determine: did the experiment actually EXECUTE on real sm_120 hardware with real genotype data and produce the recorded measurements (accuracy ladder + throughput), or is it merely an unrun design? VERDICT on C2 with evidence (binaries present, data present, recorded numbers traceable).',
  ].join('\n'), { label: 'C2-did-it-run', phase: 'Verify' }),

  () => agent([CONTEXT, '', CLAIMS, '',
    'VERIFY C3 + C4 (the numerics + current state) — the most important. Read ' + LOCAL + '/experiments/f2_emu_spike/{f2_emu_spike.cu, f2_prec_acc.cu, f2_timing.cu}, ' + LOCAL + '/src/device/cuda/f2_block_kernel.cu, and ' + LOCAL + '/docs/architecture.md (§12) + ROADMAP.md (§0). Assess:',
    '(a) Is C3\'s error analysis CORRECT — the cancellation is in the native-FP64 subtraction, the GEMMs are well-conditioned, emu vs native differ only by relErr_GEMM, κ amplifies both equally, and at DYNAMIC mantissa emu≈native (so a "must stay native" conclusion would be contradicted)? Reason it through; you may WebSearch the Ozaki / cuBLAS-emulation accuracy literature.',
    '(b) Does C3 AGREE with our MEASURED result that DYNAMIC-mantissa Ozaki gives PARITY with native on real data (no speedup), and that both the speed win and the accuracy danger live in the FIXED mantissa-bit knob? (i.e. is the reviewer independently re-deriving what we measured?)',
    '(c) Is C3\'s "the file never sweeps mantissa bits / sets MaxMantissaBitCount=0 / EAGER gated off" claim TRUE of the CURRENT state? Check whether f2_prec_acc.cu / f2_timing.cu DO sweep fixed bits {24,32,40,48,53} and DO use EAGER — i.e. is this critique aimed at only the one original spike file?',
    '(d) Does the CURRENT architecture.md §12 still say "f2 GEMMs must stay native FP64", or has it been corrected to fixed-slice Ozaki with dynamic rejected? Quote it.',
    '(e) Assess C4: the reviewer argues [0,1] inputs → narrow dynamic range → best case for Ozaki. But the GEMM input is S=[Qsq; Hc] where Hc = q(1-q)/(N-1) and per-pop N varies hugely (singletons vs large pops) + masked zeros — does that widen the effective dynamic range, consistent with our measurement that dynamic-mantissa OVERSHOT (~60 bits) → parity? Reconcile the reviewer\'s "narrow range / best case" with our "dynamic = slow on real data" measurement.',
    'VERDICT: for C3 and C4, state which points are CORRECT, which are TRUE-BUT-ALREADY-ADDRESSED (we independently reached the same conclusion and revised the architecture), and which are STALE/FALSE for the current state. Be fair: the reviewer\'s numerics insight may be exactly right and simply already incorporated.',
  ].join('\n'), { label: 'C3C4-numerics-state', phase: 'Verify' }),
])
const [c1,c2,c34] = checks.map(r=>r||'(none)')

phase('Synthesize')
const verdict = await agent([
  CONTEXT, '', CLAIMS, '',
  'Synthesize the three verification results into a fair, precise verdict for the user.',
  '== C1 (symbol/build) ==', c1, '', '== C2 (did it run) ==', c2, '', '== C3+C4 (numerics + current state) ==', c34, '',
  'Produce: (1) a per-claim verdict table — for C1..C5: TRUE / FALSE / CORRECT-BUT-ALREADY-ADDRESSED / STALE — each with the one-line evidence. (2) A short "what this actually changes for steppe" — separate the genuinely-actionable (if any) from the already-done. (3) An assessment of C5\'s paper suggestion: do we already have the pieces for the bits×κ×M tolerance-frontier figure (we have the bit sweep + κ via the synthetic sigma sweep + real-data points), what is genuinely missing (a full 2D bits×κ sweep at fixed M, tied to the est-tier tolerance), and is it worth doing. Be generous about the reviewer\'s strong craft/numerics observations while clearly flagging the factually stale claims (those premised on "it never ran" / "it never sweeps bits" / "must stay native"). Keep it tight and evidence-anchored.',
].join('\n'), { label: 'synthesis', phase: 'Synthesize' })

return { verdict }
