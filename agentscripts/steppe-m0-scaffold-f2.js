export const meta = {
  name: 'steppe-m0-scaffold-f2',
  description: 'M0: lift the validated f2 kernel into the architecture structure — scaffold (CMake/RAII/ComputeBackend/config) + f2 kernel with magic-numbers→config, CPU-reference test. Authored locally, built/tested on the remote 5090 box, stale comments swept.',
  phases: [
    { title: 'Contracts', detail: 'foundational headers: config (precision knob), ComputeBackend, shared f2 primitive, Q/V/N views' },
    { title: 'Implement', detail: 'parallel: device RAII, f2 kernel+CUDA backend, CPU reference, CMake build, equivalence test' },
    { title: 'BuildVerify', detail: 'rsync to box, build (CUDA 13/sm_120), run f2 CPU-vs-GPU equivalence test, fix-loop' },
    { title: 'CommentSweep', detail: 'fix stale comments (native-FP64-for-f2, dynamic-mantissa, resolved FIXMEs)' },
  ],
}

const LOCAL = '/home/suzunik/steppe'
const BOX   = 'ssh -i ~/.ssh/id_vastai -p 63709 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 root@108.255.76.60'
const STD = [
  'PROJECT: steppe — GPU/CUDA reimplementation of admixtools2/qpAdm. You are doing Milestone M0: lift the VALIDATED f2 kernel into the production structure.',
  'AUTHOR ALL FILES LOCALLY under ' + LOCAL + ' (real paths per architecture §4). Do NOT build locally (local box is an RTX 2070/CUDA 11.8 — wrong arch). Building/running happens later on the remote box.',
  'READ FIRST: ' + LOCAL + '/docs/architecture.md (§2 principles, §4 layout, §6 build, §7 CUDA idioms, §8 helpers, §9 config, §12 precision) and ' + LOCAL + '/docs/ROADMAP.md (§0 measured findings, §2 Q/V/N contract, §4 magic-number→config inventory, §5 standards).',
  'VALIDATED LOGIC TO LIFT (do not re-derive): ' + LOCAL + '/experiments/f2_emu_spike/{f2_emu_spike.cu (assemble_f2_kernel, run_f2_gemms, loader), f2_prec_acc.cu (long-double cancellation-free reference + fixed-slice Ozaki setup), f2_timing.cu (fixed-bit Ozaki engagement)} and ' + LOCAL + '/experiments/aadr/build_tgeno_matrix.py.',
  'PRECISION POLICY (MEASURED — this is the law): the f2 GEMMs use FIXED-slice Ozaki emulation, default mantissa_bits=40 (≈native accuracy), 32 = faster/8.6e-9; native Fp64 is oracle/fallback; the small numerator/divide stays native Fp64; Tf32 screening-only; DYNAMIC mantissa is the rejected trap — never use it. Engage via cublasSetMathMode(CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)+cublasSetEmulationStrategy(EAGER)+cublasSetFixedPointEmulationMantissaControl(CUDA_EMULATION_MANTISSA_CONTROL_FIXED)+cublasSetFixedPointEmulationMaxMantissaBitCount(bits). Build flag -DSTEPPE_HAVE_EMU_TUNING=1.',
  'STANDARDS: RAII for all device memory/handles (no raw cudaMalloc outside wrappers); strict layering (CUDA PRIVATE to device layer; core/api never include CUDA); DRY (one CUDA_CHECK, one loader, one f2 primitive shared by CPU ref + GPU); NO magic numbers (promote to config.hpp / named constants per ROADMAP §4); correct, current comments. No synthetic data for precision/throughput claims.',
].join('\n')

phase('Contracts')
const contracts = await agent([
  STD, '',
  'Author the FOUNDATIONAL CONTRACT files locally (these define the interfaces every other file builds against). Create:',
  '- include/steppe/config.hpp — host-only header. `struct Precision { enum class Kind { Fp64, EmulatedFp64, Tf32 }; Kind kind = Kind::EmulatedFp64; int mantissa_bits = 40; };` (comments: fixed-slice Ozaki 40≈native/32 faster; dynamic = the trap, not offered). `struct DeviceConfig { std::vector<int> devices; Precision precision; std::size_t stream_count=1, search_streams=4; bool use_mem_pool=true, enable_peer_access=true; };` `struct FilterConfig { double maf_min=0.0; double geno_max_missing=1.0; double mind_max_missing=1.0; };` plus named constants replacing spike magic numbers (e.g. kCdivBlock=16, kRelFloor=1e-12, kAbsFloor=1e-300, default block_size_cm=5.0). Doxygen comments; NO magic numbers; comments reflect the MEASURED precision policy.',
  '- include/steppe/error.hpp — minimal `enum class Status { Ok, DeviceOom, RankDeficient, NonSpdCovariance, InvalidConfig };` (the §10 taxonomy, trimmed).',
  '- src/core/internal/views.hpp — the Q/V/N contract as non-owning column-major [P x M] views (a small `MatView` struct {const double* data; int P; long M;} with element(i,s)=data[i+P*s]); document the contract (Q=ref-allele freq, V=validity, N=non-missing haploid count, column-major lda=P).',
  '- src/core/internal/f2_estimator.hpp — the shared `__host__ __device__` per-element bias-corrected f2 primitive (the (p_i-p_j)^2 - hc_i - hc_j formula and the hc = q(1-q)/max(N-1,1) helper) so the CPU reference and the GPU feeder CANNOT diverge; plus launch_config helpers `cdiv(n,b)` and a `grid_for`.',
  '- src/device/backend.hpp — `class ComputeBackend` abstract interface, CUDA-FREE header: a method to compute the f2 matrix from Q/V/N views at a given Precision, returning f2 [P x P] + Vpair [P x P] (host vectors or out-spans). Both CPU and CUDA backends implement it. Keep it minimal but real.',
  '- src/core/domain/block_partition_rule.hpp — host-pure SNP→block stub: `int block_of(double genpos_morgans, double block_size_morgans)` + doc; this is the single source of the block rule.',
  '',
  'Return the EXACT signatures (verbatim) of: Precision, DeviceConfig, MatView, ComputeBackend, and the f2_estimator primitive — so the implementation agents match them exactly.',
].join('\n'), { label: 'contracts', phase: 'Contracts' })

phase('Implement')
const impl = await parallel([
  () => agent([STD, '', '== CONTRACTS (build against these verbatim) ==', contracts, '',
    'Author the DEVICE RAII + checks (architecture §7), locally: src/device/cuda/check.cuh (STEPPE_CUDA_CHECK with std::source_location + throw typed CudaError; CUBLAS_CHECK; post-launch STEPPE_CUDA_CHECK_KERNEL), src/device/cuda/device_buffer.cuh (DeviceBuffer<T> move-only RAII, full move-ctor+move-assign, dtor logs-on-error never throws — the ONLY place cudaMalloc/cudaFree appear), src/device/cuda/stream.hpp (Stream/Event RAII), src/device/cuda/handles.hpp (CublasHandle RAII, created once, sets stream). Match the spike CUDA_CHECK behavior but as the single shared header. Correct comments.',
  ].join('\n'), { label: 'device-raii', phase: 'Implement' }),

  () => agent([STD, '', '== CONTRACTS ==', contracts, '',
    'Author the f2 KERNEL + CUDA backend (the core lift), locally: src/device/cuda/f2_block_kernel.cu — the fused elementwise pre-pass (decoded tile -> Q(masked),V,Qsq,Hc per the f2_estimator primitive), the 3 GEMMs (G=Q*Qᵀ, Vpair=V*Vᵀ, R=[Qsq;Hc]*Vᵀ; column-major args EXACTLY as in the spike run_f2_gemms), and the assemble_f2 kernel (numerator/divide in native FP64). Precision: when Precision::EmulatedFp64, engage FIXED-slice Ozaki at precision.mantissa_bits (the cublasSet* sequence in STD), under #if STEPPE_HAVE_EMU_TUNING; when Precision::Fp64, native CUBLAS_COMPUTE_64F; Tf32 path optional. Plus src/device/cuda/cuda_backend.cu implementing ComputeBackend using DeviceBuffer + this kernel + a CublasHandle. LIFT from experiments/f2_emu_spike/f2_emu_spike.cu (run_f2_gemms, assemble_f2_kernel) and f2_timing.cu (fixed-bit engagement). PROMOTE every magic number to config.hpp/named constants (16x16 block -> kCdivBlock via grid_for; the bias-correction floor -> the f2_estimator helper). Comments must reflect the MEASURED policy (fixed-slice Ozaki for f2; do NOT carry the stale "native FP64 / no throughput" spike comments).',
  ].join('\n'), { label: 'f2-kernel', phase: 'Implement' }),

  () => agent([STD, '', '== CONTRACTS ==', contracts, '',
    'Author the CPU REFERENCE backend, locally: src/core/cpu/cpu_backend.cpp — implements ComputeBackend by computing f2 with the cancellation-FREE long-double per-SNP reference (lift the reference from experiments/f2_emu_spike/f2_prec_acc.cu: for each pop pair, sum over jointly-valid SNPs of (p_i-p_j)^2 - hc_i - hc_j in long double, divide by the pairwise-complete count), using the SHARED f2_estimator primitive so it cannot diverge from the GPU feeder. This is the correctness oracle. Pure host C++ (no CUDA include). Correct comments.',
  ].join('\n'), { label: 'cpu-reference', phase: 'Implement' }),

  () => agent([STD, '', '== CONTRACTS ==', contracts, '',
    'Author the BUILD SYSTEM (architecture §6), locally: top-level CMakeLists.txt (project, C++20, enable_language(CUDA), CMAKE_CUDA_ARCHITECTURES=120, targets: steppe_core [host], steppe_device [CUDA PRIVATE, links CUDA::cublas], a test exe; -DSTEPPE_HAVE_EMU_TUNING=1; warnings sane), CMakePresets.json (dev/release/ci), cmake/ minimal modules if needed, .clang-format, .clang-tidy, .gitignore. ALSO a fallback ' + LOCAL + '/build_m0.sh that compiles the equivalence test directly with `nvcc -O3 -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1 <sources> -lcublas -o test_f2_equivalence` (so the slice builds even if CMake needs polish). Reference the actual source file paths the other agents create (device/cuda/*.cu, core/cpu/*.cpp, tests/*).',
  ].join('\n'), { label: 'build-system', phase: 'Implement' }),

  () => agent([STD, '', '== CONTRACTS ==', contracts, '',
    'Author the REFERENCE-EQUIVALENCE TEST, locally: tests/reference/test_f2_equivalence.cu — load a small REAL Q/V/N matrix from a dir (reuse the spike loader; default /workspace/data/aadr/derived_acc which exists on the box, P=50 M=100000), compute f2 via the CUDA backend at Precision::EmulatedFp64{mantissa_bits=40} AND Precision::Fp64, and via the CPU reference backend; assert max relative error (off-diagonal, |ref|>floor) is within the tight tier (emu vs ref < 1e-6; native vs ref < 1e-9). Print a small pass/fail table. This is a CORRECTNESS test (real input), not a precision benchmark. Exit nonzero on failure.',
  ].join('\n'), { label: 'equiv-test', phase: 'Implement' }),
])
const implOut = impl.map((r,i)=>['device-raii','f2-kernel','cpu-reference','build-system','equiv-test'][i]+':\n'+(r||'(none)')).join('\n---\n')

phase('BuildVerify')
const build = await agent([
  STD, '', '== what was authored ==', implOut, '',
  'Build and verify the M0 slice ON THE REMOTE BOX. Steps:',
  '1. rsync the local tree to the box: `rsync -az --mkpath -e "ssh -i ~/.ssh/id_vastai -p 63709 -o StrictHostKeyChecking=accept-new" ' + LOCAL + '/{include,src,tests,cmake,CMakeLists.txt,CMakePresets.json,build_m0.sh,.clang-format,.clang-tidy} root@108.255.76.60:/workspace/steppe/` (skip any path that does not exist).',
  '2. Build on the box (' + BOX + '): try CMake first (`cmake -S /workspace/steppe -B /workspace/steppe/build -GNinja && cmake --build /workspace/steppe/build`); if it fails to configure/build after 2 attempts, fall back to `bash /workspace/steppe/build_m0.sh`.',
  '3. Run the f2 equivalence test against /workspace/data/aadr/derived_acc.',
  '4. FIX-LOOP: on any compile or test error, edit the offending LOCAL files (' + LOCAL + '/...), re-rsync, rebuild — up to ~6 iterations. Common issues: include paths, the column-major GEMM args, the cublas emulation engage sequence, ComputeBackend signature mismatches.',
  'STOP after ~6 iterations regardless. Report: final build status (CMake and/or build_m0.sh), the equivalence-test output (emu-vs-ref and native-vs-ref max rel error, PASS/FAIL), and any unresolved errors with the exact message. Do not claim success unless the test actually printed PASS.',
].join('\n'), { label: 'build-verify', phase: 'BuildVerify' })

phase('CommentSweep')
const sweep = await agent([
  STD, '', '== build/verify result ==', build, '',
  'STALE-COMMENT SWEEP. Now that the precision policy is MEASURED, fix comments that contradict it. grep the new ' + LOCAL + '/{include,src,tests} AND ' + LOCAL + '/experiments for stale claims and fix each to match docs/architecture.md §12 + docs/ROADMAP.md §0:',
  '- "native FP64 ... f2 GEMM(s)", "f2 GEMMs run native", "exempt from emulation", "emulation buys no throughput", "bandwidth-bound so emulation" → the f2 GEMMs use fixed-slice Ozaki (40-bit default), measured 8–17× over native at native accuracy.',
  '- "dynamic mantissa" used approvingly → dynamic is the rejected trap; we use FIXED.',
  '- resolved FIXMEs/TODOs: "verify-against-headers", cuBLASLt-fallback, "tuning symbols not confirmed" → the emulation tuning symbols are CONFIRMED working in standard cuBLAS 13; remove/resolve those notes.',
  '- any other spike-era comment that misstates the current design.',
  'Do NOT change code logic — comments only. Leave the experiments/ spike files runnable but with corrected comments. Report every file touched and a one-line summary of each change.',
].join('\n'), { label: 'comment-sweep', phase: 'CommentSweep' })

return { contracts, build, sweep }
