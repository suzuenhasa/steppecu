export const meta = {
  name: 'cuda-qpadm-scaffold-prompt',
  description: 'Research CUDA 13+ best practices and author a senior-grade scaffolding prompt for a GPU qpAdm/admixtools2 reimplementation',
  phases: [
    { title: 'Research', detail: '6 parallel experts: build/toolchain, C++ idioms, linalg libs, architecture, QA/CI, qpAdm domain' },
    { title: 'Draft', detail: 'synthesize research into the scaffolding prompt MD' },
    { title: 'Critique', detail: 'adversarial staff-engineer review' },
    { title: 'Finalize', detail: 'revise into the final document' },
  ],
}

phase('Research')

const PROMPTS = [
  // 1. CUDA 13+ toolkit, compiler, build system
  'You are a CUDA build-systems expert. Research CURRENT best practices for CUDA Toolkit 13.0+ (released 2025) project setup. Use WebSearch and WebFetch against NVIDIA CUDA 13 release notes, the CUDA C++ Programming/Best-Practices guides, and CMake CUDA documentation. Cover concretely: (1) What changed in CUDA 13.x vs 12.x relevant to a new project: dropped/added GPU architectures (e.g. removal of older sm_xx, addition of Blackwell sm_100/sm_103/sm_120 family-conditional/arch-conditional features), unified Arm+x86 packaging, CCCL being bundled, nvcc and nvc++ behavior, default C++ standard, tile-based / cluster features. (2) Modern target-based CMake (3.2x+) for CUDA: enable_language(CUDA), CMAKE_CUDA_ARCHITECTURES (native / all-major / explicit, and arch-conditional a_/f_ suffixes if applicable), CUDA_SEPARABLE_COMPILATION, device LTO (INTERPROCEDURAL_OPTIMIZATION), CMakePresets.json layout, CMAKE_EXPORT_COMPILE_COMMANDS, ccache/sccache wiring, ninja. (3) Dependency management options: CPM.cmake vs vcpkg vs Conan; how to fetch CCCL. (4) Recommended flag policy: warnings-as-errors, -lineinfo, debug vs release, -G caveats. Return a detailed, well-structured markdown section with concrete CMake / preset code snippets in fenced blocks and cite your sources inline as URLs. Be specific and current; explicitly flag anything you are unsure about.',

  // 2. Modern CUDA C++ idioms, CCCL, RAII
  'You are a senior CUDA C++ engineer. Document modern CUDA C++ idioms for a clean, DRY, strongly-separated codebase targeting CUDA 13+ and C++20. Use WebSearch and WebFetch against NVIDIA CCCL (Thrust / CUB / libcudacxx) docs, the CUDA C++ Best Practices Guide, and cuda::std references. Cover with code sketches in fenced blocks: (1) RAII resource wrappers: a DeviceBuffer<T> owning device memory, Stream, Event, and library-handle wrappers (cuBLAS/cuSOLVER) -- and WHY raw cudaMalloc/cudaFree must never be scattered through the code. (2) Error handling: a CUDA_CHECK macro, checking cudaGetLastError after every kernel launch, translating to host exceptions, and how this interacts with compute-sanitizer. (3) Streams, events, and CUDA graphs for orchestrating multi-step pipelines. (4) Async memory: cudaMallocAsync + memory pools, pinned host memory, and avoiding allocations in hot loops. (5) CCCL usage: when to use Thrust algorithms, CUB device-wide primitives (reductions/scans/sort), cuda::std::span and mdspan for typed non-owning views, cooperative groups. (6) Host/device separation discipline: keeping kernels tiny and testable, __host__ __device__ boundaries, header hygiene. Return detailed markdown with citations as inline URLs.',

  // 3. GPU linear algebra / math libraries grounded in qpAdm workload
  'You are an expert in GPU numerical linear algebra. The target is a GPU reimplementation of admixtools2 / qpAdm population-genetics tools. The workload: very large genotype / allele-frequency matrices; f2/f3/f4 statistics (covariance-like inner products across populations); block jackknife / bootstrap resampling producing MANY medium matrices; and constrained / generalized least-squares and SVD/QR rank tests for model fitting (qpWave/qpAdm). Research the right NVIDIA libraries and patterns using WebSearch and WebFetch against cuBLAS, cuBLASLt, cuSOLVER, cuSOLVERMp, cuSPARSE, and cuRAND docs. Cover: which routine maps to which step (GEMM, SYRK/SYR2K for covariance, batched/strided-batched GETRF/GETRS/GEQRF/GESVD for many small systems, least-squares GELS, generalized eigen/SVD for rank tests); batched vs strided-batched tradeoffs for thousands of small solves; mixed precision FP64 vs FP32 vs TF32 and the correctness implications for population-genetics statistics; handle/workspace lifetime and RAII; deterministic reductions; and clear guidance on when to call a library vs write a custom fused kernel. Return detailed markdown with exact API names, short code sketches in fenced blocks, and inline URL citations.',

  // 4. Software architecture, DRY, config, separation, layout, bindings, packaging
  'You are a staff software architect for scientific / HPC C++/CUDA projects. Design an architecture and engineering practices that would impress a senior engineer for a GPU qpAdm/admixtools2 reimplementation. Research current practice via WebSearch (modern C++/CUDA repo structure, scikit-build-core, nanobind vs pybind11, Architecture Decision Records, include-what-you-use). Cover in depth: (1) A concrete repository layout with STRICT separation of concerns: public API headers (stable include/), core compute library, a device/kernel layer, an I/O layer (genotype formats), a CLI application, Python bindings, tests, benchmarks, docs, cmake/ modules, tools/scripts, third_party/external. Present it as an annotated directory tree. (2) Layered architecture and enforced dependency direction (api -> compute -> device; io isolated; nothing depends upward) and how to enforce it (separate CMake targets, PRIVATE/PUBLIC link visibility, no cyclic includes). (3) DRY and shared helpers: one source of truth for error checking, logging, configuration, launch-config math, type traits, and a CPU reference backend that tests and the GPU path both validate against. (4) Configuration management: typed immutable config objects, layered resolution (compiled defaults < config file < env vars < CLI flags), validation, NO global mutable state, dependency injection of resources (device id, streams, allocators), runtime device and precision selection. (5) Logging abstraction + NVTX for profiling. (6) Python bindings (nanobind + scikit-build-core wheel) and why a CPU reference backend matters. (7) ADRs, semantic versioning, API/ABI stability. (8) Coding standards: clang-format, clang-tidy (with CUDA), .editorconfig, naming conventions, header guards/pragma once. Return detailed markdown with the directory tree in a fenced block, rationale, and inline URL citations.',

  // 5. Testing, sanitizers, CI, profiling, numerical correctness
  'You are a CUDA QA and performance engineer. Research best practices for testing, validation, sanitizing, CI/CD, and profiling a CUDA 13+ project. Use WebSearch and WebFetch against compute-sanitizer docs, GoogleTest/Catch2 with CUDA, NVIDIA nvbench, Nsight Systems and Nsight Compute, and GitHub Actions self-hosted GPU runners. Cover: (1) Unit-testing device code: GoogleTest or Catch2 patterns for kernels, CPU-vs-GPU golden tests, and regression tests pinned against admixtools2 reference outputs. (2) compute-sanitizer tools (memcheck, racecheck, initcheck, synccheck) and wiring them into CI. (3) Numerical correctness: FP64 vs FP32 vs TF32 policy, deterministic vs non-deterministic reductions, pairwise/Kahan summation, choosing ULP/relative/absolute tolerances, and seed control so jackknife/bootstrap resampling is reproducible. (4) Benchmarking with nvbench, performance-regression gates, and NVTX range annotation. (5) Profiling workflow: Nsight Systems first (timeline/overlap) then Nsight Compute (occupancy, roofline, memory throughput). (6) CI/CD: matrix over CUDA versions and GPU arches, clang-tidy/clang-format gates, pre-commit hooks, sccache caching, and building Python wheels. Return detailed markdown with concrete commands and snippets in fenced blocks plus inline URL citations.',

  // 6. qpAdm / admixtools2 domain grounding
  'You are a computational population geneticist and software engineer. Summarize the COMPUTATIONAL structure of admixtools2 / qpAdm so a GPU architecture can be designed around it -- focus on data flow and compute stages, not biology. Use WebSearch and WebFetch on admixtools2 (Robert Maier), qpAdm/qpWave methodology, Patterson f-statistics (f2, f3, f4, f4-ratio), the original ADMIXTOOLS, and PLINK/EIGENSTRAT formats. Cover: (1) Core primitives: allele-frequency matrices, f2 as the cacheable building block, and how f3/f4 are derived from f2 (the admixtools2 f2-blocks precompute model). (2) Block jackknife (leave-one-genome-block-out) as the dominant resampling pattern for standard errors and covariance -- and how that drives compute shape. (3) The qpWave/qpAdm fitting algorithm: building the f4 statistic matrix, rank tests via SVD, weighted/generalized least squares using the jackknife covariance, and nested-model p-values. (4) Input formats: PLINK bed/bim/fam, EIGENSTRAT/PACKEDANCESTRYMAP, allele counts. (5) Realistic data scale (thousands of samples, hundreds of thousands to millions of SNPs) and where the memory and compute bottlenecks are. End with an explicit list of the natural pipeline STAGES/MODULES and their parallelism so the software architecture can map one module per stage. Return detailed markdown with a data-flow description and inline URL citations.',
]

const LABELS = ['build-toolchain', 'cpp-idioms', 'linalg-libs', 'architecture', 'qa-ci', 'qpadm-domain']

const research = await parallel(
  PROMPTS.map((p, i) => () => agent(p, { label: LABELS[i], phase: 'Research' }))
)

const safe = research.map((r, i) => '## RESEARCH BUNDLE [' + LABELS[i] + ']\n\n' + (r || '(this researcher returned nothing; rely on the others)'))
const bundle = safe.join('\n\n---\n\n')

phase('Draft')

const DRAFT_SPEC = [
  'You are a principal engineer writing the canonical architecture & best-practices document for a new project: a GPU/CUDA (Toolkit 13+) reimplementation/reimagining of the admixtools2 / qpAdm population-genetics toolkit.',
  '',
  'This document is BOTH (a) the engineering spec the human team follows and (b) a precise prompt that an AI coding agent can be handed to scaffold the repository correctly from day one. It must be authoritative and OPINIONATED: pick concrete defaults and justify them in one line; do not present wishy-washy menus of options.',
  '',
  'You are given six research bundles below (CUDA 13 build/toolchain, modern CUDA C++ idioms, GPU linear-algebra libraries, software architecture, QA/CI, and the qpAdm domain). Synthesize them -- do NOT just concatenate. Resolve conflicts, keep concrete snippets, and ground every architectural choice in the actual qpAdm workload (large genotype matrices, f2 block precompute, block jackknife resampling, GLS/SVD rank tests).',
  '',
  'Produce a single, polished, self-contained Markdown document with these sections, in order:',
  '0. Title + a short "Purpose & how to use this document" preamble (it is a spec AND an AI scaffolding prompt) and an explicit "Assumptions & chosen defaults" list (CUDA 13.x, C++20, CMake + Ninja, CCCL, cuBLAS/cuSOLVER, nanobind, GoogleTest, nvbench, clang-tooling).',
  '1. Project vision & non-goals (what this GPU qpAdm tool is and is NOT).',
  '2. Engineering principles (DRY / single source of truth, strict separation of concerns, RAII-everywhere, no global mutable state, fail-fast, testability, numerical reproducibility/determinism, correctness-before-speed). Each principle = one crisp paragraph with how it shows up concretely.',
  '3. Tech stack & pinned versions with one-line rationale each.',
  '4. Repository layout: an annotated directory tree in a fenced block, with a one-line purpose per directory, plus a dependency-direction rule (api -> compute -> device; io isolated; nothing depends upward) and how CMake target visibility enforces it.',
  '5. Layered architecture mapped onto the qpAdm pipeline stages (data load -> allele frequencies -> f2 blocks -> jackknife -> f4 / qpAdm GLS+SVD solve -> results), naming the module that owns each stage and its parallelism.',
  '6. Build system: modern target-based CMake patterns, CMAKE_CUDA_ARCHITECTURES policy (incl. Blackwell), separable compilation + device LTO, CMakePresets.json, compile_commands.json, sccache, dependency fetching (CPM), and sanitizer/clang-tidy build variants. Include real fenced CMake/preset snippets.',
  '7. CUDA coding standards & idioms: RAII wrappers (DeviceBuffer<T>, Stream, Event, handle wrappers) with short code sketches, the CUDA_CHECK macro + post-launch error checks, launch-config helpers, CCCL (Thrust/CUB/mdspan) usage policy, async memory pools, streams/graphs, and host/device separation rules.',
  '8. DRY & shared utilities: the single-source-of-truth helpers (error, logging, config, launch math, type traits, span/mdspan) and the CPU reference backend shared by tests and validation.',
  '9. Configuration management: typed immutable config, layered resolution (defaults < file < env < CLI), validation, dependency injection of device/stream/allocator, runtime device & precision selection. Include a config struct sketch.',
  '10. Error handling, logging & observability (NVTX ranges).',
  '11. Memory & performance strategy (pools, pinned host memory, async, occupancy) and the profiling workflow (Nsight Systems then Compute).',
  '12. Numerical correctness & reproducibility: precision policy (FP64 default for stats, where TF32/FP32 is acceptable), deterministic reductions, tolerance policy, seed control for resampling, and validation against admixtools2 reference outputs.',
  '13. Testing strategy: unit/golden/regression/property tests, compute-sanitizer in CI, and how the CPU reference enables it.',
  '14. CI/CD & tooling: pipeline stages, GPU runner matrix (CUDA versions x arch), format/tidy/sanitizer gates, pre-commit, wheel builds.',
  '15. Python bindings & packaging (nanobind + scikit-build-core).',
  '16. Documentation & ADRs.',
  '17. Scaffold manifest: an ORDERED, actionable checklist of the exact files/targets to create first (CMakeLists.txt, presets, helper headers, etc.) so someone can begin immediately. Be concrete enough to act on.',
  '18. Definition of Done / quality bar.',
  '',
  'Rules: Use real code in fenced blocks where it adds value (CMake, RAII C++ sketches, config struct, directory tree). Keep prose tight -- no filler. Be explicit and verifiable about any CUDA-13-specific claim; if a fact is uncertain, mark it clearly rather than stating it confidently. The whole thing should read like it was written by a meticulous staff engineer. Your ENTIRE response is the markdown document (it will be saved to a file verbatim) -- no preamble, no meta-commentary.',
  '',
  '==== RESEARCH BUNDLES ====',
  '',
  bundle,
].join('\n')

const draft = await agent(DRAFT_SPEC, { label: 'draft-document', phase: 'Draft' })

phase('Critique')

const CRITIQUE_SPEC = [
  'You are a skeptical, highly experienced staff/principal engineer reviewing the draft architecture document below. You specialize in CUDA/HPC and you have shipped large GPU codebases. Your job is to find everything that would make this document UNIMPRESSIVE or WRONG to a senior reviewer.',
  '',
  'Critique ruthlessly and specifically (cite section numbers). Check for:',
  '- Generic/boilerplate advice that says nothing actionable ("write clean code"), and where it should be concrete.',
  '- Incorrect or outdated CUDA 13+ claims (architectures, CMake variables, CCCL, deprecated APIs, flags). Call out anything stated with false confidence.',
  '- Architecture smells: leaky layering, missing module boundaries, places the qpAdm workload (f2 blocks, block jackknife, GLS/SVD) is NOT actually reflected in the design.',
  '- Missing topics a senior engineer would expect (e.g. ABI/versioning, error taxonomy, multi-GPU, determinism guarantees, large-data out-of-core / streaming when genotype matrices exceed VRAM, workspace/memory budgeting, build reproducibility).',
  '- DRY/config/separation gaps -- the user explicitly cares about DRY, config & helpers, and separation of everything.',
  '- Code snippets that are wrong, unsafe, or not idiomatic.',
  '',
  'Return a prioritized, numbered list of concrete fixes (most important first). Each item: what is wrong + the specific change to make. Do not rewrite the document; just produce the punch list. Be terse and high-signal.',
  '',
  '==== DRAFT DOCUMENT ====',
  '',
  draft,
].join('\n')

const critique = await agent(CRITIQUE_SPEC, { label: 'staff-critique', phase: 'Critique' })

phase('Finalize')

const FINAL_SPEC = [
  'You are the principal engineer finalizing the architecture & best-practices document. Below is your DRAFT followed by a staff-engineer CRITIQUE punch list. Produce the FINAL document by applying every valid critique item: fix incorrect/outdated CUDA-13 claims, replace generic advice with concrete guidance, close the missing-topic gaps (e.g. out-of-core/streaming when genotype matrices exceed VRAM, multi-GPU strategy, determinism guarantees, error taxonomy, workspace/memory budgeting, ABI/versioning), and make sure DRY, configuration/helpers, and strict separation of concerns are concretely addressed since the user cares most about those. Keep all the strong concrete snippets from the draft.',
  '',
  'Keep the same section structure (0 through 18). Tighten prose. Ensure every CUDA-13-specific claim is either verifiable or explicitly marked uncertain. The result must read like a meticulous staff engineer wrote it and must be directly usable BOTH as a human spec and as an AI scaffolding prompt.',
  '',
  'Your ENTIRE response is the final markdown document -- it will be saved to a file verbatim. No preamble, no commentary, no code fence around the whole thing.',
  '',
  '==== DRAFT DOCUMENT ====',
  '',
  draft,
  '',
  '==== STAFF CRITIQUE PUNCH LIST ====',
  '',
  critique,
].join('\n')

const finalDoc = await agent(FINAL_SPEC, { label: 'finalize-document', phase: 'Finalize' })

return { document: finalDoc }
