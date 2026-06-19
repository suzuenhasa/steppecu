export const meta = {
  name: 'at2-timing-comparison',
  description: 'RESEARCH (read-only, web + steppe measured numbers): how long would the EQUIVALENT of what steppe just did take in ADMIXTOOLS 2 (extract_f2 / f2_from_geno) and classic ADMIXTOOLS (qpfstats/qpAdm)? steppe measured a full-autosome f2 precompute (P populations x M=584131 SNPs, n_block=757 block-jackknife, FP64-accurate EmulatedFp64) on ONE RTX 5090: P=512 ~0.67s device-resident / ~3.6s materialized; P=2500 = 51.5s (streamed, 76GB result). Find AT2/qpAdm timing + scaling + RAM/disk for the same workload, with SOURCES + confidence (do NOT fabricate numbers). Three lenses (AT2 extract_f2 precompute timing; AT2/classic qpAdm fit timing; the steppe reference numbers) -> an honest comparison doc with confidence + caveats. Writes docs/research/at2-timing-comparison.md.',
  phases: [
    { title: 'Research', detail: 'parallel: AT2 extract_f2 timing + qpAdm fit timing + the steppe measured reference' },
    { title: 'Compare', detail: 'honest apples-to-apples estimate + confidence + caveats -> doc' },
  ],
}

const R = '/home/suzunik/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell reimplementation of ADMIXTOOLS 2 f-statistics (precompute-once / fit-many). The PRECOMPUTE produces f2_blocks[P x P x n_block] FP64 (per-block leave-one-out jackknife tensor) via a 3-GEMM reformulation + Ozaki emulated-FP64 (FP64-accurate, bit-identical to a native-FP64 oracle). The FIT phase (qpAdm model rotation) is NOT built yet.',
  'STEPPE MEASURED NUMBERS (the reference for the comparison — real, on ONE consumer RTX 5090, full-autosome AADR: M=584131 SNPs, n_block=757 jackknife blocks, EmulatedFp64{40} = FP64-accurate): the f2 PRECOMPUTE (= ADMIXTOOLS-2 extract_f2 equivalent) took: P=512 ~0.67s (device-resident compute) / ~3.6s (with host materialization); P=1000 ~10.4s; P=1500 ~20.2s; P=2000 ~34.0s; P=2500 ~51.5s (the full 2500-population all-pairs f2 over 584k SNPs, 76GB result streamed). These are PRECOMPUTE-ONCE numbers; the many-model qpAdm fit is separate and not yet built.',
  'THE QUESTION: for the SAME workload — all-pairs f2 for P populations over ~584k SNPs with a block-jackknife (n_block~757), and separately a qpAdm model fit — how long does ADMIXTOOLS 2 (the R package, extract_f2 / f2_from_geno / qpadm by Robert Maier, eLife 2023) and/or classic ADMIXTOOLS (Patterson qpfstats/qp3Pop/qpAdm) take, on typical CPU hardware? And the RAM/disk it needs.',
  'HONESTY RULES (the user is allergic to fabrication): use web search + fetch (available via ToolSearch: query "select:WebSearch,WebFetch" then call them). CITE every timing/figure with a source URL. Give an explicit CONFIDENCE level. If authoritative timings are NOT found, SAY SO and reason from algorithmic complexity (f2 is O(P^2 * M) compute) + typical CPU throughput + the reported RAM/disk constraints — clearly labeled as an ESTIMATE, not a measurement. Do NOT invent a specific number and present it as fact.',
  'SOURCES to mine: the ADMIXTOOLS 2 eLife paper (Maier et al. 2023, "Ancient DNA and the population structure ..." / "On the limits of fitting complex models"), the AT2 documentation/vignettes (uqrmaie1.github.io/admixtools), AT2 GitHub issues/discussions (timing/RAM complaints for many populations), the classic ADMIXTOOLS README/paper (Patterson 2012), and any blog/forum benchmarks. Note AT2 extract_f2 is CPU (R/Rcpp), reads packed genotypes, and is known to be RAM/disk-heavy at many populations — find the actual reported scaling.',
].join('\n')

phase('Research')
const lenses = await parallel([
  () => agent([
    'You are a research analyst. Find how long ADMIXTOOLS 2 extract_f2 / f2_from_geno (the f2 PRECOMPUTE) takes and what RAM/disk it needs, for many populations over ~hundreds-of-thousands of SNPs. Web search/fetch (ToolSearch select:WebSearch,WebFetch). READ-ONLY.', STD, '',
    'FIND, with sources + confidence: (1) how extract_f2 computes f2 (CPU? all-pairs? per-block jackknife? blgsize default?) and its time complexity; (2) any REPORTED wall-clock timings for extract_f2 at scale (hundreds-to-thousands of pops, ~100k-1M SNPs) — from the paper, docs, GitHub issues, or benchmarks; (3) its RAM + DISK requirements at many populations (the f2 array is O(P^2 * n_block); AT2 is known to warn about memory — find the actual numbers/warnings); (4) whether AT2 precomputes-once-to-disk then fits-many (same model as steppe). If no hard timing exists, ESTIMATE from O(P^2 * M) on a typical multicore CPU and SAY it is an estimate. Return the findings + sources + confidence; be explicit about verified-vs-estimated.',
  ].join('\n'), { label: 'research:at2-extract-f2', phase: 'Research' }),
  () => agent([
    'You are a research analyst. Find how long the qpAdm FIT takes — both ADMIXTOOLS 2 qpadm() and classic ADMIXTOOLS qpAdm — per model and for a rotation/model-search over many models. Web search/fetch. READ-ONLY.', STD, '',
    'FIND, with sources + confidence: (1) AT2 qpadm() per-model fit time once f2 is precomputed (it reads f2_blocks + does small linear algebra — should be fast); (2) classic ADMIXTOOLS qpAdm runtime (reads genotypes per run — known slower); (3) the qpAdm ROTATION / model-search cost (running many models, e.g. the Harney/rotating-outgroup approach) — how many models and total time reported; (4) the precompute-vs-fit split in AT2 (extract_f2 once = the expensive part; qpadm many = cheap). Return findings + sources + confidence; verified-vs-estimated explicit.',
  ].join('\n'), { label: 'research:qpadm-fit', phase: 'Research' }),
  () => agent([
    'You are a steppe engineer. Assemble the steppe-side reference numbers for the comparison from the repo (read-only, no web). Confirm the measured full-autosome precompute timings + the workload definition so the comparison is apples-to-apples.', STD, '',
    'Read docs/cleanup/m4.5/*.md and any sweep results in the repo + the numbers in STD. Confirm/assemble: the steppe f2 PRECOMPUTE timings (P=512/1000/1500/2000/2500 full-autosome, M=584131, n_block=757, one RTX 5090), the precision (EmulatedFp64 = FP64-accurate), the result size (2*P^2*n_block*8), the hardware (one consumer 5090, 32GB), and the precompute-once/fit-many framing (fit not built yet). State clearly what steppe MEASURED (the precompute) vs what it has NOT built (the qpAdm fit), so the comparison does not overclaim. Return the consolidated steppe reference.',
  ].join('\n'), { label: 'research:steppe-reference', phase: 'Research' }),
]).then(a => a.filter(Boolean))

phase('Compare')
const compare = await agent([
  'You are the lead. Write an HONEST timing comparison: how long the equivalent of steppe`s f2 precompute (and the qpAdm fit) would take in ADMIXTOOLS 2 / classic qpAdm. WRITE to ' + R + '/docs/research/at2-timing-comparison.md (Write tool). READ-ONLY on code.', STD, '',
  'THE THREE LENSES:\n<<<\n' + lenses.map((a,i)=>'### lens '+(i+1)+'\n'+a).join('\n\n') + '\n>>>', '',
  'The doc must contain: (1) THE APPLES-TO-APPLES — the f2 precompute: steppe MEASURED (P=2500 full-autosome = 51.5s on one 5090; the smaller-P points) vs AT2 extract_f2 for the SAME (P=2500, M~584k, n_block~757) — the AT2 number with its SOURCE + CONFIDENCE (or an explicitly-labeled complexity-based ESTIMATE if no source); the speedup factor, honestly bounded (a range if the AT2 number is an estimate); (2) the RAM/DISK comparison (AT2 needs the f2 array + genotype data in RAM/disk; steppe streams it — note both); (3) the FIT phase — AT2 qpadm per-model + a rotation, vs steppe (not built yet, so DO NOT claim a steppe fit number; only note where steppe would sit once built); (4) CAVEATS that keep it honest — CPU cores assumed, AT2 is FP64-native (steppe is FP64-accurate-emulated, bit-identical), extract_f2 is precompute-once (amortized), steppe`s number is the precompute only, hardware differences; (5) BOTTOM LINE — a defensible statement of the speedup for the precompute (e.g. "X-Y faster, source/estimate") without overclaiming. Tag every number verified-vs-estimated with its source. If the AT2 timing is genuinely unknown, say the comparison is an estimate and give the range + the assumptions.',
  'Return a tight 8-12 line executive summary: the AT2 extract_f2 time for the P=2500 full-autosome workload (sourced or estimated, with confidence), steppe`s measured 51.5s, the honest speedup range, the RAM/disk contrast, and the fit-phase note + caveats.',
].join('\n'), { label: 'compare-timing', phase: 'Compare' })

return { lenses, compare }
