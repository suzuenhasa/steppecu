export const meta = {
  name: 'steppe-aadr-dataprep',
  description: 'Research AADR (Reich Lab) + author verified on-box scripts to turn AADR EIGENSTRAT into a per-population allele-freq matrix (Q/V/N) for the f2 emulation test. Scripts only — main loop executes on the box.',
  phases: [
    { title: 'Research', detail: '2 parallel: AADR dataset facts/URLs; EIGENSTRAT packed format + conversion to per-pop freqs' },
    { title: 'Author', detail: 'write download/convert/build-matrix scripts targeting the agreed Q/V/N binary format' },
    { title: 'Verify', detail: 'adversarial: allele-freq math, A1 polarity consistency, packed parsing, output layout' },
  ],
}

// AGREED OUTPUT FORMAT (both data-prep and the CUDA spike loader rely on this):
//   /workspace/data/aadr/derived/{Q.f64,V.f64,N.f64,meta.json}
//   Q/V/N: raw little-endian float64, written from a numpy array of shape (M, P) C-contiguous
//          via .tofile() -> element (pop i, snp s) sits at index i + P*s == cuBLAS column-major [P x M], lda=P.
//   Q = reference-allele frequency in [0,1] per (pop,snp); V = 1.0 if N>0 else 0.0; N = non-missing haploid count.
//   meta.json = {P, M, pops:[...], n_indiv_per_pop:[...], source, snp_cap}
const FMT = 'OUTPUT FORMAT (load-bearing, both sides depend on it): write /workspace/data/aadr/derived/{Q.f64,V.f64,N.f64,meta.json}. Q,V,N are raw little-endian float64 produced from a numpy array of shape (M,P) that is C-contiguous, written with .tofile() — so element (pop i, snp s) lands at flat index i + P*s, which equals a cuBLAS COLUMN-MAJOR [P x M] matrix with leading dimension P. Q = frequency of a FIXED reference allele (A1, identical allele across all pops) in [0,1]; V = 1.0 if that (pop,snp) has any non-missing data else 0.0; N = non-missing HAPLOID count (= 2 x non-missing diploids). meta.json = {P, M, pops, n_indiv_per_pop, source, snp_cap}.'

phase('Research')
const research = await parallel([
  () => agent([
    'Research the Allen Ancient DNA Resource (AADR), Reich Lab, on Harvard Dataverse. Use WebSearch/WebFetch.',
    'Find the CURRENT AADR version and EXACT public, wget/curl-able download URLs (Dataverse file-access API form https://dataverse.harvard.edu/api/access/datafile/<fileId>, plus the dataset DOI/landing page) for the PACKEDANCESTRYMAP .geno/.snp/.ind files of BOTH compilations: the "1240K" set and the "1240K+HO / Human Origins (HO)" set.',
    'For each: approx #SNPs, #individuals, #populations, per-file and total download size, and the exact file names.',
    'Recommend which to use for an f2 emulated-FP64 ACCURACY test that wants (a) many CLOSELY-RELATED modern populations (maximal small-f2 cancellation stress) and (b) manageable download+convert time — weigh HO (~600k SNPs, more modern pops) vs 1240K (~1.2M SNPs).',
    'Return exact URLs + a facts table, tagging each claim [VERIFIED]/[UNCERTAIN] with the source URL. Do not download anything.',
  ].join('\n'), { label: 'aadr-dataset-facts', phase: 'Research' }),

  () => agent([
    'Establish the EXACT, correct method to compute per-population allele-frequency matrices from AADR PACKEDANCESTRYMAP (.geno/.snp/.ind). Cite sources (EIGENSOFT/convertf docs, PLINK docs, format references).',
    '(1) Format: .ind columns (id, sex, population); .snp columns (id, chrom, genpos, physpos, ref/A1, alt/A2); packed .geno byte layout — header record, 2 bits per genotype, 4 samples per byte, record length = max(48, ceil(nIndiv/4)) bytes, genotype values 0/1/2 = copies of the REFERENCE allele and 3 = missing (9). Confirm the exact bit/byte order and the reference-allele convention (does value 2 mean 2 copies of .snp col5 ref allele?).',
    '(2) ROBUST conversion on a Linux box with conda/mamba: install via `mamba install -c bioconda eigensoft plink plink2`; EIGENSOFT `convertf` PACKEDANCESTRYMAP->PACKEDPED with an exact par file; then PLINK1.9 `--bfile X --within clusters.txt --freq --out f` producing per-cluster `.frq.strat` (CLST, SNP, A1, A2, MAF, MAC, NCHROBS; NCHROBS = non-missing chromosome count). Give the exact par file, the exact plink invocation, and how to build the cluster (pop) file from .ind/.fam population labels.',
    '(3) CRITICAL for cross-population f2: the per-pop frequency MUST be the frequency of the SAME fixed allele (A1) in every population — NOT the per-pop minor-allele frequency (which would flip alleles between pops and corrupt f2). Explain exactly how plink `--freq --within` keeps A1 consistent across clusters and the pitfall to avoid. Map: p_pop = (A1 count)/NCHROBS; N = NCHROBS; V = (NCHROBS>0).',
    '(4) Also give a fully-correct pure-Python+numpy fallback reader of packed .geno (no eigensoft), in case bioconda install fails.',
    'Return the verified format details + the exact end-to-end command pipeline.',
  ].join('\n'), { label: 'eigenstrat-conversion', phase: 'Research' }),
])
const [facts, conv] = research.map(r => r || '(missing)')

phase('Author')
const scripts = await agent([
  'Author the on-box AADR data-prep pipeline for the f2 emulation accuracy test. Box: CUDA 13, conda/mamba available, python venv at /venv/main, work under /workspace/data/aadr. Use the VERIFIED URLs/format/commands from the research below.',
  FMT,
  'Produce these files, each in its own fenced block with the filename on the preceding line:',
  '1. `aadr/00_setup.sh` — install tooling (mamba install -c bioconda eigensoft plink plink2; ensure numpy in /venv/main). Idempotent; echo versions.',
  '2. `aadr/01_download.sh` — mkdir -p /workspace/data/aadr/raw; wget the recommended (HO) AADR .geno/.snp/.ind via the verified Dataverse URLs (file IDs parameterized at top, with a comment of the source version). Skip files already present (idempotent). Print sizes.',
  '3. `aadr/02_build_matrix.py` — run under /venv/main. Inputs: the AADR triple + a POPULATION LIST (default a curated ~40-pop set that INCLUDES several closely-related clusters — e.g. multiple European (French, Sardinian, Tuscan, Basque, Orcadian...), multiple East-Asian (Han, Japanese, Korean, Dai...), multiple African — to maximize f2 cancellation stress) + optional --snp-cap and --maf-min. Compute per-pop reference-allele frequency, haploid N, validity V using the ROBUST path from the research (prefer convertf+plink `.frq.strat` then pivot; otherwise the verified pure-numpy packed reader). Drop SNPs/pops with no data. WRITE the Q.f64/V.f64/N.f64 + meta.json exactly per the OUTPUT FORMAT. Echo P, M, and a few sanity stats (mean freq, mean missingness).',
  '4. `aadr/03_run.sh` — documents/calls the spike in load mode: `/workspace/.../f2_emu_spike --load /workspace/data/aadr/derived` plus a note that the spike loader is wired separately. ',
  'All scripts fail loud (set -euo pipefail), are idempotent, keep everything under /workspace, and echo progress. Do NOT execute anything — just emit the files.',
  '',
  '==== RESEARCH: AADR dataset facts/URLs ====',
  facts,
  '',
  '==== RESEARCH: EIGENSTRAT format + conversion ====',
  conv,
].join('\n'), { label: 'author-scripts', phase: 'Author' })

phase('Verify')
const verified = await agent([
  'Adversarially review the AADR data-prep scripts below and emit corrected final versions. Check hard:',
  '(1) Allele-frequency math: missing (genotype 9 / NCHROBS gaps) excluded from BOTH numerator and denominator; N = non-missing haploid (chromosome) count; frequency is of a FIXED reference allele (A1) identical across ALL populations — NOT per-pop minor-allele freq (that flip would corrupt cross-pop f2). If using plink .frq.strat, confirm A1 is held consistent across clusters and the script does not re-polarize per pop.',
  '(2) If a pure-python packed reader is used: record length max(48,ceil(nIndiv/4)), header skip, 2-bit unpack order, value 3->missing, value->copies-of-ref correctness.',
  '(3) Output layout EXACTLY matches the agreed format: float64 [P x M] column-major (element (i,s) at i+P*s) via numpy (M,P) C-order .tofile(); meta.json fields present; Q in [0,1]; V in {0,1}; N>=0.',
  '(4) The default population set genuinely includes closely-related groups (cancellation stress); snp-cap/maf handling sane; SNPs with all-missing or monomorphic-everywhere handled.',
  '(5) Robustness: set -euo pipefail, idempotent, all paths under /workspace, fails loud on missing tools/files, URLs are the verified ones.',
  'Emit the corrected files, each in a fenced block with its filename, and nothing else. Mark any residual uncertainty with a clear # FIXME.',
  '',
  '==== SCRIPTS TO REVIEW ====',
  scripts,
].join('\n'), { label: 'verify-scripts', phase: 'Verify' })

return { verified, facts }
