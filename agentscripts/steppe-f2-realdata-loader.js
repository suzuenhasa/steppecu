export const meta = {
  name: 'steppe-f2-realdata-loader',
  description: 'Make build_matrix.py conda-free (pure-numpy packed AADR reader) + add --load mode to f2_emu_spike.cu; the two must agree byte-for-byte on the Q/V/N layout. Adversarially verified. Code only.',
  phases: [
    { title: 'Author', detail: 'one author writes BOTH files against a shared binary-format spec (coherence)' },
    { title: 'Verify', detail: '2 parallel: packed-reader numerics/format; spike loader + cross-format agreement' },
    { title: 'Finalize', detail: 'apply both verifies; emit final files' },
  ],
}

const CU = '/home/suzunik/steppe/experiments/f2_emu_spike/f2_emu_spike.cu'
const PY = '/home/suzunik/steppe/experiments/aadr/02_build_matrix.py'

const FMT = [
  'SHARED BINARY FORMAT (both files MUST agree exactly):',
  '- Output dir: /workspace/data/aadr/derived/ containing: Q.f64, V.f64, N.f64, meta.json, shape.txt.',
  '- Q.f64/V.f64/N.f64: raw little-endian float64, written from a numpy array of shape (M, P) that is C-contiguous via .tofile(). Therefore element (population i, SNP s) sits at flat index (i + P*s) — i.e. a cuBLAS COLUMN-MAJOR [P x M] matrix with leading dimension P. (Reader and writer must both use this exact convention.)',
  '- Q = frequency of the REFERENCE allele in [0,1] (the .geno value counts copies of the reference allele, so this is automatically consistent across populations — do NOT re-polarize to minor allele).',
  '- N = non-missing HAPLOID count = 2 * (non-missing diploid individuals in that pop at that SNP). V = 1.0 if N>0 else 0.0.',
  '- shape.txt: a single ASCII line "P M\\n" (two ints) for trivial C fscanf. meta.json: {P, M, pops:[...], n_indiv_per_pop:[...], source}.',
].join('\n')

phase('Author')
const authored = await agent([
  'Author TWO coupled files for the steppe f2 emulation real-data test. Read the current versions from disk first:',
  '  - CUDA spike:  Read ' + CU,
  '  - build script: Read ' + PY,
  'Then emit FINAL full replacements for both.',
  '',
  FMT,
  '',
  'FILE 1 — ' + PY + ' (rewrite as a SELF-CONTAINED pure-NumPy reader; the box has NO conda/eigensoft/plink, only python+numpy at /venv/main). It must:',
  '- Parse AADR PACKEDANCESTRYMAP from /workspace/data/aadr/raw: read the .ind (cols: id, sex, population) and .snp (id, chrom, genpos, physpos, ref, alt), and the packed .geno directly in numpy. Packed .geno layout: ASCII header record then one record per SNP; record length rlen = max(48, ceil(nIndiv/4)) bytes; 2 bits per individual, 4 individuals per byte; verify the bit order (typically the highest-order 2 bits is the first individual); genotype value 0/1/2 = copies of the REFERENCE allele, 3 = missing. Confirm by reading the GENO magic + nind/nsnp from the header. Use np.fromfile/np.unpackbits, process SNPs in CHUNKS (e.g. 20k rows) so peak RAM stays modest (nIndiv may be ~16-21k).',
  '- Population selection: support `--pops a,b,c` (exact .ind labels), `--list-pops` (print every population with its individual count, then exit — so the user can pick), and `--auto-top K` (pick the K populations with the most individuals). DEFAULT: a curated set of closely-related-cluster names to maximize f2 cancellation stress (multiple European e.g. French, Sardinian, Tuscan, Spanish, Basque, Orcadian; multiple East-Asian e.g. Han, Japanese, Korean, Dai; some African e.g. Yoruba, Mbuti) — BUT if fewer than ~5 of the curated names match the real .ind labels, FALL BACK to --auto-top 40 and print a clear warning (AADR labels vary, e.g. suffixes like .DG/.SG/.HO).',
  '- For each selected pop and SNP: p_ref = (sum of genotype values over non-missing individuals)/(2*n_nonmissing); N = 2*n_nonmissing; V = 1.0 if n_nonmissing>0 else 0.0. Optional --snp-cap K (take first K SNPs) and --maf-min (drop SNPs whose mean ref-freq is <maf or >1-maf across selected pops). Drop SNPs with no data in any selected pop.',
  '- Write derived/{Q.f64,V.f64,N.f64,meta.json,shape.txt} per the SHARED FORMAT. Echo P, M, mean freq, mean missingness, and the matched pop list.',
  '- set sane argparse defaults; fail loud if raw files missing.',
  '',
  'FILE 2 — ' + CU + ' (add a --load mode; keep everything else, especially the emulation-engagement guard and the long-double reference, intact). Requirements:',
  '- New CLI: `./f2_emu_spike --load <dir>` reads <dir>/shape.txt (P M via fscanf) then <dir>/{Q.f64,V.f64,N.f64} (each P*M little-endian doubles, column-major [P x M] per the SHARED FORMAT) into host vectors.',
  '- From the loaded raw reference-allele freq Q_raw, validity V, and haploid N, reconstruct EXACTLY the same host arrays the synthetic path builds before the GEMMs: Q_masked = Q_raw * V; Hc = Q_raw*(1-Q_raw)/max(N-1,1) * V; Qsq = Q_masked*Q_masked (or Q_raw*Q_raw*V — match whatever the existing synthetic code does); S = [Qsq ; Hc]. Then run the IDENTICAL native + emulated GEMM paths, the long-double reference, the error stats, the emulation-engagement check, and the table print. The "sigma"/"miss" columns can show NaN or "real"; add a label column or print the source dir + P + M.',
  '- Preserve the existing `P M sigma [miss] [seed]` synthetic mode when argv[1] is not "--load".',
  '- Must still compile with: nvcc -O3 -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1 f2_emu_spike.cu -lcublas',
  '',
  'Output EXACTLY two fenced code blocks, each preceded by a line with its absolute filename: first ' + PY + ', then ' + CU + '. No other prose.',
].join('\n'), { label: 'author-both', phase: 'Author' })

phase('Verify')
const checks = await parallel([
  () => agent([
    'Adversarially verify the PYTHON packed-AADR reader below (file ' + PY + '). Find and list defects precisely:',
    '(1) Packed .geno parsing: rlen = max(48, ceil(nIndiv/4)); header record skipped correctly; 2-bit unpack ORDER (which individual is the high vs low bits of each byte — getting this wrong silently scrambles genotypes); value 3 => missing; value 0/1/2 => copies of REFERENCE allele.',
    '(2) Allele-frequency math: missing excluded from BOTH numerator and denominator; p_ref=(sum genos)/(2*n_nonmissing); N=2*n_nonmissing; V=(N>0). It must be the SAME reference allele across pops (it is, since .geno counts the .snp reference allele) — confirm no per-pop minor-allele re-polarization.',
    '(3) Output layout EXACTLY: numpy (M,P) C-order float64 .tofile() so (pop i, snp s) is at i+P*s == column-major [P x M]; shape.txt "P M"; meta.json fields.',
    '(4) Memory: chunked over SNPs, no full nIndiv x nSNP unpacked array; works for nIndiv~21k, M~600k.',
    '(5) Robustness: --list-pops / --auto-top / curated-fallback; fails loud on missing files.',
    'Return a terse numbered punch list of concrete fixes (or "no defects" with reasoning). Read the file from disk too if useful.',
    '',
    '==== AUTHORED FILES ====',
    authored,
  ].join('\n'), { label: 'verify-python', phase: 'Verify' }),
  () => agent([
    'Adversarially verify the CUDA spike --load mode below (file ' + CU + '). Find and list defects precisely:',
    '(1) Does the loader read shape.txt (P,M) then Q/V/N as P*M little-endian doubles in COLUMN-MAJOR [P x M] (element (i,s) at i+P*s) — matching the writer? A row/col-major mismatch would transpose the data silently.',
    '(2) Does it reconstruct Q_masked, Hc, Qsq, S IDENTICALLY to the existing synthetic path (compare against the actual synthetic code in the same file)? Any divergence (e.g. forgetting to mask Q, or a different Hc denominator) corrupts the comparison.',
    '(3) Are the native + emulated GEMM calls, the long-double reference, the emulation-engagement (emuEqNat) guard, and the table print all still exercised in --load mode?',
    '(4) Synthetic mode (P M sigma ...) still intact; arg parsing unambiguous (--load vs numeric P).',
    '(5) Compiles with nvcc -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1; no fscanf/fread misuse; file-size sanity check (fread returns P*M).',
    'Return a terse numbered punch list of concrete fixes (or "no defects"). Read the file from disk too if useful.',
    '',
    '==== AUTHORED FILES ====',
    authored,
  ].join('\n'), { label: 'verify-cuda', phase: 'Verify' }),
])
const [vpy, vcu] = checks.map(r => r || '(none)')

phase('Finalize')
const finalFiles = await agent([
  'Apply EVERY valid fix from the two punch lists to the authored files and emit the FINAL versions. Keep the SHARED BINARY FORMAT (writer and loader must agree: column-major [P x M] float64, (i,s) at i+P*s; shape.txt "P M"; reference-allele freq; N=non-missing haploid; V=(N>0)). Preserve the spike emulation guard and long-double reference. Ensure the .cu compiles with nvcc -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1 -lcublas.',
  'Output EXACTLY two fenced code blocks, each preceded by its absolute filename line: first ' + PY + ', then ' + CU + '. No other prose.',
  '',
  '==== AUTHORED FILES ====',
  authored,
  '',
  '==== PUNCH LIST: python reader ====',
  vpy,
  '',
  '==== PUNCH LIST: cuda loader ====',
  vcu,
].join('\n'), { label: 'finalize', phase: 'Finalize' })

return { finalFiles }
