# DATES goldens (oracle gate, 2026-06-24)

Reference: github.com/priyamoorjani/DATES (DATES Version 750), built on box5090
(CUDA 13.0 host; deps GSL 2.7.1, OpenBLAS 0.3.26, FFTW3 3.3.10). Default weight =
population allele-freq difference of the two sources (dates.c:604 wt=w1-w2; runmode 1).
FFT autocorrelation engine: fftsubs.c fftauto (FWD FFT -> |F|^2 -> INV FFT, /n).

## example_French_Yoruba (bit-identical reproduction of the bundled golden)
Simulated French+Yoruba admixture (real 1000G CEU/YRI haplotypes). 
date = 53.546 generations, SE = 3.346. Covariance curve .out BIT-IDENTICAL to committed golden.

## aadr_PUR_CEU_YRI (REAL AADR)
Target PUR (Puerto Rican, n=100), sources CEU+YRI, from v66.p1_HO.aadr.patch.PUB
TGENO (autosomes, 579720 SNPs). TGENO decoded to EIGENSTRAT by documented layout
(build_tgeno_matrix.py spec), convertf v8621 -> packedancestrymap, then DATES.
date = 9.742 generations, SE = 0.317. Literature-consistent (PUR Euro-African
admixture ~9-12 gen / colonial era; Moreno-Estrada 2013, Browning 2018).
