# gen_golden.R — freeze a kalis Li-Stephens forward-backward posterior on a tiny
# REAL phased 1000G chr22 panel with a REAL (HapMap-interpolated) cM map, as the
# near-bit oracle for steppe's CpuBackend reference forward-backward.
#
# Panel: 12 real 1000G phase3 samples -> 24 phased haplotypes (the kalis cache),
# a subset of biallelic polymorphic SNPs, cM interpolated from the HapMap map.
# kalis copies each recipient from all OTHER haplotypes (Pi uniform, self excluded
# = leave-one-out). We freeze, for a few recipients, the FULL per-SNP posterior
# gamma (N donors x M SNPs), plus the exact donor allele matrix, rho (steppe
# convention), mu, and each recipient's Pi — so the C++ gate feeds kalis's own
# inputs to steppe's FB and diffs gamma.

suppressMessages(library(kalis))
setwd("/workspace/ls_panel")

set.seed(1)
M_KEEP    <- 256L        # SNPs in the tiny panel
RHO_S     <- 1.0e4       # CalcRho scale (spreads rho into an informative range)
MU        <- 0.002       # emission/miscopy rate (match -> 1-mu, mismatch -> mu)
RECIPS    <- c(1L, 2L, 13L)  # recipient haplotype indices (1-indexed into the cache)

## --- Build the phased haplotype matrix (loci x haplotypes, 0/1) --------------
d   <- read.table("gt.tsv", header = FALSE, stringsAsFactors = FALSE)
pos <- d[[1]]
gt  <- as.matrix(d[, -1])
haps <- do.call(cbind, lapply(seq_len(ncol(gt)), function(j) {
  s <- strsplit(gt[, j], "[|]")
  cbind(sapply(s, function(x) as.integer(x[1])),
        sapply(s, function(x) as.integer(x[2])))
}))
poly <- apply(haps, 1L, function(r) length(unique(r)) > 1L)
keep <- which(poly)
keep <- keep[seq_len(min(M_KEEP, length(keep)))]
haps <- haps[keep, , drop = FALSE]      # M x N
pos  <- pos[keep]
storage.mode(haps) <- "integer"
M <- nrow(haps); N <- ncol(haps)
cat(sprintf("panel: M=%d SNPs x N=%d haplotypes\n", M, N))

## --- Interpolate the REAL cM map at these positions --------------------------
mp   <- read.table(gzfile("map.gz"), header = FALSE, stringsAsFactors = FALSE)
mpos <- mp[[2]]; mcm <- mp[[3]]
cM   <- approx(mpos, mcm, xout = pos, rule = 2)$y
dcm  <- diff(cM)                         # length M-1, per-gap cM
dcm[dcm < 0] <- 0                        # guard any interpolation wobble

## --- kalis parameters --------------------------------------------------------
CacheHaplotypes(haps)                        # loci x haplotypes (must precede CalcRho)
stopifnot(L() == M, N() == N)
rho_kalis <- CalcRho(cM = dcm, s = RHO_S)   # length M-1, recomb between SNP l and l+1
cat(sprintf("rho range [%.4g, %.4g], median %.4g\n",
            min(rho_kalis), max(rho_kalis), median(rho_kalis)))
pars <- Parameters(rho = rho_kalis, mu = MU)  # Pi = NULL => uniform, self-excluded

## --- Per-recipient full posterior gamma (N x M) ------------------------------
# For each SNP l we build a fresh forward table advanced to l and a fresh backward
# table advanced to l, then read PostProbs (the exact per-position posterior).
gamma_for <- function(r) {
  g <- matrix(0.0, nrow = N, ncol = M)
  for (l in seq_len(M)) {
    fwd <- MakeForwardTable(pars, r, r)
    Forward(fwd, pars, t = l, nthreads = 1L)
    bck <- MakeBackwardTable(pars, r, r)
    Backward(bck, pars, t = l, nthreads = 1L)
    pp <- PostProbs(fwd, bck, nthreads = 1L)   # N x 1
    g[, l] <- as.numeric(pp)
  }
  g
}

## --- Write the golden --------------------------------------------------------
# steppe-convention rho: length M, rho[1]=1 (unused; the first column has no
# predecessor), rho[l] = recomb entering column l = rho_kalis[l-1].
rho_steppe <- c(1.0, rho_kalis)             # length M
mu_vec <- rep(MU, M)

con <- file("golden.txt", "w")
writeLines(sprintf("# steppe Li-Stephens paint golden (kalis oracle, real 1000G chr22 phased panel)"), con)
writeLines(sprintf("K %d", N), con)          # K donor states == N cache haplotypes
writeLines(sprintf("M %d", M), con)
writeLines(sprintf("R %d", length(RECIPS)), con)
writeLines(sprintf("mu_scalar %.17g", MU), con)
# donor allele matrix: N rows (donor-major), each M ints (0/1)
for (k in seq_len(N)) {
  writeLines(paste0("D ", paste(haps[, k], collapse = " ")), con)
}
# rho (steppe convention), length M
writeLines(paste0("RHO ", paste(sprintf("%.17g", rho_steppe), collapse = " ")), con)
# mu per-site, length M
writeLines(paste0("MU ", paste(sprintf("%.17g", mu_vec), collapse = " ")), con)
# each recipient: self index (0-indexed), alleles (M), pi (N), gamma (N*M donor-major)
for (r in RECIPS) {
  g <- gamma_for(r)
  pi_r <- rep(1.0 / (N - 1), N); pi_r[r] <- 0.0
  writeLines(sprintf("REC %d", r - 1L), con)   # 0-indexed self donor
  writeLines(paste0("A ", paste(haps[, r], collapse = " ")), con)
  writeLines(paste0("PI ", paste(sprintf("%.17g", pi_r), collapse = " ")), con)
  # gamma donor-major flat [k*M + l]: as.numeric(t(g)) walks donor k outer, SNP l inner.
  writeLines(paste0("G ", paste(sprintf("%.17g", as.numeric(t(g))), collapse = " ")), con)
}
close(con)
cat("wrote golden.txt\n")
cat(sprintf("golden bytes: %d\n", file.info("golden.txt")$size))
